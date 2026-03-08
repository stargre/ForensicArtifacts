# test.py

import os
import sys
import yaml
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import json
from datetime import datetime

# 导入数据处理模块
from pre_data.dataprocess import ForensicFeatureDataset

# 导入模型组件
from feature.Scene.scene_encoder import SceneEncoder
from feature.Imaging.image_encoder import ImagingEncoder
from feature.Signal.signal_encoder import SignalEncoder
from feature.MHSA import GatedFusionModule


# ======================== 配置解析 ========================
def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='虚假图像检测测试')
    parser.add_argument('--config', type=str, required=True, 
                       help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='模型权重路径（优先级高于配置文件）')
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='分布式训练本地rank')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='结果保存目录')
    return parser.parse_args()


# ======================== 分布式设置 ========================
def setup_distributed():
    """设置分布式环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
        torch.cuda.set_device(local_rank)
        
        return True, rank, world_size, local_rank
    else:
        return False, 0, 1, 0


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


# ======================== 模型定义（与训练一致）========================
class ClassificationHead(nn.Module):
    """分类头"""
    def __init__(self, in_channels=64, hidden_dim=256, dropout=0.1):
        super(ClassificationHead, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, F_unified):
        f_global = self.gap(F_unified).view(F_unified.size(0), -1)
        h = self.relu(self.bn1(self.fc1(f_global)))
        h = self.dropout(h)
        p = self.sigmoid(self.fc2(h))
        return p


class ForensicDetectionModel(nn.Module):
    """完整检测模型"""
    def __init__(self, config):
        super(ForensicDetectionModel, self).__init__()
        
        model_cfg = config['model']
        
        self.scene_encoder = SceneEncoder(
            in_channels=model_cfg.get('scene_in_channels', 4),
            out_channels=model_cfg.get('scene_channels', 64)
        )
        self.imaging_encoder = ImagingEncoder(
            in_channels=model_cfg.get('imaging_in_channels', 32),
            out_channels=model_cfg.get('imaging_channels', 64)
        )
        self.signal_encoder = SignalEncoder(
            in_channels=model_cfg.get('signal_in_channels', 3),
            out_channels=model_cfg.get('signal_channels', 64)
        )
        
        self.fusion_module = GatedFusionModule(
            feature_channels=model_cfg.get('fusion_channels', 64),
            reduction=model_cfg.get('reduction', 4)
        )
        
        self.classifier = ClassificationHead(
            in_channels=model_cfg.get('fusion_channels', 64),
            hidden_dim=model_cfg.get('hidden_dim', 256),
            dropout=model_cfg.get('dropout', 0.1)
        )
    
    def forward(self, scene_feat, signal_feat, imaging_feat):
        A1 = self.scene_encoder(scene_feat)
        A2 = self.imaging_encoder(imaging_feat)
        A3 = self.signal_encoder(signal_feat)
        
        F_unified, weights = self.fusion_module(A1, A2, A3)
        p = self.classifier(F_unified)
        
        return p, F_unified, weights


# ======================== 加载模型 ========================
def load_checkpoint(model, checkpoint_path, device, rank=0):
    """加载模型权重"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"权重文件不存在: {checkpoint_path}")
    
    if rank == 0:
        print(f"\n📥 加载模型权重: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 提取模型状态字典
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # 处理DDP包装的模型
    if isinstance(model, DDP):
        model.module.load_state_dict(state_dict, strict=True)
    else:
        model.load_state_dict(state_dict, strict=True)
    
    if rank == 0:
        print("✅ 模型权重加载成功!")
        if 'epoch' in checkpoint:
            print(f"  训练轮数: {checkpoint['epoch']}")
        if 'val_acc' in checkpoint:
            print(f"  验证准确率: {checkpoint['val_acc']:.2f}%")
        if 'val_auc' in checkpoint:
            print(f"  验证AUC: {checkpoint['val_auc']:.4f}")
    
    return model


# ======================== 测试函数 ========================
@torch.no_grad()
def test_model(model, dataloader, device, config, rank=0, save_predictions=False):
    """
    测试模型
    
    Returns:
        dict: 包含各种评估指标的字典
    """
    model.eval()
    
    # 存储预测结果
    all_predictions = []
    all_labels = []
    all_domains = []
    all_mani_types = []
    all_paths = []
    all_probabilities = []
    
    # 按域统计
    domain_stats = defaultdict(lambda: {
        'predictions': [],
        'labels': [],
        'probabilities': [],
        'correct': 0,
        'total': 0
    })
    
    # 按操作类型统计
    mani_type_stats = defaultdict(lambda: {
        'predictions': [],
        'labels': [],
        'probabilities': [],
        'correct': 0,
        'total': 0
    })
    
    if rank == 0:
        pbar = tqdm(dataloader, desc='Testing')
    else:
        pbar = dataloader
    
    for batch in pbar:
        scene_feat = batch['scene'].to(device)
        signal_feat = batch['signal'].to(device)
        imaging_feat = batch['imaging'].to(device)
        labels = batch['label'].float().to(device).unsqueeze(1)
        domains = batch['domain']
        mani_types = batch['mani_type']
        paths = batch['path']
        
        # 前向传播
        predictions, _, _ = model(scene_feat, signal_feat, imaging_feat)
        
        # 转换为numpy
        pred_probs = predictions.cpu().numpy().flatten()
        pred_labels = (pred_probs > 0.5).astype(int)
        true_labels = labels.cpu().numpy().flatten().astype(int)
        
        # 全局统计
        all_probabilities.extend(pred_probs.tolist())
        all_predictions.extend(pred_labels.tolist())
        all_labels.extend(true_labels.tolist())
        all_domains.extend(domains)
        all_mani_types.extend(mani_types)
        all_paths.extend(paths)
        
        # 按域统计
        for i in range(len(labels)):
            domain = domains[i]
            mani_type = mani_types[i]
            
            is_correct = (pred_labels[i] == true_labels[i])
            
            # 域统计
            domain_stats[domain]['predictions'].append(pred_labels[i])
            domain_stats[domain]['labels'].append(true_labels[i])
            domain_stats[domain]['probabilities'].append(pred_probs[i])
            domain_stats[domain]['correct'] += int(is_correct)
            domain_stats[domain]['total'] += 1
            
            # 操作类型统计（仅对伪造样本）
            if true_labels[i] == 1:
                mani_type_stats[mani_type]['predictions'].append(pred_labels[i])
                mani_type_stats[mani_type]['labels'].append(true_labels[i])
                mani_type_stats[mani_type]['probabilities'].append(pred_probs[i])
                mani_type_stats[mani_type]['correct'] += int(is_correct)
                mani_type_stats[mani_type]['total'] += 1
    
    # 计算全局指标
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, f1_score, 
        precision_score, recall_score, confusion_matrix,
        classification_report
    )
    
    # 全局指标
    global_metrics = {
        'accuracy': accuracy_score(all_labels, all_predictions),
        'auc': roc_auc_score(all_labels, all_probabilities) if len(np.unique(all_labels)) > 1 else 0.0,
        'f1': f1_score(all_labels, all_predictions, zero_division=0),
        'precision': precision_score(all_labels, all_predictions, zero_division=0),
        'recall': recall_score(all_labels, all_predictions, zero_division=0),
        'confusion_matrix': confusion_matrix(all_labels, all_predictions).tolist(),
        'total_samples': len(all_labels)
    }
    
    # 按域计算指标
    domain_metrics = {}
    for domain, stats in domain_stats.items():
        if stats['total'] > 0:
            preds = np.array(stats['predictions'])
            labels = np.array(stats['labels'])
            probs = np.array(stats['probabilities'])
            
            domain_metrics[domain] = {
                'accuracy': accuracy_score(labels, preds),
                'auc': roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0,
                'f1': f1_score(labels, preds, zero_division=0),
                'precision': precision_score(labels, preds, zero_division=0),
                'recall': recall_score(labels, preds, zero_division=0),
                'total_samples': stats['total'],
                'correct_samples': stats['correct']
            }
    
    # 按操作类型计算指标
    mani_type_metrics = {}
    for mani_type, stats in mani_type_stats.items():
        if stats['total'] > 0:
            preds = np.array(stats['predictions'])
            labels = np.array(stats['labels'])
            probs = np.array(stats['probabilities'])
            
            mani_type_metrics[mani_type] = {
                'accuracy': accuracy_score(labels, preds),
                'auc': roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0,
                'f1': f1_score(labels, preds, zero_division=0),
                'precision': precision_score(labels, preds, zero_division=0),
                'recall': recall_score(labels, preds, zero_division=0),
                'total_samples': stats['total'],
                'correct_samples': stats['correct']
            }
    
    # 汇总结果
    results = {
        'global_metrics': global_metrics,
        'domain_metrics': domain_metrics,
        'mani_type_metrics': mani_type_metrics,
    }
    
    # 保存预测结果
    if save_predictions and rank == 0:
        results['predictions'] = {
            'probabilities': all_probabilities.tolist(),
            'predictions': all_predictions.tolist(),
            'labels': all_labels.tolist(),
            'domains': all_domains,
            'mani_types': all_mani_types,
            'paths': all_paths
        }
    
    return results


# ======================== 打印测试结果 ========================
def print_test_results(results, dataset_name="Test"):
    """格式化打印测试结果"""
    print(f"\n{'='*80}")
    print(f"{dataset_name} 数据集测试结果")
    print(f"{'='*80}")
    
    # 全局指标
    global_metrics = results['global_metrics']
    print(f"\n📊 全局指标 (总样本数: {global_metrics['total_samples']})")
    print(f"  {'指标':<15} {'值':<10}")
    print(f"  {'-'*25}")
    print(f"  {'Accuracy':<15} {global_metrics['accuracy']*100:>8.2f}%")
    print(f"  {'AUC':<15} {global_metrics['auc']:>8.4f}")
    print(f"  {'F1 Score':<15} {global_metrics['f1']:>8.4f}")
    print(f"  {'Precision':<15} {global_metrics['precision']:>8.4f}")
    print(f"  {'Recall':<15} {global_metrics['recall']:>8.4f}")
    
    # 混淆矩阵
    cm = np.array(global_metrics['confusion_matrix'])
    print(f"\n📈 混淆矩阵:")
    print(f"              预测Real  预测Fake")
    print(f"  真实Real    {cm[0,0]:>8}  {cm[0,1]:>8}")
    print(f"  真实Fake    {cm[1,0]:>8}  {cm[1,1]:>8}")
    
    # 按域统计
    if 'domain_metrics' in results and results['domain_metrics']:
        print(f"\n🌍 按域统计:")
        print(f"  {'域名':<20} {'准确率':<10} {'AUC':<10} {'F1':<10} {'样本数':<10}")
        print(f"  {'-'*60}")
        for domain, metrics in sorted(results['domain_metrics'].items(), 
                                     key=lambda x: -x[1]['total_samples']):
            print(f"  {domain:<20} "
                  f"{metrics['accuracy']*100:>8.2f}% "
                  f"{metrics['auc']:>8.4f} "
                  f"{metrics['f1']:>8.4f} "
                  f"{metrics['total_samples']:>8}")
    
    # 按操作类型统计
    if 'mani_type_metrics' in results and results['mani_type_metrics']:
        print(f"\n🔧 按操作类型统计 (仅伪造样本):")
        print(f"  {'操作类型':<20} {'准确率':<10} {'AUC':<10} {'F1':<10} {'样本数':<10}")
        print(f"  {'-'*60}")
        for mani_type, metrics in sorted(results['mani_type_metrics'].items(), 
                                        key=lambda x: -x[1]['total_samples']):
            print(f"  {mani_type:<20} "
                  f"{metrics['accuracy']*100:>8.2f}% "
                  f"{metrics['auc']:>8.4f} "
                  f"{metrics['f1']:>8.4f} "
                  f"{metrics['total_samples']:>8}")
    
    print(f"{'='*80}\n")


# ======================== 保存结果 ========================
def save_results(results, save_dir, dataset_name="test"):
    """保存测试结果到JSON文件"""
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存指标
    metrics_file = os.path.join(save_dir, f'{dataset_name}_metrics_{timestamp}.json')
    
    # 准备可序列化的数据
    save_data = {
        'timestamp': timestamp,
        'dataset': dataset_name,
        'global_metrics': results['global_metrics'],
        'domain_metrics': results['domain_metrics'],
        'mani_type_metrics': results.get('mani_type_metrics', {})
    }
    
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    print(f"📁 指标已保存至: {metrics_file}")
    
    # 保存预测结果（如果有）
    if 'predictions' in results:
        pred_file = os.path.join(save_dir, f'{dataset_name}_predictions_{timestamp}.json')
        with open(pred_file, 'w', encoding='utf-8') as f:
            json.dump(results['predictions'], f, indent=2, ensure_ascii=False)
        print(f"📁 预测结果已保存至: {pred_file}")
    
    # 保存简要报告
    report_file = os.path.join(save_dir, f'{dataset_name}_report_{timestamp}.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"测试报告 - {dataset_name}\n")
        f.write(f"{'='*80}\n\n")
        
        f.write(f"测试时间: {timestamp}\n\n")
        
        f.write("全局指标:\n")
        for key, value in results['global_metrics'].items():
            if key != 'confusion_matrix':
                f.write(f"  {key}: {value}\n")
        
        f.write(f"\n混淆矩阵:\n{np.array(results['global_metrics']['confusion_matrix'])}\n")
        
        f.write("\n按域统计:\n")
        for domain, metrics in sorted(results['domain_metrics'].items()):
            f.write(f"  {domain}: Acc={metrics['accuracy']:.4f}, "
                   f"AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}\n")
    
    print(f"📁 测试报告已保存至: {report_file}")


# ======================== 主测试流程 ========================
def main():
    args = parse_args()
    config = load_config(args.config)
    
    # 分布式设置
    is_distributed, rank, world_size, local_rank = setup_distributed()
    
    if is_distributed:
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device(config['system']['device'])
    
    if rank == 0:
        print("\n" + "="*80)
        print("🧪 虚假图像检测测试")
        print("="*80)
        print(f"配置文件: {args.config}")
        print(f"设备: {device}")
        print(f"分布式: {is_distributed}, World Size: {world_size}")
    
    # ==================== 加载模型 ====================
    if rank == 0:
        print("\n🏗️ 初始化模型...")
    
    model = ForensicDetectionModel(config).to(device)
    
    # 加载权重
    checkpoint_path = args.checkpoint or config.get('checkpoint_path')
    if checkpoint_path is None:
        # 尝试从save_dir加载best_model.pth
        save_dir = config.get('save_dir', './checkpoints')
        checkpoint_path = os.path.join(save_dir, 'best_model.pth')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到模型权重文件: {checkpoint_path}")
    
    model = load_checkpoint(model, checkpoint_path, device, rank)
    
    # DDP包装
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], 
                   output_device=local_rank,
                   find_unused_parameters=True)
    
    # ==================== 加载测试数据集 ====================
    test_cfg = config.get('test_datasets', config.get('val_dataset'))
    
    if rank == 0:
        print(f"\n📦 加载测试数据...")
    
    # 支持多个测试集
    test_datasets_config = []
    
    if isinstance(test_cfg, dict):
        # 单个测试集
        test_datasets_config = [test_cfg]
    elif isinstance(test_cfg, list):
        # 多个测试集
        test_datasets_config = test_cfg
    
    # 结果保存目录
    save_dir = args.save_dir or config.get('log_dir', './test_results')
    save_predictions = config.get('testing', {}).get('save_predictions', False)
    
    # ==================== 测试每个数据集 ====================
    all_results = {}
    
    for test_idx, test_config in enumerate(test_datasets_config):
        json_path = test_config['path']
        target_domains = test_config.get('target_domains')
        target_mani_types = test_config.get('target_mani_types')
        
        # 数据集名称
        dataset_name = os.path.splitext(os.path.basename(json_path))[0]
        if target_domains:
            dataset_name += f"_{'_'.join(target_domains)}"
        
        if rank == 0:
            print(f"\n{'='*80}")
            print(f"测试数据集 {test_idx+1}: {dataset_name}")
            print(f"{'='*80}")
        
        # 加载数据集
        test_dataset = ForensicFeatureDataset(
            json_path=json_path,
            is_train=False,
            target_domains=target_domains,
            target_mani_types=target_mani_types,
            strict_mode=config['data'].get('strict_mode', False)
        )
        
        # 创建DataLoader
        if is_distributed:
            test_sampler = DistributedSampler(test_dataset, shuffle=False)
        else:
            test_sampler = None
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.get('testing', {}).get('batch_size', 64),
            shuffle=False,
            sampler=test_sampler,
            num_workers=config['system']['num_workers'],
            pin_memory=config['system']['pin_memory']
        )
        
        if rank == 0:
            print(f"测试样本数: {len(test_dataset)}, 批次数: {len(test_loader)}")
        
        # 测试
        results = test_model(
            model, test_loader, device, config, 
            rank=rank, save_predictions=save_predictions
        )
        
        # 打印结果
        if rank == 0:
            print_test_results(results, dataset_name)
            
            # 保存结果
            save_results(results, save_dir, dataset_name)
        
        all_results[dataset_name] = results
    
    # ==================== 汇总多数据集结果 ====================
    if rank == 0 and len(all_results) > 1:
        print(f"\n{'='*80}")
        print("📊 多数据集测试汇总")
        print(f"{'='*80}")
        print(f"  {'数据集':<30} {'准确率':<12} {'AUC':<12} {'F1':<12} {'样本数':<10}")
        print(f"  {'-'*76}")
        
        for dataset_name, results in all_results.items():
            metrics = results['global_metrics']
            print(f"  {dataset_name:<30} "
                  f"{metrics['accuracy']*100:>10.2f}% "
                  f"{metrics['auc']:>10.4f} "
                  f"{metrics['f1']:>10.4f} "
                  f"{metrics['total_samples']:>8}")
        
        print(f"{'='*80}\n")
        
        # 保存汇总结果
        summary_file = os.path.join(save_dir, 'summary_all_datasets.json')
        summary_data = {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'datasets': {
                name: results['global_metrics']
                for name, results in all_results.items()
            }
        }
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        print(f"📁 汇总结果已保存至: {summary_file}")
    
    # 清理
    cleanup_distributed()
    
    if rank == 0:
        print("\n✅ 测试完成!")


if __name__ == '__main__':
    main()