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

from pre_data.dataprocess import ForensicFeatureDataset
from feature.Scene.scene_encoder import SceneEncoder
from feature.Imaging.image_encoder import ImagingEncoder
from feature.Signal.signal_encoder import SignalEncoder
from feature.MHSA import GatedFusionModule


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description='虚假图像检测测试')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default=None, help='模型权重路径')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--save_dir', type=str, default=None, help='结果保存目录')
    return parser.parse_args()


def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl', init_method='env://',
                                world_size=world_size, rank=rank)
        torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    else:
        return False, 0, 1, 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


class ClassificationHead(nn.Module):
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
    def __init__(self, config):
        super(ForensicDetectionModel, self).__init__()
        model_cfg = config['model']
        self.scene_encoder = SceneEncoder(
            in_channels=model_cfg.get('scene_in_channels', 4),
            out_channels=model_cfg.get('scene_channels', 64))
        self.imaging_encoder = ImagingEncoder(
            in_channels=model_cfg.get('imaging_in_channels', 32),
            out_channels=model_cfg.get('imaging_channels', 64))
        self.signal_encoder = SignalEncoder(
            in_channels=model_cfg.get('signal_in_channels', 3),
            out_channels=model_cfg.get('signal_channels', 64))
        self.fusion_module = GatedFusionModule(
            feature_channels=model_cfg.get('fusion_channels', 64),
            reduction=model_cfg.get('reduction', 4))
        self.classifier = ClassificationHead(
            in_channels=model_cfg.get('fusion_channels', 64),
            hidden_dim=model_cfg.get('hidden_dim', 256),
            dropout=model_cfg.get('dropout', 0.1))

    def forward(self, scene_feat, signal_feat, imaging_feat):
        A1 = self.scene_encoder(scene_feat)
        A2 = self.imaging_encoder(imaging_feat)
        A3 = self.signal_encoder(signal_feat)
        F_unified, weights = self.fusion_module(A1, A2, A3)
        p = self.classifier(F_unified)
        return p, F_unified, weights


# ======================== 指标函数 ========================
def compute_all_metrics(all_preds, all_labels, threshold=0.5):
    """计算全面的二分类评估指标"""
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, f1_score,
        precision_score, recall_score, confusion_matrix,
        matthews_corrcoef, balanced_accuracy_score,
        average_precision_score, cohen_kappa_score,
        log_loss
    )

    pred_labels = (all_preds > threshold).astype(int)

    cm = confusion_matrix(all_labels, pred_labels, labels=[0, 1])  
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(all_labels, pred_labels)
    precision = precision_score(all_labels, pred_labels, zero_division=0)
    recall = recall_score(all_labels, pred_labels, zero_division=0)
    f1 = f1_score(all_labels, pred_labels, zero_division=0)

    specificity = tn / (tn + fp + 1e-8)
    npv = tn / (tn + fn + 1e-8)
    fpr = fp / (fp + tn + 1e-8)
    fnr = fn / (fn + tp + 1e-8)
    balanced_acc = balanced_accuracy_score(all_labels, pred_labels)
    mcc = matthews_corrcoef(all_labels, pred_labels)
    kappa = cohen_kappa_score(all_labels, pred_labels)

    try:                                                          
        auc_roc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc_roc = 0.0
    try:                                                         
        auc_pr = average_precision_score(all_labels, all_preds)
    except ValueError:
        auc_pr = 0.0
    try:                                                          
        logloss = log_loss(all_labels, np.clip(all_preds, 1e-7, 1 - 1e-7))
    except ValueError:
        logloss = 0.0


    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'npv': npv,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'mcc': mcc,
        'kappa': kappa,
        'log_loss': logloss,
        'fpr': fpr,
        'fnr': fnr,
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'confusion_matrix': cm.tolist(),
        'threshold': threshold,
    }


def find_optimal_threshold(all_preds, all_labels):
    """在测试集上搜索最优阈值（仅用于分析参考，不用于实际判断）"""
    best_score = -1.0
    best_threshold = 0.5
    best_acc = 0.0
    best_f1 = 0.0

    for threshold in np.arange(0.20, 0.80, 0.01):
        pred_labels = (all_preds > threshold).astype(int)
        acc = np.mean(pred_labels == all_labels)
        tp = np.sum((pred_labels == 1) & (all_labels == 1))
        fp = np.sum((pred_labels == 1) & (all_labels == 0))
        fn = np.sum((pred_labels == 0) & (all_labels == 1))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        score = 0.5 * acc + 0.5 * f1
        if score > best_score:
            best_score = score
            best_acc = acc
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_acc, best_f1


def load_checkpoint(model, checkpoint_path, device, rank=0):
    """加载模型权重，同时返回训练时的最优阈值"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"权重文件不存在: {checkpoint_path}")

    if rank == 0:
        print(f"\n 加载模型权重: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    if isinstance(model, DDP):
        model.module.load_state_dict(state_dict, strict=True)
    else:
        model.load_state_dict(state_dict, strict=True)

    #  提取最优阈值
    optimal_threshold = checkpoint.get('optimal_threshold', 0.5)

    if rank == 0:
        print("✅ 模型权重加载成功!")
        if 'epoch' in checkpoint:
            print(f"  训练轮数: {checkpoint['epoch']}")
        if 'val_acc' in checkpoint:
            print(f"  验证准确率: {checkpoint['val_acc']:.2f}%")
        if 'val_auc' in checkpoint:
            print(f"  验证AUC: {checkpoint['val_auc']:.4f}")
        print(f"  最优阈值: {optimal_threshold:.2f}")       

        #  打印训练时保存的全部指标
        if 'all_metrics' in checkpoint:
            train_metrics = checkpoint['all_metrics']
            print(f"  训练时验证 MCC: {train_metrics.get('mcc', 'N/A')}")
            print(f"  训练时验证 F1:  {train_metrics.get('f1', 'N/A')}")

    return model, optimal_threshold  #  返回阈值


# ======================== 测试函数 ========================
@torch.no_grad()
def test_model(model, dataloader, device, config, rank=0,
               save_predictions=False, threshold=0.5):
    """测试模型（使用指定阈值）"""
    model.eval()

    all_probabilities = []
    all_labels = []
    all_domains = []
    all_mani_types = []
    all_paths = []

    # 按域/操作类型收集原始概率和标签
    domain_data = defaultdict(lambda: {'probs': [], 'labels': []})
    mani_type_data = defaultdict(lambda: {'probs': [], 'labels': []})

    if rank == 0:
        pbar = tqdm(dataloader, desc=f'Testing (threshold={threshold:.2f})')
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

        predictions, _, _ = model(scene_feat, signal_feat, imaging_feat)

        pred_probs = predictions.cpu().numpy().flatten()
        true_labels = labels.cpu().numpy().flatten().astype(int)

        all_probabilities.extend(pred_probs.tolist())
        all_labels.extend(true_labels.tolist())
        all_domains.extend(domains)
        all_mani_types.extend(mani_types)
        all_paths.extend(paths)

        for i in range(len(true_labels)):
            domain_data[domains[i]]['probs'].append(pred_probs[i])
            domain_data[domains[i]]['labels'].append(true_labels[i])
            if true_labels[i] == 1:
                mani_type_data[mani_types[i]]['probs'].append(pred_probs[i])
                mani_type_data[mani_types[i]]['labels'].append(true_labels[i])

    all_probabilities = np.array(all_probabilities)
    all_labels = np.array(all_labels)

    #  使用阈值计算全面指标
    global_metrics = compute_all_metrics(all_probabilities, all_labels, threshold=threshold)
    global_metrics['total_samples'] = len(all_labels)

    #  同时算默认 0.5 阈值的指标作为对比
    if abs(threshold - 0.5) > 0.01:
        default_metrics = compute_all_metrics(all_probabilities, all_labels, threshold=0.5)
    else:
        default_metrics = None

    #  搜索测试集上的最优阈值
    test_optimal_threshold, _, _ = find_optimal_threshold(all_probabilities, all_labels)
    test_optimal_metrics = compute_all_metrics(all_probabilities, all_labels,
                                               threshold=test_optimal_threshold)

    # 按域计算指标
    domain_metrics = {}
    for domain, data in domain_data.items():
        d_probs = np.array(data['probs'])
        d_labels = np.array(data['labels'])
        if len(d_labels) > 0:
            d_metrics = compute_all_metrics(d_probs, d_labels, threshold=threshold)
            d_metrics['total_samples'] = len(d_labels)
            domain_metrics[domain] = d_metrics

    # 按操作类型计算指标
    mani_type_metrics = {}
    for mani_type, data in mani_type_data.items():
        m_probs = np.array(data['probs'])
        m_labels = np.array(data['labels'])
        if len(m_labels) > 0:
            m_metrics = compute_all_metrics(m_probs, m_labels, threshold=threshold)
            m_metrics['total_samples'] = len(m_labels)
            mani_type_metrics[mani_type] = m_metrics

    results = {
        'global_metrics': global_metrics,
        'default_metrics': default_metrics,
        'test_optimal_threshold': test_optimal_threshold,
        'test_optimal_metrics': test_optimal_metrics,
        'domain_metrics': domain_metrics,
        'mani_type_metrics': mani_type_metrics,
        'threshold_used': threshold,
    }

    if save_predictions and rank == 0:
        pred_labels = (all_probabilities > threshold).astype(int)
        results['predictions'] = {
            'probabilities': all_probabilities.tolist(),
            'predictions': pred_labels.tolist(),
            'labels': all_labels.tolist(),
            'domains': all_domains,
            'mani_types': all_mani_types,
            'paths': all_paths
        }

    return results


# ======================== 打印结果 ========================
def print_test_results(results, dataset_name="Test"):
    """打印测试结果"""
    print(f"\n{'='*80}")
    print(f" {dataset_name} 数据集测试结果")
    print(f"{'='*80}")

    gm = results['global_metrics']
    threshold = results['threshold_used']
    dm = results.get('default_metrics')
    test_opt_t = results.get('test_optimal_threshold', threshold)
    test_opt_m = results.get('test_optimal_metrics')

    print(f"\n  总样本数: {gm['total_samples']}")
    print(f"  使用阈值: {threshold:.2f} (来自训练验证集)")
    if test_opt_t:
        print(f"  测试集最优阈值: {test_opt_t:.2f} (仅供参考)")

    print(f"\n  {'指标':<20}", end="")
    if dm:
        print(f"{'阈值=0.50':<14}", end="")
    print(f"{'阈值={:.2f}'.format(threshold):<14}", end="")
    if test_opt_m:
        print(f"{'测试最优={:.2f}'.format(test_opt_t):<16}", end="")
    print()
    print(f"  {'-'*62}")

    def row(name, key, pct=False):
        print(f"  {name:<20}", end="")
        fmt = "{:>10.2f}%   " if pct else "{:>10.4f}    "
        if dm:
            val = dm[key] * 100 if pct else dm[key]
            print(fmt.format(val), end="")
        val = gm[key] * 100 if pct else gm[key]
        print(fmt.format(val), end="")
        if test_opt_m:
            val = test_opt_m[key] * 100 if pct else test_opt_m[key]
            print(fmt.format(val), end="")
        print()

    row('Accuracy', 'accuracy', pct=True)
    row('Balanced Acc', 'balanced_accuracy', pct=True)
    row('Precision', 'precision')
    row('Recall(TPR)', 'recall')
    row('Specificity(TNR)', 'specificity')
    row('F1 Score', 'f1')
    row('MCC', 'mcc')
    row('Cohen Kappa', 'kappa')
    row('FPR', 'fpr')
    row('FNR(漏检率)', 'fnr')
    print(f"  {'-'*62}")
    print(f"  {'AUC-ROC':<20} {gm['auc_roc']:>10.4f}      ")
    print(f"  {'AUC-PR (AP)':<20} {gm['auc_pr']:>10.4f}   ")
    print(f"  {'Log Loss':<20} {gm['log_loss']:>10.4f}    ")

    # 混淆矩阵
    print(f"\n  混淆矩阵 (阈值={threshold:.2f}):")
    print(f"              预测Real  预测Fake")
    print(f"    真实Real  {gm['tn']:>8}  {gm['fp']:>8}")
    print(f"    真实Fake  {gm['fn']:>8}  {gm['tp']:>8}")

    # 按域统计
    if results['domain_metrics']:
        print(f"\n   按域统计 (阈值={threshold:.2f}):")
        print(f"  {'域名':<15} {'Acc':>7} {'AUC':>7} {'MCC':>7} {'F1':>7} {'FNR':>7} {'样本':>6}")
        print(f"  {'-'*58}")
        for domain, metrics in sorted(results['domain_metrics'].items(),
                                      key=lambda x: -x[1]['total_samples']):
            print(f"  {domain:<15} "
                  f"{metrics['accuracy']*100:>6.1f}% "
                  f"{metrics['auc_roc']:>6.3f} "
                  f"{metrics['mcc']:>6.3f} "
                  f"{metrics['f1']:>6.3f} "
                  f"{metrics['fnr']:>6.3f} "
                  f"{metrics['total_samples']:>6}")

    # 按操作类型统计
    if results['mani_type_metrics']:
        print(f"\n   按操作类型统计 (仅伪造样本, 阈值={threshold:.2f}):")
        print(f"  {'操作类型':<25} {'检出率':>7} {'样本':>6}")
        print(f"  {'-'*40}")
        for mani_type, metrics in sorted(results['mani_type_metrics'].items(),
                                         key=lambda x: -x[1]['total_samples']):
            detection_rate = metrics['recall']  # 对于全是 label=1 的子集，recall = 检出率
            print(f"  {mani_type:<25} "
                  f"{detection_rate*100:>6.1f}% "
                  f"{metrics['total_samples']:>6}")

    print(f"{'='*80}\n")


# ======================== 保存结果 ========================
def save_results(results, save_dir, dataset_name="test"):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存指标
    metrics_file = os.path.join(save_dir, f'{dataset_name}_metrics_{timestamp}.json')
    save_data = {
        'timestamp': timestamp,
        'dataset': dataset_name,
        'threshold_used': results['threshold_used'],
        'test_optimal_threshold': results.get('test_optimal_threshold'),
        'global_metrics': results['global_metrics'],
        'domain_metrics': results['domain_metrics'],
        'mani_type_metrics': results.get('mani_type_metrics', {})
    }
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f" 指标已保存至: {metrics_file}")

    # 保存预测结果
    if 'predictions' in results:
        pred_file = os.path.join(save_dir, f'{dataset_name}_predictions_{timestamp}.json')
        with open(pred_file, 'w', encoding='utf-8') as f:
            json.dump(results['predictions'], f, indent=2, ensure_ascii=False)
        print(f" 预测结果已保存至: {pred_file}")

    # 保存报告
    report_file = os.path.join(save_dir, f'{dataset_name}_report_{timestamp}.txt')
    gm = results['global_metrics']
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"测试报告 - {dataset_name}\n{'='*80}\n\n")
        f.write(f"测试时间: {timestamp}\n")
        f.write(f"使用阈值: {results['threshold_used']:.2f}\n\n")
        f.write("全局指标:\n")
        for key in ['accuracy', 'balanced_accuracy', 'auc_roc', 'auc_pr',
                     'f1', 'precision', 'recall', 'specificity',
                     'mcc', 'kappa', 'log_loss', 'fpr', 'fnr']:
            if key in gm:
                f.write(f"  {key}: {gm[key]:.4f}\n")
        f.write(f"\n混淆矩阵:\n{np.array(gm['confusion_matrix'])}\n")
        f.write("\n按域统计:\n")
        for domain, metrics in sorted(results['domain_metrics'].items()):
            f.write(f"  {domain}: Acc={metrics['accuracy']:.4f}, "
                    f"AUC={metrics['auc_roc']:.4f}, MCC={metrics['mcc']:.4f}, "
                    f"F1={metrics['f1']:.4f}\n")
    print(f" 测试报告已保存至: {report_file}")


# ======================== 主测试流程 ========================
def main():
    args = parse_args()
    config = load_config(args.config)

    is_distributed, rank, world_size, local_rank = setup_distributed()

    if is_distributed:
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device(config['system']['device'])

    if rank == 0:
        print("\n" + "=" * 80)
        print(" 虚假图像检测测试")
        print("=" * 80)
        print(f"配置文件: {args.config}")
        print(f"设备: {device}")
        print(f"分布式: {is_distributed}, World Size: {world_size}")

    if rank == 0:
        print("\n🏗️ 初始化模型...")

    model = ForensicDetectionModel(config).to(device)

    checkpoint_path = args.checkpoint or config.get('checkpoint_path')
    if checkpoint_path is None:
        save_dir = config.get('save_dir', './checkpoints')
        checkpoint_path = os.path.join(save_dir, 'best_model.pth')

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到模型权重文件: {checkpoint_path}")

    #  加载模型
    model, optimal_threshold = load_checkpoint(model, checkpoint_path, device, rank)

    # 测试时不需要 DDP
    if rank != 0:
        cleanup_distributed()
        return

    # ==================== 以下只在 rank 0 执行 ====================
    test_cfg = config.get('test_datasets', config.get('val_dataset'))

    print(f"\n 加载测试数据...")

    test_datasets_config = []
    if isinstance(test_cfg, dict):
        test_datasets_config = [test_cfg]
    elif isinstance(test_cfg, list):
        test_datasets_config = test_cfg

    save_dir = args.save_dir or config.get('log_dir', './test_results')
    save_predictions = config.get('testing', {}).get('save_predictions', False)

    all_results = {}

    for test_idx, test_config in enumerate(test_datasets_config):
        json_path = test_config['path']
        target_domains = test_config.get('target_domains')
        target_mani_types = test_config.get('target_mani_types')

        dataset_name = os.path.splitext(os.path.basename(json_path))[0]
        if target_domains:
            dataset_name += f"_{'_'.join(target_domains)}"

        print(f"\n{'='*80}")
        print(f"测试数据集 {test_idx+1}: {dataset_name}")
        print(f"{'='*80}")

        test_dataset = ForensicFeatureDataset(
            json_path=json_path,
            is_train=False,
            target_domains=target_domains,
            target_mani_types=target_mani_types,
            strict_mode=config['data'].get('strict_mode', False)
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config.get('testing', {}).get('batch_size', 64),
            shuffle=False,
            sampler=None,
            num_workers=config['system']['num_workers'],
            pin_memory=config['system']['pin_memory']
        )

        print(f"测试样本数: {len(test_dataset)}, 批次数: {len(test_loader)}")

        #  传入最优阈值
        results = test_model(
            model, test_loader, device, config,
            rank=0, save_predictions=save_predictions,
            threshold=optimal_threshold
        )

        print_test_results(results, dataset_name)
        save_results(results, save_dir, dataset_name)
        all_results[dataset_name] = results

    # 汇总
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print(" 多数据集测试汇总")
        print(f"{'='*80}")
        print(f"  {'数据集':<35} {'ACC':>7} {'AUC':>7} {'MCC':>7} {'F1':>7} {'样本':>6}")
        print(f"  {'-'*72}")

        for dataset_name, results in all_results.items():
            gm = results['global_metrics']
            print(f"  {dataset_name:<35} "
                  f"{gm['accuracy']*100:>6.1f}% "
                  f"{gm['auc_roc']:>6.3f} "
                  f"{gm['mcc']:>6.3f} "
                  f"{gm['f1']:>6.3f} "
                  f"{gm['total_samples']:>6}")

        print(f"{'='*80}\n")

        summary_file = os.path.join(save_dir, 'summary_all_datasets.json')
        summary_data = {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'threshold_used': optimal_threshold,
            'datasets': {
                name: results['global_metrics']
                for name, results in all_results.items()
            }
        }
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        print(f" 汇总结果已保存至: {summary_file}")

    cleanup_distributed()
    print("\n✅ 测试完成!")


if __name__ == '__main__':
    main()