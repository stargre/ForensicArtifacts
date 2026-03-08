# train.py

import os
import sys
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import numpy as np
from collections import defaultdict

# 导入数据处理模块
from pre_data.dataprocess import ForensicFeatureDataset

# 导入课程学习管理器
from curriculum.static_curriculum_management import StaticCurriculumManager
from curriculum.reverse_curriculum_management import ReverseCurriculumManager 
from curriculum.adaptive_curriculum_management import AdaptiveCurriculumManager

# 导入各个编码器和融合模块
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
    parser = argparse.ArgumentParser(description='虚假图像检测训练')
    parser.add_argument('--config', type=str, required=True, 
                       help='配置文件路径')
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='分布式训练本地rank')
    return parser.parse_args()


# ======================== 分布式训练设置 ========================
def setup_distributed():
    """设置分布式训练环境"""
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


# ======================== 随机种子 ========================
def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# ======================== 课程学习管理器工厂 ========================
def create_curriculum_manager(dataset, curriculum_cfg, total_training_epochs):
    """根据配置创建课程学习管理器"""
    if not curriculum_cfg.get('enabled', False):
        return None
    
    manager_type = curriculum_cfg.get('manager_type', 'static').lower()
    
    if manager_type == 'static':
        return StaticCurriculumManager(
            dataset=dataset,
            total_epochs=curriculum_cfg.get('total_epochs', total_training_epochs),
            schedule_type=curriculum_cfg.get('schedule_type', 'linear'),
            start_ratio=curriculum_cfg.get('start_ratio', 0.3),
            end_ratio=curriculum_cfg.get('end_ratio', 1.0),
            warmup_epochs=curriculum_cfg.get('warmup_epochs', 0),
        )
    
    elif manager_type == 'reverse':
        return ReverseCurriculumManager(
            dataset=dataset,
            total_epochs=curriculum_cfg.get('total_epochs', total_training_epochs),
            schedule_type=curriculum_cfg.get('schedule_type', 'linear'),
            start_ratio=curriculum_cfg.get('start_ratio', 0.3),
            end_ratio=curriculum_cfg.get('end_ratio', 1.0),
            warmup_epochs=curriculum_cfg.get('warmup_epochs', 0),
        )
    
    elif manager_type == 'adaptive':
        adaptive_cfg = curriculum_cfg.get('adaptive', {})
        return AdaptiveCurriculumManager(
            dataset=dataset,
            total_epochs=total_training_epochs,
            
            # 数据量增长
            start_ratio=curriculum_cfg.get('start_ratio', 0.3),
            end_ratio=curriculum_cfg.get('end_ratio', 1.0),
            ratio_update_frequency=adaptive_cfg.get('ratio_update_frequency', 2),
            
            # 置信度更新
            warmup_epochs=curriculum_cfg.get('warmup_epochs', 3),
            confidence_update_frequency=adaptive_cfg.get('confidence_update_frequency', 1),
            initial_weight=adaptive_cfg.get('initial_weight', 0.5),
            loss_weight=adaptive_cfg.get('loss_weight', 0.5),
            confidence_momentum=adaptive_cfg.get('confidence_momentum', 0.9),
            loss_normalization=adaptive_cfg.get('loss_normalization', 'percentile'),
            use_kl_regularization=adaptive_cfg.get('use_kl_regularization', True),
            kl_weight=adaptive_cfg.get('kl_weight', 0.1),
            
            # 采样策略
            confidence_weight_schedule=adaptive_cfg.get('confidence_weight_schedule', 'linear'),
            initial_confidence_weight=adaptive_cfg.get('initial_confidence_weight', 0.0),
            final_confidence_weight=adaptive_cfg.get('final_confidence_weight', 0.7),
            
            seed=42,
        )
    
    else:
        raise ValueError(f"未知的课程管理器类型: {manager_type}. "
                        f"支持的类型: ['static', 'reverse', 'adaptive']")


# ======================== 特征级数据增强 ========================
class FeatureAugmentation:
    """特征级数据增强"""
    def __init__(self, flip_prob=0.5, rotate_prob=0.3, 
                 channel_drop_prob=0.2, noise_prob=0.3, noise_std=0.05):
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.channel_drop_prob = channel_drop_prob
        self.noise_prob = noise_prob
        self.noise_std = noise_std
    
    def __call__(self, feat):
        if random.random() < self.flip_prob:
            feat = torch.flip(feat, dims=[-1])
        if random.random() < self.rotate_prob:
            k = random.choice([1, 2, 3])
            feat = torch.rot90(feat, k, dims=[-2, -1])
        if random.random() < self.channel_drop_prob:
            C = feat.shape[0]
            num_drop = max(1, int(C * 0.2))
            drop_idx = random.sample(range(C), num_drop)
            mask = torch.ones(C, 1, 1, device=feat.device, dtype=feat.dtype)
            mask[drop_idx] = 0
            feat = feat * mask
        if random.random() < self.noise_prob:
            noise = torch.randn_like(feat) * self.noise_std
            feat = feat + noise
        return feat


# ======================== 分类头定义 ========================
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


# ======================== 完整模型定义 ========================
class ForensicDetectionModel(nn.Module):
    """虚假图像检测完整模型"""
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


# ======================== 训练函数 ========================
def train_one_epoch(model, dataloader, criterion, optimizer, device, 
                    epoch, config, feat_aug=None, rank=0,
                    curriculum_manager=None):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    grad_clip = config['training'].get('grad_clip', 0)
    is_adaptive = isinstance(curriculum_manager, AdaptiveCurriculumManager)
    
    if rank == 0:
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} [Train]')
    else:
        pbar = dataloader
    
    for batch_idx, batch in enumerate(pbar):
        scene_feat = batch['scene'].to(device)
        signal_feat = batch['signal'].to(device)
        imaging_feat = batch['imaging'].to(device)
        labels = batch['label'].float().to(device).unsqueeze(1)
        
        # 特征级数据增强
        if feat_aug is not None:
            batch_size = scene_feat.size(0)
            for i in range(batch_size):
                scene_feat[i] = feat_aug(scene_feat[i])
                signal_feat[i] = feat_aug(signal_feat[i])
                imaging_feat[i] = feat_aug(imaging_feat[i])
        
        optimizer.zero_grad()
        
        predictions, F_unified, weights = model(scene_feat, signal_feat, imaging_feat)
        
        # 计算每个样本的损失（用于自适应课程学习）
        
        sample_losses = nn.functional.binary_cross_entropy(
            predictions, labels, reduction='none'
        ).squeeze(1)  # 只 squeeze 通道维，保持 (batch_size,)
        if sample_losses.dim() == 0:
            sample_losses = sample_losses.unsqueeze(0)

        
        loss = sample_losses.mean()
        
        # 稀疏正则化
        if config['training']['loss_weights'].get('sparse_loss', 0) > 0:
            W1, W2, W3 = weights
            sparse_loss = (W1.abs().mean() + W2.abs().mean() + W3.abs().mean())
            loss = loss + config['training']['loss_weights']['sparse_loss'] * sparse_loss
        
        loss.backward()
        
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        running_loss += loss.item()
        predicted = (predictions > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        # 记录损失用于自适应课程学习
        if is_adaptive and 'index' in batch:
            curriculum_manager.record_batch_losses(batch['index'], sample_losses)
        
        if rank == 0:
            pbar.set_postfix({
                'loss': f'{running_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


# ======================== 验证函数 ========================
def validate(model, dataloader, criterion, device, epoch, rank=0):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    
    all_preds = []
    all_labels = []
    domain_stats = defaultdict(lambda: {'preds': [], 'labels': [], 'correct': 0, 'total': 0})
    
    if rank == 0:
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} [Val]')
    else:
        pbar = dataloader
    
    with torch.no_grad():
        for batch in pbar:
            scene_feat = batch['scene'].to(device)
            signal_feat = batch['signal'].to(device)
            imaging_feat = batch['imaging'].to(device)
            labels = batch['label'].float().to(device).unsqueeze(1)
            domains = batch['domain']
            
            predictions, _, _ = model(scene_feat, signal_feat, imaging_feat)
            loss = criterion(predictions, labels)
            
            running_loss += loss.item()
            
            all_preds.extend(predictions.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            
            predicted_labels = (predictions > 0.5).float()
            for i in range(len(labels)):
                domain = domains[i]
                pred = predictions[i].item()
                label = labels[i].item()
                is_correct = (predicted_labels[i].item() == label)
                
                domain_stats[domain]['preds'].append(pred)
                domain_stats[domain]['labels'].append(label)
                domain_stats[domain]['correct'] += int(is_correct)
                domain_stats[domain]['total'] += 1
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100.0 * np.mean((all_preds > 0.5) == all_labels)
    
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
    
    try:
        val_auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        val_auc = 0.0
    
    val_f1 = f1_score(all_labels, all_preds > 0.5, zero_division=0)
    val_precision = precision_score(all_labels, all_preds > 0.5, zero_division=0)
    val_recall = recall_score(all_labels, all_preds > 0.5, zero_division=0)
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1} 验证结果")
        print(f"{'='*60}")
        print(f"全局指标:")
        print(f"  Loss:      {val_loss:.4f}")
        print(f"  Accuracy:  {val_acc:.2f}%")
        print(f"  AUC:       {val_auc:.4f}")
        print(f"  F1:        {val_f1:.4f}")
        print(f"  Precision: {val_precision:.4f}")
        print(f"  Recall:    {val_recall:.4f}")
        
        print(f"\n按域统计:")
        for domain, stats in sorted(domain_stats.items(), key=lambda x: -x[1]['total']):
            if stats['total'] > 0:
                acc = 100.0 * stats['correct'] / stats['total']
                try:
                    auc = roc_auc_score(stats['labels'], stats['preds']) if len(set(stats['labels'])) > 1 else 0.0
                except ValueError:
                    auc = 0.0
                print(f"  {domain:15s}: Acc={acc:5.1f}%  AUC={auc:.3f}  ({stats['correct']}/{stats['total']})")
        print(f"{'='*60}\n")
    
    return val_loss, val_acc, val_auc, val_f1


# ======================== 主训练流程 ========================
def main():
    args = parse_args()
    config = load_config(args.config)
    
    is_distributed, rank, world_size, local_rank = setup_distributed()
    
    if is_distributed:
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device(config['system']['device'])
    
    set_seed(config['system'].get('seed', 42))
    
    if rank == 0:
        print("\n" + "="*60)
        print("🚀 虚假图像检测训练（预提取特征版本）")
        print("="*60)
        print(f"配置文件: {args.config}")
        print(f"设备: {device}")
        print(f"分布式: {is_distributed}, World Size: {world_size}")
    
    # ==================== 数据加载 ====================
    if rank == 0:
        print("\n📦 加载特征数据集...")
    
    train_dataset = ForensicFeatureDataset(
        json_path=config['train_dataset']['path'],
        is_train=True,
        target_domains=config['train_dataset'].get('target_domains'),
        target_mani_types=config['train_dataset'].get('target_mani_types'),
        strict_mode=config['data'].get('strict_mode', False)
    )
    
    val_dataset = ForensicFeatureDataset(
        json_path=config['val_dataset']['path'],
        is_train=False,
        target_domains=config['val_dataset'].get('target_domains'),
        target_mani_types=config['val_dataset'].get('target_mani_types'),
        strict_mode=config['data'].get('strict_mode', False)
    )
    
    # ==================== 课程学习管理器 ====================
    curriculum_cfg = config.get('curriculum', {})
    total_training_epochs = config['training']['epochs']
    
    curriculum_manager = create_curriculum_manager(
        train_dataset, 
        curriculum_cfg, 
        total_training_epochs
    )
    
    is_adaptive = isinstance(curriculum_manager, AdaptiveCurriculumManager)
    
    if curriculum_manager is not None:
        manager_type = curriculum_cfg.get('manager_type', 'static')
        if rank == 0:
            print(f"\n📚 课程学习已启用 (类型: {manager_type})")
            if is_adaptive:
                print(f"  模式: 全量数据 + 难度排序训练")
    else:
        if rank == 0:
            print("\n⏭️ 课程学习已禁用，使用全部训练数据")
    
    # ==================== 验证集DataLoader ====================
    if is_distributed:
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        val_sampler = None
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        sampler=val_sampler,
        num_workers=config['system']['num_workers'],
        pin_memory=config['system']['pin_memory']
    )
    
    if rank == 0:
        print(f"训练集: {len(train_dataset)} 样本")
        print(f"验证集: {len(val_dataset)} 样本, {len(val_loader)} 批次")
    
    # ==================== 特征增强器 ====================
    use_feat_aug = config['training'].get('use_feature_augment', False)
    feat_aug = None
    if use_feat_aug:
        feat_aug = FeatureAugmentation(
            flip_prob=config['training'].get('flip_prob', 0.5),
            rotate_prob=config['training'].get('rotate_prob', 0.3),
            channel_drop_prob=config['training'].get('channel_drop_prob', 0.2),
            noise_prob=config['training'].get('noise_prob', 0.3),
            noise_std=config['training'].get('noise_std', 0.05)
        )
        if rank == 0:
            print(f"\n✅ 特征增强已启用")
    
    # ==================== 模型初始化 ====================
    if rank == 0:
        print("\n🏗️ 初始化模型...")
    
    model = ForensicDetectionModel(config).to(device)
    
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], 
                   output_device=local_rank,
                   find_unused_parameters=True)
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  总参数量: {total_params:,}")
    
    # ==================== 优化器 ====================
    opt_cfg = config['training']['optimizer']
    optimizer = optim.AdamW(
        model.parameters(),
        lr=opt_cfg['lr'],
        weight_decay=opt_cfg['weight_decay'],
        betas=tuple(opt_cfg['betas'])
    )
    
    # ==================== 学习率调度器 ====================
    sched_cfg = config['training']['scheduler']
    warmup_epochs = config['training'].get('warmup_epochs', 0)
    total_epochs = config['training']['epochs']
    
    if warmup_epochs > 0:
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
                return sched_cfg.get('eta_min', 1e-6) / opt_cfg['lr'] + \
                       (1 - sched_cfg.get('eta_min', 1e-6) / opt_cfg['lr']) * \
                       0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sched_cfg['T_max'],
            eta_min=sched_cfg['eta_min']
        )
    
    criterion = nn.BCELoss()
    
    # ==================== 训练循环 ====================
    if rank == 0:
        print("\n" + "="*60)
        print("🏃 开始训练...")
        print("="*60)
    
    best_val_acc = 0.0
    best_val_auc = 0.0
    save_dir = config.get('save_dir', 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(config['training']['epochs']):
        
        # ==================== 创建训练DataLoader ====================
        if is_adaptive:
            # 自适应课程学习：使用全量数据 + 难度排序
            train_sampler = curriculum_manager.get_sampler()
            train_sampler.set_epoch(epoch) 
            
            if rank == 0:
                stats = curriculum_manager.get_stats()
                warmup_mark = " [预热期]" if stats.get('is_warmup', False) else ""
                
                print(f"\n[Curriculum-Adaptive] Epoch {epoch+1}:{warmup_mark}")
                print(f"  数据量: {stats['num_samples']}/{stats['total_samples']} "
                    f"({stats['ratio']:.1%})")
                print(f"  置信度权重: {stats.get('confidence_weight', 0):.2f}")
                print(f"  平均置信度: {stats.get('mean_confidence', 0.5):.4f} "
                    f"(std={stats.get('std_confidence', 0):.4f})")
                print(f"  已更新样本: {stats.get('samples_updated', 0)}")
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=config['training']['batch_size'],
                sampler=train_sampler,
                num_workers=config['system']['num_workers'],
                pin_memory=config['system']['pin_memory'],
                drop_last=True
            )
        
        elif curriculum_manager is not None:
            # 静态/反向课程学习：使用子集
            current_subset = curriculum_manager.get_current_subset(epoch)
            
            if rank == 0:
                stats = curriculum_manager.get_stats()
                manager_type = curriculum_cfg.get('manager_type', 'static')
                print(f"\n[Curriculum-{manager_type}] Epoch {epoch+1}: "
                      f"使用 {stats['num_samples']}/{stats['total_samples']} 样本 "
                      f"({stats['ratio']:.1%})")
            
            if is_distributed:
                train_sampler = DistributedSampler(current_subset, shuffle=True)
                train_sampler.set_epoch(epoch)
            else:
                train_sampler = None
            
            train_loader = DataLoader(
                current_subset,
                batch_size=config['training']['batch_size'],
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                num_workers=config['system']['num_workers'],
                pin_memory=config['system']['pin_memory'],
                drop_last=True
            )
        
        else:
            # 不使用课程学习
            if is_distributed:
                train_sampler = DistributedSampler(train_dataset, shuffle=True)
                train_sampler.set_epoch(epoch)
            else:
                train_sampler = None
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=config['training']['batch_size'],
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                num_workers=config['system']['num_workers'],
                pin_memory=config['system']['pin_memory'],
                drop_last=True
            )
        
        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, 
            epoch, config, feat_aug=feat_aug, rank=rank,
            curriculum_manager=curriculum_manager
        )
        
        # 验证
        val_loss, val_acc, val_auc, val_f1 = validate(
            model, val_loader, criterion, device, epoch, rank
        )
        
        # 学习率调整
        scheduler.step()
        
        # 课程学习步进
        if curriculum_manager is not None:
            curriculum_manager.step()
        
        # 主进程保存模型
        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"\nEpoch {epoch+1} 总结:")
            print(f"  训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.2f}%")
            print(f"  验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.2f}%")
            print(f"  验证 AUC: {val_auc:.4f} | 验证 F1: {val_f1:.4f}")
            print(f"  学习率: {current_lr:.6f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_auc = val_auc
                
                model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_acc': val_acc,
                    'val_auc': val_auc,
                    'val_f1': val_f1,
                    'config': config
                }
                
                torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
                print(f"  ✓ 最佳模型已保存! (准确率: {val_acc:.2f}%, AUC: {val_auc:.4f})")
    
    cleanup_distributed()
    
    if rank == 0:
        print("\n" + "="*60)
        print("🎉 训练完成!")
        print(f"最佳验证准确率: {best_val_acc:.2f}%")
        print(f"最佳验证 AUC: {best_val_auc:.4f}")
        print("="*60)


if __name__ == '__main__':
    main()