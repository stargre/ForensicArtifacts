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

from pre_data.dataprocess import ForensicFeatureDataset
from curriculum.static_curriculum_management import StaticCurriculumManager
from curriculum.reverse_curriculum_management import ReverseCurriculumManager 
from curriculum.adaptive_curriculum_management import AdaptiveCurriculumManager
from curriculum.domainweighted_curriculum_management import DomainWeightedCurriculumManager  
from feature.Scene.scene_encoder import SceneEncoder
from feature.Imaging.image_encoder import ImagingEncoder
from feature.Signal.signal_encoder import SignalEncoder
from feature.MHSA import GatedFusionModule

# ==================== FLOPs 计算支持 ====================
try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description='虚假图像检测训练')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--local_rank', type=int, default=-1, help='分布式训练本地rank')
    return parser.parse_args()


def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
        torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    else:
        return False, 0, 1, 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def create_curriculum_manager(dataset, curriculum_cfg, total_training_epochs):
    """
    根据配置创建不同类型的课程管理器
    
    支持的类型:
    - static: 静态置信度课程学习
    - reverse: 反向课程学习
    - adaptive: 自适应课程学习
    - domain_weighted: 域权重课程学习
    """
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
            start_ratio=curriculum_cfg.get('start_ratio', 0.3),
            end_ratio=curriculum_cfg.get('end_ratio', 1.0),
            ratio_update_frequency=adaptive_cfg.get('ratio_update_frequency', 2),
            warmup_epochs=curriculum_cfg.get('warmup_epochs', 3),
            confidence_update_frequency=adaptive_cfg.get('confidence_update_frequency', 1),
            initial_weight=adaptive_cfg.get('initial_weight', 0.5),
            loss_weight=adaptive_cfg.get('loss_weight', 0.5),
            confidence_momentum=adaptive_cfg.get('confidence_momentum', 0.9),
            loss_normalization=adaptive_cfg.get('loss_normalization', 'percentile'),
            use_kl_regularization=adaptive_cfg.get('use_kl_regularization', True),
            kl_weight=adaptive_cfg.get('kl_weight', 0.1),
            confidence_weight_schedule=adaptive_cfg.get('confidence_weight_schedule', 'linear'),
            initial_confidence_weight=adaptive_cfg.get('initial_confidence_weight', 0.0),
            final_confidence_weight=adaptive_cfg.get('final_confidence_weight', 0.7),
            seed=42,
        )
    
    elif manager_type == 'domain_weighted':
        domain_cfg = curriculum_cfg.get('domain_weighted', {})
        return DomainWeightedCurriculumManager(
            dataset=dataset,
            total_epochs=total_training_epochs,
            domain_names=domain_cfg.get('domain_names', None),
            initial_domain_weights=domain_cfg.get('initial_domain_weights', None),
            difficulty_metric=domain_cfg.get('difficulty_metric', 'auc'),
            min_domain_weight=domain_cfg.get('min_domain_weight', 0.15),
            warmup_epochs=domain_cfg.get('warmup_epochs', 3),
            dro_start_epoch=domain_cfg.get('dro_start_epoch', 5),
            dro_weight=domain_cfg.get('dro_weight', 0.0),
            dro_final_weight=domain_cfg.get('dro_final_weight', 0.8),
            eta=domain_cfg.get('eta', 1.0),
            start_ratio=domain_cfg.get('start_ratio', 0.5),
            end_ratio=domain_cfg.get('end_ratio', 1.0),
            plateau_epochs=domain_cfg.get('plateau_epochs', 5),
            loss_ema_decay=domain_cfg.get('loss_ema_decay', 0.9),
            seed=domain_cfg.get('seed', 42),
        )
    
    else:
        raise ValueError(f"未知的课程管理器类型: {manager_type}，支持: static, reverse, adaptive, domain_weighted")


class FeatureAugmentation:
    """特征增强"""
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


class ClassificationHead(nn.Module):
    """GAP+GMP双池化 + 单层MLP（与旧版结构一致，仅增加GMP）"""
    def __init__(self, in_channels=64, hidden_dim=256, dropout=0.1):
        super(ClassificationHead, self).__init__()
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        
        concat_dim = in_channels * 2
        
        self.fc1 = nn.Linear(concat_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, F_unified):
        f_avg = self.gap(F_unified).view(F_unified.size(0), -1)
        f_max = self.gmp(F_unified).view(F_unified.size(0), -1)
        f_combined = torch.cat([f_avg, f_max], dim=1)
        
        h = self.relu(self.bn1(self.fc1(f_combined)))
        h = self.dropout(h)
        p = self.sigmoid(self.fc2(h))
        return p


class ForensicDetectionModel(nn.Module):
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


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred, target):
        bce = nn.functional.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.where(target == 1, pred, 1 - pred)
        alpha_weight = torch.where(target == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha_weight * (1 - pt) ** self.gamma
        loss = focal_weight * bce
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'none':
            return loss.squeeze(1) if loss.dim() > 1 else loss
        return loss.sum()


class EarlyStopping:
    """早停机制：当监控指标连续 patience 个 epoch 没有改善时停止训练"""
    
    def __init__(self, patience=10, min_delta=0.001, monitor='val_acc', verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.verbose = verbose
        self.mode = 'min' if 'loss' in monitor else 'max'
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == 'max':
            improved = score > (self.best_score + self.min_delta)
        else:
            improved = score < (self.best_score - self.min_delta)
        
        if improved:
            if self.verbose:
                print(f"  ↑ {self.monitor} 改善: {self.best_score:.4f} → {score:.4f} "
                      f"(+{abs(score - self.best_score):.4f}), 计数器重置")
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"  → {self.monitor} 未改善 ({self.counter}/{self.patience}), "
                      f"当前最佳: {self.best_score:.4f} @ Epoch {self.best_epoch + 1}")
            
            if self.counter >= self.patience:
                if not self.early_stop:
                    self.early_stop = True
                    if self.verbose:
                        print(f"\n  ⚠ 早停触发！连续 {self.patience} 个 epoch 无改善")
                        print(f"    最佳 {self.monitor}: {self.best_score:.4f} @ Epoch {self.best_epoch + 1}")
        
        return self.early_stop


def find_optimal_threshold(all_preds, all_labels):
    """在验证集上搜索最优分类阈值"""
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


def compute_all_metrics(all_preds, all_labels, threshold=0.5):
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, f1_score, 
        precision_score, recall_score, confusion_matrix,
        matthews_corrcoef, balanced_accuracy_score,
        average_precision_score, cohen_kappa_score,
        log_loss
    )
    
    pred_labels = (all_preds > threshold).astype(int)
    
    cm = confusion_matrix(all_labels, pred_labels)
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
    
    auc_roc = roc_auc_score(all_labels, all_preds)
    auc_pr = average_precision_score(all_labels, all_preds)
    logloss = log_loss(all_labels, np.clip(all_preds, 1e-7, 1-1e-7))
    
    metrics = {
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
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'confusion_matrix': cm.tolist(),
        'threshold': threshold,
    }
    
    return metrics


# ==================== 新增：FLOPs/Params 计算函数 ====================
def compute_model_flops_from_dataloader(model, dataloader, device):
    """
    使用 dataloader 中真实 batch 的单个样本计算 FLOPs/Params
    优点：输入尺寸与真实数据一致
    """
    if not THOP_AVAILABLE:
        return None
    
    model_for_profile = model.module if hasattr(model, 'module') else model
    model_for_profile.eval()

    try:
        batch = next(iter(dataloader))
        scene_feat = batch['scene'][:1].to(device)
        signal_feat = batch['signal'][:1].to(device)
        imaging_feat = batch['imaging'][:1].to(device)

        with torch.no_grad():
            flops, params = profile(
                model_for_profile,
                inputs=(scene_feat, signal_feat, imaging_feat),
                verbose=False
            )

        flops_str, params_str = clever_format([flops, params], "%.3f")

        return {
            'flops': flops,
            'params': params,
            'flops_str': flops_str,
            'params_str': params_str,
            'input_shape': {
                'scene': list(scene_feat.shape),
                'signal': list(signal_feat.shape),
                'imaging': list(imaging_feat.shape)
            }
        }
    except Exception as e:
        return {'error': str(e)}


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
    is_domain_weighted = isinstance(curriculum_manager, DomainWeightedCurriculumManager)
    
    if isinstance(criterion, FocalLoss):
        criterion_none = FocalLoss(
            alpha=criterion.alpha, gamma=criterion.gamma, reduction='none'
        )

    if rank == 0:
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} [Train]')
    else:
        pbar = dataloader
    
    for batch_idx, batch in enumerate(pbar):
        scene_feat = batch['scene'].to(device)
        signal_feat = batch['signal'].to(device)
        imaging_feat = batch['imaging'].to(device)
        labels = batch['label'].float().to(device).unsqueeze(1)
        
        domains = batch.get('domain', None)
        
        if feat_aug is not None:
            batch_size = scene_feat.size(0)
            for i in range(batch_size):
                scene_feat[i] = feat_aug(scene_feat[i])
                signal_feat[i] = feat_aug(signal_feat[i])
                imaging_feat[i] = feat_aug(imaging_feat[i])
        
        optimizer.zero_grad()
        predictions, F_unified, weights = model(scene_feat, signal_feat, imaging_feat)
        
        if isinstance(criterion, FocalLoss):
            sample_losses_for_curriculum = criterion_none(predictions, labels)
            loss = criterion(predictions, labels)
        else:
            sample_losses_for_curriculum = nn.functional.binary_cross_entropy(
                predictions, labels, reduction='none'
            ).squeeze(1)
            if sample_losses_for_curriculum.dim() == 0:
                sample_losses_for_curriculum = sample_losses_for_curriculum.unsqueeze(0)
            loss = sample_losses_for_curriculum.mean()
        
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
        
        if is_adaptive and 'index' in batch:
            curriculum_manager.record_batch_losses(
                batch['index'], 
                sample_losses_for_curriculum.detach()
            )
        elif is_domain_weighted and domains is not None:
            curriculum_manager.record_batch_losses(
                domains, 
                sample_losses_for_curriculum.detach(),
                predictions.detach(),
                labels.detach()
            )  
        
        if rank == 0:
            pbar.set_postfix({
                'loss': f'{running_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device, epoch, rank=0):
    """验证函数"""
    model.eval()
    running_loss = 0.0
    
    all_preds = []
    all_labels = []
    domain_stats = defaultdict(lambda: {'preds': [], 'labels': []})
    
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
            
            for i in range(len(labels)):
                domain = domains[i]
                domain_stats[domain]['preds'].append(predictions[i].item())
                domain_stats[domain]['labels'].append(labels[i].item())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    val_loss = running_loss / len(dataloader)
    
    optimal_threshold, _, _ = find_optimal_threshold(all_preds, all_labels)
    
    metrics_optimal = compute_all_metrics(all_preds, all_labels, threshold=optimal_threshold)
    metrics_default = compute_all_metrics(all_preds, all_labels, threshold=0.5)
    
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1} 验证结果")
        print(f"{'='*70}")
        print(f"  Loss: {val_loss:.4f}")
        print(f"")
        print(f"  {'指标':<20} {'阈值=0.50':<14} {'阈值={:.2f}'.format(optimal_threshold):<14}")
        print(f"  {'-'*48}")
        print(f"  {'Accuracy':<20} {metrics_default['accuracy']*100:>10.2f}%   {metrics_optimal['accuracy']*100:>10.2f}%")
        print(f"  {'Balanced Acc':<20} {metrics_default['balanced_accuracy']*100:>10.2f}%   {metrics_optimal['balanced_accuracy']*100:>10.2f}%")
        print(f"  {'Precision':<20} {metrics_default['precision']:>10.4f}    {metrics_optimal['precision']:>10.4f}")
        print(f"  {'Recall(TPR)':<20} {metrics_default['recall']:>10.4f}    {metrics_optimal['recall']:>10.4f}")
        print(f"  {'Specificity(TNR)':<20} {metrics_default['specificity']:>10.4f}    {metrics_optimal['specificity']:>10.4f}")
        print(f"  {'F1 Score':<20} {metrics_default['f1']:>10.4f}    {metrics_optimal['f1']:>10.4f}")
        print(f"  {'MCC':<20} {metrics_default['mcc']:>10.4f}    {metrics_optimal['mcc']:>10.4f}")
        print(f"  {'Cohen Kappa':<20} {metrics_default['kappa']:>10.4f}    {metrics_optimal['kappa']:>10.4f}")
        print(f"  {'FPR':<20} {metrics_default['fpr']:>10.4f}    {metrics_optimal['fpr']:>10.4f}")
        print(f"  {'FNR(漏检率)':<20} {metrics_default['fnr']:>10.4f}    {metrics_optimal['fnr']:>10.4f}")
        print(f"  {'-'*48}")
        print(f"  {'AUC-ROC':<20} {metrics_optimal['auc_roc']:>10.4f}    (不依赖阈值)")
        print(f"  {'AUC-PR (AP)':<20} {metrics_optimal['auc_pr']:>10.4f}    (不依赖阈值)")
        print(f"  {'Log Loss':<20} {metrics_optimal['log_loss']:>10.4f}    (不依赖阈值)")
        
        print(f"\n  混淆矩阵 (阈值={optimal_threshold:.2f}):")
        print(f"              预测Real  预测Fake")
        print(f"    真实Real  {metrics_optimal['tn']:>8}  {metrics_optimal['fp']:>8}")
        print(f"    真实Fake  {metrics_optimal['fn']:>8}  {metrics_optimal['tp']:>8}")
        
        real_preds = all_preds[all_labels == 0]
        fake_preds = all_preds[all_labels == 1]
        print(f"\n  预测概率分布:")
        print(f"    Real样本: mean={real_preds.mean():.3f}, std={real_preds.std():.3f}, "
              f"median={np.median(real_preds):.3f}")
        print(f"    Fake样本: mean={fake_preds.mean():.3f}, std={fake_preds.std():.3f}, "
              f"median={np.median(fake_preds):.3f}")
        
        print(f"\n  按域统计 (阈值={optimal_threshold:.2f}):")
        print(f"  {'域名':<15} {'Acc':>7} {'AUC':>7} {'MCC':>7} {'F1':>7} {'样本':>6}")
        print(f"  {'-'*50}")
        for domain, stats in sorted(domain_stats.items(), key=lambda x: -len(x[1]['labels'])):
            d_preds = np.array(stats['preds'])
            d_labels = np.array(stats['labels'])
            if len(d_labels) > 0:
                d_metrics = compute_all_metrics(d_preds, d_labels, threshold=optimal_threshold)
                print(f"  {domain:<15} "
                      f"{d_metrics['accuracy']*100:>6.1f}% "
                      f"{d_metrics['auc_roc']:>6.3f} "
                      f"{d_metrics['mcc']:>6.3f} "
                      f"{d_metrics['f1']:>6.3f} "
                      f"{len(d_labels):>6}")
        print(f"{'='*70}\n")
    
    val_acc = metrics_optimal['accuracy'] * 100.0
    val_auc = metrics_optimal['auc_roc']
    val_f1 = metrics_optimal['f1']
    
    return val_loss, val_acc, val_auc, val_f1, optimal_threshold, metrics_optimal


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
        print(" 虚假图像检测训练（预提取特征）")
        print("="*60)
    
    # ==================== 数据集加载 ====================
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
    
    curriculum_cfg = config.get('curriculum', {})
    total_training_epochs = config['training']['epochs']
    curriculum_manager = create_curriculum_manager(train_dataset, curriculum_cfg, total_training_epochs)
    
    is_adaptive = isinstance(curriculum_manager, AdaptiveCurriculumManager)
    is_domain_weighted = isinstance(curriculum_manager, DomainWeightedCurriculumManager)
    
    if rank == 0 and curriculum_manager is not None:
        manager_type = curriculum_cfg.get('manager_type', 'static')
        print(f"\n  课程学习类型: {manager_type}")
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        sampler=None,
        num_workers=config['system']['num_workers'],
        pin_memory=config['system']['pin_memory']
    )
    
    # ==================== 模型初始化 ====================
    model = ForensicDetectionModel(config).to(device)
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  总参数量: {total_params:,}")
        print(f"  可训练参数量: {trainable_params:,}")
    
        # ==================== FLOPs 计算 ====================
        print(f"  正在计算模型 FLOPs / Params ...")
        flops_info = compute_model_flops_from_dataloader(model, val_loader, device)
        
        if flops_info is None:
            print("  ⚠ 未安装 thop，跳过 FLOPs 计算。请执行: pip install thop")
        elif 'error' in flops_info:
            print(f"  ⚠ FLOPs 计算失败: {flops_info['error']}")
        else:
            print(f"  输入尺寸:")
            print(f"    Scene:   {flops_info['input_shape']['scene']}")
            print(f"    Signal:  {flops_info['input_shape']['signal']}")
            print(f"    Imaging: {flops_info['input_shape']['imaging']}")
            print(f"  模型参数量(THOP): {flops_info['params_str']}")
            print(f"  模型 FLOPs:       {flops_info['flops_str']}")
    
    # ==================== 优化器初始化 ====================
    opt_cfg = config['training']['optimizer']
    optimizer = optim.AdamW(
        model.parameters(),
        lr=opt_cfg['lr'],
        weight_decay=opt_cfg['weight_decay'],
        betas=tuple(opt_cfg['betas'])
    )
    
    # ==================== 调度器初始化 ====================
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
            optimizer, T_max=sched_cfg['T_max'], eta_min=sched_cfg['eta_min']
        )
    
    # ==================== 损失函数 ====================
    loss_cfg = config['training'].get('loss', {})
    loss_type = loss_cfg.get('type', 'bce')
    
    if loss_type == 'focal':
        criterion = FocalLoss(
            alpha=loss_cfg.get('focal_alpha', 0.75),
            gamma=loss_cfg.get('focal_gamma', 2.0)
        )
        if rank == 0:
            print(f"  使用 Focal Loss (alpha={loss_cfg.get('focal_alpha', 0.75)}, "
                  f"gamma={loss_cfg.get('focal_gamma', 2.0)})")
    else:
        criterion = nn.BCELoss()
        if rank == 0:
            print(f"  使用 BCE Loss")
    
    # ======================== 断点续训加载 ========================
    start_epoch = 0
    best_val_acc = 0.0
    best_val_auc = 0.0
    best_threshold = 0.5
    
    checkpoint_path = config.get('checkpoint_path', None)
    resume = config.get('resume', False)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        if rank == 0:
            if resume:
                print(f"\n  📥 从断点恢复训练: {checkpoint_path}")
            else:
                print(f"\n  📥 加载预训练权重: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            if resume:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
                start_epoch = checkpoint['epoch']
                best_val_acc = checkpoint.get('val_acc', 0.0)
                best_val_auc = checkpoint.get('val_auc', 0.0)
                best_threshold = checkpoint.get('optimal_threshold', 0.5)
                
                if rank == 0:
                    print(f"  ✅ 完全恢复成功!")
                    print(f"     起始 Epoch: {start_epoch}")
                    print(f"     最佳验证准确率: {best_val_acc:.2f}%")
                    print(f"     最佳验证 AUC: {best_val_auc:.4f}")
                    print(f"     最佳阈值: {best_threshold:.2f}")
                    print(f"     当前学习率: {optimizer.param_groups[0]['lr']:.6f}")
            else:
                if rank == 0:
                    print(f"  ✅ 权重加载成功! (训练状态重置)")
                    print(f"     起始 Epoch: 0")
                    print(f"     学习率: {opt_cfg['lr']:.6f} (初始值)")
        
        except Exception as e:
            if rank == 0:
                print(f"  ⚠️  加载检查点失败: {e}")
                print(f"  从头开始训练...")
            start_epoch = 0
            best_val_acc = 0.0
            best_val_auc = 0.0
            best_threshold = 0.5
    elif checkpoint_path and not os.path.exists(checkpoint_path):
        if rank == 0:
            print(f"\n  ⚠️  检查点文件不存在: {checkpoint_path}")
            print(f"  从头开始训练...")
    
    # ==================== DDP 包装 ====================
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    # ==================== 特征增强 ====================
    use_feat_aug = config['training'].get('use_feature_augment', False)
    feat_aug = FeatureAugmentation(
        flip_prob=config['training'].get('flip_prob', 0.5),
        rotate_prob=config['training'].get('rotate_prob', 0.3),
        channel_drop_prob=config['training'].get('channel_drop_prob', 0.2),
        noise_prob=config['training'].get('noise_prob', 0.3),
        noise_std=config['training'].get('noise_std', 0.05),
    ) if use_feat_aug else None
    
    # ==================== 早停初始化 ====================
    es_cfg = config['training'].get('early_stopping', {})
    early_stopper = None
    if es_cfg.get('enabled', False):
        early_stopper = EarlyStopping(
            patience=es_cfg.get('patience', 10),
            min_delta=es_cfg.get('min_delta', 0.001),
            monitor=es_cfg.get('monitor', 'val_acc'),
            verbose=(rank == 0)
        )
        if rank == 0:
            print(f"  早停已启用: monitor={es_cfg.get('monitor', 'val_acc')}, "
                  f"patience={es_cfg.get('patience', 10)}, "
                  f"min_delta={es_cfg.get('min_delta', 0.001)}")
    
    # ==================== 保存目录 ====================
    save_dir = config.get('save_dir', 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    
    # ======================== 训练循环 ========================
    final_epoch = 0
    
    for epoch in range(start_epoch, config['training']['epochs']):
        final_epoch = epoch + 1
        
        if is_adaptive:
            train_sampler = curriculum_manager.get_sampler()
            train_sampler.set_epoch(epoch)
            train_loader = DataLoader(
                train_dataset,
                batch_size=config['training']['batch_size'],
                sampler=train_sampler,
                num_workers=config['system']['num_workers'],
                pin_memory=config['system']['pin_memory'],
                drop_last=True
            )
        
        elif is_domain_weighted:
            train_sampler = curriculum_manager.get_sampler()
            train_sampler.set_epoch(epoch)
            train_loader = DataLoader(
                train_dataset,
                batch_size=config['training']['batch_size'],
                sampler=train_sampler,
                num_workers=config['system']['num_workers'],
                pin_memory=config['system']['pin_memory'],
                drop_last=True
            )
        
        elif curriculum_manager is not None:
            current_subset = curriculum_manager.get_current_subset(epoch)
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
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, config, feat_aug=feat_aug, rank=rank,
            curriculum_manager=curriculum_manager
        )
        
        val_loss, val_acc, val_auc, val_f1, optimal_threshold, all_metrics = validate(
            model, val_loader, criterion, device, epoch, rank
        )
        
        scheduler.step()
        
        if curriculum_manager is not None:
            curriculum_manager.step()
            
            if is_domain_weighted and rank == 0:
                stats = curriculum_manager.get_stats()
                print(f"\n[Domain Curriculum Stats]")
                print(f"  数据比例: {stats['data_ratio']:.1%} ({stats['total_samples']} 样本)")
                print(f"  DRO权重: {stats['dro_weight']:.3f}")
                print(f"  域权重: ", end="")
                for d, w in stats['domain_weights'].items():
                    print(f"{d}={w:.3f} ", end="")
                print()
                print(f"  域Loss EMA: ", end="")
                for d, l in stats['domain_losses'].items():
                    print(f"{d}={l:.4f} ", end="")
                print()
        
        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch+1} 总结:")
            print(f"  训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.2f}%")
            print(f"  验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.2f}%")
            print(f"  验证 AUC-ROC: {val_auc:.4f} | AUC-PR: {all_metrics['auc_pr']:.4f}")
            print(f"  验证 F1: {val_f1:.4f} | MCC: {all_metrics['mcc']:.4f} | Kappa: {all_metrics['kappa']:.4f}")
            print(f"  最优阈值: {optimal_threshold:.2f}")
            print(f"  学习率: {current_lr:.6f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_auc = val_auc
                best_threshold = optimal_threshold
                
                model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_acc': val_acc,
                    'val_auc': val_auc,
                    'val_f1': val_f1,
                    'val_mcc': all_metrics['mcc'],
                    'val_auc_pr': all_metrics['auc_pr'],
                    'val_kappa': all_metrics['kappa'],
                    'optimal_threshold': optimal_threshold,
                    'all_metrics': all_metrics,
                    'config': config
                }
                torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
                print(f"  ✓ 最佳模型已保存! "
                      f"(ACC: {val_acc:.2f}%, AUC: {val_auc:.4f}, "
                      f"MCC: {all_metrics['mcc']:.4f}, Threshold: {optimal_threshold:.2f})")
            
            save_freq = config.get('logging', {}).get('save_freq', 5)
            if (epoch + 1) % save_freq == 0:
                model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_acc': val_acc,
                    'val_auc': val_auc,
                    'val_f1': val_f1,
                    'val_mcc': all_metrics['mcc'],
                    'val_auc_pr': all_metrics['auc_pr'],
                    'val_kappa': all_metrics['kappa'],
                    'optimal_threshold': optimal_threshold,
                    'all_metrics': all_metrics,
                    'config': config
                }
                checkpoint_name = f'checkpoint_epoch_{epoch+1}.pth'
                torch.save(checkpoint, os.path.join(save_dir, checkpoint_name))
                print(f"  💾 检查点已保存: {checkpoint_name}")
        
        if early_stopper is not None:
            monitor = es_cfg.get('monitor', 'val_acc')
            monitor_map = {
                'val_acc': val_acc,
                'val_auc': val_auc,
                'val_f1': val_f1,
                'val_loss': val_loss,
                'val_mcc': all_metrics['mcc'],
                'val_auc_pr': all_metrics['auc_pr'],
                'val_kappa': all_metrics['kappa'],
            }
            current_score = monitor_map.get(monitor, val_acc)
            
            should_stop = early_stopper(current_score, epoch)
            
            if is_distributed:
                stop_tensor = torch.tensor([1.0 if should_stop else 0.0], device=device)
                dist.broadcast(stop_tensor, src=0)
                should_stop = stop_tensor.item() > 0.5
            
            if should_stop:
                if rank == 0:
                    print(f"\n{'='*60}")
                    print(f" 早停! 在 Epoch {epoch+1} 停止训练")
                    print(f" 最佳 {monitor}: {early_stopper.best_score:.4f} "
                          f"@ Epoch {early_stopper.best_epoch + 1}")
                    print(f"{'='*60}")
                break
    
    cleanup_distributed()
    
    if rank == 0:
        print("\n" + "="*60)
        print(" 训练完成!")
        print(f"  最佳验证准确率: {best_val_acc:.2f}%")
        print(f"  最佳验证 AUC: {best_val_auc:.4f}")
        print(f"  最佳阈值: {best_threshold:.2f}")
        if early_stopper is not None and early_stopper.early_stop:
            print(f"  训练因早停在 Epoch {early_stopper.best_epoch + 1} 后终止 "
                  f"(实际运行至 Epoch {final_epoch})")
        else:
            print(f"  训练完整运行了 {final_epoch} 个 epoch")
        print("="*60)


if __name__ == '__main__':
    main()