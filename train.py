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
from tqdm import tqdm
import random
import numpy as np

# 导入数据处理模块
from pre_data.dataprocess import ForensicDataset

# 导入各个编码器和融合模块
from Scene.scene_encoder import SceneEncoder
from Imaging.image_encoder import ImagingEncoder
from Signal.signal_encoder import SignalEncoder
from MHSA import GatedFusionModule


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


# ======================== 分类头定义 ========================
class ClassificationHead(nn.Module):
    """分类头：全局平均池化 + 全连接层"""
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
        image_size = config['data']['image_size']
        
        # 三个并行编码器
        self.scene_encoder = SceneEncoder(
            img_size=image_size,
            out_channels=model_cfg.get('scene_channels', 64)
        )
        self.imaging_encoder = ImagingEncoder(
            out_channels=model_cfg.get('imaging_channels', 64)
        )
        self.signal_encoder = SignalEncoder(
            out_channels=model_cfg.get('signal_channels', 64)
        )
        
        # 门控融合模块
        self.fusion_module = GatedFusionModule(
            feature_channels=model_cfg.get('fusion_channels', 64),
            num_heads=model_cfg.get('num_heads', 4)
        )
        
        # 分类头
        self.classifier = ClassificationHead(
            in_channels=model_cfg.get('fusion_channels', 64),
            hidden_dim=model_cfg.get('hidden_dim', 256),
            dropout=model_cfg.get('dropout', 0.1)
        )
    
    def forward(self, x):
        """
        前向传播流程:
        1. RGB图像 → 三个编码器 → A1, A2, A3
        2. A1, A2, A3 → 门控融合 → F_unified
        3. F_unified → 分类头 → 预测概率
        """
        # 三路特征提取
        A1 = self.scene_encoder(x)    # [B, 1, H, W]
        A2 = self.imaging_encoder(x)  # [B, 64, H, W]
        A3 = self.signal_encoder(x)   # [B, 64, H, W]
        
        # 跨维度门控融合
        F_unified, weights = self.fusion_module(A1, A2, A3)  # [B, 64, H, W]
        
        # 分类预测
        p = self.classifier(F_unified)  # [B, 1]
        
        return p, F_unified, weights


# ======================== 训练函数 ========================
def train_one_epoch(model, dataloader, criterion, optimizer, device, 
                    epoch, config, rank=0):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    if rank == 0:
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} [Train]')
    else:
        pbar = dataloader
    
    for batch_idx, batch in enumerate(pbar):
        # 从字典中提取数据
        images = batch['image'].to(device)
        labels = batch['label'].float().to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        predictions, F_unified, weights = model(images)
        
        # 计算损失
        loss = criterion(predictions, labels)
        
        # 稀疏正则化（可选）
        if config['training']['loss_weights'].get('sparse_loss', 0) > 0:
            W1, W2, W3 = weights
            sparse_loss = (W1.abs().mean() + W2.abs().mean() + W3.abs().mean())
            loss = loss + config['training']['loss_weights']['sparse_loss'] * sparse_loss
        
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        predicted = (predictions > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        # 更新进度条
        if rank == 0:
            pbar.set_postfix({
                'loss': f'{running_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device, epoch, rank=0):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    if rank == 0:
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} [Val]')
    else:
        pbar = dataloader
    
    with torch.no_grad():
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].float().to(device).unsqueeze(1)
            
            predictions, _, _ = model(images)
            loss = criterion(predictions, labels)
            
            running_loss += loss.item()
            predicted = (predictions > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            if rank == 0:
                pbar.set_postfix({
                    'loss': f'{running_loss/(len(dataloader)):.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


# ======================== 主训练流程 ========================
def main():
    # 解析参数
    args = parse_args()
    config = load_config(args.config)
    
    # 设置分布式训练
    is_distributed, rank, world_size, local_rank = setup_distributed()
    
    # 设置设备
    if is_distributed:
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device(config['system']['device'])
    
    # 设置随机种子
    set_seed(config['system'].get('seed', 42))
    
    # 只在主进程打印
    if rank == 0:
        print("\n" + "="*60)
        print("🚀 虚假图像检测训练")
        print("="*60)
        print(f"配置文件: {args.config}")
        print(f"设备: {device}")
        print(f"分布式: {is_distributed}, World Size: {world_size}")
    
    # ==================== 数据加载 ====================
    if rank == 0:
        print("\n📦 加载数据集...")
    
    # 训练集
    train_dataset = ForensicDataset(
        json_path=config['train_dataset']['path'],
        image_size=config['data']['image_size'],
        norm_type=config['data']['norm_type'],
        is_train=True,
        use_mask=config['data'].get('use_mask', False),
        target_domains=config['train_dataset'].get('target_domains'),
        target_mani_types=config['train_dataset'].get('target_mani_types'),
        use_heavy_augment=config['data'].get('use_heavy_augment', False),
        strict_mode=config['data'].get('strict_mode', False)
    )
    
    # 验证集
    val_dataset = ForensicDataset(
        json_path=config['val_dataset']['path'],
        image_size=config['data']['image_size'],
        norm_type=config['data']['norm_type'],
        is_train=False,
        use_mask=config['data'].get('use_mask', False),
        target_domains=config['val_dataset'].get('target_domains'),
        target_mani_types=config['val_dataset'].get('target_mani_types'),
        strict_mode=config['data'].get('strict_mode', False)
    )
    
    # 分布式采样器
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    
    # DataLoader
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config['system']['num_workers'],
        pin_memory=config['system']['pin_memory'],
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        sampler=val_sampler,
        num_workers=config['system']['num_workers'],
        pin_memory=config['system']['pin_memory']
    )
    
    if rank == 0:
        print(f"训练集: {len(train_dataset)} 样本, {len(train_loader)} 批次")
        print(f"验证集: {len(val_dataset)} 样本, {len(val_loader)} 批次")
    
    # ==================== 模型初始化 ====================
    if rank == 0:
        print("\n🏗️ 初始化模型...")
    
    model = ForensicDetectionModel(config).to(device)
    
    # DDP包装
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
    
    # 学习率调度器
    sched_cfg = config['training']['scheduler']
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=sched_cfg['T_max'],
        eta_min=sched_cfg['eta_min']
    )
    
    # 损失函数
    criterion = nn.BCELoss()
    
    # ==================== 训练循环 ====================
    if rank == 0:
        print("\n" + "="*60)
        print("🏃 开始训练...")
        print("="*60)
    
    best_val_acc = 0.0
    save_dir = config.get('save_dir', 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(config['training']['epochs']):
        if is_distributed:
            train_sampler.set_epoch(epoch)
        
        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config, rank
        )
        
        # 验证
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch, rank
        )
        
        # 学习率调整
        scheduler.step()
        
        # 主进程保存模型
        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"\nEpoch {epoch+1} 总结:")
            print(f"  训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.2f}%")
            print(f"  验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.2f}%")
            print(f"  学习率: {current_lr:.6f}")
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                
                model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_acc': val_acc,
                    'config': config
                }
                
                torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
                print(f"  ✓ 最佳模型已保存! (准确率: {val_acc:.2f}%)")
    
    # 清理
    cleanup_distributed()
    
    if rank == 0:
        print("\n" + "="*60)
        print(" 训练完成!")
        print(f"最佳验证准确率: {best_val_acc:.2f}%")
        print("="*60)


if __name__ == '__main__':
    main()