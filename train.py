import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# 导入数据处理模块
from pre_data.dataprocess import create_dataloaders, validate_and_analyze_json

# 导入各个编码器和融合模块
from Scene.scene_encoder import SceneEncoder
from Imaging.image_encoder import ImagingEncoder
from Signal.signal_encoder import SignalEncoder
from MHSA import GatedFusionModule


# ======================== 分类头定义 ========================
class ClassificationHead(nn.Module):
    """
    分类头：全局平均池化 + 全连接层
    """
    def __init__(self, in_channels=64, hidden_dim=256):
        """
        Args:
            in_channels: 输入特征通道数（统一伪影F的通道数）
            hidden_dim: 隐藏层维度
        """
        super(ClassificationHead, self).__init__()
        
        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # 全连接层
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        
        # 分类层
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, F_unified):
        """
        Args:
            F_unified: 统一伪影表征 [B, C, H, W]
        Returns:
            p: 预测概率 [B, 1]
        """
        # 全局平均池化: [B, C, H, W] -> [B, C, 1, 1]
        f_global = self.gap(F_unified)
        
        # 展平: [B, C, 1, 1] -> [B, C]
        f_global = f_global.view(f_global.size(0), -1)
        
        # 全连接层
        h = self.fc1(f_global)
        h = self.bn1(h)
        h = self.relu(h)
        
        # 分类输出
        logits = self.fc2(h)
        p = self.sigmoid(logits)
        
        return p


# ======================== 完整模型定义 ========================
class ForensicDetectionModel(nn.Module):
    """
    虚假图像检测完整模型
    """
    def __init__(self, fusion_channels=64, hidden_dim=256):
        """
        Args:
            fusion_channels: 融合后统一伪影的通道数
            hidden_dim: 分类头隐藏层维度
        """
        super(ForensicDetectionModel, self).__init__()
        
        # 三个并行编码器
        self.scene_encoder = SceneEncoder()      # 场景一致性（ViT-Tiny）
        self.imaging_encoder = ImagingEncoder()  # 成像真实性（CNN）
        self.signal_encoder = SignalEncoder()    # 信号自然性（CNN）
        
        # 门控融合模块
        self.fusion_module = GatedFusionModule(
            feature_channels=fusion_channels
        )
        
        # 分类头
        self.classifier = ClassificationHead(
            in_channels=fusion_channels,
            hidden_dim=hidden_dim
        )
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入RGB图像 [B, 3, 512, 512]
        Returns:
            p: 分类概率 [B, 1]
            F_unified: 统一伪影表征 [B, C, H, W]（用于可视化）
            weights: 融合权重 [W1, W2, W3]（用于可解释性）
        """
        # 三路特征提取
        A1 = self.scene_encoder(x)    # 场景一致性特征 [B, C, H, W]
        A2 = self.imaging_encoder(x)  # 成像真实性特征 [B, C, H, W]
        A3 = self.signal_encoder(x)   # 信号自然性特征 [B, C, H, W]
        
        # 跨维度门控融合
        F_unified, weights = self.fusion_module(A1, A2, A3)
        # F_unified: [B, C, H, W]
        # weights: (W1, W2, W3)，每个 [B, H, W]
        
        # 分类预测
        p = self.classifier(F_unified)  # [B, 1]
        
        return p, F_unified, weights


# ======================== 训练函数 ========================
def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    训练一个epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} [Train]')
    
    for batch_idx, (images, labels) in enumerate(pbar):
        # 数据移到设备
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # [B] -> [B, 1]
        
        # 前向传播
        optimizer.zero_grad()
        predictions, F_unified, weights = model(images)
        
        # 计算损失（这里先只用分类损失）
        loss = criterion(predictions, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        predicted = (predictions > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{running_loss/(batch_idx+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


# ======================== 验证函数 ========================
def validate(model, dataloader, criterion, device, epoch):
    """
    验证模型
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} [Val]')
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            # 前向传播
            predictions, _, _ = model(images)
            
            # 计算损失
            loss = criterion(predictions, labels)
            
            # 统计
            running_loss += loss.item()
            predicted = (predictions > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f'{running_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


# ======================== 主训练流程 ========================
def main():
    # ==================== 配置参数 ====================
    # 数据路径
    train_json = 'data/train.json'  # 训练集JSON文件
    val_json = 'data/val.json'      # 验证集JSON文件
    
    # 训练参数
    batch_size = 8
    num_epochs = 50
    learning_rate = 1e-4
    weight_decay = 1e-5
    image_size = 512
    num_workers = 4
    
    # 模型参数
    fusion_channels = 64
    hidden_dim = 256
    
    # 保存路径
    save_dir = 'checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n使用设备: {device}\n')
    
    # ==================== 验证JSON文件 ====================
    print("验证JSON文件格式...")
    
    if not validate_and_analyze_json(train_json):
        raise ValueError(f"训练集JSON验证失败: {train_json}")
    
    if not validate_and_analyze_json(val_json):
        raise ValueError(f"验证集JSON验证失败: {val_json}")
    
    # ==================== 数据加载 ====================
    print("\n开始加载数据集...")
    
    train_loader, val_loader = create_dataloaders(
        train_json=train_json,
        val_json=val_json,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        pin_memory=True
    )
    
    # ==================== 模型初始化 ====================
    print("\n初始化模型...")
    model = ForensicDetectionModel(
        fusion_channels=fusion_channels,
        hidden_dim=hidden_dim
    ).to(device)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  - 总参数量: {total_params:,}')
    print(f'  - 可训练参数量: {trainable_params:,}')
    
    # ==================== 损失函数和优化器 ====================
    criterion = nn.BCELoss()  # 二元交叉熵损失
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )
    
    # ==================== 训练循环 ====================
    print('\n' + '='*60)
    print('开始训练...')
    print('='*60)
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'\n{"="*60}')
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'{"="*60}')
        
        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # 验证
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # 学习率调整
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 打印结果
        print(f'\nEpoch {epoch+1} 总结:')
        print(f'  训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.2f}%')
        print(f'  验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.2f}%')
        print(f'  学习率: {current_lr:.6f}')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'train_loss': train_loss
            }
            save_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(checkpoint, save_path)
            print(f'  ✓ 最佳模型已保存! (验证准确率: {val_acc:.2f}%)')
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }
            save_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, save_path)
            print(f'  ✓ 检查点已保存!')
    
    print('\n' + '='*60)
    print('训练完成!')
    print(f'最佳验证准确率: {best_val_acc:.2f}%')
    print('='*60)


if __name__ == '__main__':
    main()