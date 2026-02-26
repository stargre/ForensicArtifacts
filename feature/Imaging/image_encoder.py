import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# 导入特征提取器
from .SRM_manipulation import extract_srm_feature
from .prnu_feature import extract_prnu_feature
from .CFA import extract_cfa_feature


class ImagingFeatureExtractor(nn.Module):
    """
    成像真实性特征提取器
    整合 PRNU、SRM、CFA 三个维度的底层成像痕迹
    
    输入: RGB 图像 [B, 3, H, W] (已归一化)
    输出: 拼接特征 [B, 32, H, W]
        - 通道0:     PRNU 特征图 (1)
        - 通道1-30:  SRM 纹理响应 (30)
        - 通道31:    CFA 插值残差 (1)
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] torch.Tensor, 归一化后的 RGB 图像
               (ImageNet normalization: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        
        Returns:
            features: [B, 32, H, W] torch.Tensor
        """
        B, C, H, W = x.shape
        device = x.device
        
        # 存储批次特征
        batch_features = []
        
        for i in range(B):
            # 1. 反归一化到 [0, 1]
            img_tensor = x[i]  # [3, H, W]
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
            img_tensor = img_tensor * std + mean
            img_tensor = torch.clamp(img_tensor, 0, 1)
            
            # 2. 转为 numpy 格式 (H, W, 3), uint8
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            
            # 3. 提取三个特征
            # PRNU: (H, W) float32
            prnu_map = extract_prnu_feature(img_np)  
            
            # SRM: (30, H, W) float32
            srm_map = extract_srm_feature(img_np)    
            
            # CFA: (H, W) float32
            cfa_map = extract_cfa_feature(img_np)    
            
            # 4. 拼接为 (32, H, W)
            prnu_map = np.expand_dims(prnu_map, axis=0)  # (1, H, W)
            cfa_map = np.expand_dims(cfa_map, axis=0)    # (1, H, W)
            
            combined = np.concatenate([prnu_map, srm_map, cfa_map], axis=0)  # (32, H, W)
            
            # 5. 转为 Tensor
            combined_tensor = torch.from_numpy(combined).float()  # (32, H, W)
            batch_features.append(combined_tensor)
        
        # 6. 堆叠批次
        features = torch.stack(batch_features, dim=0).to(device)  # [B, 32, H, W]
        
        return features


class ImagingEncoder(nn.Module):
    """
    成像真实性分支编码器（Imaging Authenticity Encoder）
    
    完整流程:
        输入RGB图像 → 特征提取器 → CNN编码器 → 输出特征
    
    输入: 
        x ∈ ℝ^(B × 3 × H × W) —— 原始 RGB 图像 (已归一化)
    
    输出:
        A₂ ∈ ℝ^(B × 64 × H × W) —— 保留空间分辨率，用于后续门控融合
    
    设计依据:
        - 特征提取: PRNU(1) + SRM(30) + CFA(1) = 32 通道
        - 4 层浅层 CNN
        - 每层: Conv(3×3, padding=1) → BatchNorm → ReLU
        - 无池化、无下采样
        - 通道数: 32 → 64 → 64 → 64 → 64
        - 参数量 ≈ 0.38M
    """
    def __init__(self, out_channels=64):
        super().__init__()
        
        # 特征提取器（整合 PRNU + SRM + CFA）
        self.feature_extractor = ImagingFeatureExtractor()
        
        # CNN 编码器
        # Layer 1: 32 → 64
        self.conv1 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Layer 2: 64 → 64
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Layer 3: 64 → 64
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        
        # Layer 4: 64 → out_channels
        self.conv4 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1, stride=1)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.relu4 = nn.ReLU(inplace=True)

        self._initialize_weights()

    def _initialize_weights(self):
        """标准 Kaiming 初始化（推荐用于 ReLU 网络）"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        完整前向传播
        
        Args:
            x: [B, 3, H, W] torch.Tensor —— 原始 RGB 图像（已归一化）
        
        Returns:
            out: [B, 64, H, W] torch.Tensor —— 成像真实性特征
        """
        # Step 1: 特征提取 (RGB → 32通道特征)
        features = self.feature_extractor(x)  # [B, 32, H, W]
        
        # Step 2: CNN 编码
        x = self.relu1(self.bn1(self.conv1(features)))  # [B, 64, H, W]
        x = self.relu2(self.bn2(self.conv2(x)))         # [B, 64, H, W]
        x = self.relu3(self.bn3(self.conv3(x)))         # [B, 64, H, W]
        x = self.relu4(self.bn4(self.conv4(x)))         # [B, 64, H, W]
        
        return x

    @property
    def num_params(self):
        """计算总参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# # ==================== 测试代码 ====================
# if __name__ == '__main__':
#     print("="*60)
#     print("测试成像真实性编码器")
#     print("="*60)
    
#     # 1. 创建模型
#     model = ImagingEncoder(out_channels=64)
#     print(f"✓ 模型参数量: {model.num_params:,}")
    
#     # 2. 创建测试输入 (模拟已归一化的 RGB 图像)
#     batch_size = 2
#     H, W = 512, 512
    
#     # 模拟 ImageNet 归一化后的图像
#     x = torch.randn(batch_size, 3, H, W)
    
#     print(f"\n输入张量:")
#     print(f"  - Shape: {x.shape}")
#     print(f"  - Range: [{x.min():.3f}, {x.max():.3f}]")
    
#     # 3. 前向传播
#     print("\n开始前向传播...")
#     with torch.no_grad():
#         output = model(x)
    
#     print(f"\n输出张量:")
#     print(f"  - Shape: {output.shape}")
#     print(f"  - Range: [{output.min():.3f}, {output.max():.3f}]")
    
#     # 4. 验证形状
#     expected_shape = (batch_size, 64, H, W)
#     assert output.shape == expected_shape, f"输出形状错误! 期望 {expected_shape}, 得到 {output.shape}"
    
#     print("\n" + "="*60)
#     print("✓ 测试通过!")
#     print("="*60)