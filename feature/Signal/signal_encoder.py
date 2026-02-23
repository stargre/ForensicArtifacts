import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

# 导入信号自然性特征提取器
from .Local_Spectral import extract_spectral_feature
from .Laplacian import extract_resampling_feature
from .JPEG import extract_jpeg_feature


class SignalFeatureExtractor(nn.Module):
    """
    信号自然性特征提取器
    整合三个维度的数字信号处理痕迹
    
    输入: RGB 图像 [B, 3, H, W] (已归一化)
    输出: 拼接特征 [B, 3, H, W]
        - 通道0: 局部频谱异常 M_spec
        - 通道1: 重采样伪影 M_resamp
        - 通道2: JPEG 压缩不一致 M_jpeg
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] torch.Tensor, 归一化后的 RGB 图像
        
        Returns:
            features: [B, 3, H, W] torch.Tensor
        """
        B, C, H, W = x.shape
        device = x.device
        
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
            
            # 3. 提取三个特征 (每个都是 (H, W) float32, [0,1])
            M_spec = extract_spectral_feature(img_np)
            
            M_resamp = extract_resampling_feature(img_np)
            
            M_jpeg = extract_jpeg_feature(img_np)
            
            # 4. 归一化到 [0, 1] (如果特征提取器未归一化)
            M_spec = self._normalize(M_spec)
            M_resamp = self._normalize(M_resamp)
            M_jpeg = self._normalize(M_jpeg)
            
            # 5. 堆叠为 (3, H, W)
            combined = np.stack([M_spec, M_resamp, M_jpeg], axis=0)
            
            # 6. 转为 Tensor
            combined_tensor = torch.from_numpy(combined).float()
            batch_features.append(combined_tensor)
        
        # 7. 堆叠批次
        features = torch.stack(batch_features, dim=0).to(device)  # [B, 3, H, W]
        
        return features
    
    def _normalize(self, feat):
        """归一化特征到 [0, 1]"""
        feat_min = feat.min()
        feat_max = feat.max()
        if feat_max - feat_min < 1e-8:
            return np.zeros_like(feat, dtype=np.float32)
        return ((feat - feat_min) / (feat_max - feat_min)).astype(np.float32)


class MultiScaleCNN(nn.Module):
    """
    多尺度感知野 CNN 编码器
    
    架构:
        - 三个并行分支: 3×3@r=3, 5×5@r=2, 7×7@r=3
        - 有效感受野: 7×7, 9×9, 19×19
        - 拼接后通过 1×1 卷积融合
    """
    def __init__(self, in_channels=3, base_channels=32, use_bn=True, output_sigmoid=False):
        super().__init__()
        self.output_sigmoid = output_sigmoid
        
        # Branch 1: 3×3 @ dilation=3 → 有效感受野 = 7×7
        padding1 = (3 - 1) * 3 // 2  # = 3
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, 
                     padding=padding1, dilation=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(base_channels) if use_bn else nn.Identity()
        )
        
        # Branch 2: 5×5 @ dilation=2 → 有效感受野 = 9×9
        padding2 = (5 - 1) * 2 // 2  # = 4
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=5, 
                     padding=padding2, dilation=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(base_channels) if use_bn else nn.Identity()
        )
        
        # Branch 3: 7×7 @ dilation=3 → 有效感受野 = 19×19
        padding3 = (7 - 1) * 3 // 2  # = 9
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, 
                     padding=padding3, dilation=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(base_channels) if use_bn else nn.Identity()
        )
        
        # 融合层: 1×1 卷积
        total_channels = base_channels * 3
        self.fusion = nn.Sequential(
            nn.Conv2d(total_channels, base_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(base_channels) if use_bn else nn.Identity()
        )

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W]
        
        Returns:
            out: [B, base_channels, H, W]
        """
        b1 = self.branch1(x)  # [B, C, H, W]
        b2 = self.branch2(x)  # [B, C, H, W]
        b3 = self.branch3(x)  # [B, C, H, W]
        
        out = torch.cat([b1, b2, b3], dim=1)  # [B, 3C, H, W]
        out = self.fusion(out)                # [B, C, H, W]
        
        return out


class SignalEncoder(nn.Module):
    """
    信号自然性分支编码器（Signal Naturalness Encoder）
    
    完整流程:
        输入RGB图像 → 特征提取器(3通道) → 多尺度CNN → 输出特征
    
    输入: 
        x ∈ ℝ^(B × 3 × H × W) —— 原始 RGB 图像 (已归一化)
    
    输出:
        A₃ ∈ ℝ^(B × 64 × H × W) —— 保留空间分辨率，用于后续门控融合
    
    设计依据:
        - 特征提取: 3维数字信号痕迹 (频谱/重采样/JPEG)
        - 多尺度CNN: 3个并行分支，不同感受野
        - 无下采样: 保持空间分辨率
        - 通道数: 3 → 32×3 → 64
    """
    def __init__(self, out_channels=64, base_channels=32, use_bn=True):
        super().__init__()
        
        # 特征提取器
        self.feature_extractor = SignalFeatureExtractor()
        
        # 多尺度 CNN 编码器
        self.multi_scale_cnn = MultiScaleCNN(
            in_channels=3,
            base_channels=base_channels,
            use_bn=use_bn
        )
        
        # 最终投影层 (可选，用于统一通道数)
        self.proj = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        )

    def forward(self, x):
        """
        完整前向传播
        
        Args:
            x: [B, 3, H, W] torch.Tensor —— 原始 RGB 图像（已归一化）
        
        Returns:
            out: [B, 64, H, W] torch.Tensor —— 信号自然性特征
        """
        # Step 1: 特征提取 (RGB → 3通道信号特征)
        features = self.feature_extractor(x)  # [B, 3, H, W]
        
        # Step 2: 多尺度 CNN 编码
        encoded = self.multi_scale_cnn(features)  # [B, 32, H, W]
        
        # Step 3: 投影到目标通道数
        out = self.proj(encoded)  # [B, 64, H, W]
        
        return out

    @property
    def num_params(self):
        """计算总参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# # ==================== 测试代码 ====================
# if __name__ == "__main__":
#     print("="*60)
#     print("测试信号自然性编码器")
#     print("="*60)
    
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"使用设备: {device}\n")
    
#     # 1. 创建模型
#     model = SignalEncoder(out_channels=64, base_channels=32, use_bn=True).to(device)
#     print(f"✓ 模型参数量: {model.num_params:,}")
    
#     # 2. 创建测试输入 (模拟已归一化的 RGB 图像)
#     batch_size = 2
#     H, W = 512, 512
#     dummy_input = torch.randn(batch_size, 3, H, W).to(device)
    
#     print(f"\n输入张量:")
#     print(f"  - Shape: {dummy_input.shape}")
#     print(f"  - Range: [{dummy_input.min():.3f}, {dummy_input.max():.3f}]")
    
#     # 3. 前向传播
#     print("\n开始前向传播...")
#     with torch.no_grad():
#         output = model(dummy_input)
    
#     print(f"\n输出张量:")
#     print(f"  - Shape: {output.shape}")
#     print(f"  - Range: [{output.min():.3f}, {output.max():.3f}]")
    
#     # 4. 验证形状
#     expected_shape = (batch_size, 64, H, W)
#     assert output.shape == expected_shape, \
#         f"输出形状错误! 期望 {expected_shape}, 得到 {output.shape}"
    
#     # 5. 测试多尺度分支独立性
#     print("\n测试多尺度分支:")
#     test_input = torch.randn(1, 3, 512, 512).to(device)
#     with torch.no_grad():
#         b1_out = model.multi_scale_cnn.branch1(test_input)
#         b2_out = model.multi_scale_cnn.branch2(test_input)
#         b3_out = model.multi_scale_cnn.branch3(test_input)
    
#     print(f"  - Branch 1 (3×3@r=3) output: {b1_out.shape}")
#     print(f"  - Branch 2 (5×5@r=2) output: {b2_out.shape}")
#     print(f"  - Branch 3 (7×7@r=3) output: {b3_out.shape}")
    
#     print("\n" + "="*60)
#     print("✓ 测试通过!")
#     print("="*60)