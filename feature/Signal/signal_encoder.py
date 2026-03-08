import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleCNN(nn.Module):
    """
    多尺度感知野 CNN 编码器
    
    架构:
        - 三个并行分支: 3×3@r=3, 5×5@r=2, 7×7@r=3
        - 有效感受野: 7×7, 9×9, 19×19
        - 拼接后通过 1×1 卷积融合
    """
    def __init__(self, in_channels=3, base_channels=32, use_bn=True):
        super().__init__()
        
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
            x: [B, in_channels, H, W]
        
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
    信号自然性分支编码器（Signal Naturalness Encoder）—— 预提取特征版本
    
    完整流程:
        预提取信号特征 → 多尺度CNN → 输出特征
    
    输入: 
        signal_feat ∈ ℝ^(B × 3 × H × W) —— 预提取的信号特征
            - 通道0: 局部频谱异常 (spectral)
            - 通道1: 重采样伪影 (resampling)
            - 通道2: JPEG 压缩不一致 (jpeg)
    
    输出:
        A₃ ∈ ℝ^(B × out_channels × H × W) —— 信号自然性特征
    
    关键改动:
        ✅ 移除 SignalFeatureExtractor（特征已预提取）
        ✅ 直接接收 3 通道特征作为输入
        ✅ 保持多尺度 CNN 架构不变
    """
    def __init__(self, in_channels=3, out_channels=64, base_channels=32, use_bn=True):
        """
        Args:
            in_channels: 输入特征通道数（信号特征固定为3）
            out_channels: 输出通道数（默认64）
            base_channels: 中间层通道数（默认32）
            use_bn: 是否使用 BatchNorm
        """
        super().__init__()
        
        # === 移除了 SignalFeatureExtractor ===
        
        # 多尺度 CNN 编码器
        self.multi_scale_cnn = MultiScaleCNN(
            in_channels=in_channels,
            base_channels=base_channels,
            use_bn=use_bn
        )
        
        # 最终投影层（统一通道数）
        self.proj = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        )

    def forward(self, signal_feat):
        """
        完整前向传播
        
        Args:
            signal_feat: [B, 3, H, W] torch.Tensor —— 预提取的信号特征
        
        Returns:
            out: [B, out_channels, H, W] torch.Tensor —— 信号自然性特征
        """
        # Step 1: 直接使用预提取特征（无需再提取）
        # features = signal_feat  # [B, 3, H, W]
        
        # Step 2: 多尺度 CNN 编码
        encoded = self.multi_scale_cnn(signal_feat)  # [B, 32, H, W]
        
        # Step 3: 投影到目标通道数
        out = self.proj(encoded)  # [B, out_channels, H, W]
        
        return out

    @property
    def num_params(self):
        """计算总参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# # ==================== 测试代码 ====================
# if __name__ == "__main__":
#     print("="*60)
#     print("测试信号自然性编码器（预提取特征版本）")
#     print("="*60)
    
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"使用设备: {device}\n")
    
#     # 1. 创建模型
#     model = SignalEncoder(in_channels=3, out_channels=64, base_channels=32, use_bn=True).to(device)
#     print(f"✓ 模型参数量: {model.num_params:,}")
    
#     # 2. 创建测试输入（模拟预提取的信号特征）
#     batch_size = 2
#     H, W = 512, 512
#     signal_feat = torch.randn(batch_size, 3, H, W).to(device)
    
#     print(f"\n输入特征:")
#     print(f"  - Shape: {signal_feat.shape}")
#     print(f"  - Range: [{signal_feat.min():.3f}, {signal_feat.max():.3f}]")
    
#     # 3. 前向传播
#     print("\n开始前向传播...")
#     with torch.no_grad():
#         output = model(signal_feat)
    
#     print(f"\n输出特征:")
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