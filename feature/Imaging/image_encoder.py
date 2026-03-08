import torch
import torch.nn as nn


class ImagingEncoder(nn.Module):
    """
    成像真实性分支编码器（Imaging Authenticity Encoder）—— 预提取特征版本
    
    完整流程:
        预提取成像特征 → CNN编码器 → 输出特征
    
    输入: 
        imaging_feat ∈ ℝ^(B × N × H × W) —— 预提取的成像特征
            - 通道0:     PRNU 特征图 (1)
            - 通道1-3:   SRM 纹理响应 (3)
            - 通道4:     CFA 插值残差 (1)
            总共 N = 5 通道（根据你的提取代码）
    
    输出:
        A₂ ∈ ℝ^(B × out_channels × H × W) —— 成像真实性特征
    
    关键改动:
        ✅ 移除 ImagingFeatureExtractor（特征已预提取）
        ✅ 直接接收 N 通道特征作为输入（N=5）
        ✅ 保持 4 层浅层 CNN 架构不变
    """
    def __init__(self, in_channels=5, out_channels=64):
        """
        Args:
            in_channels: 输入特征通道数（成像特征，根据提取代码为5）
            out_channels: 输出通道数（默认64）
        """
        super().__init__()
        
        # === 移除了 ImagingFeatureExtractor ===
        
        # CNN 编码器
        # Layer 1: in_channels → 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, stride=1)
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

    def forward(self, imaging_feat):
        """
        完整前向传播
        
        Args:
            imaging_feat: [B, in_channels, H, W] torch.Tensor —— 预提取的成像特征
        
        Returns:
            out: [B, out_channels, H, W] torch.Tensor —— 成像真实性特征
        """
        # Step 1: 直接使用预提取特征（无需再提取）
        # features = imaging_feat  # [B, N, H, W]
        
        # Step 2: CNN 编码
        x = self.relu1(self.bn1(self.conv1(imaging_feat)))  # [B, 64, H, W]
        x = self.relu2(self.bn2(self.conv2(x)))             # [B, 64, H, W]
        x = self.relu3(self.bn3(self.conv3(x)))             # [B, 64, H, W]
        x = self.relu4(self.bn4(self.conv4(x)))             # [B, out_channels, H, W]
        
        return x

    @property
    def num_params(self):
        """计算总参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# # ==================== 测试代码 ====================
# if __name__ == '__main__':
#     print("="*60)
#     print("测试成像真实性编码器（预提取特征版本）")
#     print("="*60)
    
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"使用设备: {device}\n")
    
#     # 1. 创建模型
#     model = ImagingEncoder(in_channels=5, out_channels=64).to(device)
#     print(f"✓ 模型参数量: {model.num_params:,}")
    
#     # 2. 创建测试输入（模拟预提取的成像特征）
#     batch_size = 2
#     H, W = 512, 512
    
#     # 模拟预提取的成像特征: PRNU(1) + SRM(3) + CFA(1) = 5 通道
#     imaging_feat = torch.randn(batch_size, 5, H, W).to(device)
    
#     print(f"\n输入特征:")
#     print(f"  - Shape: {imaging_feat.shape}")
#     print(f"  - Range: [{imaging_feat.min():.3f}, {imaging_feat.max():.3f}]")
    
#     # 3. 前向传播
#     print("\n开始前向传播...")
#     with torch.no_grad():
#         output = model(imaging_feat)
    
#     print(f"\n输出特征:")
#     print(f"  - Shape: {output.shape}")
#     print(f"  - Range: [{output.min():.3f}, {output.max():.3f}]")
    
#     # 4. 验证形状
#     expected_shape = (batch_size, 64, H, W)
#     assert output.shape == expected_shape, \
#         f"输出形状错误! 期望 {expected_shape}, 得到 {output.shape}"
    
#     print("\n" + "="*60)
#     print("✓ 测试通过!")
#     print("="*60)