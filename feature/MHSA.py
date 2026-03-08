import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention2D(nn.Module):
    """
    标准 2D 多头自注意力模块
    输入: (B, C_in, H, W)
    输出: (B, C_out, H, W)
    """
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x: (B, C, H, W) → (B, H*W, C)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, C), N = H*W

        Q = self.q_proj(x)  # (B, N, C)
        K = self.k_proj(x)  # (B, N, C)
        V = self.v_proj(x)  # (B, N, C)

        # Split into heads
        Q = Q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)

        out = attn @ V  # (B, num_heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, -1, self.embed_dim)
        out = self.out_proj(out)
        out = out.transpose(1, 2).view(B, self.embed_dim, H, W)
        
        return out


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    Input: (B, C, H, W)
    Output: (B, C, H, W) with channel-wise weights
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, _, _ = x.shape
        y = self.avg_pool(x).view(B, C)      # (B, C)
        y = self.fc(y).view(B, C, 1, 1)      # (B, C, 1, 1)
        return x * y                         # 广播到 (B, C, H, W)


class GatedFusionModule(nn.Module):
    """
    跨维度门控协同融合模块（SE注意力平替版）
    
    输入:
        A1: (B, C, H, W)  —— Scene consistency
        A2: (B, C, H, W)  —— Imaging authenticity
        A3: (B, C, H, W)  —— Signal naturalness
    
    输出:
        F_unified: (B, C, H, W)
        weights: (w1, w2, w3), 每个 (B, 1, H, W)
    """
    def __init__(self, feature_channels=64, reduction=4):
        """
        Args:
            feature_channels: 输入特征通道数（三个分支统一）
            reduction: SE 模块的压缩比（默认 4，即 bottleneck 维度 = C//4）
        """
        super().__init__()
        self.feature_channels = feature_channels

        # 可选投影层（保持灵活性）
        self.proj1 = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)
        self.proj2 = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)
        self.proj3 = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)

        # 拼接后通道数
        fuse_channels = 3 * feature_channels  # 192

        # 用 SE 替代 MHSA：对拼接特征做通道注意力
        self.se_block = SEBlock(fuse_channels, reduction=reduction)

        # 权重预测头（从 SE 增强后的特征预测每个分支的权重）
        self.weight_head = nn.Conv2d(fuse_channels, 3, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, A1, A2, A3):
        """
        Args:
            A1: [B, 64, H, W]
            A2: [B, 64, H, W]
            A3: [B, 64, H, W]
        
        Returns:
            F_unified: [B, 64, H, W]
            weights: (w1, w2, w3), 每个 [B, 1, H, W]
        """
        # Step 1: 投影（可选，但保留以增强表达）
        A1_hat = self.proj1(A1)  # [B, 64, H, W]
        A2_hat = self.proj2(A2)
        A3_hat = self.proj3(A3)

        # Step 2: 拼接 → [B, 192, H, W]
        X_fuse = torch.cat([A1_hat, A2_hat, A3_hat], dim=1)

        # Step 3: SE 注意力增强（替代 MHSA）
        X_enhanced = self.se_block(X_fuse)  # [B, 192, H, W]

        # Step 4: 预测权重 → [B, 3, H, W]
        weights = self.weight_head(X_enhanced)
        weights = F.softmax(weights, dim=1)  # w1 + w2 + w3 = 1 per pixel

        # Step 5: 分割权重
        w1, w2, w3 = torch.chunk(weights, 3, dim=1)  # each [B, 1, H, W]

        # Step 6: 加权融合
        F_unified = w1 * A1_hat + w2 * A2_hat + w3 * A3_hat  # [B, 64, H, W]

        return F_unified, (w1, w2, w3)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
# class GatedFusionModule(nn.Module):
#     """
#     跨维度门控协同融合模块
    
#     输入:
#         A1: (B, C, H, W)  —— Scene consistency (已改为64通道)
#         A2: (B, C, H, W)  —— Imaging authenticity
#         A3: (B, C, H, W)  —— Signal naturalness
    
#     输出:
#         F_unified: (B, C, H, W)
#         weights: (w1, w2, w3), 每个 (B, 1, H, W)
#     """
#     def __init__(self, feature_channels=64, num_heads=4):
#         """
#         Args:
#             feature_channels: 输入特征通道数（三个分支统一）
#             num_heads: 多头注意力头数
#         """
#         super().__init__()
#         self.feature_channels = feature_channels

#         # 可选的投影层（用于特征变换，可以保留或去除）
#         self.proj1 = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)
#         self.proj2 = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)
#         self.proj3 = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)

#         # MHSA 门控网络
#         fuse_channels = 3 * feature_channels  # 192 (64*3)
#         self.mhsa = MultiHeadSelfAttention2D(
#             embed_dim=fuse_channels, 
#             num_heads=num_heads
#         )

#         # 权重预测头
#         self.weight_head = nn.Conv2d(fuse_channels, 3, kernel_size=1)

#         self._init_weights()

#     def _init_weights(self):
#         """初始化权重"""
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.xavier_normal_(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)

#     def forward(self, A1, A2, A3):
#         """
#         Args:
#             A1: [B, 64, H, W] 场景一致性特征
#             A2: [B, 64, H, W] 成像真实性特征
#             A3: [B, 64, H, W] 信号自然性特征
        
#         Returns:
#             F_unified: [B, 64, H, W] 统一伪影特征
#             weights: (w1, w2, w3), 每个 [B, 1, H, W]
#         """
#         B, C, H, W = A1.shape
        
#         # Step 1: 可选的特征投影（保持维度不变）
#         A1_hat = self.proj1(A1)  # [B, 64, H, W]
#         A2_hat = self.proj2(A2)  # [B, 64, H, W]
#         A3_hat = self.proj3(A3)  # [B, 64, H, W]

#         # Step 2: 拼接 → [B, 192, H, W]
#         X_fuse = torch.cat([A1_hat, A2_hat, A3_hat], dim=1)

#         # Step 3: 多头自注意力建模跨维度依赖
#         X_enhanced = self.mhsa(X_fuse)  # [B, 192, H, W]

#         # Step 4: 预测权重 → [B, 3, H, W]
#         weights = self.weight_head(X_enhanced)
#         weights = F.softmax(weights, dim=1)  # 归一化: w1 + w2 + w3 = 1

#         # Step 5: 分割权重
#         w1, w2, w3 = torch.chunk(weights, 3, dim=1)  # 每个 [B, 1, H, W]

#         # Step 6: 加权融合（公式25）
#         F_unified = w1 * A1_hat + w2 * A2_hat + w3 * A3_hat  # [B, 64, H, W]

#         return F_unified, (w1, w2, w3)

#     @property
#     def num_params(self):
#         """计算总参数量"""
#         return sum(p.numel() for p in self.parameters() if p.requires_grad)

