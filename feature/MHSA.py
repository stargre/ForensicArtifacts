import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention2D(nn.Module):
    """
    标准 2D 多头自注意力模块（无位置编码，因输入已是空间特征图）
    输入: (B, C_in, H, W)
    输出: (B, C_out, H, W)，通常 C_out == C_in
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

        # Split into heads: (B, N, num_heads, head_dim) → (B, num_heads, N, head_dim)
        Q = Q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, num_heads, N, N)
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        out = attn @ V  # (B, num_heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, -1, self.embed_dim)  # (B, N, C)

        out = self.out_proj(out)  # (B, N, C)
        out = out.transpose(1, 2).view(B, self.embed_dim, H, W)  # (B, C, H, W)
        return out


class CrossDimensionalGatedFusion(nn.Module):
    """
    跨维度门控协同融合模块（严格遵循论文 2.2 节）
    
    输入:
        A1: (B, 1, H, W)   —— Scene consistency
        A2: (B, C, H, W)   —— Imaging authenticity
        A3: (B, C, H, W)   —— Signal naturalness
    
    输出:
        F_unified: (B, C, H, W)
    """
    def __init__(self, common_channels=64, num_heads=4):
        super().__init__()
        self.common_channels = common_channels

        # Step 1: 1×1 Conv for channel alignment (公式后处理)
        self.proj1 = nn.Conv2d(1, common_channels, kernel_size=1)
        self.proj2 = nn.Conv2d(common_channels, common_channels, kernel_size=1)
        self.proj3 = nn.Conv2d(common_channels, common_channels, kernel_size=1)

        # Step 2: MHSA-based gating network (图25)
        fuse_channels = 3 * common_channels
        self.mhsa = MultiHeadSelfAttention2D(embed_dim=fuse_channels, num_heads=num_heads)

        # Step 3: Weight prediction head → 3 channels → softmax over dim=1
        self.weight_head = nn.Conv2d(fuse_channels, 3, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, A1, A2, A3):
        """
        A1: (B, 1, H, W)
        A2: (B, C, H, W)
        A3: (B, C, H, W)
        """
        B, _, H, W = A1.shape
        C = self.common_channels

        # Channel alignment (Â1, Â2, Â3)
        A1_hat = self.proj1(A1)      # (B, C, H, W)
        A2_hat = self.proj2(A2)      # (B, C, H, W)
        A3_hat = self.proj3(A3)      # (B, C, H, W)

        # Concatenate: X_fuse = [Â1; Â2; Â3] → (B, 3C, H, W)
        X_fuse = torch.cat([A1_hat, A2_hat, A3_hat], dim=1)  # (B, 3C, H, W)

        # MHSA to model inter-dimensional & spatial dependencies
        X_enhanced = self.mhsa(X_fuse)  # (B, 3C, H, W)

        # Predict gating weights: (B, 3, H, W)
        weights = self.weight_head(X_enhanced)  # (B, 3, H, W)
        weights = F.softmax(weights, dim=1)     # Normalize: w1 + w2 + w3 = 1 per pixel

        # Split weights
        w1, w2, w3 = torch.chunk(weights, 3, dim=1)  # each: (B, 1, H, W)

        # Weighted fusion (公式 25)
        F_unified = w1 * A1_hat + w2 * A2_hat + w3 * A3_hat  # (B, C, H, W)

        return F_unified, (w1, w2, w3)  # 返回权重图用于可视化/分析