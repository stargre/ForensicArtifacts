import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from PIL import Image

# 导入场景一致性特征提取器
from .Semantic_Illusion import extract_semantic_feature
from .Geo_consistency import extract_geometric_feature
from .Lighting_shadow_anomaly import extract_lighting_feature
from .Layout import extract_layout_feature


class SceneFeatureExtractor(nn.Module):
    """
    场景一致性特征提取器
    整合四个维度的高层语义伪影检测
    
    输入: RGB 图像 [B, 3, H, W] (已归一化)
    输出: 拼接特征 [B, 4, H, W]
        - 通道0: 语义对齐异常 M_align
        - 通道1: 几何一致性 M_depth
        - 通道2: 光照阴影矛盾 M_light
        - 通道3: 布局语义异常 M_layout
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] torch.Tensor, 归一化后的 RGB 图像
        
        Returns:
            features: [B, 4, H, W] torch.Tensor
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
            
            # 3. 提取四个特征 (每个都是 (H, W) float32)
            M_align = extract_semantic_feature(img_np)
       
            M_depth = extract_geometric_feature(img_np)

            M_light = extract_lighting_feature(img_np)
            
            M_layout = extract_layout_feature(img_np)
            
            
            # 4. 堆叠为 (4, H, W)
            combined = np.stack([M_align, M_depth, M_light, M_layout], axis=0)
            
            # 5. 转为 Tensor
            combined_tensor = torch.from_numpy(combined).float()
            batch_features.append(combined_tensor)
        
        # 6. 堆叠批次
        features = torch.stack(batch_features, dim=0).to(device)  # [B, 4, H, W]
        
        return features


class PatchEmbed(nn.Module):
    """
    将特征图切分为 patches 并嵌入到高维空间
    """
    def __init__(self, img_size=512, patch_size=16, in_chans=4, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        Args:
            x: [B, 4, H, W]
        Returns:
            [B, num_patches, embed_dim]
        """
        x = self.proj(x)  # [B, embed_dim, H/16, W/16]
        x = rearrange(x, 'b d h w -> b (h w) d')
        return x


class Attention(nn.Module):
    """
    多头自注意力机制
    """
    def __init__(self, dim, num_heads=3, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    """
    Transformer Block: Attention + MLP
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                            attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        attn_out, attn_weights = self.attn(self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights


class ViTTinyEncoder(nn.Module):
    """
    ViT-Tiny 编码器 (4层, 192维, 3头注意力)
    """
    def __init__(self, img_size=512, patch_size=16, in_chans=4, 
                 embed_dim=192, depth=4, num_heads=3):
        super().__init__()
        
        self.patch_embed = PatchEmbed(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=in_chans, 
            embed_dim=embed_dim
        )
        
        num_patches = self.patch_embed.num_patches
        
        # CLS token
        self.cls_token_param = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Transformer Blocks
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads) 
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        
        # 初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token_param, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        
        # 添加 CLS token
        cls_tokens = self.cls_token_param.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, num_patches+1, embed_dim]
        
        # 添加位置编码
        x = x + self.pos_embed
        
        # Transformer blocks
        attn_weights_list = []
        for blk in self.blocks:
            x, attn = blk(x)
            attn_weights_list.append(attn)
        
        x = self.norm(x)
        
        return x, attn_weights_list


def attention_rollout(attn_weights_list):
    """
    Attention Rollout 算法
    将多层注意力权重聚合为单一的空间显著性图
    
    Args:
        attn_weights_list: List of [B, num_heads, N, N] 注意力权重
    
    Returns:
        cls_rollout: [B, num_patches] 归一化的空间显著性分数
    """
    B = attn_weights_list[0].shape[0]
    N_total = attn_weights_list[0].shape[-1]  # num_patches + 1 (包含CLS)
    N_patches = N_total - 1
    
    # 对每层的多头注意力取平均
    avg_attn = [attn.mean(dim=1) for attn in attn_weights_list]  # [B, N, N]
    
    # 初始化单位矩阵
    rollout = torch.eye(N_total, device=avg_attn[0].device).unsqueeze(0).repeat(B, 1, 1)
    
    # 逐层累积注意力
    for attn in avg_attn:
        rollout = torch.bmm(attn, rollout)
    
    # 提取 CLS token 对所有 patch 的注意力
    cls_rollout = rollout[:, 0, 1:]  # [B, num_patches]
    
    # 归一化到 [0, 1]
    cls_rollout = cls_rollout - cls_rollout.min(dim=1, keepdim=True)[0]
    cls_rollout = cls_rollout / (cls_rollout.max(dim=1, keepdim=True)[0] + 1e-8)
    
    return cls_rollout


class SceneEncoder(nn.Module):
    """
    场景一致性分支编码器（Scene Consistency Encoder）
    
    完整流程:
        输入RGB图像 → 特征提取器(4通道) → ViT-Tiny → Attention Rollout → 上采样
    
    输入: 
        x ∈ ℝ^(B × 3 × H × W) —— 原始 RGB 图像 (已归一化)
    
    输出:
        A₁ ∈ ℝ^(B × 1 × H × W) —— 场景一致性异常热力图
    
    设计依据:
        - 特征提取: 4维高层语义特征 (语义/几何/光照/布局)
        - ViT-Tiny: 4层, 192维, 3头注意力
        - Attention Rollout: 聚合跨层注意力
        - 双线性插值上采样至原始分辨率
    """
    def __init__(self, img_size=512, out_channels=64):
        super().__init__()
        
        # 特征提取器
        self.feature_extractor = SceneFeatureExtractor()
        
        # ViT-Tiny 编码器
        self.encoder = ViTTinyEncoder(
            img_size=img_size,
            patch_size=16,
            in_chans=4,  # 4个场景特征通道
            embed_dim=192,
            depth=4,
            num_heads=3
        )
        
        self.img_size = img_size
        self.patch_size = 16
        self.num_patches_per_side = img_size // 16  # 512/16 = 32
        self.out_channels = out_channels

        self.channel_expand = nn.Conv2d(1, out_channels, kernel_size=1)

    def forward(self, x):
        """
        完整前向传播
        
        Args:
            x: [B, 3, H, W] torch.Tensor —— 原始 RGB 图像（已归一化）
        
        Returns:
            anomaly_map: [B, 1, H, W] torch.Tensor —— 场景一致性异常热力图
        """
        B = x.shape[0]
        
        # Step 1: 特征提取 (RGB → 4通道场景特征)
        features = self.feature_extractor(x)  # [B, 4, H, W]
        
        # Step 2: ViT 编码
        _, attn_list = self.encoder(features)  # attn_list: List of [B, heads, N, N]
        
        # Step 3: Attention Rollout
        rollout_scores = attention_rollout(attn_list)  # [B, num_patches]
        
        # Step 4: 重塑为空间分辨率
        heatmap_lowres = rollout_scores.view(
            B, self.num_patches_per_side, self.num_patches_per_side
        )  # [B, 32, 32]
        
        # Step 5: 双线性插值上采样到原始分辨率
        anomaly_map = F.interpolate(
            heatmap_lowres.unsqueeze(1),  # [B, 1, 32, 32]
            size=(self.img_size, self.img_size),
            mode='bilinear',
            align_corners=False
        )  # [B, 1, 512, 512]

        # Step 6: 扩展到64通道
        anomaly_map = self.channel_expand(anomaly_map)  # [B, 64, 512, 512]
        
        return anomaly_map

    @property
    def num_params(self):
        """计算总参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# # ==================== 测试代码 ====================
# if __name__ == "__main__":
#     print("="*60)
#     print("测试场景一致性编码器")
#     print("="*60)
    
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"使用设备: {device}\n")
    
#     # 1. 创建模型
#     model = SceneEncoder(img_size=512, out_channels=1).to(device)
#     print(f"✓ 模型参数量: {model.num_params:,}")
    
#     # 2. 创建测试输入 (模拟已归一化的 RGB 图像)
#     batch_size = 2
#     dummy_input = torch.randn(batch_size, 3, 512, 512).to(device)
    
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
#     expected_shape = (batch_size, 1, 512, 512)
#     assert output.shape == expected_shape, \
#         f"输出形状错误! 期望 {expected_shape}, 得到 {output.shape}"
    
#     print("\n" + "="*60)
#     print("✓ 测试通过!")
#     print("="*60)