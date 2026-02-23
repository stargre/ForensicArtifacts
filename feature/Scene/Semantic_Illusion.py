
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt


# ==============================================================================
# 精简 DINOv2 ViT-B/14 模型（适配 SAM2 风格权重：518x518, 无 register, 有 mask_token）
# 权重来源: https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth
# ==============================================================================

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, proj_drop=0.0, drop_path=0.0, init_values=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_drop=proj_drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=proj_drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=518, patch_size=14, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 37*37 = 1369
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class DinoVisionTransformerForHighRes(nn.Module):
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 1369
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))  # [1, 1370, 768]

        dpr = [x.item() for x in torch.linspace(0, 0.0, depth)]
        init_values = 1e-5
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=0.0,
                drop_path=dpr[i],
                init_values=init_values,
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self.init_weights()

    def init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, 1369, 768]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 1370, 768]
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x


# ==============================================================================
# Semantic Consistency Checker（适配高分辨率 DINOv2）
# ==============================================================================

class SemanticConsistencyChecker:
    def __init__(self, device=None, checkpoint_path="/mnt/data3/zhiyu/checkpoint/dinov2/dinov2_vitb14_pretrain.pth"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"[INFO] Loading DINOv2 weights from: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        
        # 🔥 关键：移除 mask_token（当前权重包含它，但模型不需要）
        if "mask_token" in state_dict:
            del state_dict["mask_token"]
            print("[INFO] Removed 'mask_token' from state_dict.")

        # 创建适配 518x518 的模型（无 register tokens）
        self.model = DinoVisionTransformerForHighRes(
            img_size=518,
            patch_size=14,
            embed_dim=768,
            depth=12,
            num_heads=12,
            qkv_bias=True,
        )
        
        # 加载权重（现在 keys 应该完全匹配）
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device).eval()
        print("[INFO] Model loaded successfully (518x518, 37x37 patches).")

    @torch.no_grad()
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        # 🔥 必须 resize 到 518x518
        transform = transforms.Compose([
            transforms.Resize((518, 518), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        pixel_values = transform(image).unsqueeze(0).to(self.device)
        return pixel_values

    @torch.no_grad()
    def extract_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.model(pixel_values)  # [B, 1370, 768]
        return features.squeeze(0)  # [1370, 768]

    def compute_feature_similarity(self, features: torch.Tensor) -> torch.Tensor:
        # 只使用 patch tokens: [1:1370] → [1369, 768]
        patch_features = features[1:]  # [1369, 768]
        norm_features = patch_features / (patch_features.norm(dim=-1, keepdim=True) + 1e-8)
        similarity_matrix = torch.mm(norm_features, norm_features.t())  # [1369, 1369]
        return similarity_matrix

    def generate_semantic_map(self, similarity_matrix: torch.Tensor) -> np.ndarray:
        avg_similarity = similarity_matrix.mean(dim=0)  # [1369]
        avg_similarity = avg_similarity.reshape(37, 37)  # 518//14 = 37

        # 上采样到 512x512
        heatmap = F.interpolate(
            avg_similarity.unsqueeze(0).unsqueeze(0),
            size=(512, 512),
            mode='bilinear',
            align_corners=False
        ).squeeze().cpu().numpy()

        # 归一化到 [0, 1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        anomaly_map = 1.0 - heatmap  # 高异常 = 低一致性
        
        return anomaly_map
    
    def extract_semantic_features(self, image_512: np.ndarray) -> np.ndarray:
        """
        输入: (512, 512, 3) RGB uint8
        输出: (512, 512, 768) float32 —— DINOv2 patch features
        """
        assert image_512.shape == (512, 512, 3) and image_512.dtype == np.uint8

        # 转为 PIL 并 resize to 518x518 internally
        image_pil = Image.fromarray(image_512)
        pixel_values = self.preprocess_image(image_pil)  # (1, 3, 518, 518)

        features = self.model(pixel_values)  # (1, 1370, 768)
        patch_tokens = features[0, 1:, :]    # (1369, 768)

        # Reshape to (37, 37, 768)
        patch_tokens = patch_tokens.view(37, 37, 768)

        # Upsample to (512, 512, 768)
        patch_tokens = patch_tokens.permute(2, 0, 1).unsqueeze(0)  # (1, 768, 37, 37)
        patch_tokens = F.interpolate(
            patch_tokens,
            size=(512, 512),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).permute(1, 2, 0)  # (512, 512, 768)

        return patch_tokens.cpu().numpy().astype(np.float32)
    

    @torch.no_grad()
    def check_semantic_consistency(self, image: Image.Image) -> np.ndarray:
        # 不再强制要求 512x512 输入（自动 resize 到 518x518）
        pixel_values = self.preprocess_image(image)
        features = self.extract_features(pixel_values)
        similarity_matrix = self.compute_feature_similarity(features)
        semantic_map = self.generate_semantic_map(similarity_matrix)
        return semantic_map


# ==============================================================================
# 主程序
# ==============================================================================

# if __name__ == "__main__":
#     checker = SemanticConsistencyChecker(device="cuda")

#     image = Image.open("/mnt/data3/public_datasets/OpenMMSec/3/e685278670eb41f1bb35f9f88510a1c1.jpg").convert("RGB")
    
#     semantic_map = checker.check_semantic_consistency(image)

#     print(f"Semantic Map shape: {semantic_map.shape}, range: [{semantic_map.min():.3f}, {semantic_map.max():.3f}]")
    
#     plt.imshow(semantic_map, cmap='jet')
#     plt.colorbar()
#     plt.title('Semantic Consistency Heatmap (DINOv2 ViT-B/14 - 518x518)')
#     plt.savefig("semantic_map.png", dpi=150, bbox_inches='tight')
#     print("✅ Heatmap saved to semantic_map.png")


if __name__ == "__main__":
    checker = SemanticConsistencyChecker(device="cuda")

    image_path = "/mnt/data3/public_datasets/OpenMMSec/3/e685278670eb41f1bb35f9f88510a1c1.jpg"
    original_image = Image.open(image_path).convert("RGB")
    
    semantic_map = checker.check_semantic_consistency(original_image)

    print(f"Semantic Map shape: {semantic_map.shape}, range: [{semantic_map.min():.3f}, {semantic_map.max():.3f}]")
    
    # --- 保存单独的热力图（保持原有功能） ---
    plt.figure(figsize=(6, 6))
    plt.imshow(semantic_map, cmap='jet')
    plt.colorbar()
    plt.title('Semantic Consistency Heatmap (DINOv2 ViT-B/14 - 518x518)')
    plt.savefig("semantic_map.png", dpi=150, bbox_inches='tight')
    plt.close()

    # --- 新增：原图 + 热力图对比 ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 原图（resize 到 512x512 以便对齐）
    original_resized = original_image.resize((512, 512), Image.BICUBIC)
    axes[0].imshow(original_resized)
    axes[0].set_title("Original Image (512×512)")
    axes[0].axis('off')

    # 热力图
    im = axes[1].imshow(semantic_map, cmap='jet')
    axes[1].set_title("Semantic Consistency Heatmap")
    axes[1].axis('off')
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("semantic_map_comparison.png", dpi=200, bbox_inches='tight')
    plt.close()

    print("✅ Heatmap saved to semantic_map.png")
    print("✅ Comparison (original + heatmap) saved to semantic_map_comparison.png")
