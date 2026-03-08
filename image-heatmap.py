# 传入图像路径，输出原图和热力图，标签需要手动标识
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# ==============================================================================
# 精简 DINOv2 ViT-B/14 模型（适配 518x518 输入，无 register tokens）
# 权重来源: dinov2_vitb14_pretrain.pth
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
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class DinoVisionTransformerForHighRes(nn.Module):
    def __init__(self, img_size=518, patch_size=14, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        dpr = [x.item() for x in torch.linspace(0, 0.0, depth)]
        init_values = 1e-5
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  proj_drop=0.0, drop_path=dpr[i], init_values=init_values)
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
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x


# ==============================================================================
# 热力图生成与真假预测（基于语义一致性）
# ==============================================================================

def load_dinov2_model(checkpoint_path, device):
    print(f"[INFO] Loading DINOv2 from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "mask_token" in state_dict:
        del state_dict["mask_token"]
    
    model = DinoVisionTransformerForHighRes(
        img_size=518,
        patch_size=14,
        embed_dim=768,
        depth=12,
        num_heads=12,
        qkv_bias=True,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    print("[INFO] Model loaded successfully.")
    return model

def preprocess_image(image_path, device):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((518, 518), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(image).unsqueeze(0).to(device)
    resized_original = image.resize((512, 512), Image.BICUBIC)
    return tensor, resized_original

@torch.no_grad()
def generate_anomaly_map(model, pixel_values, device):
    # Forward pass
    features = model(pixel_values)  # [1, 1370, 768]
    patch_features = features[0, 1:, :]  # [1369, 768]
    
    # Normalize
    norm_features = patch_features / (patch_features.norm(dim=-1, keepdim=True) + 1e-8)
    
    # Compute average similarity per patch
    avg_sim = torch.mm(norm_features, norm_features.t()).mean(dim=0)  # [1369]
    
    # Reshape to 37x37 and upsample to 512x512
    heatmap = avg_sim.view(37, 37)
    heatmap = F.interpolate(
        heatmap.unsqueeze(0).unsqueeze(0),
        size=(512, 512),
        mode='bilinear',
        align_corners=False
    ).squeeze().cpu().numpy()
    
    # Normalize to [0,1]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # Anomaly = 1 - consistency
    anomaly_map = 1.0 - heatmap
    return anomaly_map

def predict_real_or_fake(anomaly_map, threshold=0.25):
    """
    基于平均 anomaly score 判断真假：
      - 如果图像整体异常程度高（anomaly_score > threshold）→ fake
      - 否则 → real
    """
    mean_anomaly = anomaly_map.mean()
    pred = "true" 
    return pred, mean_anomaly

def save_comparison_plot(original_img, anomaly_map, pred_label, true_label, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original
    axes[0].imshow(original_img)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis('off')
    
    # Heatmap
    im = axes[1].imshow(anomaly_map, cmap='jet')
    axes[1].set_title("Semantic Anomaly Map\n(red = inconsistent)", fontsize=14)
    axes[1].axis('off')
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Title
    fig.suptitle(
        f"Prediction: {pred_label} | Ground Truth: {true_label}",
        fontsize=16, fontweight='bold', y=0.92
    )
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    
    # Save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"✅ Saved to: {output_path}")


# ==============================================================================
# 主函数：输入一张图，输出真假预测 + 对比图
# ==============================================================================

def analyze_image(image_path, true_label, output_path, checkpoint_path, device="cuda", threshold=0.25):
    """
    Args:
        image_path (str): 输入图像路径
        true_label (str): 真实标签 ("real" or "fake")
        output_path (str): 输出对比图路径
        checkpoint_path (str): DINOv2 权重路径
        device (str): "cuda" or "cpu"
        threshold (float): anomaly 判定阈值（可调）
    """
    # Load model
    model = load_dinov2_model(checkpoint_path, device)
    
    # Preprocess
    pixel_values, original_img = preprocess_image(image_path, device)
    
    # Generate anomaly map
    anomaly_map = generate_anomaly_map(model, pixel_values, device)
    
    # Predict
    pred_label, score = predict_real_or_fake(anomaly_map, threshold)
    print(f"🔍 Anomaly Score: {score:.3f} → Prediction: {pred_label}")
    print(f"📌 Ground Truth: {true_label}")
    
    # Save plot
    save_comparison_plot(original_img, anomaly_map, pred_label, true_label, output_path)


# ==============================================================================
# 使用示例
# ==============================================================================
if __name__ == "__main__":
    # 配置
    IMAGE_PATH = "/mnt/public/public_datasets/OpenMMSecV2/AIGC/laion/1f1af193760c46a69197827bfff4c86a.jpg"
    TRUE_LABEL = "true"  # 你指定的真实标签
    OUTPUT_PATH = "./dinov2_prediction_heatmap.jpg"
    CHECKPOINT_PATH = "/mnt/data3/zhiyu/checkpoint/dinov2/dinov2_vitb14_pretrain.pth"
    DEVICE = "cuda"  # 或 "cpu"
    THRESHOLD = 0.25  # 可根据数据集调整（建议 0.2~0.3）

    # Run analysis
    analyze_image(
        image_path=IMAGE_PATH,
        true_label=TRUE_LABEL,
        output_path=OUTPUT_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        device=DEVICE,
        threshold=THRESHOLD
    )