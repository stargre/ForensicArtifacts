import sys
import os
import torch
import numpy as np
from PIL import Image
import cv2

# === 添加本地 MiDaS 路径 ===
MIDAS_ROOT = "/mnt/data3/zhiyu/MiDaS"
sys.path.insert(0, MIDAS_ROOT)

# 使用官方 load_model（完全离线）
from midas.model_loader import load_model

# === 配置：BEiT-Large @ 384 ===
WEIGHT_PATH = "/mnt/data3/zhiyu/MiDaS/weights/dpt_beit_large_384.pt"
MODEL_TYPE = "dpt_beit_large_384"  # ✅ 必须严格匹配！

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_geometric_anomaly_map(image_pil):
    print("Loading MiDaS BEiT-Large 384 model from local path...")
    
    # --- 使用官方 loader 加载（自动匹配架构）---
    model, transform, net_w, net_h = load_model(
        device,
        WEIGHT_PATH,
        MODEL_TYPE,
        optimize=False,
        height=None,
        square=True  # BEiT 模型要求 square 输入
    )
    model.eval()
    print("✅ Model loaded successfully.")

    # --- 预处理图像 ---
    image_rgb = np.array(image_pil)  # (H, W, 3), uint8, RGB
    transformed = transform({"image": image_rgb})
    input_tensor = transformed["image"]  # 假设这里是 numpy.ndarray
    input_batch = torch.from_numpy(input_tensor).to(device).unsqueeze(0)

    # --- 预测深度图 D = M(I) ---
    with torch.no_grad():
        depth_pred = model(input_batch)  # (1, H_out, W_out)
        # 插值回原始图像尺寸 (H, W)
        depth_pred = torch.nn.functional.interpolate(
            depth_pred.unsqueeze(1),
            size=image_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze(0).squeeze(0)  # (H, W)
    
    depth = depth_pred.cpu().numpy().astype(np.float32)
    print(f"✅ Depth map shape: {depth.shape}")

    # --- Step: 计算表面法向量 N(x,y) = [-∂D/∂x, -∂D/∂y, 1]^T ---
    dz_dx, dz_dy = np.gradient(depth)
    normal = np.stack([-dz_dx, -dz_dy, np.ones_like(depth)], axis=-1)  # (H, W, 3)
    norm = np.linalg.norm(normal, axis=2, keepdims=True)
    normal = normal / (norm + 1e-8)  # 单位法向量

    # --- Step: 计算几何异常图 L_geo = ||∇N||_F^2 ---
    loss = np.zeros_like(depth, dtype=np.float32)
    for i in range(3):  # 对 Nx, Ny, Nz 分别求梯度
        Ni = normal[:, :, i].astype(np.float32)
        dNi_dx = cv2.Sobel(Ni, cv2.CV_32F, 1, 0, ksize=3)
        dNi_dy = cv2.Sobel(Ni, cv2.CV_32F, 0, 1, ksize=3)
        loss += dNi_dx ** 2 + dNi_dy ** 2

    print("✅ Geometric anomaly map computed.")
    return loss  # (H, W), float32

def extract_geometric_feature(image_512: np.ndarray) -> np.ndarray:
    """
    输入: (512, 512, 3) RGB uint8
    输出: (512, 512, 3) float32 —— 表面法向量 [Nx, Ny, Nz]
    """
    assert image_512.shape == (512, 512, 3) and image_512.dtype == np.uint8

    print("Loading MiDaS BEiT-Large 384 model for feature extraction...")
    model, transform, net_w, net_h = load_model(
        device,
        WEIGHT_PATH,
        MODEL_TYPE,
        optimize=False,
        height=None,
        square=True
    )
    model.eval()

    # 转回 PIL 用于 transform（因为 MiDaS 的 transform 需要原始 HxW）
    image_pil = Image.fromarray(image_512)
    image_rgb = np.array(image_pil)  # (512,512,3)

    transformed = transform({"image": image_rgb})
    input_tensor = transformed["image"]  # (3, H, W)
    input_batch = torch.from_numpy(input_tensor).to(device).unsqueeze(0)

    with torch.no_grad():
        depth_pred = model(input_batch)  # (1, H_out, W_out)
        depth_pred = torch.nn.functional.interpolate(
            depth_pred.unsqueeze(1),
            size=(512, 512),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0).squeeze(0)  # (512, 512)

    depth = depth_pred.cpu().numpy().astype(np.float32)

    # 计算法向量
    dz_dx, dz_dy = np.gradient(depth)
    normal = np.stack([-dz_dx, -dz_dy, np.ones_like(depth)], axis=-1)  # (512,512,3)
    norm = np.linalg.norm(normal, axis=2, keepdims=True)
    normal = normal / (norm + 1e-8)

    return normal.astype(np.float32)  # (512, 512, 3)
    
# # === 主程序 ===
# if __name__ == "__main__":
#     image_path = "/mnt/data3/public_datasets/OpenMMSec/3/e685278670eb41f1bb35f9f88510a1c1.jpg"
#     image = Image.open(image_path).convert("RGB")
    
#     geo_anomaly_map = compute_geometric_anomaly_map(image)

#     # === 保存热力图（可视化）===
#     vis = (geo_anomaly_map - geo_anomaly_map.min()) / (geo_anomaly_map.max() - geo_anomaly_map.min() + 1e-8)
#     vis = (vis * 255).astype(np.uint8)
#     heatmap = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
#     output_path = "geometric_consistency_heatmap_large384.png"
#     cv2.imwrite(output_path, heatmap)
#     print(f"🔥 Heatmap saved to: {output_path}")
#     print(f"Final anomaly map shape: {geo_anomaly_map.shape}")
