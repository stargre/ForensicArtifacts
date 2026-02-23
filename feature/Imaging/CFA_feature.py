import os
import numpy as np
import cv2
from PIL import Image

def extract_cfa_feature(image):
    """
    提取 CFA 插值痕迹缺失特征（严格实现公式 R_cfa = I - D(D^{-1}(I))）
    
    原理：
      - D^{-1}(I): 将 RGB 图像按 RGGB 模式降采样为 Bayer 单通道
      - D(·): 使用 OpenCV 双线性去马赛克重建 RGB
      - R_cfa: 计算重建误差（AIGC 误差大，真实图像误差小）
    
    优化：
      - 仅使用绿色通道残差（G 占 50%，重建最稳定）
      - 输出 (H, W) float32，值越大表示越可能为 AIGC
    
    Args:
        image: PIL.Image 或 (H, W, 3) uint8 numpy array (RGB)
    
    Returns:
        cfa_map: (H, W) float32, ≥0
    """
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = image.astype(np.uint8)

    H, W = img.shape[:2]
    
    # Step 1: D^{-1}(I) —— 模拟 Bayer 采样（RGGB）
    bayer = np.zeros((H, W), dtype=np.uint8)
    bayer[0::2, 0::2] = img[0::2, 0::2, 0]   # R
    bayer[0::2, 1::2] = img[0::2, 1::2, 1]   # G
    bayer[1::2, 0::2] = img[1::2, 0::2, 1]   # G
    bayer[1::2, 1::2] = img[1::2, 1::2, 2]   # B

    # Step 2: D(bayer) —— 双线性去马赛克重建
    rgb_recon = cv2.cvtColor(bayer, cv2.COLOR_BayerRG2RGB)

    # Step 3: R_cfa = |I - D(D^{-1}(I))| —— 仅用绿色通道（更稳定）
    residual = np.abs(img[:, :, 1].astype(np.float32) - rgb_recon[:, :, 1].astype(np.float32))
    
    return residual.astype(np.float32)


def save_cfa_outputs(cfa_map, heatmap_path, raw_npy_path, raw_png_path):
    """
    保存 CFA 特征图：热力图 + 原始数据 + 灰度图
    
    Args:
        cfa_map: (H, W) float32
        heatmap_path: str, Jet 热力图路径
        raw_npy_path: str, .npy 原始特征
        raw_png_path: str, 灰度残差图（0~255）
    """
    # 保存原始特征（用于后续 32 通道拼接）
    np.save(raw_npy_path, cfa_map)

    # 保存灰度图（便于快速查看）
    gray_vis = np.clip(cfa_map, 0, 255).astype(np.uint8)
    cv2.imwrite(raw_png_path, gray_vis)

    # 生成抗分层热力图（1%~99% 截断）
    p1 = np.percentile(cfa_map, 1)
    p99 = np.percentile(cfa_map, 99)
    cfa_norm = np.clip(cfa_map, p1, p99)
    cfa_norm = (cfa_norm - p1) / (p99 - p1 + 1e-8)
    heatmap = cv2.applyColorMap((cfa_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(heatmap_path, heatmap)


# # ========== Main Function ==========
# if __name__ == "__main__":
#     # 🔧 参数配置（仅配置，无计算）
#     image_path = "/mnt/data3/public_datasets/OpenMMSec/3/e685278670eb41f1bb35f9f88510a1c1.jpg"
#     heatmap_path = "cfa_residual_heatmap.png"
#     raw_npy_path = "cfa_feature.npy"      # → 将拼接到 [PRNU, SRM, CFA]
#     raw_png_path = "cfa_residual_raw.png"

#     # ✅ 输入验证
#     assert os.path.exists(image_path), f"❌ Image not found: {image_path}"

#     # 📷 加载图像
#     image = Image.open(image_path).convert("RGB")
#     print(f"✅ Loaded image: {image_path}, size: {image.size}")

#     # 🔍 提取 CFA 特征（单通道 float32）
#     cfa_map = extract_cfa_feature(image)

#     # 💾 保存输出
#     save_cfa_outputs(cfa_map, heatmap_path, raw_npy_path, raw_png_path)

#     # 📊 打印统计（供调试参考，不影响 pipeline）
#     score = cfa_map.mean()
#     print(f"📊 CFA Feature — Mean: {score:.2f}")
#     print(f"🔥 Heatmap saved: {heatmap_path}")
#     print(f"💾 Raw feature (for fusion): {raw_npy_path}")
#     print(f"🖼️  Gray residual: {raw_png_path}")