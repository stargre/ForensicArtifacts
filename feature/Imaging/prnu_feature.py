import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image
import cv2

def extract_prnu_feature(image):
    """
    提取 PRNU 特征图 M_prnu
    输入: image (H, W, 3) numpy array, uint8 or float32 in [0,1]
    输出: prnu_map (H, W) float32 in [0,1], 单通道
    """
    # Step 1: 转换为 float32 并归一化
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    else:
        assert image.max() <= 1.0, "Image should be normalized"

    H, W, C = image.shape

    # Step 2: 高通滤波 HPR = I - Gσ * I, 只处理绿色通道
    sigma = 3.0
    green_channel = image[:, :, 1]  # 使用绿色通道
    blurred_green = gaussian_filter(green_channel, sigma=sigma)
    hpr_green = green_channel - blurred_green  # H×W

    # Step 3: 去均值并归一化
    hpr_green -= np.mean(hpr_green)
    hpr_green /= np.std(hpr_green)

    # Step 4: 计算自相关（FFT 快速）
    fft_hpr = np.fft.fft2(hpr_green)
    autocorr = np.real(np.fft.ifft2(fft_hpr * np.conj(fft_hpr)))
    autocorr = autocorr / np.max(autocorr)

    # Step 5: 取中心区域（避免边缘效应）
    center_size = min(H, W) // 8  # 缩小中心区域大小
    center_y, center_x = H//2, W//2
    center_region = autocorr[center_y-center_size:center_y+center_size,
                             center_x-center_size:center_x+center_size]

    # Step 6: 插值回原图大小
    prnu_map = cv2.resize(center_region, (W, H), interpolation=cv2.INTER_CUBIC)

    # Step 7: 归一化到 [0,1]
    prnu_map = (prnu_map - prnu_map.min()) / (prnu_map.max() - prnu_map.min() + 1e-8)

    return prnu_map.astype(np.float32)

# # === 主程序 ===
# if __name__ == "__main__":
#     # 加载图像
#     image_path = "/mnt/data3/public_datasets/OpenMMSec/3/e685278670eb41f1bb35f9f88510a1c1.jpg"
#     image = Image.open(image_path).convert("RGB")
#     image_np = np.array(image)

#     # 提取 PRNU 特征图
#     prnu_map = extract_prnu_feature(image_np)

#     # === 保存热力图（可视化）===
#     vis = (prnu_map - prnu_map.min()) / (prnu_map.max() - prnu_map.min() + 1e-8)
#     vis = (vis * 255).astype(np.uint8)
#     heatmap = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    
#     # 将热图叠加到原图上
#     overlay = image_np.copy()
#     alpha = 0.6  # 设置透明度
#     heatmap_on_image = cv2.addWeighted(heatmap, alpha, overlay, 1 - alpha, 0)

#     output_path_heatmap = "prnu_heatmap.png"
#     output_path_overlay = "prnu_heatmap_on_image.png"
#     cv2.imwrite(output_path_heatmap, heatmap)
#     cv2.imwrite(output_path_overlay, heatmap_on_image)
#     print(f"🔥 Heatmap saved to: {output_path_heatmap}")
#     print(f"🔥 Heatmap on image saved to: {output_path_overlay}")
#     print(f"Final PRNU map shape: {prnu_map.shape}")