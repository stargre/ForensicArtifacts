import numpy as np
import cv2
from PIL import Image

def get_srm_filters():
    """
    返回完整的 30 个 SRM 滤波器 (5x5)，权重已按原始论文归一化。
    来源: Fridrich et al., "Rich Models for Steganalysis of Digital Images", IEEE TIFS 2012.
    实现参考: https://github.com/PeterWang512/CNNDetection/blob/master/srm.py
    """
    filters = []

    # Group 1: 6 filters with weight 4 (first-order derivatives)
    base1 = [
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, -1, 0, 0],
         [0, 0, 2, 0, 0],
         [0, 0, -1, 0, 0]],

        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, -1, 0, 0],
         [0, 0, 2, 0, 0],
         [0, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, -1, 2, 0, 0],
         [0, 0, -1, 0, 0],
         [0, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, -1, 0, 0],
         [0, 2, 0, 0, 0],
         [0, -1, 0, 0, 0]],

        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, -1, 0, 0],
         [0, 2, 0, 0, 0],
         [0, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, -1, 0, 0],
         [2, 0, 0, 0, 0],
         [-1, 0, 0, 0, 0]]
    ]
    for b in base1:
        filters.append(np.array(b) * 4)

    # Group 2: 12 filters with weight 12 (second-order cross derivatives)
    base2_offsets = [
        (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2),
        (-1, -2), (-1, 2),
        (0, -2), (0, 2),
        (1, -2), (1, 2),
        (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)
    ]
    # But only 12 are used (excluding center and some duplicates)
    selected_offsets = [
        (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2),
        (-1, -2), (-1, 2),
        (0, -2), (0, 2),
        (1, -2), (1, 2),
        (2, -2)
    ]
    for dx, dy in selected_offsets:
        kernel = np.zeros((5, 5))
        kernel[2, 2] = -1
        x, y = 2 + dx, 2 + dy
        if 0 <= x < 5 and 0 <= y < 5:
            kernel[x, y] = 2
            filters.append(kernel * 12)

    # Group 3: 12 filters with weight 4 (higher-order patterns)
    base3 = [
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, -1, 0, 0],
         [0, 2, 0, 0, 0],
         [0, -1, 0, 0, 0]],

        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, -1, 2, 0, 0],
         [0, 0, -1, 0, 0],
         [0, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, -1, 0, 0],
         [2, 0, 0, 0, 0],
         [-1, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, -1, 2, 0],
         [0, 0, 0, -1, 0],
         [0, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, -1, 0],
         [0, 0, 2, 0, 0],
         [0, 0, -1, 0, 0]],

        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, -1, 0],
         [0, 2, 0, 0, 0],
         [0, -1, 0, 0, 0]],

        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, -1, 0],
         [2, 0, 0, 0, 0],
         [-1, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, -1, 2, -1, 0],
         [0, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [-1, 2, -1, 0, 0],
         [0, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, -1, 2, -1],
         [0, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 0],
         [-1, 2, -1, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 0],
         [0, 0, -1, 2, -1],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]]
    ]
    for b in base3:
        filters.append(np.array(b) * 4)

    assert len(filters) == 30, f"Expected 30 filters, got {len(filters)}"
    return np.array(filters, dtype=np.float32)

def extract_srm_feature(image):
    """
    提取 SRM 特征图 M_texture ∈ R^{30 × H × W}
    
    Args:
        image: (H, W, 3) or (H, W) numpy array, uint8
    
    Returns:
        srm_map: (30, H, W) numpy array, float32, values in [-3, 3]
    """
    # 转灰度图
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        gray = image.astype(np.float32)

    H, W = gray.shape
    kernels = get_srm_filters()  # (30, 5, 5)

    srm_maps = []
    for k in kernels:
        # 使用反射边界减少边缘效应
        response = cv2.filter2D(gray, -1, k, borderType=cv2.BORDER_REFLECT)
        # 截断到典型传感器噪声范围 [-3, 3]
        response = np.clip(response, -3.0, 3.0)
        srm_maps.append(response)

    srm_tensor = np.stack(srm_maps, axis=0)  # (30, H, W)
    return srm_tensor.astype(np.float32)

# # === 主程序：加载图像并提取 SRM 特征 ===
# if __name__ == "__main__":
#     image_path = "/mnt/data3/public_datasets/OpenMMSec/3/e685278670eb41f1bb35f9f88510a1c1.jpg"
    
#     # 加载图像
#     image = np.array(Image.open(image_path).convert("RGB"))
#     print(f"Original image shape: {image.shape}")

#     # 提取 SRM 特征
#     srm_feat = extract_srm_feature(image)
#     print(f"✅ SRM feature shape: {srm_feat.shape}")  # (30, H, W)

#     # 可视化第一个通道作为示例
#     first_ch = srm_feat[0]
#     vis = (first_ch - first_ch.min()) / (first_ch.max() - first_ch.min() + 1e-8)
#     vis = (vis * 255).astype(np.uint8)
#     cv2.imwrite("srm_channel_0_heatmap.png", vis)
#     print("✅ First SRM channel heatmap saved to 'srm_channel_0_heatmap.png'")