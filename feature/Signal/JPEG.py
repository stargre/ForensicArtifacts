import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from scipy.ndimage import zoom


def normalize_to_01(x):
    """
    Normalize the input array to [0, 1] range.
    """
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

def rgb2ycbcr(rgb):
    """Convert RGB to YCbCr (only Y channel used for JPEG analysis)"""
    if rgb.shape[0] == 3:
        r, g, b = rgb[0], rgb[1], rgb[2]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        return y
    else:
        return rgb.squeeze(0)  # already grayscale

def block_dct(image_y, block_size=8):
    """Apply DCT to non-overlapping 8x8 blocks"""
    H, W = image_y.shape
    H_pad = (block_size - H % block_size) % block_size
    W_pad = (block_size - W % block_size) % block_size
    img_pad = np.pad(image_y, ((0, H_pad), (0, W_pad)), mode='edge')
    
    H_new, W_new = img_pad.shape
    blocks_h = H_new // block_size
    blocks_w = W_new // block_size
    
    dct_blocks = np.zeros((blocks_h, blocks_w, block_size, block_size))
    for i in range(blocks_h):
        for j in range(blocks_w):
            block = img_pad[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            dct_blocks[i, j] = dct(dct(block.T, norm='ortho').T, norm='ortho')
    return dct_blocks, H, W

def benford_estimate_qf(dct_block):
    """
    Estimate local QF using Benford's law on AC coefficients.
    Returns a scalar representing quantization strength (lower = higher QF).
    """
    # Flatten AC coefficients (skip DC at [0,0])
    ac_coeffs = dct_block[1:, :].flatten()
    ac_coeffs = np.abs(ac_coeffs)
    ac_coeffs = ac_coeffs[ac_coeffs > 1e-6]  # avoid log(0)
    
    if len(ac_coeffs) == 0:
        return 0.0
    
    # Get first significant digit
    first_digits = np.floor(ac_coeffs / (10 ** np.floor(np.log10(ac_coeffs)))).astype(int)
    first_digits = first_digits[(first_digits >= 1) & (first_digits <= 9)]
    
    if len(first_digits) == 0:
        return 0.0
    
    # Benford expected distribution
    benford = np.log10(1 + 1 / np.arange(1, 10))
    observed = np.bincount(first_digits, minlength=10)[1:10]
    observed = observed / observed.sum() if observed.sum() > 0 else np.zeros(9)
    
    # Use KL divergence or L2 distance as inconsistency measure
    # Here we use negative correlation: higher deviation → lower "effective QF"
    deviation = np.sum((observed - benford) ** 2)
    return 1.0 / (1.0 + deviation)  # normalize to [0,1]

def extract_jpeg_feature(image_np):
    """
    Compute double JPEG inconsistency map
    
    输入: (H, W, 3) uint8 RGB numpy array
    输出: (H, W) float32 numpy array
    """
    
    if isinstance(image_np, np.ndarray):
        # numpy → tensor
        image_tensor = transforms.ToTensor()(image_np)  # (3, H, W)
    else:
        image_tensor = image_np
    
    # Step 1: Convert to Y channel
    y_channel = rgb2ycbcr(image_tensor.numpy())  # (H, W)
    
    # Step 2: Block-wise DCT
    dct_blocks, H_orig, W_orig = block_dct(y_channel, block_size=8)
    blocks_h, blocks_w = dct_blocks.shape[:2]
    
    # Step 3: Estimate local QF for each block
    qf_map = np.zeros((blocks_h, blocks_w))
    for i in range(blocks_h):
        for j in range(blocks_w):
            qf_map[i, j] = benford_estimate_qf(dct_blocks[i, j])
    
    # Step 4: Compute D_jpeg(k) = |Q_est(center) - mean(Q_est(neighbors))|
    d_jpeg = np.zeros_like(qf_map)
    for i in range(blocks_h):
        for j in range(blocks_w):
            center = qf_map[i, j]
            neighbors = []
            for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < blocks_h and 0 <= nj < blocks_w:
                    neighbors.append(qf_map[ni, nj])
            if neighbors:
                neighbor_mean = np.mean(neighbors)
                d_jpeg[i, j] = abs(center - neighbor_mean)
            else:
                d_jpeg[i, j] = 0.0
    
    # Step 5: Upsample to full resolution via interpolation
    scale_h = H_orig / blocks_h
    scale_w = W_orig / blocks_w
    m_jpeg = zoom(d_jpeg, (scale_h, scale_w), order=1)  # bilinear interpolation
    m_jpeg = m_jpeg[:H_orig, :W_orig]  # crop to original size
    
    # 归一化
    m_jpeg = (m_jpeg - m_jpeg.min()) / (m_jpeg.max() - m_jpeg.min() + 1e-8)
    
    return m_jpeg.astype(np.float32)  # 返回 (H, W)

# # ======================
# # Example Usage
# # ======================
# if __name__ == "__main__":
#     image_path = "/mnt/data3/public_datasets/OpenMMSec/3/e685278670eb41f1bb35f9f88510a1c1.jpg"
#     original_image = Image.open(image_path).convert("RGB")
#     image_tensor = transforms.ToTensor()(original_image)  # (3, H, W)

#     # Compute double JPEG inconsistency map
#     m_jpeg = compute_double_jpeg_map(image_tensor)

#     # Visualization
#     plt.figure(figsize=(12, 4))
#     plt.subplot(1, 3, 1)
#     plt.title('Original Image')
#     plt.imshow(original_image)
#     plt.axis('off')

#     plt.subplot(1, 3, 2)
#     plt.title('Y Channel')
#     y_img = rgb2ycbcr(image_tensor.numpy())
#     plt.imshow(y_img, cmap='gray')
#     plt.axis('off')

#     plt.subplot(1, 3, 3)
#     plt.title('Double JPEG Inconsistency Map')
#     plt.imshow(m_jpeg, cmap='hot')
#     plt.colorbar()
#     plt.axis('off')

#     plt.tight_layout()
#     plt.savefig("double_jpeg_detection.png", dpi=200, bbox_inches='tight')
#     print("✅ Double JPEG map saved to 'double_jpeg_detection.png'")