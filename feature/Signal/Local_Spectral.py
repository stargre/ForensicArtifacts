import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# ======================
# Helper: Crop to divisible by block_size (default=8)
# ======================

def normalize_to_01(x):
    """
    Normalize the input array to [0, 1] range.
    """
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def make_divisible(img_tensor, block_size=8):
    C, H, W = img_tensor.shape
    new_H = (H // block_size) * block_size
    new_W = (W // block_size) * block_size
    top = (H - new_H) // 2
    left = (W - new_W) // 2
    cropped = img_tensor[:, top:top + new_H, left:left + new_W]
    return cropped, (top, left, new_H, new_W)

# ======================
# CORRECT & LOSSLESS Spectral Feature Extraction
# Returns:
#   spec_lr: (H//8, W//8) —— RAW BLOCK-LEVEL FEATURE (USE THIS FOR DOWNSTREAM)
#   spec_hr: (H, W)       —— BILINEAR UPSAMPLED (ONLY FOR VISUALIZATION)
# ======================
def extract_spectral_feature(image, block_size=8, lam=0.5, eps=1e-6):
    """
    Compute spectral anomaly feature as per Eq.(17), WITHOUT any post-processing.
    Input:
        image: (C, H, W) tensor in [0, 255] or [0, 1]
    Output:
        spec_lr: (H//bs, W//bs) —— raw block-level feature (use for analysis)
        spec_hr: (H, W)         —— upsampled for visualization only
    """
    device = image.device
    dtype = image.dtype

    # Normalize to [0,1] if needed (preserve linearity)
    if image.max() > 1.0:
        image = image / 255.0

    # Convert to grayscale: (C, H, W) -> (1, H, W)
    if image.shape[0] == 3:
        image = torch.mean(image, dim=0, keepdim=True)  # (1, H, W)
    elif image.shape[0] != 1:
        raise ValueError("Input must be RGB or grayscale")

    C, H, W = image.shape
    assert H % block_size == 0 and W % block_size == 0, f"Image {H}x{W} not divisible by {block_size}"

    # Precompute DCT matrix (standard DCT-II)
    N = block_size
    n = torch.arange(N, dtype=dtype, device=device)
    k = n.view(-1, 1)
    DCT_mat = torch.cos(np.pi / N * (n + 0.5) * k)
    DCT_mat[0, :] *= 1 / np.sqrt(2)
    DCT_mat *= np.sqrt(2.0 / N)
    DCT_mat = DCT_mat.to(device)

    # Unfold into non-overlapping blocks: (1, H, W) -> (N_blocks, bs, bs)
    image_unsq = image.unsqueeze(0)  # (1, 1, H, W)
    blocks = F.unfold(image_unsq, kernel_size=block_size, stride=block_size)  # (1, bs*bs, N)
    N_blocks = blocks.shape[-1]
    blocks = blocks.view(1, block_size, block_size, N_blocks)
    blocks = blocks.permute(0, 3, 1, 2).squeeze(0)  # (N, bs, bs)

    # Apply DCT: C = D @ X @ D^T
    dct_coeffs = DCT_mat @ blocks @ DCT_mat.t()  # (N, bs, bs)

    # Extract DC and AC
    dc = dct_coeffs[:, 0, 0]  # (N,)
    ac = dct_coeffs[:, 1:, 1:]  # (N, 7, 7)

    # Compute AC entropy (Shannon)
    ac_flat = ac.reshape(N_blocks, -1)  # ✅ Use reshape to avoid contiguous error
    ac_abs = torch.abs(ac_flat)
    ac_energy = ac_abs + eps
    p_uv = ac_energy / ac_energy.sum(dim=1, keepdim=True)  # (N, 49)
    entropy = -(p_uv * torch.log(p_uv + eps)).sum(dim=1)  # (N,)

    # Reshape DC to spatial grid
    h_blocks = H // block_size
    w_blocks = W // block_size
    dc_grid = dc.view(h_blocks, w_blocks)  # (h_b, w_b)

    # Compute |∇DC| via central differences
    dc_pad = F.pad(dc_grid.unsqueeze(0).unsqueeze(0), (1,1,1,1), mode='replicate').squeeze()
    dx = dc_pad[2:, 1:-1] - dc_pad[:-2, 1:-1]
    dy = dc_pad[1:-1, 2:] - dc_pad[1:-1, :-2]
    dc_grad_mag = torch.sqrt(dx**2 + dy**2 + eps)  # (h_b, w_b)

    # Combine: S_spec = entropy + λ * |∇DC|
    spec_lr = entropy.view(h_blocks, w_blocks) + lam * dc_grad_mag  # (H//8, W//8)

    # Upsample ONLY for visualization (do NOT use this for features!)
    spec_hr = F.interpolate(
        spec_lr.unsqueeze(0).unsqueeze(0),
        size=(H, W),
        mode='bilinear',
        align_corners=False
    ).squeeze(0).squeeze(0)  # (H, W)

    spec_lr_normalized = normalize_to_01(spec_lr.cpu().numpy())
    spec_hr_normalized = normalize_to_01(spec_hr.cpu().numpy())

    return spec_lr_normalized, spec_hr_normalized  # RETURN BOTH

# # ======================
# # Main: Feature Extraction Pipeline
# # ======================
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     image_path = "/mnt/data3/public_datasets/OpenMMSec/3/e685278670eb41f1bb35f9f88510a1c1.jpg"
#     original_image = Image.open(image_path).convert("RGB")
    
#     # To tensor
#     img_tensor = transforms.ToTensor()(original_image)  # (3, H, W)
#     print(f"Original size: {img_tensor.shape[1]}x{img_tensor.shape[2]}")

#     # Crop to 8-divisible
#     cropped_tensor, (top, left, H_crop, W_crop) = make_divisible(img_tensor, block_size=8)
#     print(f"Cropped to: {H_crop}x{W_crop}")
#     cropped_tensor = cropped_tensor.to(device)

#     # ✅ EXTRACT FEATURES (NO POST-PROCESSING)
#     spec_lr, spec_hr = compute_local_spectral_anomaly(cropped_tensor, block_size=8, lam=0.5)
    
#     # Move to CPU for saving/analysis
#     spec_lr_np = spec_lr.cpu().numpy()  # Shape: (H//8, W//8) —— USE THIS!
#     spec_hr_np = spec_hr.cpu().numpy()  # Shape: (H, W) —— for viz only

#     print(f"✅ Raw feature shape (for downstream): {spec_lr_np.shape}")
#     print(f"   Range: [{spec_lr_np.min():.4f}, {spec_lr_np.max():.4f}]")
#     print(f"   Mean: {spec_lr_np.mean():.4f}, Std: {spec_lr_np.std():.4f}")

#     # Save raw feature (e.g., for later loading in NumPy)
#     np.save("spectral_feature_raw.npy", spec_lr_np)
#     print("✅ Raw spectral feature saved to 'spectral_feature_raw.npy'")

#     # --- Visualization (optional) ---
#     original_cropped = original_image.crop((left, top, left + W_crop, top + H_crop))

#     plt.figure(figsize=(12, 5))
    
#     plt.subplot(1, 3, 1)
#     plt.imshow(original_cropped)
#     plt.title("Original (Cropped)")
#     plt.axis('off')

#     plt.subplot(1, 3, 2)
#     plt.imshow(spec_lr_np, cmap='jet')
#     plt.title(f"Raw Feature\n({spec_lr_np.shape[0]}×{spec_lr_np.shape[1]})")
#     plt.colorbar()

#     plt.subplot(1, 3, 3)
#     plt.imshow(spec_hr_np, cmap='jet')
#     plt.title("Upsampled (Viz Only)")
#     plt.colorbar()

#     plt.tight_layout()
#     plt.savefig("feature_extraction_pipeline.png", dpi=200, bbox_inches='tight')
#     plt.close()
#     print("✅ Visualization saved to 'feature_extraction_pipeline.png'")

