import numpy as np
from PIL import Image
import cv2
from sklearn.decomposition import PCA
import os

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

def retinex_decompose_lime_bilateral(img_rgb):
    """
    High-quality Retinex decomposition without guidedFilter.
    Uses max-RGB + morphological closing + bilateral filtering.
    """
    img_f = img_rgb.astype(np.float32) / 255.0
    H, W = img_f.shape[:2]

    # Step 1: Initial illumination (max-RGB)
    L_init = np.max(img_f, axis=2)  # (H, W)

    # Step 2: Morphological closing to fill small dark holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    L_closed = cv2.morphologyEx(L_init, cv2.MORPH_CLOSE, kernel)

    # Step 3: Bilateral filtering for edge-preserving smoothing
    L_bf = cv2.bilateralFilter(
        (L_closed * 255).astype(np.uint8),
        d=25,                # diameter of pixel neighborhood
        sigmaColor=80,       # color space sigma
        sigmaSpace=80        # coordinate space sigma
    ).astype(np.float32) / 255.0

    # Step 4: Enforce physical constraint: L >= max(R,G,B)
    L_final = np.maximum(L_bf, np.max(img_f, axis=2))
    L_final = np.clip(L_final, 1e-6, 1.0)

    # Reflectance (not used, but computed for completeness)
    R = img_f / L_final[..., np.newaxis]
    return L_final, R

def compute_lighting_shadow_anomaly_map(image_pil, tau_deg=30.0):
    image_rgb = np.array(image_pil)  # (H, W, 3), uint8
    H, W = image_rgb.shape[:2]

    # --- Step 1: Retinex Decomposition ---
    print("🔍 Performing Retinex decomposition (max-RGB + morph + bilateral)...")
    L, _ = retinex_decompose_lime_bilateral(image_rgb)
    print("✅ Illumination map L obtained.")

    # --- Step 2: Compute gradients of L ---
    grad_x = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=5)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # --- Step 3: Estimate global light direction via PCA on strong gradients ---
    mask = grad_mag > np.percentile(grad_mag, 85)  # top 15%
    if np.sum(mask) < 50:
        mask = grad_mag > np.mean(grad_mag)
    
    grad_vectors = np.stack([grad_x[mask], grad_y[mask]], axis=1)
    if grad_vectors.shape[0] < 2:
        l_2d = np.array([1.0, 0.0])
    else:
        pca = PCA(n_components=1)
        pca.fit(grad_vectors)
        l_2d = pca.components_[0]
        l_2d = l_2d / (np.linalg.norm(l_2d) + 1e-8)
    print(f"✅ Global light direction (2D): [{l_2d[0]:.3f}, {l_2d[1]:.3f}]")

    # --- Step 4: Shadow direction = -∇L ---
    d_shadow_x = -grad_x
    d_shadow_y = -grad_y
    norm = np.sqrt(d_shadow_x**2 + d_shadow_y**2)
    norm = np.maximum(norm, 1e-8)
    d_shadow_x /= norm
    d_shadow_y /= norm

    # --- Step 5: Angle between d_shadow and l ---
    dot = d_shadow_x * l_2d[0] + d_shadow_y * l_2d[1]
    dot = np.clip(dot, -1.0, 1.0)
    theta_deg = np.degrees(np.arccos(np.abs(dot)))  # [0, 90]

    # --- Step 6: Sigmoid heatmap ---
    k = 0.25
    M = 1.0 / (1.0 + np.exp(-k * (theta_deg - tau_deg)))

    return M.astype(np.float32), theta_deg, L


def extract_lighting_feature(image_512: np.ndarray) -> np.ndarray:
    """
    输入: (512, 512, 3) RGB uint8
    输出: (512, 512, 3) float32 —— [L, dx, dy] 
           L = illumination, (dx, dy) = shadow direction unit vector
    """
    assert image_512.shape == (512, 512, 3) and image_512.dtype == np.uint8

    img_rgb = image_512.copy()
    H, W = img_rgb.shape[:2]

    # Retinex 分解
    L, _ = retinex_decompose_lime_bilateral(img_rgb)  # (512,512)

    # 梯度
    grad_x = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=5)

    # Shadow direction = -∇L, normalized
    d_shadow_x = -grad_x
    d_shadow_y = -grad_y
    norm = np.sqrt(d_shadow_x**2 + d_shadow_y**2)
    norm = np.maximum(norm, 1e-8)
    d_shadow_x /= norm
    d_shadow_y /= norm

    features = np.stack([L, d_shadow_x, d_shadow_y], axis=-1)  # (512,512,3)
    return features.astype(np.float32)


# === Main ===
if __name__ == "__main__":
    image_path = "/mnt/data3/public_datasets/OpenMMSec/3/e685278670eb41f1bb35f9f88510a1c1.jpg"
    assert os.path.exists(image_path), f"Image not found: {image_path}"
    
    image = Image.open(image_path).convert("RGB")
    light_map, theta_map, L_map = compute_lighting_shadow_anomaly_map(image, tau_deg=30.0)

    # Save heatmap
    vis = (light_map * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    output_path = "lighting_shadow_heatmap.png"
    cv2.imwrite(output_path, heatmap)
    print(f"🔥 Heatmap saved to: {output_path}")

    # Optional: save L for inspection
    cv2.imwrite("illumination_L.png", (L_map * 255).astype(np.uint8))

