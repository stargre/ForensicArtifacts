import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

def normalize_to_01(x):
    """
    Normalize the input array to [0, 1] range.
    """
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def apply_high_order_differences(image_tensor):
    """
    Apply 4th-order difference along horizontal and vertical directions.
    Output residuals have the same spatial size as input image (H, W).
    """
    device = image_tensor.device
    x = image_tensor.squeeze(0)  # (H, W)
    H, W = x.shape

    kernel = torch.tensor([1, -4, 6, -4, 1], dtype=torch.float32, device=device)

    # Horizontal: pad left/right
    x_pad_h = torch.nn.functional.pad(x.unsqueeze(0).unsqueeze(0), (2, 2, 0, 0), mode='replicate')
    res_h = torch.nn.functional.conv2d(x_pad_h, kernel.view(1, 1, 1, -1)).squeeze()  # (H, W)

    # Vertical: pad top/bottom
    x_pad_v = torch.nn.functional.pad(x.unsqueeze(0).unsqueeze(0), (0, 0, 2, 2), mode='replicate')
    res_v = torch.nn.functional.conv2d(x_pad_v, kernel.view(1, 1, -1, 1)).squeeze()  # (H, W)

    return [res_h, res_v]

def extract_resampling_feature(residual_maps, shifts=None):
    """
    Compute shifted correlation maps. All outputs are same size as input residuals.
    """
    if shifts is None:
        shifts = [(2, 0), (0, 2), (3, 0), (0, 3)]  # support bilinear & bicubic

    responses = []
    for res in residual_maps:
        H, W = res.shape
        for dx, dy in shifts:
            shifted = torch.roll(res, shifts=(dx, dy), dims=(0, 1))
            
            # Create mask to zero out wrapped-around edges
            mask = torch.ones_like(res)
            if dx > 0:
                mask[:dx, :] = 0
            elif dx < 0:
                mask[dx:, :] = 0
            if dy > 0:
                mask[:, :dy] = 0
            elif dy < 0:
                mask[:, dy:] = 0

            # Compute absolute correlation, keep full size
            corr = torch.abs(res * shifted) * mask
            responses.append(corr)

    # Now all tensors in responses have same shape (H, W)
    stacked = torch.stack(responses, dim=0)  # (N, H, W)
    confidence_map = torch.max(stacked, dim=0)[0]  # (H, W)

    # Step 3: Apply normalization to ensure values are in [0, 1]
    confidence_map_normalized = normalize_to_01(confidence_map.cpu().numpy())
    
    return confidence_map_normalized

# # ======================
# # Main
# # ======================
# if __name__ == "__main__":
#     image_path = "/mnt/data3/zhiyu/fake_resampled.jpg"
#     original_image = Image.open(image_path).convert("L")
#     image_tensor = transforms.ToTensor()(original_image)  # (1, H, W)

#     assert image_tensor.shape[0] == 1, "Must be grayscale"

#     # Step 1: Get residuals (same size as input)
#     residual_maps = apply_high_order_differences(image_tensor)

#     # Step 2: Compute confidence map
#     confidence_map = compute_shifted_correlation(residual_maps)

#     # Visualization
#     plt.figure(figsize=(15, 4))

#     plt.subplot(1, 4, 1)
#     plt.title('Original')
#     plt.imshow(original_image, cmap='gray')
#     plt.axis('off')

#     plt.subplot(1, 4, 2)
#     plt.title('Residual (Horizontal)')
#     plt.imshow(residual_maps[0].cpu().numpy(), cmap='seismic')
#     plt.axis('off')

#     plt.subplot(1, 4, 3)
#     plt.title('Residual (Vertical)')
#     plt.imshow(residual_maps[1].cpu().numpy(), cmap='seismic')
#     plt.axis('off')

#     plt.subplot(1, 4, 4)
#     plt.title('Resampling Confidence')
#     plt.imshow(confidence_map, cmap='hot')
#     plt.colorbar()
#     plt.axis('off')

#     plt.tight_layout()
#     plt.savefig("resampling_detection_optimal.png", dpi=200, bbox_inches='tight')
#     print("✅ Saved to 'resampling_detection_optimal.png'")