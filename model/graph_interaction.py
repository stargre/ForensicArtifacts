import torch
import torch.nn as nn


class PatchGraphInteraction(nn.Module):
    """
    Local neighborhood patch interaction.

    Uses depthwise convolution to propagate
    local spatial information among patches.

    Input:
        patch_tokens: [B, N, C]

    Output:
        enhanced_patch_tokens: [B, N, C]
    """

    def __init__(
        self,
        embed_dim=768,
        grid_size=16,
        dropout=0.1,
    ):
        super().__init__()

        self.grid_size = grid_size

        self.norm = nn.LayerNorm(embed_dim)

        # depthwise conv = local graph interaction
        self.dwconv = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=3,
            padding=1,
            groups=embed_dim,
        )

        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, patch_tokens):
        """
        Args:
            patch_tokens: [B, N, C]

        Returns:
            enhanced_patch_tokens: [B, N, C]
        """

        B, N, C = patch_tokens.shape

        H = W = self.grid_size

        assert H * W == N, (
            f"grid_size^2 != num_patches "
            f"({H}x{W} != {N})"
        )

        residual = patch_tokens

        # LN
        x = self.norm(patch_tokens)

        # [B, N, C] -> [B, C, H, W]
        x = x.transpose(1, 2).reshape(B, C, H, W)

        # local interaction
        x = self.dwconv(x)

        # [B, C, H, W] -> [B, N, C]
        x = x.flatten(2).transpose(1, 2)

        # channel mixing
        x = self.proj(x)

        # residual connection
        x = x + residual

        return x