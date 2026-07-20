import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    """
    Learnable attention pooling over patch tokens.

    Input:
        patch_tokens: [B, N, C]

    Output:
        pooled_feat: [B, C]
    """

    def __init__(
        self,
        embed_dim=768,
        hidden_dim=256,
        dropout=0.1,
    ):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, patch_tokens):
        """
        Args:
            patch_tokens:
                [B, N, C]

        Returns:
            pooled_feat:
                [B, C]
        """

        # [B, N, 1]
        attn_scores = self.attention(patch_tokens)

        # [B, N, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)

        # weighted sum
        pooled_feat = (patch_tokens * attn_weights).sum(dim=1)

        return pooled_feat