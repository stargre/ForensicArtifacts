import torch
import torch.nn as nn

from model.attention_pooling import AttentionPooling


class DinoClassificationHead(nn.Module):
    """
    DINOv2 classification head with optional attention pooling.

    Supported pooling_type:
        - cls
        - patch_mean
        - patch_max
        - cls_patch_mean
        - cls_patch_max
        - patch_mean_max
        - cls_patch_mean_max
        - cls_patch_attention_mean
    """

    def __init__(
        self,
        embed_dim=768,
        pooling_type="cls_patch_mean",
        hidden_dim=512,
        dropout=0.1,
        enable_attention_pooling=False,
        attention_hidden_dim=256,
    ):
        super().__init__()

        self.pooling_type = pooling_type
        self.embed_dim = embed_dim
        self.enable_attention_pooling = enable_attention_pooling

        supported_pooling = [
            "cls",
            "patch_mean",
            "patch_max",
            "cls_patch_mean",
            "cls_patch_max",
            "patch_mean_max",
            "cls_patch_mean_max",
            "cls_patch_attention_mean",
        ]

        assert pooling_type in supported_pooling, (
            f"Unsupported pooling_type={pooling_type}"
        )

        # =====================================================
        # Attention Pooling
        # =====================================================
        if self.enable_attention_pooling:
            self.attention_pool = AttentionPooling(
                embed_dim=embed_dim,
                hidden_dim=attention_hidden_dim,
                dropout=dropout,
            )

        # =====================================================
        # Input Dimension
        # =====================================================
        if pooling_type in ["cls", "patch_mean", "patch_max"]:
            in_dim = embed_dim

        elif pooling_type in [
            "cls_patch_mean",
            "cls_patch_max",
            "patch_mean_max",
        ]:
            in_dim = embed_dim * 2

        elif pooling_type == "cls_patch_mean_max":
            in_dim = embed_dim * 3

        elif pooling_type == "cls_patch_attention_mean":
            in_dim = embed_dim * 3

        else:
            raise ValueError(f"Unknown pooling_type: {pooling_type}")

        self.in_dim = in_dim

        # =====================================================
        # Classifier
        # =====================================================
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, cls_token, patch_tokens):

        patch_mean = patch_tokens.mean(dim=1)
        patch_max = patch_tokens.max(dim=1).values

        if self.enable_attention_pooling:
            patch_attention = self.attention_pool(
                patch_tokens
            )

        # =====================================================
        # Feature Aggregation
        # =====================================================
        if self.pooling_type == "cls":
            feat = cls_token

        elif self.pooling_type == "patch_mean":
            feat = patch_mean

        elif self.pooling_type == "patch_max":
            feat = patch_max

        elif self.pooling_type == "cls_patch_mean":
            feat = torch.cat(
                [cls_token, patch_mean],
                dim=-1
            )

        elif self.pooling_type == "cls_patch_max":
            feat = torch.cat(
                [cls_token, patch_max],
                dim=-1
            )

        elif self.pooling_type == "patch_mean_max":
            feat = torch.cat(
                [patch_mean, patch_max],
                dim=-1
            )

        elif self.pooling_type == "cls_patch_mean_max":
            feat = torch.cat(
                [cls_token, patch_mean, patch_max],
                dim=-1
            )

        elif self.pooling_type == "cls_patch_attention_mean":

            if not self.enable_attention_pooling:
                raise ValueError(
                    "Attention pooling not enabled."
                )

            feat = torch.cat(
                [
                    cls_token,
                    patch_mean,
                    patch_attention
                ],
                dim=-1
            )

        else:
            raise ValueError(
                f"Unknown pooling_type: {self.pooling_type}"
            )

        logits = self.classifier(feat)

        return logits