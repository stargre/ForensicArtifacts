import torch
import torch.nn as nn

from model.dino_wrapper import DinoV2Wrapper
from model.classifier_head import DinoClassificationHead
from model.domain_head import DomainHead
from model.grl import GradientReversalLayer


class SharedProjection(nn.Module):
    """
    将 cls + patch_mean 聚合后的高维特征投影到共享特征空间
    """
    def __init__(self, in_dim, out_dim=512, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.proj(x)


class ForgeryHeadFromShared(nn.Module):
    """
    从 shared feature 做真假分类
    输入: [B, D]
    输出: [B, 1]
    """
    def __init__(self, in_dim=512, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.head(x)


class ForensicDinoDANN(nn.Module):
    """
    DINO + 域对抗版本
    """

    def __init__(self, config):
        super().__init__()
        model_cfg = config["model"]

        self.backbone = DinoV2Wrapper(
            repo_path=model_cfg.get("repo_path", "/mnt/data3/zhiyu/dino_clip/dinov2_repo"),
            model_name=model_cfg.get("backbone_name", "dinov2_vitb14_reg"),
            pretrained=model_cfg.get("pretrained", True),
            freeze_backbone=model_cfg.get("freeze_backbone", True)
        )

        pooling_type = model_cfg.get("pooling_type", "cls_patch_mean")
        assert pooling_type in ["cls", "patch_mean", "cls_patch_mean"]
        self.pooling_type = pooling_type

        embed_dim = self.backbone.embed_dim
        if pooling_type == "cls_patch_mean":
            shared_in_dim = embed_dim * 2
        else:
            shared_in_dim = embed_dim

        self.shared_proj = SharedProjection(
            in_dim=shared_in_dim,
            out_dim=model_cfg.get("shared_dim", 512),
            dropout=model_cfg.get("dropout", 0.1)
        )

        self.forgery_head = ForgeryHeadFromShared(
            in_dim=model_cfg.get("shared_dim", 512),
            hidden_dim=model_cfg.get("hidden_dim", 256),
            dropout=model_cfg.get("dropout", 0.1)
        )

        self.grl = GradientReversalLayer(
            lambda_grl=model_cfg.get("lambda_grl", 1.0)
        )

        self.domain_head = DomainHead(
            in_dim=model_cfg.get("shared_dim", 512),
            hidden_dim=model_cfg.get("domain_hidden_dim", 256),
            num_domains=model_cfg.get("num_domains", 4),
            dropout=model_cfg.get("dropout", 0.1)
        )

    def aggregate_features(self, cls_token, patch_tokens):
        if self.pooling_type == "cls":
            feat = cls_token
        elif self.pooling_type == "patch_mean":
            feat = patch_tokens.mean(dim=1)
        elif self.pooling_type == "cls_patch_mean":
            patch_mean = patch_tokens.mean(dim=1)
            feat = torch.cat([cls_token, patch_mean], dim=-1)
        return feat

    def forward(self, images):
        cls_token, patch_tokens = self.backbone(images)

        shared_input = self.aggregate_features(cls_token, patch_tokens)
        shared_feat = self.shared_proj(shared_input)              # [B, shared_dim]

        forgery_logits = self.forgery_head(shared_feat)          # [B, 1]

        domain_feat = self.grl(shared_feat)
        domain_logits = self.domain_head(domain_feat)            # [B, num_domains]

        return {
            "forgery_logits": forgery_logits,
            "domain_logits": domain_logits,
            "cls_token": cls_token,
            "patch_tokens": patch_tokens,
            "shared_feat": shared_feat,
        }