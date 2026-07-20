# model/forensic_dino_orth.py
import torch
import torch.nn as nn

from model.dino_wrapper import DinoV2Wrapper
from model.domain_head import DomainHead
from model.orth_projector import OrthProjector


class SharedProjection(nn.Module):
    """
    将 cls + patch_mean 聚合后的特征映射到共享空间
    """
    def __init__(self, in_dim, out_dim=512, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.proj(x)


class ForgeryHeadFromFeature(nn.Module):
    """
    从 forgery feature 做真假分类
    输入: [B, D]
    输出: [B, 1]
    """
    def __init__(self, in_dim=256, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.head(x)


class ForensicDinoOrth(nn.Module):
    """
    Feature-level 正交分解版（无 residual）:
        shared_feat -> forgery_feat + domain_feat
        forgery_feat 用于真假分类
        domain_feat 用于域分类
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

        self.pooling_type = model_cfg.get("pooling_type", "cls_patch_mean")
        assert self.pooling_type in ["cls", "patch_mean", "cls_patch_mean"]

        embed_dim = self.backbone.embed_dim
        if self.pooling_type == "cls_patch_mean":
            shared_in_dim = embed_dim * 2
        else:
            shared_in_dim = embed_dim

        shared_dim = model_cfg.get("shared_dim", 512)
        forgery_dim = model_cfg.get("forgery_dim", 256)
        domain_dim = model_cfg.get("domain_dim", 128)
        dropout = model_cfg.get("dropout", 0.1)

        # shared feature
        self.shared_proj = SharedProjection(
            in_dim=shared_in_dim,
            out_dim=shared_dim,
            dropout=dropout
        )

        # forgery projector（恢复成普通 projector）
        self.forgery_projector = OrthProjector(
            in_dim=shared_dim,
            hidden_dim=model_cfg.get("forgery_hidden_dim", shared_dim),
            out_dim=forgery_dim,
            dropout=dropout
        )

        # domain projector（仍可保留你现在更小容量的版本）
        self.domain_projector = OrthProjector(
            in_dim=shared_dim,
            hidden_dim=model_cfg.get("domain_projector_hidden_dim", 256),
            out_dim=domain_dim,
            dropout=dropout
        )

        # forgery head
        self.forgery_head = ForgeryHeadFromFeature(
            in_dim=forgery_dim,
            hidden_dim=model_cfg.get("hidden_dim", 128),
            dropout=dropout
        )

        # domain head
        self.domain_head = DomainHead(
            in_dim=domain_dim,
            hidden_dim=model_cfg.get("domain_hidden_dim", 64),
            num_domains=model_cfg.get("num_domains", 4),
            dropout=dropout
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

        # 聚合 DINO 特征
        shared_input = self.aggregate_features(cls_token, patch_tokens)

        # shared feature
        shared_feat = self.shared_proj(shared_input)

        # 正交分解出的两类特征
        forgery_feat = self.forgery_projector(shared_feat)
        domain_feat = self.domain_projector(shared_feat)

        # 两个 head
        forgery_logits = self.forgery_head(forgery_feat)
        domain_logits = self.domain_head(domain_feat)

        return {
            "forgery_logits": forgery_logits,
            "domain_logits": domain_logits,
            "cls_token": cls_token,
            "patch_tokens": patch_tokens,
            "shared_feat": shared_feat,
            "forgery_feat": forgery_feat,
            "domain_feat": domain_feat,
        }