# model/dino_fixed_topx.py
import torch
import torch.nn as nn

from model.dino_wrapper import DinoV2Wrapper


class ForensicDinoFixedTopX(nn.Module):
    """
    固定硬分流版本（pre-pooling channel suppression）：
      - 不改 baseline 的 classifier_head.py
      - 在本模型内部复制 baseline 的 pooling 和 classifier 结构
      - 保证 topx=0 时与 baseline 结构等价

    改动点：
      1. mask 不再作用于 pooled_feat
      2. mask 前移到 backbone 输出后的 cls_token / patch_tokens
      3. mask 维度 = embed_dim，而不是 feat_dim
    """

    def __init__(self, config):
        super().__init__()
        model_cfg = config["model"]

        self.backbone = DinoV2Wrapper(
            repo_path=model_cfg.get("repo_path", "/mnt/data3/zhiyu/dino_clip/dinov2_repo"),
            model_name=model_cfg.get("backbone_name", "dinov2_vitb14_reg"),
            pretrained=model_cfg.get("pretrained", True),
            freeze_backbone=model_cfg.get("freeze_backbone", True),
            unfreeze_last_n_blocks=model_cfg.get("unfreeze_last_n_blocks", 0),
            unfreeze_norm=model_cfg.get("unfreeze_norm", True),
            verbose=True,
        )

        self.pooling_type = model_cfg.get("pooling_type", "cls_patch_mean")
        assert self.pooling_type in ["cls", "patch_mean", "cls_patch_mean"]

        self.embed_dim = self.backbone.embed_dim

        if self.pooling_type == "cls_patch_mean":
            self.feat_dim = self.embed_dim * 2
        else:
            self.feat_dim = self.embed_dim

        hidden_dim = model_cfg.get("hidden_dim", 512)
        dropout = model_cfg.get("dropout", 0.1)

        # 这里严格复制 baseline 的 classifier 结构
        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        # 注意：mask 维度现在是 embed_dim，而不是 feat_dim
        self.register_buffer("shortcut_mask", torch.zeros(self.embed_dim))
        self.register_buffer("core_mask", torch.ones(self.embed_dim))

    def pool_features(self, cls_token, patch_tokens):
        """
        严格复制 baseline 中 DinoClassificationHead 的 pooling 逻辑
        """
        if self.pooling_type == "cls":
            feat = cls_token

        elif self.pooling_type == "patch_mean":
            feat = patch_tokens.mean(dim=1)

        elif self.pooling_type == "cls_patch_mean":
            patch_mean = patch_tokens.mean(dim=1)
            feat = torch.cat([cls_token, patch_mean], dim=-1)

        else:
            raise ValueError(f"Unknown pooling_type: {self.pooling_type}")

        return feat

    @torch.no_grad()
    def set_fixed_shortcut_mask(self, shortcut_mask):
        shortcut_mask = shortcut_mask.float().view(-1)
        if shortcut_mask.numel() != self.embed_dim:
            raise ValueError(
                f"shortcut_mask dim={shortcut_mask.numel()} != embed_dim={self.embed_dim}"
            )

        shortcut_mask = (shortcut_mask > 0.5).float()
        core_mask = 1.0 - shortcut_mask

        self.shortcut_mask.copy_(shortcut_mask)
        self.core_mask.copy_(core_mask)

    def get_mask_mean(self):
        return float(self.core_mask.mean().item())

    def extract_pooled_features(self, images):
        """
        用于 probe / routing score 估计
        这里仍然返回 baseline 风格的 pooled_feat（未mask）
        """
        cls_token, patch_tokens = self.backbone(images)
        pooled_feat = self.pool_features(cls_token, patch_tokens)
        return pooled_feat, cls_token, patch_tokens

    def forward(self, images, grl_lambda=0.0):
        """
        返回格式尽量兼容你现有 train 脚本
        """
        cls_token, patch_tokens = self.backbone(images)   # [B, C], [B, N, C]

        # 原始 pooled feature（仅用于日志 / 分析）
        pooled_feat = self.pool_features(cls_token, patch_tokens)

        # ===== pre-pooling channel suppression =====
        core_mask = self.core_mask.view(1, -1)                # [1, C]
        shortcut_mask = self.shortcut_mask.view(1, -1)        # [1, C]

        cls_core = cls_token * core_mask                      # [B, C]
        cls_short = cls_token * shortcut_mask                 # [B, C]

        patch_core = patch_tokens * core_mask.unsqueeze(1)    # [B, N, C]
        patch_short = patch_tokens * shortcut_mask.unsqueeze(1)

        core_feat = self.pool_features(cls_core, patch_core)          # [B, feat_dim]
        shortcut_feat = self.pool_features(cls_short, patch_short)    # [B, feat_dim]

        cls_logits = self.classifier(core_feat)

        # 为了兼容原日志，mask 仍然返回 pooled 后同维度的 mask
        if self.pooling_type == "cls_patch_mean":
            pooled_mask = torch.cat([self.core_mask, self.core_mask], dim=0)   # [2C]
        else:
            pooled_mask = self.core_mask                                        # [C]

        mask = pooled_mask.unsqueeze(0).expand(core_feat.size(0), -1)

        return {
            "cls_logits": cls_logits,
            "pooled_feat": pooled_feat,      # 未mask原特征
            "core_feat": core_feat,          # mask后用于分类
            "shortcut_feat": shortcut_feat,  # shortcut分支
            "mask": mask,
            "cls_token": cls_token,
            "patch_tokens": patch_tokens,
        }