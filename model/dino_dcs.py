import torch
import torch.nn as nn
from torch.autograd import Function

from model.dino_wrapper import DinoV2Wrapper


# =========================================================
# Gradient Reversal Layer
# =========================================================
class _GradientReverseFn(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class GradientReverseLayer(nn.Module):
    def forward(self, x, lambd=1.0):
        return _GradientReverseFn.apply(x, lambd)


# =========================================================
# Feature pooling
# =========================================================
class DinoFeaturePooler(nn.Module):
    """
    将 cls_token 和 patch_tokens 汇聚成一个向量
    支持:
        - cls
        - patch_mean
        - cls_patch_mean
    """
    def __init__(self, pooling_type="cls_patch_mean"):
        super().__init__()
        self.pooling_type = pooling_type

    def get_out_dim(self, embed_dim):
        if self.pooling_type in ["cls", "patch_mean"]:
            return embed_dim
        elif self.pooling_type == "cls_patch_mean":
            return embed_dim * 2
        else:
            raise ValueError(f"Unknown pooling_type: {self.pooling_type}")

    def forward(self, cls_token, patch_tokens):
        if self.pooling_type == "cls":
            return cls_token
        elif self.pooling_type == "patch_mean":
            return patch_tokens.mean(dim=1)
        elif self.pooling_type == "cls_patch_mean":
            patch_mean = patch_tokens.mean(dim=1)
            return torch.cat([cls_token, patch_mean], dim=1)
        else:
            raise ValueError(f"Unknown pooling_type: {self.pooling_type}")


# =========================================================
# Heads
# =========================================================
class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


# =========================================================
# Main model
# =========================================================
class ForensicDinoCoreShortcut(nn.Module):
    """
    输出:
        cls_logits        : [B, 1]   真假预测（core_feat）
        domain_logits     : [B, D]   域预测（shortcut_feat）
        domain_adv_logits : [B, D]   对抗域预测（core_feat + GRL）
        pooled_feat       : [B, F]   原始聚合特征 h
        core_feat         : [B, F]
        shortcut_feat     : [B, F]
        mask              : [B, F]
        cls_token         : [B, C]
        patch_tokens      : [B, N, C]
    """

    def __init__(self, config):
        super().__init__()
        model_cfg = config["model"]

        self.backbone = DinoV2Wrapper(
            repo_path=model_cfg.get("repo_path", ""),
            model_name=model_cfg.get("backbone_name", "dinov2_vitb14_reg"),
            pretrained=model_cfg.get("pretrained", True),
            freeze_backbone=model_cfg.get("freeze_backbone", True),
            unfreeze_last_n_blocks=model_cfg.get("unfreeze_last_n_blocks", 0),
            unfreeze_norm=model_cfg.get("unfreeze_norm", True),
            verbose=True,
        )

        self.pooler = DinoFeaturePooler(
            pooling_type=model_cfg.get("pooling_type", "cls_patch_mean")
        )

        self.feat_dim = self.pooler.get_out_dim(self.backbone.embed_dim)

        hidden_dim = model_cfg.get("hidden_dim", 512)
        dropout = model_cfg.get("dropout", 0.1)
        mask_hidden_dim = model_cfg.get("mask_hidden_dim", 256)
        self.mask_temperature = model_cfg.get("mask_temperature", 1.0)

        self.num_domains = int(model_cfg["num_domains"])

        use_feature_norm = model_cfg.get("use_feature_norm", False)
        self.feature_norm = nn.LayerNorm(self.feat_dim) if use_feature_norm else nn.Identity()

        # 产生互补掩码：每个通道流向 core 的比例
        self.mask_net = nn.Sequential(
            nn.Linear(self.feat_dim, mask_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mask_hidden_dim, self.feat_dim)
        )

        # 真假分类（看 core）
        self.cls_head = MLPHead(
            in_dim=self.feat_dim,
            hidden_dim=hidden_dim,
            out_dim=1,
            dropout=dropout
        )

        # 域分类（看 shortcut）
        self.domain_head = MLPHead(
            in_dim=self.feat_dim,
            hidden_dim=hidden_dim,
            out_dim=self.num_domains,
            dropout=dropout
        )

        # 对抗域分类（看 core）
        self.domain_adv_head = MLPHead(
            in_dim=self.feat_dim,
            hidden_dim=hidden_dim,
            out_dim=self.num_domains,
            dropout=dropout
        )

        self.grl = GradientReverseLayer()

    def forward(self, images, grl_lambda=1.0):
        cls_token, patch_tokens = self.backbone(images)

        pooled_feat = self.pooler(cls_token, patch_tokens)
        pooled_feat = self.feature_norm(pooled_feat)

        # 如果 backbone 冻结且 no_grad，这里给 pooled_feat 补上 grad，
        # 方便后面做 DCS 的梯度归因
        if not pooled_feat.requires_grad:
            pooled_feat = pooled_feat.detach().requires_grad_(True)

        mask_logits = self.mask_net(pooled_feat) / max(self.mask_temperature, 1e-6)
        mask = torch.sigmoid(mask_logits)

        # 互补掩码分流
        core_feat = mask * pooled_feat
        shortcut_feat = (1.0 - mask) * pooled_feat

        # 主任务：真假
        cls_logits = self.cls_head(core_feat)

        # 捷径分支：域
        domain_logits = self.domain_head(shortcut_feat)

        # 核心分支：域对抗
        rev_core_feat = self.grl(core_feat, grl_lambda)
        domain_adv_logits = self.domain_adv_head(rev_core_feat)

        return {
            "cls_logits": cls_logits,
            "domain_logits": domain_logits,
            "domain_adv_logits": domain_adv_logits,
            "pooled_feat": pooled_feat,
            "core_feat": core_feat,
            "shortcut_feat": shortcut_feat,
            "mask": mask,
            "cls_token": cls_token,
            "patch_tokens": patch_tokens,
        }


# =========================================================
# Loss utils
# =========================================================
def cross_covariance_loss(core_feat, shortcut_feat):
    """
    软去相关，不做硬正交
    """
    if core_feat.size(0) <= 1:
        return core_feat.new_tensor(0.0)

    z1 = core_feat - core_feat.mean(dim=0, keepdim=True)
    z2 = shortcut_feat - shortcut_feat.mean(dim=0, keepdim=True)
    cov = (z1.t() @ z2) / max(1, z1.shape[0] - 1)
    return (cov ** 2).mean()


def compute_mask_regularization(mask, target_keep_ratio=0.5):
    """
    1) 防止 mask 全塌到 0 或 1
    2) 轻微鼓励更接近二值
    """
    loss_ratio = (mask.mean() - target_keep_ratio) ** 2
    loss_binary = (mask * (1.0 - mask)).mean()
    return loss_ratio, loss_binary


def compute_dcs_loss(
    pooled_feat,
    mask,
    loss_cls,
    loss_domain,
    alpha=1.0,
    eps=1e-6
):
    """
    DCS:
      cls_score(c)    = E[ |dL_cls/dh_c| * |h_c| ]
      domain_score(c) = E[ |dL_dom/dh_c| * |h_c| ]
      dcs_score(c)    = ReLU(domain_score - alpha * cls_score)

    高 dcs_score 的通道更偏域，不应大量流入 core。
    """
    grad_cls = torch.autograd.grad(
        loss_cls,
        pooled_feat,
        retain_graph=True,
        create_graph=False,
        allow_unused=True
    )[0]

    grad_domain = torch.autograd.grad(
        loss_domain,
        pooled_feat,
        retain_graph=True,
        create_graph=False,
        allow_unused=True
    )[0]

    if grad_cls is None or grad_domain is None:
        zero = pooled_feat.new_tensor(0.0)
        return zero, {
            "cls_score_mean": 0.0,
            "domain_score_mean": 0.0,
            "dcs_score_mean": 0.0
        }

    cls_score = (grad_cls.detach().abs() * pooled_feat.detach().abs()).mean(dim=0)
    domain_score = (grad_domain.detach().abs() * pooled_feat.detach().abs()).mean(dim=0)

    dcs_score = torch.relu(domain_score - alpha * cls_score)
    dcs_score = dcs_score / (dcs_score.mean() + eps)

    # mask 越大，表示越流向 core
    # dcs_score 越大，表示越不应该进入 core
    loss_dcs = (mask.mean(dim=0) * dcs_score).mean()

    info = {
        "cls_score_mean": float(cls_score.mean().item()),
        "domain_score_mean": float(domain_score.mean().item()),
        "dcs_score_mean": float(dcs_score.mean().item()),
    }
    return loss_dcs, info