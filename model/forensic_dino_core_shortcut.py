import torch
import torch.nn as nn
from torch.autograd import Function

from model.dino_wrapper import DinoV2Wrapper


# =========================================================
# 反向梯度层
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
# 特征池化
# =========================================================
class DinoFeaturePooler(nn.Module):
    """
    把 cls_token 和 patch_tokens 变成一个向量 h
    支持：
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
            raise ValueError(f"未知 pooling_type: {self.pooling_type}")

    def forward(self, cls_token, patch_tokens):
        if self.pooling_type == "cls":
            return cls_token
        elif self.pooling_type == "patch_mean":
            return patch_tokens.mean(dim=1)
        elif self.pooling_type == "cls_patch_mean":
            patch_mean = patch_tokens.mean(dim=1)
            return torch.cat([cls_token, patch_mean], dim=1)
        else:
            raise ValueError(f"未知 pooling_type: {self.pooling_type}")


# =========================================================
# 小头
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
# 主模型：核心特征 / 捷径特征
# =========================================================
class ForensicDinoCoreShortcut(nn.Module):
    """
    输出:
        cls_logits        : [B, 1]      真假预测（用 core_feat）
        domain_logits     : [B, D]      域预测（用 shortcut_feat）
        domain_adv_logits : [B, D]      对抗域预测（用 core_feat + 反向梯度）
        pooled_feat       : [B, F]      原始聚合特征 h
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
            repo_path=model_cfg.get("repo_path", "/mnt/data3/zhiyu/dino_clip/dinov2_repo"),
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

        # 域数
        self.num_domains = int(model_cfg["num_domains"])

        # 掩码网络：产生每个通道保留到 core 的比例
        self.mask_net = nn.Sequential(
            nn.Linear(self.feat_dim, mask_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mask_hidden_dim, self.feat_dim)
        )

        # 真假分类头：看 core_feat
        self.cls_head = MLPHead(
            in_dim=self.feat_dim,
            hidden_dim=hidden_dim,
            out_dim=1,
            dropout=dropout
        )

        # 域分类头：看 shortcut_feat
        self.domain_head = MLPHead(
            in_dim=self.feat_dim,
            hidden_dim=hidden_dim,
            out_dim=self.num_domains,
            dropout=dropout
        )

        # 对抗域分类头：看 core_feat
        self.domain_adv_head = MLPHead(
            in_dim=self.feat_dim,
            hidden_dim=hidden_dim,
            out_dim=self.num_domains,
            dropout=dropout
        )

        self.grl = GradientReverseLayer()

    def forward(self, images, grl_lambda=1.0):
        cls_token, patch_tokens = self.backbone(images)

        # h: 原始聚合特征
        pooled_feat = self.pooler(cls_token, patch_tokens)

        # 如果 backbone 被 no_grad 包住了，那么 pooled_feat 本身不带梯度
        # 这里把它变成一个可求导的“叶子”，这样后面可以算 DCS 的梯度归因
        if not pooled_feat.requires_grad:
            pooled_feat = pooled_feat.detach().requires_grad_(True)

        # mask: 每个通道给 core_feat 的比例
        mask_logits = self.mask_net(pooled_feat) / max(self.mask_temperature, 1e-6)
        mask = torch.sigmoid(mask_logits)

        # 互补分解
        core_feat = mask * pooled_feat
        shortcut_feat = (1.0 - mask) * pooled_feat

        # 主任务：真假分类
        cls_logits = self.cls_head(core_feat)

        # 捷径分支：域分类
        domain_logits = self.domain_head(shortcut_feat)

        # 核心分支：对抗域分类
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
# 训练时会用到的几个损失工具
# =========================================================
def cross_covariance_loss(core_feat, shortcut_feat):
    """
    软去相关：不做硬正交，只做统计意义上的去相关
    """
    z1 = core_feat - core_feat.mean(dim=0, keepdim=True)
    z2 = shortcut_feat - shortcut_feat.mean(dim=0, keepdim=True)

    denom = max(1, z1.shape[0] - 1)
    cov = (z1.t() @ z2) / denom
    return (cov ** 2).mean()


def compute_mask_regularization(mask, target_keep_ratio=0.5):
    """
    两个约束：
    1. mask 平均值别塌掉，全进 core 或全进 shortcut 都不好
    2. 鼓励 mask 更接近 0/1，不要全是 0.5
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
    DCS 核心：
    - 看 pooled_feat 每个通道对“真假损失”有多敏感
    - 看 pooled_feat 每个通道对“域损失”有多敏感
    - 如果某通道更偏域、却不怎么服务于真假，就减少它进入 core 的比例

    返回:
        loss_dcs
        info: 便于日志打印
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

    # 每个通道的重要性：|梯度| * |激活|
    cls_score = (grad_cls.detach().abs() * pooled_feat.detach().abs()).mean(dim=0)      # [F]
    domain_score = (grad_domain.detach().abs() * pooled_feat.detach().abs()).mean(dim=0) # [F]

    # 抑制分数：域敏感 > 真假敏感 的部分
    dcs_score = torch.relu(domain_score - alpha * cls_score)

    # 归一化，避免数值太大
    dcs_score = dcs_score / (dcs_score.mean() + eps)

    # mask 大，说明这个通道更流向 core
    # 高 dcs_score 的通道，不应该太多流向 core
    loss_dcs = (mask.mean(dim=0) * dcs_score).mean()

    info = {
        "cls_score_mean": float(cls_score.mean().item()),
        "domain_score_mean": float(domain_score.mean().item()),
        "dcs_score_mean": float(dcs_score.mean().item()),
    }
    return loss_dcs, info