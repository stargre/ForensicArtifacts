# model/orth_projector.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class OrthProjector(nn.Module):
    """
    两层 projector:
        in_dim -> hidden_dim -> out_dim
    用于从 shared feature 中提取子空间特征
    """
    def __init__(self, in_dim=512, hidden_dim=512, out_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


def sample_dot_orthogonal_loss(f_forgery, f_domain, normalize=True):
    """
    逐样本点积正交损失
    要求 forgery/domain feature 维度相同

    L = mean( (f_i^T d_i)^2 )

    Args:
        f_forgery: [B, D]
        f_domain:  [B, D]
        normalize: 是否先对每个样本做 L2 normalize
    """
    if f_forgery.shape[1] != f_domain.shape[1]:
        raise ValueError(
            f"[sample_dot_orthogonal_loss] forgery/domain 维度必须一致，"
            f"但得到 {f_forgery.shape} vs {f_domain.shape}"
        )

    if normalize:
        f_forgery = F.normalize(f_forgery, dim=1)
        f_domain = F.normalize(f_domain, dim=1)

    dot = torch.sum(f_forgery * f_domain, dim=1)   # [B]
    loss = torch.mean(dot ** 2)
    return loss


def matrix_corr_orthogonal_loss(f_forgery, f_domain, normalize=True, center=True):
    """
    矩阵正交 / 交叉相关正交损失
    支持 forgery/domain feature 维度不同

    L = || F_f^T F_d ||_F^2 / (d_f * d_d)

    推荐:
        - 先 center
        - 再 normalize

    Args:
        f_forgery: [B, d_f]
        f_domain:  [B, d_d]
        normalize: 是否按特征维做 L2 normalize
        center:    是否先按 batch 维去均值
    """
    # [B, d_f], [B, d_d]
    if center:
        f_forgery = f_forgery - f_forgery.mean(dim=0, keepdim=True)
        f_domain = f_domain - f_domain.mean(dim=0, keepdim=True)

    if normalize:
        f_forgery = F.normalize(f_forgery, dim=0)
        f_domain = F.normalize(f_domain, dim=0)

    # [d_f, B] @ [B, d_d] -> [d_f, d_d]
    corr = torch.matmul(f_forgery.transpose(0, 1), f_domain)

    d_f = corr.shape[0]
    d_d = corr.shape[1]

    loss = torch.sum(corr ** 2) / (d_f * d_d)
    return loss


def orthogonal_loss(
    f_forgery,
    f_domain,
    loss_type="sample_dot",
    normalize=True,
    center=True
):
    """
    统一入口：根据配置切换 orth loss

    Args:
        f_forgery: [B, d_f]
        f_domain:  [B, d_d]
        loss_type: "sample_dot" 或 "matrix_corr"
        normalize: 是否做归一化
        center:    是否做中心化（仅 matrix_corr 有意义）
    """
    if loss_type == "sample_dot":
        return sample_dot_orthogonal_loss(
            f_forgery=f_forgery,
            f_domain=f_domain,
            normalize=normalize
        )

    elif loss_type == "matrix_corr":
        return matrix_corr_orthogonal_loss(
            f_forgery=f_forgery,
            f_domain=f_domain,
            normalize=normalize,
            center=center
        )

    else:
        raise ValueError(
            f"未知 orth loss type: {loss_type}，"
            f"支持 ['sample_dot', 'matrix_corr']"
        )