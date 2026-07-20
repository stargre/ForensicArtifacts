import torch
import torch.nn as nn


class DomainHead(nn.Module):
    """
    域分类头
    输入: shared feature [B, D]
    输出: domain logits [B, num_domains]
    """
    def __init__(self, in_dim=512, hidden_dim=256, num_domains=4, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_domains)
        )

    def forward(self, x):
        return self.head(x)