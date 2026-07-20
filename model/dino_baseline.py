import os
import sys
import torch
import torch.nn as nn

from model.dino_wrapper import DinoV2Wrapper
from model.classifier_head import DinoClassificationHead


# =========================================================
# Common utils
# =========================================================
def _insert_repo_path(repo_path):
    if repo_path is None:
        return

    if not os.path.isdir(repo_path):
        raise FileNotFoundError(f"repo_path 不存在: {repo_path}")

    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)


def _clean_state_dict(
    state_dict,
    strip_prefixes=(
        "module.",
        "model.",
        "backbone.",
        "encoder.",
        "teacher.",
        "student.",
    ),
    drop_keywords=(),
):
    """
    兼容不同 checkpoint 命名：
        module.backbone.xxx -> xxx
        teacher.backbone.xxx -> xxx
        student.backbone.xxx -> xxx
        model.xxx -> xxx
    """

    clean_state = {}

    for k, v in state_dict.items():
        if any(keyword in k for keyword in drop_keywords):
            continue

        changed = True
        while changed:
            changed = False
            for prefix in strip_prefixes:
                if k.startswith(prefix):
                    k = k[len(prefix):]
                    changed = True

        clean_state[k] = v

    return clean_state


def _load_checkpoint_to_backbone(
    backbone,
    checkpoint_path,
    checkpoint_key=None,
    strict=False,
    drop_keywords=(),
    verbose=True,
):
    if checkpoint_path is None:
        raise ValueError("pretrained=True 时必须提供 model.backbone_checkpoint")

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"backbone_checkpoint 不存在: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if checkpoint_key is not None and isinstance(ckpt, dict) and checkpoint_key in ckpt:
        state_dict = ckpt[checkpoint_key]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise TypeError(f"Unsupported checkpoint format: {type(ckpt)}")

    state_dict = _clean_state_dict(
        state_dict,
        drop_keywords=drop_keywords,
    )

    msg = backbone.load_state_dict(state_dict, strict=strict)

    if verbose:
        print(f"[Backbone] Loaded checkpoint: {checkpoint_path}")
        print(f"[Backbone] checkpoint_key: {checkpoint_key}")
        print(f"[Backbone] strict: {strict}")
        print(f"[Backbone] missing_keys: {len(msg.missing_keys)}")
        print(f"[Backbone] unexpected_keys: {len(msg.unexpected_keys)}")

        if len(msg.missing_keys) > 0:
            print("[Backbone] first 20 missing_keys:")
            for k in msg.missing_keys[:20]:
                print(f"  - {k}")

        if len(msg.unexpected_keys) > 0:
            print("[Backbone] first 20 unexpected_keys:")
            for k in msg.unexpected_keys[:20]:
                print(f"  - {k}")

    return msg


def _set_vit_trainable(
    backbone,
    freeze_backbone=True,
    unfreeze_last_n_blocks=0,
    unfreeze_norm=False,
    verbose=True,
    name="Backbone",
):
    """
    适用于 MAE / iBOT 这种标准 ViT：
        backbone.blocks
        backbone.norm
    """

    if freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad_(False)

    if unfreeze_last_n_blocks > 0:
        if not hasattr(backbone, "blocks"):
            raise AttributeError(
                f"{name} 没有 blocks，无法 unfreeze_last_n_blocks"
            )

        for block in backbone.blocks[-unfreeze_last_n_blocks:]:
            for p in block.parameters():
                p.requires_grad_(True)

    if unfreeze_norm:
        for n, p in backbone.named_parameters():
            if "norm" in n:
                p.requires_grad_(True)

    if verbose:
        total = sum(p.numel() for p in backbone.parameters())
        trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        print(f"[{name}] trainable params: {trainable:,} / {total:,}")


# =========================================================
# MAE backbone wrapper
# =========================================================
class MAEBackboneWrapper(nn.Module):
    """
    MAE encoder wrapper.

    输出保持和 DinoV2Wrapper 一致：
        cls_token:    [B, C]
        patch_tokens: [B, N, C]
        aux:          dict
    """

    def __init__(
        self,
        repo_path,
        model_name="mae_vit_base_patch16",
        checkpoint_path=None,
        checkpoint_key="model",
        pretrained=True,
        freeze_backbone=True,
        unfreeze_last_n_blocks=0,
        unfreeze_norm=False,
        strict_load=False,
        verbose=True,
    ):
        super().__init__()

        _insert_repo_path(repo_path)

        import models_mae

        if not hasattr(models_mae, model_name):
            raise ValueError(
                f"MAE repo 中找不到模型 {model_name}。"
                f"可用模型一般包括 mae_vit_base_patch16 / mae_vit_large_patch16 / mae_vit_huge_patch14"
            )

        self.backbone = getattr(models_mae, model_name)()
        self.model_name = model_name

        if pretrained:
            _load_checkpoint_to_backbone(
                backbone=self.backbone,
                checkpoint_path=checkpoint_path,
                checkpoint_key=checkpoint_key,
                strict=strict_load,
                drop_keywords=(),
                verbose=verbose,
            )

        # MAE 官方模型通常没有 self.embed_dim 字段，这里从 cls_token 推断
        self.embed_dim = int(self.backbone.cls_token.shape[-1])

        _set_vit_trainable(
            backbone=self.backbone,
            freeze_backbone=freeze_backbone,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
            unfreeze_norm=unfreeze_norm,
            verbose=verbose,
            name="MAE",
        )

        if verbose:
            print(f"[MAE] model_name={model_name}")
            print(f"[MAE] embed_dim={self.embed_dim}")

    def forward_encoder_no_mask(self, images):
        """
        不调用官方 forward_encoder(mask_ratio=0.0)。

        原因：
        MAE 的 random_masking 即使 mask_ratio=0.0，也会经过随机 shuffle。
        对 patch_mean 影响不大，但会引入不必要的随机性。
        这里直接手动走完整 encoder，保持 patch 顺序稳定。
        """

        x = self.backbone.patch_embed(images)

        expected_tokens = x.shape[1] + 1
        if self.backbone.pos_embed.shape[1] != expected_tokens:
            raise ValueError(
                f"MAE pos_embed 长度不匹配："
                f"pos_embed={self.backbone.pos_embed.shape[1]}, "
                f"expected={expected_tokens}. "
                f"请确认 data.image_size 与 MAE 预训练尺寸一致，通常是 224。"
            )

        x = x + self.backbone.pos_embed[:, 1:, :]

        cls_token = self.backbone.cls_token + self.backbone.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.backbone.blocks:
            x = blk(x)

        x = self.backbone.norm(x)

        return x

    def forward(self, images):
        tokens = self.forward_encoder_no_mask(images)

        cls_token = tokens[:, 0]
        patch_tokens = tokens[:, 1:]

        return cls_token, patch_tokens, {"tokens": tokens}


# =========================================================
# iBOT backbone wrapper
# =========================================================
class IBOTBackboneWrapper(nn.Module):
    """
    iBOT ViT wrapper.

    输出保持和 DinoV2Wrapper 一致：
        cls_token:    [B, C]
        patch_tokens: [B, N, C]
        aux:          dict
    """

    def __init__(
        self,
        repo_path,
        model_name="vit_base",
        patch_size=16,
        checkpoint_path=None,
        checkpoint_key="teacher",
        pretrained=True,
        freeze_backbone=True,
        unfreeze_last_n_blocks=0,
        unfreeze_norm=False,
        strict_load=False,
        verbose=True,
    ):
        super().__init__()

        _insert_repo_path(repo_path)

        import models as ibot_models

        if model_name not in ibot_models.__dict__:
            raise ValueError(
                f"iBOT repo 中找不到模型 {model_name}。"
                f"可用模型通常包括 vit_tiny / vit_small / vit_base / vit_large"
            )

        self.backbone = ibot_models.__dict__[model_name](
            patch_size=patch_size,
            return_all_tokens=True,
        )

        self.model_name = model_name
        self.patch_size = patch_size

        if hasattr(self.backbone, "embed_dim"):
            self.embed_dim = int(self.backbone.embed_dim)
        elif hasattr(self.backbone, "num_features"):
            self.embed_dim = int(self.backbone.num_features)
        else:
            raise AttributeError("无法从 iBOT backbone 推断 embed_dim")

        if pretrained:
            _load_checkpoint_to_backbone(
                backbone=self.backbone,
                checkpoint_path=checkpoint_path,
                checkpoint_key=checkpoint_key,
                strict=strict_load,
                drop_keywords=(
                    "head",
                    "last_layer",
                    "projection",
                    "prototypes",
                ),
                verbose=verbose,
            )

        _set_vit_trainable(
            backbone=self.backbone,
            freeze_backbone=freeze_backbone,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
            unfreeze_norm=unfreeze_norm,
            verbose=verbose,
            name="iBOT",
        )

        if verbose:
            print(f"[iBOT] model_name={model_name}")
            print(f"[iBOT] patch_size={patch_size}")
            print(f"[iBOT] embed_dim={self.embed_dim}")

    def forward(self, images):
        try:
            tokens = self.backbone(images, return_all_tokens=True)
        except TypeError:
            tokens = self.backbone(images)

        if isinstance(tokens, (list, tuple)):
            tokens = tokens[0]

        if tokens.dim() != 3:
            raise RuntimeError(
                f"iBOT backbone 没有返回 all tokens，得到 shape={tokens.shape}。"
                f"请确认构造时 return_all_tokens=True。"
            )

        cls_token = tokens[:, 0]
        patch_tokens = tokens[:, 1:]

        return cls_token, patch_tokens, {"tokens": tokens}


# =========================================================
# Main model
# =========================================================
class ForensicDinoBaseline(nn.Module):
    """
    为了兼容你的 train.py，类名仍然叫 ForensicDinoBaseline。

    现在支持三种 backbone:
        - dinov2
        - mae
        - ibot

    统一输出:
        logits, cls_token, patch_tokens
    """

    def __init__(self, config):
        super().__init__()

        model_cfg = config["model"]

        backbone_type = model_cfg.get("backbone_type", "dinov2").lower()
        self.backbone_type = backbone_type

        # =====================================================
        # Backbone
        # =====================================================
        if backbone_type in ["dinov2", "dino", "dino_v2"]:
            self.backbone = DinoV2Wrapper(
                repo_path=model_cfg.get(
                    "repo_path",
                    "/mnt/data3/zhiyu/dino_clip/dinov2_repo",
                ),
                model_name=model_cfg.get(
                    "backbone_name",
                    "dinov2_vitb14",
                ),
                pretrained=model_cfg.get("pretrained", True),
                freeze_backbone=model_cfg.get(
                    "freeze_backbone",
                    True,
                ),
                unfreeze_last_n_blocks=model_cfg.get(
                    "unfreeze_last_n_blocks",
                    0,
                ),
                unfreeze_norm=model_cfg.get(
                    "unfreeze_norm",
                    False,
                ),
                verbose=True,
            )

        elif backbone_type == "mae":
            self.backbone = MAEBackboneWrapper(
                repo_path=model_cfg.get(
                    "repo_path",
                    "/mnt/data3/zhiyu/mae",
                ),
                model_name=model_cfg.get(
                    "backbone_name",
                    "mae_vit_base_patch16",
                ),
                checkpoint_path=model_cfg.get(
                    "backbone_checkpoint",
                    None,
                ),
                checkpoint_key=model_cfg.get(
                    "checkpoint_key",
                    "model",
                ),
                pretrained=model_cfg.get(
                    "pretrained",
                    True,
                ),
                freeze_backbone=model_cfg.get(
                    "freeze_backbone",
                    True,
                ),
                unfreeze_last_n_blocks=model_cfg.get(
                    "unfreeze_last_n_blocks",
                    0,
                ),
                unfreeze_norm=model_cfg.get(
                    "unfreeze_norm",
                    False,
                ),
                strict_load=model_cfg.get(
                    "strict_load",
                    False,
                ),
                verbose=True,
            )

        elif backbone_type == "ibot":
            self.backbone = IBOTBackboneWrapper(
                repo_path=model_cfg.get(
                    "repo_path",
                    "/mnt/data3/zhiyu/ibot",
                ),
                model_name=model_cfg.get(
                    "backbone_name",
                    "vit_base",
                ),
                patch_size=model_cfg.get(
                    "patch_size",
                    16,
                ),
                checkpoint_path=model_cfg.get(
                    "backbone_checkpoint",
                    None,
                ),
                checkpoint_key=model_cfg.get(
                    "checkpoint_key",
                    "teacher",
                ),
                pretrained=model_cfg.get(
                    "pretrained",
                    True,
                ),
                freeze_backbone=model_cfg.get(
                    "freeze_backbone",
                    True,
                ),
                unfreeze_last_n_blocks=model_cfg.get(
                    "unfreeze_last_n_blocks",
                    0,
                ),
                unfreeze_norm=model_cfg.get(
                    "unfreeze_norm",
                    False,
                ),
                strict_load=model_cfg.get(
                    "strict_load",
                    False,
                ),
                verbose=True,
            )

        else:
            raise ValueError(
                f"Unsupported backbone_type={backbone_type}. "
                f"Expected one of: dinov2, mae, ibot"
            )

        embed_dim = self.backbone.embed_dim

        # =====================================================
        # Classifier
        # =====================================================
        self.classifier = DinoClassificationHead(
            embed_dim=embed_dim,
            pooling_type=model_cfg.get(
                "pooling_type",
                "cls_patch_mean",
            ),
            hidden_dim=model_cfg.get(
                "hidden_dim",
                512,
            ),
            dropout=model_cfg.get(
                "dropout",
                0.2,
            ),
            enable_attention_pooling=model_cfg.get(
                "enable_attention_pooling",
                False,
            ),
            attention_hidden_dim=model_cfg.get(
                "attention_hidden_dim",
                256,
            ),
        )

        self.feature_dim = embed_dim
        self.embed_dim = embed_dim

        print("=" * 80)
        print(f"[ForensicDinoBaseline] backbone_type = {self.backbone_type}")
        print(f"[ForensicDinoBaseline] embed_dim      = {self.embed_dim}")
        print(f"[ForensicDinoBaseline] pooling_type   = {model_cfg.get('pooling_type', 'cls_patch_mean')}")
        print("=" * 80)

    def forward_features(self, images):
        cls_token, patch_tokens, _ = self.backbone(images)
        return cls_token, patch_tokens

    def forward(self, images):
        cls_token, patch_tokens = self.forward_features(images)

        logits = self.classifier(
            cls_token,
            patch_tokens,
        )

        return logits, cls_token, patch_tokens