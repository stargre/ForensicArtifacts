import os
import sys
import torch
import torch.nn as nn


class IBOTWrapper(nn.Module):
    """
    iBOT ViT wrapper.

    输出格式：
        cls_token:    [B, C]
        patch_tokens: [B, N, C]
        aux:          None
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
        verbose=True,
    ):
        super().__init__()

        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        from models import vision_transformer as vits

        self.backbone = getattr(vits, model_name)(
            patch_size=patch_size,
            return_all_tokens=True,
        )

        self.embed_dim = self.backbone.embed_dim
        self.model_name = model_name

        if pretrained:
            if checkpoint_path is None:
                raise ValueError(
                    "iBOT pretrained=True requires model.backbone_checkpoint"
                )

            ckpt = torch.load(checkpoint_path, map_location="cpu")

            if checkpoint_key in ckpt:
                state_dict = ckpt[checkpoint_key]
            elif "model" in ckpt:
                state_dict = ckpt["model"]
            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt

            clean_state = {}
            for k, v in state_dict.items():
                # 兼容 teacher.backbone.xxx / module.backbone.xxx / backbone.xxx
                for prefix in [
                    "module.",
                    "teacher.",
                    "student.",
                    "backbone.",
                ]:
                    if k.startswith(prefix):
                        k = k[len(prefix):]

                # 不加载预训练投影头 / 分类头
                if k.startswith("head.") or "last_layer" in k:
                    continue

                clean_state[k] = v

            msg = self.backbone.load_state_dict(clean_state, strict=False)

            if verbose:
                print(f"[IBOTWrapper] Loaded checkpoint: {checkpoint_path}")
                print(f"[IBOTWrapper] checkpoint_key={checkpoint_key}")
                print(f"[IBOTWrapper] Missing keys: {len(msg.missing_keys)}")
                print(f"[IBOTWrapper] Unexpected keys: {len(msg.unexpected_keys)}")

        self._set_trainable(
            freeze_backbone=freeze_backbone,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
            unfreeze_norm=unfreeze_norm,
        )

        if verbose:
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            print(f"[IBOTWrapper] model={model_name}, embed_dim={self.embed_dim}")
            print(f"[IBOTWrapper] trainable={trainable:,} / total={total:,}")

    def _set_trainable(
        self,
        freeze_backbone=True,
        unfreeze_last_n_blocks=0,
        unfreeze_norm=False,
    ):
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        if unfreeze_last_n_blocks > 0:
            for block in self.backbone.blocks[-unfreeze_last_n_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True

        if unfreeze_norm:
            if hasattr(self.backbone, "norm"):
                for p in self.backbone.norm.parameters():
                    p.requires_grad = True

            if hasattr(self.backbone, "fc_norm") and self.backbone.fc_norm is not None:
                for p in self.backbone.fc_norm.parameters():
                    p.requires_grad = True

    def forward(self, images):
        tokens = self.backbone(
            images,
            return_all_tokens=True,
        )

        cls_token = tokens[:, 0]
        patch_tokens = tokens[:, 1:]

        return cls_token, patch_tokens, None