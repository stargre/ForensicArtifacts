import os
import sys
import torch
import torch.nn as nn


class MAEWrapper(nn.Module):
    """
    MAE encoder wrapper.

    输出格式保持和 DinoV2Wrapper 一致：
        cls_token:    [B, C]
        patch_tokens: [B, N, C]
        aux:          None
    """

    def __init__(
        self,
        repo_path,
        model_name="mae_vit_base_patch16",
        checkpoint_path=None,
        pretrained=True,
        freeze_backbone=True,
        unfreeze_last_n_blocks=0,
        unfreeze_norm=False,
        verbose=True,
    ):
        super().__init__()

        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        import models_mae

        self.backbone = getattr(models_mae, model_name)()

        self.embed_dim = self.backbone.cls_token.shape[-1]
        self.model_name = model_name

        if pretrained:
            if checkpoint_path is None:
                raise ValueError(
                    "MAE pretrained=True requires model.backbone_checkpoint"
                )

            ckpt = torch.load(checkpoint_path, map_location="cpu")
            state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))

            clean_state = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    k = k[len("module."):]
                clean_state[k] = v

            msg = self.backbone.load_state_dict(clean_state, strict=False)

            if verbose:
                print(f"[MAEWrapper] Loaded checkpoint: {checkpoint_path}")
                print(f"[MAEWrapper] Missing keys: {len(msg.missing_keys)}")
                print(f"[MAEWrapper] Unexpected keys: {len(msg.unexpected_keys)}")

        self._set_trainable(
            freeze_backbone=freeze_backbone,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
            unfreeze_norm=unfreeze_norm,
        )

        if verbose:
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            print(f"[MAEWrapper] model={model_name}, embed_dim={self.embed_dim}")
            print(f"[MAEWrapper] trainable={trainable:,} / total={total:,}")

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

        if unfreeze_norm and hasattr(self.backbone, "norm"):
            for p in self.backbone.norm.parameters():
                p.requires_grad = True

    def forward(self, images):
        # 关键：mask_ratio=0.0，保留全部 patch token
        latent, _, _ = self.backbone.forward_encoder(
            images,
            mask_ratio=0.0,
        )

        cls_token = latent[:, 0]
        patch_tokens = latent[:, 1:]

        return cls_token, patch_tokens, None