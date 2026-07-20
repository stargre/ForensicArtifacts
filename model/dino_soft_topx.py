import torch
import torch.nn as nn

from model.dino_wrapper import DinoV2Wrapper


class ForensicDinoSoftTopX(nn.Module):
    """
    stochastic shortcut suppression with configurable position

    支持位置:
      1) none
      2) after_pool
      3) before_pool
      4) before_block

    新增:
      - model.use_reg_token: 是否把 reg_mean 拼进 pooled feature 参与训练
      - 只有当 backbone 真的是 with-registers 时，use_reg_token 才会生效
    """

    def __init__(self, config):
        super().__init__()
        model_cfg = config["model"]
        routing_cfg = config.get("routing", {})

        self.backbone = DinoV2Wrapper(
            repo_path=model_cfg.get("repo_path", "/mnt/data3/zhiyu/dino_clip/dinov2_repo"),
            model_name=model_cfg.get("backbone_name", "dinov2_vitb14"),
            pretrained=model_cfg.get("pretrained", True),
            freeze_backbone=model_cfg.get("freeze_backbone", True),
            unfreeze_last_n_blocks=model_cfg.get("unfreeze_last_n_blocks", 0),
            unfreeze_norm=model_cfg.get("unfreeze_norm", True),
            verbose=True,
        )

        self.with_registers = bool(getattr(self.backbone, "with_registers", False))
        self.num_register_tokens = int(getattr(self.backbone, "num_register_tokens", 0))

        self.pooling_type = model_cfg.get("pooling_type", "cls_patch_mean")
        assert self.pooling_type in ["cls", "patch_mean", "cls_patch_mean"]

        # ===== 新增：是否把 reg token 用进训练 =====
        self.use_reg_token = bool(model_cfg.get("use_reg_token", False))
        self.use_reg_token_effective = (
            self.use_reg_token and self.with_registers and self.num_register_tokens > 0
        )

        if self.use_reg_token and not self.use_reg_token_effective:
            print("[SoftTopX Warning] use_reg_token=True，但当前 backbone 没有可用的 register tokens，"
                  "将自动退化为不使用 reg token。")

        self.embed_dim = self.backbone.embed_dim
        self.num_pool_parts = self._get_num_pool_parts()
        self.feat_dim = self.embed_dim * self.num_pool_parts

        hidden_dim = model_cfg.get("hidden_dim", 512)
        dropout = model_cfg.get("dropout", 0.1)

        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        # for logging / analysis only
        self.register_buffer("shortcut_mask", torch.zeros(self.embed_dim))
        self.register_buffer("core_mask", torch.ones(self.embed_dim))

        # actual stochastic sampling probs
        self.register_buffer("drop_probs", torch.ones(self.embed_dim) / self.embed_dim)

        # stochastic config
        self.stochastic_enabled = bool(routing_cfg.get("stochastic_enabled", True))
        self.drop_ratio = float(routing_cfg.get("drop_ratio", 0.03))
        self.drop_beta = float(routing_cfg.get("drop_beta", 0.3))
        self.drop_active_p = float(routing_cfg.get("drop_active_p", 0.5))
        self.dual_view_train = bool(routing_cfg.get("dual_view_train", True))

        # configurable suppress position
        suppress_position = str(routing_cfg.get("suppress_position", "before_pool")).strip().lower()
        if suppress_position == "before_classifier":
            suppress_position = "after_pool"

        valid_positions = ["none", "after_pool", "before_pool", "before_block"]
        if suppress_position not in valid_positions:
            raise ValueError(f"Unknown suppress_position={suppress_position}, valid={valid_positions}")

        self.suppress_position = suppress_position

        num_blocks = self.backbone.get_num_blocks() if hasattr(self.backbone, "get_num_blocks") else 12
        suppress_block_index = int(routing_cfg.get("suppress_block_index", num_blocks - 1))

        if self.suppress_position == "before_block":
            if suppress_block_index < 0:
                suppress_block_index = num_blocks - 1
            if suppress_block_index > num_blocks:
                raise ValueError(
                    f"suppress_block_index={suppress_block_index} 超出范围，当前 num_blocks={num_blocks}"
                )

        self.suppress_block_index = suppress_block_index

        if getattr(self.backbone, "verbose", False):
            print(f"[SoftTopX] with_registers = {self.with_registers}")
            print(f"[SoftTopX] num_register_tokens = {self.num_register_tokens}")
            print(f"[SoftTopX] use_reg_token = {self.use_reg_token}")
            print(f"[SoftTopX] use_reg_token_effective = {self.use_reg_token_effective}")
            print(f"[SoftTopX] pooling_type = {self.pooling_type}")
            print(f"[SoftTopX] feat_dim = {self.feat_dim}")
            print(f"[SoftTopX] suppress_position = {self.suppress_position}")
            if self.suppress_position == "before_block":
                print(f"[SoftTopX] suppress_block_index = {self.suppress_block_index}")

    # =========================================================
    # pooling helpers
    # =========================================================
    def _get_num_pool_parts(self):
        if self.pooling_type in ["cls", "patch_mean"]:
            n = 1
        elif self.pooling_type == "cls_patch_mean":
            n = 2
        else:
            raise ValueError(f"Unknown pooling_type: {self.pooling_type}")

        if self.use_reg_token_effective:
            n += 1
        return n

    def _get_reg_mean(self, reg_tokens):
        if (not self.use_reg_token_effective) or (reg_tokens is None):
            return None
        if reg_tokens.ndim != 3 or reg_tokens.shape[1] <= 0:
            return None
        return reg_tokens.mean(dim=1)

    def pool_features(self, cls_token, patch_tokens, reg_tokens=None):
        """
        约定:
          - pooling_type='cls'             -> [cls] (+ reg_mean if enabled)
          - pooling_type='patch_mean'      -> [patch_mean] (+ reg_mean if enabled)
          - pooling_type='cls_patch_mean'  -> [cls, patch_mean] (+ reg_mean if enabled)

        这里 reg_mean 总是追加在最后。
        """
        parts = []

        if self.pooling_type == "cls":
            parts.append(cls_token)

        elif self.pooling_type == "patch_mean":
            parts.append(patch_tokens.mean(dim=1))

        elif self.pooling_type == "cls_patch_mean":
            parts.append(cls_token)
            parts.append(patch_tokens.mean(dim=1))

        else:
            raise ValueError(f"Unknown pooling_type: {self.pooling_type}")

        reg_mean = self._get_reg_mean(reg_tokens)
        if reg_mean is not None:
            parts.append(reg_mean)

        if len(parts) == 1:
            return parts[0]
        return torch.cat(parts, dim=-1)

    # =========================================================
    # buffers
    # =========================================================
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

    @torch.no_grad()
    def set_drop_probs(self, drop_probs):
        drop_probs = drop_probs.float().view(-1)
        if drop_probs.numel() != self.embed_dim:
            raise ValueError(
                f"drop_probs dim={drop_probs.numel()} != embed_dim={self.embed_dim}"
            )

        drop_probs = torch.clamp(drop_probs, min=0.0)
        s = float(drop_probs.sum().item())
        if s <= 0:
            drop_probs = torch.ones_like(drop_probs) / drop_probs.numel()
        else:
            drop_probs = drop_probs / drop_probs.sum()

        self.drop_probs.copy_(drop_probs)

    def get_mask_mean(self):
        return 1.0

    # =========================================================
    # routing feature extraction
    # =========================================================
    def extract_pooled_features(self, images):
        """
        为兼容你当前主程序，保持旧接口:
            return pooled_feat, cls_token, patch_tokens
        """
        cls_token, patch_tokens, reg_tokens = self.backbone(images)
        pooled_feat = self.pool_features(cls_token, patch_tokens, reg_tokens=reg_tokens)
        return pooled_feat, cls_token, patch_tokens

    def extract_pooled_features_with_reg(self, images):
        """
        分析用新接口:
            return pooled_feat, cls_token, patch_tokens, reg_tokens
        """
        cls_token, patch_tokens, reg_tokens = self.backbone(images)
        pooled_feat = self.pool_features(cls_token, patch_tokens, reg_tokens=reg_tokens)
        return pooled_feat, cls_token, patch_tokens, reg_tokens

    # =========================================================
    # stochastic helpers
    # =========================================================
    def _sample_drop_mask(self, batch_size, device, dtype):
        C = self.embed_dim
        k = int(round(C * self.drop_ratio))
        k = max(0, min(k, C))

        if k <= 0:
            return torch.zeros(batch_size, C, device=device, dtype=dtype)

        probs = torch.clamp(self.drop_probs, min=0.0)
        probs_sum = probs.sum()
        if float(probs_sum.item()) <= 0:
            probs = torch.ones_like(probs) / probs.numel()
        else:
            probs = probs / probs_sum

        probs_2d = probs.unsqueeze(0).expand(batch_size, -1)
        sampled_idx = torch.multinomial(probs_2d, num_samples=k, replacement=False)

        drop_mask = torch.zeros(batch_size, C, device=device, dtype=dtype)
        drop_mask.scatter_(1, sampled_idx, 1.0)
        return drop_mask

    def _build_channel_scale(self, batch_size, device, dtype, apply_stochastic=True):
        if (
            (not self.training)
            or (not self.stochastic_enabled)
            or (not apply_stochastic)
            or (self.suppress_position == "none")
        ):
            return torch.ones(batch_size, self.embed_dim, device=device, dtype=dtype)

        rand_v = torch.rand(1, device=device).item()
        if rand_v < self.drop_active_p:
            drop_mask = self._sample_drop_mask(batch_size, device, dtype)
            scale = 1.0 - self.drop_beta * drop_mask
        else:
            scale = torch.ones(batch_size, self.embed_dim, device=device, dtype=dtype)

        return scale

    def _build_pooled_scale_from_channel_scale(self, scale):
        parts = []

        if self.pooling_type == "cls":
            parts.append(scale)
        elif self.pooling_type == "patch_mean":
            parts.append(scale)
        elif self.pooling_type == "cls_patch_mean":
            parts.append(scale)  # cls
            parts.append(scale)  # patch_mean
        else:
            raise ValueError(f"Unknown pooling_type: {self.pooling_type}")

        if self.use_reg_token_effective:
            parts.append(scale)  # reg_mean 追加在最后

        if len(parts) == 1:
            return parts[0]
        return torch.cat(parts, dim=1)

    # =========================================================
    # view forward: final features path
    # =========================================================
    def _forward_view_from_final_features(
        self,
        cls_token,
        patch_tokens,
        pooled_feat,
        reg_tokens=None,
        apply_stochastic=True
    ):
        B, C = cls_token.shape
        device = cls_token.device
        dtype = cls_token.dtype

        # none
        if self.suppress_position == "none":
            scale = torch.ones(B, C, device=device, dtype=dtype)
            pooled_scale = self._build_pooled_scale_from_channel_scale(scale)
            core_feat = pooled_feat
            shortcut_feat = None
            cls_logits = self.classifier(core_feat)
            return {
                "cls_logits": cls_logits,
                "core_feat": core_feat,
                "shortcut_feat": shortcut_feat,
                "mask": pooled_scale,
                "pooled_feat": pooled_feat,
                "reg_tokens": reg_tokens,
            }

        # after_pool
        if self.suppress_position == "after_pool":
            scale = self._build_channel_scale(B, device, dtype, apply_stochastic=apply_stochastic)
            pooled_scale = self._build_pooled_scale_from_channel_scale(scale)

            core_feat = pooled_feat * pooled_scale
            shortcut_feat = pooled_feat * (1.0 - pooled_scale)
            cls_logits = self.classifier(core_feat)

            return {
                "cls_logits": cls_logits,
                "core_feat": core_feat,
                "shortcut_feat": shortcut_feat,
                "mask": pooled_scale,
                "pooled_feat": pooled_feat,
                "reg_tokens": reg_tokens,
            }

        # before_pool
        if self.suppress_position == "before_pool":
            scale = self._build_channel_scale(B, device, dtype, apply_stochastic=apply_stochastic)

            cls_core = cls_token * scale
            patch_core = patch_tokens * scale.unsqueeze(1)

            shortcut_scale = 1.0 - scale
            cls_short = cls_token * shortcut_scale
            patch_short = patch_tokens * shortcut_scale.unsqueeze(1)

            if reg_tokens is not None:
                reg_core = reg_tokens * scale.unsqueeze(1)
                reg_short = reg_tokens * shortcut_scale.unsqueeze(1)
            else:
                reg_core = None
                reg_short = None

            core_feat = self.pool_features(cls_core, patch_core, reg_tokens=reg_core)
            shortcut_feat = self.pool_features(cls_short, patch_short, reg_tokens=reg_short)

            cls_logits = self.classifier(core_feat)
            pooled_scale = self._build_pooled_scale_from_channel_scale(scale)

            return {
                "cls_logits": cls_logits,
                "core_feat": core_feat,
                "shortcut_feat": shortcut_feat,
                "mask": pooled_scale,
                "pooled_feat": pooled_feat,
                "reg_tokens": reg_core,
            }

        raise ValueError(f"_forward_view_from_final_features 不支持 suppress_position={self.suppress_position}")

    # =========================================================
    # view forward: before_block path
    # =========================================================
    def _forward_view_from_block_input(self, tokens_before_block, block_idx, apply_stochastic=True):
        B, T, C = tokens_before_block.shape
        device = tokens_before_block.device
        dtype = tokens_before_block.dtype

        scale = self._build_channel_scale(B, device, dtype, apply_stochastic=apply_stochastic)

        # 所有 token 一起按 channel 做 suppress
        tokens_core = tokens_before_block * scale.unsqueeze(1)

        cls_core, patch_core, reg_core = self.backbone.forward_from_block_input(tokens_core, block_idx)
        core_feat = self.pool_features(cls_core, patch_core, reg_tokens=reg_core)
        cls_logits = self.classifier(core_feat)

        pooled_scale = self._build_pooled_scale_from_channel_scale(scale)

        return {
            "cls_logits": cls_logits,
            "core_feat": core_feat,
            "shortcut_feat": None,
            "mask": pooled_scale,
            "pooled_feat": None,
            "reg_tokens": reg_core,
        }

    # =========================================================
    # main forward
    # =========================================================
    def forward(self, images, grl_lambda=0.0, dual_view=False):
        # -----------------------------------------------------
        # before_block
        # -----------------------------------------------------
        if self.suppress_position == "before_block":
            block_idx = self.suppress_block_index
            tokens_before_block = self.backbone.forward_to_block_input(images, block_idx)

            if self.training and dual_view and self.dual_view_train:
                out1 = self._forward_view_from_block_input(
                    tokens_before_block, block_idx, apply_stochastic=True
                )
                out2 = self._forward_view_from_block_input(
                    tokens_before_block, block_idx, apply_stochastic=True
                )

                cls_logits1 = out1["cls_logits"]
                cls_logits2 = out2["cls_logits"]

                return {
                    "cls_logits1": cls_logits1,
                    "cls_logits2": cls_logits2,
                    "cls_logits": 0.5 * (cls_logits1 + cls_logits2),

                    "pooled_feat": None,

                    "core_feat1": out1["core_feat"],
                    "core_feat2": out2["core_feat"],
                    "shortcut_feat1": out1["shortcut_feat"],
                    "shortcut_feat2": out2["shortcut_feat"],

                    "mask1": out1["mask"],
                    "mask2": out2["mask"],
                    "mask": 0.5 * (out1["mask"] + out2["mask"]),

                    "cls_token": None,
                    "patch_tokens": None,

                    "reg_tokens1": out1.get("reg_tokens", None),
                    "reg_tokens2": out2.get("reg_tokens", None),
                    "reg_tokens": None,
                }

            out = self._forward_view_from_block_input(
                tokens_before_block, block_idx, apply_stochastic=self.training
            )

            return {
                "cls_logits": out["cls_logits"],
                "pooled_feat": out["pooled_feat"],
                "core_feat": out["core_feat"],
                "shortcut_feat": out["shortcut_feat"],
                "mask": out["mask"],
                "cls_token": None,
                "patch_tokens": None,
                "reg_tokens": out.get("reg_tokens", None),
            }

        # -----------------------------------------------------
        # none / after_pool / before_pool
        # -----------------------------------------------------
        cls_token, patch_tokens, reg_tokens = self.backbone(images)
        pooled_feat = self.pool_features(cls_token, patch_tokens, reg_tokens=reg_tokens)

        if self.training and dual_view and self.dual_view_train:
            out1 = self._forward_view_from_final_features(
                cls_token, patch_tokens, pooled_feat, reg_tokens=reg_tokens, apply_stochastic=True
            )
            out2 = self._forward_view_from_final_features(
                cls_token, patch_tokens, pooled_feat, reg_tokens=reg_tokens, apply_stochastic=True
            )

            cls_logits1 = out1["cls_logits"]
            cls_logits2 = out2["cls_logits"]

            return {
                "cls_logits1": cls_logits1,
                "cls_logits2": cls_logits2,
                "cls_logits": 0.5 * (cls_logits1 + cls_logits2),

                "pooled_feat": pooled_feat,

                "core_feat1": out1["core_feat"],
                "core_feat2": out2["core_feat"],
                "shortcut_feat1": out1["shortcut_feat"],
                "shortcut_feat2": out2["shortcut_feat"],

                "mask1": out1["mask"],
                "mask2": out2["mask"],
                "mask": 0.5 * (out1["mask"] + out2["mask"]),

                "cls_token": cls_token,
                "patch_tokens": patch_tokens,

                "reg_tokens": reg_tokens,
                "reg_tokens1": out1.get("reg_tokens", None),
                "reg_tokens2": out2.get("reg_tokens", None),
            }

        out = self._forward_view_from_final_features(
            cls_token, patch_tokens, pooled_feat, reg_tokens=reg_tokens, apply_stochastic=self.training
        )

        return {
            "cls_logits": out["cls_logits"],
            "pooled_feat": pooled_feat,
            "core_feat": out["core_feat"],
            "shortcut_feat": out["shortcut_feat"],
            "mask": out["mask"],
            "cls_token": cls_token,
            "patch_tokens": patch_tokens,
            "reg_tokens": out.get("reg_tokens", reg_tokens),
        }