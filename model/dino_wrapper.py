import torch
import torch.nn as nn


class DinoV2Wrapper(nn.Module):
    """
    DINOv2 backbone wrapper

    支持：
        - no-register 版本，如: dinov2_vitb14
        - with-registers 版本，如: dinov2_vitb14_reg

    forward 统一返回：
        cls_token:    [B, C]
        patch_tokens: [B, N, C]
        reg_tokens:   [B, R, C] or None

    说明：
        - 对 no-reg 模型，reg_tokens = None
        - 对 with-reg 模型，reg_tokens.shape 通常为 [B, 4, C]

    同时支持：
        1) 全冻结
        2) 解冻最后 n 个 transformer blocks
        3) 全部解冻（可选）

    并提供：
        - forward_to_block_input(x, block_idx)
        - forward_from_block_input(tokens, block_idx)
      以支持“在某个 block 之前做 suppress”
    """

    def __init__(
        self,
        repo_path="/mnt/data3/zhiyu/dino_clip/dinov2_repo",
        model_name="dinov2_vitb14",
        pretrained=True,
        freeze_backbone=True,
        unfreeze_last_n_blocks=0,
        unfreeze_norm=True,
        verbose=True,
    ):
        super().__init__()
        self.repo_path = repo_path
        self.model_name = model_name
        self.verbose = verbose

        # -----------------------------------------------------
        # load backbone
        # -----------------------------------------------------
        self.backbone = torch.hub.load(
            repo_path,
            model_name,
            source="local",
            pretrained=pretrained
        )

        # -----------------------------------------------------
        # infer embed_dim from model_name
        # -----------------------------------------------------
        if "vits14" in model_name:
            self.embed_dim = 384
        elif "vitb14" in model_name:
            self.embed_dim = 768
        elif "vitl14" in model_name:
            self.embed_dim = 1024
        elif "vitg14" in model_name:
            self.embed_dim = 1536
        else:
            raise ValueError(f"无法根据 model_name 推断 embed_dim: {model_name}")

        # -----------------------------------------------------
        # register info
        # -----------------------------------------------------
        # 配置名推断
        expected_with_registers = str(model_name).endswith("_reg")

        # 实际模型属性
        self.num_register_tokens = int(getattr(self.backbone, "num_register_tokens", 0))
        actual_with_registers = self.num_register_tokens > 0

        # 以实际模型为准，避免切 token 时出错
        self.with_registers = actual_with_registers

        if expected_with_registers and not actual_with_registers:
            print(
                f"[DINO Warning] model_name={model_name} 看起来是 with-registers，"
                f"但加载后的模型 num_register_tokens={self.num_register_tokens}。"
                f"当前将按 no-register 模型处理。"
            )

        if (not expected_with_registers) and actual_with_registers:
            print(
                f"[DINO Warning] model_name={model_name} 看起来是 no-register，"
                f"但加载后的模型 num_register_tokens={self.num_register_tokens}。"
                f"当前将按 with-registers 模型处理。"
            )

        if self.verbose:
            print(f"[DINO] model_name = {self.model_name}")
            print(f"[DINO] with_registers = {self.with_registers}")
            print(f"[DINO] num_register_tokens = {self.num_register_tokens}")

        # -----------------------------------------------------
        # trainable config
        # -----------------------------------------------------
        self._configure_trainable_layers(
            freeze_backbone=freeze_backbone,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
            unfreeze_norm=unfreeze_norm
        )

    # =========================================================
    # trainable control
    # =========================================================
    def _configure_trainable_layers(
        self,
        freeze_backbone=True,
        unfreeze_last_n_blocks=0,
        unfreeze_norm=True
    ):
        # 先全部冻结
        for p in self.backbone.parameters():
            p.requires_grad = False

        if not freeze_backbone:
            # 全部解冻
            for p in self.backbone.parameters():
                p.requires_grad = True

            if self.verbose:
                print(f"[DINO] Backbone 全部解冻")
            return

        # freeze_backbone=True 时，允许只解冻最后 n 层
        if unfreeze_last_n_blocks > 0:
            if not hasattr(self.backbone, "blocks"):
                raise AttributeError(
                    "[DINO] backbone 没有 blocks 属性，无法执行 unfreeze_last_n_blocks"
                )

            total_blocks = len(self.backbone.blocks)
            start_idx = max(0, total_blocks - unfreeze_last_n_blocks)

            for i in range(start_idx, total_blocks):
                for p in self.backbone.blocks[i].parameters():
                    p.requires_grad = True

            if unfreeze_norm and hasattr(self.backbone, "norm"):
                for p in self.backbone.norm.parameters():
                    p.requires_grad = True

            if self.verbose:
                print(f"[DINO] Backbone 默认冻结，仅解冻最后 {unfreeze_last_n_blocks} 个 blocks")
                if unfreeze_norm and hasattr(self.backbone, "norm"):
                    print(f"[DINO] 同时解冻最终 norm")
        else:
            if self.verbose:
                print(f"[DINO] Backbone 全冻结")

    def freeze(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def unfreeze_last_n_blocks(self, n=1, unfreeze_norm=True):
        self.freeze()

        if not hasattr(self.backbone, "blocks"):
            raise AttributeError(
                "[DINO] backbone 没有 blocks 属性，无法执行 unfreeze_last_n_blocks"
            )

        total_blocks = len(self.backbone.blocks)
        start_idx = max(0, total_blocks - n)

        for i in range(start_idx, total_blocks):
            for p in self.backbone.blocks[i].parameters():
                p.requires_grad = True

        if unfreeze_norm and hasattr(self.backbone, "norm"):
            for p in self.backbone.norm.parameters():
                p.requires_grad = True

        if self.verbose:
            print(f"[DINO] 手动解冻最后 {n} 个 blocks")
            if unfreeze_norm and hasattr(self.backbone, "norm"):
                print(f"[DINO] 同时解冻最终 norm")

    def is_backbone_trainable(self):
        return any(p.requires_grad for p in self.backbone.parameters())

    def get_trainable_param_count(self):
        total = sum(p.numel() for p in self.backbone.parameters())
        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        return total, trainable

    def print_trainable_status(self):
        total, trainable = self.get_trainable_param_count()
        ratio = 100.0 * trainable / total if total > 0 else 0.0

        print(f"[DINO] backbone 参数量: total={total:,}, trainable={trainable:,} ({ratio:.2f}%)")
        print(f"[DINO] with_registers={self.with_registers}, num_register_tokens={self.num_register_tokens}")

        if hasattr(self.backbone, "blocks"):
            print("[DINO] block trainable status:")
            for i, blk in enumerate(self.backbone.blocks):
                blk_trainable = any(p.requires_grad for p in blk.parameters())
                print(f"  block[{i:02d}]: {'trainable' if blk_trainable else 'frozen'}")

        if hasattr(self.backbone, "norm"):
            norm_trainable = any(p.requires_grad for p in self.backbone.norm.parameters())
            print(f"  norm: {'trainable' if norm_trainable else 'frozen'}")

    def get_backbone_trainable_params(self):
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def get_backbone_frozen_params(self):
        return [p for p in self.backbone.parameters() if not p.requires_grad]

    def get_last_n_block_params(self, n=1, include_norm=True):
        params = []

        if hasattr(self.backbone, "blocks"):
            total_blocks = len(self.backbone.blocks)
            start_idx = max(0, total_blocks - n)
            for i in range(start_idx, total_blocks):
                params.extend(list(self.backbone.blocks[i].parameters()))

        if include_norm and hasattr(self.backbone, "norm"):
            params.extend(list(self.backbone.norm.parameters()))

        return params

    # =========================================================
    # helpers for block-level suppress
    # =========================================================
    def get_num_blocks(self):
        if not hasattr(self.backbone, "blocks"):
            raise AttributeError("[DINO] backbone 没有 blocks 属性")
        return len(self.backbone.blocks)

    def _prepare_tokens(self, x):
        """
        兼容 dinov2 的 token 准备流程
        """
        if hasattr(self.backbone, "prepare_tokens_with_masks"):
            return self.backbone.prepare_tokens_with_masks(x, masks=None)
        elif hasattr(self.backbone, "prepare_tokens"):
            return self.backbone.prepare_tokens(x)
        else:
            raise AttributeError(
                "[DINO] backbone 既没有 prepare_tokens_with_masks，也没有 prepare_tokens，"
                "无法支持 before_block 抑制"
            )

    def _split_normed_tokens(self, x_norm):
        """
        x_norm: [B, T, C]

        返回:
            cls_token: [B, C]
            patch_tokens: [B, N, C]
            reg_tokens: [B, R, C] or None
        """
        cls_token = x_norm[:, 0]  # [B, C]

        if self.with_registers and self.num_register_tokens > 0:
            reg_tokens = x_norm[:, 1:1 + self.num_register_tokens]   # [B, R, C]
            patch_tokens = x_norm[:, 1 + self.num_register_tokens:]  # [B, N, C]
        else:
            reg_tokens = None
            patch_tokens = x_norm[:, 1:]                             # [B, N, C]

        return cls_token, patch_tokens, reg_tokens

    def _extract_reg_tokens_from_feats(self, feats):
        """
        从 forward_features 的输出中提取 reg tokens
        no-reg 时返回 None
        """
        if not self.with_registers or self.num_register_tokens <= 0:
            return None

        # 常见官方 key
        if "x_norm_regtokens" in feats:
            reg_tokens = feats["x_norm_regtokens"]
            if reg_tokens is not None and reg_tokens.ndim == 3 and reg_tokens.shape[1] > 0:
                return reg_tokens

        # 有些实现可能返回完整 norm token
        if "x_norm" in feats:
            x_norm = feats["x_norm"]
            if x_norm.ndim == 3 and x_norm.shape[1] >= 1 + self.num_register_tokens:
                return x_norm[:, 1:1 + self.num_register_tokens]

        # 有些实现可能返回 prenorm
        if "x_prenorm" in feats and hasattr(self.backbone, "norm"):
            x_norm = self.backbone.norm(feats["x_prenorm"])
            if x_norm.ndim == 3 and x_norm.shape[1] >= 1 + self.num_register_tokens:
                return x_norm[:, 1:1 + self.num_register_tokens]

        return None

    # =========================================================
    # forward to / from block input
    # =========================================================
    def forward_to_block_input(self, x, block_idx):
        """
        前向到某个 block 的输入位置（即 block_idx 之前）
        返回 token 序列，形状 [B, T, C]

        例如:
            block_idx=0  -> 刚 prepare_tokens 后
            block_idx=11 -> 第11个 block 之前
            block_idx=12 -> 所有 blocks 后、norm 前（如果总共12层）
        """
        if not hasattr(self.backbone, "blocks"):
            raise AttributeError("[DINO] backbone 没有 blocks 属性，无法支持 before_block 抑制")

        total_blocks = len(self.backbone.blocks)
        if block_idx < 0 or block_idx > total_blocks:
            raise ValueError(f"block_idx={block_idx} 超出范围 [0, {total_blocks}]")

        if self.is_backbone_trainable():
            tokens = self._prepare_tokens(x)
            for i in range(block_idx):
                tokens = self.backbone.blocks[i](tokens)
        else:
            with torch.no_grad():
                tokens = self._prepare_tokens(x)
                for i in range(block_idx):
                    tokens = self.backbone.blocks[i](tokens)

        return tokens

    def forward_from_block_input(self, tokens, block_idx):
        """
        从某个 block 的输入位置继续前向到最终输出

        输入:
            tokens: [B, T, C]
            block_idx: 从第 block_idx 个 block 开始继续

        输出:
            cls_token: [B, C]
            patch_tokens: [B, N, C]
            reg_tokens: [B, R, C] or None
        """
        if not hasattr(self.backbone, "blocks"):
            raise AttributeError("[DINO] backbone 没有 blocks 属性，无法支持 before_block 抑制")

        total_blocks = len(self.backbone.blocks)
        if block_idx < 0 or block_idx > total_blocks:
            raise ValueError(f"block_idx={block_idx} 超出范围 [0, {total_blocks}]")

        if not hasattr(self.backbone, "norm"):
            raise AttributeError("[DINO] backbone 没有 norm 属性，无法完成 forward_from_block_input")

        if self.is_backbone_trainable():
            x = tokens
            for i in range(block_idx, total_blocks):
                x = self.backbone.blocks[i](x)
            x_norm = self.backbone.norm(x)
        else:
            with torch.no_grad():
                x = tokens
                for i in range(block_idx, total_blocks):
                    x = self.backbone.blocks[i](x)
                x_norm = self.backbone.norm(x)

        cls_token, patch_tokens, reg_tokens = self._split_normed_tokens(x_norm)
        return cls_token, patch_tokens, reg_tokens

    # =========================================================
    # main forward
    # =========================================================
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W]

        Returns:
            cls_token:    [B, C]
            patch_tokens: [B, N, C]
            reg_tokens:   [B, R, C] or None
        """
        if self.is_backbone_trainable():
            feats = self.backbone.forward_features(x)
        else:
            with torch.no_grad():
                feats = self.backbone.forward_features(x)

        if "x_norm_clstoken" not in feats or "x_norm_patchtokens" not in feats:
            raise KeyError(
                "forward_features 输出中未找到 x_norm_clstoken / x_norm_patchtokens，"
                "请检查本地 dinov2_repo 的模型实现。"
            )

        cls_token = feats["x_norm_clstoken"]       # [B, C]
        patch_tokens = feats["x_norm_patchtokens"] # [B, N, C]
        reg_tokens = self._extract_reg_tokens_from_feats(feats)

        return cls_token, patch_tokens, reg_tokens