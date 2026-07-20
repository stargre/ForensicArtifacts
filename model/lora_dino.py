import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    y = W0 x + scale * B A x

    W0: 原始 Linear，冻结
    A/B: LoRA 可训练参数
    """

    def __init__(self, base_layer: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()

        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"LoRALinear 只能包装 nn.Linear，但得到 {type(base_layer)}")

        if r <= 0:
            raise ValueError(f"LoRA rank 必须 > 0，但得到 r={r}")

        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        self.lora_A = nn.Parameter(
            torch.empty(r, in_features, device=device, dtype=dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, r, device=device, dtype=dtype)
        )

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # 冻结原始 Linear
        for p in self.base_layer.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        base_out = self.base_layer(x)

        lora_hidden = F.linear(self.dropout(x), self.lora_A)
        lora_out = F.linear(lora_hidden, self.lora_B)

        return base_out + self.scaling * lora_out


def _get_parent_module(root: nn.Module, module_name: str):
    parts = module_name.split(".")
    parent = root

    for p in parts[:-1]:
        if p.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)

    return parent, parts[-1]


def _set_child_module(parent: nn.Module, child_name: str, new_module: nn.Module):
    if child_name.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(child_name)] = new_module
    else:
        setattr(parent, child_name, new_module)


def inject_lora_to_linear_layers(
    root: nn.Module,
    target_keywords,
    r=8,
    alpha=16,
    dropout=0.0,
    verbose=True,
):
    if isinstance(target_keywords, str):
        target_keywords = [target_keywords]

    matched = []

    for name, module in root.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        if any(key in name for key in target_keywords):
            matched.append((name, module))

    for name, module in matched:
        parent, child_name = _get_parent_module(root, name)
        lora_module = LoRALinear(
            base_layer=module,
            r=r,
            alpha=alpha,
            dropout=dropout,
        )
        _set_child_module(parent, child_name, lora_module)

    if verbose:
        print(f"[LoRA] matched Linear layers: {len(matched)}")
        for name, _ in matched:
            print(f"  + {name}")

    return len(matched)


def set_trainable_lora_and_head(
    model: nn.Module,
    head_keywords=("classifier",),
    verbose=True,
):
    """
    冻结所有参数，只打开：
    1. LoRA 参数
    2. 分类头参数
    """

    for p in model.parameters():
        p.requires_grad_(False)

    trainable_names = []

    for name, p in model.named_parameters():
        is_lora = ("lora_A" in name) or ("lora_B" in name)
        is_head = any(key in name for key in head_keywords)

        if is_lora or is_head:
            p.requires_grad_(True)
            trainable_names.append(name)

    if verbose:
        print("[LoRA] trainable parameters:")
        for name in trainable_names:
            print(f"  trainable: {name}")

        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[LoRA] total params     : {total:,}")
        print(f"[LoRA] trainable params : {trainable:,}")
        print(f"[LoRA] trainable ratio  : {100.0 * trainable / max(1, total):.4f}%")

    return trainable_names

def _resolve_lora_target_root(model: nn.Module, verbose=True):
    """
    兼容三种结构：

    ForensicDinoBaseline
        .backbone = DinoV2Wrapper / MAEBackboneWrapper / IBOTBackboneWrapper
            .backbone = 真正的 ViT

    因此优先使用 model.backbone.backbone。
    """

    if hasattr(model, "backbone") and hasattr(model.backbone, "backbone"):
        root = model.backbone.backbone
        root_name = "model.backbone.backbone"
    elif hasattr(model, "backbone"):
        root = model.backbone
        root_name = "model.backbone"
    else:
        root = model
        root_name = "model"

    if verbose:
        backbone_type = getattr(model, "backbone_type", "unknown")
        print(f"[LoRA] backbone_type: {backbone_type}")
        print(f"[LoRA] target root  : {root_name}")

    return root


def apply_lora_to_forensic_dino(model: nn.Module, config: dict, rank=0):
    """
    名字保留不变，兼容你的 train.py。

    支持：
        - DINOv2
        - MAE
        - iBOT

    默认 target_modules:
        - attn.qkv
        - attn.proj

    这三种 ViT 主干通常都能匹配到：
        blocks.*.attn.qkv
        blocks.*.attn.proj
    """

    lora_cfg = config.get("lora", {})
    enabled = lora_cfg.get("enabled", False)

    if not enabled:
        if rank == 0:
            print("[LoRA] disabled")
        return model

    r = lora_cfg.get("r", 8)
    alpha = lora_cfg.get("alpha", 16)
    dropout = lora_cfg.get("dropout", 0.0)
    target_modules = lora_cfg.get(
        "target_modules",
        ["attn.qkv", "attn.proj"],
    )
    head_keywords = tuple(
        lora_cfg.get("head_keywords", ["classifier"])
    )

    verbose = rank == 0

    target_root = _resolve_lora_target_root(
        model=model,
        verbose=verbose,
    )

    num_matched = inject_lora_to_linear_layers(
        root=target_root,
        target_keywords=target_modules,
        r=r,
        alpha=alpha,
        dropout=dropout,
        verbose=verbose,
    )

    if num_matched == 0:
        if verbose:
            print("[LoRA] 可用 Linear 层如下，前 200 个：")
            count = 0
            for name, module in target_root.named_modules():
                if isinstance(module, nn.Linear):
                    print(f"  Linear: {name}")
                    count += 1
                    if count >= 200:
                        break

        raise RuntimeError(
            "[LoRA] 没有匹配到任何 Linear 层。"
            f" 当前 target_modules={target_modules}。"
            "请检查 backbone 的 attention 模块命名。"
        )

    set_trainable_lora_and_head(
        model=model,
        head_keywords=head_keywords,
        verbose=verbose,
    )

    return model