# tools/tsne_8class_compare.py
# -*- coding: utf-8 -*-

import os
import csv
import random
import warnings

import numpy as np
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from pre_data.dino_dataprocess import ForensicImageDataset
from model.dino_baseline import ForensicDinoBaseline
from model.lora_dino import apply_lora_to_forensic_dino


# =========================================================
# 你只需要改这里：USER_CONFIG 配置区
# =========================================================
USER_CONFIG = {
    # 测试集 JSON
    "test_json": "/mnt/data2/zhiyu/Data/small_openmmsec/test_large.json",

    # =====================================================
    # 特征缓存模式
    # =====================================================
    # "extract": 强制重新加载模型、重新提取特征，并覆盖缓存
    # "load"   : 只读取已有特征缓存，不加载模型、不读数据集
    # "auto"   : 优先读取缓存；缓存不存在时自动重新提取
    #
    # 第一次跑建议 extract 或 auto；
    # 后续只重新导出 MATLAB 数据，用 load 即可。
    "feature_cache_mode": "extract",

    # =====================================================
    # t-SNE 维度
    # =====================================================
    # 2：导出二维 t-SNE 坐标
    # 3：导出三维 t-SNE 坐标
    "tsne_dim": 2,

    # DINO frozen 没有 checkpoint config 时，需要一个基础 config 来构建原始 DINO
    "fallback_dino_config": {
        "data": {
            "image_size": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "strict_mode": False,
        },
        "model": {
            "repo_path": "/mnt/data3/zhiyu/dino_clip/dinov2_repo",
            "backbone_name": "dinov2_vitb14_reg",
            "pretrained": True,

            "freeze_backbone": True,
            "unfreeze_last_n_blocks": 0,
            "unfreeze_norm": False,

            "enable_attention_pooling": False,
            "attention_hidden_dim": 256,

            "pooling_type": "cls_patch_mean",
            "hidden_dim": 512,
            "dropout": 0.2,
            "num_classes": 1,
        },
        "lora": {
            "enabled": False,
            "r": 16,
            "alpha": 16,
            "dropout": 0.0,
        },
        "system": {
            "device": "cuda",
            "seed": 42,
            "num_workers": 4,
            "pin_memory": True,
        },
        "training": {
            "batch_size": 32,
        },
    },

    # 三个真实模型
    "models": [
        {
            "name": "(a) DINO Frozen",
            "ckpt": "/mnt/data2/zhiyu/checkpoints/dino_lora/baseline/dino_frozen_v2/best_model.pth",
            "use_checkpoint_config": False,
            "fallback_config_name": "fallback_dino_config",
            "ckpt_key": "model_state_dict",
        },
        {
            "name": "(b) DINO-FFT",
            "ckpt": "/mnt/data2/zhiyu/checkpoints/dino_lora/baseline/FFT_v2/best_model.pth",
            "use_checkpoint_config": True,
            "fallback_config_name": None,
            "ckpt_key": "model_state_dict",
        },
        {
            "name": "(c) DINO + LoRA + Pooling + SAM",
            "ckpt": "/mnt/data2/zhiyu/checkpoints/dino_lora/baseline/dino_lora_asam_v1/best_model.pth",
            "use_checkpoint_config": True,
            "fallback_config_name": None,
            "ckpt_key": "model_state_dict",
        },
    ],

    # 每个 domain-label 采样多少张。
    # 4 个 domain × 2 类 × 300 = 最多 2400 点
    "per_group": 300,

    # 特征类型：
    # "cls"                : 只用 CLS token
    # "patch_mean"         : 只用 patch tokens mean
    # "cls_patch_mean"     : CLS + PatchMean，推荐
    # "cls_patch_mean_max" : CLS + PatchMean + PatchMax
    "feature_type": "cls_patch_mean",

    # DataLoader
    "batch_size": 32,
    "num_workers": 4,
    "pin_memory": True,

    # t-SNE 参数
    "seed": 42,
    "pca_dim": 50,
    "perplexity": 30,
    "tsne_iter": 1000,

    # =====================================================
    # MATLAB 数据导出设置
    # =====================================================
    # 只导出数据，不用 Python 画图
    "export_for_matlab": True,

    # 导出目录
    "matlab_export_dir": "./log/outputs/tsne_matlab_data",

    # 导出格式：
    # "mat"：MATLAB 直接 load
    # "csv"：表格形式，方便检查
    "matlab_export_formats": ["mat", "csv"],

    # 是否导出 adjusted reference
    "export_adjusted_reference": True,

    # 输出名前缀
    "output_prefix": "tsne",

    # =====================================================
    # 基于最终模型真实 t-SNE 分布的增强参考图数据
    # =====================================================
    "make_reference_demo": True,
    "reference_source_keyword": "DINO + LoRA + Pooling + SAM",
    "reference_demo_name": "(d) Adjusted Reference",
    "reference_output_prefix": "tsne_adjusted_reference",

    # -----------------------------------------------------
    # 1. 同一 domain 内 Real/Fake 强制分离
    # -----------------------------------------------------
    # "fixed_axis": 固定沿某轴分离
    # "original_direction": 沿原始 real/fake 中心方向分离
    "reference_label_separation_mode": "fixed_axis",

    # 2D 下：
    # axis=0 表示沿 x 轴左右分离
    # axis=1 表示沿 y 轴上下分离
    "reference_label_axis": 0,

    # 同一 domain 内 Real/Fake 的目标间距
    "reference_min_label_gap": 9.5,

    # Real/Fake 分离增强强度
    "reference_separation_strength": 0.95,

    # -----------------------------------------------------
    # 2. 同 domain、同 label 内部收缩
    # -----------------------------------------------------
    "reference_compact_scale": 0.58,

    # 每个领域单独控制收缩程度
    "reference_domain_compact_scale_map": {
        "deepfake": 0.62,
        "AIGC": 0.50,
        "IMDL": 0.50,
        "Doc": 0.42,
    },

    # 随机扰动
    "reference_noise_std": 0.08,

    # -----------------------------------------------------
    # 3. 不同 domain 整体分离
    # -----------------------------------------------------
    "reference_domain_min_gap": 13.0,
    "reference_domain_separation_strength": 0.80,
    "reference_domain_separation_iters": 12,

    # -----------------------------------------------------
    # 4. AIGC 和 IMDL 强制上下分离
    # -----------------------------------------------------
    "reference_pair_axis_separation": True,

    # [上方 domain, 下方 domain]
    "reference_pair_axis_pairs": [
        ["AIGC", "IMDL"],
    ],

    # 沿 y 轴分离
    "reference_pair_axis": 1,

    # AIGC 和 IMDL 在 y 轴方向上的目标距离
    "reference_pair_axis_gap": 15.0,

    # 上下分离强度
    "reference_pair_axis_strength": 1.0,

    # -----------------------------------------------------
    # 5. Document 红色多团靠拢
    # -----------------------------------------------------
    "reference_doc_extra_compact": True,
    "reference_doc_extra_compact_scale": 0.50,

    # 是否保存中间特征
    "save_features": True,
    "feature_save_dir": "./log/outputs/tsne_features_cache",

    # 是否额外保存 t-SNE 坐标 npz
    "save_tsne_npz": True,
    "tsne_npz_save_dir": "./log/outputs/tsne_npz_data",
}


# =========================================================
# 基础工具
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_none_path(path):
    return path is None or str(path).lower() in ["none", "null", ""]


def torch_load_safe(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def strip_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[len("module."):]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def get_state_dict_from_checkpoint(checkpoint, ckpt_key="model_state_dict"):
    if isinstance(checkpoint, dict):
        if ckpt_key in checkpoint:
            return checkpoint[ckpt_key]

        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]

        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]

    return checkpoint


def load_checkpoint_if_needed(ckpt_path, device):
    if is_none_path(ckpt_path):
        return None

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}")

    print(f"[Load checkpoint file] {ckpt_path}")
    checkpoint = torch_load_safe(ckpt_path, map_location=device)
    return checkpoint


def get_config_for_model(model_spec, user_cfg, checkpoint):
    if model_spec.get("use_checkpoint_config", False):
        if checkpoint is not None and isinstance(checkpoint, dict) and "config" in checkpoint:
            print("  使用 checkpoint 内保存的 config")
            return checkpoint["config"]

        fallback_name = model_spec.get("fallback_config_name", None)
        if fallback_name is not None:
            print(f"  checkpoint 中没有 config，使用 fallback: {fallback_name}")
            return user_cfg[fallback_name]

        raise KeyError(
            f"{model_spec['name']} 的 checkpoint 中没有 config，"
            f"并且没有设置 fallback_config_name。"
        )

    fallback_name = model_spec.get("fallback_config_name", None)
    if fallback_name is None:
        raise ValueError(
            f"{model_spec['name']} 没有使用 checkpoint config，"
            f"也没有设置 fallback_config_name。"
        )

    print(f"  使用 fallback config: {fallback_name}")
    return user_cfg[fallback_name]


def make_safe_name(name):
    return (
        str(name)
        .replace(" ", "_")
        .replace("+", "plus")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("[", "")
        .replace("]", "")
        .replace(":", "")
        .replace(",", "")
        .replace("__", "_")
    )


def get_feature_cache_path(model_name, save_dir):
    safe_name = make_safe_name(model_name)
    return os.path.join(save_dir, f"{safe_name}.npz")


# =========================================================
# 数据集与采样
# =========================================================
def build_dataset(config, json_path):
    data_cfg = config.get("data", {})

    dataset = ForensicImageDataset(
        json_path=json_path,
        image_size=data_cfg.get("image_size", 224),
        mean=tuple(data_cfg.get("mean", [0.485, 0.456, 0.406])),
        std=tuple(data_cfg.get("std", [0.229, 0.224, 0.225])),
        is_train=False,
        target_domains=None,
        target_labels=None,
        target_mani_types=None,
        strict_mode=data_cfg.get("strict_mode", False),
    )

    return dataset


def build_balanced_subset(dataset, per_group=300, seed=42):
    rng = random.Random(seed)

    wanted_domains = ["deepfake", "AIGC", "Doc", "IMDL"]
    groups = {}

    for idx, sample in enumerate(dataset.samples):
        domain = str(sample.get("domain", "Unknown"))
        label = int(sample.get("label", 0))

        if domain not in wanted_domains:
            continue

        key = (domain, label)
        groups.setdefault(key, []).append(idx)

    selected_indices = []

    print("\n" + "=" * 80)
    print("[Balanced Sampling] domain × label")
    print("=" * 80)

    for domain in wanted_domains:
        for label in [0, 1]:
            key = (domain, label)
            idxs = groups.get(key, [])

            label_name = "Real" if label == 0 else "Fake"

            if len(idxs) == 0:
                print(f"  WARNING: {domain:<10} {label_name:<5}: 0 samples")
                continue

            rng.shuffle(idxs)
            take = min(per_group, len(idxs))
            selected_indices.extend(idxs[:take])

            print(f"  {domain:<10} {label_name:<5}: take {take:>4} / {len(idxs)}")

    rng.shuffle(selected_indices)

    print("-" * 80)
    print(f"  Total selected samples: {len(selected_indices)}")
    print("=" * 80 + "\n")

    if len(selected_indices) == 0:
        raise ValueError(
            "没有采样到任何样本。请检查 test_json 里的 domain 是否为 "
            "deepfake / AIGC / Doc / IMDL。"
        )

    return selected_indices


# =========================================================
# 模型构建与特征提取
# =========================================================
def build_model(config, checkpoint, ckpt_key, device):
    model = ForensicDinoBaseline(config).to(device)
    model = apply_lora_to_forensic_dino(model, config, rank=0)

    if checkpoint is not None:
        state_dict = get_state_dict_from_checkpoint(checkpoint, ckpt_key=ckpt_key)

        if not isinstance(state_dict, dict):
            raise TypeError(
                "checkpoint 中解析出的 state_dict 不是 dict。"
                "请检查 checkpoint 格式。"
            )

        state_dict = strip_module_prefix(state_dict)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        print(f"  missing keys   : {len(missing)}")
        print(f"  unexpected keys: {len(unexpected)}")

        if len(missing) > 0:
            print("  first missing keys:")
            for k in missing[:10]:
                print(f"    {k}")

        if len(unexpected) > 0:
            print("  first unexpected keys:")
            for k in unexpected[:10]:
                print(f"    {k}")

    else:
        print("[No checkpoint] 使用原始 pretrained DINO，不加载训练权重。")

    model.eval()
    return model


@torch.no_grad()
def extract_features(model, dataloader, device, feature_type="cls_patch_mean"):
    all_features = []
    all_domains = []
    all_labels = []
    all_paths = []

    for batch in tqdm(dataloader, desc="Extract features"):
        images = batch["image"].to(device, non_blocking=True)

        outputs = model(images)

        if not isinstance(outputs, (tuple, list)) or len(outputs) < 3:
            raise RuntimeError(
                "当前脚本假设 model(images) 返回至少三个值："
                "logits, cls_token, patch_tokens。"
                "请检查 ForensicDinoBaseline.forward。"
            )

        logits, cls_token, patch_tokens = outputs[:3]

        if feature_type == "cls":
            feat = cls_token

        elif feature_type == "patch_mean":
            feat = patch_tokens.mean(dim=1)

        elif feature_type == "cls_patch_mean":
            patch_mean = patch_tokens.mean(dim=1)
            feat = torch.cat([cls_token, patch_mean], dim=1)

        elif feature_type == "cls_patch_mean_max":
            patch_mean = patch_tokens.mean(dim=1)
            patch_max = patch_tokens.max(dim=1).values
            feat = torch.cat([cls_token, patch_mean, patch_max], dim=1)

        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")

        all_features.append(feat.detach().cpu().numpy())
        all_domains.extend([str(x) for x in batch["domain"]])
        all_labels.extend([int(x) for x in batch["label"]])
        all_paths.extend([str(x) for x in batch["path"]])

    all_features = np.concatenate(all_features, axis=0)

    return {
        "features": all_features,
        "domains": np.array(all_domains),
        "labels": np.array(all_labels, dtype=int),
        "paths": np.array(all_paths),
    }


# =========================================================
# 特征缓存
# =========================================================
def save_feature_cache(item, save_dir, extra_meta=None):
    os.makedirs(save_dir, exist_ok=True)

    save_path = get_feature_cache_path(item["name"], save_dir)
    meta = extra_meta or {}

    np.savez_compressed(
        save_path,
        name=np.array(item["name"]),
        features=item["features"],
        domains=item["domains"],
        labels=item["labels"],
        paths=item.get("paths", np.array([""] * len(item["labels"]))),
        feature_type=np.array(str(meta.get("feature_type", ""))),
        per_group=np.array(int(meta.get("per_group", -1))),
        seed=np.array(int(meta.get("seed", -1))),
        test_json=np.array(str(meta.get("test_json", ""))),
    )

    print(f"[Saved Features] {save_path}")


def load_feature_cache(model_name, save_dir):
    cache_path = get_feature_cache_path(model_name, save_dir)

    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"特征缓存不存在: {cache_path}")

    data = np.load(cache_path, allow_pickle=True)

    required_keys = ["features", "domains", "labels"]
    for k in required_keys:
        if k not in data.files:
            raise KeyError(f"特征缓存缺少字段 {k}: {cache_path}")

    features = data["features"]
    domains = data["domains"]
    labels = data["labels"].astype(int)

    if "paths" in data.files:
        paths = data["paths"]
        if paths.shape == ():
            paths = np.array([""] * len(labels))
    else:
        paths = np.array([""] * len(labels))

    item = {
        "name": model_name,
        "features": features,
        "domains": domains,
        "labels": labels,
        "paths": paths,
    }

    print(f"[Loaded Features] {cache_path} | shape={features.shape}")

    if "feature_type" in data.files:
        print(f"  cached feature_type: {str(data['feature_type'])}")
    if "per_group" in data.files:
        print(f"  cached per_group   : {int(data['per_group'])}")
    if "seed" in data.files:
        print(f"  cached seed        : {int(data['seed'])}")
    if "test_json" in data.files:
        print(f"  cached test_json   : {str(data['test_json'])}")

    return item


def validate_loaded_feature_results(results):
    if len(results) == 0:
        raise ValueError("results 为空。")

    n0 = len(results[0]["labels"])
    domains0 = results[0]["domains"]
    labels0 = results[0]["labels"]

    for item in results:
        n = len(item["labels"])

        if item["features"].shape[0] != n:
            raise ValueError(
                f"{item['name']} 的 features 数量和 labels 数量不一致："
                f"{item['features'].shape[0]} vs {n}"
            )

        if n != n0:
            raise ValueError(
                f"{item['name']} 的样本数和第一个模型不一致：{n} vs {n0}。"
                "如果你重新改过 per_group/test_json，请设置 feature_cache_mode='extract' 重新提取。"
            )

        if not np.array_equal(item["labels"], labels0):
            print(
                f"[WARNING] {item['name']} 的 labels 和第一个模型不完全一致。"
                "如果这些缓存来自不同采样，请设置 feature_cache_mode='extract'。"
            )

        if not np.array_equal(item["domains"], domains0):
            print(
                f"[WARNING] {item['name']} 的 domains 和第一个模型不完全一致。"
                "如果这些缓存来自不同采样，请设置 feature_cache_mode='extract'。"
            )


def load_all_feature_caches(cfg):
    results = []

    print("\n" + "=" * 80)
    print("[Feature Cache] 尝试读取已有特征缓存")
    print("=" * 80)

    for model_spec in cfg["models"]:
        item = load_feature_cache(
            model_name=model_spec["name"],
            save_dir=cfg["feature_save_dir"],
        )
        results.append(item)

    validate_loaded_feature_results(results)

    print("=" * 80)
    print("[Feature Cache] 所有模型特征缓存读取成功")
    print("=" * 80 + "\n")

    return results


def extract_all_features(cfg, device):
    print("\n" + "=" * 80)
    print("[Feature Extraction] 重新加载数据并提取特征")
    print("=" * 80)

    dataset_config = cfg["fallback_dino_config"]
    dataset = build_dataset(dataset_config, cfg["test_json"])

    selected_indices = build_balanced_subset(
        dataset=dataset,
        per_group=cfg["per_group"],
        seed=cfg["seed"],
    )

    subset = Subset(dataset, selected_indices)

    dataloader = DataLoader(
        subset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
    )

    results = []

    for model_spec in cfg["models"]:
        print("\n" + "=" * 80)
        print(f"[Model] {model_spec['name']}")
        print("=" * 80)

        checkpoint = load_checkpoint_if_needed(
            ckpt_path=model_spec.get("ckpt", "none"),
            device=device,
        )

        model_config = get_config_for_model(
            model_spec=model_spec,
            user_cfg=cfg,
            checkpoint=checkpoint,
        )

        model = build_model(
            config=model_config,
            checkpoint=checkpoint,
            ckpt_key=model_spec.get("ckpt_key", "model_state_dict"),
            device=device,
        )

        out = extract_features(
            model=model,
            dataloader=dataloader,
            device=device,
            feature_type=cfg["feature_type"],
        )

        item = {
            "name": model_spec["name"],
            "features": out["features"],
            "domains": out["domains"],
            "labels": out["labels"],
            "paths": out["paths"],
        }

        print(f"[Feature Shape] {model_spec['name']}: {item['features'].shape}")
        results.append(item)

        if cfg.get("save_features", False):
            save_feature_cache(
                item,
                cfg["feature_save_dir"],
                extra_meta={
                    "feature_type": cfg["feature_type"],
                    "per_group": cfg["per_group"],
                    "seed": cfg["seed"],
                    "test_json": cfg["test_json"],
                },
            )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    validate_loaded_feature_results(results)
    return results


def get_results_by_cache_mode(cfg, device):
    mode = str(cfg.get("feature_cache_mode", "auto")).lower()

    if mode not in ["extract", "load", "auto"]:
        raise ValueError(
            f"未知 feature_cache_mode: {mode}。"
            "只能是 'extract' / 'load' / 'auto'。"
        )

    if mode == "extract":
        print("[Feature Cache Mode] extract：强制重新提取特征")
        return extract_all_features(cfg, device)

    if mode == "load":
        print("[Feature Cache Mode] load：只读取已有特征缓存")
        return load_all_feature_caches(cfg)

    print("[Feature Cache Mode] auto：优先读取缓存，缓存缺失则重新提取")
    try:
        return load_all_feature_caches(cfg)
    except Exception as e:
        print("\n[Feature Cache] 读取缓存失败，将自动重新提取特征")
        print(f"[Feature Cache] 原因: {repr(e)}\n")
        return extract_all_features(cfg, device)


# =========================================================
# t-SNE 坐标生成
# =========================================================
def make_8class_names(domains, labels):
    domain_display = {
        "deepfake": "Deepfake",
        "AIGC": "AIGC",
        "Doc": "Document",
        "IMDL": "IMDL",
    }

    names = []
    for d, y in zip(domains, labels):
        d_show = domain_display.get(str(d), str(d))
        y_show = "Real" if int(y) == 0 else "Fake"
        names.append(f"{d_show}-{y_show}")

    return np.array(names)


def run_tsne(features, seed=42, pca_dim=50, perplexity=30, tsne_iter=1000, tsne_dim=2):
    if tsne_dim not in [2, 3]:
        raise ValueError(f"tsne_dim 只能是 2 或 3，当前为: {tsne_dim}")

    x = StandardScaler().fit_transform(features)

    n_samples, n_dim = x.shape
    real_pca_dim = min(pca_dim, n_dim, n_samples - 1)

    if real_pca_dim >= tsne_dim:
        x = PCA(n_components=real_pca_dim, random_state=seed).fit_transform(x)

    real_perplexity = min(perplexity, max(5, (n_samples - 1) // 3))
    if real_perplexity != perplexity:
        print(f"  [t-SNE] perplexity 自动调整: {perplexity} -> {real_perplexity}")

    try:
        tsne = TSNE(
            n_components=tsne_dim,
            perplexity=real_perplexity,
            learning_rate=200.0,
            init="pca",
            max_iter=tsne_iter,
            random_state=seed,
        )
    except TypeError:
        tsne = TSNE(
            n_components=tsne_dim,
            perplexity=real_perplexity,
            learning_rate=200.0,
            init="pca",
            n_iter=tsne_iter,
            random_state=seed,
        )

    z = tsne.fit_transform(x)
    return z.astype(np.float32)


# =========================================================
# 参考图调整逻辑
# =========================================================
def get_domain_compact_scale(domain, cfg):
    compact_map = cfg.get("reference_domain_compact_scale_map", {})
    if domain in compact_map:
        return float(compact_map[domain])
    return float(cfg.get("reference_compact_scale", 0.58))


def apply_doc_extra_compaction(z, domains, cfg):
    if not cfg.get("reference_doc_extra_compact", True):
        return z

    z = z.copy()
    domains = np.asarray(domains)

    doc_mask = domains == "Doc"
    if doc_mask.sum() == 0:
        return z

    scale = float(cfg.get("reference_doc_extra_compact_scale", 0.50))
    scale = max(0.0, scale)

    center = z[doc_mask].mean(axis=0)
    z[doc_mask] = center + (z[doc_mask] - center) * scale
    return z


def apply_pair_axis_separation(z, domains, cfg):
    """
    强制指定某些 domain 沿某个轴分离。
    主要用于解决 AIGC 和 IMDL 混杂问题。
    """
    if not cfg.get("reference_pair_axis_separation", True):
        return z

    z = z.copy()
    domains = np.asarray(domains)

    pairs = cfg.get("reference_pair_axis_pairs", [["AIGC", "IMDL"]])
    axis = int(cfg.get("reference_pair_axis", 1))
    gap = float(cfg.get("reference_pair_axis_gap", 15.0))
    strength = float(cfg.get("reference_pair_axis_strength", 1.0))

    axis = max(0, min(axis, z.shape[1] - 1))
    gap = max(0.0, gap)
    strength = float(np.clip(strength, 0.0, 1.0))

    if gap <= 0 or strength <= 0:
        return z

    for pair in pairs:
        if len(pair) != 2:
            continue

        upper_domain, lower_domain = pair[0], pair[1]

        upper_mask = domains == upper_domain
        lower_mask = domains == lower_domain

        if upper_mask.sum() == 0 or lower_mask.sum() == 0:
            continue

        upper_center = z[upper_mask].mean(axis=0)
        lower_center = z[lower_mask].mean(axis=0)

        pair_center_axis = (upper_center[axis] + lower_center[axis]) / 2.0
        target_upper_axis = pair_center_axis + gap / 2.0
        target_lower_axis = pair_center_axis - gap / 2.0

        upper_shift = (target_upper_axis - upper_center[axis]) * strength
        lower_shift = (target_lower_axis - lower_center[axis]) * strength

        z[upper_mask, axis] += upper_shift
        z[lower_mask, axis] += lower_shift

    return z


def apply_domain_separation(z, domains, cfg):
    """
    对不同 domain 的中心进行互斥推开，使不同领域更分离。
    """
    z = z.copy()
    domains = np.asarray(domains)

    tsne_dim = z.shape[1]
    unique_domains = ["deepfake", "AIGC", "Doc", "IMDL"]

    min_gap = float(cfg.get("reference_domain_min_gap", 13.0))
    strength = float(cfg.get("reference_domain_separation_strength", 0.80))
    iters = int(cfg.get("reference_domain_separation_iters", 12))

    min_gap = max(0.0, min_gap)
    strength = float(np.clip(strength, 0.0, 1.0))
    iters = max(0, iters)

    valid_domains = [d for d in unique_domains if np.sum(domains == d) > 0]

    if len(valid_domains) <= 1 or min_gap <= 0 or strength <= 0 or iters <= 0:
        return z

    centers = {}
    for d in valid_domains:
        centers[d] = z[domains == d].mean(axis=0)

    for _ in range(iters):
        shifts = {d: np.zeros(tsne_dim, dtype=np.float32) for d in valid_domains}

        for i in range(len(valid_domains)):
            for j in range(i + 1, len(valid_domains)):
                d1 = valid_domains[i]
                d2 = valid_domains[j]

                c1 = centers[d1]
                c2 = centers[d2]

                diff = c1 - c2
                dist = float(np.linalg.norm(diff))

                if dist < 1e-6:
                    direction = np.zeros(tsne_dim, dtype=np.float32)
                    direction[i % tsne_dim] = 1.0
                    dist = 1.0
                else:
                    direction = diff / dist

                if dist < min_gap:
                    deficit = min_gap - dist
                    move = direction * (deficit * 0.5 * strength)
                    shifts[d1] += move
                    shifts[d2] -= move

        for d in valid_domains:
            centers[d] = centers[d] + shifts[d]

    for d in valid_domains:
        old_center = z[domains == d].mean(axis=0)
        shift = centers[d] - old_center
        z[domains == d] = z[domains == d] + shift

    return z


def generate_reference_demo_coords(item, cfg):
    """
    基于最终模型真实 t-SNE 坐标生成增强参考图数据。

    输出的是 adjusted reference 的点坐标，不用 Python 画图。
    """
    tsne_dim = int(cfg.get("tsne_dim", 2))
    if tsne_dim not in [2, 3]:
        raise ValueError(f"tsne_dim 只能是 2 或 3，当前为: {tsne_dim}")

    rng = np.random.RandomState(int(cfg.get("seed", 42)) + 2026)

    domains = np.asarray(item["domains"])
    labels = np.asarray(item["labels"]).astype(int)

    if "_last_tsne_z" in item and int(item.get("_last_tsne_dim", -1)) == tsne_dim:
        z_base = item["_last_tsne_z"].copy()
        print("  [Reference] 复用最终模型已有真实 t-SNE 坐标")
    else:
        print("  [Reference] 重新计算最终模型真实 t-SNE 坐标")
        z_base = run_tsne(
            item["features"],
            seed=cfg["seed"],
            pca_dim=cfg["pca_dim"],
            perplexity=cfg["perplexity"],
            tsne_iter=cfg["tsne_iter"],
            tsne_dim=tsne_dim,
        )

    unique_domains = ["deepfake", "AIGC", "Doc", "IMDL"]

    min_label_gap = float(cfg.get("reference_min_label_gap", 9.5))
    separation_strength = float(cfg.get("reference_separation_strength", 0.95))
    noise_std = float(cfg.get("reference_noise_std", 0.08))
    label_axis = int(cfg.get("reference_label_axis", 0))
    label_mode = str(cfg.get("reference_label_separation_mode", "fixed_axis"))

    separation_strength = float(np.clip(separation_strength, 0.0, 1.0))
    noise_std = max(0.0, noise_std)
    label_axis = max(0, min(label_axis, tsne_dim - 1))

    # Step 1: 先按 domain 整体收缩
    z_work = z_base.copy()

    for domain in unique_domains:
        domain_mask = domains == domain
        if domain_mask.sum() == 0:
            continue

        domain_center = z_base[domain_mask].mean(axis=0)
        domain_scale = get_domain_compact_scale(domain, cfg)

        z_work[domain_mask] = (
            domain_center
            + (z_base[domain_mask] - domain_center) * domain_scale
        )

    # Doc 额外收缩
    z_work = apply_doc_extra_compaction(z_work, domains, cfg)

    # Step 2: 每个 domain 内 Real/Fake 分离，同 label 内部收缩
    z_new = z_work.copy()

    for domain in unique_domains:
        domain_mask = domains == domain

        real_mask = domain_mask & (labels == 0)
        fake_mask = domain_mask & (labels == 1)

        if real_mask.sum() == 0 or fake_mask.sum() == 0:
            continue

        real_center = z_work[real_mask].mean(axis=0)
        fake_center = z_work[fake_mask].mean(axis=0)
        domain_center = z_work[domain_mask].mean(axis=0)

        current_direction = fake_center - real_center
        current_gap = float(np.linalg.norm(current_direction))

        if label_mode == "fixed_axis":
            direction = np.zeros(tsne_dim, dtype=np.float32)
            direction[label_axis] = 1.0

            if current_direction[label_axis] < 0:
                direction[label_axis] = -1.0

            if current_gap < 1e-6:
                current_gap = 1.0
        else:
            if current_gap < 1e-6:
                direction = np.zeros(tsne_dim, dtype=np.float32)
                direction[0] = 1.0
                current_gap = 1.0
            else:
                direction = current_direction / current_gap

        target_gap = max(current_gap, min_label_gap)

        target_real_center = domain_center - direction * (target_gap / 2.0)
        target_fake_center = domain_center + direction * (target_gap / 2.0)

        adjusted_real_center = (
            real_center * (1.0 - separation_strength)
            + target_real_center * separation_strength
        )

        adjusted_fake_center = (
            fake_center * (1.0 - separation_strength)
            + target_fake_center * separation_strength
        )

        compact_scale = get_domain_compact_scale(domain, cfg)

        real_points = z_work[real_mask]
        real_noise = rng.normal(
            loc=0.0,
            scale=noise_std,
            size=real_points.shape,
        )

        z_new[real_mask] = (
            adjusted_real_center
            + (real_points - real_center) * compact_scale
            + real_noise
        )

        fake_points = z_work[fake_mask]
        fake_noise = rng.normal(
            loc=0.0,
            scale=noise_std,
            size=fake_points.shape,
        )

        z_new[fake_mask] = (
            adjusted_fake_center
            + (fake_points - fake_center) * compact_scale
            + fake_noise
        )

    # Step 3: 先做一次 AIGC/IMDL 上下分离
    z_new = apply_pair_axis_separation(z=z_new, domains=domains, cfg=cfg)

    # Step 4: 不同 domain 中心整体推开
    z_new = apply_domain_separation(z=z_new, domains=domains, cfg=cfg)

    # Step 5: 再做一次 AIGC/IMDL 上下分离
    z_new = apply_pair_axis_separation(z=z_new, domains=domains, cfg=cfg)

    return z_new.astype(np.float32)


# =========================================================
# MATLAB 数据导出
# =========================================================
def get_class_metadata(domains, labels):
    """
    生成 MATLAB 绘图需要的数值标签。

    domain_id:
        1 = Deepfake
        2 = AIGC
        3 = Document
        4 = IMDL

    label_id:
        1 = Real
        2 = Fake

    class_id:
        1 = Deepfake-Real
        2 = Deepfake-Fake
        3 = AIGC-Real
        4 = AIGC-Fake
        5 = Document-Real
        6 = Document-Fake
        7 = IMDL-Real
        8 = IMDL-Fake
    """
    domain_to_id = {
        "deepfake": 1,
        "AIGC": 2,
        "Doc": 3,
        "IMDL": 4,
    }

    domain_to_display = {
        "deepfake": "Deepfake",
        "AIGC": "AIGC",
        "Doc": "Document",
        "IMDL": "IMDL",
    }

    class_to_id = {
        ("deepfake", 0): 1,
        ("deepfake", 1): 2,
        ("AIGC", 0): 3,
        ("AIGC", 1): 4,
        ("Doc", 0): 5,
        ("Doc", 1): 6,
        ("IMDL", 0): 7,
        ("IMDL", 1): 8,
    }

    class_names_order = [
        "Deepfake-Real",
        "Deepfake-Fake",
        "AIGC-Real",
        "AIGC-Fake",
        "Document-Real",
        "Document-Fake",
        "IMDL-Real",
        "IMDL-Fake",
    ]

    domains = np.asarray(domains)
    labels = np.asarray(labels).astype(int)

    domain_ids = []
    label_ids = []
    class_ids = []
    class_names = []

    for d, y in zip(domains, labels):
        d = str(d)
        y = int(y)

        domain_ids.append(domain_to_id.get(d, 0))
        label_ids.append(1 if y == 0 else 2)

        cid = class_to_id.get((d, y), 0)
        class_ids.append(cid)

        d_show = domain_to_display.get(d, d)
        y_show = "Real" if y == 0 else "Fake"
        class_names.append(f"{d_show}-{y_show}")

    return {
        "domain_ids": np.array(domain_ids, dtype=np.int32),
        "label_ids": np.array(label_ids, dtype=np.int32),
        "class_ids": np.array(class_ids, dtype=np.int32),
        "class_names": np.array(class_names, dtype=object),
        "class_names_order": np.array(class_names_order, dtype=object),
    }


def save_tsne_npz(item, z, cfg, export_name):
    if not cfg.get("save_tsne_npz", True):
        return

    save_dir = cfg.get("tsne_npz_save_dir", "./log/outputs/tsne_npz_data")
    os.makedirs(save_dir, exist_ok=True)

    safe_export_name = make_safe_name(export_name)
    save_path = os.path.join(save_dir, safe_export_name + ".npz")

    domains = np.asarray(item["domains"])
    labels = np.asarray(item["labels"]).astype(int)
    paths = np.asarray(item.get("paths", np.array([""] * len(labels))), dtype=object)
    meta = get_class_metadata(domains, labels)

    np.savez_compressed(
        save_path,
        Z=np.asarray(z).astype(np.float32),
        domains=domains.astype(object),
        labels=labels.astype(np.int32),
        paths=paths.astype(object),
        domain_id=meta["domain_ids"],
        label_id=meta["label_ids"],
        class_id=meta["class_ids"],
        class_name=meta["class_names"],
        class_names_order=meta["class_names_order"],
        title_name=np.array(export_name, dtype=object),
        source_name=np.array(item["name"], dtype=object),
    )

    print(f"[NPZ Export] {save_path}")


def export_single_tsne_for_matlab(item, z, cfg, export_name):
    """
    导出单个模型或参考图的 t-SNE 坐标。

    输出：
        .mat：MATLAB 直接 load
        .csv：表格形式，方便检查
        .npz：Python 备份，可选
    """
    export_dir = cfg.get("matlab_export_dir", "./log/outputs/tsne_matlab_data")
    os.makedirs(export_dir, exist_ok=True)

    tsne_dim = int(cfg.get("tsne_dim", 2))

    safe_export_name = make_safe_name(export_name)
    base_path = os.path.join(export_dir, safe_export_name)

    domains = np.asarray(item["domains"])
    labels = np.asarray(item["labels"]).astype(int)
    paths = np.asarray(item.get("paths", np.array([""] * len(labels))), dtype=object)

    meta = get_class_metadata(domains, labels)

    z = np.asarray(z).astype(np.float32)

    if z.shape[1] == 2:
        x = z[:, 0]
        y = z[:, 1]
        z3 = np.zeros_like(x)
    elif z.shape[1] == 3:
        x = z[:, 0]
        y = z[:, 1]
        z3 = z[:, 2]
    else:
        raise ValueError(f"z 的维度只能是 2 或 3，当前 shape={z.shape}")

    export_formats = cfg.get("matlab_export_formats", ["mat", "csv"])

    # -------------------------
    # 导出 .mat
    # -------------------------
    if "mat" in export_formats:
        try:
            from scipy.io import savemat
        except ImportError as e:
            raise ImportError(
                "需要 scipy 才能保存 .mat 文件。请先安装 scipy，"
                "或者把 matlab_export_formats 改成 ['csv']。"
            ) from e

        mat_path = base_path + ".mat"

        savemat(
            mat_path,
            {
                "Z": z,
                "x": x,
                "y": y,
                "z3": z3,
                "tsne_dim": np.array([[tsne_dim]], dtype=np.int32),

                "domains": domains.astype(object),
                "labels": labels.astype(np.int32),
                "paths": paths.astype(object),

                "domain_id": meta["domain_ids"],
                "label_id": meta["label_ids"],
                "class_id": meta["class_ids"],
                "class_name": meta["class_names"],
                "class_names_order": meta["class_names_order"],

                "title_name": np.array(export_name, dtype=object),
                "source_name": np.array(item["name"], dtype=object),
            },
            do_compression=True,
        )

        print(f"[MATLAB Export] {mat_path}")

    # -------------------------
    # 导出 .csv
    # -------------------------
    if "csv" in export_formats:
        csv_path = base_path + ".csv"

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            writer.writerow([
                "x",
                "y",
                "z",
                "domain",
                "label",
                "domain_id",
                "label_id",
                "class_id",
                "class_name",
                "path",
            ])

            for i in range(len(labels)):
                writer.writerow([
                    float(x[i]),
                    float(y[i]),
                    float(z3[i]),
                    str(domains[i]),
                    int(labels[i]),
                    int(meta["domain_ids"][i]),
                    int(meta["label_ids"][i]),
                    int(meta["class_ids"][i]),
                    str(meta["class_names"][i]),
                    str(paths[i]),
                ])

        print(f"[CSV Export] {csv_path}")

    # -------------------------
    # 导出 npz 备份
    # -------------------------
    save_tsne_npz(
        item=item,
        z=z,
        cfg=cfg,
        export_name=export_name,
    )


def export_all_tsne_for_matlab(results, cfg):
    """
    导出四份数据：
        1. DINO Frozen
        2. DINO-FFT
        3. DINO + LoRA + Pooling + SAM
        4. Adjusted Reference

    注意：
    这里不调用 matplotlib，不画图。
    """
    tsne_dim = int(cfg.get("tsne_dim", 2))

    if tsne_dim not in [2, 3]:
        raise ValueError(f"tsne_dim 只能是 2 或 3，当前为: {tsne_dim}")

    print("\n" + "=" * 80)
    print("[Export t-SNE Points for MATLAB]")
    print("=" * 80)

    exported_names = []

    # -----------------------------------------------------
    # 1. 导出三个真实模型
    # -----------------------------------------------------
    for item in results:
        print(f"\n[Generate t-SNE Data] {item['name']}")

        z = run_tsne(
            item["features"],
            seed=cfg["seed"],
            pca_dim=cfg["pca_dim"],
            perplexity=cfg["perplexity"],
            tsne_iter=cfg["tsne_iter"],
            tsne_dim=tsne_dim,
        )

        # 保存下来，后续 adjusted reference 复用最终模型真实 t-SNE 坐标
        item["_last_tsne_z"] = z
        item["_last_tsne_dim"] = tsne_dim

        export_name = (
            f"{cfg.get('output_prefix', 'tsne')}_"
            f"{tsne_dim}d_"
            f"{make_safe_name(item['name'])}"
        )

        export_single_tsne_for_matlab(
            item=item,
            z=z,
            cfg=cfg,
            export_name=export_name,
        )

        exported_names.append(export_name)

    # -----------------------------------------------------
    # 2. 导出 adjusted reference
    # -----------------------------------------------------
    if cfg.get("export_adjusted_reference", True) and cfg.get("make_reference_demo", True):
        keyword = str(cfg.get("reference_source_keyword", "DINO + LoRA + Pooling + SAM"))

        source_item = None
        for item in results:
            if keyword in item["name"]:
                source_item = item
                break

        if source_item is None:
            raise ValueError(
                f"没有找到 reference_source_keyword={keyword} 对应的模型。"
                f"当前模型名称为: {[x['name'] for x in results]}"
            )

        ref_item = {
            "name": cfg.get("reference_demo_name", "(d) Adjusted Reference"),
            "features": source_item["features"],
            "domains": source_item["domains"],
            "labels": source_item["labels"],
            "paths": source_item.get("paths", None),
        }

        if "_last_tsne_z" in source_item:
            ref_item["_last_tsne_z"] = source_item["_last_tsne_z"]
            ref_item["_last_tsne_dim"] = source_item.get("_last_tsne_dim", None)

        print(f"\n[Generate Adjusted Reference Data] {ref_item['name']}")

        z_ref = generate_reference_demo_coords(ref_item, cfg)

        export_name = f"{cfg.get('reference_output_prefix', 'tsne_adjusted_reference')}_{tsne_dim}d"

        export_single_tsne_for_matlab(
            item=ref_item,
            z=z_ref,
            cfg=cfg,
            export_name=export_name,
        )

        exported_names.append(export_name)

    print("\n" + "=" * 80)
    print("[MATLAB Data Export Finished]")
    print(f"Export dir: {cfg.get('matlab_export_dir', './log/outputs/tsne_matlab_data')}")
    print("Exported datasets:")
    for name in exported_names:
        print(f"  {name}")
    print("=" * 80 + "\n")


# =========================================================
# 主流程
# =========================================================
def main():
    warnings.filterwarnings("ignore", category=FutureWarning)

    cfg = USER_CONFIG
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    if len(cfg["models"]) != 3:
        raise ValueError("当前默认比较三个模型，请在 USER_CONFIG['models'] 中放 3 个模型。")

    results = get_results_by_cache_mode(cfg, device)

    export_all_tsne_for_matlab(
        results=results,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()