import os
import csv
import random
import numpy as np
import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from tqdm import tqdm

from model.dino_baseline import ForensicDinoBaseline
from pre_data.dino_dataprocess import ForensicImageDataset


# =========================================================
# 用户设置区：这里改成你自己的路径和参数
# =========================================================
DEVICE = "cuda:0"

# 你的模型权重
CHECKPOINT_PATH = "/mnt/data3/zhiyu/ForensicArtifacts/checkpoints/dino_baseline/checkpoint_epoch_20.pth"

# 是否用 EMA 权重
USE_EMA = False

# 要分析哪个 json
JSON_PATH = "/mnt/data2/zhiyu/Data/small_openmmsec/test_small.json"

# 保存目录
SAVE_DIR = "./analysis/occlusion_heatmap"

# 训练时模型结构配置：必须和你训练时一致
MODEL_CONFIG = {
    "model": {
        "repo_path": "/mnt/data3/zhiyu/dino_clip/dinov2_repo",
        "backbone_name": "dinov2_vitb14_reg",
        "pretrained": True,
        "freeze_backbone": True,
        "unfreeze_last_n_blocks": 0,
        "unfreeze_norm": True,
        "pooling_type": "cls_patch_mean",
        "hidden_dim": 512,
        "dropout": 0.1,
    },
    "data": {
        "image_size": 224,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "strict_mode": False
    }
}

# 数据过滤，可选
TARGET_DOMAINS = ["IMDL"]        # 例如 ["Doc", "AIGC"]，不筛选就用 None
TARGET_LABELS = None         # 例如 [1] 只看 fake，不筛选就 None
TARGET_MANI_TYPES = None     # 不筛选就 None

# 选哪些样本来画图
# 如果 SAMPLE_INDICES 不为 None，就只分析这些下标
SAMPLE_INDICES = None        # 例如 [0, 5, 8, 20]
START_INDEX = 0
MAX_SAMPLES = 50

# 遮挡参数
PATCH_SIZE = 16
STRIDE = 16
OCC_BATCH_SIZE = 64

# 解释哪个分数
# "pred"：解释模型当前预测的类别（推荐）
# "fake"：一直解释假图分数
# "real"：一直解释真图分数
TARGET_MODE = "pred"

# 遮挡块填充值
# "mean"：用整张图均值填
# "zero"：用 0 填
FILL_MODE = "mean"

# 随机种子
SEED = 42


# =========================================================
# 工具函数
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def to_device(device_str):
    return torch.device(device_str if torch.cuda.is_available() else "cpu")


def denormalize_image(x, mean, std):
    """
    x: [3, H, W]，归一化后的 tensor
    return: [H, W, 3]，范围 0~1
    """
    x = x.detach().cpu().float().clone()
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    x = x * std + mean
    x = torch.clamp(x, 0.0, 1.0)
    x = x.permute(1, 2, 0).numpy()
    return x


def get_logits(model, x):
    out = model(x)
    if isinstance(out, (tuple, list)):
        logits = out[0]
    else:
        logits = out
    return logits


def load_model(device):
    model = ForensicDinoBaseline(MODEL_CONFIG).to(device)

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)

    if USE_EMA and ckpt.get("ema_model_state_dict", None) is not None:
        ema_state = ckpt["ema_model_state_dict"]
        if isinstance(ema_state, dict) and "ema_state_dict" in ema_state:
            state_dict = ema_state["ema_state_dict"]
        else:
            state_dict = ema_state
        print("[Info] 使用 EMA 权重")
    else:
        state_dict = ckpt["model_state_dict"]
        print("[Info] 使用 Student 权重")

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def build_dataset():
    dataset = ForensicImageDataset(
        json_path=JSON_PATH,
        image_size=MODEL_CONFIG["data"]["image_size"],
        mean=tuple(MODEL_CONFIG["data"]["mean"]),
        std=tuple(MODEL_CONFIG["data"]["std"]),
        is_train=False,
        target_domains=TARGET_DOMAINS,
        target_labels=TARGET_LABELS,
        target_mani_types=TARGET_MANI_TYPES,
        strict_mode=MODEL_CONFIG["data"].get("strict_mode", False)
    )
    return dataset


@torch.no_grad()
def build_occlusion_heatmap(model, x):
    """
    x: [1, 3, H, W]
    返回:
        heat: [H, W]
        info: 一些基本信息
    """
    device = x.device
    _, _, H, W = x.shape

    # 原图分数
    logits = get_logits(model, x)                      # [1, 1]
    prob_fake = torch.sigmoid(logits).view(-1)[0].item()
    pred_label = 1 if prob_fake >= 0.5 else 0

    if TARGET_MODE == "pred":
        target_label = pred_label
    elif TARGET_MODE == "fake":
        target_label = 1
    elif TARGET_MODE == "real":
        target_label = 0
    else:
        raise ValueError(f"TARGET_MODE 只能是 pred/fake/real，当前: {TARGET_MODE}")

    if target_label == 1:
        base_score = prob_fake
    else:
        base_score = 1.0 - prob_fake

    # 遮挡值
    if FILL_MODE == "mean":
        fill_value = x.mean(dim=(2, 3), keepdim=True)  # [1, 3, 1, 1]
    elif FILL_MODE == "zero":
        fill_value = torch.zeros((1, 3, 1, 1), device=device, dtype=x.dtype)
    else:
        raise ValueError(f"FILL_MODE 只能是 mean/zero，当前: {FILL_MODE}")

    occ_inputs = []
    positions = []

    for top in range(0, H - PATCH_SIZE + 1, STRIDE):
        for left in range(0, W - PATCH_SIZE + 1, STRIDE):
            x_occ = x.clone()
            x_occ[:, :, top:top + PATCH_SIZE, left:left + PATCH_SIZE] = fill_value
            occ_inputs.append(x_occ)
            positions.append((top, left))

    occ_scores = []
    for start in range(0, len(occ_inputs), OCC_BATCH_SIZE):
        batch_x = torch.cat(occ_inputs[start:start + OCC_BATCH_SIZE], dim=0)
        batch_logits = get_logits(model, batch_x)
        batch_prob_fake = torch.sigmoid(batch_logits).view(-1)

        if target_label == 1:
            batch_score = batch_prob_fake
        else:
            batch_score = 1.0 - batch_prob_fake

        occ_scores.extend(batch_score.detach().cpu().numpy().tolist())

    heat_sum = np.zeros((H, W), dtype=np.float32)
    heat_cnt = np.zeros((H, W), dtype=np.float32)

    for s_occ, (top, left) in zip(occ_scores, positions):
        # 遮住后分数下降越多，说明这块越重要
        delta = base_score - s_occ
        heat_sum[top:top + PATCH_SIZE, left:left + PATCH_SIZE] += delta
        heat_cnt[top:top + PATCH_SIZE, left:left + PATCH_SIZE] += 1.0

    heat = heat_sum / (heat_cnt + 1e-8)

    info = {
        "prob_fake": float(prob_fake),
        "pred_label": int(pred_label),
        "target_label": int(target_label),
        "base_score": float(base_score),
    }
    return heat, info


def normalize_positive_heat(heat):
    heat_pos = np.maximum(heat, 0.0)
    if heat_pos.max() > 0:
        heat_norm = heat_pos / (heat_pos.max() + 1e-8)
    else:
        heat_norm = heat_pos
    return heat_pos, heat_norm


def make_overlay(img, heat_norm, alpha=0.55, cmap_name="jet"):
    cmap = plt.get_cmap(cmap_name)
    heat_color = cmap(heat_norm)[..., :3]
    alpha_map = alpha * heat_norm[..., None]
    overlay = img * (1.0 - alpha_map) + heat_color * alpha_map
    overlay = np.clip(overlay, 0.0, 1.0)
    return overlay


def save_visuals(img, heat, info, meta, sample_name):
    heat_pos, heat_norm = normalize_positive_heat(heat)
    overlay = make_overlay(img, heat_norm)

    # 保存原始热力图数据
    np.save(os.path.join(SAVE_DIR, f"{sample_name}_heat.npy"), heat)

    # 1. 只保存叠加图
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        os.path.join(SAVE_DIR, f"{sample_name}_overlay.png"),
        dpi=200,
        bbox_inches="tight",
        pad_inches=0
    )
    plt.close()

    # 2. 保存三联图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(img)
    axes[0].set_title(
        f"原图\nlabel={meta['label']}  pred={info['pred_label']}  prob_fake={info['prob_fake']:.4f}"
    )
    axes[0].axis("off")

    im = axes[1].imshow(heat, cmap="seismic")
    axes[1].set_title(
        f"遮挡热力图\ntarget={info['target_label']}  score={info['base_score']:.4f}"
    )
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay)
    axes[2].set_title("叠加图")
    axes[2].axis("off")

    fig.suptitle(
        f"domain={meta['domain']} | source={meta['source']}\n{meta['path']}",
        fontsize=10
    )
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, f"{sample_name}_all.png"), dpi=200)
    plt.close(fig)


def main():
    set_seed(SEED)
    ensure_dir(SAVE_DIR)
    device = to_device(DEVICE)

    dataset = build_dataset()
    print(f"[Info] 数据集样本数: {len(dataset)}")

    model = load_model(device)

    if SAMPLE_INDICES is not None:
        chosen_indices = SAMPLE_INDICES
    else:
        chosen_indices = list(range(START_INDEX, min(START_INDEX + MAX_SAMPLES, len(dataset))))

    print(f"[Info] 实际分析样本数: {len(chosen_indices)}")

    csv_rows = []

    for idx in tqdm(chosen_indices, desc="Heatmap"):
        sample = dataset[idx]

        # image tensor
        x = sample["image"].unsqueeze(0).to(device)

        # 可视化原图
        vis_img = denormalize_image(
            sample["image"],
            MODEL_CONFIG["data"]["mean"],
            MODEL_CONFIG["data"]["std"]
        )

        heat, info = build_occlusion_heatmap(model, x)

        img_path = sample["path"]
        image_stem = os.path.splitext(os.path.basename(img_path))[0]

        source = sample.get("ori_dataset", None)
        if source is None:
            source = sample.get("real_source", "UnknownSource")

        meta = {
            "label": int(sample["label"]),
            "domain": str(sample["domain"]),
            "source": str(source),
            "path": str(img_path),
        }

        sample_name = f"idx{idx:05d}_{meta['domain']}_y{meta['label']}_{image_stem}"

        save_visuals(
            img=vis_img,
            heat=heat,
            info=info,
            meta=meta,
            sample_name=sample_name
        )

        csv_rows.append({
            "index": idx,
            "sample_name": sample_name,
            "path": meta["path"],
            "domain": meta["domain"],
            "source": meta["source"],
            "label": meta["label"],
            "pred_label": info["pred_label"],
            "prob_fake": info["prob_fake"],
            "target_label": info["target_label"],
            "base_score": info["base_score"]
        })

    csv_path = os.path.join(SAVE_DIR, "summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index", "sample_name", "path", "domain", "source",
                "label", "pred_label", "prob_fake", "target_label", "base_score"
            ]
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n[Done] 热力图结果保存在: {SAVE_DIR}")
    print(f"[Done] 汇总表: {csv_path}")


if __name__ == "__main__":
    main()