import os
import csv
import random
import numpy as np

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from model.dino_baseline import ForensicDinoBaseline

from pre_data.dino_dataprocess import ForensicImageDataset


# =========================================================
# 用户设置区
# =========================================================
DEVICE = "cuda:0"

# 已训练好的主模型权重
CHECKPOINT_PATH = "/mnt/data3/zhiyu/ForensicArtifacts/checkpoints/dino_baseline/checkpoint_epoch_20.pth"
USE_EMA = False

# 你可以两种模式二选一：
# 模式A：单个 JSON，内部自动切分 train/eval
USE_SINGLE_JSON_SPLIT = False
ALL_JSON_PATH = "./your_val_or_test.json"

# 模式B：手动给 train/eval 两个 JSON
TRAIN_JSON_PATH = "/mnt/data2/zhiyu/Data/small_openmmsec/train_medium.json"
EVAL_JSON_PATH = "/mnt/data2/zhiyu/Data/small_openmmsec/val_small.json"

# 保存目录
SAVE_DIR = "./analysis/probe_all_tasks"

# 模型结构配置，要和训练时一致
MODEL_CONFIG = {
    "model": {
        "repo_path": "/mnt/data3/zhiyu/dino_clip/dinov2_repo",
        "backbone_name": "dinov2_vitb14",
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

# DataLoader
BATCH_SIZE = 64
NUM_WORKERS = 4
PIN_MEMORY = True

# 特征类型：一次性全跑
FEATURE_TYPES = [
    "cls_token",
    "patch_mean",
    "cls_patch_mean",
]

# 一次性测试所有任务
# 只要数据里有这些字段，就会自动跑
ALL_TASKS = [
    "label",
    "domain",
    "source",
    "mani_type",
    "sub_mani_type",
    "real_source"
]

# 分三种条件：
# 全部样本 / 只 real / 只 fake
LABEL_CONDITIONS = ["all", "0", "1"]

# probe 小分类器
PROBE_MODEL = "ridge"   # "ridge" 或 "logreg"

# 单 json 自动切分比例
TEST_SIZE = 0.3

# 是否保存特征
SAVE_FEATURE_NPZ = False

SEED = 42


# =========================================================
# 基础函数
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


def load_model(device):
    model = ForensicDinoBaseline(MODEL_CONFIG).to(device)

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device,weights_only=False)

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


def build_dataset(json_path):
    dataset = ForensicImageDataset(
        json_path=json_path,
        image_size=MODEL_CONFIG["data"]["image_size"],
        mean=tuple(MODEL_CONFIG["data"]["mean"]),
        std=tuple(MODEL_CONFIG["data"]["std"]),
        is_train=False,
        target_domains=None,
        target_labels=None,
        target_mani_types=None,
        strict_mode=MODEL_CONFIG["data"].get("strict_mode", False)
    )
    return dataset


def build_loader(dataset):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )


# =========================================================
# 特征提取
# =========================================================
@torch.no_grad()
def extract_feature(model, images, feature_type):
    outputs = model(images)
    if not isinstance(outputs, (tuple, list)) or len(outputs) < 3:
        raise ValueError("当前模型 forward 返回值不是 (logits, cls_token, patch_tokens)")

    logits, cls_token, patch_tokens = outputs

    if feature_type == "cls_token":
        feat = cls_token
    elif feature_type == "patch_mean":
        feat = patch_tokens.mean(dim=1)
    elif feature_type == "cls_patch_mean":
        feat = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)
    else:
        raise ValueError(f"未知 feature_type: {feature_type}")

    return feat


@torch.no_grad()
def collect_features(model, loader, device, feature_type):
    all_X = []
    all_meta = []

    for batch in tqdm(loader, desc=f"Extract {feature_type}"):
        images = batch["image"].to(device, non_blocking=True)
        feat = extract_feature(model, images, feature_type)
        feat = feat.detach().cpu().numpy().astype(np.float32)
        all_X.append(feat)

        batch_size = len(batch["label"])

        labels = batch["label"].cpu().numpy().tolist()
        domains = list(batch["domain"]) if "domain" in batch else ["Unknown"] * batch_size
        paths = list(batch["path"]) if "path" in batch else ["UnknownPath"] * batch_size
        ori_datasets = list(batch["ori_dataset"]) if "ori_dataset" in batch else [None] * batch_size
        mani_types = list(batch["mani_type"]) if "mani_type" in batch else [None] * batch_size
        sub_mani_types = list(batch["sub_mani_type"]) if "sub_mani_type" in batch else [None] * batch_size
        real_sources = list(batch["real_source"]) if "real_source" in batch else [None] * batch_size

        for i in range(batch_size):
            all_meta.append({
                "label": int(labels[i]),
                "domain": str(domains[i]) if domains[i] is not None else "None",
                "source": str(ori_datasets[i]) if ori_datasets[i] is not None else "None",
                "mani_type": str(mani_types[i]) if mani_types[i] is not None else "None",
                "sub_mani_type": str(sub_mani_types[i]) if sub_mani_types[i] is not None else "None",
                "real_source": str(real_sources[i]) if real_sources[i] is not None else "None",
                "path": str(paths[i]),
            })

    all_X = np.concatenate(all_X, axis=0)
    return all_X, all_meta


# =========================================================
# probe 小分类器
# =========================================================
def build_probe_model():
    if PROBE_MODEL == "ridge":
        clf = RidgeClassifier(class_weight="balanced", random_state=SEED)
    elif PROBE_MODEL == "logreg":
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=SEED
        )
    else:
        raise ValueError(f"PROBE_MODEL 只能是 ridge/logreg，当前: {PROBE_MODEL}")

    return make_pipeline(StandardScaler(), clf)


def get_targets(meta, task_name):
    if task_name == "label":
        return [int(m["label"]) for m in meta]
    elif task_name == "domain":
        return [str(m["domain"]) for m in meta]
    elif task_name == "source":
        return [str(m["source"]) for m in meta]
    elif task_name == "mani_type":
        return [str(m["mani_type"]) for m in meta]
    elif task_name == "sub_mani_type":
        return [str(m["sub_mani_type"]) for m in meta]
    elif task_name == "real_source":
        return [str(m["real_source"]) for m in meta]
    else:
        raise ValueError(f"未知 task_name: {task_name}")


def filter_by_label(X, meta, label_condition):
    if label_condition == "all":
        return X, meta

    target_label = int(label_condition)
    indices = [i for i, m in enumerate(meta) if int(m["label"]) == target_label]
    X_new = X[indices]
    meta_new = [meta[i] for i in indices]
    return X_new, meta_new


def remove_invalid_classes(X, meta, task_name):
    """
    去掉 None / null / 空字符串 这种无效类别
    """
    y = get_targets(meta, task_name)
    keep_idx = []
    for i, v in enumerate(y):
        if v not in ["None", "null", "NULL", "", "nan", "NaN"]:
            keep_idx.append(i)

    X_new = X[keep_idx]
    meta_new = [meta[i] for i in keep_idx]
    return X_new, meta_new


def calc_metrics(y_true, y_pred):
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def majority_baseline(y_train, y_eval):
    values, counts = np.unique(y_train, return_counts=True)
    major = values[np.argmax(counts)]
    acc = float(np.mean(np.array(y_eval) == major))
    return acc, major


def run_probe(train_X, train_meta, eval_X, eval_meta, task_name, label_condition):
    # 1. 按真假筛
    train_X_use, train_meta_use = filter_by_label(train_X, train_meta, label_condition)
    eval_X_use, eval_meta_use = filter_by_label(eval_X, eval_meta, label_condition)

    # label 任务不做 only real / only fake
    if task_name == "label" and label_condition != "all":
        return None

    # 2. 去掉无效类别
    train_X_use, train_meta_use = remove_invalid_classes(train_X_use, train_meta_use, task_name)
    eval_X_use, eval_meta_use = remove_invalid_classes(eval_X_use, eval_meta_use, task_name)

    if len(train_meta_use) == 0 or len(eval_meta_use) == 0:
        return None

    y_train_raw = get_targets(train_meta_use, task_name)
    y_eval_raw = get_targets(eval_meta_use, task_name)

    # 如果类别只有1个，没法分类
    if len(set(y_train_raw)) < 2 or len(set(y_eval_raw)) < 2:
        return {
            "task": task_name,
            "label_condition": label_condition,
            "note": "类别数少于2，无法做分类",
            "num_train": len(y_train_raw),
            "num_eval": len(y_eval_raw),
        }

    # 3. 过滤评估集中训练没见过的类别
    seen_classes = set(y_train_raw)
    keep_idx = [i for i, y in enumerate(y_eval_raw) if y in seen_classes]
    unseen_num = len(y_eval_raw) - len(keep_idx)

    if len(keep_idx) == 0:
        return {
            "task": task_name,
            "label_condition": label_condition,
            "note": "评估集中没有训练出现过的类别，无法评估",
            "num_train": len(y_train_raw),
            "num_eval": len(y_eval_raw),
            "num_eval_seen": 0,
            "num_unseen_eval_class_samples": unseen_num,
        }

    eval_X_seen = eval_X_use[keep_idx]
    y_eval_seen_raw = [y_eval_raw[i] for i in keep_idx]

    # 4. 编码标签
    encoder = LabelEncoder()
    encoder.fit(y_train_raw)

    y_train = encoder.transform(y_train_raw)
    y_eval = encoder.transform(y_eval_seen_raw)

    # 5. 训练 probe
    clf = build_probe_model()
    clf.fit(train_X_use, y_train)

    pred_train = clf.predict(train_X_use)
    pred_eval = clf.predict(eval_X_seen)

    train_metrics = calc_metrics(y_train, pred_train)
    eval_metrics = calc_metrics(y_eval, pred_eval)

    maj_acc, maj_class = majority_baseline(y_train, y_eval)

    return {
        "task": task_name,
        "label_condition": label_condition,
        "num_train": len(y_train),
        "num_eval": len(y_eval_raw),
        "num_eval_seen": len(y_eval),
        "num_unseen_eval_class_samples": int(unseen_num),
        "num_classes_train": int(len(encoder.classes_)),
        "majority_eval_acc": float(maj_acc),
        "majority_class_id": str(maj_class),

        "train_acc": train_metrics["acc"],
        "train_balanced_acc": train_metrics["balanced_acc"],
        "train_macro_f1": train_metrics["macro_f1"],

        "eval_acc": eval_metrics["acc"],
        "eval_balanced_acc": eval_metrics["balanced_acc"],
        "eval_macro_f1": eval_metrics["macro_f1"],
    }


# =========================================================
# 单 json 自动切分
# =========================================================
def build_split_from_single_json():
    full_dataset = build_dataset(ALL_JSON_PATH)
    n = len(full_dataset)
    indices = np.arange(n)

    train_idx, eval_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=SEED,
        shuffle=True
    )

    train_dataset = Subset(full_dataset, train_idx.tolist())
    eval_dataset = Subset(full_dataset, eval_idx.tolist())

    return train_dataset, eval_dataset


# =========================================================
# 主程序
# =========================================================
def main():
    set_seed(SEED)
    ensure_dir(SAVE_DIR)
    device = to_device(DEVICE)

    if USE_SINGLE_JSON_SPLIT:
        print("[Info] 使用单个 JSON 自动切分 train/eval")
        train_dataset, eval_dataset = build_split_from_single_json()
    else:
        print("[Info] 使用手动指定 train/eval JSON")
        train_dataset = build_dataset(TRAIN_JSON_PATH)
        eval_dataset = build_dataset(EVAL_JSON_PATH)

    train_loader = build_loader(train_dataset)
    eval_loader = build_loader(eval_dataset)

    print(f"[Info] train样本数: {len(train_dataset)}")
    print(f"[Info] eval样本数 : {len(eval_dataset)}")

    model = load_model(device)

    all_results = []

    for feature_type in FEATURE_TYPES:
        print("\n" + "=" * 80)
        print(f"[Feature] {feature_type}")
        print("=" * 80)

        train_X, train_meta = collect_features(model, train_loader, device, feature_type)
        eval_X, eval_meta = collect_features(model, eval_loader, device, feature_type)

        print(f"[Info] train feature shape: {train_X.shape}")
        print(f"[Info] eval  feature shape: {eval_X.shape}")

        if SAVE_FEATURE_NPZ:
            np.savez_compressed(os.path.join(SAVE_DIR, f"{feature_type}_train_features.npz"), X=train_X)
            np.savez_compressed(os.path.join(SAVE_DIR, f"{feature_type}_eval_features.npz"), X=eval_X)

        for task_name in ALL_TASKS:
            if task_name == "label":
                conditions = ["all"]
            else:
                conditions = LABEL_CONDITIONS

            for cond in conditions:
                result = run_probe(
                    train_X=train_X,
                    train_meta=train_meta,
                    eval_X=eval_X,
                    eval_meta=eval_meta,
                    task_name=task_name,
                    label_condition=cond
                )

                if result is None:
                    continue

                result["feature_type"] = feature_type
                all_results.append(result)

                print(f"\n[Probe] feature={feature_type} | task={task_name} | label_condition={cond}")
                if "note" in result:
                    print(f"  note: {result['note']}")
                    continue

                print(f"  train_num={result['num_train']}  eval_num={result['num_eval']}  eval_seen={result['num_eval_seen']}")
                print(f"  unseen_eval_class_samples={result.get('num_unseen_eval_class_samples', 0)}")
                print(f"  num_classes_train={result.get('num_classes_train', 0)}")
                print(f"  majority_eval_acc={result.get('majority_eval_acc', 0):.4f}")
                print(f"  train_acc={result.get('train_acc', 0):.4f}  train_bal_acc={result.get('train_balanced_acc', 0):.4f}  train_f1={result.get('train_macro_f1', 0):.4f}")
                print(f"  eval_acc={result.get('eval_acc', 0):.4f}  eval_bal_acc={result.get('eval_balanced_acc', 0):.4f}  eval_f1={result.get('eval_macro_f1', 0):.4f}")

    csv_path = os.path.join(SAVE_DIR, "probe_all_results.csv")
    if len(all_results) > 0:
        fieldnames = sorted(list({k for row in all_results for k in row.keys()}))
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

    print(f"\n[Done] 所有 probe 结果已保存到: {csv_path}")


if __name__ == "__main__":
    main()