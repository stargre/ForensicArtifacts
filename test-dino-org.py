import os
import csv
import yaml
import argparse
import random
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from pre_data.dino_dataprocess import ForensicImageDataset, print_dataset_summary
from model.dino_orth import ForensicDinoOrth  # ←←← 关键修改：导入 Orth 模型


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="DINOv2 Forensic Detection Testing")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    return parser.parse_args()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def find_optimal_threshold(all_preds, all_labels):
    best_score = -1.0
    best_threshold = 0.5
    best_acc = 0.0
    best_f1 = 0.0

    for threshold in np.arange(0.20, 0.80, 0.01):
        pred_labels = (all_preds > threshold).astype(int)
        acc = np.mean(pred_labels == all_labels)

        tp = np.sum((pred_labels == 1) & (all_labels == 1))
        fp = np.sum((pred_labels == 1) & (all_labels == 0))
        fn = np.sum((pred_labels == 0) & (all_labels == 1))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        score = 0.5 * acc + 0.5 * f1
        if score > best_score:
            best_score = score
            best_acc = acc
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_acc, best_f1


def compute_all_metrics(all_preds, all_labels, threshold=0.5):
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, f1_score,
        precision_score, recall_score, confusion_matrix,
        average_precision_score, matthews_corrcoef,
        balanced_accuracy_score, cohen_kappa_score, log_loss
    )

    pred_labels = (all_preds > threshold).astype(int)
    cm = confusion_matrix(all_labels, pred_labels, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    if len(np.unique(all_labels)) < 2:
        auc_roc = 0.5
        auc_pr = 0.5
    else:
        auc_roc = roc_auc_score(all_labels, all_preds)
        auc_pr = average_precision_score(all_labels, all_preds)

    metrics = {
        "accuracy": accuracy_score(all_labels, pred_labels),
        "balanced_accuracy": balanced_accuracy_score(all_labels, pred_labels),
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "f1": f1_score(all_labels, pred_labels, zero_division=0),
        "precision": precision_score(all_labels, pred_labels, zero_division=0),
        "recall": recall_score(all_labels, pred_labels, zero_division=0),
        "specificity": tn / (tn + fp + 1e-8),
        "mcc": matthews_corrcoef(all_labels, pred_labels),
        "kappa": cohen_kappa_score(all_labels, pred_labels),
        "log_loss": log_loss(all_labels, np.clip(all_preds, 1e-7, 1 - 1e-7)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "threshold": threshold,
    }
    return metrics


def compute_domain_metrics(all_probs, all_labels, all_domains, threshold):
    from sklearn.metrics import roc_auc_score

    domain_stats = defaultdict(lambda: {"preds": [], "labels": []})
    for prob, label, domain in zip(all_probs, all_labels, all_domains):
        domain_stats[domain]["preds"].append(float(prob))
        domain_stats[domain]["labels"].append(float(label))

    domain_metrics = {}
    for domain, stats in sorted(domain_stats.items()):
        d_preds = np.array(stats["preds"])
        d_labels = np.array(stats["labels"])

        metrics = compute_all_metrics(d_preds, d_labels, threshold=threshold)
        if len(np.unique(d_labels)) < 2:
            metrics["auc_roc"] = 0.5
        else:
            metrics["auc_roc"] = roc_auc_score(d_labels, d_preds)

        domain_metrics[domain] = metrics

    return domain_metrics


def save_predictions_to_csv(save_path, all_paths, all_domains, all_labels, all_probs, threshold):
    save_dir = os.path.dirname(save_path)
    if save_dir != "":
        os.makedirs(save_dir, exist_ok=True)

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "domain", "label", "prob", "pred"])

        for path, domain, label, prob in zip(all_paths, all_domains, all_labels, all_probs):
            pred = 1 if prob > threshold else 0
            writer.writerow([path, domain, int(label), float(prob), pred])

    print(f"✓ 预测结果已保存到: {save_path}")


@torch.no_grad()
def test(model, dataloader, criterion, device, threshold=None, print_freq=20):
    model.eval()

    running_loss = 0.0
    all_probs = []
    all_labels = []
    all_paths = []
    all_domains = []

    pbar = tqdm(dataloader, desc="Testing")

    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].float().to(device, non_blocking=True).unsqueeze(1)
        domains = batch["domain"]
        paths = batch.get("path", ["unknown"] * labels.size(0))

        # 关键修改：调用 Orth 模型并取 forgery_logits
        outputs = model(images)
        logits = outputs["forgery_logits"]  # ←←← 正确获取伪造检测 logits

        loss = criterion(logits, labels)
        running_loss += loss.item()

        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        labels_np = labels.cpu().numpy().flatten()

        all_probs.extend(probs.tolist())
        all_labels.extend(labels_np.tolist())
        all_paths.extend(paths)
        all_domains.extend(domains)

        if (batch_idx + 1) % print_freq == 0:
            pbar.set_postfix({
                "processed": len(all_probs),
                "loss": f"{running_loss/(batch_idx+1):.4f}"
            })

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    test_loss = running_loss / max(1, len(dataloader))

    if threshold is None:
        threshold, _, _ = find_optimal_threshold(all_probs, all_labels)

    metrics = compute_all_metrics(all_probs, all_labels, threshold=threshold)
    metrics["loss"] = test_loss

    domain_metrics = compute_domain_metrics(all_probs, all_labels, all_domains, threshold)

    return metrics, domain_metrics, all_probs, all_labels, all_paths, all_domains


def main():
    args = parse_args()
    config = load_config(args.config)

    set_seed(config["system"].get("seed", 42))

    gpu_id = str(config.get("gpus", "0")).split(",")[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device(config["system"].get("device", "cuda"))

    print("\n" + "=" * 70)
    print(" DINOv2 Forensic Detection Testing ")
    print("=" * 70)
    print(f"使用GPU: {gpu_id}")
    print(f"Checkpoint: {config['checkpoint_path']}")

    use_ema_for_test = config.get("testing", {}).get("use_ema_for_test", False)
    print(f"使用EMA模型测试: {use_ema_for_test}")
    print("=" * 70)

    # ==================== Dataset ====================
    test_cfg = config["test_dataset"]
    test_dataset = ForensicImageDataset(
        json_path=test_cfg["path"],
        image_size=config["data"].get("image_size", 224),
        mean=tuple(config["data"].get("mean", [0.485, 0.456, 0.406])),
        std=tuple(config["data"].get("std", [0.229, 0.224, 0.225])),
        is_train=False,
        target_domains=test_cfg.get("target_domains"),
        target_labels=test_cfg.get("target_labels"),
        target_mani_types=test_cfg.get("target_mani_types"),
        strict_mode=config["data"].get("strict_mode", False)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["testing"].get("batch_size", 32),
        shuffle=False,
        num_workers=config["system"].get("num_workers", 8),
        pin_memory=config["system"].get("pin_memory", True)
    )

    print_dataset_summary(test_dataset, test_loader, name="Test")
    print(f"测试集样本数: {len(test_dataset)}")
    print(f"测试批次数: {len(test_loader)}")

    # ==================== Model ====================
    model = ForensicDinoOrth(config).to(device)  # ←←← 使用 Orth 模型

    checkpoint_path = config.get("checkpoint_path", None)
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint_path}")

    print(f"加载 checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_dict_to_load = None

    if use_ema_for_test and checkpoint.get("ema_model_state_dict") is not None:
        print("✓ 使用 EMA 模型权重 (key: 'ema_model_state_dict')")
        ema_state = checkpoint["ema_model_state_dict"]

        if isinstance(ema_state, dict) and "ema_state_dict" in ema_state:
            state_dict_to_load = ema_state["ema_state_dict"]
        else:
            state_dict_to_load = ema_state

    elif "model_state_dict" in checkpoint:
        print("✓ 使用 Student 模型权重 (key: 'model_state_dict')")
        state_dict_to_load = checkpoint["model_state_dict"]
    else:
        print("⚠ 未找到标准 key，尝试直接加载整个 checkpoint")
        state_dict_to_load = checkpoint

    model.load_state_dict(state_dict_to_load, strict=True)
    model.eval()
    print("✓ 模型加载成功")

    # ==================== Testing ====================
    criterion = nn.BCEWithLogitsLoss()
    test_threshold = config["testing"].get("threshold", None)
    print_freq = config.get("logging", {}).get("print_freq", 20)

    metrics, domain_metrics, all_probs, all_labels, all_paths, all_domains = test(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        threshold=test_threshold,
        print_freq=print_freq
    )

    # ==================== Print Results ====================
    print("\n" + "=" * 70)
    print(" Overall Test Metrics ")
    print("=" * 70)
    print(f"Loss               : {metrics['loss']:.4f}")
    print(f"Threshold          : {metrics['threshold']:.2f}")
    print(f"Accuracy           : {metrics['accuracy']*100:.2f}%")
    print(f"Balanced Accuracy  : {metrics['balanced_accuracy']*100:.2f}%")
    print(f"AUC-ROC            : {metrics['auc_roc']:.4f}")
    print(f"AUC-PR             : {metrics['auc_pr']:.4f}")
    print(f"F1 Score           : {metrics['f1']:.4f}")
    print(f"Precision          : {metrics['precision']:.4f}")
    print(f"Recall             : {metrics['recall']:.4f}")
    print(f"Specificity        : {metrics['specificity']:.4f}")
    print(f"MCC                : {metrics['mcc']:.4f}")
    print(f"Cohen Kappa        : {metrics['kappa']:.4f}")
    print(f"Log Loss           : {metrics['log_loss']:.4f}")

    print("\nConfusion Matrix:")
    print(f"  TN: {metrics['tn']}, FP: {metrics['fp']}")
    print(f"  FN: {metrics['fn']}, TP: {metrics['tp']}")

    print("\n" + "=" * 70)
    print(" Per-Domain Metrics ")
    print("=" * 70)
    print(f"{'Domain':<20} {'Samples':>8} {'ACC':>8} {'AUC':>8} {'F1':>8} {'MCC':>8}")
    print("-" * 70)

    for domain, dm in sorted(domain_metrics.items(), key=lambda x: x[0]):
        count = sum(1 for d in all_domains if d == domain)
        print(f"{domain:<20} {count:>8} {dm['accuracy']*100:>7.2f}% {dm['auc_roc']:>8.4f} {dm['f1']:>8.4f} {dm['mcc']:>8.4f}")

    print("=" * 70)

    # ==================== Save Predictions ====================
    if config["testing"].get("save_predictions", False):
        save_path = config["testing"].get("prediction_save_path", "./test_predictions.csv")
        save_predictions_to_csv(
            save_path=save_path,
            all_paths=all_paths,
            all_domains=all_domains,
            all_labels=all_labels,
            all_probs=all_probs,
            threshold=metrics["threshold"]
        )

    print("\n✅ 测试完成!")


if __name__ == "__main__":
    main()