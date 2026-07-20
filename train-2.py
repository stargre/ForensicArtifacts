import os
import yaml
import argparse
import random
import numpy as np
from collections import defaultdict
import copy
import time
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from pre_data.dino_dataprocess import ForensicImageDataset, print_dataset_summary
from model.dino_baseline import ForensicDinoBaseline
from model.lora_dino import apply_lora_to_forensic_dino


# =========================================================
# utils
# =========================================================
def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="DINOv2 Forensic Detection Training")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args()


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
        torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_broadcast_device(device):
    if device.type == "cuda":
        return device
    return None


def broadcast_object(obj, rank, is_distributed, device):
    if not is_distributed:
        return obj
    obj_list = [obj if rank == 0 else None]
    dist.broadcast_object_list(obj_list, src=0, device=get_broadcast_device(device))
    return obj_list[0]


def is_main_process(rank):
    return rank == 0


def get_model(model):
    return model.module if hasattr(model, "module") else model



# =========================================================
# validation-guided training control
# =========================================================
def build_val_control_score(val_result):
    """
    用于调度器、早停、rollback、best checkpoint 的目标域控制分数。

    当 evaluate_loader(..., report_mani_macro=True) 时，
    val_result["summary_metrics"] 已经是：
        mani_type -> domain -> overall macro-average
    所以这里默认取 macro AUC / macro F1，而不是 pooled 指标。
    """
    metrics = val_result.get("summary_metrics", val_result["metrics"])
    val_auc = float(metrics["auc_roc"])
    val_f1 = float(metrics["f1"])
    return 0.5 * val_auc + 0.5 * val_f1


def clone_state_to_cpu(obj):
    """
    递归 clone state_dict 到 CPU，避免 rollback snapshot 长期占 GPU 显存。
    """
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: clone_state_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clone_state_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(clone_state_to_cpu(v) for v in obj)
    return copy.deepcopy(obj)


def move_state_to_device(obj, device):
    """
    递归把 CPU snapshot 移回当前 device。
    """
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_state_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_state_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move_state_to_device(v, device) for v in obj)
    return obj


def make_training_snapshot(model, ema_model=None):
    """
    保存当前训练状态，用于 validation-guided rollback。
    这里不保存 optimizer state：目标是回滚参数后清空 AdamW 动量，
    避免动量继续沿 source-overfit 方向推进。
    """
    base_model = model.module if hasattr(model, "module") else model
    snapshot = {
        "model_state_dict": clone_state_to_cpu(base_model.state_dict()),
        "ema_state_dict": None,
    }
    if ema_model is not None:
        snapshot["ema_state_dict"] = clone_state_to_cpu(ema_model.state_dict())
    return snapshot


def restore_training_snapshot(
    model,
    snapshot,
    device,
    ema_model=None,
    optimizer=None,
    clear_optimizer_state=True,
):
    """
    回滚到 val_control_score 最优状态。
    默认清空 optimizer state，使下一轮从更小 LR + 无旧动量开始。
    """
    if snapshot is None:
        return

    base_model = model.module if hasattr(model, "module") else model
    model_state = move_state_to_device(snapshot["model_state_dict"], device)
    base_model.load_state_dict(model_state)

    if ema_model is not None and snapshot.get("ema_state_dict", None) is not None:
        ema_state = move_state_to_device(snapshot["ema_state_dict"], device)
        ema_model.load_state_dict(ema_state)

    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        if clear_optimizer_state:
            optimizer.state.clear()


def build_scheduler(optimizer, config):
    """
    支持两类 scheduler：
    1. plateau: val_control_score 不改善就降 LR；推荐用于跨域/目标域平台期问题。
    2. cosine: 保留原始 cosine 逻辑，便于对照实验。
    """
    sched_cfg = config["training"].get("scheduler", {})
    scheduler_type = sched_cfg.get("type", sched_cfg.get("name", "plateau")).lower()

    if scheduler_type in ["plateau", "reducelronplateau"]:
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=sched_cfg.get("factor", 0.5),
            patience=sched_cfg.get("patience", 0),
            threshold=sched_cfg.get("threshold", 5e-4),
            threshold_mode=sched_cfg.get("threshold_mode", "abs"),
            cooldown=sched_cfg.get("cooldown", 0),
            min_lr=sched_cfg.get("min_lr", sched_cfg.get("eta_min", 1e-6)),
        )

    if scheduler_type in ["cosine", "cosineannealinglr"]:
        total_epochs = config["training"]["epochs"]
        eta_min = sched_cfg.get("eta_min", 1e-6)
        flat_epochs = sched_cfg.get("flat_epochs", 0)

        if flat_epochs <= 0:
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_epochs, eta_min=eta_min
            )

        if flat_epochs >= total_epochs:
            return optim.lr_scheduler.ConstantLR(
                optimizer, factor=1.0, total_iters=total_epochs
            )

        return optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                optim.lr_scheduler.ConstantLR(
                    optimizer, factor=1.0, total_iters=flat_epochs
                ),
                optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=total_epochs - flat_epochs,
                    eta_min=eta_min,
                ),
            ],
            milestones=[flat_epochs],
        )

    raise ValueError(f"Unknown scheduler type/name: {scheduler_type}")

# =========================================================
# early stop
# =========================================================
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, monitor="val_auc", verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.verbose = verbose

        self.mode = "min" if "loss" in monitor else "max"
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.mode == "max":
            improved = score > (self.best_score + self.min_delta)
        else:
            improved = score < (self.best_score - self.min_delta)

        if improved:
            if self.verbose:
                print(f"  ↑ {self.monitor} 改善: {self.best_score:.4f} → {score:.4f}")
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"  → {self.monitor} 未改善 ({self.counter}/{self.patience})")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"  ⚠ 早停触发! 最佳 {self.monitor}: {self.best_score:.4f} @ epoch {self.best_epoch+1}")

        return self.early_stop


# =========================================================
# metrics
# =========================================================
def _safe_div(numerator, denominator, eps=1e-8):
    return float(numerator) / float(denominator + eps)


def _binary_stats_from_probs(all_probs, all_labels, threshold):
    all_probs = np.asarray(all_probs, dtype=float)
    all_labels = np.asarray(all_labels, dtype=int)

    pred_labels = (all_probs >= threshold).astype(int)

    tp = int(np.sum((pred_labels == 1) & (all_labels == 1)))
    tn = int(np.sum((pred_labels == 0) & (all_labels == 0)))
    fp = int(np.sum((pred_labels == 1) & (all_labels == 0)))
    fn = int(np.sum((pred_labels == 0) & (all_labels == 1)))

    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    recall = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    balanced_acc = 0.5 * (recall + specificity)

    precision = tp / (tp + fp + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "acc": acc,
        "balanced_acc": balanced_acc,
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
    }


def find_optimal_threshold(
    all_probs,
    all_labels,
    objective="balanced_acc",
    coarse_min=0.02,
    coarse_max=0.98,
    coarse_step=0.01,
    fine_radius=0.05,
    fine_step=0.001,
):
    """
    默认目标：最大化 Balanced ACC。

    注意：
    - 输入必须是 sigmoid 后的 probability，不是 raw logits。
    - 阈值判断统一使用 >=。
    - 先粗搜，再在最优点附近细搜。
    """

    all_probs = np.asarray(all_probs, dtype=float)
    all_labels = np.asarray(all_labels, dtype=int)

    best_thr = 0.5
    best_score = -1.0
    best_stats = None

    # 第一阶段：全范围粗搜
    coarse_thresholds = np.arange(coarse_min, coarse_max + 1e-12, coarse_step)

    for thr in coarse_thresholds:
        stats = _binary_stats_from_probs(all_probs, all_labels, thr)

        if objective == "balanced_acc":
            score = stats["balanced_acc"]
        elif objective == "accuracy":
            score = stats["acc"]
        elif objective == "f1":
            score = stats["f1"]
        elif objective == "youden":
            score = stats["recall"] + stats["specificity"] - 1.0
        else:
            raise ValueError(f"Unknown objective: {objective}")

        if score > best_score:
            best_score = score
            best_thr = float(thr)
            best_stats = stats

    # 第二阶段：局部细搜
    fine_min = max(0.001, best_thr - fine_radius)
    fine_max = min(0.999, best_thr + fine_radius)
    fine_thresholds = np.arange(fine_min, fine_max + 1e-12, fine_step)

    for thr in fine_thresholds:
        stats = _binary_stats_from_probs(all_probs, all_labels, thr)

        if objective == "balanced_acc":
            score = stats["balanced_acc"]
        elif objective == "accuracy":
            score = stats["acc"]
        elif objective == "f1":
            score = stats["f1"]
        elif objective == "youden":
            score = stats["recall"] + stats["specificity"] - 1.0
        else:
            raise ValueError(f"Unknown objective: {objective}")

        if score > best_score:
            best_score = score
            best_thr = float(thr)
            best_stats = stats

    return best_thr, best_score, best_stats

class TemperatureScaler(nn.Module):
    def __init__(self, init_temperature=1.0):
        super().__init__()
        self.log_temperature = nn.Parameter(
            torch.log(torch.ones(1) * init_temperature)
        )

    @property
    def temperature(self):
        return torch.exp(self.log_temperature).clamp(min=1e-6, max=100.0)

    def forward(self, logits):
        return logits / self.temperature

    def fit(self, val_logits, val_labels, device="cuda", max_iter=50):
        self.to(device)

        val_logits = torch.as_tensor(
            val_logits, dtype=torch.float32, device=device
        ).view(-1)
        val_labels = torch.as_tensor(
            val_labels, dtype=torch.float32, device=device
        ).view(-1)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.LBFGS(
            [self.log_temperature],
            lr=0.01,
            max_iter=max_iter,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            scaled_logits = self.forward(val_logits)
            loss = criterion(scaled_logits, val_labels)
            loss.backward()
            return loss

        before_loss = criterion(val_logits, val_labels).item()
        optimizer.step(closure)

        with torch.no_grad():
            scaled_logits = self.forward(val_logits)
            after_loss = criterion(scaled_logits, val_labels).item()

        print(
            f"[Temperature Scaling] "
            f"T={self.temperature.item():.4f}, "
            f"val_BCE_before={before_loss:.6f}, "
            f"val_BCE_after={after_loss:.6f}"
        )

        return self

    @torch.no_grad()
    def transform_logits(self, logits, device="cuda"):
        self.to(device)
        logits = torch.as_tensor(
            logits, dtype=torch.float32, device=device
        ).view(-1)
        scaled_logits = self.forward(logits)
        return scaled_logits.detach().cpu().numpy()

    @torch.no_grad()
    def transform_probs(self, logits, device="cuda"):
        scaled_logits = self.transform_logits(logits, device=device)
        scaled_logits = np.clip(scaled_logits, -50, 50)
        probs = 1.0 / (1.0 + np.exp(-scaled_logits))
        return probs


@torch.no_grad()
@torch.no_grad()
def collect_logits_labels(
    model,
    dataloader,
    criterion,
    device,
    epoch=0,
    split_name="Collect",
    collect_mani_type=False,
):
    """
    Collect logits / labels / domains, and optionally mani_type.

    Important:
    - Train/Val/Test pooled metrics only need logits/labels/domains.
    - Val/Test hierarchical macro metrics need mani_type:
      mani_type -> domain -> overall macro-average.
    """
    model.eval()

    running_loss = 0.0
    total_samples = 0

    all_logits = []
    all_labels = []
    all_domains = []
    all_mani_types = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [{split_name}]")

    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].float().to(device, non_blocking=True).unsqueeze(1)

        domains = _to_list(batch["domain"])
        mani_types = get_batch_mani_types(batch) if collect_mani_type else None

        logits, _, _ = model_forward(model, images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        all_logits.extend(logits.detach().cpu().numpy().flatten().tolist())
        all_labels.extend(labels.detach().cpu().numpy().flatten().tolist())
        all_domains.extend(domains)

        if collect_mani_type:
            all_mani_types.extend(mani_types)

    avg_loss = running_loss / max(1, total_samples)

    out = {
        "loss": avg_loss,
        "logits": np.array(all_logits, dtype=float),
        "labels": np.array(all_labels, dtype=int),
        "domains": all_domains,
    }

    if collect_mani_type:
        out["mani_types"] = all_mani_types

    return out

def build_domain_stats_from_arrays(probs, labels, domains):
    domain_stats = defaultdict(lambda: {"preds": [], "labels": []})

    for p, y, d in zip(probs, labels, domains):
        domain_stats[d]["preds"].append(float(p))
        domain_stats[d]["labels"].append(float(y))

    return domain_stats

def evaluate_with_temperature_and_thresholds(
    model,
    val_loader,
    test_loader,
    criterion,
    device,
    epoch=0,
    split_name="Student",
    use_temperature_scaling=True,
):
    """
    Final test evaluation.

    Val is evaluated by the same 14-mani_type hierarchical protocol:
        mani_type -> domain -> overall macro-average.

    The selected validation threshold is the threshold that maximizes Val
    hierarchical macro Balanced Accuracy, not pooled Balanced Accuracy.
    """
    # 1. 收集 val / test 的 raw logits
    # Val 和 Test 都收集 mani_type，因为二者都要按同一套层级宏平均协议计算。
    val_out = collect_logits_labels(
        model=model,
        dataloader=val_loader,
        criterion=criterion,
        device=device,
        epoch=epoch,
        split_name=f"{split_name}-Val-Collect",
        collect_mani_type=True,
    )

    test_out = collect_logits_labels(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        epoch=epoch,
        split_name=f"{split_name}-Test-Collect",
        collect_mani_type=True,
    )

    val_logits = val_out["logits"]
    val_labels = val_out["labels"]
    val_domains = val_out["domains"]
    val_mani_types = val_out["mani_types"]

    test_logits = test_out["logits"]
    test_labels = test_out["labels"]
    test_domains = test_out["domains"]
    test_mani_types = test_out["mani_types"]

    # 2. Temperature Scaling：只在 val 上拟合
    if use_temperature_scaling:
        scaler = TemperatureScaler()
        scaler.fit(val_logits, val_labels, device=device)

        val_probs = scaler.transform_probs(val_logits, device=device)
        test_probs = scaler.transform_probs(test_logits, device=device)
    else:
        scaler = None
        val_logits_clip = np.clip(val_logits, -50, 50)
        test_logits_clip = np.clip(test_logits, -50, 50)

        val_probs = 1.0 / (1.0 + np.exp(-val_logits_clip))
        test_probs = 1.0 / (1.0 + np.exp(-test_logits_clip))

    # 3. 在 val 上按 mani_type -> domain -> overall macro 协议搜索 Balanced ACC 最优阈值
    val_best_thr, val_best_bal_acc, _ = find_optimal_threshold_for_manitype_macro(
        probs=val_probs,
        labels=val_labels,
        domains=val_domains,
        mani_types=val_mani_types,
        objective="balanced_acc",
        coarse_min=0.02,
        coarse_max=0.98,
        coarse_step=0.01,
        fine_radius=0.05,
        fine_step=0.001,
    )

    # 4. val pooled + val hierarchical macro 结果
    val_metrics = compute_all_metrics(
        val_probs,
        val_labels,
        threshold=val_best_thr,
    )

    val_domain_stats = build_domain_stats_from_arrays(
        val_probs,
        val_labels,
        val_domains,
    )
    val_domain_metrics = compute_domain_metrics(
        val_domain_stats,
        threshold=val_best_thr,
    )

    val_mani_macro = compute_manitype_domain_macro_metrics(
        probs=val_probs,
        labels=val_labels,
        domains=val_domains,
        mani_types=val_mani_types,
        threshold=val_best_thr,
    )

    # 5. test 使用 val macro 最佳阈值
    test_metrics_with_val_thr = compute_all_metrics(
        test_probs,
        test_labels,
        threshold=val_best_thr,
    )

    test_domain_stats = build_domain_stats_from_arrays(
        test_probs,
        test_labels,
        test_domains,
    )
    test_domain_metrics_with_val_thr = compute_domain_metrics(
        test_domain_stats,
        threshold=val_best_thr,
    )

    test_mani_macro_with_val_thr = compute_manitype_domain_macro_metrics(
        probs=test_probs,
        labels=test_labels,
        domains=test_domains,
        mani_types=test_mani_types,
        threshold=val_best_thr,
    )

    # 6. test 自己按 hierarchical macro 搜索最佳阈值：oracle / upper bound，仅用于参考
    test_best_thr, test_best_bal_acc, _ = find_optimal_threshold_for_manitype_macro(
        probs=test_probs,
        labels=test_labels,
        domains=test_domains,
        mani_types=test_mani_types,
        objective="balanced_acc",
        coarse_min=0.02,
        coarse_max=0.98,
        coarse_step=0.01,
        fine_radius=0.05,
        fine_step=0.001,
    )

    test_metrics_with_test_thr = compute_all_metrics(
        test_probs,
        test_labels,
        threshold=test_best_thr,
    )

    test_domain_metrics_with_test_thr = compute_domain_metrics(
        test_domain_stats,
        threshold=test_best_thr,
    )

    test_mani_macro_with_test_thr = compute_manitype_domain_macro_metrics(
        probs=test_probs,
        labels=test_labels,
        domains=test_domains,
        mani_types=test_mani_types,
        threshold=test_best_thr,
    )

    # 7. 打印结果
    print("\n" + "=" * 80)
    print(f"[{split_name}] Validation best threshold selected by ManiType-Domain-Macro Balanced ACC")
    print("=" * 80)
    print(f"val_best_thr               : {val_best_thr:.4f}")
    print(f"val_macro_balanced_acc     : {val_mani_macro['overall']['balanced_accuracy']*100:.2f}%")
    print(f"val_macro_acc              : {val_mani_macro['overall']['accuracy']*100:.2f}%")
    print(f"val_macro_auc              : {val_mani_macro['overall']['auc_roc']:.4f}")
    print(f"val_macro_ap               : {val_mani_macro['overall']['ap']:.4f}")
    print(f"val_macro_f1               : {val_mani_macro['overall']['f1']:.4f}")
    print(f"val_pooled_acc_debug       : {val_metrics['accuracy']*100:.2f}%")
    print(f"val_pooled_auc_debug       : {val_metrics['auc_roc']:.4f}")
    print(f"val_pooled_ap_debug        : {val_metrics['ap']:.4f}")
    print(f"val_pooled_f1_debug        : {val_metrics['f1']:.4f}")

    print_full_metrics(
        val_metrics,
        title=f"[{split_name}] Val Pooled Metrics with Val-Macro-Best Threshold",
        loss=val_out["loss"],
    )
    print_domain_metrics(
        val_domain_metrics,
        title=f"[{split_name}] Val Pooled Per-Domain Metrics with Val-Macro-Best Threshold",
    )
    print_manitype_domain_macro_metrics(
        val_mani_macro,
        title=f"[{split_name}] Val ManiType -> Domain -> Overall Macro with Val-Macro-Best Threshold",
    )

    print("\n" + "=" * 80)
    print(f"[{split_name}] Test using VAL macro best threshold")
    print("=" * 80)
    print(f"threshold                  : {val_best_thr:.4f}")
    print(f"test_macro_acc             : {test_mani_macro_with_val_thr['overall']['accuracy']*100:.2f}%")
    print(f"test_macro_auc             : {test_mani_macro_with_val_thr['overall']['auc_roc']:.4f}")
    print(f"test_macro_ap              : {test_mani_macro_with_val_thr['overall']['ap']:.4f}")
    print(f"test_macro_f1              : {test_mani_macro_with_val_thr['overall']['f1']:.4f}")
    print(f"test_pooled_acc_debug      : {test_metrics_with_val_thr['accuracy']*100:.2f}%")
    print(f"test_pooled_auc_debug      : {test_metrics_with_val_thr['auc_roc']:.4f}")
    print(f"test_pooled_ap_debug       : {test_metrics_with_val_thr['ap']:.4f}")
    print(f"test_pooled_f1_debug       : {test_metrics_with_val_thr['f1']:.4f}")
    print(
        f"TP/TN/FP/FN pooled debug   : "
        f"{test_metrics_with_val_thr['tp']}/"
        f"{test_metrics_with_val_thr['tn']}/"
        f"{test_metrics_with_val_thr['fp']}/"
        f"{test_metrics_with_val_thr['fn']}"
    )

    print_full_metrics(
        test_metrics_with_val_thr,
        title=f"[{split_name}] Test Pooled Metrics with VAL Macro Threshold",
        loss=test_out["loss"],
    )
    print_domain_metrics(
        test_domain_metrics_with_val_thr,
        title=f"[{split_name}] Test Pooled Per-Domain Metrics with VAL Macro Threshold",
    )
    print_manitype_domain_macro_metrics(
        test_mani_macro_with_val_thr,
        title=f"[{split_name}] Test ManiType -> Domain -> Overall Macro with VAL Macro Threshold",
    )

    print("\n" + "=" * 80)
    print(f"[{split_name}] Test searching TEST macro best threshold -- oracle / upper bound")
    print("=" * 80)
    print(f"test_best_thr              : {test_best_thr:.4f}")
    print(f"test_macro_acc_oracle      : {test_mani_macro_with_test_thr['overall']['accuracy']*100:.2f}%")
    print(f"test_macro_auc_oracle      : {test_mani_macro_with_test_thr['overall']['auc_roc']:.4f}")
    print(f"test_macro_ap_oracle       : {test_mani_macro_with_test_thr['overall']['ap']:.4f}")
    print(f"test_macro_f1_oracle       : {test_mani_macro_with_test_thr['overall']['f1']:.4f}")
    print(f"test_pooled_acc_debug      : {test_metrics_with_test_thr['accuracy']*100:.2f}%")
    print(f"test_pooled_auc_debug      : {test_metrics_with_test_thr['auc_roc']:.4f}")
    print(f"test_pooled_ap_debug       : {test_metrics_with_test_thr['ap']:.4f}")
    print(f"test_pooled_f1_debug       : {test_metrics_with_test_thr['f1']:.4f}")
    print(
        f"TP/TN/FP/FN pooled debug   : "
        f"{test_metrics_with_test_thr['tp']}/"
        f"{test_metrics_with_test_thr['tn']}/"
        f"{test_metrics_with_test_thr['fp']}/"
        f"{test_metrics_with_test_thr['fn']}"
    )

    print_full_metrics(
        test_metrics_with_test_thr,
        title=f"[{split_name}] Test Pooled Metrics with TEST Macro-Best Threshold -- Oracle",
        loss=test_out["loss"],
    )
    print_domain_metrics(
        test_domain_metrics_with_test_thr,
        title=f"[{split_name}] Test Pooled Per-Domain Metrics with TEST Macro-Best Threshold -- Oracle",
    )
    print_manitype_domain_macro_metrics(
        test_mani_macro_with_test_thr,
        title=f"[{split_name}] Test ManiType -> Domain -> Overall Macro with TEST Macro-Best Threshold -- Oracle",
    )

    return {
        "temperature_scaler": scaler,
        "use_temperature_scaling": use_temperature_scaling,

        "val_best_thr": val_best_thr,
        "test_best_thr": test_best_thr,

        "val_loss": val_out["loss"],
        "test_loss": test_out["loss"],

        "val_metrics": val_metrics,
        "val_domain_metrics": val_domain_metrics,
        "val_mani_macro": val_mani_macro,

        "test_metrics_with_val_thr": test_metrics_with_val_thr,
        "test_domain_metrics_with_val_thr": test_domain_metrics_with_val_thr,
        "test_mani_macro_with_val_thr": test_mani_macro_with_val_thr,

        "test_metrics_with_test_thr": test_metrics_with_test_thr,
        "test_domain_metrics_with_test_thr": test_domain_metrics_with_test_thr,
        "test_mani_macro_with_test_thr": test_mani_macro_with_test_thr,
    }

def compute_all_metrics(all_preds, all_labels, threshold=0.5):
    """
    统一计算二分类常用指标。

    输出说明：
    - AP / AUC-PR: average_precision_score，越高越好，尤其适合类别不均衡场景
    - AUC-ROC: 阈值无关指标
    - ACC / F1 / Precision / Recall / Specificity 等：阈值相关指标
    """
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, f1_score,
        precision_score, recall_score, confusion_matrix,
        average_precision_score, matthews_corrcoef,
        balanced_accuracy_score, cohen_kappa_score,
        log_loss, brier_score_loss
    )

    all_preds = np.asarray(all_preds).astype(float)
    all_labels = np.asarray(all_labels).astype(int)

    clipped_preds = np.clip(all_preds, 1e-7, 1.0 - 1e-7)
    pred_labels = (all_preds >= threshold).astype(int)

    cm = confusion_matrix(all_labels, pred_labels, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    n = len(all_labels)
    positives = int(np.sum(all_labels == 1))
    negatives = int(np.sum(all_labels == 0))
    pred_positives = int(np.sum(pred_labels == 1))
    pred_negatives = int(np.sum(pred_labels == 0))

    if n == 0:
        raise ValueError("compute_all_metrics received empty predictions/labels.")

    if len(np.unique(all_labels)) < 2:
        auc_roc = 0.5
        ap = float(positives / max(1, n))
        ll = 0.0
    else:
        auc_roc = roc_auc_score(all_labels, all_preds)
        ap = average_precision_score(all_labels, all_preds)
        ll = log_loss(all_labels, clipped_preds, labels=[0, 1])

    precision = precision_score(all_labels, pred_labels, zero_division=0)
    recall = recall_score(all_labels, pred_labels, zero_division=0)
    specificity = _safe_div(tn, tn + fp)
    npv = _safe_div(tn, tn + fn)

    fpr = _safe_div(fp, fp + tn)
    fnr = _safe_div(fn, fn + tp)
    fdr = _safe_div(fp, fp + tp)
    false_omission_rate = _safe_div(fn, fn + tn)

    metrics = {
        # basic counts
        "samples": int(n),
        "positive": positives,
        "negative": negatives,
        "pred_positive": pred_positives,
        "pred_negative": pred_negatives,
        "threshold": float(threshold),

        # threshold-independent ranking/probability metrics
        "auc_roc": float(auc_roc),
        "auc_pr": float(ap),
        "ap": float(ap),
        "log_loss": float(ll),
        "brier": float(brier_score_loss(all_labels, clipped_preds)),

        # threshold-dependent classification metrics
        "accuracy": float(accuracy_score(all_labels, pred_labels)),
        "balanced_accuracy": float(balanced_accuracy_score(all_labels, pred_labels)),
        "f1": float(f1_score(all_labels, pred_labels, zero_division=0)),
        "f1_macro": float(f1_score(all_labels, pred_labels, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(all_labels, pred_labels, average="weighted", zero_division=0)),
        "precision": float(precision),
        "recall": float(recall),
        "sensitivity": float(recall),
        "specificity": float(specificity),
        "npv": float(npv),

        # correlation/agreement metrics
        "mcc": float(matthews_corrcoef(all_labels, pred_labels)),
        "kappa": float(cohen_kappa_score(all_labels, pred_labels)),

        # error-rate diagnostics
        "fpr": float(fpr),
        "fnr": float(fnr),
        "fdr": float(fdr),
        "for": float(false_omission_rate),
        "youden_j": float(recall + specificity - 1.0),
        "prevalence": float(positives / max(1, n)),
        "pred_positive_rate": float(pred_positives / max(1, n)),

        # confusion matrix
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }
    return metrics


def compute_domain_metrics(domain_stats, threshold=0.5):
    """
    每个 domain 也输出完整常用指标，而不是只输出 AUC。
    若某个 domain 只有单类样本，AUC-ROC 置为 0.5，AP 置为该 domain 的正样本比例。
    """
    domain_metrics = {}
    for d, stats in sorted(domain_stats.items()):
        d_preds = np.array(stats["preds"])
        d_labels = np.array(stats["labels"])
        if len(d_labels) == 0:
            continue
        domain_metrics[d] = compute_all_metrics(d_preds, d_labels, threshold=threshold)
    return domain_metrics

# =========================================================
# ManiType -> Domain -> Overall macro metrics for TEST
# =========================================================
def _to_list(x):
    """
    把 batch 里的字段统一转成 Python list。
    兼容 list / tuple / torch.Tensor / numpy.ndarray。
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def get_batch_mani_types(batch):
    """
    兼容不同 dataset 里 mani_type 的键名。
    你的代码 config 里叫 target_mani_types，所以最可能是 mani_type 或 manitype。
    """
    candidate_keys = [
        "mani_type",
        "manitype",
        "mani_types",
        "manipulation_type",
        "manipulation_types",
    ]

    for k in candidate_keys:
        if k in batch:
            return _to_list(batch[k])

    raise KeyError(
        "Cannot find mani_type in batch. "
        "Please check your ForensicImageDataset.__getitem__ return dict. "
        "Expected one of: " + ", ".join(candidate_keys)
    )


def _build_manitype_eval_groups(
    probs,
    labels,
    domains,
    mani_types,
    real_mani_names=None,
):
    """
    Build per-mani_type binary evaluation groups.

    For each fake mani_type:
        evaluation group = fake samples of this mani_type + real samples from the same domain.

    This handles the common data format where real samples do not have a specific fake mani_type.
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    domains = [str(x) for x in domains]
    mani_types = [str(x) for x in mani_types]

    if real_mani_names is None:
        real_mani_names = {
            "real", "Real", "REAL",
            "authentic", "Authentic",
            "original", "Original",
            "none", "None", "NONE",
            "clean", "Clean",
            "0",
        }

    if not (len(probs) == len(labels) == len(domains) == len(mani_types)):
        raise ValueError(
            f"Length mismatch: "
            f"probs={len(probs)}, labels={len(labels)}, "
            f"domains={len(domains)}, mani_types={len(mani_types)}"
        )

    real_pool_by_domain = defaultdict(lambda: {"preds": [], "labels": []})
    real_group_by_manitype = defaultdict(lambda: {"preds": [], "labels": []})
    fake_group_by_manitype = defaultdict(lambda: {"preds": [], "labels": []})

    for p, y, d, m in zip(probs, labels, domains, mani_types):
        y = int(y)
        p = float(p)

        if y == 0:
            real_pool_by_domain[d]["preds"].append(p)
            real_pool_by_domain[d]["labels"].append(y)

            # If real samples are explicitly assigned to one of the 14 mani_types,
            # use those real samples preferentially for the corresponding mani_type.
            if m not in real_mani_names:
                real_group_by_manitype[(d, m)]["preds"].append(p)
                real_group_by_manitype[(d, m)]["labels"].append(y)
        else:
            fake_group_by_manitype[(d, m)]["preds"].append(p)
            fake_group_by_manitype[(d, m)]["labels"].append(y)

    eval_groups = {}

    for key, fake_stats in sorted(fake_group_by_manitype.items()):
        d, m = key

        fake_preds = fake_stats["preds"]
        fake_labels = fake_stats["labels"]

        if key in real_group_by_manitype and len(real_group_by_manitype[key]["labels"]) > 0:
            real_preds = real_group_by_manitype[key]["preds"]
            real_labels = real_group_by_manitype[key]["labels"]
            real_source = "same_mani_type"
        else:
            real_preds = real_pool_by_domain[d]["preds"]
            real_labels = real_pool_by_domain[d]["labels"]
            real_source = "same_domain"

        if len(real_labels) == 0:
            raise ValueError(
                f"Cannot compute mani_type metrics for {key}: "
                f"found fake samples N={len(fake_labels)}, but no real samples in domain={d}. "
                "Please make sure this split contains real samples for each domain."
            )

        g_probs = np.asarray(real_preds + fake_preds, dtype=float)
        g_labels = np.asarray(real_labels + fake_labels, dtype=int)

        if len(np.unique(g_labels)) < 2:
            raise ValueError(
                f"ManiType group {key} still has only one class after real matching: "
                f"labels={np.unique(g_labels).tolist()}, N={len(g_labels)}."
            )

        eval_groups[key] = {
            "preds": g_probs,
            "labels": g_labels,
            "real_samples": int(len(real_labels)),
            "fake_samples": int(len(fake_labels)),
            "real_source": real_source,
        }

    if len(eval_groups) == 0:
        raise ValueError(
            "No fake mani_type groups were found. "
            "Please check labels and mani_type fields in this split."
        )

    return eval_groups


def _compute_manitype_domain_macro_objective_from_groups(
    eval_groups,
    threshold,
    objective="balanced_acc",
):
    """
    Compute only the macro objective for threshold search.
    This avoids recomputing AUC/AP hundreds of times.
    """
    domain_values = defaultdict(list)

    for (d, m), group in eval_groups.items():
        stats = _binary_stats_from_probs(group["preds"], group["labels"], threshold)

        if objective == "balanced_acc":
            score = stats["balanced_acc"]
        elif objective == "accuracy":
            score = stats["acc"]
        elif objective == "f1":
            score = stats["f1"]
        elif objective == "youden":
            score = stats["recall"] + stats["specificity"] - 1.0
        else:
            raise ValueError(f"Unknown objective: {objective}")

        domain_values[d].append(float(score))

    domain_scores = [
        float(np.mean(vals))
        for _, vals in sorted(domain_values.items())
        if len(vals) > 0
    ]

    if len(domain_scores) == 0:
        raise ValueError("No domain scores available for mani_type macro threshold search.")

    return float(np.mean(domain_scores))


def find_optimal_threshold_for_manitype_macro(
    probs,
    labels,
    domains,
    mani_types,
    objective="balanced_acc",
    coarse_min=0.02,
    coarse_max=0.98,
    coarse_step=0.01,
    fine_radius=0.05,
    fine_step=0.001,
):
    """
    Select threshold by maximizing the same hierarchical macro protocol:
        mani_type -> domain -> overall.

    This is used for Val when Val is also evaluated by the 14-mani_type protocol.
    """
    eval_groups = _build_manitype_eval_groups(
        probs=probs,
        labels=labels,
        domains=domains,
        mani_types=mani_types,
    )

    best_thr = 0.5
    best_score = -1.0

    coarse_thresholds = np.arange(coarse_min, coarse_max + 1e-12, coarse_step)

    for thr in coarse_thresholds:
        score = _compute_manitype_domain_macro_objective_from_groups(
            eval_groups,
            threshold=float(thr),
            objective=objective,
        )
        if score > best_score:
            best_score = float(score)
            best_thr = float(thr)

    fine_min = max(0.001, best_thr - fine_radius)
    fine_max = min(0.999, best_thr + fine_radius)
    fine_thresholds = np.arange(fine_min, fine_max + 1e-12, fine_step)

    for thr in fine_thresholds:
        score = _compute_manitype_domain_macro_objective_from_groups(
            eval_groups,
            threshold=float(thr),
            objective=objective,
        )
        if score > best_score:
            best_score = float(score)
            best_thr = float(thr)

    return best_thr, best_score, None


def compute_manitype_domain_macro_metrics(
    probs,
    labels,
    domains,
    mani_types,
    threshold=0.5,
    real_mani_names=None,
):
    """
    Hierarchical macro metrics for Val/Test:
        1. For each fake mani_type, combine:
             fake samples of this mani_type + real samples from the same domain.
        2. Compute metrics for every mani_type.
        3. Average mani_type metrics inside each domain, not sample-weighted.
        4. Average domain metrics for final overall, not sample-weighted.

    This supports the case where real samples do not have a specific fake mani_type.
    """
    eval_groups = _build_manitype_eval_groups(
        probs=probs,
        labels=labels,
        domains=domains,
        mani_types=mani_types,
        real_mani_names=real_mani_names,
    )

    per_mani_type = {}

    for key, group in sorted(eval_groups.items()):
        metrics = compute_all_metrics(
            group["preds"],
            group["labels"],
            threshold=threshold,
        )
        metrics["fake_samples"] = int(group["fake_samples"])
        metrics["real_samples"] = int(group["real_samples"])
        metrics["real_source"] = group["real_source"]
        per_mani_type[key] = metrics

    domain_to_keys = defaultdict(list)
    for key in per_mani_type.keys():
        d, _ = key
        domain_to_keys[d].append(key)

    macro_metric_keys = [
        "accuracy",
        "balanced_accuracy",
        "auc_roc",
        "auc_pr",
        "ap",
        "f1",
        "f1_macro",
        "f1_weighted",
        "precision",
        "recall",
        "specificity",
        "mcc",
        "kappa",
        "log_loss",
        "brier",
    ]

    domain_avg = {}

    for d in sorted(domain_to_keys.keys()):
        keys = sorted(domain_to_keys[d], key=lambda x: x[1])

        domain_avg[d] = {
            "samples": int(sum(per_mani_type[k]["samples"] for k in keys)),
            "fake_samples": int(sum(per_mani_type[k].get("fake_samples", 0) for k in keys)),
            "real_samples_sum": int(sum(per_mani_type[k].get("real_samples", 0) for k in keys)),
            "num_mani_types": int(len(keys)),
            "threshold": float(threshold),
            "mani_types": [k[1] for k in keys],
        }

        for metric_name in macro_metric_keys:
            vals = [
                per_mani_type[k][metric_name]
                for k in keys
                if metric_name in per_mani_type[k]
            ]
            domain_avg[d][metric_name] = float(np.mean(vals)) if len(vals) > 0 else float("nan")

    overall = {
        "samples": int(len(labels)),
        "num_domains": int(len(domain_avg)),
        "num_mani_types": int(len(per_mani_type)),
        "threshold": float(threshold),
    }

    for metric_name in macro_metric_keys:
        vals = [
            domain_avg[d][metric_name]
            for d in sorted(domain_avg.keys())
            if metric_name in domain_avg[d]
        ]
        overall[metric_name] = float(np.mean(vals)) if len(vals) > 0 else float("nan")

    return {
        "per_mani_type": per_mani_type,
        "domain_avg": domain_avg,
        "overall": overall,
    }

def print_manitype_domain_macro_metrics(
    result,
    title="[ManiType -> Domain -> Overall Macro Metrics]",
):
    """
    打印：
        1. 每个 mani_type 的指标
        2. 每个 domain 的 mani_type macro average
        3. overall domain macro average
    """
    if result is None:
        return

    per_mani_type = result["per_mani_type"]
    domain_avg = result["domain_avg"]
    overall = result["overall"]

    print("\n" + "=" * 130)
    print(title)
    print("=" * 130)

    print("\n[Per ManiType Metrics]")
    header = (
        f"{'Domain':<16} "
        f"{'ManiType':<24} "
        f"{'N':>7} "
        f"{'Real':>7} "
        f"{'Fake':>7} "
        f"{'ACC':>8} "
        f"{'AUC':>8} "
        f"{'AP':>8} "
        f"{'F1':>8} "
        f"{'P':>8} "
        f"{'R':>8} "
        f"{'Spec':>8} "
        f"{'MCC':>8} "
        f"{'Thr':>7}"
    )
    print(header)
    print("-" * len(header))

    for (d, m), metrics in sorted(per_mani_type.items(), key=lambda x: (x[0][0], x[0][1])):
        print(
            f"{d:<16} "
            f"{m:<24} "
            f"{metrics['samples']:>7d} "
            f"{metrics.get('real_samples', 0):>7d} "
            f"{metrics.get('fake_samples', 0):>7d} "
            f"{metrics['accuracy']*100:>7.2f}% "
            f"{metrics['auc_roc']:>8.4f} "
            f"{metrics['ap']:>8.4f} "
            f"{metrics['f1']:>8.4f} "
            f"{metrics['precision']:>8.4f} "
            f"{metrics['recall']:>8.4f} "
            f"{metrics['specificity']:>8.4f} "
            f"{metrics['mcc']:>8.4f} "
            f"{metrics['threshold']:>7.4f}"
        )

    print("\n[Domain Average: mean over ManiTypes, NOT sample-weighted]")
    header = (
        f"{'Domain':<16} "
        f"{'#Types':>7} "
        f"{'N(sum)':>8} "
        f"{'RealSum':>8} "
        f"{'FakeSum':>8} "
        f"{'ACC':>8} "
        f"{'AUC':>8} "
        f"{'AP':>8} "
        f"{'F1':>8} "
        f"{'P':>8} "
        f"{'R':>8} "
        f"{'Spec':>8} "
        f"{'MCC':>8}"
    )
    print(header)
    print("-" * len(header))

    for d, metrics in sorted(domain_avg.items()):
        print(
            f"{d:<16} "
            f"{metrics['num_mani_types']:>7d} "
            f"{metrics['samples']:>8d} "
            f"{metrics.get('real_samples_sum', 0):>8d} "
            f"{metrics.get('fake_samples', 0):>8d} "
            f"{metrics['accuracy']*100:>7.2f}% "
            f"{metrics['auc_roc']:>8.4f} "
            f"{metrics['ap']:>8.4f} "
            f"{metrics['f1']:>8.4f} "
            f"{metrics['precision']:>8.4f} "
            f"{metrics['recall']:>8.4f} "
            f"{metrics['specificity']:>8.4f} "
            f"{metrics['mcc']:>8.4f}"
        )

    print("\n[Overall Average: mean over Domain Averages, NOT sample-weighted]")
    print(
        f"Domains={overall['num_domains']} | "
        f"ManiTypes={overall['num_mani_types']} | "
        f"OriginalSamples={overall['samples']} | "
        f"Thr={overall['threshold']:.4f}"
    )
    print(
        f"ACC={overall['accuracy']*100:.2f}% | "
        f"BalACC={overall['balanced_accuracy']*100:.2f}% | "
        f"AUC={overall['auc_roc']:.4f} | "
        f"AP={overall['ap']:.4f} | "
        f"F1={overall['f1']:.4f} | "
        f"P={overall['precision']:.4f} | "
        f"R={overall['recall']:.4f} | "
        f"Spec={overall['specificity']:.4f} | "
        f"MCC={overall['mcc']:.4f}"
    )

    print("=" * 130 + "\n")
    
def compute_domain_auc(domain_stats):
    """
    兼容旧逻辑：只需要 domain AUC 时仍可使用。
    """
    domain_metrics_full = compute_domain_metrics(domain_stats, threshold=0.5)
    return {
        d: {"auc_roc": m["auc_roc"], "ap": m["ap"], "auc_pr": m["auc_pr"]}
        for d, m in domain_metrics_full.items()
    }

def metric_line(result):
    """
    用于 Epoch Summary / Final Summary 的紧凑单行输出。

    如果 result 里有 summary_metrics，则优先使用 summary_metrics。
    对 Test 来说，summary_metrics 是 mani_type -> domain -> overall 的层级宏平均。
    对 Train/Val 来说，默认仍是 pooled metrics。
    """
    m = result.get("summary_metrics", result["metrics"])

    metric_source = "Macro" if result.get("mani_macro_metrics", None) is not None else "Pooled"

    return (
        f"[{metric_source}] "
        f"loss={result['loss']:.4f} | "
        f"ACC={m['accuracy']*100:.2f}% | "
        f"BalACC={m['balanced_accuracy']*100:.2f}% | "
        f"AUC={m['auc_roc']:.4f} | "
        f"AP={m['ap']:.4f} | "
        f"F1={m['f1']:.4f} | "
        f"P={m['precision']:.4f} | "
        f"R={m['recall']:.4f} | "
        f"Spec={m['specificity']:.4f} | "
        f"MCC={m['mcc']:.4f} | "
        f"Kappa={m['kappa']:.4f} | "
        f"Thr={m['threshold']:.4f}"
    )

def print_full_metrics(metrics, title="Metrics", loss=None):
    """
    分块打印完整指标，Train / Val / Test 共用。
    """
    print(f"\n{'='*70}")
    print(title)
    print(f"{'='*70}")

    if loss is not None:
        print(f"Loss                 : {loss:.6f}")

    print(f"Samples              : {metrics['samples']}")
    print(f"Positive / Negative  : {metrics['positive']} / {metrics['negative']}")
    print(f"Pred Pos / Pred Neg  : {metrics['pred_positive']} / {metrics['pred_negative']}")
    print(f"Threshold            : {metrics['threshold']:.4f}")

    print("\n[Ranking / Probability]")
    print(f"AUC-ROC              : {metrics['auc_roc']:.6f}")
    print(f"AP / AUC-PR          : {metrics['ap']:.6f}")
    print(f"LogLoss              : {metrics['log_loss']:.6f}")
    print(f"Brier Score          : {metrics['brier']:.6f}")

    print("\n[Classification]")
    print(f"Accuracy             : {metrics['accuracy']*100:.2f}%")
    print(f"Balanced Accuracy    : {metrics['balanced_accuracy']*100:.2f}%")
    print(f"F1                   : {metrics['f1']:.6f}")
    print(f"Macro F1             : {metrics['f1_macro']:.6f}")
    print(f"Weighted F1          : {metrics['f1_weighted']:.6f}")
    print(f"Precision / PPV      : {metrics['precision']:.6f}")
    print(f"Recall / Sensitivity : {metrics['recall']:.6f}")
    print(f"Specificity / TNR    : {metrics['specificity']:.6f}")
    print(f"NPV                  : {metrics['npv']:.6f}")

    print("\n[Agreement / Error]")
    print(f"MCC                  : {metrics['mcc']:.6f}")
    print(f"Cohen Kappa          : {metrics['kappa']:.6f}")
    print(f"FPR                  : {metrics['fpr']:.6f}")
    print(f"FNR                  : {metrics['fnr']:.6f}")
    print(f"FDR                  : {metrics['fdr']:.6f}")
    print(f"FOR                  : {metrics['for']:.6f}")
    print(f"Youden J             : {metrics['youden_j']:.6f}")
    print(f"Prevalence           : {metrics['prevalence']:.6f}")
    print(f"Pred Positive Rate   : {metrics['pred_positive_rate']:.6f}")

    print("\n[Confusion Matrix]")
    print(f"TP={metrics['tp']} | FP={metrics['fp']} | TN={metrics['tn']} | FN={metrics['fn']}")
    print(f"{'='*70}\n")


def print_domain_metrics(domain_metrics, title="[Per-Domain Metrics]"):
    """
    分 domain 打印常用指标。
    """
    if domain_metrics is None or len(domain_metrics) == 0:
        return

    print(f"\n{title}")
    header = (
        f"{'Domain':<24} "
        f"{'N':>6} {'ACC':>8} {'AUC':>8} {'AP':>8} {'F1':>8} "
        f"{'P':>8} {'R':>8} {'Spec':>8} {'MCC':>8} {'Thr':>6}"
    )
    print(header)
    print("-" * len(header))
    for d, m in domain_metrics.items():
        print(
            f"{str(d):<24} "
            f"{m['samples']:>6d} "
            f"{m['accuracy']*100:>7.2f}% "
            f"{m['auc_roc']:>8.4f} "
            f"{m['ap']:>8.4f} "
            f"{m['f1']:>8.4f} "
            f"{m['precision']:>8.4f} "
            f"{m['recall']:>8.4f} "
            f"{m['specificity']:>8.4f} "
            f"{m['mcc']:>8.4f} "
            f"{m['threshold']:>6.2f}"
        )


def print_train_val_domain_gap(train_result, val_result, title="[Train vs Val Domain AUC/AP/F1 Gap]"):
    train_domains = train_result.get("domain_metrics", {})
    val_domains = val_result.get("domain_metrics", {})

    common_domains = sorted(set(train_domains.keys()) & set(val_domains.keys()))
    if len(common_domains) == 0:
        return

    print(f"\n{title}")
    header = (
        f"{'Domain':<24} "
        f"{'AUC_gap':>10} {'AP_gap':>10} {'F1_gap':>10} {'ACC_gap':>10}"
    )
    print(header)
    print("-" * len(header))

    for d in common_domains:
        train_m = train_domains[d]
        val_m = val_domains[d]
        auc_gap = train_m["auc_roc"] - val_m["auc_roc"]
        ap_gap = train_m["ap"] - val_m["ap"]
        f1_gap = train_m["f1"] - val_m["f1"]
        acc_gap = train_m["accuracy"] - val_m["accuracy"]
        print(
            f"{str(d):<24} "
            f"{auc_gap:+10.4f} {ap_gap:+10.4f} {f1_gap:+10.4f} {acc_gap:+10.4f}"
        )

# =========================================================
# optimizer
# =========================================================
def build_optimizer(model, config, rank=0):
    opt_cfg = config["training"]["optimizer"]

    base_lr = opt_cfg["lr"]
    head_lr = opt_cfg.get("head_lr", base_lr)
    lora_lr = opt_cfg.get("lora_lr", base_lr)
    backbone_lr = opt_cfg.get("backbone_lr", base_lr * 0.1)

    weight_decay = opt_cfg["weight_decay"]
    betas = tuple(opt_cfg["betas"])

    head_params = []
    lora_params = []
    backbone_params = []
    other_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if ("lora_A" in name) or ("lora_B" in name):
            lora_params.append(p)
        elif any(key in name for key in ["classifier", "head", "fc"]):
            head_params.append(p)
        elif "backbone.backbone" in name:
            backbone_params.append(p)
        else:
            other_params.append(p)

    param_groups = []

    if len(head_params) > 0:
        param_groups.append({
            "params": head_params,
            "lr": head_lr,
            "weight_decay": weight_decay,
            "name": "head"
        })

    if len(lora_params) > 0:
        param_groups.append({
            "params": lora_params,
            "lr": lora_lr,
            "weight_decay": weight_decay,
            "name": "lora"
        })

    if len(other_params) > 0:
        param_groups.append({
            "params": other_params,
            "lr": base_lr,
            "weight_decay": weight_decay,
            "name": "other"
        })

    if len(backbone_params) > 0:
        param_groups.append({
            "params": backbone_params,
            "lr": backbone_lr,
            "weight_decay": weight_decay,
            "name": "backbone"
        })

    optimizer = optim.AdamW(param_groups, betas=betas)

    if rank == 0:
        print(f"[Optimizer] head lr     = {head_lr}")
        print(f"[Optimizer] lora lr     = {lora_lr}")
        print(f"[Optimizer] other lr    = {base_lr}")
        print(f"[Optimizer] backbone lr = {backbone_lr}")
        print(
            "[Optimizer] trainable groups: "
            f"head={len(head_params)}, "
            f"lora={len(lora_params)}, "
            f"other={len(other_params)}, "
            f"backbone={len(backbone_params)}"
        )

    return optimizer


# =========================================================
# EMA
# =========================================================
class ModelEMA:
    def __init__(
        self,
        model,
        decay=0.999,
        dynamic_decay=False,
        decay_start=0.99,
        decay_end=0.9995,
        total_steps=1000,
        schedule="cosine",
    ):
        base_model = model.module if hasattr(model, "module") else model
        self.ema = copy.deepcopy(base_model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

        self.dynamic_decay = dynamic_decay
        self.decay = decay
        self.decay_start = decay_start
        self.decay_end = decay_end
        self.total_steps = max(1, int(total_steps))
        self.schedule = schedule
        self.num_updates = 0

    def _get_decay(self):
        if not self.dynamic_decay:
            return float(self.decay)

        progress = min(1.0, self.num_updates / self.total_steps)
        if self.schedule == "linear":
            return float(self.decay_start + (self.decay_end - self.decay_start) * progress)
        elif self.schedule == "cosine":
            cosine_progress = 0.5 * (1.0 - np.cos(np.pi * progress))
            return float(self.decay_start + (self.decay_end - self.decay_start) * cosine_progress)
        else:
            raise ValueError(f"Unknown EMA schedule: {self.schedule}")

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        cur_decay = self._get_decay()

        student = model.module if hasattr(model, "module") else model
        ema_state = self.ema.state_dict()
        model_state = student.state_dict()

        for k, v in ema_state.items():
            model_v = model_state[k]
            if not torch.is_floating_point(model_v):
                v.copy_(model_v)
            else:
                v.mul_(cur_decay).add_(model_v, alpha=1.0 - cur_decay)

        return cur_decay

    def state_dict(self):
        return {
            "ema_state_dict": self.ema.state_dict(),
            "num_updates": self.num_updates,
        }

    def load_state_dict(self, state_dict):
        if isinstance(state_dict, dict) and "ema_state_dict" in state_dict:
            self.ema.load_state_dict(state_dict["ema_state_dict"])
            self.num_updates = state_dict.get("num_updates", 0)
        else:
            self.ema.load_state_dict(state_dict)
            self.num_updates = 0

# =========================================================
# SAM / ASAM
# =========================================================
class SAMASAM:
    """
    同时支持 SAM 和 ASAM。

    sam_type:
        - "sam"  : 普通 SAM，扰动方向只看 grad
        - "asam" : Adaptive SAM，扰动方向看 (abs(w) + eta) * grad

    eta:
        只对 ASAM 生效。
        作用是避免 LoRA 中零初始化或极小参数导致扰动为 0。
    """

    def __init__(self, model, optimizer, rho=0.2, sam_type="asam", eta=0.01, eps=1e-12):
        self.model = model
        self.optimizer = optimizer
        self.rho = rho
        self.sam_type = sam_type.lower()
        self.eta = eta
        self.eps = eps
        self.state = {}

        assert self.sam_type in ["sam", "asam"], \
            f"sam_type must be 'sam' or 'asam', got {self.sam_type}"

    @torch.no_grad()
    def first_step(self):
        grad_norm = self._grad_norm()

        if grad_norm is None:
            return

        for p in self.model.parameters():
            if p.grad is None:
                continue

            if self.sam_type == "asam":
                # ASAM: adaptive perturbation
                # 加 eta 是为了让 LoRA 中接近 0 或等于 0 的参数也能得到扰动
                e_w = (torch.abs(p) + self.eta) * p.grad
            else:
                # SAM: standard perturbation
                e_w = p.grad

            e_w = e_w * (self.rho / (grad_norm + self.eps))

            p.add_(e_w)
            self.state[p] = e_w

    @torch.no_grad()
    def second_step(self):
        for p in self.model.parameters():
            if p in self.state:
                p.sub_(self.state[p])

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.state = {}

    def _grad_norm(self):
        norms = []

        for p in self.model.parameters():
            if p.grad is None:
                continue

            if self.sam_type == "asam":
                grad = (torch.abs(p) + self.eta) * p.grad
            else:
                grad = p.grad

            norms.append(grad.norm(p=2))

        if len(norms) == 0:
            return None

        return torch.norm(torch.stack(norms), p=2)

# =========================================================
# gather train outputs for metrics in DDP
# =========================================================
def gather_train_epoch_outputs(local_preds, local_labels, local_domains, rank, is_distributed):
    if not is_distributed:
        return local_preds, local_labels, local_domains

    gathered = [None for _ in range(dist.get_world_size())] if rank == 0 else None
    dist.gather_object(
        obj={"preds": local_preds, "labels": local_labels, "domains": local_domains},
        object_gather_list=gathered,
        dst=0
    )

    if rank == 0:
        all_preds, all_labels, all_domains = [], [], []
        for item in gathered:
            all_preds.extend(item["preds"])
            all_labels.extend(item["labels"])
            all_domains.extend(item["domains"])
        return all_preds, all_labels, all_domains
    return None, None, None


# =========================================================
# model forward
# =========================================================
def model_forward(model, images):
    logits, cls_token, patch_tokens = model(images)
    return logits, cls_token, patch_tokens


# =========================================================
# eval
# =========================================================
@torch.no_grad()
@torch.no_grad()
def evaluate_loader(
    model,
    dataloader,
    criterion,
    device,
    epoch,
    split_name="Val",
    verbose=True,
    report_mani_macro=False,
):
    model.eval()
    running_loss = 0.0

    all_preds, all_labels = [], []
    all_domains, all_mani_types = [], []

    domain_stats = defaultdict(lambda: {"preds": [], "labels": []})

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [{split_name}]") if verbose else dataloader

    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].float().to(device, non_blocking=True).unsqueeze(1)

        domains = _to_list(batch["domain"])
        mani_types = get_batch_mani_types(batch) if report_mani_macro else None

        logits, _, _ = model_forward(model, images)
        loss = criterion(logits, labels)
        running_loss += loss.item()

        probs = torch.sigmoid(logits)
        probs_np = probs.cpu().numpy().flatten()
        labels_np = labels.cpu().numpy().flatten()

        all_preds.extend(probs_np.tolist())
        all_labels.extend(labels_np.tolist())
        all_domains.extend(domains)

        if report_mani_macro:
            all_mani_types.extend(mani_types)

        for i in range(len(labels_np)):
            d = domains[i]
            domain_stats[d]["preds"].append(float(probs_np[i]))
            domain_stats[d]["labels"].append(float(labels_np[i]))

    all_preds = np.array(all_preds, dtype=float)
    all_labels = np.array(all_labels, dtype=int)

    eval_loss = running_loss / max(1, len(dataloader))

    # If report_mani_macro=True, the threshold is also selected by the same
    # mani_type -> domain -> overall macro protocol.
    if report_mani_macro:
        optimal_threshold, _, _ = find_optimal_threshold_for_manitype_macro(
            probs=all_preds,
            labels=all_labels,
            domains=all_domains,
            mani_types=all_mani_types,
            objective="balanced_acc",
        )
    else:
        optimal_threshold, _, _ = find_optimal_threshold(all_preds, all_labels)

    # pooled/global 指标，保留用于 debug
    metrics = compute_all_metrics(
        all_preds,
        all_labels,
        threshold=optimal_threshold,
    )

    # pooled domain 指标，保留用于 debug
    domain_metrics = compute_domain_metrics(
        domain_stats,
        threshold=optimal_threshold,
    )

    # mani_type -> domain -> overall 的层级宏平均
    mani_macro_metrics = None
    if report_mani_macro:
        mani_macro_metrics = compute_manitype_domain_macro_metrics(
            probs=all_preds,
            labels=all_labels,
            domains=all_domains,
            mani_types=all_mani_types,
            threshold=optimal_threshold,
        )

    if verbose:
        print_full_metrics(
            metrics,
            title=f"Epoch {epoch+1} {split_name} Pooled Metrics",
            loss=eval_loss
        )

        print_domain_metrics(
            domain_metrics,
            title=f"[{split_name} Pooled Per-Domain Metrics]"
        )

        if report_mani_macro:
            print_manitype_domain_macro_metrics(
                mani_macro_metrics,
                title=f"[{split_name} ManiType -> Domain -> Overall Macro Metrics]"
            )

    # Val/Test 如果开启 report_mani_macro，则 summary 返回层级宏平均；
    # 这样 best model / early stopping / checkpoint 都基于同一套宏平均协议。
    summary_metrics = (
        mani_macro_metrics["overall"]
        if mani_macro_metrics is not None
        else metrics
    )

    return {
        "loss": eval_loss,

        # summary 用
        "acc": summary_metrics["accuracy"] * 100.0,
        "auc": summary_metrics["auc_roc"],
        "f1": summary_metrics["f1"],
        "precision": summary_metrics["precision"],
        "recall": summary_metrics["recall"],
        "optimal_threshold": optimal_threshold,

        # summary metrics: for model selection / early stopping
        "summary_metrics": summary_metrics,

        # pooled debug metrics
        "metrics": metrics,
        "domain_metrics": domain_metrics,

        # hierarchical macro metrics
        "mani_macro_metrics": mani_macro_metrics,
    }

def run_rank0_full_val_and_broadcast(
    model_for_eval, dataloader, criterion, device,
    epoch, rank, is_distributed, split_name="Val", verbose=True,
    report_mani_macro=True,
):
    if is_distributed:
        dist.barrier()

    if rank == 0:
        result = evaluate_loader(
            model=model_for_eval,
            dataloader=dataloader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            split_name=split_name,
            verbose=verbose,
            report_mani_macro=report_mani_macro,
        )
    else:
        result = None

    result = broadcast_object(result, rank, is_distributed, device)
    return result


# =========================================================
# auto test
# =========================================================
def _print_final_eval_summary(name, result):
    """
    Final summary prioritizes the requested Test hierarchical metric:
        mani_type -> domain -> overall macro-average.
    Pooled metrics are printed only as debug/reference.
    """
    official_pooled = result["test_metrics_with_val_thr"]
    oracle_pooled = result["test_metrics_with_test_thr"]

    official_macro_pack = result.get("test_mani_macro_with_val_thr", None)
    oracle_macro_pack = result.get("test_mani_macro_with_test_thr", None)

    if official_macro_pack is not None:
        official = official_macro_pack["overall"]
        official_name = "VAL-thr ManiType-Domain-Macro"
    else:
        official = official_pooled
        official_name = "VAL-thr Pooled"

    if oracle_macro_pack is not None:
        oracle = oracle_macro_pack["overall"]
        oracle_name = "TEST-thr ManiType-Domain-Macro"
    else:
        oracle = oracle_pooled
        oracle_name = "TEST-thr Pooled"

    print(
        f"{name} | {official_name} | "
        f"ACC={official['accuracy']*100:.2f}% | "
        f"BalACC={official['balanced_accuracy']*100:.2f}% | "
        f"AUC={official['auc_roc']:.4f} | "
        f"AP={official['ap']:.4f} | "
        f"F1={official['f1']:.4f} | "
        f"P={official['precision']:.4f} | "
        f"R={official['recall']:.4f} | "
        f"Spec={official['specificity']:.4f} | "
        f"MCC={official['mcc']:.4f} | "
        f"Thr={result['val_best_thr']:.4f}"
    )

    print(
        f"{name} | {oracle_name} | "
        f"ACC={oracle['accuracy']*100:.2f}% | "
        f"BalACC={oracle['balanced_accuracy']*100:.2f}% | "
        f"AUC={oracle['auc_roc']:.4f} | "
        f"AP={oracle['ap']:.4f} | "
        f"F1={oracle['f1']:.4f} | "
        f"P={oracle['precision']:.4f} | "
        f"R={oracle['recall']:.4f} | "
        f"Spec={oracle['specificity']:.4f} | "
        f"MCC={oracle['mcc']:.4f} | "
        f"Thr={result['test_best_thr']:.4f}"
    )

    print(
        f"{name} | Pooled Debug | "
        f"VAL-thr ACC={official_pooled['accuracy']*100:.2f}% | "
        f"AUC={official_pooled['auc_roc']:.4f} | "
        f"AP={official_pooled['ap']:.4f} | "
        f"F1={official_pooled['f1']:.4f}"
    )


def _compact_metric_dict(metrics):
    """
    只保存最终对比最需要的核心指标，避免把大对象和不可 JSON 序列化对象写进去。
    """
    if metrics is None:
        return None

    keep_keys = [
        "samples",
        "num_domains",
        "num_mani_types",
        "positive",
        "negative",
        "threshold",
        "accuracy",
        "balanced_accuracy",
        "auc_roc",
        "auc_pr",
        "ap",
        "f1",
        "f1_macro",
        "f1_weighted",
        "precision",
        "recall",
        "specificity",
        "mcc",
        "kappa",
        "log_loss",
        "brier",
        "tp",
        "tn",
        "fp",
        "fn",
    ]

    out = {}
    for k in keep_keys:
        if k not in metrics:
            continue
        v = metrics[k]
        if isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, (np.floating,)):
            v = float(v)
        out[k] = v
    return out


def _compact_one_final_eval_result(result):
    """
    保存一个模型 Student/EMA 的最终结果。
    official_val_thr_macro 是正式结果：使用 val macro 最优阈值，在 test 上按 mani_type->domain->overall macro 计算。
    oracle_test_thr_macro 是 test 自己搜阈值的上限结果，仅作参考。
    """
    if result is None:
        return None

    official_macro_pack = result.get("test_mani_macro_with_val_thr", None)
    oracle_macro_pack = result.get("test_mani_macro_with_test_thr", None)

    official_macro = (
        official_macro_pack.get("overall", None)
        if official_macro_pack is not None else None
    )
    oracle_macro = (
        oracle_macro_pack.get("overall", None)
        if oracle_macro_pack is not None else None
    )

    return {
        "use_temperature_scaling": bool(result.get("use_temperature_scaling", True)),
        "val_best_thr": float(result.get("val_best_thr", 0.0)),
        "test_best_thr": float(result.get("test_best_thr", 0.0)),
        "val_loss": float(result.get("val_loss", 0.0)),
        "test_loss": float(result.get("test_loss", 0.0)),

        "official_val_thr_macro": _compact_metric_dict(official_macro),
        "oracle_test_thr_macro": _compact_metric_dict(oracle_macro),

        # debug/reference: pooled 指标也保存，方便你排查样本加权与 macro 的差异。
        "official_val_thr_pooled": _compact_metric_dict(
            result.get("test_metrics_with_val_thr", None)
        ),
        "oracle_test_thr_pooled": _compact_metric_dict(
            result.get("test_metrics_with_test_thr", None)
        ),
    }


def save_final_eval_summary(final_result, save_dir, filename="final_eval_summary.json"):
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, filename)

    compact = {
        "student": _compact_one_final_eval_result(final_result.get("student", None)),
        "ema": _compact_one_final_eval_result(final_result.get("ema", None)),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(compact, f, ensure_ascii=False, indent=2)

    print(f"[FinalEval] Saved final evaluation summary to: {out_path}")
    return out_path


def run_final_test(
    best_ckpt_path,
    config,
    criterion,
    device,
    val_loader,
    test_loader,
    ema_enabled,
    save_ema,
    final_epoch=0,
    use_temperature_scaling=True,
):
    print("\n" + "=" * 70)
    print("加载 Val-Control 最佳模型，并进行最终测试")
    print("=" * 70)
    print(f"Best checkpoint: {best_ckpt_path}")
    print(f"Use Temperature Scaling: {use_temperature_scaling}")

    checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=False)

    # =====================================================
    # Student final eval
    # =====================================================
    test_student_model = ForensicDinoBaseline(config).to(device)
    test_student_model = apply_lora_to_forensic_dino(
        test_student_model,
        config,
        rank=0,
    )
    test_student_model.load_state_dict(checkpoint["model_state_dict"])
    test_student_model.eval()

    test_student_result = evaluate_with_temperature_and_thresholds(
        model=test_student_model,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        epoch=final_epoch,
        split_name="Final-Student",
        use_temperature_scaling=use_temperature_scaling,
    )

    # =====================================================
    # EMA final eval
    # =====================================================
    test_ema_result = None

    if ema_enabled and save_ema and checkpoint.get("ema_model_state_dict", None) is not None:
        print("\n" + "-" * 70)
        print("加载 EMA 模型，并进行最终测试")
        print("-" * 70)

        test_ema_model = ForensicDinoBaseline(config).to(device)
        test_ema_model = apply_lora_to_forensic_dino(
            test_ema_model,
            config,
            rank=0,
        )

        ema_state = checkpoint["ema_model_state_dict"]

        if isinstance(ema_state, dict) and "ema_state_dict" in ema_state:
            test_ema_model.load_state_dict(ema_state["ema_state_dict"])
        else:
            test_ema_model.load_state_dict(ema_state)

        test_ema_model.eval()

        test_ema_result = evaluate_with_temperature_and_thresholds(
            model=test_ema_model,
            val_loader=val_loader,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
            epoch=final_epoch,
            split_name="Final-EMA",
            use_temperature_scaling=use_temperature_scaling,
        )

    # =====================================================
    # Summary
    # =====================================================
    print("\n" + "=" * 70)
    print("最终测试结果汇总")
    print("=" * 70)
    print("说明：")
    print("  VAL-thr ManiType-Domain-Macro  = test 按 mani_type 平均，再按 domain 平均，再 overall 平均")
    print("  TEST-thr ManiType-Domain-Macro = test 自己搜索阈值后的 macro 上限，仅作参考")
    print("  Pooled Debug                   = 全 test pooled 指标，仅作调试参考")
    print("-" * 70)

    _print_final_eval_summary("Final-Student", test_student_result)

    if test_ema_result is not None:
        _print_final_eval_summary("Final-EMA    ", test_ema_result)

    print("=" * 70)


    final_result = {
        "student": test_student_result,
        "ema": test_ema_result,
    }

    save_final_eval_summary(
        final_result=final_result,
        save_dir=config.get("save_dir", "./checkpoints"),
        filename="final_eval_summary.json",
    )

    return final_result

# =========================================================
# train one epoch
# =========================================================
def train_one_epoch(
    model,
    ema_model,
    dataloader,
    criterion,
    optimizer,
    asam,
    device,
    epoch,
    rank,
    is_distributed,
    grad_clip=0.0
):
    model.train()

    running_loss = 0.0
    correct = 0.0
    total = 0.0
    num_batches = 0
    last_ema_decay = None

    local_preds, local_labels, local_domains = [], [], []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]") if rank == 0 else dataloader

    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].float().to(device, non_blocking=True).unsqueeze(1)
        domains = batch["domain"]

        optimizer.zero_grad()

        logits, _, _ = model_forward(model, images)
        loss = criterion(logits, labels)

        if asam is not None:
            # ---- ASAM first step ----
            loss.backward()
            asam.first_step()

            optimizer.zero_grad()

            logits2, _, _ = model_forward(model, images)
            loss2 = criterion(logits2, labels)
            loss2.backward()

            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            asam.second_step()
        else:
            # ---- Normal backward ----
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if ema_model is not None:
            last_ema_decay = ema_model.update(model)

        running_loss += loss.item()
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        num_batches += 1

        probs_np = probs.detach().cpu().numpy().flatten().tolist()
        labels_np = labels.detach().cpu().numpy().flatten().tolist()
        local_preds.extend(probs_np)
        local_labels.extend(labels_np)
        local_domains.extend(list(domains))

        if rank == 0:
            pbar.set_postfix({
                "loss": f"{running_loss/(batch_idx+1):.4f}",
                "acc": f"{100.*correct/max(1,total):.2f}%"
            })

    # reduce stats across GPUs
    if is_distributed:
        stats = torch.tensor([running_loss, correct, total, num_batches],
                             dtype=torch.float64, device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss_sum, total_correct, total_samples, total_batches = stats.tolist()
        epoch_loss = total_loss_sum / max(1.0, total_batches)
        epoch_acc = 100.0 * total_correct / max(1.0, total_samples)
    else:
        epoch_loss = running_loss / max(1, num_batches)
        epoch_acc = 100.0 * correct / max(1, total)

    # gather preds/labels for train metrics
    gathered_preds, gathered_labels, gathered_domains = gather_train_epoch_outputs(
        local_preds=local_preds,
        local_labels=local_labels,
        local_domains=local_domains,
        rank=rank,
        is_distributed=is_distributed
    )

    train_result = None
    if rank == 0:
        gathered_preds = np.array(gathered_preds)
        gathered_labels = np.array(gathered_labels)

        train_threshold, _, _ = find_optimal_threshold(gathered_preds, gathered_labels)
        train_metrics = compute_all_metrics(gathered_preds, gathered_labels, threshold=train_threshold)

        domain_stats = defaultdict(lambda: {"preds": [], "labels": []})
        for p, y, d in zip(gathered_preds, gathered_labels, gathered_domains):
            domain_stats[d]["preds"].append(float(p))
            domain_stats[d]["labels"].append(float(y))

        train_domain_metrics = compute_domain_metrics(domain_stats, threshold=train_threshold)

        train_result = {
            "loss": epoch_loss,
            "acc": epoch_acc,
            "auc": train_metrics["auc_roc"],
            "f1": train_metrics["f1"],
            "precision": train_metrics["precision"],
            "recall": train_metrics["recall"],
            "optimal_threshold": train_threshold,
            "metrics": train_metrics,
            "domain_metrics": train_domain_metrics,
            "ema_decay": last_ema_decay,
        }

    train_result = broadcast_object(train_result, rank, is_distributed, device)
    return train_result


# =========================================================
# main
# =========================================================
def main():
    args = parse_args()
    config = load_config(args.config)

    is_distributed, rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}") if is_distributed else torch.device(config["system"]["device"])

    set_seed(config["system"].get("seed", 42))

    if rank == 0:
        print("\n" + "=" * 70)
        print(" DINOv2 Forensic Detection Training ")
        print("=" * 70)

    # -------- dataset --------
    data_cfg = config["data"]

    train_dataset = ForensicImageDataset(
        json_path=config["train_dataset"]["path"],
        image_size=data_cfg.get("image_size", 224),
        mean=tuple(data_cfg.get("mean", [0.485, 0.456, 0.406])),
        std=tuple(data_cfg.get("std", [0.229, 0.224, 0.225])),
        is_train=True,
        target_domains=config["train_dataset"].get("target_domains"),
        target_labels=config["train_dataset"].get("target_labels"),
        target_mani_types=config["train_dataset"].get("target_mani_types"),
        strict_mode=data_cfg.get("strict_mode", False)
    )

    val_dataset = ForensicImageDataset(
        json_path=config["val_dataset"]["path"],
        image_size=data_cfg.get("image_size", 224),
        mean=tuple(data_cfg.get("mean", [0.485, 0.456, 0.406])),
        std=tuple(data_cfg.get("std", [0.229, 0.224, 0.225])),
        is_train=False,
        target_domains=config["val_dataset"].get("target_domains"),
        target_labels=config["val_dataset"].get("target_labels"),
        target_mani_types=config["val_dataset"].get("target_mani_types"),
        strict_mode=data_cfg.get("strict_mode", False)
    )

    test_dataset = None
    if config.get("test_datasets", {}).get("path", None) is not None:
        test_dataset = ForensicImageDataset(
            json_path=config["test_datasets"]["path"],
            image_size=data_cfg.get("image_size", 224),
            mean=tuple(data_cfg.get("mean", [0.485, 0.456, 0.406])),
            std=tuple(data_cfg.get("std", [0.229, 0.224, 0.225])),
            is_train=False,
            target_domains=config["test_datasets"].get("target_domains"),
            target_labels=config["test_datasets"].get("target_labels"),
            target_mani_types=config["test_datasets"].get("target_mani_types"),
            strict_mode=data_cfg.get("strict_mode", False)
        )

    # loaders (val/test only on rank0)
    if rank == 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=config["system"]["num_workers"],
            pin_memory=config["system"]["pin_memory"]
        )

        test_loader = None
        if test_dataset is not None:
            test_loader = DataLoader(
                test_dataset,
                batch_size=config["training"]["batch_size"],
                shuffle=False,
                num_workers=config["system"]["num_workers"],
                pin_memory=config["system"]["pin_memory"]
            )
    else:
        val_loader = None
        test_loader = None

    if rank == 0:
        print_dataset_summary(train_dataset, None, name="Train")
        print_dataset_summary(val_dataset, val_loader, name="Validation")
        if test_dataset is not None:
            print_dataset_summary(test_dataset, test_loader, name="Test")

    # -------- model --------
    model = ForensicDinoBaseline(config).to(device)
    model = apply_lora_to_forensic_dino(model, config, rank=rank)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        if hasattr(model, "backbone") and hasattr(model.backbone, "print_trainable_status"):
            model.backbone.print_trainable_status()

    if is_distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            broadcast_buffers=False
        )

    # -------- optimizer --------
    optimizer = build_optimizer(model, config, rank=rank)

    # -------- SAM / ASAM --------
    sam_cfg = config["training"]["optimizer"].get("sam", {})
    sam_enabled = sam_cfg.get("enabled", False)

    sam_type = sam_cfg.get("type", "asam").lower()

    if sam_type not in ["sam", "asam"]:
        raise ValueError(f"training.optimizer.sam.type must be 'sam' or 'asam', got {sam_type}")

    # 推荐默认值：
    # SAM 一般 rho 小一点，常用 0.05
    # ASAM 一般 rho 大一点，常用 0.1 / 0.2
    default_rho = 0.05 if sam_type == "sam" else 0.2

    sam_rho = sam_cfg.get("rho", default_rho)
    sam_eta = sam_cfg.get("eta", 0.01)

    base_model_for_asam = model.module if hasattr(model, "module") else model

    asam = None
    if sam_enabled:
        asam = SAMASAM(
            model=base_model_for_asam,
            optimizer=optimizer,
            rho=sam_rho,
            sam_type=sam_type,
            eta=sam_eta
        )

    if rank == 0:
        if sam_enabled:
            print(
                f"[SAM/ASAM] enabled=True | "
                f"type={sam_type} | rho={sam_rho} | eta={sam_eta}"
            )
        else:
            print("[SAM/ASAM] enabled=False")
            
    # -------- scheduler --------
    # 默认使用 ReduceLROnPlateau：val_control_score 不改善就立刻降 LR。
    scheduler = build_scheduler(optimizer, config)

    if rank == 0:
        sched_cfg = config["training"].get("scheduler", {})
        print(f"[Scheduler] type={sched_cfg.get('type', sched_cfg.get('name', 'plateau'))}")

    # -------- loss --------
    criterion = nn.BCEWithLogitsLoss()

    # -------- EMA --------
    ema_cfg = config.get("ema", {})
    ema_enabled = ema_cfg.get("enabled", False)

    dynamic_decay = ema_cfg.get("dynamic_decay", False)
    decay = ema_cfg.get("decay", 0.999)
    decay_start = ema_cfg.get("decay_start", 0.99)
    decay_end = ema_cfg.get("decay_end", 0.9995)
    schedule = ema_cfg.get("schedule", "cosine")

    use_ema_for_val = ema_cfg.get("use_ema_for_val", True)
    save_ema = ema_cfg.get("save_ema", True)

    ema_model = None
    if ema_enabled:
        batch_size = config["training"]["batch_size"]
        steps_per_epoch_est = max(1, len(train_dataset) // batch_size)
        total_steps_est = steps_per_epoch_est * total_epochs

        ema_model = ModelEMA(
            model=model,
            decay=decay,
            dynamic_decay=dynamic_decay,
            decay_start=decay_start,
            decay_end=decay_end,
            total_steps=total_steps_est,
            schedule=schedule
        )

    if rank == 0:
        if ema_enabled:
            if dynamic_decay:
                print(f"[EMA] enabled=True, dynamic_decay=True, schedule={schedule}, "
                      f"decay_start={decay_start}, decay_end={decay_end}")
            else:
                print(f"[EMA] enabled=True, dynamic_decay=False, decay={decay}")
        else:
            print("[EMA] enabled=False")

    # -------- early stopping --------
    es_cfg = config["training"].get("early_stopping", {})
    early_stopper = None
    if es_cfg.get("enabled", False):
        early_stopper = EarlyStopping(
            patience=es_cfg.get("patience", 8),
            min_delta=es_cfg.get("min_delta", 0.0005),
            monitor=es_cfg.get("monitor", "val_auc"),
            verbose=(rank == 0)
        )

    # -------- checkpoint --------
    save_dir = config.get("save_dir", "./checkpoints/dino")
    os.makedirs(save_dir, exist_ok=True)

    start_epoch = 0
    final_epoch = 0
    best_val_auc = 0.0
    best_val_ap = 0.0
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_val_precision = 0.0
    best_val_recall = 0.0
    best_val_mcc = 0.0
    best_threshold = 0.5

    val_control_cfg = config["training"].get("val_control", {})
    val_control_min_delta = val_control_cfg.get(
        "min_delta",
        config["training"].get("val_control_min_delta", 5e-4),
    )
    rollback_on_no_improve = val_control_cfg.get(
        "rollback_on_no_improve",
        config["training"].get("rollback_on_no_improve", True),
    )
    clear_optimizer_on_rollback = val_control_cfg.get("clear_optimizer_on_rollback", True)
    best_val_control_score = -float("inf")
    best_val_control_snapshot = None

    checkpoint_path = config.get("checkpoint_path", None)
    resume = config.get("resume", False)

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        if rank == 0:
            print(f"加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if hasattr(model, "module"):
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])

        if ema_enabled and checkpoint.get("ema_model_state_dict") is not None:
            ema_model.load_state_dict(checkpoint["ema_model_state_dict"])

        if resume:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"]
            best_val_acc = checkpoint.get("val_acc", 0.0)
            best_val_auc = checkpoint.get("val_auc", 0.0)
            best_val_ap = checkpoint.get("val_ap", 0.0)
            best_val_f1 = checkpoint.get("val_f1", 0.0)
            best_val_precision = checkpoint.get("val_precision", 0.0)
            best_val_recall = checkpoint.get("val_recall", 0.0)
            best_val_mcc = checkpoint.get("val_mcc", 0.0)
            best_threshold = checkpoint.get("optimal_threshold", 0.5)
            best_val_control_score = checkpoint.get("best_val_control_score", checkpoint.get("val_control_score", -float("inf")))
            if np.isfinite(best_val_control_score):
                best_val_control_snapshot = make_training_snapshot(
                    model=model,
                    ema_model=ema_model if ema_enabled else None,
                )

    # -------- train loop --------
    grad_clip = config["training"].get("grad_clip", 0.0)

    for epoch in range(start_epoch, config["training"]["epochs"]):
        final_epoch = epoch + 1

        if is_distributed:
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
            train_sampler.set_epoch(epoch)
        else:
            train_sampler = None

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=config["system"]["num_workers"],
            pin_memory=config["system"]["pin_memory"],
            drop_last=True
        )

        if rank == 0:
            print(f"\n[Epoch {epoch+1}] train batches: {len(train_loader)}")

        # ===== train =====
        train_result = train_one_epoch(
            model=model,
            ema_model=ema_model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            asam=asam,
            device=device,
            epoch=epoch,
            rank=rank,
            is_distributed=is_distributed,
            grad_clip=grad_clip
        )

        # ===== val student =====
        student_eval_model = model.module if hasattr(model, "module") else model
        student_val_result = run_rank0_full_val_and_broadcast(
            model_for_eval=student_eval_model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            rank=rank,
            is_distributed=is_distributed,
            split_name="Val-Student",
            verbose=True,
            report_mani_macro=True,
        )

        # ===== val ema =====
        ema_val_result = None
        if ema_enabled:
            ema_val_result = run_rank0_full_val_and_broadcast(
                model_for_eval=ema_model.ema,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                epoch=epoch,
                rank=rank,
                is_distributed=is_distributed,
                split_name="Val-EMA",
                verbose=True,
                report_mani_macro=True,
            )

        # choose main val result
        if ema_enabled and use_ema_for_val and ema_val_result is not None:
            main_val_result = ema_val_result
            main_val_name = "Val-EMA"
        else:
            main_val_result = student_val_result
            main_val_name = "Val-Student"

        train_loss = train_result["loss"]
        train_acc = train_result["acc"]
        train_auc = train_result["auc"]
        train_f1 = train_result["f1"]
        train_precision = train_result["precision"]
        train_recall = train_result["recall"]
        train_domain_metrics = train_result["domain_metrics"]
        current_ema_decay = train_result.get("ema_decay", None)

        val_loss = main_val_result["loss"]
        val_acc = main_val_result["acc"]
        val_auc = main_val_result["auc"]
        val_f1 = main_val_result["f1"]
        val_precision = main_val_result["precision"]
        val_recall = main_val_result["recall"]
        optimal_threshold = main_val_result["optimal_threshold"]

        # From now on, validation model selection uses the requested
        # mani_type -> domain -> overall macro-average metrics.
        # pooled_metrics are kept only for debug/diagnostics.
        metrics = main_val_result.get("summary_metrics", main_val_result["metrics"])
        pooled_metrics = main_val_result["metrics"]
        mani_macro_metrics = main_val_result.get("mani_macro_metrics", None)
        domain_metrics = main_val_result["domain_metrics"]

        # Val control score: 0.5 * macro AUC + 0.5 * macro F1
        val_control_score = build_val_control_score(main_val_result)
        improved_by_val_control = val_control_score > (best_val_control_score + val_control_min_delta)

        if improved_by_val_control:
            best_val_control_score = val_control_score
            best_val_control_snapshot = make_training_snapshot(
                model=model,
                ema_model=ema_model if ema_enabled else None,
            )

        # Plateau scheduler 必须吃 val_control_score；Cosine 等保留无参 step。
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_control_score)
        else:
            scheduler.step()

        # ===== test every epoch =====
        test_student_result = None
        test_ema_result = None
        if test_dataset is not None and test_loader is not None and rank == 0:
            student_eval_model = model.module if hasattr(model, "module") else model
            test_student_result = evaluate_loader(
                model=student_eval_model,
                dataloader=test_loader,
                criterion=criterion,
                device=device,
                epoch=epoch,
                split_name="Test-Student",
                verbose=True,
                report_mani_macro=True,
            )

            if ema_enabled:
                test_ema_result = evaluate_loader(
                    model=ema_model.ema,
                    dataloader=test_loader,
                    criterion=criterion,
                    device=device,
                    epoch=epoch,
                    split_name="Test-EMA",
                    verbose=True,
                    report_mani_macro=True,
                )

        if rank == 0:
            lr_info = " | ".join([
                f"group{i}={pg['lr']:.6e}"
                for i, pg in enumerate(optimizer.param_groups)
            ])

            print_full_metrics(
                train_result["metrics"],
                title=f"Epoch {epoch+1} Train Metrics",
                loss=train_loss
            )
            print_domain_metrics(
                train_domain_metrics,
                title="[Train Per-Domain Metrics]"
            )

            print_train_val_domain_gap(
                train_result, student_val_result,
                title="[Train vs Val-Student Domain AUC/AP/F1/ACC Gap]"
            )
            if ema_val_result is not None:
                print_train_val_domain_gap(
                    train_result, ema_val_result,
                    title="[Train vs Val-EMA Domain AUC/AP/F1/ACC Gap]"
                )

            if current_ema_decay is not None:
                print(f"\n[EMA] decay(last): {current_ema_decay:.6f}")

            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train       | {metric_line(train_result)}")
            print(f"  {main_val_name:<11} | {metric_line(main_val_result)}")
            print(f"  val_control_score=0.5*AUC+0.5*F1 = {val_control_score:.6f}")
            print(f"  best_val_control_score = {best_val_control_score:.6f}")
            print(f"  lr: {lr_info}")

            if test_student_result is not None:
                print(f"  Test-Student | {metric_line(test_student_result)}")
            if test_ema_result is not None:
                print(f"  Test-EMA     | {metric_line(test_ema_result)}")

            # save best by val_control_score, not single val_auc
            if improved_by_val_control:
                best_val_auc = val_auc
                best_val_ap = metrics["ap"]
                best_val_acc = val_acc
                best_val_f1 = val_f1
                best_val_precision = val_precision
                best_val_recall = val_recall
                best_val_mcc = metrics["mcc"]
                best_threshold = optimal_threshold

                student_state = (
                    model.module.state_dict()
                    if hasattr(model, "module")
                    else model.state_dict()
                )
                ckpt = {
                    "epoch": epoch + 1,
                    "model_state_dict": student_state,
                    "ema_model_state_dict": (
                        ema_model.state_dict() if (ema_enabled and save_ema) else None
                    ),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_result": train_result,
                    "student_val_result": student_val_result,
                    "ema_val_result": ema_val_result,
                    "main_val_name": main_val_name,
                    "val_acc": val_acc,
                    "val_auc": val_auc,
                    "val_ap": metrics["ap"],
                    "val_auc_pr": metrics["auc_pr"],
                    "val_f1": val_f1,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "val_mcc": metrics["mcc"],
                    "val_balanced_accuracy": metrics["balanced_accuracy"],
                    "optimal_threshold": optimal_threshold,
                    "metrics": metrics,
                    "summary_metrics": metrics,
                    "pooled_metrics": pooled_metrics,
                    "domain_metrics": domain_metrics,
                    "mani_macro_metrics": mani_macro_metrics,
                    "val_control_score": val_control_score,
                    "best_val_control_score": best_val_control_score,
                    "val_control_min_delta": val_control_min_delta,
                    "config": config,
                }
                torch.save(ckpt, os.path.join(save_dir, "best_model.pth"))
                print(
                    f"✓ Best model saved! ({main_val_name} "
                    f"Control={val_control_score:.6f}, "
                    f"AUC={val_auc:.4f}, AP={metrics['ap']:.4f}, "
                    f"F1={val_f1:.4f}, ACC={val_acc:.2f}%)"
                )

            # periodic save
            save_freq = config.get("logging", {}).get("save_freq", 5)
            if (epoch + 1) % save_freq == 0:
                student_state = (
                    model.module.state_dict()
                    if hasattr(model, "module")
                    else model.state_dict()
                )
                ckpt = {
                    "epoch": epoch + 1,
                    "model_state_dict": student_state,
                    "ema_model_state_dict": (
                        ema_model.state_dict() if (ema_enabled and save_ema) else None
                    ),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_result": train_result,
                    "student_val_result": student_val_result,
                    "ema_val_result": ema_val_result,
                    "main_val_name": main_val_name,
                    "val_acc": val_acc,
                    "val_auc": val_auc,
                    "val_ap": metrics["ap"],
                    "val_auc_pr": metrics["auc_pr"],
                    "val_f1": val_f1,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "val_mcc": metrics["mcc"],
                    "val_balanced_accuracy": metrics["balanced_accuracy"],
                    "optimal_threshold": optimal_threshold,
                    "metrics": metrics,
                    "summary_metrics": metrics,
                    "pooled_metrics": pooled_metrics,
                    "domain_metrics": domain_metrics,
                    "mani_macro_metrics": mani_macro_metrics,
                    "val_control_score": val_control_score,
                    "best_val_control_score": best_val_control_score,
                    "val_control_min_delta": val_control_min_delta,
                    "config": config,
                }
                ckpt_name = f"checkpoint_epoch_{epoch+1}.pth"
                torch.save(ckpt, os.path.join(save_dir, ckpt_name))
                print(f"💾 Checkpoint saved: {ckpt_name}")

        # validation-guided rollback
        # 当前 epoch 的 test 已经保留打印；如果 val_control_score 没改善，
        # 下一轮训练从历史 val 最优点 + 更小 LR 重新出发。
        if rollback_on_no_improve and (not improved_by_val_control):
            if best_val_control_snapshot is not None:
                restore_training_snapshot(
                    model=model,
                    snapshot=best_val_control_snapshot,
                    device=device,
                    ema_model=ema_model if ema_enabled else None,
                    optimizer=optimizer,
                    clear_optimizer_state=clear_optimizer_on_rollback,
                )
                if rank == 0:
                    print(
                        "  ↩ rollback: restored model/EMA to best val_control_score "
                        "state for the next epoch."
                    )
                    if clear_optimizer_on_rollback:
                        print("  ↩ rollback: optimizer state cleared; current LR is kept.")
            else:
                if rank == 0:
                    print("  [rollback skipped] no best_val_control_snapshot available yet.")

        if is_distributed:
            dist.barrier()

        # early stopping
        if early_stopper is not None:
            monitor = es_cfg.get("monitor", "val_auc")
            monitor_map = {
                "val_acc": val_acc,
                "val_auc": val_auc,
                "val_ap": metrics["ap"],
                "val_auc_pr": metrics["auc_pr"],
                "val_f1": val_f1,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_balanced_acc": metrics["balanced_accuracy"],
                "val_mcc": metrics["mcc"],
                "val_loss": val_loss,
                "val_control_score": val_control_score,
            }
            current_score = monitor_map.get(monitor, val_control_score)
            should_stop = early_stopper(current_score, epoch)

            if is_distributed:
                stop_tensor = torch.tensor(
                    [1.0 if should_stop else 0.0], device=device
                )
                dist.broadcast(stop_tensor, src=0)
                should_stop = stop_tensor.item() > 0.5

            if should_stop:
                if rank == 0:
                    print(f"\nEarly stop at epoch {epoch+1}")
                break

    cleanup_distributed()

    if rank == 0:
        print("\n" + "=" * 70)
        print("训练完成!")
        print(f"最佳验证 ACC       : {best_val_acc:.2f}%")
        print(f"最佳验证 AUC-ROC   : {best_val_auc:.4f}")
        print(f"最佳验证 AP/AUC-PR : {best_val_ap:.4f}")
        print(f"最佳验证 F1        : {best_val_f1:.4f}")
        print(f"最佳验证 Precision : {best_val_precision:.4f}")
        print(f"最佳验证 Recall    : {best_val_recall:.4f}")
        print(f"最佳验证 MCC       : {best_val_mcc:.4f}")
        print(f"最佳 Val-Control   : {best_val_control_score:.6f}")
        print(f"最佳阈值           : {best_threshold:.2f}")
        print("=" * 70)

        if test_dataset is not None and test_loader is not None:
            best_ckpt_path = os.path.join(save_dir, "best_model.pth")
            if os.path.exists(best_ckpt_path):
                run_final_test(
                    best_ckpt_path=best_ckpt_path,
                    config=config,
                    criterion=criterion,
                    device=device,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    ema_enabled=ema_enabled,
                    save_ema=save_ema,
                    final_epoch=final_epoch,
                    use_temperature_scaling=True,
                )
            else:
                print(f"[Warning] 未找到 best_model.pth，跳过测试: {best_ckpt_path}")


if __name__ == "__main__":
    main()
