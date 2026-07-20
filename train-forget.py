import os
import yaml
import argparse
import random
import numpy as np
from collections import defaultdict
import copy
import time

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


# =========================================================
# utils
# =========================================================
def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="DINOv2 + Forgetting Curriculum + ASAM + EMA")
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


def get_student_model(model):
    return model.module if hasattr(model, "module") else model


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


def compute_domain_auc(domain_stats):
    from sklearn.metrics import roc_auc_score
    domain_metrics = {}
    for d, stats in sorted(domain_stats.items()):
        d_preds = np.array(stats["preds"])
        d_labels = np.array(stats["labels"])
        if len(d_labels) > 0 and len(np.unique(d_labels)) > 1:
            d_auc = roc_auc_score(d_labels, d_preds)
        else:
            d_auc = 0.5
        domain_metrics[d] = {"auc_roc": d_auc}
    return domain_metrics


def print_train_val_domain_gap(train_result, val_result, title="[Train vs Val Domain AUC Gap]"):
    train_domains = train_result.get("domain_metrics", {})
    val_domains = val_result.get("domain_metrics", {})

    common_domains = sorted(set(train_domains.keys()) & set(val_domains.keys()))
    if len(common_domains) == 0:
        return

    print(f"\n{title}")
    for d in common_domains:
        train_auc = train_domains[d]["auc_roc"]
        val_auc = val_domains[d]["auc_roc"]
        gap = train_auc - val_auc
        print(f"  {d}: train={train_auc:.4f} | val={val_auc:.4f} | gap={gap:+.4f}")


# =========================================================
# optimizer
# =========================================================
def build_optimizer(model, config, rank=0):
    opt_cfg = config["training"]["optimizer"]

    base_lr = opt_cfg["lr"]
    backbone_lr = opt_cfg.get("backbone_lr", base_lr * 0.1)
    weight_decay = opt_cfg["weight_decay"]
    betas = tuple(opt_cfg["betas"])

    backbone_params = []
    other_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "backbone.backbone" in name:
            backbone_params.append(p)
        else:
            other_params.append(p)

    param_groups = []
    if len(other_params) > 0:
        param_groups.append({
            "params": other_params,
            "lr": base_lr,
            "weight_decay": weight_decay
        })
    if len(backbone_params) > 0:
        param_groups.append({
            "params": backbone_params,
            "lr": backbone_lr,
            "weight_decay": weight_decay
        })

    optimizer = optim.AdamW(param_groups, betas=betas)

    if rank == 0:
        print(f"[Optimizer] head/other lr = {base_lr}")
        print(f"[Optimizer] backbone lr   = {backbone_lr}")
        print(f"[Optimizer] trainable groups: other={len(other_params)}, backbone={len(backbone_params)}")

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
# ASAM
# =========================================================
class ASAM:
    def __init__(self, model, optimizer, rho=0.3, eps=1e-12):
        self.model = model
        self.optimizer = optimizer
        self.rho = rho
        self.eps = eps
        self.state = {}

    @torch.no_grad()
    def first_step(self):
        grad_norm = self._grad_norm()
        for p in self.model.parameters():
            if p.grad is None:
                continue
            e_w = torch.abs(p) * p.grad
            e_w = e_w * (self.rho / (grad_norm + self.eps))
            p.add_(e_w)
            self.state[p] = e_w

    @torch.no_grad()
    def second_step(self):
        for p in self.model.parameters():
            if p in self.state:
                p.sub_(self.state[p])
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.state = {}

    def _grad_norm(self):
        norm = torch.norm(
            torch.stack([
                (torch.abs(p) * p.grad).norm(p=2)
                for p in self.model.parameters()
                if p.grad is not None
            ]),
            p=2
        )
        return norm


# =========================================================
# Forgetting Curriculum Tracker
# =========================================================
class ForgettingTracker:
    """
    per-sample history:
      - forget_count : correct->error 遗忘次数
      - correct_count: 累计预测正确次数
      - prev_correct : 上一次是否预测正确

    DDP 支持：
      每个 batch 先写入 local cache，
      每个 epoch 结束后 all_reduce 同步到全局。
    """

    def __init__(self, dataset_size, device):
        self.dataset_size = dataset_size
        self.device = device

        # 全局统计
        self.forget_count = torch.zeros(dataset_size, dtype=torch.float32, device=device)
        self.correct_count = torch.zeros(dataset_size, dtype=torch.float32, device=device)
        self.prev_correct = torch.full((dataset_size,), -1.0, dtype=torch.float32, device=device)

        # 每 epoch local cache（用于 DDP sync）
        self.local_seen = torch.zeros(dataset_size, dtype=torch.float32, device=device)
        self.local_forget = torch.zeros(dataset_size, dtype=torch.float32, device=device)
        self.local_correct = torch.zeros(dataset_size, dtype=torch.float32, device=device)
        self.local_prev = torch.full((dataset_size,), -1.0, dtype=torch.float32, device=device)

    @torch.no_grad()
    def update_batch(self, indices, current_correct):
        """
        在每个 batch 后调用，写入 local cache。
        indices:         [B] LongTensor
        current_correct: [B] FloatTensor, 0 or 1
        """
        for idx, corr in zip(indices, current_correct):
            i = int(idx.item())
            old_prev = self.prev_correct[i]

            new_forget = self.forget_count[i]
            if old_prev >= 0 and old_prev.item() == 1.0 and corr.item() == 0.0:
                new_forget = new_forget + 1.0

            new_correct = self.correct_count[i]
            if corr.item() == 1.0:
                new_correct = new_correct + 1.0

            self.local_seen[i] = 1.0
            self.local_forget[i] = new_forget
            self.local_correct[i] = new_correct
            self.local_prev[i] = corr

    @torch.no_grad()
    def sync_epoch(self, is_distributed):
        """
        epoch 结束后调用：
        DDP 下 all_reduce，然后写回全局统计，清空 local cache。
        """
        if is_distributed:
            dist.all_reduce(self.local_seen, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.local_forget, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.local_correct, op=dist.ReduceOp.SUM)
            # prev_correct: 每个样本只有一个 rank 更新过，sum 等价于 max
            dist.all_reduce(self.local_prev, op=dist.ReduceOp.SUM)

        seen = self.local_seen > 0
        self.forget_count[seen] = self.local_forget[seen]
        self.correct_count[seen] = self.local_correct[seen]
        self.prev_correct[seen] = self.local_prev[seen]

        # reset local cache
        self.local_seen.zero_()
        self.local_forget.zero_()
        self.local_correct.zero_()
        self.local_prev.fill_(-1.0)

    def get_batch_stats(self, indices):
        return self.forget_count[indices], self.correct_count[indices]

    def state_dict(self):
        return {
            "forget_count": self.forget_count.detach().cpu(),
            "correct_count": self.correct_count.detach().cpu(),
            "prev_correct": self.prev_correct.detach().cpu(),
        }

    def load_state_dict(self, sd):
        self.forget_count.copy_(sd["forget_count"].to(self.device))
        self.correct_count.copy_(sd["correct_count"].to(self.device))
        self.prev_correct.copy_(sd["prev_correct"].to(self.device))


# =========================================================
# Curriculum Weight Computation
# =========================================================
def compute_curriculum_weights(forget, correct_cnt, p_min=0.05, p_max=0.95, alpha=3.0):
    """
    difficulty = 0.7 * forget_norm + 0.3 * (1 - correct_norm)
    drop_prob  = p_min + (p_max - p_min) * exp(-alpha * difficulty)
    weight     = 1 - drop_prob

    difficulty=0 → drop≈p_max → weight≈p_min (简单样本低权重)
    difficulty=1 → drop≈p_min → weight≈p_max (困难样本高权重)
    """
    forget_norm = forget / (forget.max() + 1e-6)
    correct_norm = correct_cnt / (correct_cnt.max() + 1e-6)

    difficulty = 0.7 * forget_norm + 0.3 * (1.0 - correct_norm)
    drop_prob = p_min + (p_max - p_min) * torch.exp(-alpha * difficulty)
    weights = 1.0 - drop_prob
    return weights


# =========================================================
# forward helper
# =========================================================
def student_forward(model, images):
    logits, cls_token, patch_tokens = model(images)
    return logits, cls_token, patch_tokens


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
# eval
# =========================================================
@torch.no_grad()
def evaluate_loader(model, dataloader, criterion, device, epoch, split_name="Val", verbose=True):
    model.eval()
    running_loss = 0.0

    all_preds, all_labels = [], []
    domain_stats = defaultdict(lambda: {"preds": [], "labels": []})

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [{split_name}]") if verbose else dataloader

    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].float().to(device, non_blocking=True).unsqueeze(1)
        domains = batch["domain"]

        logits, _, _ = student_forward(model, images)
        loss = criterion(logits, labels)
        running_loss += loss.item()

        probs = torch.sigmoid(logits)
        probs_np = probs.cpu().numpy().flatten()
        labels_np = labels.cpu().numpy().flatten()

        all_preds.extend(probs_np.tolist())
        all_labels.extend(labels_np.tolist())

        for i in range(len(labels_np)):
            d = domains[i]
            domain_stats[d]["preds"].append(float(probs_np[i]))
            domain_stats[d]["labels"].append(float(labels_np[i]))

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    eval_loss = running_loss / max(1, len(dataloader))
    optimal_threshold, _, _ = find_optimal_threshold(all_preds, all_labels)
    metrics = compute_all_metrics(all_preds, all_labels, threshold=optimal_threshold)
    domain_metrics = compute_domain_auc(domain_stats)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1} {split_name}结果")
        print(f"{'='*60}")
        print(f"Loss: {eval_loss:.4f}")
        print(f"ACC : {metrics['accuracy']*100:.2f}%")
        print(f"PRE : {metrics['precision']:.4f}")
        print(f"REC : {metrics['recall']:.4f}")
        print(f"F1  : {metrics['f1']:.4f}")
        print(f"AUC : {metrics['auc_roc']:.4f}")
        print(f"最佳阈值: {optimal_threshold:.2f}")
        print(f"{'='*60}\n")

        print(f"[{split_name} Per-Domain AUC]")
        for d, m in domain_metrics.items():
            print(f"  {d}: {m['auc_roc']:.4f}")

    return {
        "loss": eval_loss,
        "acc": metrics["accuracy"] * 100.0,
        "auc": metrics["auc_roc"],
        "f1": metrics["f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "optimal_threshold": optimal_threshold,
        "metrics": metrics,
        "domain_metrics": domain_metrics
    }


def run_rank0_full_val_and_broadcast(
    model_for_eval, dataloader, criterion, device, epoch,
    rank, is_distributed, split_name="Val", verbose=True
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
            verbose=verbose
        )
    else:
        result = None

    result = broadcast_object(result, rank, is_distributed, device)
    return result


# =========================================================
# auto test
# =========================================================
@torch.no_grad()
def run_final_test(
    best_ckpt_path, config, criterion, device, test_loader,
    ema_enabled, save_ema, final_epoch=0
):
    print("\n" + "=" * 70)
    print("加载最佳模型并在测试集上评估")
    print("=" * 70)
    print(f"Best checkpoint: {best_ckpt_path}")

    checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=False)

    test_student_model = ForensicDinoBaseline(config).to(device)
    test_student_model.load_state_dict(checkpoint["model_state_dict"])
    test_student_model.eval()

    test_student_result = evaluate_loader(
        model=test_student_model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        epoch=final_epoch,
        split_name="Test-Student",
        verbose=True
    )

    test_ema_result = None
    if ema_enabled and save_ema and checkpoint.get("ema_model_state_dict", None) is not None:
        print("\n" + "-" * 70)
        print("加载 EMA 模型并在测试集上评估")
        print("-" * 70)

        test_ema_model = ForensicDinoBaseline(config).to(device)
        ema_state = checkpoint["ema_model_state_dict"]

        if isinstance(ema_state, dict) and "ema_state_dict" in ema_state:
            test_ema_model.load_state_dict(ema_state["ema_state_dict"])
        else:
            test_ema_model.load_state_dict(ema_state)

        test_ema_model.eval()

        test_ema_result = evaluate_loader(
            model=test_ema_model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
            epoch=final_epoch,
            split_name="Test-EMA",
            verbose=True
        )

    print("\n" + "=" * 70)
    print("最终测试结果汇总")
    print("=" * 70)
    print(f"Test-Student | ACC={test_student_result['acc']:.2f}% | "
          f"AUC={test_student_result['auc']:.4f} | "
          f"PRE={test_student_result['precision']:.4f} | "
          f"REC={test_student_result['recall']:.4f} | "
          f"F1={test_student_result['f1']:.4f}")

    if test_ema_result is not None:
        print(f"Test-EMA     | ACC={test_ema_result['acc']:.2f}% | "
              f"AUC={test_ema_result['auc']:.4f} | "
              f"PRE={test_ema_result['precision']:.4f} | "
              f"REC={test_ema_result['recall']:.4f} | "
              f"F1={test_ema_result['f1']:.4f}")
    print("=" * 70)


# =========================================================
# train one epoch (Forgetting Curriculum + ASAM)
# =========================================================
def train_one_epoch(
    model,
    ema_model,
    dataloader,
    criterion,
    criterion_none,
    optimizer,
    asam,
    device,
    epoch,
    rank,
    is_distributed,
    tracker,
    warmup_epochs,
    grad_clip=0.0,
    curriculum_mode="prob",   
    p_min=0.05,
    p_max=0.95,
    alpha=3.0,
):
    model.train()

    running_loss = 0.0
    correct = 0.0
    total = 0.0
    num_batches = 0
    last_ema_decay = None

    local_preds, local_labels, local_domains = [], [], []
    # ================= Curriculum Debug Stats =================
    forget_mean_accum = 0.0
    forget_max_global = 0.0
    correct_mean_accum = 0.0
    difficulty_mean_accum = 0.0
    difficulty_max_global = 0.0
    low_weight_ratio_accum = 0.0
    # ===========================================================

    is_warmup = (epoch < warmup_epochs)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train|{'warmup' if is_warmup else 'curriculum'}]") \
        if rank == 0 else dataloader

    weight_mean_accum = 0.0

    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].float().to(device, non_blocking=True).unsqueeze(1)
        domains = batch["domain"]
        indices = batch["index"]
        if not torch.is_tensor(indices):
            indices = torch.tensor(indices, dtype=torch.long)
        indices = indices.to(device)

        optimizer.zero_grad()

        logits, _, _ = student_forward(model, images)
        per_sample_loss = criterion_none(logits, labels).squeeze(1)  # [B]

        # ---- compute loss ----
        if is_warmup:
            loss = per_sample_loss.mean()
            weights = torch.ones_like(per_sample_loss)
        else:
            if curriculum_mode == "forget":
                forget, correct_cnt = tracker.get_batch_stats(indices)

                forget_norm = forget / (forget.max() + 1e-6)
                correct_norm = correct_cnt / (correct_cnt.max() + 1e-6)
                difficulty = 0.7 * forget_norm + 0.3 * (1.0 - correct_norm)

            elif curriculum_mode == "prob":
                with torch.no_grad():
                    probs = torch.sigmoid(logits).squeeze(1)
                    difficulty = 1.0 - 2.0 * torch.abs(probs - 0.5)
                    difficulty = torch.clamp(difficulty, 0.0, 1.0)

            else:
                raise ValueError(f"Unknown curriculum mode: {curriculum_mode}")

            # 统一计算权重
            drop_prob = p_min + (p_max - p_min) * torch.exp(-alpha * difficulty)
            weights = 1.0 - drop_prob

            loss = (per_sample_loss * weights).sum() / (weights.sum() + 1e-8)

            # ===================== DEBUG STATISTICS =====================
            if curriculum_mode == "forget":
                forget_mean_accum += forget.mean().item()
                forget_max_global = max(forget_max_global, forget.max().item())
                correct_mean_accum += correct_cnt.mean().item()

            difficulty_mean_accum += difficulty.mean().item()
            difficulty_max_global = max(difficulty_max_global, difficulty.max().item())
            low_weight_ratio_accum += (weights < 0.1).float().mean().item()
            # =============================================================
        if not is_warmup:
            weight_mean_accum += weights.mean().item()

        # ---- ASAM or normal update ----
        if asam is not None:
            # first step
            loss.backward()
            asam.first_step()

            # second forward
            optimizer.zero_grad()
            logits2, _, _ = student_forward(model, images)
            per_sample_loss2 = criterion_none(logits2, labels).squeeze(1)

            if is_warmup:
                loss2 = per_sample_loss2.mean()
            else:
                loss2 = (per_sample_loss2 * weights).sum() / (weights.sum() + 1e-8)

            loss2.backward()

            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            asam.second_step()
        else:
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        # ---- EMA update ----
        if ema_model is not None:
            last_ema_decay = ema_model.update(model)

        # ---- update tracker ----
        with torch.no_grad():
            probs_det = torch.sigmoid(logits.detach())
            preds_det = (probs_det > 0.5).float().squeeze(1)
            correct_flag = (preds_det == labels.squeeze(1)).float()
            tracker.update_batch(indices, correct_flag)

        # ---- stats ----
        running_loss += loss.item()
        probs = torch.sigmoid(logits.detach())
        preds = (probs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        num_batches += 1

        probs_np = probs.cpu().numpy().flatten().tolist()
        labels_np = labels.cpu().numpy().flatten().tolist()
        local_preds.extend(probs_np)
        local_labels.extend(labels_np)
        local_domains.extend(list(domains))

        if rank == 0:
            pbar.set_postfix({
                "loss": f"{running_loss / (batch_idx + 1):.4f}",
                "acc": f"{100. * correct / max(1, total):.2f}%",
                "w": f"{weight_mean_accum / max(1, num_batches):.3f}",
            })

    # ---- sync tracker across GPUs ----
    tracker.sync_epoch(is_distributed)

    # ---- reduce stats ----
    if is_distributed:
        stats = torch.tensor(
            [running_loss, correct, total, num_batches],
            dtype=torch.float64, device=device
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss_sum, total_correct, total_samples, total_batches = stats.tolist()
        epoch_loss = total_loss_sum / max(1.0, total_batches)
        epoch_acc = 100.0 * total_correct / max(1.0, total_samples)
    else:
        epoch_loss = running_loss / max(1, num_batches)
        epoch_acc = 100.0 * correct / max(1, total)

    # ---- gather preds for train metrics ----
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
        train_domain_metrics = compute_domain_auc(domain_stats)

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
            "phase": "warmup" if is_warmup else "curriculum",
            "mean_weight": weight_mean_accum / max(1, num_batches),
        }

        if not is_warmup:
            train_result.update({
                "forget_mean": forget_mean_accum / max(1, num_batches),
                "forget_max": forget_max_global,
                "correct_mean": correct_mean_accum / max(1, num_batches),
                "difficulty_mean": difficulty_mean_accum / max(1, num_batches),
                "difficulty_max": difficulty_max_global,
                "low_weight_ratio": low_weight_ratio_accum / max(1, num_batches),
            })

    train_result = broadcast_object(train_result, rank, is_distributed, device)
    return train_result


# =========================================================
# main
# =========================================================
def main():
    args = parse_args()
    config = load_config(args.config)

    is_distributed, rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}") if is_distributed else \
        torch.device(config["system"]["device"])

    set_seed(config["system"].get("seed", 42))

    if rank == 0:
        print("\n" + "=" * 70)
        print(" DINOv2 + Forgetting Curriculum + ASAM + EMA ")
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

    test_datasets = []

    if config.get("test_datasets", None) is not None:
        for test_cfg in config["test_datasets"]:
            dataset = ForensicImageDataset(
                json_path=test_cfg["path"],
                image_size=data_cfg.get("image_size", 224),
                mean=tuple(data_cfg.get("mean", [0.485, 0.456, 0.406])),
                std=tuple(data_cfg.get("std", [0.229, 0.224, 0.225])),
                is_train=False,
                target_domains=test_cfg.get("target_domains"),
                target_labels=test_cfg.get("target_labels"),
                target_mani_types=test_cfg.get("target_mani_types"),
                strict_mode=data_cfg.get("strict_mode", False)
            )

            test_datasets.append({
                "name": test_cfg.get("name", "test"),
                "dataset": dataset
            })


    # val/test loader only on rank0
    if rank == 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=config["system"]["num_workers"],
            pin_memory=config["system"]["pin_memory"]
        )

        if rank == 0 and len(test_datasets) > 0:
            for item in test_datasets:
                loader = DataLoader(
                    item["dataset"],
                    batch_size=config["training"]["batch_size"],
                    shuffle=False,
                    num_workers=config["system"]["num_workers"],
                    pin_memory=config["system"]["pin_memory"]
                )
                item["loader"] = loader
                
    else:
        val_loader = None

    if rank == 0:
        print_dataset_summary(train_dataset, None, name="Train")
        print_dataset_summary(val_dataset, val_loader, name="Validation")
        if len(test_datasets) > 0:
            for item in test_datasets:
                print_dataset_summary(
                    item["dataset"],
                    item.get("loader", None),
                    name=f"Test-{item['name']}"
                )

    # -------- model --------
    model = ForensicDinoBaseline(config).to(device)

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

    # -------- ASAM --------
    sam_cfg = config["training"]["optimizer"].get("sam", {})
    sam_enabled = sam_cfg.get("enabled", False)
    sam_rho = sam_cfg.get("rho", 0.2)
    base_model_for_asam = model.module if hasattr(model, "module") else model
    asam = None
    if sam_enabled:
        asam = ASAM(model=base_model_for_asam, optimizer=optimizer, rho=sam_rho)
        if rank == 0:
            print(f"[ASAM] enabled=True, rho={sam_rho}")
    else:
        if rank == 0:
            print("[ASAM] enabled=False")

    # -------- scheduler --------
    sched_cfg = config["training"]["scheduler"]
    total_epochs = config["training"]["epochs"]
    eta_min = sched_cfg.get("eta_min", 1e-6)
    flat_epochs = sched_cfg.get("flat_epochs", 0)

    if flat_epochs <= 0:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs, eta_min=eta_min
        )
    elif flat_epochs >= total_epochs:
        scheduler = optim.lr_scheduler.ConstantLR(
            optimizer, factor=1.0, total_iters=total_epochs
        )
    else:
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                optim.lr_scheduler.ConstantLR(
                    optimizer, factor=1.0, total_iters=flat_epochs
                ),
                optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=total_epochs - flat_epochs, eta_min=eta_min
                ),
            ],
            milestones=[flat_epochs]
        )

    # -------- loss --------
    criterion = nn.BCEWithLogitsLoss()
    criterion_none = nn.BCEWithLogitsLoss(reduction="none")

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
            if dynamic_decay:
                print(f"[EMA] enabled=True, dynamic_decay=True, schedule={schedule}, "
                      f"decay_start={decay_start}, decay_end={decay_end}")
            else:
                print(f"[EMA] enabled=True, dynamic_decay=False, decay={decay}")
    else:
        if rank == 0:
            print("[EMA] enabled=False")

    # -------- Curriculum config --------
    curriculum_cfg = config.get("curriculum", {})
    warmup_epochs = int(curriculum_cfg.get("warmup_epochs", 2))
    p_min = float(curriculum_cfg.get("p_min", 0.05))
    p_max = float(curriculum_cfg.get("p_max", 0.95))
    alpha = float(curriculum_cfg.get("alpha", 3.0))
    curriculum_mode = curriculum_cfg.get("mode", "forget")  

    if rank == 0:
        print("\n" + "=" * 70)
        print("Forgetting Curriculum Config")
        print("=" * 70)
        print(f"  warmup_epochs : {warmup_epochs}")
        print(f"  p_min         : {p_min}")
        print(f"  p_max         : {p_max}")
        print(f"  alpha         : {alpha}")
        print("=" * 70)

    # -------- Forgetting Tracker --------
    tracker = ForgettingTracker(dataset_size=len(train_dataset), device=device)

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
    save_dir = config.get("save_dir", "./checkpoints/dino_curriculum")
    os.makedirs(save_dir, exist_ok=True)

    start_epoch = 0
    final_epoch = 0
    best_val_auc = 0.0
    best_val_acc = 0.0
    best_threshold = 0.5

    checkpoint_path = config.get("checkpoint_path", None)
    resume = config.get("resume", False)

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        if rank == 0:
            print(f"加载检查点: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if hasattr(model, "module"):
            model.module.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt["model_state_dict"])

        if ema_enabled and ckpt.get("ema_model_state_dict") is not None:
            ema_model.load_state_dict(ckpt["ema_model_state_dict"])

        if ckpt.get("tracker_state_dict", None) is not None:
            tracker.load_state_dict(ckpt["tracker_state_dict"])

        if resume:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt["epoch"]
            best_val_acc = ckpt.get("val_acc", 0.0)
            best_val_auc = ckpt.get("val_auc", 0.0)
            best_threshold = ckpt.get("optimal_threshold", 0.5)
            if rank == 0:
                print(f"Resume from epoch {start_epoch}, best_val_auc={best_val_auc:.4f}")

    # -------- grad clip --------
    grad_clip = config["training"].get("grad_clip", 0.0)

    # =========================================================
    # Training Loop
    # =========================================================
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
            print(f"\n[Epoch {epoch+1}] train batches: {len(train_loader)} | "
                  f"phase: {'warmup' if epoch < warmup_epochs else 'curriculum'}")

        # ===== train =====
        train_result = train_one_epoch(
            model=model,
            ema_model=ema_model,
            dataloader=train_loader,
            criterion=criterion,
            criterion_none=criterion_none,
            optimizer=optimizer,
            asam=asam,
            device=device,
            epoch=epoch,
            rank=rank,
            is_distributed=is_distributed,
            tracker=tracker,
            warmup_epochs=warmup_epochs,
            grad_clip=grad_clip,
            curriculum_mode=curriculum_mode,  # NEW
            p_min=p_min,
            p_max=p_max,
            alpha=alpha,
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
            verbose=True
        )

        # ===== val EMA =====
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
                verbose=True
            )

        # choose main val
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
        metrics = main_val_result["metrics"]
        domain_metrics = main_val_result["domain_metrics"]

        scheduler.step()

        # ===== test every epoch =====
        test_student_result = None
        test_ema_result = None
        if rank == 0 and len(test_datasets) > 0:
            student_eval_model = model.module if hasattr(model, "module") else model

            for item in test_datasets:
                print(f"\nRunning Test: {item['name']}")

                evaluate_loader(
                    model=student_eval_model,
                    dataloader=item["loader"],
                    criterion=criterion,
                    device=device,
                    epoch=epoch,
                    split_name=f"Test-{item['name']}-Student",
                    verbose=True
                )

                if ema_enabled:
                    evaluate_loader(
                        model=ema_model.ema,
                        dataloader=item["loader"],
                        criterion=criterion,
                        device=device,
                        epoch=epoch,
                        split_name=f"Test-{item['name']}-EMA",
                        verbose=True
                    )

        # ===== logging =====
        if rank == 0:
            lr_info = " | ".join(
                [f"group{i}={pg['lr']:.6e}" for i, pg in enumerate(optimizer.param_groups)]
            )

            print("\n[Train Per-Domain AUC]")
            for d, m in train_domain_metrics.items():
                print(f"  {d}: {m['auc_roc']:.4f}")

            print_train_val_domain_gap(
                train_result, student_val_result,
                title="[Train vs Val-Student Domain AUC Gap]"
            )
            if ema_val_result is not None:
                print_train_val_domain_gap(
                    train_result, ema_val_result,
                    title="[Train vs Val-EMA Domain AUC Gap]"
                )

            if current_ema_decay is not None:
                print(f"\n[EMA] decay(last): {current_ema_decay:.6f}")

            print(f"\n[Curriculum]")
            print(f"  phase       : {train_result.get('phase')}")
            print(f"  mean weight : {train_result.get('mean_weight', 0.0):.4f}")
            if train_result.get("phase") == "curriculum":
                print("\n[Curriculum Debug]")
                print(f"  forget_mean      : {train_result['forget_mean']:.4f}")
                print(f"  forget_max       : {train_result['forget_max']:.4f}")
                print(f"  correct_mean     : {train_result['correct_mean']:.4f}")
                print(f"  difficulty_mean  : {train_result['difficulty_mean']:.4f}")
                print(f"  difficulty_max   : {train_result['difficulty_max']:.4f}")
                print(f"  low_weight_ratio : {train_result['low_weight_ratio']:.4f}")

            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train loss: {train_loss:.4f} | acc: {train_acc:.2f}% | auc: {train_auc:.4f}")
            print(f"  Val({main_val_name}) loss: {val_loss:.4f} | acc: {val_acc:.2f}% | auc: {val_auc:.4f}")
            print(f"  lr: {lr_info}")


            # ===== save best =====
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_val_acc = val_acc
                best_threshold = optimal_threshold

                student_state = model.module.state_dict() \
                    if hasattr(model, "module") else model.state_dict()

                ckpt = {
                    "epoch": epoch + 1,
                    "model_state_dict": student_state,
                    "ema_model_state_dict": ema_model.state_dict()
                    if (ema_enabled and save_ema) else None,
                    "tracker_state_dict": tracker.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_result": train_result,
                    "student_val_result": student_val_result,
                    "ema_val_result": ema_val_result,
                    "main_val_name": main_val_name,
                    "val_acc": val_acc,
                    "val_auc": val_auc,
                    "val_f1": val_f1,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "optimal_threshold": optimal_threshold,
                    "metrics": metrics,
                    "domain_metrics": domain_metrics,
                    "config": config,
                }
                torch.save(ckpt, os.path.join(save_dir, "best_model.pth"))
                print(f"✓ Best model saved! ({main_val_name} AUC: {val_auc:.4f})")

            # ===== periodic save =====
            save_freq = config.get("logging", {}).get("save_freq", 5)
            if (epoch + 1) % save_freq == 0:
                student_state = model.module.state_dict() \
                    if hasattr(model, "module") else model.state_dict()
                ckpt = {
                    "epoch": epoch + 1,
                    "model_state_dict": student_state,
                    "ema_model_state_dict": ema_model.state_dict()
                    if (ema_enabled and save_ema) else None,
                    "tracker_state_dict": tracker.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_result": train_result,
                    "student_val_result": student_val_result,
                    "ema_val_result": ema_val_result,
                    "main_val_name": main_val_name,
                    "val_acc": val_acc,
                    "val_auc": val_auc,
                    "val_f1": val_f1,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "optimal_threshold": optimal_threshold,
                    "metrics": metrics,
                    "domain_metrics": domain_metrics,
                    "config": config,
                }
                ckpt_name = f"checkpoint_epoch_{epoch+1}.pth"
                torch.save(ckpt, os.path.join(save_dir, ckpt_name))
                print(f"💾 Checkpoint saved: {ckpt_name}")

        # ===== early stopping =====
        if early_stopper is not None:
            monitor = es_cfg.get("monitor", "val_auc")
            monitor_map = {
                "val_acc": val_acc,
                "val_auc": val_auc,
                "val_f1": val_f1,
                "val_loss": val_loss
            }
            current_score = monitor_map.get(monitor, val_auc)
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
        print("\n" + "=" * 60)
        print("训练完成!")
        print(f"最佳验证准确率: {best_val_acc:.2f}%")
        print(f"最佳验证AUC: {best_val_auc:.4f}")
        print(f"最佳阈值: {best_threshold:.2f}")
        print("=" * 60)

        if rank == 0 and len(test_datasets) > 0:
            best_ckpt_path = os.path.join(save_dir, "best_model.pth")
            if os.path.exists(best_ckpt_path):
                for item in test_datasets:
                    print(f"\nFinal Test: {item['name']}")
                    run_final_test(
                        best_ckpt_path=best_ckpt_path,
                        config=config,
                        criterion=criterion,
                        device=device,
                        test_loader=item["loader"],
                        ema_enabled=ema_enabled,
                        save_ema=save_ema,
                        final_epoch=final_epoch
                    )
                    
            
if __name__ == "__main__":
    main()