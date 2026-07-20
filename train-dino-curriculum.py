import os
import yaml
import argparse
import random
import numpy as np
from collections import defaultdict
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from pre_data.dino_dataprocess import ForensicImageDataset, print_dataset_summary
from model.dino_baseline import ForensicDinoBaseline


# ------------------------- utils -------------------------
def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="DINOv2 train with EMA + Loss Profile Curriculum + Auto Test")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args()


def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])

        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    else:
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


# ------------------------- early stop -------------------------
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, monitor='val_auc', verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.verbose = verbose

        self.mode = 'min' if 'loss' in monitor else 'max'
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.mode == 'max':
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


# ------------------------- metrics -------------------------
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


# ------------------------- optimizer -------------------------
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


# ------------------------- EMA -------------------------
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


# ------------------------- loss profile curriculum -------------------------
class LossProfileCurriculumManager:
    def __init__(
        self,
        dataset,
        top_ratio=0.5,
        small_weight=0.2,
        domain_names=None,
        enable_after_profiling=True,
        curriculum_epochs=6,
        rank=0
    ):
        self.dataset = dataset
        self.top_ratio = top_ratio
        self.small_weight = small_weight
        self.enable_after_profiling = enable_after_profiling
        self.curriculum_epochs = curriculum_epochs
        self.rank = rank

        self.n_total = len(dataset)
        self.domain_indices = defaultdict(list)

        for idx in range(self.n_total):
            d = dataset.get_domain(idx)
            self.domain_indices[d].append(idx)

        if domain_names is not None:
            self.domain_names = [d for d in domain_names if d in self.domain_indices]
        else:
            self.domain_names = sorted(self.domain_indices.keys())

        self.sample_losses = np.zeros(self.n_total, dtype=np.float32)
        self.sample_weights = np.ones(self.n_total, dtype=np.float32)
        self.is_ready = False

        if self.rank == 0:
            print("\n" + "=" * 72)
            print("LossProfileCurriculumManager")
            print("=" * 72)
            print(f"  top_ratio         : {self.top_ratio:.2f}")
            print(f"  small_weight      : {self.small_weight:.3f}")
            print(f"  curriculum_epochs : {self.curriculum_epochs}")
            print(f"  策略              : 前期每域 top-k 高loss样本权重=1，其余样本保留小权重")
            print("=" * 72)

    def is_active(self, epoch):
        return epoch < self.curriculum_epochs

    def update_sample_losses(self, sample_losses_dict):
        for idx, loss_val in sample_losses_dict.items():
            self.sample_losses[int(idx)] = float(loss_val)

        self._rebuild_weights()
        self.is_ready = True

    def _rebuild_weights(self):
        weights = np.full(self.n_total, self.small_weight, dtype=np.float32)
        domain_summary = {}

        for d in self.domain_names:
            indices = self.domain_indices[d]
            if len(indices) == 0:
                continue

            domain_losses = self.sample_losses[indices]
            order = np.argsort(-domain_losses)  # descending
            k = max(1, int(len(indices) * self.top_ratio))

            selected_local = order[:k]
            selected_global = [indices[i] for i in selected_local]

            weights[selected_global] = 1.0

            domain_summary[d] = {
                "total": len(indices),
                "selected": len(selected_global),
                "mean_loss": float(domain_losses.mean()) if len(domain_losses) > 0 else 0.0,
                "max_loss": float(domain_losses.max()) if len(domain_losses) > 0 else 0.0,
            }

        self.sample_weights = weights

        if self.rank == 0:
            print("\n[LossProfileCurriculum] 样本权重已重建")
            for d in self.domain_names:
                if d in domain_summary:
                    info = domain_summary[d]
                    print(
                        f"  {d}: selected={info['selected']}/{info['total']} "
                        f"| mean_loss={info['mean_loss']:.4f} | max_loss={info['max_loss']:.4f}"
                    )

    def get_weight_for_indices(self, indices):
        return self.sample_weights[indices]

    def is_enabled_for_training(self, epoch):
        return self.enable_after_profiling and self.is_ready and self.is_active(epoch)


def create_loss_profile_curriculum_manager(dataset, config, rank=0):
    cfg = config.get("loss_profile_curriculum", {})
    enabled = cfg.get("enabled", False)

    if not enabled:
        if rank == 0:
            print("[LossProfileCurriculum] 未启用")
        return None

    return LossProfileCurriculumManager(
        dataset=dataset,
        top_ratio=cfg.get("top_ratio", 0.5),
        small_weight=cfg.get("small_weight", 0.2),
        domain_names=cfg.get("domain_names", None),
        enable_after_profiling=cfg.get("enable_after_profiling", True),
        curriculum_epochs=cfg.get("curriculum_epochs", 6),
        rank=rank
    )


# ------------------------- profiling -------------------------
@torch.no_grad()
def profile_full_train_losses(model_for_eval, dataloader, criterion_none, device, rank=0, split_name="Train-LossProfile"):
    model_for_eval.eval()
    sample_loss_map = {}

    pbar = tqdm(dataloader, desc=split_name) if rank == 0 else dataloader

    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].float().to(device, non_blocking=True).unsqueeze(1)
        indices = batch["index"]

        logits, _, _ = model_for_eval(images)
        sample_losses = criterion_none(logits, labels).squeeze(1).detach().cpu().numpy()

        if torch.is_tensor(indices):
            indices = indices.cpu().numpy().tolist()
        else:
            indices = list(indices)

        for idx, loss_val in zip(indices, sample_losses.tolist()):
            sample_loss_map[int(idx)] = float(loss_val)

    return sample_loss_map


# ------------------------- train/eval -------------------------
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
    else:
        return None, None, None


def train_one_epoch(model, ema_model, dataloader, criterion_none, optimizer,
                    device, epoch, rank, is_distributed,
                    loss_profile_manager=None,
                    grad_clip=0.0):

    model.train()
    running_loss = 0.0
    correct = 0.0
    total = 0.0
    num_batches = 0

    local_preds, local_labels, local_domains = [], [], []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]") if rank == 0 else dataloader
    last_ema_decay = None
    weighted_batches = 0

    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].float().to(device, non_blocking=True).unsqueeze(1)
        domains = batch["domain"]
        indices = batch["index"]

        optimizer.zero_grad()
        logits, _, _ = model(images)

        sample_losses = criterion_none(logits, labels).squeeze(1)

        if loss_profile_manager is not None and loss_profile_manager.is_enabled_for_training(epoch):
            if torch.is_tensor(indices):
                indices_np = indices.cpu().numpy()
            else:
                indices_np = np.array(indices)

            sample_weights = loss_profile_manager.get_weight_for_indices(indices_np)
            sample_weights = torch.tensor(sample_weights, dtype=sample_losses.dtype, device=sample_losses.device)

            loss = (sample_losses * sample_weights).sum() / (sample_weights.sum() + 1e-8)
            weighted_batches += 1
        else:
            loss = sample_losses.mean()

        loss.backward()

        if grad_clip is not None and grad_clip > 0:
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
            "weighted_training": (loss_profile_manager is not None and loss_profile_manager.is_enabled_for_training(epoch)),
            "weighted_batches": weighted_batches,
        }

    train_result = broadcast_object(train_result, rank, is_distributed, device)
    return train_result


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

        logits, _, _ = model(images)
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


def run_rank0_full_val_and_broadcast(model_for_eval, dataloader, criterion, device, epoch, rank, is_distributed, split_name="Val", verbose=True):
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


# ------------------------- auto test -------------------------
@torch.no_grad()
def run_final_test(best_ckpt_path, config, criterion, device, test_loader, ema_enabled, save_ema, final_epoch=0):
    print("\n" + "=" * 70)
    print("加载最佳模型并在测试集上评估")
    print("=" * 70)
    print(f"Best checkpoint: {best_ckpt_path}")

    checkpoint = torch.load(best_ckpt_path, map_location=device)

    # Student
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


# ------------------------- main -------------------------
def main():
    args = parse_args()
    config = load_config(args.config)

    is_distributed, rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}") if is_distributed else torch.device(config["system"]["device"])

    set_seed(config["system"].get("seed", 42))

    if rank == 0:
        print("\n" + "=" * 70)
        print(" DINOv2 + EMA + Loss Profile Curriculum + Auto Test ")
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

    loss_profile_manager = create_loss_profile_curriculum_manager(train_dataset, config, rank=rank)

    if rank == 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=config["system"]["num_workers"],
            pin_memory=config["system"]["pin_memory"]
        )

        train_profile_loader = DataLoader(
            train_dataset,
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
        train_profile_loader = None
        test_loader = None

    if rank == 0:
        print_dataset_summary(train_dataset, None, name="Train")
        print_dataset_summary(val_dataset, val_loader, name="Validation")
        if test_dataset is not None:
            print_dataset_summary(test_dataset, test_loader, name="Test")

    # -------- model --------
    model = ForensicDinoBaseline(config).to(device)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")

        if hasattr(model, "backbone"):
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
                optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=flat_epochs),
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
        if ema_enabled:
            if dynamic_decay:
                print(f"[EMA] enabled=True, dynamic_decay=True, schedule={schedule}, decay_start={decay_start}, decay_end={decay_end}")
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
    save_dir = config.get("save_dir", "./checkpoints/dino_loss_profile_curriculum")
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
            best_threshold = checkpoint.get("optimal_threshold", 0.5)

    # -------- train loop --------
    grad_clip = config["training"].get("grad_clip", 0.0)

    for epoch in range(start_epoch, config["training"]["epochs"]):
        final_epoch = epoch + 1

        use_loss_profile_curriculum = (
            loss_profile_manager is not None and
            loss_profile_manager.is_active(epoch)
        )

        # ========== 每轮开始先全训练集 profiling ==========
        if use_loss_profile_curriculum:
            if rank == 0:
                profile_model = ema_model.ema if (ema_enabled and use_ema_for_val and ema_model is not None) else (model.module if hasattr(model, "module") else model)

                sample_loss_map = profile_full_train_losses(
                    model_for_eval=profile_model,
                    dataloader=train_profile_loader,
                    criterion_none=criterion_none,
                    device=device,
                    rank=rank,
                    split_name=f"Epoch {epoch+1} [Train-LossProfile]"
                )
            else:
                sample_loss_map = None

            sample_loss_map = broadcast_object(sample_loss_map, rank, is_distributed, device)
            loss_profile_manager.update_sample_losses(sample_loss_map)

        # ========== 第 1 轮只 profiling，不训练 ==========
        if use_loss_profile_curriculum and epoch == 0:
            if rank == 0:
                print(f"\n[Epoch {epoch+1}] 仅执行全训练集 loss profiling，不进行梯度更新。")

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

            main_val_result = ema_val_result if (ema_enabled and use_ema_for_val and ema_val_result is not None) else student_val_result
            val_auc = main_val_result["auc"]
            val_acc = main_val_result["acc"]
            optimal_threshold = main_val_result["optimal_threshold"]

            if rank == 0:
                print(f"\n[Epoch {epoch+1}] Profiling 完成，下一轮开始使用 loss-profile curriculum 训练。")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_val_acc = val_acc
                best_threshold = optimal_threshold

            scheduler.step()
            continue

        # ========== 训练 ==========
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
            if use_loss_profile_curriculum:
                print(f"\n[Epoch {epoch+1}] 当前训练批次数: {len(train_loader)} | sampler=BaselineSampler + LossWeightedCurriculum")
            else:
                print(f"\n[Epoch {epoch+1}] 当前训练批次数: {len(train_loader)} | sampler=BaselineSampler")

        train_result = train_one_epoch(
            model=model,
            ema_model=ema_model,
            dataloader=train_loader,
            criterion_none=criterion_none,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            rank=rank,
            is_distributed=is_distributed,
            loss_profile_manager=loss_profile_manager if use_loss_profile_curriculum else None,
            grad_clip=grad_clip
        )

        # ---------- val student ----------
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

        # ---------- val ema ----------
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

        # ---------- choose main val ----------
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
        weighted_training = train_result.get("weighted_training", False)

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

        if rank == 0:
            lr_info = " | ".join([f"group{i}={pg['lr']:.6e}" for i, pg in enumerate(optimizer.param_groups)])

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
                print(f"\n[EMA]")
                print(f"  EMA decay (last step): {current_ema_decay:.6f}")

            print(f"\nEpoch {epoch+1} 总结:")
            print(f"  是否使用加权课程训练: {weighted_training}")
            print(f"  训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.2f}%")
            print(f"  训练AUC: {train_auc:.4f} | PRE: {train_precision:.4f} | REC: {train_recall:.4f} | F1: {train_f1:.4f}")

            print(f"  Val-Student 损失: {student_val_result['loss']:.4f} | 准确率: {student_val_result['acc']:.2f}%")
            print(f"  Val-Student AUC: {student_val_result['auc']:.4f} | PRE: {student_val_result['precision']:.4f} | REC: {student_val_result['recall']:.4f} | F1: {student_val_result['f1']:.4f}")

            if ema_val_result is not None:
                print(f"  Val-EMA 损失: {ema_val_result['loss']:.4f} | 准确率: {ema_val_result['acc']:.2f}%")
                print(f"  Val-EMA AUC: {ema_val_result['auc']:.4f} | PRE: {ema_val_result['precision']:.4f} | REC: {ema_val_result['recall']:.4f} | F1: {ema_val_result['f1']:.4f}")

            print(f"  主验证指标来源: {main_val_name}")
            print(f"  学习率: {lr_info}")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_val_acc = val_acc
                best_threshold = optimal_threshold

                student_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                checkpoint = {
                    "epoch": epoch + 1,
                    "model_state_dict": student_state,
                    "ema_model_state_dict": ema_model.state_dict() if (ema_enabled and save_ema) else None,
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
                torch.save(checkpoint, os.path.join(save_dir, "best_model.pth"))
                print(f"✓ 最佳模型已保存! ({main_val_name} AUC: {val_auc:.4f})")

            save_freq = config.get("logging", {}).get("save_freq", 5)
            if (epoch + 1) % save_freq == 0:
                student_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                checkpoint = {
                    "epoch": epoch + 1,
                    "model_state_dict": student_state,
                    "ema_model_state_dict": ema_model.state_dict() if (ema_enabled and save_ema) else None,
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
                torch.save(checkpoint, os.path.join(save_dir, ckpt_name))
                print(f"💾 检查点已保存: {ckpt_name}")

        if early_stopper is not None:
            monitor = es_cfg.get("monitor", "val_auc")
            monitor_map = {
                "val_acc": val_acc,
                "val_auc": val_auc,
                "val_f1": val_f1,
                "val_loss": val_loss,
            }
            current_score = monitor_map.get(monitor, val_auc)
            should_stop = early_stopper(current_score, epoch)

            if is_distributed:
                stop_tensor = torch.tensor([1.0 if should_stop else 0.0], device=device)
                dist.broadcast(stop_tensor, src=0)
                should_stop = stop_tensor.item() > 0.5

            if should_stop:
                if rank == 0:
                    print(f"\n训练提前停止于 epoch {epoch+1}")
                break

    cleanup_distributed()

    if rank == 0:
        print("\n" + "=" * 60)
        print("训练完成!")
        print(f"最佳验证准确率: {best_val_acc:.2f}%")
        print(f"最佳验证AUC: {best_val_auc:.4f}")
        print(f"最佳阈值: {best_threshold:.2f}")
        print("=" * 60)

        # -------- auto test --------
        if test_dataset is not None and test_loader is not None:
            best_ckpt_path = os.path.join(save_dir, "best_model.pth")
            if os.path.exists(best_ckpt_path):
                run_final_test(
                    best_ckpt_path=best_ckpt_path,
                    config=config,
                    criterion=criterion,
                    device=device,
                    test_loader=test_loader,
                    ema_enabled=ema_enabled,
                    save_ema=save_ema,
                    final_epoch=final_epoch
                )
            else:
                print(f"[Warning] 未找到 best_model.pth，跳过测试: {best_ckpt_path}")


if __name__ == "__main__":
    main()