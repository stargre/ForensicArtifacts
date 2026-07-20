import os
import sys
import json
import yaml
import argparse
import random
import numpy as np
from collections import defaultdict
import copy
import traceback
from contextlib import contextmanager
from datetime import timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

import pre_data.dino_dataprocess as dino_dp_module
from pre_data.dino_dataprocess import ForensicImageDataset, print_dataset_summary
from model.dino_soft_topx import ForensicDinoSoftTopX
from curriculum.domainweighted_curriculum_management import DomainWeightedCurriculumManager


# =========================================================
# utils
# =========================================================
def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="DINOv2 stochastic shortcut suppression with configurable position"
    )
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args()


def setup_distributed(timeout_minutes=180):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])

        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank,
            timeout=timedelta(minutes=timeout_minutes)
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


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def state_dict_to_cpu(state_dict):
    new_state = {}
    for k, v in state_dict.items():
        if torch.is_tensor(v):
            new_state[k] = v.detach().cpu()
        else:
            new_state[k] = v
    return new_state


def get_model_state_dict(model):
    base_model = model.module if hasattr(model, "module") else model
    return base_model.state_dict()


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    raise TypeError(f"Type not serializable: {type(obj)}")


def save_json(data, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=json_default)


def get_test_dataset_cfg(config):
    if config.get("test_dataset", None) is not None:
        return config["test_dataset"]
    if config.get("test_datasets", None) is not None:
        return config["test_datasets"]
    return {}


def get_loss_cfg(config):
    loss_cfg = config.get("loss", {})
    default_cfg = {
        "lambda_cls": 1.0,
        "lambda_cons": 0.05,
        "cons_temperature": 2.0,
    }
    for k, v in default_cfg.items():
        loss_cfg.setdefault(k, v)
    return loss_cfg


def get_domain_targets(batch, device):
    if "domain_label" in batch:
        t = batch["domain_label"]
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=torch.long)
        return t.long().to(device, non_blocking=True)
    raise ValueError("当前 routing 版本要求 dataset 返回 domain_label")


def get_domain_keys_from_batch(batch):
    if "domain" in batch:
        vals = batch["domain"]
        if isinstance(vals, (list, tuple)):
            return list(vals)
        if torch.is_tensor(vals):
            return vals.detach().cpu().tolist()
        return [vals]
    elif "domain_label" in batch:
        vals = batch["domain_label"]
        if torch.is_tensor(vals):
            return vals.detach().cpu().tolist()
        return list(vals)
    else:
        raise ValueError("batch 中缺少 domain 或 domain_label")


def get_routing_cfg(config):
    routing_cfg = config.get("routing", {})
    defaults = {
        "topx_ratio": 0.2,
        "score_mode": "diff",
        "alpha": 1.0,

        "probe_epochs": 5,
        "probe_batch_size": 512,
        "probe_lr": 1.0e-3,
        "probe_weight_decay": 1.0e-4,
        "probe_hidden_dim": 0,
        "probe_dropout": 0.0,

        "score_batch_size": None,
        "use_train_aug_for_score": False,

        "extra_probe_keys": ["ori_dataset", "real_source"],

        "stochastic_enabled": True,
        "dual_view_train": True,
        "drop_ratio": 0.03,
        "drop_beta": 0.3,
        "drop_active_p": 0.5,
        "prob_uniform_mix": 0.1,

        "suppress_position": "before_pool",
        "suppress_block_index": 11,
    }
    for k, v in defaults.items():
        routing_cfg.setdefault(k, v)

    if routing_cfg["score_batch_size"] is None:
        routing_cfg["score_batch_size"] = routing_cfg["probe_batch_size"]

    return routing_cfg


@contextmanager
def suppress_stdout_only(enabled=True):
    if not enabled:
        yield
        return

    old_stdout = sys.stdout
    devnull = open(os.devnull, "w")
    try:
        sys.stdout = devnull
        yield
    finally:
        sys.stdout = old_stdout
        devnull.close()


class SilentTqdm:
    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable if iterable is not None else []

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, *args, **kwargs):
        pass

    def close(self):
        pass

    def update(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


@contextmanager
def suppress_dataset_logs(enabled=True):
    if not enabled:
        yield
        return

    old_stdout = sys.stdout
    devnull = open(os.devnull, "w")
    old_tqdm = dino_dp_module.tqdm

    try:
        sys.stdout = devnull
        dino_dp_module.tqdm = SilentTqdm
        yield
    finally:
        sys.stdout = old_stdout
        dino_dp_module.tqdm = old_tqdm
        devnull.close()


def build_dataset(
    dataset_cfg,
    data_cfg,
    is_train,
    rank=0,
    silent_nonzero_rank=True,
):
    silent = (rank != 0 and silent_nonzero_rank)

    with suppress_dataset_logs(silent):
        dataset = ForensicImageDataset(
            json_path=dataset_cfg["path"],
            image_size=data_cfg.get("image_size", 224),
            mean=tuple(data_cfg.get("mean", [0.485, 0.456, 0.406])),
            std=tuple(data_cfg.get("std", [0.229, 0.224, 0.225])),
            is_train=is_train,
            target_domains=dataset_cfg.get("target_domains"),
            target_labels=dataset_cfg.get("target_labels"),
            target_mani_types=dataset_cfg.get("target_mani_types"),
            strict_mode=data_cfg.get("strict_mode", False)
        )
    return dataset


def get_routing_sync_path(save_dir):
    return os.path.join(save_dir, "routing_sync.pt")


def save_routing_sync_file(save_dir, shortcut_mask_cpu, routing_info):
    ensure_dir(save_dir)

    drop_probs = routing_info.get("drop_probs", None)
    if drop_probs is None:
        raise ValueError("routing_info 中缺少 drop_probs，无法同步给其他 rank")

    pack = {
        "shortcut_mask": shortcut_mask_cpu.detach().cpu()
        if torch.is_tensor(shortcut_mask_cpu) else torch.tensor(shortcut_mask_cpu, dtype=torch.float32),
        "drop_probs": torch.tensor(drop_probs, dtype=torch.float32),
        "routing_info": routing_info,
    }

    sync_path = get_routing_sync_path(save_dir)
    torch.save(pack, sync_path)
    return sync_path


def load_routing_sync_file(save_dir, map_location="cpu"):
    sync_path = get_routing_sync_path(save_dir)
    if not os.path.exists(sync_path):
        raise FileNotFoundError(f"routing sync file 不存在: {sync_path}")
    return torch.load(sync_path, map_location=map_location, weights_only=False)


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

    try:
        ll = log_loss(all_labels, np.clip(all_preds, 1e-7, 1 - 1e-7), labels=[0, 1])
    except Exception:
        ll = 0.0

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
        "log_loss": ll,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "threshold": float(threshold),
    }
    return metrics


def compute_domain_metrics(domain_stats, threshold=0.5):
    domain_metrics = {}
    for d, stats in sorted(domain_stats.items()):
        d_preds = np.array(stats["preds"])
        d_labels = np.array(stats["labels"])

        if len(d_labels) == 0:
            continue

        d_metrics = compute_all_metrics(d_preds, d_labels, threshold=threshold)
        domain_metrics[d] = d_metrics

    return domain_metrics


def compute_domain_auc(domain_stats, threshold=0.5):
    return compute_domain_metrics(domain_stats, threshold=threshold)


def print_full_metrics(metrics, prefix=""):
    print(f"{prefix}Accuracy           : {metrics['accuracy'] * 100:.2f}%")
    print(f"{prefix}Balanced Accuracy  : {metrics['balanced_accuracy'] * 100:.2f}%")
    print(f"{prefix}AUC-ROC            : {metrics['auc_roc']:.4f}")
    print(f"{prefix}AUC-PR             : {metrics['auc_pr']:.4f}")
    print(f"{prefix}F1 Score           : {metrics['f1']:.4f}")
    print(f"{prefix}Precision          : {metrics['precision']:.4f}")
    print(f"{prefix}Recall             : {metrics['recall']:.4f}")
    print(f"{prefix}Specificity        : {metrics['specificity']:.4f}")
    print(f"{prefix}MCC                : {metrics['mcc']:.4f}")


def print_full_domain_metrics(domain_metrics, title="Per-Domain Metrics"):
    print(f"{title}")
    if domain_metrics is None or len(domain_metrics) == 0:
        print("  (empty)")
        return

    for d, m in sorted(domain_metrics.items()):
        print(f"  [{d}]")
        print(f"    Accuracy           : {m['accuracy'] * 100:.2f}%")
        print(f"    Balanced Accuracy  : {m['balanced_accuracy'] * 100:.2f}%")
        print(f"    AUC-ROC            : {m['auc_roc']:.4f}")
        print(f"    AUC-PR             : {m['auc_pr']:.4f}")
        print(f"    F1 Score           : {m['f1']:.4f}")
        print(f"    Precision          : {m['precision']:.4f}")
        print(f"    Recall             : {m['recall']:.4f}")
        print(f"    Specificity        : {m['specificity']:.4f}")
        print(f"    MCC                : {m['mcc']:.4f}")


def broadcast_object(obj, rank, is_distributed, device):
    if not is_distributed:
        return obj
    obj_list = [obj if rank == 0 else None]
    dist.broadcast_object_list(obj_list, src=0, device=get_broadcast_device(device))
    return obj_list[0]


# =========================================================
# SAM optimizer
# =========================================================
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer_cls, rho=0.05, adaptive=False, **kwargs):
        if rho < 0.0:
            raise ValueError(f"Invalid rho, should be non-negative: {rho}")

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

        self.is_sam = True
        self.rho = rho
        self.adaptive = adaptive

    @torch.no_grad()
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norms = []

        for group in self.param_groups:
            adaptive = group.get("adaptive", self.adaptive)
            for p in group["params"]:
                if p.grad is None:
                    continue
                if adaptive:
                    scale = torch.abs(p)
                    norms.append((scale * p.grad).norm(p=2).to(shared_device))
                else:
                    norms.append(p.grad.norm(p=2).to(shared_device))

        if len(norms) == 0:
            return torch.tensor(0.0, device=shared_device)

        return torch.norm(torch.stack(norms), p=2)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        if grad_norm.item() == 0.0:
            if zero_grad:
                self.zero_grad()
            return

        for group in self.param_groups:
            rho = group.get("rho", self.rho)
            adaptive = group.get("adaptive", self.adaptive)
            scale = rho / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()

                if adaptive:
                    e_w = torch.pow(p, 2) * p.grad * scale
                else:
                    e_w = p.grad * scale

                p.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if "old_p" not in self.state[p]:
                    continue
                p.data = self.state[p]["old_p"]

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            raise RuntimeError("SAM requires closure, but current code should use first_step/second_step explicitly.")
        loss = closure()
        self.first_step(zero_grad=True)
        closure()
        self.second_step(zero_grad=True)
        return loss

    def zero_grad(self, set_to_none=False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "base_optimizer": self.base_optimizer.state_dict(),
            "param_groups": self.param_groups,
            "defaults": self.defaults,
            "sam_rho": self.rho,
            "sam_adaptive": self.adaptive,
        }

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict["base_optimizer"])
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(state_dict.get("defaults", {}))
        self.rho = state_dict.get("sam_rho", self.rho)
        self.adaptive = state_dict.get("sam_adaptive", self.adaptive)


def is_sam_optimizer(optimizer):
    return getattr(optimizer, "is_sam", False)


def build_optimizer(model, config, rank=0):
    opt_cfg = config["training"]["optimizer"]

    base_lr = opt_cfg["lr"]
    backbone_lr = opt_cfg.get("backbone_lr", base_lr * 0.1)
    weight_decay = opt_cfg.get("weight_decay", 0.05)
    betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
    optimizer_name = opt_cfg.get("name", "AdamW").lower()

    sam_cfg = opt_cfg.get("sam", {})
    sam_enabled = bool(sam_cfg.get("enabled", False))
    sam_rho = float(sam_cfg.get("rho", 0.05))
    sam_adaptive = bool(sam_cfg.get("adaptive", False))

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
            "weight_decay": weight_decay,
            "rho": sam_rho,
            "adaptive": sam_adaptive,
        })

    if len(backbone_params) > 0:
        param_groups.append({
            "params": backbone_params,
            "lr": backbone_lr,
            "weight_decay": weight_decay,
            "rho": sam_rho,
            "adaptive": sam_adaptive,
        })

    if optimizer_name != "adamw":
        raise ValueError(f"当前只支持 AdamW / SAM+AdamW，收到 optimizer.name={opt_cfg.get('name')}")

    if sam_enabled:
        optimizer = SAM(
            param_groups,
            base_optimizer_cls=optim.AdamW,
            rho=sam_rho,
            adaptive=sam_adaptive,
            betas=betas,
        )
    else:
        optimizer = optim.AdamW(param_groups, betas=betas)

    if rank == 0:
        print(f"[Optimizer] base name      = AdamW")
        print(f"[Optimizer] head/other lr  = {base_lr}")
        print(f"[Optimizer] backbone lr    = {backbone_lr}")
        print(f"[Optimizer] weight_decay   = {weight_decay}")
        print(f"[Optimizer] trainable groups: other={len(other_params)}, backbone={len(backbone_params)}")
        print(f"[Optimizer] SAM enabled    = {sam_enabled}")
        if sam_enabled:
            print(f"[Optimizer] SAM rho        = {sam_rho}")
            print(f"[Optimizer] SAM adaptive   = {sam_adaptive}")

    return optimizer


# =========================================================
# routing / probe
# =========================================================
class LinearOrMLPProbe(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=0, dropout=0.0):
        super().__init__()
        if hidden_dim is None or hidden_dim <= 0:
            self.net = nn.Linear(in_dim, out_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )

    def forward(self, x):
        return self.net(x)


def build_inverse_freq_weights(targets, num_classes):
    counts = torch.bincount(targets.long(), minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)
    weights = 1.0 / counts
    weights = weights / weights.mean()
    return weights


def pretty_probe_name(key):
    return str(key)


def normalize_text_label(v):
    if v is None:
        return None
    v = str(v).strip()
    if v == "":
        return None
    if v.lower() in ["unknown", "none", "null", "nan"]:
        return None
    return v


def build_string_label_mapping(raw_values):
    valid_values = [normalize_text_label(v) for v in raw_values]
    valid_values = [v for v in valid_values if v is not None]
    uniq = sorted(set(valid_values))
    return {v: i for i, v in enumerate(uniq)}


def encode_string_labels(raw_values, mapping):
    encoded = []
    for v in raw_values:
        v = normalize_text_label(v)
        if v is None:
            encoded.append(-1)
        else:
            encoded.append(mapping.get(v, -1))
    return torch.tensor(encoded, dtype=torch.long)

def merge_pooled_score_to_channel_level(score_vec, embed_dim, pooling_type, use_reg_token=False):
    score_vec = score_vec.view(-1).float()

    if pooling_type == "cls":
        num_parts = 1
    elif pooling_type == "patch_mean":
        num_parts = 1
    elif pooling_type == "cls_patch_mean":
        num_parts = 2
    else:
        raise ValueError(f"Unknown pooling_type: {pooling_type}")

    if use_reg_token:
        num_parts += 1

    expected_dim = embed_dim * num_parts
    if score_vec.numel() != expected_dim:
        raise ValueError(
            f"score dim={score_vec.numel()} != expected_dim={expected_dim} "
            f"(embed_dim={embed_dim}, pooling_type={pooling_type}, use_reg_token={use_reg_token})"
        )

    merged = torch.zeros(embed_dim, dtype=score_vec.dtype, device=score_vec.device)
    for i in range(num_parts):
        merged += score_vec[i * embed_dim:(i + 1) * embed_dim]

    return merged

def extract_feature_bank(
    model,
    dataloader,
    device,
    rank=0,
    desc="Extract Feature Bank",
    extra_keys=None
):
    model.eval()

    feats = []
    labels = []
    domains = []

    extra_keys = extra_keys or []
    extra_raw = {k: [] for k in extra_keys}

    pbar = tqdm(dataloader, desc=desc) if rank == 0 else dataloader

    with torch.no_grad():
        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            pooled_feat, _, _ = model.extract_pooled_features(images)

            feats.append(pooled_feat.detach().cpu())
            labels.append(batch["label"].long().cpu())
            domains.append(get_domain_targets(batch, device=torch.device("cpu")).cpu())

            bs = images.size(0)
            for k in extra_keys:
                vals = batch.get(k, None)

                if vals is None:
                    extra_raw[k].extend([None] * bs)
                elif isinstance(vals, (list, tuple)):
                    extra_raw[k].extend([normalize_text_label(v) for v in vals])
                else:
                    extra_raw[k].extend([normalize_text_label(vals)] * bs)

    feats = torch.cat(feats, dim=0).float()
    labels = torch.cat(labels, dim=0).long()
    domains = torch.cat(domains, dim=0).long()

    return feats, labels, domains, extra_raw


def train_probe_on_feature_bank(
    features,
    targets,
    probe,
    task_type,
    device,
    epochs=5,
    batch_size=512,
    lr=1e-3,
    weight_decay=1e-4,
    rank=0
):
    probe = probe.to(device)
    probe.train()

    optimizer = optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    num_samples = features.size(0)

    if task_type == "binary":
        pos = max(1, int((targets == 1).sum().item()))
        neg = max(1, int((targets == 0).sum().item()))
        pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif task_type == "multiclass":
        num_classes = int(targets.max().item()) + 1
        class_weights = build_inverse_freq_weights(targets, num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    for ep in range(epochs):
        perm = torch.randperm(num_samples)
        total_loss = 0.0
        total_steps = 0

        for start in range(0, num_samples, batch_size):
            idx = perm[start:start + batch_size]
            x = features[idx].to(device, non_blocking=True)

            if task_type == "binary":
                y = targets[idx].float().unsqueeze(1).to(device, non_blocking=True)
                logits = probe(x)
                loss = criterion(logits, y)
            else:
                y = targets[idx].long().to(device, non_blocking=True)
                logits = probe(x)
                loss = criterion(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_steps += 1

        if rank == 0:
            print(f"[Probe-{task_type}] epoch {ep+1}/{epochs} | loss={total_loss/max(1,total_steps):.4f}")

    probe.eval()
    return probe


@torch.no_grad()
def evaluate_probe_on_feature_bank(features, targets, probe, task_type, device, batch_size=512):
    probe.eval()

    all_logits = []
    all_targets = []
    num_samples = features.size(0)

    running_loss = 0.0
    steps = 0

    if task_type == "binary":
        pos = max(1, int((targets == 1).sum().item()))
        neg = max(1, int((targets == 0).sum().item()))
        pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif task_type == "multiclass":
        num_classes = int(targets.max().item()) + 1
        class_weights = build_inverse_freq_weights(targets, num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        x = features[start:end].to(device, non_blocking=True)

        if task_type == "binary":
            y = targets[start:end].float().unsqueeze(1).to(device, non_blocking=True)
            logits = probe(x)
            loss = criterion(logits, y)

            all_logits.append(logits.squeeze(1).detach().cpu())
            all_targets.append(targets[start:end].detach().cpu())
        else:
            y = targets[start:end].long().to(device, non_blocking=True)
            logits = probe(x)
            loss = criterion(logits, y)

            all_logits.append(logits.detach().cpu())
            all_targets.append(targets[start:end].detach().cpu())

        running_loss += loss.item()
        steps += 1

    avg_loss = running_loss / max(1, steps)

    if task_type == "binary":
        from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

        logits = torch.cat(all_logits, dim=0).numpy()
        y_true = torch.cat(all_targets, dim=0).numpy()

        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs > 0.5).astype(np.int64)

        acc = accuracy_score(y_true, preds)
        f1 = f1_score(y_true, preds, zero_division=0)
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, probs)
        else:
            auc = 0.5

        return {
            "loss": float(avg_loss),
            "acc": float(acc),
            "auc": float(auc),
            "f1": float(f1),
        }

    else:
        from sklearn.metrics import accuracy_score, balanced_accuracy_score

        logits = torch.cat(all_logits, dim=0).numpy()
        y_true = torch.cat(all_targets, dim=0).numpy()
        preds = logits.argmax(axis=1)

        acc = accuracy_score(y_true, preds)
        bacc = balanced_accuracy_score(y_true, preds)

        return {
            "loss": float(avg_loss),
            "acc": float(acc),
            "balanced_acc": float(bacc),
        }


def compute_single_channel_score_from_feature_bank(
    features,
    targets,
    probe,
    task_type,
    device,
    batch_size=512
):
    probe.eval()

    num_samples, feat_dim = features.shape

    if task_type == "binary":
        pos = max(1, int((targets == 1).sum().item()))
        neg = max(1, int((targets == 0).sum().item()))
        pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif task_type == "multiclass":
        num_classes = int(targets.max().item()) + 1
        class_weights = build_inverse_freq_weights(targets, num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    accum = torch.zeros(feat_dim, dtype=torch.float32, device=device)
    total = 0

    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)

        x = features[start:end].to(device, non_blocking=True).detach().requires_grad_(True)

        if task_type == "binary":
            y = targets[start:end].float().unsqueeze(1).to(device, non_blocking=True)
            logits = probe(x)
            loss = criterion(logits, y)
        else:
            y = targets[start:end].long().to(device, non_blocking=True)
            logits = probe(x)
            loss = criterion(logits, y)

        grad = torch.autograd.grad(
            loss, x, retain_graph=False, create_graph=False
        )[0]

        accum += (grad.abs() * x.detach().abs()).sum(dim=0)
        total += x.size(0)

    score = accum / max(1, total)
    return score.detach().cpu()


def build_fixed_topx_mask(suppress_score, topx_ratio):
    suppress_score = suppress_score.view(-1).float()
    feat_dim = suppress_score.numel()

    if topx_ratio <= 0:
        shortcut_mask = torch.zeros(feat_dim, dtype=torch.float32)
        core_mask = torch.ones(feat_dim, dtype=torch.float32)
        shortcut_idx = torch.empty(0, dtype=torch.long)
        return shortcut_mask, core_mask, shortcut_idx

    k = int(round(feat_dim * float(topx_ratio)))
    k = min(max(k, 1), feat_dim - 1)

    shortcut_mask = torch.zeros(feat_dim, dtype=torch.float32)
    topk = torch.topk(suppress_score, k=k, largest=True)
    shortcut_idx = topk.indices
    shortcut_mask[shortcut_idx] = 1.0

    core_mask = 1.0 - shortcut_mask
    return shortcut_mask, core_mask, shortcut_idx


def build_channel_drop_probs(suppress_score, mix_uniform=0.1, eps=1e-8):
    score = suppress_score.view(-1).float()
    score = torch.relu(score)

    if float(score.sum().item()) <= 0:
        probs = torch.ones_like(score) / score.numel()
    else:
        probs = score / (score.sum() + eps)

    if mix_uniform > 0:
        uniform = torch.full_like(probs, 1.0 / probs.numel())
        probs = (1.0 - mix_uniform) * probs + mix_uniform * uniform

    probs = probs / (probs.sum() + eps)
    return probs


def save_routing_visualizations(routing_info, save_dir):
    ensure_dir(save_dir)

    suppress_score = np.array(routing_info["suppress_score"])
    cls_score = np.array(routing_info["cls_score"])
    dom_score = np.array(routing_info["dom_score"])

    plt.figure(figsize=(8, 5))
    plt.hist(suppress_score, bins=50, alpha=0.8, density=True)
    plt.title("Suppress Score Distribution")
    plt.xlabel("Suppress Score")
    plt.ylabel("Density")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "routing_suppress_hist.png"), dpi=200)
    plt.close()

    sorted_idx = np.argsort(-suppress_score)
    topk = min(100, len(sorted_idx))
    idx = np.arange(topk)

    plt.figure(figsize=(10, 5))
    plt.plot(idx, suppress_score[sorted_idx[:topk]], marker='o', label="suppress")
    plt.plot(idx, dom_score[sorted_idx[:topk]], marker='x', label="dom")
    plt.plot(idx, cls_score[sorted_idx[:topk]], marker='s', label="cls")
    plt.title("Top Suppress Channels")
    plt.xlabel("Rank")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "routing_top_channels.png"), dpi=200)
    plt.close()


def estimate_fixed_routing_mask(model, score_dataloader, config, device, save_dir, rank=0):
    routing_cfg = get_routing_cfg(config)

    probe_epochs = routing_cfg["probe_epochs"]
    probe_batch_size = routing_cfg["probe_batch_size"]
    probe_lr = routing_cfg["probe_lr"]
    probe_weight_decay = routing_cfg["probe_weight_decay"]
    probe_hidden_dim = routing_cfg["probe_hidden_dim"]
    probe_dropout = routing_cfg["probe_dropout"]

    score_batch_size = routing_cfg["score_batch_size"]
    topx_ratio = routing_cfg["topx_ratio"]
    score_mode = routing_cfg["score_mode"]
    alpha = routing_cfg["alpha"]
    extra_probe_keys = routing_cfg.get("extra_probe_keys", ["ori_dataset", "real_source"])
    prob_uniform_mix = float(routing_cfg.get("prob_uniform_mix", 0.1))

    if rank == 0:
        print("\n" + "=" * 70)
        print("[Routing] 开始估计 stochastic shortcut drop 的通道分数 / 概率")
        print("=" * 70)

    features, labels, domains, extra_raw = extract_feature_bank(
        model=model,
        dataloader=score_dataloader,
        device=device,
        rank=rank,
        desc="Extract routing features",
        extra_keys=extra_probe_keys
    )

    pooled_feat_dim = features.shape[1]
    embed_dim = model.embed_dim
    num_domains = int(domains.max().item()) + 1

    if rank == 0:
        print(f"[Routing] pooled feature bank shape: {tuple(features.shape)}")
        print(f"[Routing] pooled_feat_dim={pooled_feat_dim}, mask_dim(embed_dim)={embed_dim}, num_domains={num_domains}")
        print(f"[Routing] labels distribution: {torch.bincount(labels).tolist()}")
        print(f"[Routing] domains distribution: {torch.bincount(domains).tolist()}")

    cls_probe = LinearOrMLPProbe(
        in_dim=pooled_feat_dim,
        out_dim=1,
        hidden_dim=probe_hidden_dim,
        dropout=probe_dropout
    )

    cls_probe = train_probe_on_feature_bank(
        features=features,
        targets=labels,
        probe=cls_probe,
        task_type="binary",
        device=device,
        epochs=probe_epochs,
        batch_size=probe_batch_size,
        lr=probe_lr,
        weight_decay=probe_weight_decay,
        rank=rank
    )

    cls_probe_metrics = evaluate_probe_on_feature_bank(
        features=features,
        targets=labels,
        probe=cls_probe,
        task_type="binary",
        device=device,
        batch_size=probe_batch_size
    )

    dom_probe = LinearOrMLPProbe(
        in_dim=pooled_feat_dim,
        out_dim=num_domains,
        hidden_dim=probe_hidden_dim,
        dropout=probe_dropout
    )

    dom_probe = train_probe_on_feature_bank(
        features=features,
        targets=domains,
        probe=dom_probe,
        task_type="multiclass",
        device=device,
        epochs=probe_epochs,
        batch_size=probe_batch_size,
        lr=probe_lr,
        weight_decay=probe_weight_decay,
        rank=rank
    )

    dom_probe_metrics = evaluate_probe_on_feature_bank(
        features=features,
        targets=domains,
        probe=dom_probe,
        task_type="multiclass",
        device=device,
        batch_size=probe_batch_size
    )

    cls_score_pooled = compute_single_channel_score_from_feature_bank(
        features=features,
        targets=labels,
        probe=cls_probe,
        task_type="binary",
        device=device,
        batch_size=score_batch_size
    )

    dom_score_pooled = compute_single_channel_score_from_feature_bank(
        features=features,
        targets=domains,
        probe=dom_probe,
        task_type="multiclass",
        device=device,
        batch_size=score_batch_size
    )

    eps = 1e-6
    cls_norm_pooled = cls_score_pooled / (cls_score_pooled.mean() + eps)
    dom_norm_pooled = dom_score_pooled / (dom_score_pooled.mean() + eps)

    if score_mode == "diff":
        suppress_score_pooled = torch.relu(dom_norm_pooled - alpha * cls_norm_pooled)
        if float(suppress_score_pooled.max().item()) <= 0:
            suppress_score_pooled = dom_norm_pooled.clone()
    elif score_mode == "ratio":
        suppress_score_pooled = dom_norm_pooled / (cls_norm_pooled + eps)
    else:
        raise ValueError(f"Unknown score_mode: {score_mode}")

    extra_probe_metrics = {}
    extra_score_pooled = {}
    extra_label_maps = {}

    for k in extra_probe_keys:
        raw_values = extra_raw.get(k, [])
        label_map = build_string_label_mapping(raw_values)
        extra_label_maps[k] = label_map

        encoded = encode_string_labels(raw_values, label_map)
        valid_mask = encoded >= 0

        if rank == 0:
            print(f"[Routing] {k}: valid={int(valid_mask.sum().item())}/{len(encoded)}, num_classes={len(label_map)}")

        if valid_mask.sum().item() < 2 or len(label_map) < 2:
            extra_probe_metrics[k] = None
            extra_score_pooled[k] = None
            continue

        f_valid = features[valid_mask]
        t_valid = encoded[valid_mask]

        probe = LinearOrMLPProbe(
            in_dim=pooled_feat_dim,
            out_dim=len(label_map),
            hidden_dim=probe_hidden_dim,
            dropout=probe_dropout
        )

        probe = train_probe_on_feature_bank(
            features=f_valid,
            targets=t_valid,
            probe=probe,
            task_type="multiclass",
            device=device,
            epochs=probe_epochs,
            batch_size=probe_batch_size,
            lr=probe_lr,
            weight_decay=probe_weight_decay,
            rank=rank
        )

        metrics = evaluate_probe_on_feature_bank(
            features=f_valid,
            targets=t_valid,
            probe=probe,
            task_type="multiclass",
            device=device,
            batch_size=probe_batch_size
        )

        score = compute_single_channel_score_from_feature_bank(
            features=f_valid,
            targets=t_valid,
            probe=probe,
            task_type="multiclass",
            device=device,
            batch_size=score_batch_size
        )

        extra_probe_metrics[k] = metrics
        extra_score_pooled[k] = score

    use_reg_for_pool = bool(getattr(model, "use_reg_token_effective", False))

    cls_score = merge_pooled_score_to_channel_level(
        cls_score_pooled,
        embed_dim=embed_dim,
        pooling_type=model.pooling_type,
        use_reg_token=use_reg_for_pool
    )
    dom_score = merge_pooled_score_to_channel_level(
        dom_score_pooled,
        embed_dim=embed_dim,
        pooling_type=model.pooling_type,
        use_reg_token=use_reg_for_pool
    )
    suppress_score = merge_pooled_score_to_channel_level(
        suppress_score_pooled,
        embed_dim=embed_dim,
        pooling_type=model.pooling_type,
        use_reg_token=use_reg_for_pool
    )


    extra_score_channel = {}
    for k, v in extra_score_pooled.items():
        if v is None:
            extra_score_channel[k] = None
        else:
            extra_score_channel[k] = merge_pooled_score_to_channel_level(
                v,
                embed_dim=embed_dim,
                pooling_type=model.pooling_type,
                use_reg_token=use_reg_for_pool
            )

    shortcut_mask, core_mask, shortcut_idx = build_fixed_topx_mask(
        suppress_score=suppress_score,
        topx_ratio=topx_ratio
    )

    drop_probs = build_channel_drop_probs(
        suppress_score=suppress_score,
        mix_uniform=prob_uniform_mix
    )

    cls_score_np = np.array(cls_score)
    dom_score_np = np.array(dom_score)
    suppress_score_np = np.array(suppress_score)
    drop_probs_np = np.array(drop_probs)

    if len(shortcut_idx) > 0:
        shortcut_idx_np = shortcut_idx.numpy()
        top_cls_mean = float(cls_score_np[shortcut_idx_np].mean())
        top_dom_mean = float(dom_score_np[shortcut_idx_np].mean())
        top_sup_mean = float(suppress_score_np[shortcut_idx_np].mean())
    else:
        top_cls_mean = 0.0
        top_dom_mean = 0.0
        top_sup_mean = 0.0

    routing_info = {
        "pooled_feat_dim": int(pooled_feat_dim),
        "mask_dim": int(embed_dim),

        "topx_ratio": float(topx_ratio),
        "shortcut_channels": int(shortcut_mask.sum().item()),
        "core_channels": int(core_mask.sum().item()),
        "core_ratio": float(core_mask.mean().item()),
        "score_mode": score_mode,
        "alpha": float(alpha),

        "cls_probe_metrics": cls_probe_metrics,
        "dom_probe_metrics": dom_probe_metrics,

        "top_shortcut_idx": shortcut_idx.tolist(),

        "cls_score_mean": float(cls_score_np.mean()),
        "cls_score_std": float(cls_score_np.std()),
        "cls_score_min": float(cls_score_np.min()),
        "cls_score_max": float(cls_score_np.max()),

        "dom_score_mean": float(dom_score_np.mean()),
        "dom_score_std": float(dom_score_np.std()),
        "dom_score_min": float(dom_score_np.min()),
        "dom_score_max": float(dom_score_np.max()),

        "suppress_score_mean": float(suppress_score_np.mean()),
        "suppress_score_std": float(suppress_score_np.std()),
        "suppress_score_min": float(suppress_score_np.min()),
        "suppress_score_max": float(suppress_score_np.max()),

        "drop_probs_mean": float(drop_probs_np.mean()),
        "drop_probs_std": float(drop_probs_np.std()),
        "drop_probs_min": float(drop_probs_np.min()),
        "drop_probs_max": float(drop_probs_np.max()),

        "top_shortcut_cls_mean": top_cls_mean,
        "top_shortcut_dom_mean": top_dom_mean,
        "top_shortcut_suppress_mean": top_sup_mean,

        "cls_score": cls_score.tolist(),
        "dom_score": dom_score.tolist(),
        "suppress_score": suppress_score.tolist(),
        "drop_probs": drop_probs.tolist(),

        "cls_score_pooled": cls_score_pooled.tolist(),
        "dom_score_pooled": dom_score_pooled.tolist(),
        "suppress_score_pooled": suppress_score_pooled.tolist(),
    }

    for k in extra_probe_keys:
        name = pretty_probe_name(k)
        metrics = extra_probe_metrics.get(k, None)
        score_ch = extra_score_channel.get(k, None)
        score_pool = extra_score_pooled.get(k, None)
        label_map = extra_label_maps.get(k, {})

        routing_info[f"{name}_label_map"] = label_map
        routing_info[f"{name}_probe_metrics"] = metrics
        routing_info[f"{name}_score"] = score_ch.tolist() if score_ch is not None else None
        routing_info[f"{name}_score_pooled"] = score_pool.tolist() if score_pool is not None else None

        if score_ch is not None:
            arr = np.array(score_ch)
            routing_info[f"{name}_score_mean"] = float(arr.mean())
            routing_info[f"{name}_score_std"] = float(arr.std())
            routing_info[f"{name}_score_min"] = float(arr.min())
            routing_info[f"{name}_score_max"] = float(arr.max())
        else:
            routing_info[f"{name}_score_mean"] = None
            routing_info[f"{name}_score_std"] = None
            routing_info[f"{name}_score_min"] = None
            routing_info[f"{name}_score_max"] = None

    ensure_dir(save_dir)
    save_json(routing_info, os.path.join(save_dir, "routing_info.json"))
    torch.save(
        {
            "shortcut_mask": shortcut_mask,
            "core_mask": core_mask,
            "drop_probs": drop_probs,
            "routing_info": routing_info,
        },
        os.path.join(save_dir, "routing_mask.pt")
    )

    vis_enabled = config.get("visualization", {}).get("enabled", True)
    if vis_enabled and rank == 0:
        routing_vis_dir = os.path.join(save_dir, "routing_visualizations")
        save_routing_visualizations(routing_info, routing_vis_dir)

    if rank == 0:
        print("\n[Routing Probe Metrics]")
        print(f"  cls_probe -> "
              f"loss={cls_probe_metrics['loss']:.4f}, "
              f"acc={cls_probe_metrics['acc']:.4f}, "
              f"auc={cls_probe_metrics['auc']:.4f}, "
              f"f1={cls_probe_metrics['f1']:.4f}")

        print(f"  dom_probe -> "
              f"loss={dom_probe_metrics['loss']:.4f}, "
              f"acc={dom_probe_metrics['acc']:.4f}, "
              f"bacc={dom_probe_metrics['balanced_acc']:.4f}")

        for k in extra_probe_keys:
            name = pretty_probe_name(k)
            m = routing_info.get(f"{name}_probe_metrics", None)
            if m is not None:
                print(f"  {name}_probe -> "
                      f"loss={m['loss']:.4f}, "
                      f"acc={m['acc']:.4f}, "
                      f"bacc={m['balanced_acc']:.4f}")
            else:
                print(f"  {name}_probe -> skipped")

        print("\n[Routing Score Stats | channel-level]")
        print(f"  cls_score      -> mean={routing_info['cls_score_mean']:.6f}, "
              f"std={routing_info['cls_score_std']:.6f}, "
              f"min={routing_info['cls_score_min']:.6f}, "
              f"max={routing_info['cls_score_max']:.6f}")

        print(f"  dom_score      -> mean={routing_info['dom_score_mean']:.6f}, "
              f"std={routing_info['dom_score_std']:.6f}, "
              f"min={routing_info['dom_score_min']:.6f}, "
              f"max={routing_info['dom_score_max']:.6f}")

        for k in extra_probe_keys:
            name = pretty_probe_name(k)
            if routing_info.get(f"{name}_score_mean", None) is not None:
                print(f"  {name}_score -> mean={routing_info[f'{name}_score_mean']:.6f}, "
                      f"std={routing_info[f'{name}_score_std']:.6f}, "
                      f"min={routing_info[f'{name}_score_min']:.6f}, "
                      f"max={routing_info[f'{name}_score_max']:.6f}")

        print(f"  suppress_score -> mean={routing_info['suppress_score_mean']:.6f}, "
              f"std={routing_info['suppress_score_std']:.6f}, "
              f"min={routing_info['suppress_score_min']:.6f}, "
              f"max={routing_info['suppress_score_max']:.6f}")

        print(f"  drop_probs     -> mean={routing_info['drop_probs_mean']:.6f}, "
              f"std={routing_info['drop_probs_std']:.6f}, "
              f"min={routing_info['drop_probs_min']:.6f}, "
              f"max={routing_info['drop_probs_max']:.6f}")

        print("\n[Top Shortcut Channel Stats | for analysis only]")
        print(f"  shortcut_channels={routing_info['shortcut_channels']}/{embed_dim}")
        print(f"  top_shortcut_cls_mean      = {routing_info['top_shortcut_cls_mean']:.6f}")
        print(f"  top_shortcut_dom_mean      = {routing_info['top_shortcut_dom_mean']:.6f}")
        print(f"  top_shortcut_suppress_mean = {routing_info['top_shortcut_suppress_mean']:.6f}")

        print(f"\n[Routing] routing info saved to: {save_dir}")

    return shortcut_mask, routing_info


def build_baseline_equivalent_mask(model, save_dir, rank=0):
    mask_dim = model.embed_dim

    shortcut_mask = torch.zeros(mask_dim, dtype=torch.float32)
    core_mask = torch.ones(mask_dim, dtype=torch.float32)
    drop_probs = torch.ones(mask_dim, dtype=torch.float32) / mask_dim

    routing_info = {
        "pooled_feat_dim": int(model.feat_dim),
        "mask_dim": int(mask_dim),

        "topx_ratio": 0.0,
        "shortcut_channels": 0,
        "core_channels": int(mask_dim),
        "core_ratio": 1.0,
        "score_mode": "baseline_equivalent",
        "alpha": 0.0,

        "cls_probe_metrics": {
            "loss": 0.0,
            "acc": 0.0,
            "auc": 0.0,
            "f1": 0.0,
        },
        "dom_probe_metrics": {
            "loss": 0.0,
            "acc": 0.0,
            "balanced_acc": 0.0,
        },

        "ori_dataset_probe_metrics": None,
        "real_source_probe_metrics": None,

        "top_shortcut_idx": [],

        "cls_score_mean": 0.0,
        "cls_score_std": 0.0,
        "cls_score_min": 0.0,
        "cls_score_max": 0.0,

        "dom_score_mean": 0.0,
        "dom_score_std": 0.0,
        "dom_score_min": 0.0,
        "dom_score_max": 0.0,

        "suppress_score_mean": 0.0,
        "suppress_score_std": 0.0,
        "suppress_score_min": 0.0,
        "suppress_score_max": 0.0,

        "drop_probs_mean": float(drop_probs.mean().item()),
        "drop_probs_std": float(drop_probs.std().item()),
        "drop_probs_min": float(drop_probs.min().item()),
        "drop_probs_max": float(drop_probs.max().item()),

        "top_shortcut_cls_mean": 0.0,
        "top_shortcut_dom_mean": 0.0,
        "top_shortcut_suppress_mean": 0.0,

        "cls_score": [0.0] * mask_dim,
        "dom_score": [0.0] * mask_dim,
        "suppress_score": [0.0] * mask_dim,
        "drop_probs": drop_probs.tolist(),

        "ori_dataset_score": None,
        "real_source_score": None,

        "cls_score_pooled": [0.0] * int(model.feat_dim),
        "dom_score_pooled": [0.0] * int(model.feat_dim),
        "suppress_score_pooled": [0.0] * int(model.feat_dim),
        "ori_dataset_score_pooled": None,
        "real_source_score_pooled": None,
    }

    ensure_dir(save_dir)
    save_json(routing_info, os.path.join(save_dir, "routing_info.json"))
    torch.save(
        {
            "shortcut_mask": shortcut_mask,
            "core_mask": core_mask,
            "drop_probs": drop_probs,
            "routing_info": routing_info,
        },
        os.path.join(save_dir, "routing_mask.pt")
    )

    if rank == 0:
        print("\n" + "=" * 70)
        print("[Routing] baseline 等价模式（无 suppress score / drop_probs 为均匀分布）")
        print("=" * 70)
        print(f"[Routing] routing info saved to: {save_dir}")

    return shortcut_mask, routing_info


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
            "ema_state_dict": state_dict_to_cpu(self.ema.state_dict()),
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
# curriculum
# =========================================================
def create_curriculum_manager(dataset, config, rank=0):
    curriculum_cfg = config.get("curriculum", {})
    enabled = curriculum_cfg.get("enabled", False)

    if not enabled:
        if rank == 0:
            print("[Curriculum] 未启用课程学习，使用普通随机采样")
        return None

    manager_type = curriculum_cfg.get("manager_type", "none").lower()
    if manager_type == "none":
        if rank == 0:
            print("[Curriculum] manager_type=none，使用普通随机采样")
        return None

    if manager_type != "domain_weighted":
        raise ValueError(f"未知 curriculum manager_type: {manager_type}")

    domain_cfg = curriculum_cfg.get("domain_weighted", {})

    manager = DomainWeightedCurriculumManager(
        dataset=dataset,
        total_epochs=config["training"]["epochs"],
        domain_names=domain_cfg.get("domain_names", None),
        difficulty_metric=domain_cfg.get("difficulty_metric", "val_auc"),

        min_domain_weight=domain_cfg.get("min_domain_weight", 0.15),
        max_domain_weight=domain_cfg.get("max_domain_weight", 0.40),

        focus_epochs=domain_cfg.get("focus_epochs", curriculum_cfg.get("stop_epoch", 12)),
        transition_epochs=domain_cfg.get("transition_epochs", None),

        eta=domain_cfg.get("eta", 2.0),
        start_ratio=domain_cfg.get("start_ratio", 0.4),
        end_ratio=domain_cfg.get("end_ratio", 1.0),

        mastery_auc=domain_cfg.get("mastery_auc", 0.97),
        max_focus_alpha=domain_cfg.get("max_focus_alpha", 0.60),
        base_weight_mode=domain_cfg.get("base_weight_mode", "dataset"),

        mastered_weight_mode=domain_cfg.get("mastered_weight_mode", "min"),
        post_focus_enabled=domain_cfg.get("post_focus_enabled", True),
        post_focus_alpha=domain_cfg.get("post_focus_alpha", 0.20),

        seed=config.get("system", {}).get("seed", 42)
    )

    if rank == 0:
        print("[Curriculum] 启用 DomainWeightedCurriculumManager")
    return manager


# =========================================================
# visualization
# =========================================================
def plot_lines(x, ys, labels, title, ylabel, save_path):
    plt.figure(figsize=(8, 5))
    for y, label in zip(ys, labels):
        if y is None or len(y) == 0:
            continue
        plt.plot(x, y, marker='o', linewidth=1.8, label=label)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_training_visualizations(history, save_dir):
    ensure_dir(save_dir)
    if len(history["epoch"]) == 0:
        return

    x = history["epoch"]

    plot_lines(
        x,
        [history["train_loss"], history["val_student_loss"], history["val_main_loss"]],
        ["Train Loss", "Val-Student Loss", "Val-Main Loss"],
        "Loss Curves",
        "Loss",
        os.path.join(save_dir, "curve_loss.png")
    )

    plot_lines(
        x,
        [history["train_auc"], history["val_student_auc"], history["val_main_auc"]],
        ["Train AUC", "Val-Student AUC", "Val-Main AUC"],
        "AUC Curves",
        "AUC",
        os.path.join(save_dir, "curve_auc.png")
    )

    plot_lines(
        x,
        [history["train_f1"], history["val_student_f1"], history["val_main_f1"]],
        ["Train F1", "Val-Student F1", "Val-Main F1"],
        "F1 Curves",
        "F1",
        os.path.join(save_dir, "curve_f1.png")
    )

    plot_lines(
        x,
        [history["train_mask_mean"], history["val_student_mask_mean"], history["val_main_mask_mean"]],
        ["Train Core Mask Mean", "Val-Student Core Mask Mean", "Val-Main Core Mask Mean"],
        "Core Mask Mean Curves",
        "Mask Mean",
        os.path.join(save_dir, "curve_mask_mean.png")
    )


def save_confusion_matrix_plot(all_labels, all_preds, threshold, save_path, title="Confusion Matrix"):
    from sklearn.metrics import confusion_matrix

    pred_labels = (all_preds > threshold).astype(int)
    cm = confusion_matrix(all_labels, pred_labels, labels=[0, 1])

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Real(0)", "Fake(1)"])
    plt.yticks(tick_marks, ["Real(0)", "Fake(1)"])

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_roc_curve_plot(all_labels, all_preds, save_path, title="ROC Curve"):
    from sklearn.metrics import roc_curve, auc

    plt.figure(figsize=(6, 5))
    if len(np.unique(all_labels)) >= 2:
        fpr, tpr, _ = roc_curve(all_labels, all_preds)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
    else:
        plt.plot([0, 1], [0, 1], '--', color='gray', label="Invalid ROC (single class)")

    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_pr_curve_plot(all_labels, all_preds, save_path, title="PR Curve"):
    from sklearn.metrics import precision_recall_curve, average_precision_score

    plt.figure(figsize=(6, 5))
    if len(np.unique(all_labels)) >= 2:
        precision, recall, _ = precision_recall_curve(all_labels, all_preds)
        ap = average_precision_score(all_labels, all_preds)
        plt.plot(recall, precision, lw=2, label=f"AP = {ap:.4f}")
    else:
        plt.plot([0, 1], [0.5, 0.5], '--', color='gray', label="Invalid PR (single class)")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_score_hist_plot(all_labels, all_preds, save_path, title="Score Distribution"):
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    real_scores = all_preds[all_labels == 0]
    fake_scores = all_preds[all_labels == 1]

    plt.figure(figsize=(7, 5))
    if len(real_scores) > 0:
        plt.hist(real_scores, bins=40, alpha=0.6, label="Real", density=True)
    if len(fake_scores) > 0:
        plt.hist(fake_scores, bins=40, alpha=0.6, label="Fake", density=True)

    plt.xlabel("Predicted Probability")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_domain_auc_bar(domain_metrics, save_path, title="Per-Domain AUC"):
    if domain_metrics is None or len(domain_metrics) == 0:
        return

    domain_names = list(domain_metrics.keys())
    domain_aucs = [domain_metrics[d]["auc_roc"] for d in domain_names]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(domain_names, domain_aucs)
    plt.ylim(0.0, 1.0)
    plt.ylabel("AUC")
    plt.title(title)
    plt.grid(axis='y', alpha=0.3)

    for bar, auc in zip(bars, domain_aucs):
        plt.text(bar.get_x() + bar.get_width() / 2, auc + 0.01, f"{auc:.3f}",
                 ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_mask_histogram(mask_values, save_path, title="Mask Value Distribution"):
    if mask_values is None or len(mask_values) == 0:
        return

    plt.figure(figsize=(7, 5))
    plt.hist(mask_values, bins=50, alpha=0.8, density=True)
    plt.xlabel("Mask Value")
    plt.ylabel("Density")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def visualize_eval_result(eval_result, save_dir, split_name="Test"):
    ensure_dir(save_dir)

    raw = eval_result.get("raw", None)
    if raw is None:
        return

    all_preds = np.array(raw["preds"])
    all_labels = np.array(raw["labels"])
    threshold = eval_result["threshold_used"]
    domain_metrics = eval_result.get("domain_metrics", {})
    mask_values = raw.get("mask_values", [])

    save_confusion_matrix_plot(
        all_labels, all_preds, threshold,
        save_path=os.path.join(save_dir, f"{split_name.lower()}_confusion_matrix.png"),
        title=f"{split_name} Confusion Matrix @ {threshold:.2f}"
    )

    save_roc_curve_plot(
        all_labels, all_preds,
        save_path=os.path.join(save_dir, f"{split_name.lower()}_roc.png"),
        title=f"{split_name} ROC Curve"
    )

    save_pr_curve_plot(
        all_labels, all_preds,
        save_path=os.path.join(save_dir, f"{split_name.lower()}_pr.png"),
        title=f"{split_name} PR Curve"
    )

    save_score_hist_plot(
        all_labels, all_preds,
        save_path=os.path.join(save_dir, f"{split_name.lower()}_score_hist.png"),
        title=f"{split_name} Score Distribution"
    )

    save_domain_auc_bar(
        domain_metrics,
        save_path=os.path.join(save_dir, f"{split_name.lower()}_domain_auc.png"),
        title=f"{split_name} Per-Domain AUC"
    )

    save_mask_histogram(
        mask_values,
        save_path=os.path.join(save_dir, f"{split_name.lower()}_mask_hist.png"),
        title=f"{split_name} Core Mask Value Distribution"
    )


# =========================================================
# train / eval
# =========================================================
def symmetric_kl_binary_with_logits(logits1, logits2, temperature=2.0):
    z1 = torch.cat([torch.zeros_like(logits1), logits1], dim=1) / temperature
    z2 = torch.cat([torch.zeros_like(logits2), logits2], dim=1) / temperature

    logp1 = F.log_softmax(z1, dim=1)
    logp2 = F.log_softmax(z2, dim=1)

    p1 = logp1.exp()
    p2 = logp2.exp()

    kl12 = F.kl_div(logp1, p2, reduction="batchmean")
    kl21 = F.kl_div(logp2, p1, reduction="batchmean")

    return 0.5 * (kl12 + kl21)


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


def forward_compute_train_loss(model, batch, device, criterion_cls_none, loss_cfg):
    images = batch["image"].to(device, non_blocking=True)
    labels = batch["label"].float().to(device, non_blocking=True).unsqueeze(1)

    outputs = model(images, grl_lambda=0.0, dual_view=True)

    if "cls_logits1" in outputs and "cls_logits2" in outputs:
        logits1 = outputs["cls_logits1"]
        logits2 = outputs["cls_logits2"]
        mask = outputs["mask"]
    else:
        logits1 = outputs["cls_logits"]
        logits2 = outputs["cls_logits"]
        mask = outputs["mask"]

    lambda_cls = loss_cfg.get("lambda_cls", 1.0)
    lambda_cons = loss_cfg.get("lambda_cons", 0.05)
    cons_temperature = loss_cfg.get("cons_temperature", 2.0)

    loss_cls1 = criterion_cls_none(logits1, labels).squeeze(1).mean()
    loss_cls2 = criterion_cls_none(logits2, labels).squeeze(1).mean()
    loss_cls = 0.5 * (loss_cls1 + loss_cls2)

    if lambda_cons > 0 and ("cls_logits1" in outputs and "cls_logits2" in outputs):
        loss_cons = symmetric_kl_binary_with_logits(
            logits1, logits2, temperature=cons_temperature
        )
    else:
        loss_cons = torch.zeros((), device=device)

    total_loss = lambda_cls * loss_cls + lambda_cons * loss_cons

    probs1 = torch.sigmoid(logits1)
    probs2 = torch.sigmoid(logits2)
    probs = 0.5 * (probs1 + probs2)

    return {
        "total_loss": total_loss,
        "loss_cls": loss_cls,
        "loss_cons": loss_cons,
        "probs": probs,
        "labels": labels,
        "mask": mask,
    }


def train_one_epoch(model, ema_model, dataloader,
                    criterion_cls_none, optimizer,
                    device, epoch, rank, is_distributed,
                    loss_cfg, grad_clip=0.0):

    model.train()

    running_total_loss = 0.0
    running_cls_loss = 0.0
    running_cons_loss = 0.0
    running_mask_mean = 0.0

    correct = 0.0
    total = 0.0
    num_batches = 0

    local_preds, local_labels, local_domains = [], [], []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]") if rank == 0 else dataloader
    last_ema_decay = None

    use_sam = is_sam_optimizer(optimizer)

    for batch_idx, batch in enumerate(pbar):
        domains = get_domain_keys_from_batch(batch)

        if not use_sam:
            optimizer.zero_grad(set_to_none=True)

            out = forward_compute_train_loss(
                model=model,
                batch=batch,
                device=device,
                criterion_cls_none=criterion_cls_none,
                loss_cfg=loss_cfg
            )

            total_loss = out["total_loss"]
            loss_cls = out["loss_cls"]
            loss_cons = out["loss_cons"]
            probs = out["probs"]
            labels = out["labels"]
            mask = out["mask"]

            total_loss.backward()

            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

        else:
            # ---------- SAM first step ----------
            optimizer.zero_grad(set_to_none=True)

            out1 = forward_compute_train_loss(
                model=model,
                batch=batch,
                device=device,
                criterion_cls_none=criterion_cls_none,
                loss_cfg=loss_cfg
            )
            total_loss_1 = out1["total_loss"]
            total_loss_1.backward()

            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.first_step(zero_grad=True)

            # ---------- SAM second step ----------
            out2 = forward_compute_train_loss(
                model=model,
                batch=batch,
                device=device,
                criterion_cls_none=criterion_cls_none,
                loss_cfg=loss_cfg
            )
            total_loss_2 = out2["total_loss"]
            total_loss_2.backward()

            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.second_step(zero_grad=True)

            # 用第二次 forward 的结果作为日志统计
            total_loss = out2["total_loss"]
            loss_cls = out2["loss_cls"]
            loss_cons = out2["loss_cons"]
            probs = out2["probs"]
            labels = out2["labels"]
            mask = out2["mask"]

        if ema_model is not None:
            last_ema_decay = ema_model.update(model)

        running_total_loss += total_loss.item()
        running_cls_loss += loss_cls.item()
        running_cons_loss += loss_cons.item()
        running_mask_mean += mask.mean().item()

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
                "loss": f"{running_total_loss/(batch_idx+1):.4f}",
                "cls": f"{running_cls_loss/(batch_idx+1):.4f}",
                "cons": f"{running_cons_loss/(batch_idx+1):.4f}",
                "mask": f"{running_mask_mean/(batch_idx+1):.3f}",
                "acc": f"{100.*correct/max(1,total):.2f}%"
            })

    if is_distributed:
        stats = torch.tensor([
            running_total_loss,
            running_cls_loss,
            running_cons_loss,
            running_mask_mean,
            correct,
            total,
            num_batches
        ], dtype=torch.float64, device=device)

        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        stats = stats.tolist()

        epoch_total_loss = stats[0] / max(1.0, stats[6])
        epoch_cls_loss = stats[1] / max(1.0, stats[6])
        epoch_cons_loss = stats[2] / max(1.0, stats[6])
        epoch_mask_mean = stats[3] / max(1.0, stats[6])
        epoch_acc = 100.0 * stats[4] / max(1.0, stats[5])
    else:
        epoch_total_loss = running_total_loss / max(1, num_batches)
        epoch_cls_loss = running_cls_loss / max(1, num_batches)
        epoch_cons_loss = running_cons_loss / max(1, num_batches)
        epoch_mask_mean = running_mask_mean / max(1, num_batches)
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

        train_domain_metrics = compute_domain_metrics(domain_stats, threshold=train_threshold)

        train_result = {
            "loss": epoch_total_loss,
            "acc": epoch_acc,
            "auc": train_metrics["auc_roc"],
            "f1": train_metrics["f1"],
            "precision": train_metrics["precision"],
            "recall": train_metrics["recall"],
            "optimal_threshold": train_threshold,
            "threshold_used": train_threshold,
            "metrics": train_metrics,
            "domain_metrics": train_domain_metrics,
            "ema_decay": last_ema_decay,
            "aux": {
                "loss_cls": epoch_cls_loss,
                "loss_cons": epoch_cons_loss,
                "loss_domain": 0.0,
                "loss_domain_adv": 0.0,
                "loss_dcs": 0.0,
                "loss_decorr": 0.0,
                "loss_mask_ratio": 0.0,
                "loss_mask_binary": 0.0,
                "mask_mean": epoch_mask_mean,
                "cls_score_mean": 0.0,
                "domain_score_mean": 0.0,
                "dcs_score_mean": 0.0,
            }
        }

    train_result = broadcast_object(train_result, rank, is_distributed, device)
    return train_result


@torch.no_grad()
def evaluate_loader(model, dataloader, criterion, device, epoch,
                    split_name="Val", verbose=True,
                    return_raw=False, max_mask_values=200000,
                    fixed_threshold=None):

    model.eval()
    running_loss = 0.0
    running_mask_mean = 0.0

    all_preds, all_labels, all_domains = [], [], []
    mask_values = []

    domain_stats = defaultdict(lambda: {"preds": [], "labels": []})

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [{split_name}]") if verbose else dataloader

    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].float().to(device, non_blocking=True).unsqueeze(1)
        domains = get_domain_keys_from_batch(batch)

        outputs = model(images, grl_lambda=0.0, dual_view=False)
        logits = outputs["cls_logits"]
        mask = outputs["mask"]

        loss = criterion(logits, labels)
        running_loss += loss.item()
        running_mask_mean += mask.mean().item()

        probs = torch.sigmoid(logits)
        probs_np = probs.cpu().numpy().flatten()
        labels_np = labels.cpu().numpy().flatten()

        all_preds.extend(probs_np.tolist())
        all_labels.extend(labels_np.tolist())
        all_domains.extend(list(domains))

        if len(mask_values) < max_mask_values:
            flat_mask = mask.detach().cpu().reshape(-1).tolist()
            remain = max_mask_values - len(mask_values)
            mask_values.extend(flat_mask[:remain])

        for i in range(len(labels_np)):
            d = domains[i]
            domain_stats[d]["preds"].append(float(probs_np[i]))
            domain_stats[d]["labels"].append(float(labels_np[i]))

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    eval_loss = running_loss / max(1, len(dataloader))
    eval_mask_mean = running_mask_mean / max(1, len(dataloader))

    split_optimal_threshold, _, _ = find_optimal_threshold(all_preds, all_labels)
    threshold_used = split_optimal_threshold if fixed_threshold is None else float(fixed_threshold)

    metrics = compute_all_metrics(all_preds, all_labels, threshold=threshold_used)
    domain_metrics = compute_domain_metrics(domain_stats, threshold=threshold_used)

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
        print(f"Threshold Used: {threshold_used:.2f}")
        if fixed_threshold is not None:
            print(f"Split Optimal Threshold (for ref only): {split_optimal_threshold:.2f}")
        print(f"Core Mask mean: {eval_mask_mean:.4f}")

        print("\n[Overall Full Metrics]")
        print_full_metrics(metrics)

        print(f"{'='*60}\n")

        print_full_domain_metrics(domain_metrics, title=f"[{split_name} Per-Domain Full Metrics]")

    result = {
        "loss": eval_loss,
        "acc": metrics["accuracy"] * 100.0,
        "auc": metrics["auc_roc"],
        "f1": metrics["f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "optimal_threshold": float(split_optimal_threshold),
        "threshold_used": float(threshold_used),
        "metrics": metrics,
        "domain_metrics": domain_metrics,
        "mask_mean": eval_mask_mean,
    }

    if return_raw:
        result["raw"] = {
            "preds": all_preds.tolist(),
            "labels": all_labels.tolist(),
            "domains": all_domains,
            "mask_values": mask_values
        }

    return result


def run_rank0_full_val_and_broadcast(model_for_eval, dataloader, criterion, device, epoch, rank, is_distributed,
                                     split_name="Val", verbose=True):
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
            return_raw=False,
            fixed_threshold=None
        )
    else:
        result = None

    result = broadcast_object(result, rank, is_distributed, device)
    return result


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


def load_best_eval_model_for_test(config, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = ForensicDinoSoftTopX(config).to(device)

    best_eval_state_dict = checkpoint.get("best_eval_state_dict", None)
    if best_eval_state_dict is not None:
        model.load_state_dict(best_eval_state_dict, strict=False)
    else:
        main_val_name = checkpoint.get("main_val_name", "Val-Student")
        if main_val_name == "Val-EMA" and checkpoint.get("ema_model_state_dict") is not None:
            ema_pack = checkpoint["ema_model_state_dict"]
            if isinstance(ema_pack, dict) and "ema_state_dict" in ema_pack:
                state_dict = ema_pack["ema_state_dict"]
            else:
                state_dict = ema_pack
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    model.eval()
    return model, checkpoint


def run_final_test_and_visualize(config, save_dir, criterion, device, test_loader):
    if test_loader is None:
        print("[Test] 未提供 test_dataset/test_datasets，跳过自动测试。")
        return None

    best_ckpt_path = os.path.join(save_dir, "best_model.pth")
    if not os.path.exists(best_ckpt_path):
        print(f"[Test] 未找到最佳模型: {best_ckpt_path}，跳过自动测试。")
        return None

    print("\n" + "=" * 70)
    print("[Final Test] 加载 best_model.pth 并执行测试")
    print("=" * 70)

    test_model, checkpoint = load_best_eval_model_for_test(config, best_ckpt_path, device)
    best_epoch = checkpoint.get("epoch", 0)
    val_threshold = checkpoint.get("optimal_threshold", 0.5)

    test_result = evaluate_loader(
        model=test_model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        epoch=max(0, best_epoch - 1),
        split_name="Test",
        verbose=True,
        return_raw=True,
        max_mask_values=config.get("visualization", {}).get("max_mask_values", 200000),
        fixed_threshold=val_threshold
    )

    test_dir = os.path.join(save_dir, "test_results")
    ensure_dir(test_dir)

    save_json(test_result, os.path.join(test_dir, "test_result.json"))

    vis_enabled = config.get("visualization", {}).get("enabled", True)
    if vis_enabled:
        vis_dir = os.path.join(test_dir, "plots")
        visualize_eval_result(test_result, vis_dir, split_name="Test")
        print(f"[Test] 可视化已保存到: {vis_dir}")

    print("[Test] 结果已保存。")
    return test_result


# =========================================================
# main
# =========================================================
def main():
    args = parse_args()
    config = load_config(args.config)

    dist_timeout_minutes = config.get("system", {}).get("dist_timeout_minutes", 180)
    is_distributed, rank, world_size, local_rank = setup_distributed(
        timeout_minutes=dist_timeout_minutes
    )

    system_cfg = config.get("system", {})
    if is_distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(system_cfg.get("device", "cuda"))

    set_seed(system_cfg.get("seed", 42))

    if rank == 0:
        print("\n" + "=" * 70)
        print(" DINOv2 Stochastic Shortcut Suppression ")
        print("=" * 70)

    save_dir = config.get("save_dir", "./checkpoints/dino_soft")
    ensure_dir(save_dir)
    vis_dir = os.path.join(save_dir, "visualizations")
    ensure_dir(vis_dir)

    # -------- dataset --------
    data_cfg = config["data"]

    train_dataset = build_dataset(
        dataset_cfg=config["train_dataset"],
        data_cfg=data_cfg,
        is_train=True,
        rank=rank,
        silent_nonzero_rank=True,
    )

    if rank == 0:
        val_dataset = build_dataset(
            dataset_cfg=config["val_dataset"],
            data_cfg=data_cfg,
            is_train=False,
            rank=rank,
            silent_nonzero_rank=False,
        )
    else:
        val_dataset = None

    test_dataset_cfg = get_test_dataset_cfg(config)
    test_dataset = None
    test_loader = None
    if rank == 0 and test_dataset_cfg is not None and test_dataset_cfg.get("path", None):
        test_dataset = build_dataset(
            dataset_cfg=test_dataset_cfg,
            data_cfg=data_cfg,
            is_train=False,
            rank=rank,
            silent_nonzero_rank=False,
        )

    curriculum_manager = create_curriculum_manager(train_dataset, config, rank=rank)

    if rank == 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=system_cfg.get("num_workers", 8),
            pin_memory=system_cfg.get("pin_memory", True)
        )
        if test_dataset is not None:
            test_loader = DataLoader(
                test_dataset,
                batch_size=config["training"]["batch_size"],
                shuffle=False,
                num_workers=system_cfg.get("num_workers", 8),
                pin_memory=system_cfg.get("pin_memory", True)
            )
    else:
        val_loader = None

    if rank == 0:
        print_dataset_summary(train_dataset, None, name="Train")
        print_dataset_summary(val_dataset, val_loader, name="Validation")
        if test_dataset is not None:
            print_dataset_summary(test_dataset, test_loader, name="Test")

    # -------- domain mapping --------
    model_cfg = config["model"]
    domain_names = model_cfg.get("domain_names", None)
    num_domains = model_cfg.get("num_domains", None)

    if domain_names is None:
        raise ValueError("新模型必须在 config['model'] 中提供 domain_names")
    if num_domains is None:
        num_domains = len(domain_names)
        config["model"]["num_domains"] = num_domains

    if len(domain_names) != int(num_domains):
        raise ValueError(
            f"model.num_domains={num_domains} 与 domain_names 长度={len(domain_names)} 不一致"
        )

    domain_to_idx = {name: idx for idx, name in enumerate(domain_names)}

    if rank == 0:
        print(f"[Domain Mapping] {domain_to_idx}")

    # -------- model --------
    with suppress_stdout_only(rank != 0):
        model = ForensicDinoSoftTopX(config).to(device)

    # -------- FLOPs calculation --------
    if rank == 0:
        try:
            from fvcore.nn import FlopCountAnalysis
            # 构造一个典型的输入张量 (batch_size=1, channels=3, height, width)
            image_size = data_cfg.get("image_size", 224)
            dummy_input = torch.randn(1, 3, image_size, image_size).to(device)
            
            # 使用 FlopCountAnalysis 分析
            flops = FlopCountAnalysis(model, dummy_input)
            total_flops = flops.total()
            gflops = total_flops / 1e9
            
            print(f"\n[Model Complexity]")
            print(f"  Input shape: (1, 3, {image_size}, {image_size})")
            print(f"  Total FLOPs: {total_flops:,}")
            print(f"  GFLOPs: {gflops:.2f} G")
            
            # 可选：保存到 config 或日志文件
            config["model_complexity"] = {
                "input_shape": [1, 3, image_size, image_size],
                "flops": total_flops,
                "gflops": round(gflops, 2)
            }
        except Exception as e:
            print(f"[Warning] Failed to compute FLOPs: {e}")
            traceback.print_exc()

    # -------- routing --------
    routing_cfg = get_routing_cfg(config)
    topx_ratio = float(routing_cfg.get("topx_ratio", 0.0))
    stochastic_enabled = bool(routing_cfg.get("stochastic_enabled", True))

    need_estimate_routing = stochastic_enabled or (topx_ratio > 0.0)
    routing_sync_path = get_routing_sync_path(save_dir)

    if rank == 0:
        if not need_estimate_routing:
            shortcut_mask_cpu, routing_info = build_baseline_equivalent_mask(
                model=model,
                save_dir=save_dir,
                rank=rank
            )
        else:
            if routing_cfg.get("use_train_aug_for_score", False):
                score_dataset = train_dataset
            else:
                score_dataset = build_dataset(
                    dataset_cfg=config["train_dataset"],
                    data_cfg=data_cfg,
                    is_train=False,
                    rank=rank,
                    silent_nonzero_rank=False,
                )

            score_loader = DataLoader(
                score_dataset,
                batch_size=routing_cfg.get("score_batch_size", config["training"]["batch_size"]),
                shuffle=False,
                num_workers=system_cfg.get("num_workers", 8),
                pin_memory=system_cfg.get("pin_memory", True)
            )

            shortcut_mask_cpu, routing_info = estimate_fixed_routing_mask(
                model=model,
                score_dataloader=score_loader,
                config=config,
                device=device,
                save_dir=save_dir,
                rank=rank
            )

        save_routing_sync_file(
            save_dir=save_dir,
            shortcut_mask_cpu=shortcut_mask_cpu,
            routing_info=routing_info
        )

    if is_distributed:
        dist.barrier()

    sync_pack = load_routing_sync_file(save_dir, map_location="cpu")
    shortcut_mask = sync_pack["shortcut_mask"].float().to(device)
    drop_probs = sync_pack["drop_probs"].float().to(device)
    routing_info = sync_pack["routing_info"]

    model.set_fixed_shortcut_mask(shortcut_mask)
    model.set_drop_probs(drop_probs)

    if rank == 0:
        print("\n[Routing Summary]")
        print(f"  stochastic_enabled : {stochastic_enabled}")
        print(f"  suppress_position  : {routing_cfg.get('suppress_position', 'before_pool')}")
        if routing_cfg.get("suppress_position", "before_pool") == "before_block":
            print(f"  suppress_block_idx : {routing_cfg.get('suppress_block_index', 11)}")
        print(f"  topx_ratio(log only): {routing_info['topx_ratio']:.4f}")
        print(f"  shortcut_channels  : {routing_info['shortcut_channels']}")
        print(f"  core_channels      : {routing_info['core_channels']}")
        print(f"  core_ratio(log)    : {routing_info['core_ratio']:.4f}")
        print(f"  drop_probs_mean    : {routing_info.get('drop_probs_mean', 0.0):.6f}")
        print(f"  drop_probs_std     : {routing_info.get('drop_probs_std', 0.0):.6f}")
        print(f"  drop_probs_min     : {routing_info.get('drop_probs_min', 0.0):.6f}")
        print(f"  drop_probs_max     : {routing_info.get('drop_probs_max', 0.0):.6f}")

        cls_probe_metrics = routing_info.get("cls_probe_metrics", {})
        dom_probe_metrics = routing_info.get("dom_probe_metrics", {})

        print("\n[Routing Summary Metrics]")
        print(f"  cls_probe_acc   : {cls_probe_metrics.get('acc', 0.0):.4f}")
        print(f"  cls_probe_auc   : {cls_probe_metrics.get('auc', 0.0):.4f}")
        print(f"  cls_probe_f1    : {cls_probe_metrics.get('f1', 0.0):.4f}")
        print(f"  dom_probe_acc   : {dom_probe_metrics.get('acc', 0.0):.4f}")
        print(f"  dom_probe_bacc  : {dom_probe_metrics.get('balanced_acc', 0.0):.4f}")
        print(f"  suppress_mean   : {routing_info.get('suppress_score_mean', 0.0):.6f}")
        print(f"  suppress_std    : {routing_info.get('suppress_score_std', 0.0):.6f}")

        if routing_info.get("ori_dataset_probe_metrics", None) is not None:
            print(f"  ori_dataset_acc : {routing_info['ori_dataset_probe_metrics'].get('acc', 0.0):.4f}")
            print(f"  ori_dataset_bacc: {routing_info['ori_dataset_probe_metrics'].get('balanced_acc', 0.0):.4f}")

        if routing_info.get("real_source_probe_metrics", None) is not None:
            print(f"  real_source_acc : {routing_info['real_source_probe_metrics'].get('acc', 0.0):.4f}")
            print(f"  real_source_bacc: {routing_info['real_source_probe_metrics'].get('balanced_acc', 0.0):.4f}")

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")

        if hasattr(model, "backbone"):
            model.backbone.print_trainable_status()

    if is_distributed:
        ddp_find_unused = config.get("training", {}).get("ddp_find_unused_parameters", False)
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=ddp_find_unused,
            broadcast_buffers=False
        )

    # -------- optimizer --------
    optimizer = build_optimizer(model, config, rank=rank)

    # -------- scheduler --------
    sched_cfg = config["training"]["scheduler"]
    total_epochs = config["training"]["epochs"]
    eta_min = sched_cfg.get("eta_min", 1e-6)
    scheduler_name = sched_cfg.get("name", "CosineAnnealingLR").lower()

    if scheduler_name == "cosineannealinglr":
        t_max = sched_cfg.get("T_max", total_epochs)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=eta_min
        )
    else:
        raise ValueError(f"当前只支持 CosineAnnealingLR，收到 scheduler.name={sched_cfg.get('name')}")

    # -------- loss --------
    criterion = nn.BCEWithLogitsLoss()
    criterion_cls_none = nn.BCEWithLogitsLoss(reduction="none")

    # -------- EMA --------
    ema_cfg = config.get("ema", {})
    ema_enabled = ema_cfg.get("enabled", False)

    dynamic_decay = ema_cfg.get("dynamic_decay", False)
    decay = ema_cfg.get("decay", 0.999)

    decay_start = ema_cfg.get("decay_start", 0.99)
    decay_end = ema_cfg.get("decay_end", 0.9995)
    schedule = ema_cfg.get("schedule", "cosine")

    use_ema_for_val = ema_cfg.get("use_ema_for_val", True)
    use_ema_for_curriculum = ema_cfg.get("use_ema_for_curriculum", True)
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
    start_epoch = 0
    best_val_auc = -1.0
    best_val_acc = 0.0
    best_threshold = 0.5

    checkpoint_path = config.get("checkpoint_path", None)
    resume = config.get("resume", False)

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        if rank == 0:
            print(f"加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if hasattr(model, "module"):
            model.module.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        if ema_enabled and checkpoint.get("ema_model_state_dict") is not None:
            ema_model.load_state_dict(checkpoint["ema_model_state_dict"])

        if resume:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"]
            best_val_acc = checkpoint.get("val_acc", 0.0)
            best_val_auc = checkpoint.get("val_auc", -1.0)
            best_threshold = checkpoint.get("optimal_threshold", 0.5)

    # -------- history --------
    history = defaultdict(list)

    # -------- training args --------
    grad_clip = config["training"].get("grad_clip", 0.0)
    loss_cfg = get_loss_cfg(config)

    # -------- train loop --------
    for epoch in range(start_epoch, config["training"]["epochs"]):
        if curriculum_manager is not None:
            train_sampler = curriculum_manager.get_sampler()
            train_sampler.set_epoch(epoch)
            train_loader = DataLoader(
                train_dataset,
                batch_size=config["training"]["batch_size"],
                sampler=train_sampler,
                num_workers=system_cfg.get("num_workers", 8),
                pin_memory=system_cfg.get("pin_memory", True),
                drop_last=True
            )
            sampler_name = "CurriculumSampler"
        else:
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
                num_workers=system_cfg.get("num_workers", 8),
                pin_memory=system_cfg.get("pin_memory", True),
                drop_last=True
            )
            sampler_name = "BaselineSampler"

        if rank == 0:
            print(f"\n[Epoch {epoch+1}] 当前训练批次数: {len(train_loader)} | sampler={sampler_name}")

        train_result = train_one_epoch(
            model=model,
            ema_model=ema_model,
            dataloader=train_loader,
            criterion_cls_none=criterion_cls_none,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            rank=rank,
            is_distributed=is_distributed,
            loss_cfg=loss_cfg,
            grad_clip=grad_clip
        )

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

        if ema_enabled and use_ema_for_val:
            main_val_result = ema_val_result
            main_val_name = "Val-EMA"
            main_eval_source = "ema"
        else:
            main_val_result = student_val_result
            main_val_name = "Val-Student"
            main_eval_source = "student"

        train_loss = train_result["loss"]
        train_acc = train_result["acc"]
        train_auc = train_result["auc"]
        train_f1 = train_result["f1"]
        train_precision = train_result["precision"]
        train_recall = train_result["recall"]
        train_domain_metrics = train_result["domain_metrics"]
        current_ema_decay = train_result.get("ema_decay", None)
        train_aux = train_result.get("aux", {})

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

        if curriculum_manager is not None:
            if ema_enabled and use_ema_for_curriculum and ema_val_result is not None:
                curriculum_manager.update_val_metrics(ema_val_result["domain_metrics"])
            else:
                curriculum_manager.update_val_metrics(student_val_result["domain_metrics"])
            curriculum_manager.step()

        if rank == 0:
            history["epoch"].append(epoch + 1)

            history["train_loss"].append(train_loss)
            history["train_auc"].append(train_auc)
            history["train_f1"].append(train_f1)
            history["train_mask_mean"].append(train_aux.get("mask_mean", 0.0))

            history["val_student_loss"].append(student_val_result["loss"])
            history["val_student_auc"].append(student_val_result["auc"])
            history["val_student_f1"].append(student_val_result["f1"])
            history["val_student_mask_mean"].append(student_val_result.get("mask_mean", 0.0))

            history["val_main_loss"].append(main_val_result["loss"])
            history["val_main_auc"].append(main_val_result["auc"])
            history["val_main_f1"].append(main_val_result["f1"])
            history["val_main_mask_mean"].append(main_val_result.get("mask_mean", 0.0))

            save_json(history, os.path.join(save_dir, "training_history.json"))
            if config.get("visualization", {}).get("enabled", True):
                save_training_visualizations(history, vis_dir)

            lr_info = " | ".join([f"group{i}={pg['lr']:.6e}" for i, pg in enumerate(optimizer.param_groups)])

            print("\n[Train Overall Full Metrics]")
            print_full_metrics(train_result["metrics"])

            print("")
            print_full_domain_metrics(train_domain_metrics, title="[Train Per-Domain Full Metrics]")

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
            print(f"  训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.2f}%")
            print(f"  训练AUC: {train_auc:.4f} | PRE: {train_precision:.4f} | REC: {train_recall:.4f} | F1: {train_f1:.4f}")
            print(f"  Train Aux -> loss_cls: {train_aux.get('loss_cls', 0.0):.4f} | loss_cons: {train_aux.get('loss_cons', 0.0):.4f}")

            print(f"  Val-Student 损失: {student_val_result['loss']:.4f} | 准确率: {student_val_result['acc']:.2f}%")
            print(f"  Val-Student AUC: {student_val_result['auc']:.4f} | PRE: {student_val_result['precision']:.4f} | REC: {student_val_result['recall']:.4f} | F1: {student_val_result['f1']:.4f}")

            if ema_val_result is not None:
                print(f"  Val-EMA 损失: {ema_val_result['loss']:.4f} | 准确率: {ema_val_result['acc']:.2f}%")
                print(f"  Val-EMA AUC: {ema_val_result['auc']:.4f} | PRE: {ema_val_result['precision']:.4f} | REC: {ema_val_result['recall']:.4f} | F1: {ema_val_result['f1']:.4f}")

            print(f"  主验证指标来源: {main_val_name}")
            print(f"  学习率: {lr_info}")
            print(f"  训练使用: L = lambda_cls * L_cls + lambda_cons * L_cons")
            print(f"  Core MaskMean -> train: {train_aux.get('mask_mean', 0.0):.4f} | "
                  f"student_val: {student_val_result.get('mask_mean', 0.0):.4f}"
                  + (f" | ema_val: {ema_val_result.get('mask_mean', 0.0):.4f}" if ema_val_result is not None else ""))

            should_save_best = (epoch == start_epoch) or (val_auc > best_val_auc)

            if should_save_best:
                best_val_auc = val_auc
                best_val_acc = val_acc
                best_threshold = optimal_threshold

                student_state = state_dict_to_cpu(get_model_state_dict(model))
                if ema_enabled and ema_model is not None:
                    ema_state_pack = ema_model.state_dict()
                else:
                    ema_state_pack = None

                if main_eval_source == "ema" and ema_enabled and ema_model is not None:
                    best_eval_state_dict = state_dict_to_cpu(ema_model.ema.state_dict())
                else:
                    best_eval_state_dict = student_state

                checkpoint = {
                    "epoch": epoch + 1,
                    "model_state_dict": student_state,
                    "best_eval_state_dict": best_eval_state_dict,
                    "best_eval_source": main_eval_source,
                    "ema_model_state_dict": ema_state_pack if (ema_enabled and save_ema) else None,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "routing_info": routing_info,
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
                student_state = state_dict_to_cpu(get_model_state_dict(model))
                if ema_enabled and ema_model is not None:
                    ema_state_pack = ema_model.state_dict()
                else:
                    ema_state_pack = None

                checkpoint = {
                    "epoch": epoch + 1,
                    "model_state_dict": student_state,
                    "best_eval_source": main_eval_source,
                    "ema_model_state_dict": ema_state_pack if (ema_enabled and save_ema) else None,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "routing_info": routing_info,
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

    if is_distributed:
        cleanup_distributed()

    if rank == 0:
        print("\n" + "=" * 60)
        print("训练完成!")
        print(f"最佳验证准确率: {best_val_acc:.2f}%")
        print(f"最佳验证AUC: {best_val_auc:.4f}")
        print(f"最佳阈值: {best_threshold:.2f}")
        print("=" * 60)

        torch.cuda.empty_cache()

        test_after_train = config.get("test_after_train", True)
        if test_after_train:
            _ = run_final_test_and_visualize(
                config=config,
                save_dir=save_dir,
                criterion=criterion,
                device=device,
                test_loader=test_loader
            )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        rank = os.environ.get("RANK", "unknown")
        local_rank = os.environ.get("LOCAL_RANK", "unknown")
        print(f"\n[FATAL] rank={rank}, local_rank={local_rank}, error={repr(e)}", file=sys.stderr)
        traceback.print_exc()
        raise