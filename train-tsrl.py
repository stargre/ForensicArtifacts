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

from torch.distributions import Normal
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
    parser = argparse.ArgumentParser(description="DINOv2 train with TSRL (Tutor-Student RL Curriculum)")
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
# TSRL: state manager (per-sample history)
# =========================================================
class TSRLStateManager:
    """
    per-sample history:
      - ema_loss
      - forget_count (standard: correct->error)
      - prev_correct
    """
    def __init__(self, dataset_size, device, ema_beta=0.9):
        self.dataset_size = dataset_size
        self.device = device
        self.ema_beta = ema_beta

        self.ema_loss = torch.zeros(dataset_size, dtype=torch.float32, device=device)
        self.forget_count = torch.zeros(dataset_size, dtype=torch.float32, device=device)
        self.prev_correct = torch.full((dataset_size,), -1.0, dtype=torch.float32, device=device)

        # local cache for epoch sync
        self.local_seen_mask = torch.zeros(dataset_size, dtype=torch.float32, device=device)
        self.local_ema_loss = torch.zeros(dataset_size, dtype=torch.float32, device=device)
        self.local_forget_count = torch.zeros(dataset_size, dtype=torch.float32, device=device)
        self.local_prev_correct = torch.full((dataset_size,), -1.0, dtype=torch.float32, device=device)

    @torch.no_grad()
    def get_batch_history(self, indices):
        return (
            self.ema_loss[indices],
            self.forget_count[indices],
            self.prev_correct[indices],
        )

    @torch.no_grad()
    def update_batch(self, indices, ce_losses, current_correct):
        for idx, cur_loss, cur_corr in zip(indices, ce_losses, current_correct):
            i = int(idx.item())
            old_ema = self.ema_loss[i]
            old_prev_corr = self.prev_correct[i]
            old_forget = self.forget_count[i]

            if old_ema.item() == 0.0:
                new_ema = cur_loss
            else:
                new_ema = self.ema_beta * old_ema + (1.0 - self.ema_beta) * cur_loss

            # standard forgetting: correct -> error
            new_forget = old_forget
            if old_prev_corr >= 0 and old_prev_corr.item() == 1.0 and cur_corr.item() == 0.0:
                new_forget = old_forget + 1.0

            self.local_seen_mask[i] = 1.0
            self.local_ema_loss[i] = new_ema
            self.local_forget_count[i] = new_forget
            self.local_prev_correct[i] = cur_corr

    @torch.no_grad()
    def sync_across_processes(self, is_distributed):
        # DistributedSampler: each index should be seen by exactly one rank each epoch
        if is_distributed:
            dist.all_reduce(self.local_seen_mask, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.local_ema_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.local_forget_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.local_prev_correct, op=dist.ReduceOp.SUM)

        seen = self.local_seen_mask > 0
        self.ema_loss[seen] = self.local_ema_loss[seen]
        self.forget_count[seen] = self.local_forget_count[seen]
        self.prev_correct[seen] = self.local_prev_correct[seen]

        # reset local cache
        self.local_seen_mask.zero_()
        self.local_ema_loss.zero_()
        self.local_forget_count.zero_()
        self.local_prev_correct.fill_(-1.0)

    def state_dict(self):
        return {
            "ema_loss": self.ema_loss.detach().cpu(),
            "forget_count": self.forget_count.detach().cpu(),
            "prev_correct": self.prev_correct.detach().cpu(),
        }

    def load_state_dict(self, state_dict):
        self.ema_loss.copy_(state_dict["ema_loss"].to(self.device))
        self.forget_count.copy_(state_dict["forget_count"].to(self.device))
        self.prev_correct.copy_(state_dict["prev_correct"].to(self.device))


# =========================================================
# TSRL: Tutor actor-critic + PPO buffer
# =========================================================
class TutorActorCritic(nn.Module):
    """
    Actor outputs z ~ N(mu, std); weight w = sigmoid(z) in [0,1]
    Critic outputs V(s)
    """
    def __init__(self, state_dim, hidden_dim=256, init_log_std=-0.5):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

        self.log_std = nn.Parameter(torch.tensor([init_log_std], dtype=torch.float32))

    def get_dist(self, states):
        mu = self.actor(states)
        std = torch.exp(self.log_std).expand_as(mu)
        return Normal(mu, std)

    def act(self, states):
        dist_obj = self.get_dist(states)
        z = dist_obj.rsample()
        log_prob = dist_obj.log_prob(z).sum(dim=-1, keepdim=True)
        entropy = dist_obj.entropy().sum(dim=-1, keepdim=True)
        value = self.critic(states)
        weight = torch.sigmoid(z)
        return z, weight, log_prob, entropy, value

    def evaluate_actions(self, states, actions_z):
        dist_obj = self.get_dist(states)
        log_prob = dist_obj.log_prob(actions_z).sum(dim=-1, keepdim=True)
        entropy = dist_obj.entropy().sum(dim=-1, keepdim=True)
        value = self.critic(states)
        weight = torch.sigmoid(actions_z)
        return log_prob, entropy, value, weight


class PPOBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states = []
        self.actions_z = []
        self.log_probs = []
        self.rewards = []
        self.values = []

    def add(self, states, actions_z, log_probs, rewards, values):
        self.states.append(states.detach().cpu())
        self.actions_z.append(actions_z.detach().cpu())
        self.log_probs.append(log_probs.detach().cpu())
        self.rewards.append(rewards.detach().cpu())
        self.values.append(values.detach().cpu())

    def is_empty(self):
        return len(self.states) == 0

    def get_all(self, device):
        states = torch.cat(self.states, dim=0).to(device)
        actions_z = torch.cat(self.actions_z, dim=0).to(device)
        log_probs = torch.cat(self.log_probs, dim=0).to(device)
        rewards = torch.cat(self.rewards, dim=0).to(device)
        values = torch.cat(self.values, dim=0).to(device)
        return states, actions_z, log_probs, rewards, values


# =========================================================
# TSRL helper functions
# =========================================================


def normalize_1d_tensor(x):
    if x.numel() <= 1:
        return torch.zeros_like(x)
    return (x - x.mean()) / (x.std() + 1e-6)


def build_tsrl_state(features, conf, correct_flag, ema_loss, forget_count):
    """
    Eq.(1): s = [f, p, e, l_ema, c_forget]
    features: [B,C], conf/correct/ema/forget: [B,1]
    """
    if conf.ndim == 1:
        conf = conf.unsqueeze(1)
    if correct_flag.ndim == 1:
        correct_flag = correct_flag.unsqueeze(1)
    if ema_loss.ndim == 1:
        ema_loss = ema_loss.unsqueeze(1)
    if forget_count.ndim == 1:
        forget_count = forget_count.unsqueeze(1)
    return torch.cat([features, conf, correct_flag, ema_loss, forget_count], dim=1)

def compute_tsrl_reward_state_based(
    conf_init,
    ema_loss,
    forget_count,
    lambda_forget=0.7,
    lambda_conf=0.3,
    lambda_loss=0.2
):
    """
    State-based reward.
    不依赖 pred_upd / delta_conf.
    """

    # normalize forgetting
    forget_norm = normalize_1d_tensor(forget_count).unsqueeze(1)

    # normalize EMA loss
    ema_norm = normalize_1d_tensor(ema_loss).unsqueeze(1)

    # confidence penalty (already [0,1])
    conf_penalty = conf_init

    reward = (
        lambda_forget * forget_norm
        + lambda_loss * ema_norm
        - lambda_conf * conf_penalty
    )

    return reward

def heuristic_expert_weight(ema_loss_raw, cur_loss_raw, correct_flag):
    """
    BC Expert heuristic: favor hard-but-learnable
    """
    score = 0.5 * normalize_1d_tensor(ema_loss_raw) + 0.5 * normalize_1d_tensor(cur_loss_raw)
    w = torch.sigmoid(score)  # [0,1]
    w = 0.2 + 0.8 * w         # [0.2,1]
    # down-weight easy correct low-loss
    easy = (correct_flag.squeeze(1) > 0.5) & (cur_loss_raw < cur_loss_raw.mean())
    w[easy] = torch.clamp(w[easy] * 0.7, min=0.05, max=1.0)
    return w.unsqueeze(1)


def student_forward(model, images):
    """
    Your ForensicDinoBaseline returns: logits, cls_token, patch_tokens
    - use cls_token as f_i
    """
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


# =========================================================
# auto test
# =========================================================
@torch.no_grad()
def run_final_test(best_ckpt_path, config, criterion, device, test_loader, ema_enabled, save_ema, final_epoch=0):
    print("\n" + "=" * 70)
    print("加载最佳模型并在测试集上评估")
    print("=" * 70)
    print(f"Best checkpoint: {best_ckpt_path}")

    checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only = False)

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


# =========================================================
# PPO update (Eq.7 simplified)
# =========================================================
def train_tutor_with_ppo(tutor_model, tutor_optimizer, ppo_buffer, device, tsrl_cfg, rank=0):
    if ppo_buffer.is_empty():
        if rank == 0:
            print("[Tutor-PPO] buffer empty, skip update.")
        return {"ppo_loss": 0.0, "actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

    states, actions_z, old_log_probs, rewards, old_values = ppo_buffer.get_all(device)
    ppo_buffer.clear()

    rewards = rewards.view(-1, 1)
    old_values = old_values.view(-1, 1)
    old_log_probs = old_log_probs.view(-1, 1)

    # advantage (simple): A = r - V_old
    advantages = rewards - old_values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    returns = rewards  # dense immediate reward

    ppo_epochs = tsrl_cfg["ppo_epochs"]
    clip_eps = tsrl_cfg["ppo_clip_eps"]
    value_coef = tsrl_cfg["value_coef"]
    entropy_coef = tsrl_cfg["entropy_coef"]
    max_grad_norm = tsrl_cfg.get("tutor_grad_clip", 1.0)

    tutor_model.train()

    avg_total_loss = 0.0
    avg_actor_loss = 0.0
    avg_critic_loss = 0.0
    avg_entropy = 0.0

    for _ in range(ppo_epochs):
        new_log_probs, entropy, new_values, _ = tutor_model.evaluate_actions(states, actions_z)

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        critic_loss = F.mse_loss(new_values, returns)
        entropy_bonus = entropy.mean()

        total_loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy_bonus

        tutor_optimizer.zero_grad()
        total_loss.backward()
        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(tutor_model.parameters(), max_grad_norm)
        tutor_optimizer.step()

        avg_total_loss += total_loss.item()
        avg_actor_loss += actor_loss.item()
        avg_critic_loss += critic_loss.item()
        avg_entropy += entropy_bonus.item()

    n = float(ppo_epochs)
    result = {
        "ppo_loss": avg_total_loss / n,
        "actor_loss": avg_actor_loss / n,
        "critic_loss": avg_critic_loss / n,
        "entropy": avg_entropy / n,
    }

    if rank == 0:
        print(f"[Tutor-PPO] total={result['ppo_loss']:.6f} | actor={result['actor_loss']:.6f} | "
              f"critic={result['critic_loss']:.6f} | entropy={result['entropy']:.6f}")

    return result


# =========================================================
# TSRL train one epoch
# =========================================================
def train_one_epoch_tsrl(
    model,
    ema_model,
    dataloader,
    criterion_none,
    optimizer,
    asam,
    device,
    epoch,
    rank,
    is_distributed,
    state_manager,
    tutor_model,
    tutor_optimizer,
    ppo_buffer,
    tsrl_cfg,
    grad_clip=0.0
):
    model.train()

    running_loss = 0.0
    correct = 0.0
    total = 0.0
    num_batches = 0

    local_preds, local_labels, local_domains = [], [], []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]") if rank == 0 else dataloader
    last_ema_decay = None

    warmup_epochs = tsrl_cfg["warmup_epochs"]
    bc_epochs = tsrl_cfg["bc_epochs"]
    reward_c_rew = tsrl_cfg["reward_conf_scale"]

    if epoch < warmup_epochs:
        phase = "warmup"
    elif epoch < warmup_epochs + bc_epochs:
        phase = "bc"
    else:
        phase = "tsrl"

    # meters
    bc_running_loss = 0.0
    bc_steps = 0
    weight_mean_meter = 0.0

    reward_mean_meter = 0.0
    reward_steps = 0

    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].float().to(device, non_blocking=True).unsqueeze(1)
        domains = batch["domain"]
        indices = batch["index"]
        if not torch.is_tensor(indices):
            indices = torch.tensor(indices, dtype=torch.long)
        indices = indices.to(device)

        # --------------------------------------------
        # (A) PRE-UPDATE state: logits/feature/conf/correct/loss + history
        # --------------------------------------------
        model.eval()
        with torch.no_grad():
            logits_init, cls_token, _ = student_forward(model, images)   # logits:[B,1], cls_token:[B,C]
            feats_init = cls_token
            conf_init = torch.sigmoid(logits_init)
            pred_init = (conf_init > 0.5).float()
            correct_init = (pred_init == labels).float()
            ce_loss_init = criterion_none(logits_init, labels).squeeze(1)  # [B]

        ema_loss_hist, forget_count_hist, _ = state_manager.get_batch_history(indices)

        # normalize history inside batch to stabilize policy input
        ema_loss_norm = normalize_1d_tensor(ema_loss_hist)
        forget_norm = normalize_1d_tensor(forget_count_hist)

        states = build_tsrl_state(
            features=feats_init,
            conf=conf_init,
            correct_flag=correct_init,
            ema_loss=ema_loss_norm,
            forget_count=forget_norm,
        )

        # --------------------------------------------
        # (B) Tutor action -> sample weights
        # --------------------------------------------
        if phase == "warmup":
            sample_weights = torch.ones_like(labels)
            actions_z = None
            log_probs = None
            values = None

        elif phase == "bc":
            expert_w = heuristic_expert_weight(
                ema_loss_raw=ema_loss_hist,
                cur_loss_raw=ce_loss_init,
                correct_flag=correct_init,
            )

            tutor_model.train()
            z, pred_w, _, _, _ = tutor_model.act(states)
            bc_loss = F.mse_loss(pred_w, expert_w)

            tutor_optimizer.zero_grad()
            bc_loss.backward()
            tutor_optimizer.step()

            sample_weights = expert_w.detach()
            actions_z = None
            log_probs = None
            values = None

            bc_running_loss += bc_loss.item()
            bc_steps += 1

        else:
            tutor_model.eval()
            with torch.no_grad():
                actions_z, sample_weights, log_probs, _, values = tutor_model.act(states)
            sample_weights = sample_weights.detach()

        sample_weights = torch.clamp(sample_weights, 0.0, 1.0)
        weight_mean_meter += sample_weights.mean().item()

        # --------------------------------------------
        # (C) Student update with weighted loss (Eq.4)
        # --------------------------------------------
        model.train()
        optimizer.zero_grad()

        logits_train, _, _ = student_forward(model, images)
        per_sample_losses = criterion_none(logits_train, labels).squeeze(1)  # [B]
        # normalize weighted mean to avoid shrink with small weights
        weighted_loss = (per_sample_losses * sample_weights.squeeze(1)).sum() / (sample_weights.sum() + 1e-8)

        # =========================
        # ASAM First Step
        # =========================
        weighted_loss.backward()
        asam.first_step()

        optimizer.zero_grad()   
        # =========================
        # Second Forward
        # =========================
        logits_train_2, _, _ = student_forward(model, images)
        per_sample_losses_2 = criterion_none(logits_train_2, labels).squeeze(1)

        weighted_loss_2 = (
            per_sample_losses_2 * sample_weights.squeeze(1)
        ).sum() / (sample_weights.sum() + 1e-8)

        weighted_loss_2.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        asam.second_step()

        if ema_model is not None:
            last_ema_decay = ema_model.update(model)

        # --------------------------------------------
        # (D) POST-UPDATE reward (Eq.5): forward again after update
        # --------------------------------------------
        if phase == "tsrl":
            with torch.no_grad():
                rewards = compute_tsrl_reward_state_based(
                    conf_init=conf_init,
                    ema_loss=ema_loss_hist,
                    forget_count=forget_count_hist,
                    lambda_forget=tsrl_cfg.get("lambda_forget", 0.7),
                    lambda_conf=tsrl_cfg.get("lambda_conf", 0.3),
                    lambda_loss=tsrl_cfg.get("lambda_loss", 0.2),
                )

                ppo_buffer.add(
                    states=states,
                    actions_z=actions_z,
                    log_probs=log_probs,
                    rewards=rewards,
                    values=values,
                )

                reward_mean_meter += rewards.mean().item()
                reward_steps += 1


        # --------------------------------------------
        # (E) Update state manager using AFTER-update status
        # --------------------------------------------
        with torch.no_grad():
            logits_state, _, _ = student_forward(model, images)
            probs_state = torch.sigmoid(logits_state)
            preds_state = (probs_state > 0.5).float()
            correct_state = (preds_state == labels).float().squeeze(1)  # [B]
            ce_state = criterion_none(logits_state, labels).squeeze(1)  # [B]

        state_manager.update_batch(indices=indices, ce_losses=ce_state, current_correct=correct_state)

        # --------------------------------------------
        # stats
        # --------------------------------------------
        running_loss += weighted_loss.item()
        probs = torch.sigmoid(logits_train)
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
            postfix = {
                "loss": f"{running_loss/(batch_idx+1):.4f}",
                "acc": f"{100.*correct/max(1,total):.2f}%",
                "phase": phase,
                "w": f"{weight_mean_meter/max(1,num_batches):.3f}"
            }
            if phase == "bc" and bc_steps > 0:
                postfix["bc"] = f"{bc_running_loss/bc_steps:.4f}"
            if phase == "tsrl" and reward_steps > 0:
                postfix["r"] = f"{reward_mean_meter/reward_steps:.4f}"
            pbar.set_postfix(postfix)

    # sync state history across GPUs
    state_manager.sync_across_processes(is_distributed)

    # PPO update at epoch end
    ppo_result = None
    if phase == "tsrl":
        if is_distributed:
            dist.barrier()
        # IMPORTANT: tutor_model might be DDP; optimize the underlying module
        ppo_result = train_tutor_with_ppo(
            tutor_model=get_student_model(tutor_model),
            tutor_optimizer=tutor_optimizer,
            ppo_buffer=ppo_buffer,
            device=device,
            tsrl_cfg=tsrl_cfg,
            rank=rank
        )

    # reduce stats
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
            "phase": phase,
            "bc_loss": (bc_running_loss / max(1, bc_steps)) if bc_steps > 0 else None,
            "mean_weight": weight_mean_meter / max(1, num_batches),
            "mean_reward": reward_mean_meter / max(1, reward_steps) if reward_steps > 0 else None,
            "ppo_result": ppo_result,
        }

    train_result = broadcast_object(train_result, rank, is_distributed, device)
    return train_result

def train_one_epoch_standard(
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

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]") if rank == 0 else dataloader

    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].float().to(device, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad()

        logits, _, _ = student_forward(model, images)
        loss = criterion(logits, labels)

        if asam is not None:
            # ---- ASAM ----
            loss.backward()
            asam.first_step()

            optimizer.zero_grad()

            logits2, _, _ = student_forward(model, images)
            loss2 = criterion(logits2, labels)

            loss2.backward()

            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            asam.second_step()
        else:
            # ---- Normal ----
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

    return {
        "loss": epoch_loss,
        "acc": epoch_acc,
        "auc": 0.0,
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "domain_metrics": {},
        "ema_decay": last_ema_decay,
    }

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
        print(" DINOv2 + TSRL + EMA + Auto Test ")
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

    # -------- model (Student) --------
    model = ForensicDinoBaseline(config).to(device)

    # feature dim (cls_token dim)
    # in your wrapper, embed_dim exists in backbone; but easiest is:
    feature_dim = int(getattr(model.backbone, "embed_dim", 768))

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        if hasattr(model, "backbone") and hasattr(model.backbone, "print_trainable_status"):
            model.backbone.print_trainable_status()
        print(f"[TSRL] detected feature_dim = {feature_dim}")

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
        asam = ASAM(
            model=base_model_for_asam,
            optimizer=optimizer,
            rho=sam_rho
        )

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

    # -------- TSRL config --------
    tsrl_cfg = config.get("tsrl", {})
    tsrl_enabled = tsrl_cfg.get("enabled", False)

    # fill defaults
    def _get(key, default):
        return tsrl_cfg[key] if key in tsrl_cfg else default

    tsrl_cfg = {
        "enabled": tsrl_enabled,
        "warmup_epochs": int(_get("warmup_epochs", 2)),
        "bc_epochs": int(_get("bc_epochs", 2)),
        "state_ema_beta": float(_get("state_ema_beta", 0.9)),
        "reward_conf_scale": float(_get("reward_conf_scale", 0.5)),
        "hidden_dim": int(_get("hidden_dim", 256)),
        "tutor_lr": float(_get("tutor_lr", 1e-4)),
        "ppo_epochs": int(_get("ppo_epochs", 4)),
        "ppo_clip_eps": float(_get("ppo_clip_eps", 0.2)),
        "value_coef": float(_get("value_coef", 0.5)),
        "entropy_coef": float(_get("entropy_coef", 0.01)),
        "tutor_grad_clip": float(_get("tutor_grad_clip", 1.0)),
        "init_log_std": float(_get("init_log_std", -0.5)),
        "lambda_forget": float(_get("lambda_forget", 0.7)),
        "lambda_conf": float(_get("lambda_conf", 0.3)),
        "lambda_loss": float(_get("lambda_loss", 0.2)),
    }

    if rank == 0:
        print("\n" + "=" * 70)
        print("TSRL Config")
        print("=" * 70)
        for k, v in tsrl_cfg.items():
            print(f"{k:<18}: {v}")
        print("=" * 70)

    # TSRL modules
    if tsrl_enabled:
        state_manager = TSRLStateManager(
            dataset_size=len(train_dataset),
            device=device,
            ema_beta=tsrl_cfg.get("state_ema_beta", 0.9),
        )

        tutor_state_dim = feature_dim + 4
        tutor_model = TutorActorCritic(
            state_dim=tutor_state_dim,
            hidden_dim=tsrl_cfg.get("hidden_dim", 256),
            init_log_std=tsrl_cfg.get("init_log_std", -0.5)
        ).to(device)

        tutor_optimizer = optim.Adam(
            tutor_model.parameters(),
            lr=tsrl_cfg.get("tutor_lr", 1e-4)
        )
        ppo_buffer = PPOBuffer()
    else:
        state_manager = None
        tutor_model = None
        tutor_optimizer = None
        ppo_buffer = None

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
    save_dir = config.get("save_dir", "./checkpoints/dino_tsrl")
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

        # student
        if hasattr(model, "module"):
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])

        # ema
        if ema_enabled and checkpoint.get("ema_model_state_dict") is not None:
            ema_model.load_state_dict(checkpoint["ema_model_state_dict"])

        # tutor & state manager
        if tsrl_enabled and checkpoint.get("tutor_model_state_dict", None) is not None:
            get_student_model(tutor_model).load_state_dict(checkpoint["tutor_model_state_dict"])
        if tsrl_enabled and checkpoint.get("state_manager", None) is not None:
            state_manager.load_state_dict(checkpoint["state_manager"])

        if resume:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if tsrl_enabled and checkpoint.get("tutor_optimizer_state_dict", None) is not None:
                tutor_optimizer.load_state_dict(checkpoint["tutor_optimizer_state_dict"])
            start_epoch = checkpoint["epoch"]
            best_val_acc = checkpoint.get("val_acc", 0.0)
            best_val_auc = checkpoint.get("val_auc", 0.0)
            best_threshold = checkpoint.get("optimal_threshold", 0.5)

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

        # ===== TSRL train =====
        if tsrl_enabled:
            train_result = train_one_epoch_tsrl(
                model=model,
                ema_model=ema_model,
                dataloader=train_loader,
                criterion_none=criterion_none,
                optimizer=optimizer,
                asam=asam,
                device=device,
                epoch=epoch,
                rank=rank,
                is_distributed=is_distributed,
                state_manager=state_manager,
                tutor_model=tutor_model,
                tutor_optimizer=tutor_optimizer,
                ppo_buffer=ppo_buffer,
                tsrl_cfg=tsrl_cfg,
                grad_clip=grad_clip
            )
        else:
            train_result = train_one_epoch_standard(
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
            verbose=True
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

        # ===== test every epoch  =====
        if test_dataset is not None and test_loader is not None:
            student_eval_model = model.module if hasattr(model, "module") else model

            if rank == 0:
                student_eval_model = model.module if hasattr(model, "module") else model
                test_student_result = evaluate_loader(
                    model=student_eval_model,
                    dataloader=test_loader,
                    criterion=criterion,
                    device=device,
                    epoch=epoch,
                    split_name="Test-Student",
                    verbose=True
                )

                test_ema_result = None
                if ema_enabled:
                    test_ema_result = evaluate_loader(
                        model=ema_model.ema,
                        dataloader=test_loader,
                        criterion=criterion,
                        device=device,
                        epoch=epoch,
                        split_name="Test-EMA",
                        verbose=True
                    )
            else:
                test_student_result = None
                test_ema_result = None

            


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
                print(f"\n[EMA] decay(last): {current_ema_decay:.6f}")

            if tsrl_enabled:
                print(f"\n[TSRL]")
                print(f"  phase       : {train_result.get('phase')}")
                print(f"  mean weight : {train_result.get('mean_weight', 0.0):.4f}")
                
            if train_result.get("bc_loss", None) is not None:
                print(f"  bc loss     : {train_result.get('bc_loss'):.6f}")
            if train_result.get("mean_reward", None) is not None:
                print(f"  mean reward : {train_result.get('mean_reward'):.6f}")
            if train_result.get("ppo_result", None) is not None:
                pr = train_result["ppo_result"]
                print(f"  ppo total   : {pr['ppo_loss']:.6f} | actor={pr['actor_loss']:.6f} | "
                      f"critic={pr['critic_loss']:.6f} | ent={pr['entropy']:.6f}")

            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train loss: {train_loss:.4f} | acc: {train_acc:.2f}% | auc: {train_auc:.4f}")
            print(f"  Val({main_val_name}) loss: {val_loss:.4f} | acc: {val_acc:.2f}% | auc: {val_auc:.4f}")
            print(f"  lr: {lr_info}")
            # ===== 打印当前epoch test结果（不参与best选择）=====
            if test_student_result is not None:
                print(f"  Test-Student acc: {test_student_result['acc']:.2f}% | "
                      f"auc: {test_student_result['auc']:.4f}")

            if test_ema_result is not None:
                print(f"  Test-EMA     acc: {test_ema_result['acc']:.2f}% | "
                      f"auc: {test_ema_result['auc']:.4f}")

            # save best
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_val_acc = val_acc
                best_threshold = optimal_threshold

                student_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                checkpoint = {
                    "epoch": epoch + 1,
                    "model_state_dict": student_state,
                    "ema_model_state_dict": ema_model.state_dict() if (ema_enabled and save_ema) else None,
                    "tutor_model_state_dict": (
                        get_student_model(tutor_model).state_dict()
                        if tsrl_enabled else None
                    ),
                    "tutor_optimizer_state_dict": (
                        tutor_optimizer.state_dict()
                        if tsrl_enabled else None
                    ),
                    "state_manager": (
                        state_manager.state_dict()
                        if tsrl_enabled else None
                    ),
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
                print(f"✓ Best model saved! ({main_val_name} AUC: {val_auc:.4f})")

            # periodic save
            save_freq = config.get("logging", {}).get("save_freq", 5)
            if (epoch + 1) % save_freq == 0:
                student_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                checkpoint = {
                    "epoch": epoch + 1,
                    "model_state_dict": student_state,
                    "ema_model_state_dict": ema_model.state_dict() if (ema_enabled and save_ema) else None,
                    "tutor_model_state_dict": (
                        get_student_model(tutor_model).state_dict()
                        if tsrl_enabled else None
                    ),
                    "tutor_optimizer_state_dict": (
                        tutor_optimizer.state_dict()
                        if tsrl_enabled else None
                    ),
                    "state_manager": (
                        state_manager.state_dict()
                        if tsrl_enabled else None
                    ),
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
                print(f"💾 Checkpoint saved: {ckpt_name}")

        # early stopping
        if early_stopper is not None:
            monitor = es_cfg.get("monitor", "val_auc")
            monitor_map = {"val_acc": val_acc, "val_auc": val_auc, "val_f1": val_f1, "val_loss": val_loss}
            current_score = monitor_map.get(monitor, val_auc)
            should_stop = early_stopper(current_score, epoch)

            if is_distributed:
                stop_tensor = torch.tensor([1.0 if should_stop else 0.0], device=device)
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

        # auto test
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