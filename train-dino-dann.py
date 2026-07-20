import os
import yaml
import argparse
import random
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from pre_data.dino_dataprocess import ForensicImageDataset, print_dataset_summary
from model.dino_orth import ForensicDinoOrth
from model.orth_projector import orthogonal_loss


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="DINO Orthogonal Decomposition training")
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
                    print(f"  ⚠ 早停触发! 最佳 {self.monitor}: {self.best_score:.4f} @ epoch {self.best_epoch + 1}")

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


def train_one_epoch(
    model,
    dataloader,
    criterion_forgery,
    criterion_domain,
    optimizer,
    device,
    epoch,
    config,
    rank=0,
    grad_clip=0.0
):
    model.train()

    running_loss = 0.0
    running_forgery_loss = 0.0
    running_domain_loss = 0.0
    running_orth_loss = 0.0

    correct = 0
    total = 0

    domain_correct = 0
    domain_total = 0

    domain_loss_weight = config["training"].get("domain_loss_weight", 0.2)
    orth_loss_weight = config["training"].get("orth_loss_weight", 0.05)

    if rank == 0:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")
    else:
        pbar = dataloader

    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].float().to(device, non_blocking=True).unsqueeze(1)
        domain_labels = batch["domain_label"].long().to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
        forgery_logits = outputs["forgery_logits"]
        domain_logits = outputs["domain_logits"]
        forgery_feat = outputs["forgery_feat"]
        domain_feat = outputs["domain_feat"]

        loss_forgery = criterion_forgery(forgery_logits, labels)
        loss_domain = criterion_domain(domain_logits, domain_labels)
        loss_orth = orthogonal_loss(
            forgery_feat,
            domain_feat,
            loss_type=config["training"].get("orth_loss_type", "sample_dot"),
            normalize=config["training"].get("orth_normalize", True),
            center=config["training"].get("orth_center", True),
        )

        loss = loss_forgery + domain_loss_weight * loss_domain + orth_loss_weight * loss_orth
        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        running_loss += loss.item()
        running_forgery_loss += loss_forgery.item()
        running_domain_loss += loss_domain.item()
        running_orth_loss += loss_orth.item()

        probs = torch.sigmoid(forgery_logits)
        preds = (probs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        domain_preds = torch.argmax(domain_logits, dim=1)
        domain_correct += (domain_preds == domain_labels).sum().item()
        domain_total += domain_labels.size(0)

        if rank == 0:
            pbar.set_postfix({
                "loss": f"{running_loss/(batch_idx+1):.4f}",
                "f_loss": f"{running_forgery_loss/(batch_idx+1):.4f}",
                "d_loss": f"{running_domain_loss/(batch_idx+1):.4f}",
                "o_loss": f"{running_orth_loss/(batch_idx+1):.4f}",
                "acc": f"{100.*correct/total:.2f}%",
                "d_acc": f"{100.*domain_correct/domain_total:.2f}%"
            })

    epoch_loss = running_loss / len(dataloader)
    epoch_forgery_loss = running_forgery_loss / len(dataloader)
    epoch_domain_loss = running_domain_loss / len(dataloader)
    epoch_orth_loss = running_orth_loss / len(dataloader)

    epoch_acc = 100.0 * correct / total
    epoch_domain_acc = 100.0 * domain_correct / domain_total

    return (
        epoch_loss,
        epoch_forgery_loss,
        epoch_domain_loss,
        epoch_orth_loss,
        epoch_acc,
        epoch_domain_acc
    )


@torch.no_grad()
def validate(
    model,
    dataloader,
    criterion_forgery,
    criterion_domain,
    device,
    epoch,
    config,
    rank=0
):
    model.eval()

    running_loss = 0.0
    running_forgery_loss = 0.0
    running_domain_loss = 0.0
    running_orth_loss = 0.0

    all_preds = []
    all_labels = []
    domain_stats = defaultdict(lambda: {"preds": [], "labels": []})

    domain_correct = 0
    domain_total = 0

    domain_loss_weight = config["training"].get("domain_loss_weight", 0.2)
    orth_loss_weight = config["training"].get("orth_loss_weight", 0.05)

    if rank == 0:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]")
    else:
        pbar = dataloader

    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].float().to(device, non_blocking=True).unsqueeze(1)
        domain_labels = batch["domain_label"].long().to(device, non_blocking=True)
        domains = batch["domain"]

        outputs = model(images)
        forgery_logits = outputs["forgery_logits"]
        domain_logits = outputs["domain_logits"]
        forgery_feat = outputs["forgery_feat"]
        domain_feat = outputs["domain_feat"]

        loss_forgery = criterion_forgery(forgery_logits, labels)
        loss_domain = criterion_domain(domain_logits, domain_labels)
        loss_orth = orthogonal_loss(
            forgery_feat,
            domain_feat,
            loss_type=config["training"].get("orth_loss_type", "sample_dot"),
            normalize=config["training"].get("orth_normalize", True),
            center=config["training"].get("orth_center", True),
        )

        loss = loss_forgery + domain_loss_weight * loss_domain + orth_loss_weight * loss_orth

        running_loss += loss.item()
        running_forgery_loss += loss_forgery.item()
        running_domain_loss += loss_domain.item()
        running_orth_loss += loss_orth.item()

        probs = torch.sigmoid(forgery_logits)
        all_preds.extend(probs.cpu().numpy().flatten())
        all_labels.extend(labels.cpu().numpy().flatten())

        domain_preds = torch.argmax(domain_logits, dim=1)
        domain_correct += (domain_preds == domain_labels).sum().item()
        domain_total += domain_labels.size(0)

        for i in range(len(labels)):
            domain = domains[i]
            domain_stats[domain]["preds"].append(probs[i].item())
            domain_stats[domain]["labels"].append(labels[i].item())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    val_loss = running_loss / len(dataloader)
    val_forgery_loss = running_forgery_loss / len(dataloader)
    val_domain_loss = running_domain_loss / len(dataloader)
    val_orth_loss = running_orth_loss / len(dataloader)
    val_domain_acc = 100.0 * domain_correct / domain_total

    optimal_threshold, _, _ = find_optimal_threshold(all_preds, all_labels)
    metrics = compute_all_metrics(all_preds, all_labels, threshold=optimal_threshold)

    if rank == 0:
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1} 验证结果")
        print(f"{'='*80}")
        print(f"  Total Loss   : {val_loss:.4f}")
        print(f"  Forgery Loss : {val_forgery_loss:.4f}")
        print(f"  Domain Loss  : {val_domain_loss:.4f}")
        print(f"  Orth Loss    : {val_orth_loss:.4f}")
        print(f"  Forgery ACC  : {metrics['accuracy']*100:.2f}%")
        print(f"  Domain ACC   : {val_domain_acc:.2f}%")
        print(f"  AUC          : {metrics['auc_roc']:.4f}")
        print(f"  AUC-PR       : {metrics['auc_pr']:.4f}")
        print(f"  F1           : {metrics['f1']:.4f}")
        print(f"  Precision    : {metrics['precision']:.4f}")
        print(f"  Recall       : {metrics['recall']:.4f}")
        print(f"  Specificity  : {metrics['specificity']:.4f}")
        print(f"  最佳阈值      : {optimal_threshold:.2f}")
        print(f"{'='*80}\n")

    return (
        val_loss,
        val_forgery_loss,
        val_domain_loss,
        val_orth_loss,
        metrics["accuracy"] * 100.0,
        val_domain_acc,
        metrics["auc_roc"],
        metrics["f1"],
        optimal_threshold,
        metrics
    )


def main():
    args = parse_args()
    config = load_config(args.config)

    is_distributed, rank, world_size, local_rank = setup_distributed()

    if is_distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(config["system"]["device"])

    set_seed(config["system"].get("seed", 42))

    if rank == 0:
        print("\n" + "=" * 80)
        print(" DINOv2 + Feature-level Orthogonal Decomposition Training ")
        print("=" * 80)

    # ==================== Dataset ====================
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

    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config["system"]["num_workers"],
        pin_memory=config["system"]["pin_memory"],
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        sampler=val_sampler,
        num_workers=config["system"]["num_workers"],
        pin_memory=config["system"]["pin_memory"]
    )

    if rank == 0:
        print_dataset_summary(train_dataset, train_loader, name="Train")
        print_dataset_summary(val_dataset, val_loader, name="Validation")

    # ==================== Model ====================
    model = ForensicDinoOrth(config).to(device)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")

    if is_distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False
        )

    # ==================== Optimizer ====================
    opt_cfg = config["training"]["optimizer"]
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=opt_cfg["lr"],
        weight_decay=opt_cfg["weight_decay"],
        betas=tuple(opt_cfg["betas"])
    )

    # ==================== Scheduler ====================
    sched_cfg = config["training"]["scheduler"]
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=sched_cfg["T_max"],
        eta_min=sched_cfg["eta_min"]
    )

    # ==================== Loss ====================
    criterion_forgery = nn.BCEWithLogitsLoss()
    criterion_domain = nn.CrossEntropyLoss()

    # ==================== Early Stopping ====================
    es_cfg = config["training"].get("early_stopping", {})
    early_stopper = None
    if es_cfg.get("enabled", False):
        early_stopper = EarlyStopping(
            patience=es_cfg.get("patience", 8),
            min_delta=es_cfg.get("min_delta", 0.0005),
            monitor=es_cfg.get("monitor", "val_auc"),
            verbose=(rank == 0)
        )

    # ==================== Checkpoint ====================
    save_dir = config.get("save_dir", "./checkpoints/dino_orth")
    os.makedirs(save_dir, exist_ok=True)

    start_epoch = 0
    best_val_auc = 0.0
    best_val_acc = 0.0
    best_threshold = 0.5

    checkpoint_path = config.get("checkpoint_path", None)
    resume = config.get("resume", False)

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        if rank == 0:
            print(f"加载检查点: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

        if hasattr(model, "module"):
            model.module.load_state_dict(state_dict, strict=True)
        else:
            model.load_state_dict(state_dict, strict=True)

        if resume and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint.get("epoch", 0)
            best_val_acc = checkpoint.get("val_acc", 0.0)
            best_val_auc = checkpoint.get("val_auc", 0.0)
            best_threshold = checkpoint.get("optimal_threshold", 0.5)

    # ==================== Train Loop ====================
    final_epoch = 0
    grad_clip = config["training"].get("grad_clip", 0.0)

    for epoch in range(start_epoch, config["training"]["epochs"]):
        final_epoch = epoch + 1

        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss, train_f_loss, train_d_loss, train_o_loss, train_acc, train_domain_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion_forgery=criterion_forgery,
            criterion_domain=criterion_domain,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            config=config,
            rank=rank,
            grad_clip=grad_clip
        )

        val_loss, val_f_loss, val_d_loss, val_o_loss, val_acc, val_domain_acc, val_auc, val_f1, optimal_threshold, metrics = validate(
            model=model,
            dataloader=val_loader,
            criterion_forgery=criterion_forgery,
            criterion_domain=criterion_domain,
            device=device,
            epoch=epoch,
            config=config,
            rank=rank
        )

        scheduler.step()

        if rank == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"\nEpoch {epoch+1} 总结:")
            print(f"  训练总损失: {train_loss:.4f}")
            print(f"  训练伪造损失: {train_f_loss:.4f} | 训练域损失: {train_d_loss:.4f} | 训练正交损失: {train_o_loss:.4f}")
            print(f"  训练真假准确率: {train_acc:.2f}% | 训练域准确率: {train_domain_acc:.2f}%")
            print(f"  验证总损失: {val_loss:.4f}")
            print(f"  验证伪造损失: {val_f_loss:.4f} | 验证域损失: {val_d_loss:.4f} | 验证正交损失: {val_o_loss:.4f}")
            print(f"  验证真假准确率: {val_acc:.2f}% | 验证域准确率: {val_domain_acc:.2f}%")
            print(f"  验证AUC: {val_auc:.4f} | 验证AUC-PR: {metrics['auc_pr']:.4f} | 验证F1: {val_f1:.4f}")
            print(f"  学习率: {current_lr:.6f}")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_val_acc = val_acc
                best_threshold = optimal_threshold

                model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                checkpoint = {
                    "epoch": epoch + 1,
                    "model_state_dict": model_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_acc": val_acc,
                    "val_auc": val_auc,
                    "val_f1": val_f1,
                    "val_domain_acc": val_domain_acc,
                    "optimal_threshold": optimal_threshold,
                    "metrics": metrics,
                    "config": config,
                }
                torch.save(checkpoint, os.path.join(save_dir, "best_model.pth"))
                print(f"✓ 最佳模型已保存! (AUC: {val_auc:.4f}, Domain ACC: {val_domain_acc:.2f}%)")

            save_freq = config.get("logging", {}).get("save_freq", 5)
            if (epoch + 1) % save_freq == 0:
                model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                checkpoint = {
                    "epoch": epoch + 1,
                    "model_state_dict": model_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_acc": val_acc,
                    "val_auc": val_auc,
                    "val_f1": val_f1,
                    "val_domain_acc": val_domain_acc,
                    "optimal_threshold": optimal_threshold,
                    "metrics": metrics,
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
        print("\n" + "=" * 80)
        print("训练完成!")
        print(f"最佳验证准确率: {best_val_acc:.2f}%")
        print(f"最佳验证AUC: {best_val_auc:.4f}")
        print(f"最佳阈值: {best_threshold:.2f}")
        print("=" * 80)


if __name__ == "__main__":
    main()