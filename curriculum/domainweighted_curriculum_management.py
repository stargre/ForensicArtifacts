import numpy as np
import torch.distributed as dist
from torch.utils.data import Sampler
from typing import Dict, List, Optional
from collections import defaultdict


class DomainWeightedSampler(Sampler):
    """
    域加权采样器：
    1) 先按 domain_weights 做 multinomial 分配各域样本数
    2) 每个域优先无放回采样
    3) 不够再补有放回采样
    """
    def __init__(self, domain_indices, domain_weights, total_samples, num_replicas=None, rank=None, seed=42):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0

        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0

        self.domain_indices = domain_indices
        self.domain_weights = domain_weights
        self.total_samples = total_samples

        self.num_samples = (total_samples + num_replicas - 1) // num_replicas
        self.total_size_padded = self.num_samples * num_replicas

    def update_params(self, domain_weights: Dict[str, float], total_samples: int):
        self.domain_weights = domain_weights.copy()
        self.total_samples = total_samples
        self.num_samples = (total_samples + self.num_replicas - 1) // self.num_replicas
        self.total_size_padded = self.num_samples * self.num_replicas

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _sample_from_domains(self, rng: np.random.RandomState):
        valid_domains = []
        probs = []

        for d, w in self.domain_weights.items():
            if d in self.domain_indices and len(self.domain_indices[d]) > 0 and w > 0:
                valid_domains.append(d)
                probs.append(w)

        if len(valid_domains) == 0:
            raise RuntimeError("没有可用于采样的域，请检查 domain_weights 和 dataset。")

        probs = np.array(probs, dtype=np.float64)
        probs = probs / probs.sum()

        counts = rng.multinomial(self.total_samples, probs)

        sampled_indices = []
        for d, n_samples in zip(valid_domains, counts):
            if n_samples <= 0:
                continue

            available = np.array(self.domain_indices[d], dtype=np.int64)

            if n_samples <= len(available):
                chosen = rng.choice(available, size=n_samples, replace=False)
            else:
                full = rng.permutation(available)
                extra = rng.choice(available, size=n_samples - len(available), replace=True)
                chosen = np.concatenate([full, extra], axis=0)

            sampled_indices.extend(chosen.tolist())

        rng.shuffle(sampled_indices)
        return sampled_indices

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        indices = self._sample_from_domains(rng)

        if len(indices) < self.total_size_padded:
            extra = self.total_size_padded - len(indices)
            indices = indices + indices[:extra]
        elif len(indices) > self.total_size_padded:
            indices = indices[:self.total_size_padded]

        indices = indices[self.rank:self.total_size_padded:self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples


class DomainWeightedCurriculumManager:
    """
    新版课程学习管理器

    核心策略：
    1. 前 focus_epochs 轮：强课程学习（base -> hard）
    2. focus_epochs 之后：后期弱自适应（仍依据 val AUC 更新，但强度更小）
    3. 达标域（val_auc >= mastery_auc）降到 min_domain_weight，而不是 base weight
    4. 数据比例只在前 focus_epochs 内从 start_ratio -> end_ratio；后期固定 100%
    """

    def __init__(
        self,
        dataset,
        total_epochs: int,
        domain_names: Optional[List[str]] = None,
        difficulty_metric: str = 'val_auc',   # 兼容保留
        min_domain_weight: float = 0.15,
        max_domain_weight: float = 0.40,
        focus_epochs: int = 12,
        transition_epochs: Optional[int] = None,  # 兼容保留
        eta: float = 2.0,
        start_ratio: float = 0.4,
        end_ratio: float = 1.0,
        mastery_auc: float = 0.97,
        max_focus_alpha: float = 0.60,
        base_weight_mode: str = "dataset",
        mastered_weight_mode: str = "min",     # "min" / "base"
        post_focus_enabled: bool = True,
        post_focus_alpha: float = 0.20,
        seed: int = 42
    ):
        self.dataset = dataset
        self.T = total_epochs
        self.eta = eta
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.min_domain_weight = min_domain_weight
        self.max_domain_weight = max_domain_weight
        self.mastery_auc = mastery_auc
        self.max_focus_alpha = max_focus_alpha
        self.base_weight_mode = base_weight_mode
        self.mastered_weight_mode = mastered_weight_mode
        self.post_focus_enabled = post_focus_enabled
        self.post_focus_alpha = post_focus_alpha
        self.seed = seed
        self.current_epoch = 0

        if focus_epochs is None and transition_epochs is not None:
            focus_epochs = transition_epochs
        self.focus_epochs = focus_epochs if focus_epochs is not None else min(12, total_epochs)

        self.is_distributed = dist.is_initialized()
        self.rank = dist.get_rank() if self.is_distributed else 0

        self._build_domain_structure(domain_names)
        self.base_domain_weights = self._build_base_weights()

        self.domain_val_auc = {d: 0.5 for d in self.domain_names}

        initial_total = self._compute_total_samples(0)
        initial_weights = self._compute_domain_weights(0)

        self.sampler = DomainWeightedSampler(
            domain_indices=self.domain_indices,
            domain_weights=initial_weights,
            total_samples=initial_total,
            seed=seed
        )

        if self.rank == 0:
            self._print_config()

    def _build_domain_structure(self, domain_names):
        self.domain_indices = defaultdict(list)
        for idx in range(len(self.dataset)):
            domain = self.dataset.get_domain(idx)
            self.domain_indices[domain].append(idx)

        if domain_names is not None:
            self.domain_names = [d for d in domain_names if d in self.domain_indices]
        else:
            self.domain_names = sorted(self.domain_indices.keys())

        self.n_total = len(self.dataset)

    def _build_base_weights(self) -> Dict[str, float]:
        if self.base_weight_mode == "uniform":
            w = 1.0 / len(self.domain_names)
            return {d: w for d in self.domain_names}

        counts = {d: len(self.domain_indices[d]) for d in self.domain_names}
        total = sum(counts.values())
        return {d: counts[d] / total for d in self.domain_names}

    def is_active(self, epoch: int) -> bool:
        # 新版里 curriculum 始终 active，只是前后阶段强度不同
        return True

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        total = sum(weights.values())
        if total <= 0:
            return {d: 1.0 / len(self.domain_names) for d in self.domain_names}
        return {d: weights.get(d, 0.0) / total for d in self.domain_names}

    def _normalize_subset(self, weights: Dict[str, float], subset: List[str]) -> Dict[str, float]:
        s = sum(weights[d] for d in subset)
        if s <= 0:
            v = 1.0 / len(subset)
            return {d: v for d in subset}
        return {d: weights[d] / s for d in subset}

    def _compute_progress(self, epoch: int) -> float:
        if self.focus_epochs <= 1:
            return 1.0
        return min(1.0, epoch / max(1, self.focus_epochs - 1))

    def _compute_alpha(self, epoch: int) -> float:
        if epoch < self.focus_epochs:
            p = self._compute_progress(epoch)
            return self.max_focus_alpha * p
        else:
            if self.post_focus_enabled:
                return self.post_focus_alpha
            else:
                return 0.0

    def _compute_data_ratio(self, epoch: int) -> float:
        if epoch < self.focus_epochs:
            p = self._compute_progress(epoch)
            ratio = self.start_ratio + (self.end_ratio - self.start_ratio) * p
            return float(np.clip(ratio, self.start_ratio, self.end_ratio))
        else:
            return self.end_ratio

    def _compute_total_samples(self, epoch: int) -> int:
        ratio = self._compute_data_ratio(epoch)
        return max(1, int(self.n_total * ratio))

    def update_val_metrics(self, domain_metrics: Dict[str, Dict]):
        updated = False
        for d in self.domain_names:
            if d in domain_metrics and "auc_roc" in domain_metrics[d]:
                self.domain_val_auc[d] = float(domain_metrics[d]["auc_roc"])
                updated = True

        if self.rank == 0 and updated:
            print(f"\n[Curriculum] 收到验证集 AUC 反馈:")
            for d in self.domain_names:
                flag = " (mastered)" if self.domain_val_auc[d] >= self.mastery_auc else ""
                print(f"  {d}: Val AUC = {self.domain_val_auc[d]:.4f}{flag}")

    def _get_mastered_fixed_weight(self, domain: str) -> float:
        if self.mastered_weight_mode == "min":
            return self.min_domain_weight
        elif self.mastered_weight_mode == "base":
            return self.base_domain_weights[domain]
        else:
            raise ValueError(f"未知 mastered_weight_mode: {self.mastered_weight_mode}")

    def _apply_weight_constraints(self, weights: Dict[str, float], fixed_domains: Optional[Dict[str, float]] = None):
        fixed_domains = fixed_domains or {}

        result = {d: 0.0 for d in self.domain_names}
        for d, v in fixed_domains.items():
            result[d] = v

        free_domains = [d for d in self.domain_names if d not in fixed_domains]
        fixed_mass = sum(result[d] for d in self.domain_names)
        free_mass = 1.0 - fixed_mass

        if free_mass <= 1e-12 or len(free_domains) == 0:
            return self._normalize_weights(result)

        raw = {d: max(weights.get(d, 0.0), 1e-12) for d in free_domains}

        while True:
            if len(raw) == 0:
                break

            raw_total = sum(raw.values())
            if raw_total <= 0:
                for d in raw:
                    raw[d] = 1.0
                raw_total = len(raw)

            proposed = {d: raw[d] / raw_total * free_mass for d in raw}

            changed = False
            to_remove = []

            for d, v in proposed.items():
                if v < self.min_domain_weight:
                    result[d] = self.min_domain_weight
                    free_mass -= self.min_domain_weight
                    to_remove.append(d)
                    changed = True
                elif v > self.max_domain_weight:
                    result[d] = self.max_domain_weight
                    free_mass -= self.max_domain_weight
                    to_remove.append(d)
                    changed = True

            for d in to_remove:
                raw.pop(d, None)

            if free_mass <= 1e-12:
                break

            if not changed:
                for d, v in proposed.items():
                    result[d] = v
                break

        return self._normalize_weights(result)

    def _compute_domain_weights(self, epoch: int) -> Dict[str, float]:
        alpha = self._compute_alpha(epoch)

        mastered_domains = [d for d in self.domain_names if self.domain_val_auc[d] >= self.mastery_auc]
        active_domains = [d for d in self.domain_names if d not in mastered_domains]

        # 如果全部达标，则回到均衡兜底
        if len(active_domains) == 0:
            return self._normalize_weights({
                d: self._get_mastered_fixed_weight(d) for d in self.domain_names
            })

        # 达标域固定到指定权重（通常是 min）
        fixed_weights = {d: self._get_mastered_fixed_weight(d) for d in mastered_domains}
        remain_mass = 1.0 - sum(fixed_weights.values())

        # 对未达标域：base -> hard 混合
        base_active = self._normalize_subset(self.base_domain_weights, active_domains)

        aucs = np.array([self.domain_val_auc[d] for d in active_domains], dtype=np.float64)
        hardness = 1.0 - aucs
        hardness = hardness - hardness.max()  # 数值稳定
        exp_hard = np.exp(self.eta * hardness)
        hard_active_arr = exp_hard / exp_hard.sum()
        hard_active = {d: hard_active_arr[i] for i, d in enumerate(active_domains)}

        blended_active = {}
        for d in active_domains:
            blended_active[d] = (1.0 - alpha) * base_active[d] + alpha * hard_active[d]

        weights = {d: 0.0 for d in self.domain_names}
        for d in mastered_domains:
            weights[d] = fixed_weights[d]
        for d in active_domains:
            weights[d] = blended_active[d] * remain_mass

        weights = self._apply_weight_constraints(weights, fixed_domains=fixed_weights)
        return weights

    def step(self):
        self.current_epoch += 1

        next_total = self._compute_total_samples(self.current_epoch)
        next_weights = self._compute_domain_weights(self.current_epoch)

        self.sampler.update_params(next_weights, next_total)
        self.sampler.set_epoch(self.current_epoch)

        if self.rank == 0:
            data_ratio = self._compute_data_ratio(self.current_epoch)
            alpha = self._compute_alpha(self.current_epoch)
            mastered_domains = [d for d in self.domain_names if self.domain_val_auc[d] >= self.mastery_auc]
            phase = "Focus" if self.current_epoch < self.focus_epochs else "Post-Focus"

            print(f"\n[Curriculum] Epoch {self.current_epoch} 准备就绪")
            print(f"  阶段: {phase}")
            print(f"  数据量比例: {data_ratio:.1%} ({next_total} 样本)")
            print(f"  强化强度 alpha: {alpha:.3f}")
            print(f"  达标域: {mastered_domains if len(mastered_domains) > 0 else '无'}")
            print(f"  达标域固定策略: {self.mastered_weight_mode}")
            print(f"  base weights: ", end="")
            for d in self.domain_names:
                print(f"{d}={self.base_domain_weights[d]:.3f} ", end="")
            print()
            print(f"  新域权重: ", end="")
            for d in self.domain_names:
                print(f"{d}={next_weights[d]:.3f} ", end="")
            print()

        return self.current_epoch

    def get_sampler(self):
        return self.sampler

    def _print_config(self):
        print(f"\n{'='*72}")
        print("升级版 DomainWeightedCurriculumManager")
        print(f"{'='*72}")
        print(f"  focus_epochs         : {self.focus_epochs}")
        print(f"  数据比例             : {self.start_ratio:.0%} -> {self.end_ratio:.0%}")
        print(f"  base_weight_mode     : {self.base_weight_mode}")
        print(f"  mastered_weight_mode : {self.mastered_weight_mode}")
        print(f"  mastery_auc          : {self.mastery_auc:.3f}")
        print(f"  eta                  : {self.eta:.3f}")
        print(f"  max_focus_alpha      : {self.max_focus_alpha:.3f}")
        print(f"  post_focus_enabled   : {self.post_focus_enabled}")
        print(f"  post_focus_alpha     : {self.post_focus_alpha:.3f}")
        print(f"  min/max_domain_w     : {self.min_domain_weight:.3f} / {self.max_domain_weight:.3f}")
        print(f"  策略                 : 前期强课程，后期弱自适应；达标域降到最小权重")
        print(f"{'='*72}\n")