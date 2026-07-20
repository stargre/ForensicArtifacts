import numpy as np
from collections import defaultdict


class LossProfileCurriculumManager:
    """
    基于全训练集 sample loss 的可插拔课程学习器

    机制：
    1. 每轮结束后，对全训练集前向，得到每个样本 loss
    2. 按 domain 分组排序
    3. 每个域取 top-k 高 loss 样本作为主训练样本（weight=1.0）
    4. 其余样本保留，但赋予较小权重（weight=small_weight）
    """

    def __init__(
        self,
        dataset,
        top_ratio=0.5,
        small_weight=0.2,
        domain_names=None,
        enable_after_profiling=True,
        rank=0
    ):
        self.dataset = dataset
        self.top_ratio = top_ratio
        self.small_weight = small_weight
        self.enable_after_profiling = enable_after_profiling
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

        # 保存每个样本的最近一次 loss
        self.sample_losses = np.zeros(self.n_total, dtype=np.float32)

        # 保存训练时使用的样本权重
        self.sample_weights = np.ones(self.n_total, dtype=np.float32)

        # 是否已经完成过一次全量 profiling
        self.is_ready = False

        if self.rank == 0:
            print("\n" + "=" * 72)
            print("LossProfileCurriculumManager")
            print("=" * 72)
            print(f"  top_ratio      : {self.top_ratio:.2f}")
            print(f"  small_weight   : {self.small_weight:.3f}")
            print(f"  策略           : 每域选 top-k 高loss样本为主训练集，其余样本保留小权重")
            print("=" * 72)

    def update_sample_losses(self, sample_losses_dict):
        """
        sample_losses_dict: {sample_idx: loss_value}
        """
        for idx, loss_val in sample_losses_dict.items():
            self.sample_losses[idx] = float(loss_val)

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

    def get_sample_weights(self):
        return self.sample_weights

    def get_weight_for_indices(self, indices):
        return self.sample_weights[indices]

    def is_enabled_for_training(self):
        return self.enable_after_profiling and self.is_ready