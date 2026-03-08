# curriculum/adaptive_curriculum_manager.py

"""
自适应课程学习管理器（Adaptive Curriculum Learning Manager）

核心特性：
1. 数据量线性增长：0.3 → 1.0，每2轮更新比例
2. 难度动态更新：难度 = f(初始置信度, 训练损失)
3. 高置信度 = 简单样本（保持原始语义）
4. 加权随机采样：避免batch内难度单一
5. 完整分布式支持：多GPU损失聚合

作者：最终优化版本
"""

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from typing import Dict, Any, List, Optional
from collections import defaultdict


class AdaptiveCurriculumSampler(Sampler):
    """
    自适应课程学习采样器
    
    功能：
    1. 根据置信度选择 top_k 个最简单的样本（高置信度 = 简单）
    2. 对选中的样本进行加权随机采样
    3. 支持分布式训练
    """
    
    def __init__(self, 
                 confidence_scores: np.ndarray,
                 top_k: int,
                 confidence_weight: float = 0.5,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 seed: int = 42):
        """
        Args:
            confidence_scores: 全部样本的置信度分数 [0, 1]
                              高置信度 = 简单样本
            top_k: 本轮选择的样本数量
            confidence_weight: 置信度对采样顺序的影响 [0, 1]
                - 0.0: 完全随机（忽略置信度）
                - 0.5: 中等引导（推荐）
                - 1.0: 严格按置信度排序
            num_replicas: 分布式进程数
            rank: 当前进程编号
            seed: 随机种子
        """
        # 分布式参数
        if num_replicas is None:
            if dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size()
            else:
                num_replicas = 1
        if rank is None:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0
        
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        
        self.confidence_scores = confidence_scores
        self.total_size = len(confidence_scores)
        self.top_k = min(top_k, self.total_size)
        self.confidence_weight = np.clip(confidence_weight, 0.0, 1.0)
        
        # 每个进程分到的样本数
        self.num_samples = (self.top_k + self.num_replicas - 1) // self.num_replicas
        self.total_size_padded = self.num_samples * self.num_replicas
    
    def _select_and_shuffle(self, rng: np.random.RandomState) -> List[int]:
        """
        两阶段采样：
        1. 选择：从全部样本中选出置信度最高的 top_k 个（高置信度=简单）
        2. 打乱：对这 top_k 个样本进行加权随机打乱
        """
        # 阶段1：选择置信度最高的 top_k 个样本
        # argsort 升序，取最后 top_k 个（最高置信度）
        sorted_by_conf = np.argsort(self.confidence_scores)
        selected_indices = sorted_by_conf[-self.top_k:]  # 取置信度最高的
        selected_confidences = self.confidence_scores[selected_indices]
        
        # 阶段2：对选中的样本进行加权打乱
        k = len(selected_indices)
        w = self.confidence_weight
        
        if w <= 0 or k <= 1:
            # 完全随机
            rng.shuffle(selected_indices)
            return selected_indices.tolist()
        
        if w >= 1.0:
            # 严格按置信度排序（从高到低）
            order = np.argsort(-selected_confidences)
            return selected_indices[order].tolist()
        
        # 混合策略：
        # - 计算每个样本在选中集合中的相对排名 [0, 1]
        # - 高置信度样本排名低（更容易排在前面）
        # - 生成随机扰动
        # - 按混合键重新排序
        
        # 置信度越高，rank 越小（排在前面）
        conf_order = np.argsort(-selected_confidences)  # 降序排列的索引
        relative_rank = np.zeros(k)
        relative_rank[conf_order] = np.arange(k) / max(k - 1, 1)
        
        random_perturbation = rng.random(k)
        mixed_key = w * relative_rank + (1 - w) * random_perturbation
        
        reorder = np.argsort(mixed_key)
        return selected_indices[reorder].tolist()
    
    def update_params(self, confidence_scores: np.ndarray, top_k: int, confidence_weight: float):
        """更新采样器参数"""
        self.confidence_scores = confidence_scores.copy()
        self.top_k = min(top_k, self.total_size)
        self.confidence_weight = np.clip(confidence_weight, 0.0, 1.0)
        
        # 重新计算分片大小
        self.num_samples = (self.top_k + self.num_replicas - 1) // self.num_replicas
        self.total_size_padded = self.num_samples * self.num_replicas
    
    def set_epoch(self, epoch: int):
        """设置epoch（分布式种子同步）"""
        self.epoch = epoch
    
    def __iter__(self):
        # 生成全局索引（所有进程用相同种子）
        rng = np.random.RandomState(self.seed + self.epoch)
        indices = self._select_and_shuffle(rng)
        
        # Padding
        if len(indices) < self.total_size_padded:
            extra = self.total_size_padded - len(indices)
            indices = indices + indices[:extra]
        
        # 分布式分片：交错取样，确保每个GPU看到不同数据
        indices = indices[self.rank:self.total_size_padded:self.num_replicas]
        
        assert len(indices) == self.num_samples, \
            f"索引数量 {len(indices)} != 预期 {self.num_samples}"
        
        return iter(indices)
    
    def __len__(self):
        return self.num_samples


class AdaptiveCurriculumManager:
    """
    自适应课程学习管理器
    
    核心机制：
    1. 数据量线性增长：0.3 → 1.0，每2轮更新比例
    2. 置信度动态更新：置信度 = α × 初始置信度 + β × (1 - 归一化损失)
       - 高置信度 = 简单样本（损失低的样本置信度高）
    3. 加权随机采样：根据置信度加权，高置信度样本更可能被采样
    4. 完整分布式支持：多GPU损失聚合
    
    训练流程：
    Epoch 1-2:   30% 数据，完全随机采样（预热期）
    Epoch 3-4:   30% 数据，开始更新置信度，轻度加权采样
    Epoch 5-6:   44% 数据，继续更新置信度
    Epoch 7-8:   58% 数据，...
    ...
    Epoch N:     100% 数据，使用最终置信度加权采样
    """
    
    def __init__(self,
                 dataset,
                 total_epochs: int = 30,
                 
                 # 数据量增长配置
                 start_ratio: float = 0.3,
                 end_ratio: float = 1.0,
                 ratio_update_frequency: int = 2,  # 每2轮更新一次数据比例
                 
                 # 置信度更新配置
                 warmup_epochs: int = 3,
                 confidence_update_frequency: int = 1,  # 每1轮更新一次置信度
                 initial_weight: float = 0.5,
                 loss_weight: float = 0.5,
                 confidence_momentum: float = 0.9,
                 loss_normalization: str = 'percentile',
                 use_kl_regularization: bool = True,
                 kl_weight: float = 0.1,
                 
                 # 采样策略配置
                 confidence_weight_schedule: str = 'linear',  # 'fixed', 'linear', 'staged'
                 initial_confidence_weight: float = 0.0,  # 初始：完全随机
                 final_confidence_weight: float = 0.7,    # 最终：强引导
                 
                 # 系统配置
                 seed: int = 42):
        """
        Args:
            dataset: 训练数据集，需要有 get_confidence(idx) 方法
            total_epochs: 总训练轮数
            
            # === 数据量增长 ===
            start_ratio: 初始数据比例（如0.3表示30%）
            end_ratio: 最终数据比例（通常1.0）
            ratio_update_frequency: 每N轮更新一次数据比例
            
            # === 置信度更新 ===
            warmup_epochs: 预热期（期间不更新置信度）
            confidence_update_frequency: 每N轮更新一次置信度
            initial_weight: 初始置信度的权重
            loss_weight: 损失反馈的权重
            confidence_momentum: EMA动量（越大越平滑）
            loss_normalization: 损失归一化方式
            use_kl_regularization: 是否使用KL散度正则化
            kl_weight: KL散度权重
            
            # === 采样策略 ===
            confidence_weight_schedule: 置信度权重调度策略
            initial_confidence_weight: 初始置信度权重（0=随机）
            final_confidence_weight: 最终置信度权重
            
            seed: 随机种子
        """
        self.dataset = dataset
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.seed = seed
        
        # === 数据量增长配置 ===
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.ratio_update_frequency = ratio_update_frequency
        
        # === 置信度更新配置 ===
        self.warmup_epochs = warmup_epochs
        self.confidence_update_frequency = confidence_update_frequency
        self.initial_weight = initial_weight
        self.loss_weight = loss_weight
        self.confidence_momentum = confidence_momentum
        self.loss_normalization = loss_normalization
        self.use_kl_regularization = use_kl_regularization
        self.kl_weight = kl_weight
        
        # 归一化权重
        self._normalize_weights()
        
        # === 采样策略配置 ===
        self.confidence_weight_schedule = confidence_weight_schedule
        self.initial_confidence_weight = initial_confidence_weight
        self.final_confidence_weight = final_confidence_weight
        
        # === 数据集信息 ===
        self.total_samples = len(dataset)
        
        # === 分布式配置 ===
        self.is_distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.is_distributed else 0
        self.world_size = dist.get_world_size() if self.is_distributed else 1
        
        # === 置信度分数（高置信度 = 简单样本）===
        self._init_confidence_scores()
        self.previous_confidence_scores = self.confidence_scores.copy()
        
        # === 损失记录 ===
        self.epoch_losses = {}
        self.loss_history = defaultdict(list)
        self.update_count = np.zeros(self.total_samples)
        
        # === 创建采样器 ===
        initial_top_k = self._compute_top_k(0)
        initial_conf_weight = self._compute_confidence_weight(0)
        
        self.sampler = AdaptiveCurriculumSampler(
            confidence_scores=self.confidence_scores,
            top_k=initial_top_k,
            confidence_weight=initial_conf_weight,
            seed=seed,
        )
        
        # === 打印配置 ===
        if self.rank == 0:
            self._print_config()
    
    def _normalize_weights(self):
        """归一化初始权重和损失权重"""
        total = self.initial_weight + self.loss_weight
        if total > 0:
            self.initial_weight /= total
            self.loss_weight /= total
        else:
            self.initial_weight = 0.5
            self.loss_weight = 0.5
    
    def _init_confidence_scores(self):
        """
        初始化置信度分数
        
        高置信度 = 简单样本
        """
        self.confidence_scores = np.zeros(self.total_samples)
        self.initial_confidence = np.zeros(self.total_samples)
        
        for i in range(self.total_samples):
            confidence = self.dataset.get_confidence(i)
            self.initial_confidence[i] = confidence
            self.confidence_scores[i] = confidence  # 保持原始语义
        
        self._normalize_confidence_scores()
    
    def _normalize_confidence_scores(self):
        """归一化置信度分数到 [0, 1]"""
        min_c = self.confidence_scores.min()
        max_c = self.confidence_scores.max()
        if max_c > min_c:
            self.confidence_scores = (self.confidence_scores - min_c) / (max_c - min_c)
        else:
            self.confidence_scores[:] = 0.5
    
    def _compute_data_ratio(self, epoch: int) -> float:
        """
        计算当前epoch应该使用的数据比例
        
        线性增长，每 ratio_update_frequency 轮更新一次
        """
        if epoch >= self.total_epochs:
            return self.end_ratio
        
        # 计算当前处于第几个"比例更新周期"
        update_step = epoch // self.ratio_update_frequency
        total_steps = self.total_epochs // self.ratio_update_frequency
        
        if total_steps <= 0:
            return self.end_ratio
        
        # 线性插值
        progress = min(update_step / total_steps, 1.0)
        ratio = self.start_ratio + (self.end_ratio - self.start_ratio) * progress
        
        return np.clip(ratio, self.start_ratio, self.end_ratio)
    
    def _compute_top_k(self, epoch: int) -> int:
        """计算当前epoch应该选择的样本数量"""
        ratio = self._compute_data_ratio(epoch)
        top_k = int(self.total_samples * ratio)
        return max(1, min(top_k, self.total_samples))
    
    def _compute_confidence_weight(self, epoch: int) -> float:
        """
        计算当前epoch的置信度权重（用于加权采样）
        
        Returns:
            float: 置信度权重 [0, 1]
        """
        if epoch < self.warmup_epochs:
            # 预热期：完全随机
            return 0.0
        
        if self.confidence_weight_schedule == 'fixed':
            return self.initial_confidence_weight
        
        elif self.confidence_weight_schedule == 'linear':
            # 从预热期结束开始线性增长
            adjusted_epoch = epoch - self.warmup_epochs
            adjusted_total = max(1, self.total_epochs - self.warmup_epochs)
            progress = min(adjusted_epoch / adjusted_total, 1.0)
            
            return (self.initial_confidence_weight + 
                    (self.final_confidence_weight - self.initial_confidence_weight) * progress)
        
        elif self.confidence_weight_schedule == 'staged':
            # 分阶段调整
            current_ratio = self._compute_data_ratio(epoch)
            
            if current_ratio <= 0.4:
                return 0.2  # 早期：轻度引导
            elif current_ratio <= 0.7:
                return 0.5  # 中期：中度引导
            else:
                return 0.7  # 后期：强引导
        
        else:
            return 0.5
    
    def _normalize_losses(self, losses: np.ndarray) -> np.ndarray:
        """归一化损失值到 [0, 1]"""
        if len(losses) == 0:
            return losses
        
        if self.loss_normalization == 'minmax':
            mn, mx = losses.min(), losses.max()
            return (losses - mn) / (mx - mn) if mx > mn else np.zeros_like(losses)
        
        elif self.loss_normalization == 'zscore':
            mean, std = losses.mean(), losses.std()
            if std > 0:
                normalized = (losses - mean) / std
                return 1.0 / (1.0 + np.exp(-normalized))
            return np.ones_like(losses) * 0.5
        
        elif self.loss_normalization == 'percentile':
            p25, p75 = np.percentile(losses, 25), np.percentile(losses, 75)
            if p75 > p25:
                return np.clip((losses - p25) / (p75 - p25), 0, 1)
            return np.ones_like(losses) * 0.5
        
        return losses
    
    def _compute_kl_divergence(self, new_scores: np.ndarray, old_scores: np.ndarray) -> float:
        """计算KL散度"""
        new_s = np.clip(new_scores, 1e-10, 1.0)
        old_s = np.clip(old_scores, 1e-10, 1.0)
        new_p = new_s / (new_s.sum() + 1e-10)
        old_p = old_s / (old_s.sum() + 1e-10)
        return float(np.sum(old_p * np.log((old_p + 1e-10) / (new_p + 1e-10))))
    
    def record_batch_losses(self, sample_indices, losses):
        """
        记录一批样本的损失
        
        Args:
            sample_indices: 样本索引 (batch_size,)
            losses: 对应的损失值 (batch_size,)
        """
        if isinstance(sample_indices, torch.Tensor):
            sample_indices = sample_indices.cpu().numpy()
        if isinstance(losses, torch.Tensor):
            losses = losses.detach().cpu().numpy()
        
        sample_indices = np.atleast_1d(sample_indices)
        losses = np.atleast_1d(losses)
        
        if len(sample_indices) != len(losses):
            # 处理维度不匹配的情况
            min_len = min(len(sample_indices), len(losses))
            sample_indices = sample_indices[:min_len]
            losses = losses[:min_len]
        
        for idx, loss in zip(sample_indices, losses):
            idx = int(idx)
            if 0 <= idx < self.total_samples:
                self.epoch_losses[idx] = float(loss)
    
    def _gather_losses_distributed(self):
        """分布式聚合所有进程的损失"""
        if not self.is_distributed:
            return
        
        local_indices = list(self.epoch_losses.keys())
        local_losses = list(self.epoch_losses.values())
        
        if len(local_indices) == 0:
            local_indices = [-1]
            local_losses = [0.0]
        
        # 获取各进程数据量
        local_count = torch.tensor([len(local_indices)], dtype=torch.long, device='cuda')
        all_counts = [torch.zeros(1, dtype=torch.long, device='cuda') 
                      for _ in range(self.world_size)]
        dist.all_gather(all_counts, local_count)
        max_count = max(c.item() for c in all_counts)
        
        # Padding
        pad_indices = local_indices + [-1] * (max_count - len(local_indices))
        pad_losses = local_losses + [0.0] * (max_count - len(local_losses))
        
        idx_tensor = torch.tensor(pad_indices, dtype=torch.long, device='cuda')
        loss_tensor = torch.tensor(pad_losses, dtype=torch.float32, device='cuda')
        
        all_idx = [torch.zeros(max_count, dtype=torch.long, device='cuda') 
                   for _ in range(self.world_size)]
        all_loss = [torch.zeros(max_count, dtype=torch.float32, device='cuda') 
                    for _ in range(self.world_size)]
        
        dist.all_gather(all_idx, idx_tensor)
        dist.all_gather(all_loss, loss_tensor)
        
        # 合并取平均
        merged_sums = defaultdict(float)
        merged_counts = defaultdict(int)
        
        for r_idx, r_loss in zip(all_idx, all_loss):
            for i, l in zip(r_idx.cpu().numpy(), r_loss.cpu().numpy()):
                if i >= 0:
                    merged_sums[int(i)] += float(l)
                    merged_counts[int(i)] += 1
        
        self.epoch_losses = {
            idx: merged_sums[idx] / merged_counts[idx]
            for idx in merged_sums
        }
    
    def update_confidence_scores(self):
        """
        根据收集的损失更新置信度分数
        
        置信度 = α × 初始置信度 + β × (1 - 归一化损失)
        
        高置信度 = 简单样本（损失低的样本置信度高）
        """
        # 分布式聚合
        if self.is_distributed:
            self._gather_losses_distributed()
        
        if len(self.epoch_losses) == 0:
            if self.rank == 0:
                print("[AdaptiveCurriculum] 警告：没有收集到损失数据，跳过更新")
            return 0
        
        indices = np.array(list(self.epoch_losses.keys()))
        losses = np.array(list(self.epoch_losses.values()))
        normalized_losses = self._normalize_losses(losses)
        
        old_scores = self.confidence_scores.copy()
        
        # 更新置信度分数
        updated_count = 0
        for idx, norm_loss in zip(indices, normalized_losses):
            # 置信度 = α × 初始置信度 + β × (1 - 归一化损失)
            # 损失越低，(1 - norm_loss) 越高，置信度越高
            loss_confidence = 1.0 - norm_loss
            
            new_confidence = (self.initial_weight * self.initial_confidence[idx] + 
                             self.loss_weight * loss_confidence)
            
            # EMA平滑更新
            self.confidence_scores[idx] = (
                self.confidence_momentum * self.confidence_scores[idx] +
                (1 - self.confidence_momentum) * new_confidence
            )
            
            self.loss_history[idx].append(norm_loss)
            self.update_count[idx] += 1
            updated_count += 1
        
        # KL散度正则化
        if self.use_kl_regularization:
            kl_div = self._compute_kl_divergence(self.confidence_scores, old_scores)
            if kl_div > self.kl_weight:
                blend = self.kl_weight / (kl_div + 1e-10)
                self.confidence_scores = (
                    blend * self.confidence_scores +
                    (1 - blend) * old_scores
                )
        
        self._normalize_confidence_scores()
        self.previous_confidence_scores = old_scores
        self.epoch_losses = {}
        
        if self.rank == 0:
            print(f"[AdaptiveCurriculum] 置信度已更新: "
                  f"样本数={updated_count}, "
                  f"平均置信度={self.confidence_scores.mean():.4f}, "
                  f"标准差={self.confidence_scores.std():.4f}")
        
        return updated_count
    
    def should_update_confidence(self, epoch: int) -> bool:
        """判断是否应该更新置信度分数"""
        if epoch < self.warmup_epochs:
            return False
        return (epoch - self.warmup_epochs) % self.confidence_update_frequency == 0
    
    def step(self):
        """
        Epoch结束时调用
        
        1. 更新置信度分数（如果需要）
        2. 更新采样器参数（数据比例、置信度权重）
        3. 同步epoch种子
        """
        self.current_epoch += 1
        
        # 更新置信度分数
        if self.should_update_confidence(self.current_epoch):
            self.update_confidence_scores()
        else:
            self.epoch_losses = {}
        
        # 计算下一轮的参数
        next_top_k = self._compute_top_k(self.current_epoch)
        next_conf_weight = self._compute_confidence_weight(self.current_epoch)
        
        # 更新采样器参数
        self.sampler.update_params(
            confidence_scores=self.confidence_scores,
            top_k=next_top_k,
            confidence_weight=next_conf_weight
        )
        
        # 同步epoch种子（分布式）
        self.sampler.set_epoch(self.current_epoch)
        
        return self.current_epoch
    
    def get_sampler(self):
        """获取当前的采样器"""
        return self.sampler
    
    def get_stats(self) -> Dict[str, Any]:
        """获取当前状态统计"""
        current_ratio = self._compute_data_ratio(self.current_epoch)
        current_top_k = self._compute_top_k(self.current_epoch)
        current_conf_weight = self._compute_confidence_weight(self.current_epoch)
        
        return {
            'epoch': self.current_epoch,
            'total_samples': self.total_samples,
            'num_samples': current_top_k,
            'ratio': current_ratio,
            'confidence_weight': current_conf_weight,
            'mean_confidence': float(self.confidence_scores.mean()),
            'std_confidence': float(self.confidence_scores.std()),
            'min_confidence': float(self.confidence_scores.min()),
            'max_confidence': float(self.confidence_scores.max()),
            'samples_updated': int(np.sum(self.update_count > 0)),
            'is_warmup': self.current_epoch < self.warmup_epochs,
        }
    
    def get_confidence_distribution(self) -> Dict[str, np.ndarray]:
        """获取置信度分数分布（用于可视化）"""
        return {
            'confidence_scores': self.confidence_scores.copy(),
            'initial_confidence': self.initial_confidence.copy(),
            'update_count': self.update_count.copy(),
        }
    
    def get_sorted_indices(self) -> np.ndarray:
        """获取按置信度排序的索引（从高到低）"""
        return np.argsort(-self.confidence_scores)
    
    def _print_config(self):
        """打印配置信息"""
        print(f"\n{'='*70}")
        print(f"🎯 自适应课程学习管理器（Adaptive Curriculum Learning）")
        print(f"{'='*70}")
        
        print(f"\n📊 数据集信息:")
        print(f"  总样本数: {self.total_samples}")
        print(f"  数据比例: {self.start_ratio:.0%} → {self.end_ratio:.0%}")
        print(f"  比例更新频率: 每 {self.ratio_update_frequency} 轮")
        print(f"  增长方式: 线性增长")
        
        print(f"\n🔄 置信度更新:")
        print(f"  预热周期: {self.warmup_epochs} epochs（不更新置信度）")
        print(f"  更新频率: 每 {self.confidence_update_frequency} epoch")
        print(f"  权重配置: 初始置信度={self.initial_weight:.2f}, "
              f"损失反馈={self.loss_weight:.2f}")
        print(f"  EMA动量: {self.confidence_momentum}")
        print(f"  损失归一化: {self.loss_normalization}")
        print(f"  KL正则化: {'启用' if self.use_kl_regularization else '禁用'} "
              f"(权重={self.kl_weight})")
        
        print(f"\n📈 采样策略:")
        print(f"  调度方式: {self.confidence_weight_schedule}")
        print(f"  置信度权重: {self.initial_confidence_weight} → "
              f"{self.final_confidence_weight}")
        print(f"  高置信度 = 简单样本（优先采样）")
        
        print(f"\n🖥️  分布式:")
        print(f"  启用: {'是' if self.is_distributed else '否'}")
        print(f"  进程数: {self.world_size}")
        print(f"  当前rank: {self.rank}")
        
        # 打印数据比例增长计划
        print(f"\n📅 数据比例增长计划:")
        for e in range(0, self.total_epochs, self.ratio_update_frequency):
            ratio = self._compute_data_ratio(e)
            top_k = self._compute_top_k(e)
            conf_w = self._compute_confidence_weight(e)
            warmup_mark = " (预热期)" if e < self.warmup_epochs else ""
            print(f"  Epoch {e+1:2d}-{e+self.ratio_update_frequency:2d}: "
                  f"数据={ratio:5.1%} ({top_k:5d}样本), "
                  f"置信度权重={conf_w:.2f}{warmup_mark}")
        
        print(f"{'='*70}\n")