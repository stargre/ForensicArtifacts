# curriculum/reverse_curriculum_manager.py
import numpy as np
from torch.utils.data import Subset


class ReverseCurriculumManager:
    """
    反向课程学习管理器（Anti-Curriculum Learning）
    
    基于预计算的置信度分数，按从难到易的顺序逐步引入训练样本
    置信度越低的样本越"困难"，应该先训练
    
    理论依据：
    - 困难样本可能包含更多判别性信息
    - 先学习困难模式可以避免过早过拟合到简单模式
    """
    
    def __init__(self, 
                 dataset, 
                 total_epochs=10, 
                 schedule_type='linear', 
                 start_ratio=0.3, 
                 end_ratio=1.0,
                 warmup_epochs=0):
        """
        Args:
            dataset: 原始数据集，需要有 get_confidence_sorted_indices() 方法
            total_epochs: 课程学习持续的总epoch数（之后使用全部数据）
            schedule_type: 调度策略 ['linear', 'exponential', 'step', 'cosine', 'root']
            start_ratio: 初始数据比例（最困难的样本）
            end_ratio: 最终数据比例
            warmup_epochs: 预热期（保持 start_ratio）
        """
        self.dataset = dataset
        self.total_epochs = total_epochs
        self.schedule_type = schedule_type
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        
        # 获取按置信度排序的索引（从高到低），然后反转为从低到高（从难到易）
        sorted_indices_easy_to_hard = dataset.get_confidence_sorted_indices()
        self.sorted_indices = sorted_indices_easy_to_hard[::-1].copy()  # 反转：从难到易
        self.total_samples = len(self.sorted_indices)
        
        print(f"\n[ReverseCurriculumManager] 初始化完成:")
        print(f"  总样本数: {self.total_samples}")
        print(f"  调度策略: {schedule_type}")
        print(f"  数据比例: {start_ratio:.0%} → {end_ratio:.0%}")
        print(f"  课程周期: {total_epochs} epochs")
        print(f"  预热周期: {warmup_epochs} epochs")
        print(f"  模式: 反向课程学习（从难到易）")
    
    def get_current_ratio(self, epoch=None):
        """根据当前epoch计算应使用的数据比例"""
        if epoch is None:
            epoch = self.current_epoch
        
        # 预热期保持初始比例
        if epoch < self.warmup_epochs:
            return self.start_ratio
        
        # 调整后的进度
        adjusted_epoch = epoch - self.warmup_epochs
        adjusted_total = max(1, self.total_epochs - self.warmup_epochs)
        progress = min(adjusted_epoch / adjusted_total, 1.0)
        
        if self.schedule_type == 'linear':
            ratio = self.start_ratio + (self.end_ratio - self.start_ratio) * progress
            
        elif self.schedule_type == 'exponential':
            ratio = self.start_ratio + (self.end_ratio - self.start_ratio) * (1 - np.exp(-3 * progress))
            
        elif self.schedule_type == 'step':
            step_size = 0.1
            stage_length = 1
            stage = adjusted_epoch // stage_length
            ratio = self.start_ratio + stage * step_size
            ratio = min(ratio, self.end_ratio)
            
        elif self.schedule_type == 'cosine':
            ratio = self.start_ratio + (self.end_ratio - self.start_ratio) * \
                    0.5 * (1 - np.cos(np.pi * progress))
            
        elif self.schedule_type == 'root':
            ratio = self.start_ratio + (self.end_ratio - self.start_ratio) * np.sqrt(progress)
            
        else:
            ratio = self.end_ratio
        
        return max(self.start_ratio, min(self.end_ratio, ratio))
    
    def get_current_indices(self, epoch=None):
        """获取当前epoch应该使用的样本索引"""
        ratio = self.get_current_ratio(epoch)
        num_samples = int(self.total_samples * ratio)
        num_samples = max(1, num_samples)
        
        # 返回置信度最低（最困难）的前 num_samples 个样本的索引
        return self.sorted_indices[:num_samples]
    
    def get_current_subset(self, epoch=None):
        """获取当前epoch的数据子集"""
        indices = self.get_current_indices(epoch)
        return Subset(self.dataset, indices)
    
    def step(self):
        """更新epoch计数并返回新的数据比例"""
        self.current_epoch += 1
        new_ratio = self.get_current_ratio()
        num_samples = int(self.total_samples * new_ratio)
        
        print(f"[ReverseCurriculum] Epoch {self.current_epoch}: "
              f"ratio={new_ratio:.1%}, samples={num_samples}/{self.total_samples}")
        
        return new_ratio
    
    def get_stats(self):
        """获取当前状态统计信息"""
        ratio = self.get_current_ratio()
        num_samples = int(self.total_samples * ratio)
        indices = self.get_current_indices()
        
        confidences = [self.dataset.get_confidence(idx) for idx in indices]
        
        return {
            'epoch': self.current_epoch,
            'ratio': ratio,
            'num_samples': num_samples,
            'total_samples': self.total_samples,
            'min_confidence': min(confidences) if confidences else 0,
            'max_confidence': max(confidences) if confidences else 0,
            'mean_confidence': np.mean(confidences) if confidences else 0,
            'mode': 'reverse'  # 标识反向课程学习
        }