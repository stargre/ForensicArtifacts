# pre_data/dataprocess.py
import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np
from collections import Counter
from PIL import ImageFile


# ======================== 数据验证器 ========================
class DataValidator:
    """
    JSON数据格式验证器（OpenMMSecV2格式，特征版本）
    """
    @staticmethod
    def validate_json_format(json_path, strict_mode=False):
        """
        验证JSON文件格式并过滤无效样本（特征版本）
        """
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON格式错误: {str(e)}")
        except Exception as e:
            raise ValueError(f"读取文件失败: {str(e)}")

        if not isinstance(data, list):
            if strict_mode:
                raise ValueError("JSON根节点必须是列表类型")
            else:
                print("[WARNING] JSON非列表格式，尝试转换...")
                data = [data] if isinstance(data, dict) else []

        if len(data) == 0:
            raise ValueError("JSON列表为空")

        # ✅ 新增：检查 confidence 字段
        required_keys = ["label", "feature_path"]
        optional_keys = ["confidence"]  # 课程学习需要

        valid_samples = []
        stats = {
            'total': len(data),
            'valid': 0,
            'missing_fields': 0,
            'missing_feature': 0,
            'corrupted_feature': 0,
            'invalid_label': 0,
            'missing_confidence': 0,  # ✅ 新增
        }

        for i, sample in enumerate(data):
            if not isinstance(sample, dict):
                stats['missing_fields'] += 1
                continue

            missing_keys = [k for k in required_keys if k not in sample]
            if missing_keys:
                stats['missing_fields'] += 1
                if strict_mode:
                    raise ValueError(f"样本 {i} 缺少必需字段: {missing_keys}")
                continue

            label = sample["label"]
            if label not in [0, 1]:
                stats['invalid_label'] += 1
                if strict_mode:
                    raise ValueError(f"样本 {i} 标签必须是0或1")
                continue

            feature_path = sample["feature_path"]
            if feature_path is None or not os.path.exists(feature_path):
                stats['missing_feature'] += 1
                if strict_mode:
                    raise FileNotFoundError(f"特征文件不存在: {feature_path}")
                continue

            # 验证特征文件完整性
            try:
                with np.load(feature_path) as npz:
                    for key in ['scene', 'signal', 'imaging']:
                        if key not in npz:
                            raise ValueError(f"特征文件缺少 '{key}' 字段")
                        arr = npz[key]
                        if arr.size == 0:
                            raise ValueError(f"特征 '{key}' 为空数组")
            except Exception as e:
                stats['corrupted_feature'] += 1
                if strict_mode:
                    raise ValueError(f"特征文件损坏或无法加载: {feature_path}") from e
                continue

            # ✅ 新增：检查并补充 confidence 字段
            if 'confidence' not in sample or sample['confidence'] is None:
                sample['confidence'] = 0.5  # 默认中等置信度
                stats['missing_confidence'] += 1

            valid_samples.append(sample)
            stats['valid'] += 1

        print(f"\n[验证统计]")
        print(f"  原始样本数: {stats['total']}")
        print(f"  有效样本数: {stats['valid']}")
        print(f"  缺少字段: {stats['missing_fields']}")
        print(f"  特征缺失: {stats['missing_feature']}")
        print(f"  特征损坏: {stats['corrupted_feature']}")
        print(f"  标签非法: {stats['invalid_label']}")
        print(f"  缺少置信度: {stats['missing_confidence']}")  # ✅ 新增

        if stats['valid'] == 0:
            raise ValueError("没有有效样本！")

        return valid_samples, stats

    @staticmethod
    def get_dataset_statistics(samples):
        """获取数据集详细统计信息"""
        total = len(samples)
        label_counts = Counter(s['label'] for s in samples)
        domain_counts = Counter(s.get('domain', 'Unknown') for s in samples)
        mani_type_counts = Counter(
            s.get('mani_type', 'Unknown')
            for s in samples if s['label'] == 1
        )

        # ✅ 新增：统计置信度分布
        confidences = [s.get('confidence', 0.5) for s in samples]
        confidence_stats = {
            'min': min(confidences) if confidences else 0,
            'max': max(confidences) if confidences else 0,
            'mean': np.mean(confidences) if confidences else 0,
            'std': np.std(confidences) if confidences else 0,
        }

        return {
            'total': total,
            'label_distribution': dict(label_counts),
            'domain_distribution': dict(domain_counts),
            'mani_type_distribution': dict(mani_type_counts),
            'confidence_stats': confidence_stats,  # ✅ 新增
        }


# ======================== 特征数据集（支持课程学习）========================
class ForensicFeatureDataset(Dataset):
    """
    虚假图像检测数据集（加载预提取特征版本，支持课程学习）

    特性:
        - 直接加载 .npz 特征文件，无需加载原图
        - 返回 scene / signal / imaging 三个分支的特征张量
        - 支持多维度筛选
        - ✅ 支持按置信度排序（课程学习）
    """
    def __init__(self,
                 json_path,
                 is_train=True,
                 target_domains=None,
                 target_labels=None,
                 target_mani_types=None,
                 strict_mode=False):
        """
        Args:
            json_path: JSON配置文件路径（含 feature_path 和 confidence 字段）
            is_train: 是否为训练模式
            target_domains: 筛选特定域
            target_labels: 筛选特定标签
            target_mani_types: 筛选特定操作类型
            strict_mode: 严格验证模式
        """
        self.json_path = json_path
        self.is_train = is_train

        # 验证JSON并加载样本
        print(f"\n{'='*60}")
        print(f"初始化特征数据集: {json_path}")
        print(f"模式: {'训练' if is_train else '验证/测试'}")
        print(f"{'='*60}")

        self.full_samples, self.validation_stats = DataValidator.validate_json_format(
            json_path,
            strict_mode=strict_mode
        )

        # 应用筛选
        self.samples = self._apply_filters(
            self.full_samples,
            target_domains,
            target_labels,
            target_mani_types
        )

        # ✅ 新增：构建置信度索引（用于课程学习）
        self._build_confidence_index()

        # 打印统计信息
        self._print_statistics()

    def _apply_filters(self, samples, domains, labels, mani_types):
        """应用多维度筛选"""
        filtered = samples

        if domains is not None:
            if isinstance(domains, str):
                domains = [domains]
            filtered = [s for s in filtered if s.get('domain') in domains]
            print(f"  [筛选] 域: {domains} → 剩余 {len(filtered)} 样本")

        if labels is not None:
            if isinstance(labels, int):
                labels = [labels]
            filtered = [s for s in filtered if s['label'] in labels]
            print(f"  [筛选] 标签: {labels} → 剩余 {len(filtered)} 样本")

        if mani_types is not None:
            if isinstance(mani_types, str):
                mani_types = [mani_types]
            filtered = [s for s in filtered if s.get('mani_type') in mani_types]
            print(f"  [筛选] 操作类型: {mani_types} → 剩余 {len(filtered)} 样本")

        if len(filtered) == 0:
            raise ValueError("筛选后无有效样本！")

        return filtered

    # 新增：构建置信度索引
    def _build_confidence_index(self):
        """
        构建置信度索引（从高到低排序）
        用于课程学习中按难易度选择样本
        """
        # 提取所有样本的置信度和索引
        self.confidences = []
        for idx, item in enumerate(self.samples):
            conf = item.get('confidence', 0.5)  # 默认置信度0.5
            self.confidences.append((idx, conf))
        
        # 按置信度从高到低排序（置信度越高越可信，越"简单"）
        self.confidences.sort(key=lambda x: -x[1])
        
        # 排序后的索引列表
        self._sorted_indices = [idx for idx, _ in self.confidences]
        
        # 打印置信度分布
        confs = [c for _, c in self.confidences]
        if confs:
            print(f"\n[置信度分布]")
            print(f"  最小值: {min(confs):.4f}")
            print(f"  最大值: {max(confs):.4f}")
            print(f"  平均值: {np.mean(confs):.4f}")
            print(f"  标准差: {np.std(confs):.4f}")
            
            # 打印分位数
            percentiles = [25, 50, 75, 90, 95, 99]
            print(f"  分位数: ", end="")
            for p in percentiles:
                val = np.percentile(confs, p)
                print(f"P{p}={val:.4f} ", end="")
            print()

    # 获取按置信度排序的索引
    def get_confidence_sorted_indices(self):
        """
        获取按置信度排序的索引列表（从高到低）
        用于课程学习管理器
        
        Returns:
            list: 排序后的索引列表
        """
        return self._sorted_indices.copy()

    # ✅ 新增：获取指定索引的置信度
    def get_confidence(self, idx):
        """
        获取指定索引样本的置信度
        
        Args:
            idx: 样本索引
            
        Returns:
            float: 置信度值
        """
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"索引 {idx} 超出范围 [0, {len(self.samples)})")
        return self.samples[idx].get('confidence', 0.5)

    def _print_statistics(self):
        """打印数据集统计信息"""
        stats = DataValidator.get_dataset_statistics(self.samples)

        print(f"\n{'='*60}")
        print(f"数据集统计")
        print(f"{'='*60}")
        print(f"总样本数: {stats['total']}")

        label_dist = stats['label_distribution']
        print(f"\n标签分布:")
        for label, count in sorted(label_dist.items()):
            label_name = "Real" if label == 0 else "Fake"
            print(f"  {label_name}: {count} ({count/stats['total']*100:.2f}%)")

        if stats['domain_distribution']:
            print(f"\n域分布:")
            for domain, count in sorted(stats['domain_distribution'].items(),
                                       key=lambda x: -x[1])[:10]:
                print(f"  {domain}: {count}")

        # ✅ 新增：打印置信度统计
        conf_stats = stats.get('confidence_stats', {})
        if conf_stats:
            print(f"\n置信度统计:")
            print(f"  范围: [{conf_stats['min']:.4f}, {conf_stats['max']:.4f}]")
            print(f"  均值: {conf_stats['mean']:.4f}")
            print(f"  标准差: {conf_stats['std']:.4f}")

        print(f"\n配置信息:")
        print(f"  训练模式: {'是' if self.is_train else '否'}")
        print(f"  数据来源: 预提取特征 (.npz)")
        print(f"  支持课程学习: 是")  # ✅ 新增
        print(f"{'='*60}\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        获取单个样本

        Returns:
            dict:
                - scene:     torch.Tensor (4, H, W)   场景特征
                - signal:    torch.Tensor (3, H, W)   信号特征
                - imaging:   torch.Tensor (N, H, W)   成像特征
                - label:     int                       标签 0/1
                - domain:    str                       域名
                - mani_type: str                       操作类型
                - confidence: float                    置信度 ✅ 新增
                - path:      str                       原始图像路径
                - feature_path: str                    特征文件路径
        """
        if idx >= len(self.samples) or idx < 0:
            raise IndexError(f"索引 {idx} 超出范围 [0, {len(self.samples)})")

        sample = self.samples[idx]

        # === 安全提取字段 ===
        feature_path = sample.get('feature_path')
        image_path = sample.get('path')
        label = sample.get('label')
        domain = sample.get('domain')
        mani_type = sample.get('mani_type')
        confidence = sample.get('confidence', 0.5)  # ✅ 新增

        # 安全转换
        safe_feature_path = str(feature_path) if feature_path is not None else ""
        safe_image_path = str(image_path) if image_path is not None else ""
        safe_domain = str(domain) if domain is not None else "Unknown"
        safe_mani_type = str(mani_type) if mani_type is not None else "Unknown"

        if label is None or label not in [0, 1]:
            print(f"[WARN] 样本 {idx} label={label} 非法，强制设为 0")
            safe_label = 0
        else:
            safe_label = int(label)

        # === 加载特征 ===
        try:
            if not os.path.exists(safe_feature_path):
                raise FileNotFoundError(f"特征文件不存在: {safe_feature_path}")

            npz_data = np.load(safe_feature_path)
            scene_feat = torch.from_numpy(npz_data['scene'].astype(np.float32))
            signal_feat = torch.from_numpy(npz_data['signal'].astype(np.float32))
            imaging_feat = torch.from_numpy(npz_data['imaging'].astype(np.float32))

        except Exception as e:
            print(f"[ERROR] 特征加载失败 (idx={idx}, path={safe_feature_path}): {e}")
            # 使用零张量作为占位符
            scene_feat = torch.zeros(4, 512, 512, dtype=torch.float32)
            signal_feat = torch.zeros(3, 512, 512, dtype=torch.float32)
            imaging_feat = torch.zeros(32, 512, 512, dtype=torch.float32)

        # === 构建输出 ===
        output = {
            'scene': scene_feat,
            'signal': signal_feat,
            'imaging': imaging_feat,
            'label': safe_label,  # ✅ 改为直接返回 int
            'domain': safe_domain,
            'mani_type': safe_mani_type,
            'confidence': confidence,  # ✅ 新增
            'path': safe_image_path,
            'feature_path': safe_feature_path,
            'index': idx, 
        }

        return output

    def update_domains(self, new_domains):
        """动态更新目标域（用于课程学习）"""
        self.samples = self._apply_filters(self.full_samples, new_domains, None, None)
        self._build_confidence_index()  # ✅ 重新构建索引
        print(f"[域切换] 更新为 {new_domains}，当前样本数: {len(self.samples)}")

    def get_class_distribution(self):
        """获取类别分布"""
        return Counter(s['label'] for s in self.samples)

    # 新增：获取置信度分布
    def get_confidence_distribution(self):
        """
        获取置信度分布统计
        
        Returns:
            dict: 包含 min, max, mean, std, percentiles
        """
        confs = [s.get('confidence', 0.5) for s in self.samples]
        if not confs:
            return {}
        
        return {
            'min': min(confs),
            'max': max(confs),
            'mean': np.mean(confs),
            'std': np.std(confs),
            'percentiles': {
                p: np.percentile(confs, p)
                for p in [25, 50, 75, 90, 95, 99]
            }
        }

    def __str__(self):
        class_dist = self.get_class_distribution()
        class_info = ", ".join([f"Label {l}: {c}" for l, c in sorted(class_dist.items())])
        conf_dist = self.get_confidence_distribution()
        conf_info = f"Confidence: [{conf_dist.get('min', 0):.3f}, {conf_dist.get('max', 0):.3f}]"
        
        return (f"ForensicFeatureDataset from: {self.json_path}\n"
                f"总样本数: {len(self.samples)}\n"
                f"类别分布: {class_info}\n"
                f"{conf_info}")


# ======================== 数据加载器工厂 ========================
def create_dataloaders(train_json,
                       val_json,
                       batch_size=8,
                       num_workers=4,
                       pin_memory=True,
                       target_domains=None,
                       target_mani_types=None,
                       strict_mode=False):
    """
    创建训练和验证数据加载器（特征版本，支持课程学习）

    Args:
        train_json: 训练集 JSON 路径（含 feature_path 和 confidence）
        val_json: 验证集 JSON 路径
        batch_size: 批大小
        num_workers: 数据加载线程数
        pin_memory: 是否锁定内存
        target_domains: 筛选特定域
        target_mani_types: 筛选特定操作类型
        strict_mode: 严格模式

    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import DataLoader

    # 训练集
    train_dataset = ForensicFeatureDataset(
        json_path=train_json,
        is_train=True,
        target_domains=target_domains,
        target_mani_types=target_mani_types,
        strict_mode=strict_mode
    )

    # 验证集
    val_dataset = ForensicFeatureDataset(
        json_path=val_json,
        is_train=False,
        target_domains=target_domains,
        target_mani_types=target_mani_types,
        strict_mode=strict_mode
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    print(f"\n{'='*60}")
    print(f"数据加载器创建完成!")
    print(f"  训练集样本数: {len(train_dataset)}")
    print(f"  验证集样本数: {len(val_dataset)}")
    print(f"  训练批次数: {len(train_loader)}")
    print(f"  验证批次数: {len(val_loader)}")
    print(f"  批大小: {batch_size}")
    print(f"{'='*60}\n")

    return train_loader, val_loader