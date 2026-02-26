import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path
from collections import Counter

# 使用 albumentations 进行数据增强
import albumentations as albu
from albumentations.pytorch import ToTensorV2


# ======================== 数据增强器 ========================
class ForensicTransform:
    """
    虚假图像检测专用数据增强器
    
    特性:
        - 训练时: 丰富的增强策略（翻转、旋转、颜色、压缩、模糊、噪声）
        - 测试时: 仅做尺寸调整
        - 支持同步变换图像和mask
        - 多种归一化策略
    """
    def __init__(self, 
                 output_size=(512, 512), 
                 norm_type='image_net',
                 is_train=True):
        """
        Args:
            output_size: 输出尺寸 (H, W)
            norm_type: 归一化类型 ('image_net' / 'standard' / 'none')
            is_train: 是否为训练模式
        """
        self.output_size = output_size
        self.norm_type = norm_type
        self.is_train = is_train
        
        # 构建变换流程
        self.transform = self._build_transform()
    
    def _build_transform(self):
        """构建完整的变换流程"""
        if self.is_train:
            common_transform = self._get_train_transform()
        else:
            common_transform = self._get_test_transform()
        
        post_transform = self._get_post_transform()
        
        # 合并变换
        return albu.Compose([
            *common_transform.transforms,
            *post_transform.transforms
        ], additional_targets={'mask': 'mask'})
    
    def _get_train_transform(self):
        """训练时的数据增强"""
        return albu.Compose([
            # 1. 尺寸调整
            albu.Resize(height=self.output_size[0], width=self.output_size[1]),
            
            # 2. 空间变换
            albu.HorizontalFlip(p=0.5),
            albu.Rotate(limit=15, p=0.5, border_mode=0),  # 旋转
            
            # 3. 颜色变换
            albu.RandomBrightnessContrast(
                brightness_limit=(-0.2, 0.2), 
                contrast_limit=0.2, 
                p=0.8
            ),
            albu.HueSaturationValue(
                hue_shift_limit=10, 
                sat_shift_limit=20, 
                val_shift_limit=10, 
                p=0.5
            ),
            
            # 4. 压缩伪影（模拟真实场景）
            albu.ImageCompression(quality_lower=80, quality_upper=100, p=0.3),
            
            # 5. 模糊
            albu.GaussianBlur(blur_limit=(3, 7), p=0.3),
            
            # 6. 噪声
            albu.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            
        ], additional_targets={'mask': 'mask'})
    
    def _get_test_transform(self):
        """测试时仅做尺寸调整"""
        return albu.Compose([
            albu.Resize(height=self.output_size[0], width=self.output_size[1]),
        ], additional_targets={'mask': 'mask'})
    
    def _get_post_transform(self):
        """后处理：归一化 + 转Tensor"""
        if self.norm_type == 'image_net':
            return albu.Compose([
                albu.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2(transpose_mask=True)
            ], additional_targets={'mask': 'mask'})
        
        elif self.norm_type == 'standard':
            return albu.Compose([
                albu.Normalize(
                    mean=[0.5, 0.5, 0.5], 
                    std=[0.5, 0.5, 0.5]
                ),
                ToTensorV2(transpose_mask=True)
            ], additional_targets={'mask': 'mask'})
        
        elif self.norm_type == 'none':
            return albu.Compose([
                albu.ToFloat(max_value=255.0),
                ToTensorV2(transpose_mask=True)
            ], additional_targets={'mask': 'mask'})
        
        else:
            raise ValueError(f"不支持的归一化类型: {self.norm_type}，"
                           f"请使用 'image_net', 'standard' 或 'none'")
    
    def __call__(self, image, mask=None):
        """
        应用变换
        
        Args:
            image: numpy array (H, W, 3), uint8
            mask: numpy array (H, W), 可选
        
        Returns:
            image: torch.Tensor (3, H, W)
            mask: torch.Tensor (1, H, W) 或 None
        """
        if mask is None:
            # 无mask时创建虚拟mask
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
            augmented = self.transform(image=image, mask=mask)
            return augmented['image'], None
        else:
            augmented = self.transform(image=image, mask=mask)
            return augmented['image'], augmented['mask']


class AdvancedForensicTransform(ForensicTransform):
    """
    高级数据增强器（可选更多增强策略）
    """
    def __init__(self, 
                 output_size=(512, 512), 
                 norm_type='image_net',
                 is_train=True,
                 use_heavy_augment=False):
        """
        Args:
            use_heavy_augment: 是否使用更强的增强（用于困难样本）
        """
        self.use_heavy_augment = use_heavy_augment
        super().__init__(output_size, norm_type, is_train)
    
    def _get_train_transform(self):
        """训练时的数据增强（增强版）"""
        base_transforms = [
            albu.Resize(height=self.output_size[0], width=self.output_size[1]),
            albu.HorizontalFlip(p=0.5),
            albu.Rotate(limit=15, p=0.5, border_mode=0),
            albu.RandomBrightnessContrast(
                brightness_limit=(-0.2, 0.2), 
                contrast_limit=0.2, 
                p=0.8
            ),
            albu.HueSaturationValue(
                hue_shift_limit=10, 
                sat_shift_limit=20, 
                val_shift_limit=10, 
                p=0.5
            ),
            albu.ImageCompression(quality_lower=80, quality_upper=100, p=0.3),
            albu.GaussianBlur(blur_limit=(3, 7), p=0.3),
            albu.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        ]
        
        if self.use_heavy_augment:
            # 额外增强策略
            heavy_transforms = [
                albu.RandomScale(scale_limit=0.2, p=0.3),
                albu.ShiftScaleRotate(
                    shift_limit=0.1, 
                    scale_limit=0.1, 
                    rotate_limit=20, 
                    p=0.5
                ),
                albu.OneOf([
                    albu.MotionBlur(blur_limit=7),
                    albu.MedianBlur(blur_limit=7),
                    albu.GaussianBlur(blur_limit=7),
                ], p=0.3),
                albu.OneOf([
                    albu.OpticalDistortion(distort_limit=0.3),
                    albu.GridDistortion(num_steps=5, distort_limit=0.3),
                ], p=0.2),
                albu.CoarseDropout(
                    max_holes=8, 
                    max_height=32, 
                    max_width=32, 
                    p=0.2
                ),
            ]
            base_transforms.extend(heavy_transforms)
        
        return albu.Compose(base_transforms, additional_targets={'mask': 'mask'})


# ======================== 数据验证器 ========================
class DataValidator:
    """
    JSON数据格式验证器（OpenMMSecV2格式）
    """
    @staticmethod
    def validate_json_format(json_path, strict_mode=False):
        """
        验证JSON文件格式并过滤无效样本
        """
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
        
        required_keys = ["path", "label"]
        
        valid_samples = []
        stats = {
            'total': len(data),
            'valid': 0,
            'missing_fields': 0,
            'missing_image': 0,
            'corrupted_image': 0,
            'invalid_label': 0,
            'bad_mask': 0
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
            
            image_path = sample["path"]
            if not os.path.exists(image_path):
                stats['missing_image'] += 1
                if strict_mode:
                    raise FileNotFoundError(f"图像文件不存在: {image_path}")
                continue
            
            # 验证图像完整性
            try:
                with Image.open(image_path) as img:
                    img.load()
                    if img.mode in ("RGBA", "LA", "P"):
                        _ = img.convert("RGB")
            except (OSError, ValueError, IOError) as e:
                stats['corrupted_image'] += 1
                if strict_mode:
                    raise ValueError(f"图像损坏: {image_path}")
                continue
            
            # 检查mask
            mask_path = sample.get("mask")
            if mask_path is not None:
                if not isinstance(mask_path, str):
                    stats['bad_mask'] += 1
                    sample["mask"] = None
                elif not os.path.exists(mask_path):
                    pass  # 不跳过样本
            
            valid_samples.append(sample)
            stats['valid'] += 1
        
        print(f"\n[验证统计]")
        print(f"  原始样本数: {stats['total']}")
        print(f"  有效样本数: {stats['valid']}")
        print(f"  缺少字段: {stats['missing_fields']}")
        print(f"  图像缺失: {stats['missing_image']}")
        print(f"  图像损坏: {stats['corrupted_image']}")
        print(f"  标签非法: {stats['invalid_label']}")
        
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
        has_mask = sum(1 for s in samples if s.get('mask') is not None)
        
        return {
            'total': total,
            'label_distribution': dict(label_counts),
            'domain_distribution': dict(domain_counts),
            'mani_type_distribution': dict(mani_type_counts),
            'samples_with_mask': has_mask,
            'mask_ratio': has_mask / total if total > 0 else 0
        }


# ======================== 数据集 ========================
class ForensicDataset(Dataset):
    """
    虚假图像检测数据集（OpenMMSecV2格式）
    
    特性:
        - 自动验证并过滤损坏样本
        - 使用 albumentations 进行高效增强
        - 支持多维度筛选
        - 可选mask加载
        - 动态域切换（课程学习）
    """
    def __init__(self, 
                 json_path, 
                 image_size=512,
                 norm_type='image_net',
                 is_train=True,
                 use_mask=False,
                 target_domains=None,
                 target_labels=None,
                 target_mani_types=None,
                 use_heavy_augment=False,
                 strict_mode=False):
        """
        Args:
            json_path: JSON配置文件路径
            image_size: 统一图像尺寸 (默认512)
            norm_type: 归一化类型 ('image_net' / 'standard' / 'none')
            is_train: 是否为训练模式
            use_mask: 是否加载mask
            target_domains: 筛选特定域
            target_labels: 筛选特定标签
            target_mani_types: 筛选特定操作类型
            use_heavy_augment: 是否使用更强的增强
            strict_mode: 严格验证模式
        """
        self.json_path = json_path
        self.image_size = image_size
        self.is_train = is_train
        self.use_mask = use_mask
        self.target_domains = target_domains
        
        # 初始化增强器
        self.transform = AdvancedForensicTransform(
            output_size=(image_size, image_size),
            norm_type=norm_type,
            is_train=is_train,
            use_heavy_augment=use_heavy_augment
        )
        
        # 验证JSON并加载样本
        print(f"\n{'='*60}")
        print(f"初始化数据集: {json_path}")
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
        
        print(f"\n配置信息:")
        print(f"  图像尺寸: {self.image_size}×{self.image_size}")
        print(f"  训练模式: {'是' if self.is_train else '否'}")
        print(f"  加载Mask: {'是' if self.use_mask else '否'}")
        print(f"  带Mask样本: {stats['samples_with_mask']} ({stats['mask_ratio']*100:.2f}%)")
        print(f"{'='*60}\n")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        sample = self.samples[idx]
        image_path = sample['path']
        label = sample['label']
        mask_path = sample.get('mask')
        
        # 加载图像
        try:
            image = Image.open(image_path).convert("RGB")
            image = np.array(image)  # (H, W, 3), uint8
        except Exception as e:
            raise RuntimeError(f"加载图像失败: {image_path}") from e
        
        # 加载mask
        if self.use_mask:
            if mask_path and os.path.exists(mask_path):
                try:
                    mask = Image.open(mask_path).convert("L")
                    mask = mask.resize((image.shape[1], image.shape[0]), Image.NEAREST)
                    mask = np.array(mask)
                    mask = (mask > 128).astype(np.float32)  # (H, W)
                except Exception as e:
                    print(f"[WARNING] Mask加载失败: {mask_path}")
                    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
            else:
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        else:
            mask = None
        
        # 应用变换
        image, mask = self.transform(image, mask)
        
        # 处理mask维度
        if mask is not None:
            if not torch.is_tensor(mask):
                mask = torch.from_numpy(mask).float()
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)  # (H, W) → (1, H, W)
        
        # 构建输出
        output = {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'domain': sample.get('domain', 'Unknown'),
            'mani_type': sample.get('mani_type', 'Unknown'),
            'path': image_path
        }
        
        if self.use_mask:
            output['mask'] = mask
        
        return output
    
    def update_domains(self, new_domains):
        """动态更新目标域（用于课程学习）"""
        self.target_domains = new_domains
        self.samples = self._apply_filters(self.full_samples, new_domains, None, None)
        print(f"[域切换] 更新为 {new_domains}，当前样本数: {len(self.samples)}")
    
    def get_class_distribution(self):
        """获取类别分布"""
        return Counter(s['label'] for s in self.samples)
    
    def __str__(self):
        class_dist = self.get_class_distribution()
        class_info = ", ".join([f"Label {l}: {c}" for l, c in sorted(class_dist.items())])
        return (f"ForensicDataset from: {self.json_path}\n"
                f"总样本数: {len(self.samples)}\n"
                f"类别分布: {class_info}")


# ======================== 数据加载器工厂 ========================
def create_dataloaders(train_json, 
                       val_json, 
                       batch_size=8,
                       num_workers=4,
                       image_size=512,
                       norm_type='image_net',
                       pin_memory=True,
                       target_domains=None,
                       target_mani_types=None,
                       use_mask=False,
                       use_heavy_augment=False,
                       strict_mode=False):
    """
    创建训练和验证数据加载器
    """
    from torch.utils.data import DataLoader
    
    # 训练集
    train_dataset = ForensicDataset(
        json_path=train_json,
        image_size=image_size,
        norm_type=norm_type,
        is_train=True,
        use_mask=use_mask,
        target_domains=target_domains,
        target_mani_types=target_mani_types,
        use_heavy_augment=use_heavy_augment,
        strict_mode=strict_mode
    )
    
    # 验证集
    val_dataset = ForensicDataset(
        json_path=val_json,
        image_size=image_size,
        norm_type=norm_type,
        is_train=False,  # 关闭增强
        use_mask=use_mask,
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
    print(f"  训练批次数: {len(train_loader)}")
    print(f"  验证批次数: {len(val_loader)}")
    print(f"{'='*60}\n")
    
    return train_loader, val_loader
