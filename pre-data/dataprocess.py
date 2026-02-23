import os
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path


class DataValidator:
    """
    JSON数据格式验证器
    """
    @staticmethod
    def validate_json_format(json_path):
        """
        验证JSON文件格式
        Args:
            json_path: JSON文件路径
        Returns:
            bool: 是否通过验证
            str: 错误信息（如果有）
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检查顶层结构
            if not isinstance(data, dict):
                return False, "JSON根节点必须是字典类型"
            
            if 'images' not in data:
                return False, "JSON缺少'images'字段"
            
            if not isinstance(data['images'], list):
                return False, "'images'字段必须是列表类型"
            
            # 检查每个图像条目
            for idx, item in enumerate(data['images']):
                if not isinstance(item, dict):
                    return False, f"第{idx}个图像条目必须是字典类型"
                
                # 必需字段检查
                if 'path' not in item:
                    return False, f"第{idx}个图像条目缺少'path'字段"
                
                if 'label' not in item:
                    return False, f"第{idx}个图像条目缺少'label'字段"
                
                # 标签值检查
                if item['label'] not in [0, 1]:
                    return False, f"第{idx}个图像的标签必须是0或1，当前为{item['label']}"
                
                # 路径存在性检查
                if not os.path.exists(item['path']):
                    print(f"警告: 第{idx}个图像路径不存在: {item['path']}")
            
            return True, "验证通过"
        
        except json.JSONDecodeError as e:
            return False, f"JSON格式错误: {str(e)}"
        except Exception as e:
            return False, f"验证过程出错: {str(e)}"
    
    @staticmethod
    def get_dataset_statistics(json_path):
        """
        获取数据集统计信息
        Args:
            json_path: JSON文件路径
        Returns:
            dict: 统计信息
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total = len(data['images'])
        real_count = sum(1 for item in data['images'] if item['label'] == 0)
        fake_count = sum(1 for item in data['images'] if item['label'] == 1)
        
        valid_paths = sum(1 for item in data['images'] if os.path.exists(item['path']))
        
        return {
            'total_images': total,
            'real_images': real_count,
            'fake_images': fake_count,
            'valid_paths': valid_paths,
            'invalid_paths': total - valid_paths,
            'real_ratio': real_count / total if total > 0 else 0,
            'fake_ratio': fake_count / total if total > 0 else 0
        }


class ForensicDataset(Dataset):
    """
    虚假图像检测数据集
    支持数据筛选、统一缩放、标准化处理
    """
    def __init__(self, 
                 json_path, 
                 image_size=512,
                 normalize=True,
                 augment=False,
                 filter_invalid=True,
                 filter_labels=None):
        """
        Args:
            json_path: JSON配置文件路径
            image_size: 统一图像尺寸 (默认512x512)
            normalize: 是否进行ImageNet标准化
            augment: 是否进行数据增强
            filter_invalid: 是否过滤无效路径
            filter_labels: 筛选特定标签 (None/0/1/[0,1])
        """
        self.json_path = json_path
        self.image_size = image_size
        self.normalize = normalize
        self.augment = augment
        
        # 验证JSON格式
        is_valid, message = DataValidator.validate_json_format(json_path)
        if not is_valid:
            raise ValueError(f"JSON验证失败: {message}")
        
        print(f"✓ JSON格式验证通过: {json_path}")
        
        # 加载数据
        self._load_data(filter_invalid, filter_labels)
        
        # 构建数据变换
        self._build_transforms()
        
        # 打印统计信息
        self._print_statistics()
    
    def _load_data(self, filter_invalid, filter_labels):
        """
        加载并筛选数据
        """
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.image_list = []
        self.labels = []
        
        invalid_count = 0
        filtered_count = 0
        
        for item in data['images']:
            img_path = item['path']
            label = item['label']
            
            # 筛选无效路径
            if filter_invalid and not os.path.exists(img_path):
                invalid_count += 1
                continue
            
            # 筛选特定标签
            if filter_labels is not None:
                if isinstance(filter_labels, list):
                    if label not in filter_labels:
                        filtered_count += 1
                        continue
                else:
                    if label != filter_labels:
                        filtered_count += 1
                        continue
            
            self.image_list.append(img_path)
            self.labels.append(label)
        
        if invalid_count > 0:
            print(f"  - 过滤了 {invalid_count} 个无效路径")
        if filtered_count > 0:
            print(f"  - 过滤了 {filtered_count} 个不符合标签要求的样本")
    
    def _build_transforms(self):
        """
        构建图像变换流程
        """
        transform_list = []
        
        # 1. 统一缩放到指定尺寸
        transform_list.append(transforms.Resize((self.image_size, self.image_size)))
        
        # 2. 数据增强（仅训练时）
        if self.augment:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            ])
        
        # 3. 转为Tensor
        transform_list.append(transforms.ToTensor())
        
        # 4. 标准化（使用ImageNet均值和标准差）
        if self.normalize:
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                )
            )
        
        self.transform = transforms.Compose(transform_list)
    
    def _print_statistics(self):
        """
        打印数据集统计信息
        """
        total = len(self.image_list)
        real_count = sum(1 for label in self.labels if label == 0)
        fake_count = sum(1 for label in self.labels if label == 1)
        
        print(f"\n数据集统计信息:")
        print(f"  - 总样本数: {total}")
        print(f"  - 真实图像: {real_count} ({real_count/total*100:.2f}%)")
        print(f"  - 伪造图像: {fake_count} ({fake_count/total*100:.2f}%)")
        print(f"  - 图像尺寸: {self.image_size}×{self.image_size}")
        print(f"  - 数据增强: {'是' if self.augment else '否'}")
        print(f"  - 标准化: {'是' if self.normalize else '否'}\n")
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        Returns:
            image: 预处理后的图像张量 [3, H, W]
            label: 标签 (0=真实, 1=伪造)
        """
        # 加载图像
        img_path = self.image_list[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"警告: 无法加载图像 {img_path}, 错误: {str(e)}")
            # 返回黑色图像作为fallback
            image = Image.new('RGB', (self.image_size, self.image_size), (0, 0, 0))
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        # 获取标签
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return image, label
    
    def get_sample_info(self, idx):
        """
        获取样本的详细信息（用于调试）
        """
        return {
            'index': idx,
            'path': self.image_list[idx],
            'label': self.labels[idx],
            'label_name': 'Real' if self.labels[idx] == 0 else 'Fake'
        }


def create_dataloaders(train_json, 
                       val_json, 
                       batch_size=8,
                       num_workers=4,
                       image_size=512,
                       pin_memory=True):
    """
    便捷函数：创建训练和验证数据加载器
    
    Args:
        train_json: 训练集JSON路径
        val_json: 验证集JSON路径
        batch_size: 批次大小
        num_workers: 数据加载线程数
        image_size: 图像尺寸
        pin_memory: 是否固定内存
    
    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
    """
    from torch.utils.data import DataLoader
    
    print("="*60)
    print("初始化训练集...")
    print("="*60)
    train_dataset = ForensicDataset(
        json_path=train_json,
        image_size=image_size,
        normalize=True,
        augment=True,  # 训练集开启数据增强
        filter_invalid=True
    )
    
    print("="*60)
    print("初始化验证集...")
    print("="*60)
    val_dataset = ForensicDataset(
        json_path=val_json,
        image_size=image_size,
        normalize=True,
        augment=False,  # 验证集关闭数据增强
        filter_invalid=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # 丢弃最后不完整的batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print("="*60)
    print("数据加载器创建完成!")
    print(f"  - 训练批次数: {len(train_loader)}")
    print(f"  - 验证批次数: {len(val_loader)}")
    print("="*60)
    
    return train_loader, val_loader


def validate_and_analyze_json(json_path):
    """
    独立工具：验证并分析JSON文件
    
    使用示例:
        python -c "from pre_data.dataprocess import validate_and_analyze_json; \
                   validate_and_analyze_json('data/train.json')"
    """
    print("\n" + "="*60)
    print(f"正在分析: {json_path}")
    print("="*60)
    
    # 验证格式
    is_valid, message = DataValidator.validate_json_format(json_path)
    
    if is_valid:
        print(f"✓ {message}")
        
        # 获取统计信息
        stats = DataValidator.get_dataset_statistics(json_path)
        
        print("\n统计信息:")
        print(f"  总图像数: {stats['total_images']}")
        print(f"  真实图像: {stats['real_images']} ({stats['real_ratio']*100:.2f}%)")
        print(f"  伪造图像: {stats['fake_images']} ({stats['fake_ratio']*100:.2f}%)")
        print(f"  有效路径: {stats['valid_paths']}")
        print(f"  无效路径: {stats['invalid_paths']}")
        
        if stats['invalid_paths'] > 0:
            print(f"\n⚠ 警告: 发现 {stats['invalid_paths']} 个无效路径!")
    else:
        print(f"✗ {message}")
    
    print("="*60 + "\n")
    
    return is_valid


# ==================== 测试代码 ====================
if __name__ == '__main__':
    # 测试示例
    
    # 1. 创建示例JSON文件
    sample_train_json = {
        "images": [
            {"path": "data/real/img1.jpg", "label": 0},
            {"path": "data/real/img2.jpg", "label": 0},
            {"path": "data/fake/img1.jpg", "label": 1},
            {"path": "data/fake/img2.jpg", "label": 1}
        ]
    }
    
    os.makedirs('data', exist_ok=True)
    with open('data/sample_train.json', 'w', encoding='utf-8') as f:
        json.dump(sample_train_json, f, indent=2, ensure_ascii=False)
    
    print("已创建示例JSON文件: data/sample_train.json\n")
    
    # 2. 验证JSON格式
    validate_and_analyze_json('data/sample_train.json')
    
    # 3. 测试数据集加载（如果有真实数据的话）
    # dataset = ForensicDataset(
    #     json_path='data/sample_train.json',
    #     image_size=512,
    #     normalize=True,
    #     augment=True
    # )
    # 
    # print(f"数据集长度: {len(dataset)}")
    # if len(dataset) > 0:
    #     img, label = dataset[0]
    #     print(f"图像shape: {img.shape}")
    #     print(f"标签: {label}")