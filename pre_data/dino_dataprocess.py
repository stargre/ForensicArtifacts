# pre_data/dino_dataprocess.py

import os
import json
from collections import Counter

import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from torchvision import transforms
from tqdm import tqdm


class DinoDataValidator:
    """
    针对原图 JSON 的验证器
    必需字段:
        - path
        - label
    可选字段:
        - domain / mani_type / ori_dataset
    """

    @staticmethod
    def validate_json_format(json_path, strict_mode=False):
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        # ── 第一步：读取 JSON ──────────────────────────────────────
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

        required_keys = ["label", "path"]

        valid_samples = []
        stats = {
            "total": len(data),
            "valid": 0,
            "missing_fields": 0,
            "missing_image": 0,
            "corrupted_image": 0,
            "invalid_label": 0,
        }

        # ── 第二步：逐样本验证，带进度条 ──────────────────────────
        pbar = tqdm(
            enumerate(data),
            total=len(data),
            desc="[数据验证] 扫描样本",
            unit="样本",
            dynamic_ncols=True,
        )

        for i, sample in pbar:
            if not isinstance(sample, dict):
                stats["missing_fields"] += 1
                pbar.set_postfix({
                    "有效": stats["valid"],
                    "缺字段": stats["missing_fields"],
                    "图像缺失": stats["missing_image"],
                })
                continue

            # 检查必需字段
            missing_keys = [k for k in required_keys if k not in sample]
            if missing_keys:
                stats["missing_fields"] += 1
                if strict_mode:
                    raise ValueError(f"样本 {i} 缺少必需字段: {missing_keys}")
                pbar.set_postfix({
                    "有效": stats["valid"],
                    "缺字段": stats["missing_fields"],
                    "图像缺失": stats["missing_image"],
                })
                continue

            # 检查标签
            label = sample["label"]
            if label not in [0, 1]:
                stats["invalid_label"] += 1
                if strict_mode:
                    raise ValueError(f"样本 {i} 标签必须是0或1")
                pbar.set_postfix({
                    "有效": stats["valid"],
                    "缺字段": stats["missing_fields"],
                    "图像缺失": stats["missing_image"],
                })
                continue

            # 检查图像路径是否存在
            image_path = sample["path"]
            if image_path is None or not os.path.exists(image_path):
                stats["missing_image"] += 1
                if strict_mode:
                    raise FileNotFoundError(f"图像不存在: {image_path}")
                pbar.set_postfix({
                    "有效": stats["valid"],
                    "缺字段": stats["missing_fields"],
                    "图像缺失": stats["missing_image"],
                })
                continue

            # 检查图像文件完整性
            try:
                with Image.open(image_path) as img:
                    img.verify()
            except Exception as e:
                stats["corrupted_image"] += 1
                if strict_mode:
                    raise ValueError(f"图像损坏或无法加载: {image_path}") from e
                pbar.set_postfix({
                    "有效": stats["valid"],
                    "缺字段": stats["missing_fields"],
                    "图像缺失": stats["missing_image"],
                    "图像损坏": stats["corrupted_image"],
                })
                continue

            valid_samples.append(sample)
            stats["valid"] += 1

            pbar.set_postfix({
                "有效": stats["valid"],
                "缺字段": stats["missing_fields"],
                "图像缺失": stats["missing_image"],
                "图像损坏": stats["corrupted_image"],
            })

        pbar.close()

        # ── 第三步：打印汇总统计 ───────────────────────────────────
        print(f"\n[验证完成]")
        print(f"  原始样本数: {stats['total']}")
        print(f"  有效样本数: {stats['valid']}")
        print(f"  缺少字段:   {stats['missing_fields']}")
        print(f"  图像缺失:   {stats['missing_image']}")
        print(f"  图像损坏:   {stats['corrupted_image']}")
        print(f"  标签非法:   {stats['invalid_label']}")

        if stats["valid"] == 0:
            raise ValueError("没有有效样本！")

        return valid_samples, stats

    @staticmethod
    def get_dataset_statistics(samples):
        total = len(samples)
        label_counts = Counter(s["label"] for s in samples)
        domain_counts = Counter(s.get("domain", "Unknown") for s in samples)
        mani_type_counts = Counter(
            s.get("mani_type", "Unknown")
            for s in samples if s["label"] == 1
        )
        return {
            "total": total,
            "label_distribution": dict(label_counts),
            "domain_distribution": dict(domain_counts),
            "mani_type_distribution": dict(mani_type_counts),
        }


class ForensicImageDataset(Dataset):
    """
    DINO baseline 用的原图数据集
    从 JSON 读取 path / label / domain 等字段
    """

    def __init__(
        self,
        json_path,
        image_size=224,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        is_train=True,
        target_domains=None,
        target_labels=None,
        target_mani_types=None,
        strict_mode=False,
    ):
        self.json_path = json_path
        self.is_train = is_train
        self.image_size = image_size

        print(f"\n{'='*70}")
        print(f" 初始化数据集: {os.path.basename(json_path)}")
        print(f" 模式: {'训练 (Train)' if is_train else '推理 (Inference)'}")
        print(f"{'='*70}")

        # 验证并加载样本
        self.full_samples, self.validation_stats = DinoDataValidator.validate_json_format(
            json_path, strict_mode=strict_mode
        )

        # 应用筛选（带进度条）
        self.samples = self._apply_filters(
            self.full_samples,
            target_domains,
            target_labels,
            target_mani_types
        )

        self.domain2id = {
            "AIGC": 0,
            "deepfake": 1,
            "IMDL": 2,
            "Doc": 3
        }

        self.transform = self._build_transform(image_size, mean, std, is_train)

    def _build_transform(self, image_size, mean, std, is_train):
        if is_train:
            return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])

    def _apply_filters(self, samples, domains, labels, mani_types):
        """
        应用筛选，带 tqdm 进度条
        """
        has_filter = any(x is not None for x in [domains, labels, mani_types])

        if not has_filter:
            return samples

        if domains is not None and isinstance(domains, str):
            domains = [domains]
        if labels is not None and isinstance(labels, int):
            labels = [labels]
        if mani_types is not None and isinstance(mani_types, str):
            mani_types = [mani_types]

        filtered = []
        pbar = tqdm(
            samples,
            total=len(samples),
            desc="[数据筛选] 过滤样本",
            unit="样本",
            dynamic_ncols=True,
        )

        for s in pbar:
            if domains is not None and s.get("domain") not in domains:
                continue
            if labels is not None and s["label"] not in labels:
                continue
            if mani_types is not None and s.get("mani_type") not in mani_types:
                continue
            filtered.append(s)
            pbar.set_postfix({"剩余": len(filtered)})

        pbar.close()

        if domains is not None:
            print(f"  [筛选] 域: {domains} → 剩余 {len(filtered)} 样本")
        if labels is not None:
            print(f"  [筛选] 标签: {labels} → 剩余 {len(filtered)} 样本")
        if mani_types is not None:
            print(f"  [筛选] 操作类型: {mani_types} → 剩余 {len(filtered)} 样本")

        if len(filtered) == 0:
            raise ValueError("筛选后无有效样本！")

        return filtered

    def get_detailed_statistics(self):
        """
        返回详细的数据集统计信息，供 print_dataset_summary 使用
        """
        total = len(self.samples)

        label_counts = Counter(s["label"] for s in self.samples)
        domain_counts = Counter(s.get("domain", "Unknown") for s in self.samples)

        # 每个 domain 下的真假分布
        domain_label_counts = {}
        for s in self.samples:
            domain = s.get("domain", "Unknown")
            label = s["label"]
            if domain not in domain_label_counts:
                domain_label_counts[domain] = {"real": 0, "fake": 0}
            if label == 0:
                domain_label_counts[domain]["real"] += 1
            else:
                domain_label_counts[domain]["fake"] += 1

        # fake 样本的 mani_type 分布
        mani_type_counts = Counter(
            s.get("mani_type", "Unknown")
            for s in self.samples if s["label"] == 1
        )

        return {
            "total": total,
            "label_counts": dict(label_counts),
            "domain_counts": dict(domain_counts),
            "domain_label_counts": domain_label_counts,
            "mani_type_counts": dict(mani_type_counts),
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image_path = str(sample.get("path", ""))
        label = int(sample.get("label", 0))
        domain = str(sample.get("domain", "Unknown"))
        mani_type = str(sample.get("mani_type", "Unknown"))
        ori_dataset = str(sample.get("ori_dataset", "Unknown"))
        real_source = str(sample.get("real_source", "Unknown"))

        if domain not in self.domain2id:
            raise ValueError(f"未知 domain: {domain}，请检查 domain2id 映射")
        domain_label = self.domain2id[domain]

        try:
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image)
        except Exception as e:
            print(f"[ERROR] 图像加载失败 idx={idx}, path={image_path}, err={e}")
            image = torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32)

        return {
            "image": image,
            "label": label,
            "domain": domain,
            "domain_label": domain_label,
            "mani_type": mani_type,
            "ori_dataset": ori_dataset,
            "real_source": real_source,
            "path": image_path,
            "index": idx,
        }

    # ===== 新增：给 curriculum manager 用 =====
    def get_domain(self, idx):
        """
        返回指定样本的 domain 字符串
        """
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"索引 {idx} 超出范围 [0, {len(self.samples)})")
        return str(self.samples[idx].get("domain", "Unknown"))

    def get_label(self, idx):
        """
        返回指定样本的标签（0/1）
        """
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"索引 {idx} 超出范围 [0, {len(self.samples)})")
        return int(self.samples[idx].get("label", 0))

    def get_class_distribution(self):
        return Counter(s["label"] for s in self.samples)

    def __str__(self):
        class_dist = self.get_class_distribution()
        class_info = ", ".join([f"Label {l}: {c}" for l, c in sorted(class_dist.items())])
        return (
            f"ForensicImageDataset | {os.path.basename(self.json_path)} | "
            f"总样本数: {len(self.samples)} | {class_info}"
        )


def print_dataset_summary(
    dataset,
    dataloader=None,
    name="Dataset",
    max_domains=20,
    max_mani_types=20,
    show_examples=3
):
    """
    打印数据集完整摘要（在 train/test 中调用）

    Args:
        dataset:      ForensicImageDataset
        dataloader:   可选，提供则打印 batch_size 和 batch 数
        name:         数据集名称，如 Train / Validation / Test
        max_domains:  最多显示几个 domain
        max_mani_types: 最多显示几个 mani_type
        show_examples:  显示几个样本路径
    """
    stats = dataset.get_detailed_statistics()

    total = stats["total"]
    label_counts = stats["label_counts"]
    domain_counts = stats["domain_counts"]
    domain_label_counts = stats["domain_label_counts"]
    mani_type_counts = stats["mani_type_counts"]

    num_real = label_counts.get(0, 0)
    num_fake = label_counts.get(1, 0)

    print("\n" + "=" * 80)
    print(f"  {name} Dataset Summary".center(80))
    print("=" * 80)

    # ── 基本信息 ──────────────────────────────────────────────────
    print(f"  {'总样本数':<16}: {total}")
    print(f"  {'Real (label=0)':<16}: {num_real}  ({num_real / total * 100:.1f}%)")
    print(f"  {'Fake (label=1)':<16}: {num_fake}  ({num_fake / total * 100:.1f}%)")
    if dataloader is not None:
        try:
            print(f"  {'Batch size':<16}: {dataloader.batch_size}")
            print(f"  {'Batch 数':<16}: {len(dataloader)}")
        except Exception:
            pass

    # ── domain 分布 ───────────────────────────────────────────────
    print(f"\n  [Domain 分布]")
    print(f"  {'Domain':<18} {'Total':>7} {'Real':>7} {'Fake':>7}  {'占比':>7}")
    print(f"  {'-'*52}")

    sorted_domains = sorted(domain_counts.items(), key=lambda x: -x[1])[:max_domains]
    for domain, count in sorted_domains:
        real_c = domain_label_counts.get(domain, {}).get("real", 0)
        fake_c = domain_label_counts.get(domain, {}).get("fake", 0)
        ratio = count / total * 100
        print(f"  {domain:<18} {count:>7} {real_c:>7} {fake_c:>7}  {ratio:>6.1f}%")

    if len(domain_counts) > max_domains:
        print(f"  ... 其余 {len(domain_counts) - max_domains} 个 domain 未显示")

    # ── mani_type 分布 ────────────────────────────────────────────
    if len(mani_type_counts) > 0:
        print(f"\n  [Fake 样本 mani_type 分布]")
        print(f"  {'mani_type':<28} {'数量':>7}")
        print(f"  {'-'*38}")
        sorted_mani = sorted(mani_type_counts.items(), key=lambda x: -x[1])[:max_mani_types]
        for mani_type, count in sorted_mani:
            print(f"  {str(mani_type):<28} {count:>7}")
        if len(mani_type_counts) > max_mani_types:
            print(f"  ... 其余 {len(mani_type_counts) - max_mani_types} 个 mani_type 未显示")

    # ── 样本示例 ──────────────────────────────────────────────────
    n_examples = min(show_examples, len(dataset.samples))
    if n_examples > 0:
        print(f"\n  [样本示例]")
        for i in range(n_examples):
            s = dataset.samples[i]
            path_short = s.get("path", "")
            if len(path_short) > 60:
                path_short = "..." + path_short[-57:]
            print(
                f"  [{i}] label={s.get('label')} | "
                f"domain={str(s.get('domain', 'Unknown')):<12} | "
                f"mani={str(s.get('mani_type', 'None')):<18} | "
                f"{path_short}"
            )

    print("=" * 80 + "\n")