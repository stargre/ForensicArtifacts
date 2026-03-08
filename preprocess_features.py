import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import sys
import torch
import cv2

# ==================== 硬编码配置 ====================
TRAIN_JSON = "/mnt/data3/zhiyu/Data/small_openmmsec/train_medium.json"
VAL_JSON = "/mnt/data3/zhiyu/Data/small_openmmsec/test_medium.json"

TRAIN_OUTPUT_DIR = "/mnt/data3/zhiyu/Data/small_openmmsec/features/train"
VAL_OUTPUT_DIR = "/mnt/data3/zhiyu/Data/small_openmmsec/features/test"

MIDAS_ROOT = "/mnt/data3/zhiyu/MiDaS"
MIDAS_WEIGHT = "/mnt/data3/zhiyu/MiDaS/weights/dpt_beit_large_384.pt"

# 选择处理模式：'train' / 'val' / 'both'
PROCESS_MODE = "val"
# ====================================================================

# ==================== 预加载所有模型（只加载一次）====================
print("="*60)
print("🔧 预加载模型...")
print("="*60)

# 1. 预加载 DINOv2 模型
from feature.Scene.Semantic_Illusion import _get_global_checker
dino_checker = _get_global_checker(device="cuda")
print("✅ DINOv2 已加载")

# 2. 预加载 MiDaS 模型
sys.path.insert(0, MIDAS_ROOT)
from midas.model_loader import load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas_model, midas_transform, _, _ = load_model(
    device,
    MIDAS_WEIGHT,
    "dpt_beit_large_384",
    optimize=False,
    height=None,
    square=True
)
midas_model.eval()
print(" MiDaS 已加载")

# 3. 预加载 OCR 模型
from feature.Scene.Layout import _get_global_detector
layout_detector = _get_global_detector()
print(" PaddleOCR 已加载")

print("="*60)
print(" 所有模型加载完成！开始特征提取...\n")

# ==================== 特征提取函数（使用预加载的模型）====================

def extract_scene_features(img_np, dino_checker, midas_model, midas_transform, layout_detector):
    """提取场景特征（使用预加载的模型）"""
    # 1. Semantic (使用预加载的 DINOv2)
    M_align_full = dino_checker.extract_semantic_feature(img_np)
    M_align = np.mean(M_align_full, axis=2)
    
    # 2. Geometric (使用预加载的 MiDaS)
    image_pil = Image.fromarray(img_np)
    image_rgb = np.array(image_pil)
    transformed = midas_transform({"image": image_rgb})
    input_tensor = transformed["image"]
    input_batch = torch.from_numpy(input_tensor).to(device).unsqueeze(0)
    
    with torch.no_grad():
        depth_pred = midas_model(input_batch)
        depth_pred = torch.nn.functional.interpolate(
            depth_pred.unsqueeze(1),
            size=(512, 512),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0).squeeze(0)
    
    depth = depth_pred.cpu().numpy().astype(np.float32)
    dz_dx, dz_dy = np.gradient(depth)
    normal = np.stack([-dz_dx, -dz_dy, np.ones_like(depth)], axis=-1)
    norm = np.linalg.norm(normal, axis=2, keepdims=True)
    normal = normal / (norm + 1e-8)
    M_depth = np.linalg.norm(normal, axis=2)
    
    # 3. Lighting (无需模型)
    from feature.Scene.Lighting_shadow_anomaly import extract_lighting_feature
    M_light_full = extract_lighting_feature(img_np)
    M_light = M_light_full[:, :, 0]
    
    # 4. Layout (使用预加载的 OCR)
    M_layout_full = layout_detector.extract_layout_feature(img_np)
    M_layout = M_layout_full[:, :, 0]
    
    return np.stack([M_align, M_depth, M_light, M_layout], axis=0)


def extract_signal_features(img_np):
    """提取信号特征（无需预加载模型）"""
    from feature.Signal.Local_Spectral import extract_spectral_feature
    from feature.Signal.Laplacian import extract_resampling_feature
    from feature.Signal.JPEG import extract_jpeg_feature
    
    M_spec = extract_spectral_feature(img_np)
    M_resamp = extract_resampling_feature(img_np)
    M_jpeg = extract_jpeg_feature(img_np)
    
    return np.stack([M_spec, M_resamp, M_jpeg], axis=0)


def extract_imaging_features(img_np):
    """提取成像特征（无需预加载模型）"""
    from feature.Imaging.prnu_feature import extract_prnu_feature
    from feature.Imaging.SRM_feature import extract_srm_feature
    from feature.Imaging.CFA_feature import extract_cfa_feature
    
    prnu_map = extract_prnu_feature(img_np)
    srm_map = extract_srm_feature(img_np)
    cfa_map = extract_cfa_feature(img_np)
    
    prnu_map = np.expand_dims(prnu_map, axis=0)
    cfa_map = np.expand_dims(cfa_map, axis=0)
    
    return np.concatenate([prnu_map, srm_map, cfa_map], axis=0)


def preprocess_single_image(image_path, output_dir):
    """处理单张图片（使用预加载的模型）"""
    # 加载图像
    image = Image.open(image_path).convert("RGB")
    image_512 = image.resize((512, 512), Image.BICUBIC)
    img_np = np.array(image_512)
    
    # 提取三个分支特征
    scene_features = extract_scene_features(
        img_np, dino_checker, midas_model, midas_transform, layout_detector
    )
    signal_features = extract_signal_features(img_np)
    imaging_features = extract_imaging_features(img_np)
    
    # 保存
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    feature_path = os.path.join(output_dir, f"{base_name}.npz")
    
    np.savez_compressed(
        feature_path,
        scene=scene_features.astype(np.float32),
        signal=signal_features.astype(np.float32),
        imaging=imaging_features.astype(np.float32)
    )
    
    return feature_path


def preprocess_dataset(json_path, output_dir):
    """批量处理数据集"""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(json_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    print(f"🚀 开始处理 {len(samples)} 个样本...\n")
    
    new_samples = []
    skipped = 0
    
    for sample in tqdm(samples, desc="提取特征"):
        image_path = sample['path']
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        feature_path = os.path.join(output_dir, f"{base_name}.npz")
        
        # 断点续传：跳过已存在的特征
        if os.path.exists(feature_path):
            new_sample = sample.copy()
            new_sample['feature_path'] = feature_path
            new_samples.append(new_sample)
            skipped += 1
            continue
        
        if not os.path.exists(image_path):
            print(f"[WARNING] 图像不存在: {image_path}")
            continue
        
        try:
            feature_path = preprocess_single_image(image_path, output_dir)
            
            new_sample = sample.copy()
            new_sample['feature_path'] = feature_path
            new_samples.append(new_sample)
            
        except Exception as e:
            print(f"[ERROR] {image_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存新 JSON
    output_json = json_path.replace('.json', '_with_features.json')
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(new_samples, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 特征提取完成!")
    print(f"  成功: {len(new_samples)}/{len(samples)}")
    print(f"  跳过（已存在）: {skipped}")
    print(f"  特征目录: {output_dir}")
    print(f"  新JSON: {output_json}")


# ==================== 主程序 ====================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("📋 配置信息")
    print("="*60)
    print(f"处理模式: {PROCESS_MODE}")
    print(f"训练集 JSON: {TRAIN_JSON}")
    print(f"训练集输出: {TRAIN_OUTPUT_DIR}")
    print(f"验证集 JSON: {VAL_JSON}")
    print(f"验证集输出: {VAL_OUTPUT_DIR}")
    print("="*60 + "\n")
    
    if PROCESS_MODE == "train":
        print("🔄 处理训练集...")
        preprocess_dataset(TRAIN_JSON, TRAIN_OUTPUT_DIR)
    
    elif PROCESS_MODE == "val":
        print("🔄 处理验证集...")
        preprocess_dataset(VAL_JSON, VAL_OUTPUT_DIR)
    
    elif PROCESS_MODE == "both":
        print("🔄 处理训练集...")
        preprocess_dataset(TRAIN_JSON, TRAIN_OUTPUT_DIR)
        
        print("\n🔄 处理验证集...")
        preprocess_dataset(VAL_JSON, VAL_OUTPUT_DIR)
    
    else:
        raise ValueError(f"无效的 PROCESS_MODE: {PROCESS_MODE}，请使用 'train', 'val' 或 'both'")
    
    print("\n" + "="*60)
    print("🎉 所有任务完成！")
    print("="*60)