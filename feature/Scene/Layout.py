# Layout.py (No PPStructure version)
import cv2
import numpy as np
from paddleocr import PaddleOCR
from skimage.filters import gaussian
from typing import List, Dict

class LayoutAnomalyDetector:
    def __init__(self, input_size=512, lang='en'):
        self.input_size = input_size
        # 使用基础 PaddleOCR，不依赖 paddlex / PPStructure
        self.ocr_engine = PaddleOCR(
            use_angle_cls=False,
            lang=lang,
            use_gpu=False,
            show_log=False,
            det_db_box_thresh=0.3,
            det_db_unclip_ratio=1.6
        )

    def _extract_boxes(self, ocr_result) -> List[Dict]:
        """从 PaddleOCR 基础输出中提取文本框"""
        boxes = []
        if not ocr_result or ocr_result[0] is None:
            return boxes
        for item in ocr_result[0]:
            points, (text, conf) = item
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            boxes.append({
                'x1': x1, 'y1': y1,
                'x2': x2, 'y2': y2,
                'w': w, 'h': h,
                'text': text,
                'conf': conf
            })
        return boxes

    def _cluster_into_lines(self, boxes: List[Dict]) -> List[List[Dict]]:
        """按垂直位置聚类成文本行"""
        if len(boxes) == 0:
            return []
        # 按 y1 排序
        sorted_boxes = sorted(boxes, key=lambda b: b['y1'])
        lines = []
        current_line = [sorted_boxes[0]]
        avg_height = np.mean([b['h'] for b in sorted_boxes])
        threshold = avg_height * 0.7  # 行高容忍度

        for box in sorted_boxes[1:]:
            last_center_y = current_line[-1]['y1'] + current_line[-1]['h'] / 2
            curr_center_y = box['y1'] + box['h'] / 2
            if abs(curr_center_y - last_center_y) < threshold:
                current_line.append(box)
            else:
                lines.append(current_line)
                current_line = [box]
        if current_line:
            lines.append(current_line)
        return lines

    def _compute_anomaly_scores(self, boxes: List[Dict]) -> np.ndarray:
        """计算每个 box 的布局异常得分"""
        N = len(boxes)
        if N == 0:
            return np.array([])

        scores = np.zeros(N)
        lines = self._cluster_into_lines(boxes)
        if len(lines) < 2:
            return scores

        # 1. 行间距统计
        line_centers = [np.mean([b['y1'] + b['h']/2 for b in line]) for line in lines]
        spacings = np.diff(sorted(line_centers))
        mean_spacing = np.median(spacings) if len(spacings) > 0 else 20.0
        std_spacing = np.std(spacings) + 1e-5

        # 2. 段首缩进统计
        first_x_list = [min(b['x1'] for b in line) for line in lines]
        indent_mean = np.mean(first_x_list)
        indent_std = np.std(first_x_list) + 1e-5

        # 3. 字体大小（用高度 proxy）
        heights = [b['h'] for b in boxes]
        height_mean = np.median(heights)
        height_std = np.std(heights) + 1e-5

        # 构建 box 到 line 映射
        box_line_map = {}
        for li, line in enumerate(lines):
            for b in line:
                # 找索引（通过坐标匹配）
                for i, bb in enumerate(boxes):
                    if (bb['x1'] == b['x1'] and bb['y1'] == b['y1']):
                        box_line_map[i] = li
                        break

        # 打分
        for i, box in enumerate(boxes):
            score = 0.0
            li = box_line_map.get(i, -1)

            # 行间距异常（非首尾行）
            if 0 < li < len(lines) - 1:
                center_y = box['y1'] + box['h'] / 2
                spacing_prev = abs(center_y - line_centers[li - 1])
                spacing_next = abs(line_centers[li + 1] - center_y)
                z1 = abs(spacing_prev - mean_spacing) / std_spacing
                z2 = abs(spacing_next - mean_spacing) / std_spacing
                score += max(z1, z2)

            # 段首缩进异常（如果是行首）
            if li >= 0 and any(b['x1'] == box['x1'] for b in lines[li] if b['y1'] == box['y1']):
                # 简化：只要是最左即视为行首
                if box['x1'] == min(b['x1'] for b in lines[li]):
                    z_indent = abs(box['x1'] - indent_mean) / indent_std
                    score += z_indent

            # 字体大小异常
            z_font = abs(box['h'] - height_mean) / height_std
            score += z_font

            scores[i] = score

        return scores

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        输入: (512, 512, 3) RGB uint8
        输出: (512, 512) float32 热力图
        """
        H, W = image.shape[:2]
        assert H == W == self.input_size, "Input must be 512x512"

        # OCR
        ocr_result = self.ocr_engine.ocr(image, cls=False)
        boxes = self._extract_boxes(ocr_result)
        scores = self._compute_anomaly_scores(boxes)

        # 反投影
        heatmap = np.zeros((H, W), dtype=np.float32)
        for box, score in zip(boxes, scores):
            x1, y1 = int(round(box['x1'])), int(round(box['y1']))
            x2, y2 = int(round(box['x2'])), int(round(box['y2']))
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 > x1 and y2 > y1:
                heatmap[y1:y2, x1:x2] += score

        # 高斯扩散
        heatmap = gaussian(heatmap, sigma=5, preserve_range=True)
        return heatmap.astype(np.float32)
    
    def extract_layout_feature(self, image: np.ndarray) -> np.ndarray:
        """
        输入: (512, 512, 3) RGB uint8
        输出: (512, 512, 3) float32 —— [box_mask, confidence_map, height_map]
        """
        H, W = image.shape[:2]
        assert H == W == self.input_size

        ocr_result = self.ocr_engine.ocr(image, cls=False)
        boxes = self._extract_boxes(ocr_result)

        # 初始化特征图
        box_mask = np.zeros((H, W), dtype=np.float32)
        conf_map = np.zeros((H, W), dtype=np.float32)
        height_map = np.zeros((H, W), dtype=np.float32)

        for box in boxes:
            x1, y1 = int(round(box['x1'])), int(round(box['y1']))
            x2, y2 = int(round(box['x2'])), int(round(box['y2']))
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 > x1 and y2 > y1:
                box_mask[y1:y2, x1:x2] = 1.0
                conf_map[y1:y2, x1:x2] = box['conf']
                height_map[y1:y2, x1:x2] = box['h'] / 512.0  # 归一化高度

        features = np.stack([box_mask, conf_map, height_map], axis=-1)  # (512,512,3)
        return features.astype(np.float32)

# 全局单例，避免重复初始化 OCR
_GLOBAL_LAYOUT_DETECTOR = None

def _get_global_detector(input_size=512, lang='en'):
    global _GLOBAL_LAYOUT_DETECTOR
    if _GLOBAL_LAYOUT_DETECTOR is None:
        _GLOBAL_LAYOUT_DETECTOR = LayoutAnomalyDetector(input_size=input_size, lang=lang)
    return _GLOBAL_LAYOUT_DETECTOR

def extract_layout_feature(image_512: np.ndarray) -> np.ndarray:
    
    detector = _get_global_detector()
    return detector.extract_layout_feature(image_512)

# if __name__ == "__main__":
#     import matplotlib.pyplot as plt

#     # 加载图像
#     img_path = "/mnt/data3/public_datasets/OpenMMSec/3/e685278670eb41f1bb35f9f88510a1c1.jpg"  # 替换为你的路径
#     image = cv2.imread(img_path)
#     original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 保留原始图像用于保存和显示
#     image_resized = cv2.resize(original_image, (512, 512))  # 调整大小以适应模型输入要求

#     detector = LayoutAnomalyDetector(input_size=512, lang='en')  # 或 'ch' 中文
#     heatmap = detector(image_resized)

#     # 单独保存原图和热力图
#     plt.imsave("original_image.png", original_image)
#     plt.imsave("layout_anomaly_heatmap.png", heatmap, cmap='jet')

#     # 可视化 - 并排显示原图和热力图
#     fig, axes = plt.subplots(1, 2, figsize=(12, 6))

#     # 原图
#     axes[0].imshow(original_image)
#     axes[0].set_title("Original Image")
#     axes[0].axis('off')

#     # 热力图
#     im = axes[1].imshow(heatmap, cmap='jet')
#     axes[1].set_title("Layout Anomaly Heatmap")
#     axes[1].axis('off')
#     fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

#     plt.tight_layout()
#     plt.savefig("comparison_original_vs_heatmap.png", dpi=150)
#     plt.show()

#     print("✅ Original image saved to original_image.png")
#     print("✅ Heatmap saved to layout_anomaly_heatmap.png")
#     print("✅ Comparison image saved to comparison_original_vs_heatmap.png")
