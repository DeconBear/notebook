# -*- coding: utf-8 -*-
"""
s12 目标检测 demo：YOLOv8 推理 + 从零实现 NMS 和 IoU
=====================================================
使用 Ultralytics YOLOv8 进行目标检测推理，
同时从零实现 IoU 计算和 NMS（非极大值抑制），
展示检测流程的每个步骤。

运行方式：python demo.py（从 s12_object_detection/code/ 目录运行）
依赖：torch, ultralytics, opencv-python, matplotlib, numpy
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.patches as patches
import os
import urllib.request

# 图片保存目录：固定为本章节的 images/ 目录（相对于本脚本的 ../images/）
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)
from typing import List, Tuple, Optional


# ============================================================
# 第 1 部分：IoU 计算 —— 从零实现
# ============================================================

def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    计算两个边界框的 IoU（交并比）

    边界框格式: [x1, y1, x2, y2]（左上角 + 右下角坐标）

    参数:
        box1: 第一个边界框，形状 (4,) → [x1, y1, x2, y2]
        box2: 第二个边界框，形状 (4,) → [x1, y1, x2, y2]

    返回:
        iou: 交并比值，范围 [0, 1]
             0 表示不重叠，1 表示完全重合
    """
    # ---------- 1. 计算交集区域 ----------
    # 交集的左上角 = 两个框左上角的最大值
    x1_inter = max(box1[0], box2[0])  # 交集区域的左边界 x
    y1_inter = max(box1[1], box2[1])  # 交集区域的上边界 y

    # 交集的右下角 = 两个框右下角的最小值
    x2_inter = min(box1[2], box2[2])  # 交集区域的右边界 x
    y2_inter = min(box1[3], box2[3])  # 交集区域的下边界 y

    # 交集宽度和高度（如果两个框不重叠，可能是负值）
    inter_width = max(0, x2_inter - x1_inter)   # 宽度，确保 >= 0
    inter_height = max(0, y2_inter - y1_inter)  # 高度，确保 >= 0
    inter_area = inter_width * inter_height      # 交集面积

    # ---------- 2. 计算并集区域 ----------
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])  # box1 面积
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])  # box2 面积
    union_area = area1 + area2 - inter_area             # 并集面积

    # ---------- 3. 计算 IoU ----------
    if union_area <= 0:
        return 0.0  # 避免除以零

    iou = inter_area / union_area
    return float(iou)


def compute_iou_batch(boxes: np.ndarray, query_box: np.ndarray) -> np.ndarray:
    """
    批量计算 query_box 与一组 boxes 之间的 IoU

    参数:
        boxes: 边界框数组，形状 (N, 4)，每行为 [x1, y1, x2, y2]
        query_box: 查询框，形状 (4,) → [x1, y1, x2, y2]

    返回:
        ious: IoU 数组，形状 (N,)，每个元素是 query_box 与对应 box 的 IoU
    """
    # ---------- 向量化计算 ----------
    # 交集左上角: (max(x1_q, x1_i), max(y1_q, y1_i))
    x1_inter = np.maximum(query_box[0], boxes[:, 0])  # 所有框的左边界最大值
    y1_inter = np.maximum(query_box[1], boxes[:, 1])  # 所有框的上边界最大值

    # 交集右下角: (min(x2_q, x2_i), min(y2_q, y2_i))
    x2_inter = np.minimum(query_box[2], boxes[:, 2])
    y2_inter = np.minimum(query_box[3], boxes[:, 3])

    # 交集面积
    inter_w = np.maximum(0, x2_inter - x1_inter)
    inter_h = np.maximum(0, y2_inter - y1_inter)
    inter_area = inter_w * inter_h

    # 各自面积
    area_query = (query_box[2] - query_box[0]) * (query_box[3] - query_box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # 并集面积
    union_area = area_query + area_boxes - inter_area

    # IoU，避免除以 0
    ious = np.divide(inter_area, union_area,
                     out=np.zeros_like(inter_area, dtype=float), where=union_area > 0)
    return ious


# ============================================================
# 第 2 部分：NMS —— 从零实现
# ============================================================

def nms(boxes: np.ndarray, scores: np.ndarray,
        iou_threshold: float = 0.5) -> np.ndarray:
    """
    非极大值抑制（Non-Maximum Suppression）

    从重叠的检测框中选出最佳的框，去除重复检测。

    参数:
        boxes: 边界框数组，形状 (N, 4)，每行格式 [x1, y1, x2, y2]
        scores: 置信度得分数组，形状 (N,)
        iou_threshold: IoU 阈值，高于此值的框被视为重复检测

    返回:
        keep_indices: 被保留的框的索引列表，按置信度降序排列

    算法步骤:
        1. 按置信度降序排序所有框
        2. 取出得分最高的框，加入保留列表
        3. 计算该框与剩余所有框的 IoU
        4. 移除所有 IoU > threshold 的框（它们被视为同一物体的重复检测）
        5. 重复步骤 2-4，直到没有剩余框
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    # ---------- 转换为 float 保证精度 ----------
    boxes = boxes.astype(np.float32)

    # ---------- 步骤 1: 按置信度降序排列 ----------
    order = scores.argsort()[::-1]  # 从大到小的索引

    keep = []  # 存放需要保留的框的索引

    while order.size > 0:
        # ---------- 步骤 2: 取置信度最高的框 ----------
        idx = order[0]
        keep.append(idx)

        if order.size == 1:
            break  # 只剩一个框了，直接结束

        # ---------- 步骤 3-4: 计算 IoU，移除重叠框 ----------
        # 当前最高分框
        current_box = boxes[idx]

        # 剩余框
        remaining_boxes = boxes[order[1:]]

        # 批量计算 IoU
        ious = compute_iou_batch(remaining_boxes, current_box)

        # 保留 IoU <= threshold 的框（非重复检测）
        remaining_indices = np.where(ious <= iou_threshold)[0]
        order = order[remaining_indices + 1]  # +1 因为 remaining_indices 是相对于 order[1:] 的

    return np.array(keep, dtype=np.int64)


# ============================================================
# 第 3 部分：YOLO 格式转换
# ============================================================

def xyxy_to_xywh(box: np.ndarray) -> np.ndarray:
    """
    将边界框格式从 [x1, y1, x2, y2] (左上+右下) 转换为 [x, y, w, h] (中心+宽高)

    参数:
        box: 形状 (4,) 或 (N, 4)，格式 [x1, y1, x2, y2]
    返回:
        形状与输入相同，格式 [cx, cy, w, h]
    """
    # 中心点 = (左上 + 右下) / 2
    cx = (box[..., 0] + box[..., 2]) / 2
    cy = (box[..., 1] + box[..., 3]) / 2
    # 宽高 = 右下 - 左上
    w = box[..., 2] - box[..., 0]
    h = box[..., 3] - box[..., 1]
    return np.stack([cx, cy, w, h], axis=-1)


def xywh_to_xyxy(box: np.ndarray) -> np.ndarray:
    """
    将边界框格式从 [cx, cy, w, h] (中心+宽高) 转换为 [x1, y1, x2, y2] (左上+右下)

    参数:
        box: 形状 (4,) 或 (N, 4)，格式 [cx, cy, w, h]
    返回:
        形状与输入相同，格式 [x1, y1, x2, y2]
    """
    # 左上角 = 中心 - 宽高/2
    x1 = box[..., 0] - box[..., 2] / 2
    y1 = box[..., 1] - box[..., 3] / 2
    # 右下角 = 中心 + 宽高/2
    x2 = box[..., 0] + box[..., 2] / 2
    y2 = box[..., 1] + box[..., 3] / 2
    return np.stack([x1, y1, x2, y2], axis=-1)


def yolo_output_to_pixel(boxes_xywh_norm: np.ndarray,
                         img_w: int, img_h: int) -> np.ndarray:
    """
    将 YOLO 格式的归一化坐标转换为像素坐标

    YOLO 输出格式: 每个框为 [cx_norm, cy_norm, w_norm, h_norm]
    其中所有值都被归一化到 [0, 1] 相对于图像宽高

    参数:
        boxes_xywh_norm: 归一化边界框，形状 (N, 4)，格式 [cx, cy, w, h] (归一化)
        img_w: 图像宽度（像素）
        img_h: 图像高度（像素）

    返回:
        boxes_xyxy: 像素坐标边界框，形状 (N, 4)，格式 [x1, y1, x2, y2]
    """
    # ---------- 将归一化的中心+宽高转为归一化的 xyxy ----------
    boxes_xyxy_norm = xywh_to_xyxy(boxes_xywh_norm)

    # ---------- 乘以图像尺寸得到像素坐标 ----------
    boxes_xyxy_pixel = boxes_xyxy_norm.copy()
    boxes_xyxy_pixel[:, [0, 2]] *= img_w  # x 坐标
    boxes_xyxy_pixel[:, [1, 3]] *= img_h  # y 坐标

    return boxes_xyxy_pixel


# ============================================================
# 第 4 部分：可视化工具
# ============================================================

# COCO 数据集的 80 个类别名称
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def generate_colors(n: int) -> List[Tuple[float, float, float]]:
    """
    生成 n 种不同的颜色用于不同类别的可视化

    参数:
        n: 需要的颜色数量
    返回:
        colors: RGB 颜色元组列表，每个元素 (r, g, b) 范围 [0, 1]
    """
    # 使用 HSV 色彩空间均匀采样，再转回 RGB
    colors = []
    for i in range(n):
        hue = i / n
        # HSV → RGB 简单转换
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors.append((r, g, b))
    return colors


def draw_detections(image: np.ndarray, boxes: np.ndarray,
                    scores: np.ndarray, class_ids: np.ndarray,
                    class_names: List[str] = None,
                    save_path: str = None, title: str = None):
    """
    在图像上绘制检测结果（边界框、类别标签、置信度）

    参数:
        image: 原始图像（RGB 格式，uint8）
        boxes: 边界框，形状 (N, 4)，格式 [x1, y1, x2, y2]
        scores: 置信度得分，形状 (N,)
        class_ids: 类别 ID，形状 (N,)
        class_names: 类别名称列表
        save_path: 保存路径
        title: 图片标题
    """
    if class_names is None:
        class_names = COCO_CLASSES

    colors = generate_colors(len(class_names))

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(image)

    for box, score, cls_id in zip(boxes, scores, class_ids):
        cls_id = int(cls_id)
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        color = colors[cls_id % len(colors)]

        # 绘制边界框
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2,
                                  edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        # 绘制标签
        cls_name = class_names[cls_id] if cls_id < len(class_names) else f"cls{cls_id}"
        label = f"{cls_name} {score:.2f}"

        # 标签背景
        ax.text(x1, y1 - 5, label, fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color,
                          alpha=0.7, edgecolor='none'),
                color='white', fontweight='bold')

    if title:
        ax.set_title(title, fontsize=14)
    ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [可视化] 检测结果已保存到 {save_path}")
    plt.close()


# ============================================================
# 第 5 部分：下载测试图像
# ============================================================

def download_test_images(save_dir: str) -> List[str]:
    """
    下载几张测试图像用于目标检测演示

    参数:
        save_dir: 图片保存目录
    返回:
        image_paths: 下载的图片路径列表
    """
    os.makedirs(save_dir, exist_ok=True)

    # 使用一些公开的测试图像 URL
    test_images = {
        "street.jpg": "https://ultralytics.com/images/bus.jpg",
        "zidane.jpg": "https://ultralytics.com/images/zidane.jpg",
    }

    paths = []
    for fname, url in test_images.items():
        fpath = os.path.join(save_dir, fname)
        if not os.path.exists(fpath):
            try:
                print(f"  下载 {fname}...")
                urllib.request.urlretrieve(url, fpath)
            except Exception as e:
                print(f"  下载失败 {fname}: {e}")
                continue
        if os.path.exists(fpath):
            paths.append(fpath)

    return paths


# ============================================================
# 第 6 部分：主流程
# ============================================================

def test_iou_calculation():
    """测试 IoU 计算函数"""
    print("\n--- IoU 计算测试 ---")

    # 测试用例 1: 两个完全重叠的框
    box1 = np.array([10, 10, 50, 50])
    box2 = np.array([10, 10, 50, 50])
    iou = compute_iou(box1, box2)
    print(f"  完全重叠: IoU = {iou:.4f} (期望 1.0)")

    # 测试用例 2: 完全不重叠
    box2 = np.array([60, 60, 100, 100])
    iou = compute_iou(box1, box2)
    print(f"  完全不重叠: IoU = {iou:.4f} (期望 0.0)")

    # 测试用例 3: 一半重叠
    box2 = np.array([30, 30, 70, 70])
    iou = compute_iou(box1, box2)
    # box1面积=40*40=1600, box2面积=40*40=1600, 交集=20*20=400
    # 并集=1600+1600-400=2800, IoU=400/2800≈0.143
    print(f"  部分重叠: IoU = {iou:.4f} (期望 ~0.143)")

    # 测试用例 4: 批量计算
    boxes = np.array([
        [10, 10, 50, 50],   # 与 query_box 完全相同 → IoU = 1.0
        [30, 30, 70, 70],   # 约 50% 重叠
        [60, 60, 100, 100], # 不重叠
        [15, 15, 45, 45],   # 包含在 query_box 内
    ])
    query = np.array([10, 10, 50, 50])
    ious = compute_iou_batch(boxes, query)
    print(f"  批量 IoU: {ious}")


def test_nms():
    """测试 NMS 算法"""
    print("\n--- NMS 测试 ---")

    # 模拟 5 个重叠的检测框
    boxes = np.array([
        [100, 100, 200, 200],  # 框 0: 高分
        [110, 110, 210, 210],  # 框 1: 高重叠
        [105, 105, 195, 195],  # 框 2: 高重叠
        [300, 100, 400, 200],  # 框 3: 不同位置
        [115, 115, 205, 205],  # 框 4: 高重叠
    ], dtype=np.float32)

    scores = np.array([0.95, 0.82, 0.76, 0.88, 0.61], dtype=np.float32)

    print(f"  输入: {len(boxes)} 个框, 置信度: {scores}")

    keep = nms(boxes, scores, iou_threshold=0.5)

    print(f"  NMS 后保留: {len(keep)} 个框")
    print(f"  保留的索引: {keep}")
    print(f"  保留的置信度: {scores[keep]}")

    # 期望：框0（0.95）被保留，框1、2、4因为高IoU被移除
    # 框3（0.88）位置不同，被保留
    # 结果: 框0 和 框3
    assert len(keep) == 2, f"期望保留 2 个框，实际 {len(keep)}"
    print("  NMS 测试通过 ✓")


def visualize_iou_examples(save_dir: str):
    """
    创建 IoU 计算的可视化示例图
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    examples = [
        ("IoU = 0.96 (Nearly Perfect)", [10, 10, 60, 60], [12, 12, 58, 58]),
        ("IoU = 0.55 (Moderate Overlap)", [10, 10, 60, 60], [35, 35, 85, 85]),
        ("IoU = 0.08 (Almost Disjoint)", [10, 10, 60, 60], [55, 55, 80, 80]),
    ]

    for ax, (title, box_a, box_b) in zip(axes, examples):
        # 绘制 box A
        x1, y1, x2, y2 = box_a
        rect_a = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2,
                                    edgecolor='green', facecolor='green',
                                    alpha=0.3, label='Ground Truth')
        ax.add_patch(rect_a)
        # 绘制 box B
        x1, y1, x2, y2 = box_b
        rect_b = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2,
                                    edgecolor='red', facecolor='red',
                                    alpha=0.3, linestyle='--', label='Prediction')
        ax.add_patch(rect_b)

        iou_val = compute_iou(np.array(box_a), np.array(box_b))
        ax.set_title(f"{title}\nActual IoU = {iou_val:.3f}", fontsize=11)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect('equal')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("IoU (Intersection over Union) Calculation Examples", fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(save_dir, "iou_examples.png")
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  [可视化] IoU 示例图已保存到 {save_path}")


def run_yolo_detection():
    """
    使用 YOLOv8 进行目标检测并可视化结果
    """
    print("\n--- YOLOv8 目标检测 ---")

    # GPU 自动检测
    import torch
    _YOLO_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"  使用设备: {_YOLO_DEVICE}")
    if _YOLO_DEVICE.type == 'cpu':
        print("  （未检测到 GPU，使用 CPU 进行 YOLO 推理，速度较慢）")

    # 检查是否安装了 ultralytics
    try:
        from ultralytics import YOLO
    except ImportError:
        print("  [跳过] 请安装 ultralytics: pip install ultralytics")
        print("  运行 'pip install ultralytics' 后重新执行本 demo 以查看完整检测效果")
        return

    # ---------- 下载测试图像 ----------
    image_dir = "../images/test_images"
    image_paths = download_test_images(image_dir)

    if not image_paths:
        print("  [警告] 没有可用的测试图像，跳过 YOLO 检测")
        return

    # ---------- 加载 YOLOv8n (nano 版本，最快) ----------
    print("  加载 YOLOv8n 模型...")
    try:
        model = YOLO("yolov8n.pt")  # 自动下载预训练权重
    except Exception as e:
        print(f"  [错误] YOLOv8n 模型下载失败 ({e})")
        print("  [回退] 跳过 YOLO 检测演示，请检查网络连接后重试")
        return

    # ---------- 对每张测试图像进行推理 ----------
    for img_path in image_paths:
        print(f"\n  检测图像: {os.path.basename(img_path)}")

        # YOLOv8 推理
        results = model(img_path, verbose=False)
        result = results[0]

        # 读取原始图像（BGR → RGB）
        import cv2
        image = cv2.imread(img_path)
        if image is None:
            print(f"    无法读取图像: {img_path}")
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # ---------- 提取检测结果 ----------
        if result.boxes is not None and len(result.boxes) > 0:
            # YOLOv8 的结果格式
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()  # 像素坐标 [x1,y1,x2,y2]
            scores = result.boxes.conf.cpu().numpy()       # 置信度
            class_ids = result.boxes.cls.cpu().numpy()     # 类别 ID

            print(f"    原始检测: {len(boxes_xyxy)} 个框")

            # ---------- 使用我们自己的 NMS ----------
            keep = nms(boxes_xyxy, scores, iou_threshold=0.5)
            boxes_nms = boxes_xyxy[keep]
            scores_nms = scores[keep]
            class_ids_nms = class_ids[keep]

            print(f"    NMS 后: {len(boxes_nms)} 个框")
            for i in range(len(boxes_nms)):
                cls_name = COCO_CLASSES[int(class_ids_nms[i])]
                print(f"      {cls_name}: {scores_nms[i]:.3f}")

            # ---------- 可视化原始检测 vs NMS 后 ----------
            fname = os.path.splitext(os.path.basename(img_path))[0]

            # 原始检测
            draw_detections(
                image_rgb.copy(), boxes_xyxy, scores, class_ids,
                save_path=os.path.join(_IMAGES_DIR, f"{fname}_detections.png"),
                title=f"Raw Detections ({len(boxes_xyxy)} boxes)"
            )

            # NMS 后
            draw_detections(
                image_rgb.copy(), boxes_nms, scores_nms, class_ids_nms,
                save_path=os.path.join(_IMAGES_DIR, f"{fname}_after_nms.png"),
                title=f"After NMS ({len(boxes_nms)} boxes, IoU Threshold=0.5)"
            )
        else:
            print("    未检测到目标")


def main():
    """主函数"""
    print("=" * 60)
    print("s12 目标检测 Demo")
    print("IoU 计算 + NMS 从零实现 + YOLOv8 推理演示")
    print("=" * 60)

    # ---------- 准备工作 ----------
    output_dir = _IMAGES_DIR

    # ---------- 1. 测试 IoU 计算 ----------
    print("\n[1/4] 测试 IoU 计算...")
    test_iou_calculation()

    # ---------- 2. 可视化 IoU 示例 ----------
    print("\n[2/4] 可视化 IoU 示例...")
    visualize_iou_examples(output_dir)

    # ---------- 3. 测试 NMS ----------
    print("\n[3/4] 测试 NMS 算法...")
    test_nms()

    # ---------- 4. YOLOv8 目标检测 ----------
    print("\n[4/4] YOLOv8 目标检测演示...")
    run_yolo_detection()

    print("\n" + "=" * 60)
    print(f"Demo 完成！查看 {_IMAGES_DIR} 目录下的可视化结果。")
    print("=" * 60)


if __name__ == "__main__":
    main()
