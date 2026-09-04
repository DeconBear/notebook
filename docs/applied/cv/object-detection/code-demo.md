---
title: "s12 目标检测 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# s12 目标检测 — demo.py 代码详解

<a href="/notebook/code/applied/cv/object-detection/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/applied/cv/object-detection/code
python demo.py
```

## 代码逐段详解

### 第1步：导入库 —— 每个库是做什么的

```python
import numpy as np                # 向量化数值计算（IoU、NMS 的核心）
import matplotlib.pyplot as plt  # 可视化边界框、检测结果
import matplotlib.patches as patches  # 绘制矩形框
import os, urllib.request         # 文件路径管理 + 下载测试图片
from typing import List, Tuple, Optional  # 类型标注，提高代码可读性
```

> **为什么 IoU 和 NMS 用 NumPy 而不是 PyTorch？** 这两个算法不涉及梯度计算，NumPy 的纯 CPU 计算已经足够快。而且目标检测的后处理通常在将 GPU tensor 转为 numpy 后进行，直接使用 NumPy 避免了不必要的 GPU-CPU 转换。

### 第2步：IoU 计算 —— 目标检测最基础的度量

IoU（Intersection over Union，交并比）是衡量两个边界框重叠程度的指标。它贯穿目标检测的三个阶段：**训练**（判断锚框是否包含目标）、**评估**（判断检测是否正确）、**后处理**（NMS 中去除冗余框）。

**数学定义**：

$$
\text{IoU}(A, B) = \frac{\text{area}(A \cap B)}{\text{area}(A \cup B)} = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}
$$

**代码实现 —— 逐行对应数学公式**：

```python
def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    # 边界框格式: [x1, y1, x2, y2]（左上角 + 右下角）

    # 1. 计算交集区域
    #    交集矩形的左上角 = 两个框左上角的最大值
    x1_inter = max(box1[0], box2[0])  # max(左边界1, 左边界2)
    y1_inter = max(box1[1], box2[1])  # max(上边界1, 上边界2)
    #    交集矩形的右下角 = 两个框右下角的最小值
    x2_inter = min(box1[2], box2[2])  # min(右边界1, 右边界2)
    y2_inter = min(box1[3], box2[3])  # min(下边界1, 下边界2)

    #    交集宽度和高度 — max(0, ...) 确保不重叠时面积为 0
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height   # |A ∩ B|

    # 2. 计算各自面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])  # |A|
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])  # |B|

    # 3. 计算并集面积和 IoU
    union_area = area1 + area2 - inter_area  # |A ∪ B| = |A| + |B| - |A ∩ B|
    if union_area <= 0:
        return 0.0

    iou = inter_area / union_area  # |A ∩ B| / |A ∪ B|
    return float(iou)
```

**关于 `max(0, ...)` 的重要性**：如果两个框不重叠，`x2_inter - x1_inter` 会是负值（因为一个框的 min 右边界 小于另一个框的 max 左边界）。`max(0, ...)` 确保不重叠时交面积为 0，IoU = 0。

**批量 IoU 计算 —— 向量化加速**：

```python
def compute_iou_batch(boxes: np.ndarray, query_box: np.ndarray) -> np.ndarray:
    # 用 np.maximum 和 np.minimum 替代循环，一次处理所有框
    x1_inter = np.maximum(query_box[0], boxes[:, 0])  # 所有框的左边界
    y1_inter = np.maximum(query_box[1], boxes[:, 1])  # 所有框的上边界
    x2_inter = np.minimum(query_box[2], boxes[:, 2])
    y2_inter = np.minimum(query_box[3], boxes[:, 3])

    inter_w = np.maximum(0, x2_inter - x1_inter)
    inter_h = np.maximum(0, y2_inter - y1_inter)
    inter_area = inter_w * inter_h

    area_query = (query_box[2] - query_box[0]) * (query_box[3] - query_box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = area_query + area_boxes - inter_area

    # np.divide with where 参数：只在 union_area > 0 处做除法，否则填 0
    ious = np.divide(inter_area, union_area,
                     out=np.zeros_like(inter_area), where=union_area > 0)
    return ious
```

**为什么需要批量版本？** NMS 算法中，每一步都需要计算当前最高分框与**所有剩余框**的 IoU。如果写成 Python 循环，对于一个有 200 个检测框的图像，需要做 $200 \times 199 / 2 \approx 20,000$ 次 Python 级别的函数调用。向量化后用 NumPy 广播一次完成，速度快数十倍。

### 第3步：NMS（非极大值抑制）—— 从零实现

NMS 是目标检测中最关键的后处理步骤。同一个物体可能被多个重叠的边界框检测到，NMS 负责去除冗余。

**算法步骤**：

```
Input:  B = {b1, ..., bN} (边界框), S = {s1, ..., sN} (置信度), Nt (IoU阈值)
Output: D (保留的框)
1. D ← {}
2. While B is not empty:
    a. m ← argmax(S)                     # 选置信度最高的框
    b. M ← bm; D ← D ∪ {M}; B ← B \ {M}  # 保留并从列表中移除
    c. For each bi in B:
        if IoU(M, bi) ≥ Nt:              # 如果与 M 重叠太多
            B ← B \ {bi}                 # 移除该框（视为重复检测）
3. Return D
```

**代码实现**：

```python
def nms(boxes, scores, iou_threshold=0.5):
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    boxes = boxes.astype(np.float32)

    # 步骤 1: 按置信度降序排列
    order = scores.argsort()[::-1]  # 从大到小的索引
    keep = []

    while order.size > 0:
        # 步骤 2a: 取置信度最高的框
        idx = order[0]
        keep.append(idx)

        if order.size == 1:
            break

        # 步骤 2c: 计算当前框与剩余所有框的 IoU
        current_box = boxes[idx]
        remaining_boxes = boxes[order[1:]]
        ious = compute_iou_batch(remaining_boxes, current_box)

        # 保留 IoU ≤ threshold 的框（非重复检测）
        remaining_indices = np.where(ious <= iou_threshold)[0]
        order = order[remaining_indices + 1]  # +1 因为 remaining_indices 相对于 order[1:]

    return np.array(keep, dtype=np.int64)
```

**IoU 阈值的选择**：`iou_threshold=0.5` 是标准选择。
- **太大**（如 0.9）：几乎不移除框，检测结果充满冗余
- **太小**（如 0.1）：可能误删拥挤场景中相邻的不同物体（如人群中两个人）

> 在 COCO 评测中，mAP@0.5 使用 IoU=0.5 作为正样本阈值，mAP@0.5:0.95 在多个阈值下平均——后者更严格，需要更精确的框定位。

### 第4步：边界框格式转换

目标检测中有两种常见的边界框表示格式，理解它们的转换很关键：

| 格式 | 表示 | 使用场景 |
|------|------|---------|
| `[x1, y1, x2, y2]` | 左上角 + 右下角 | IoU 计算、NMS、绘图 |
| `[cx, cy, w, h]` | 中心 + 宽高 | YOLO 输出、锚框偏移计算 |

**转换公式**：

$$ 
\text{cx} = \frac{x_1 + x_2}{2}, \quad \text{cy} = \frac{y_1 + y_2}{2}
$$

$$
\text{w} = x_2 - x_1, \quad \text{h} = y_2 - y_1 
$$

```python
def xyxy_to_xywh(box):
    cx = (box[..., 0] + box[..., 2]) / 2
    cy = (box[..., 1] + box[..., 3]) / 2
    w = box[..., 2] - box[..., 0]
    h = box[..., 3] - box[..., 1]
    return np.stack([cx, cy, w, h], axis=-1)

def xywh_to_xyxy(box):
    x1 = box[..., 0] - box[..., 2] / 2
    y1 = box[..., 1] - box[..., 3] / 2
    x2 = box[..., 0] + box[..., 2] / 2
    y2 = box[..., 1] + box[..., 3] / 2
    return np.stack([x1, y1, x2, y2], axis=-1)
```

**YOLO 输出格式的特殊之处**：YOLO 的框坐标是**归一化**的：
- $(x, y)$ 是中心相对于**网格单元**的偏移，归一化到 $[0, 1]$
- $(w, h)$ 是宽高相对于**整张图像**的比例，归一化到 $[0, 1]$

转换为像素坐标时需要知道图像的实际尺寸：

```python
def yolo_output_to_pixel(boxes_xywh_norm, img_w, img_h):
    # 先转成归一化的 xyxy，再乘以图像尺寸
    boxes_xyxy_norm = xywh_to_xyxy(boxes_xywh_norm)
    boxes_xyxy_pixel = boxes_xyxy_norm.copy()
    boxes_xyxy_pixel[:, [0, 2]] *= img_w  # x 坐标乘宽度
    boxes_xyxy_pixel[:, [1, 3]] *= img_h  # y 坐标乘高度
    return boxes_xyxy_pixel
```

### 第5步：YOLOv8 推理 —— 使用预训练模型

```python
def run_yolo_detection():
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")  # YOLOv8 nano（最小最快，约 3.2M 参数）

    for img_path in image_paths:
        results = model(img_path, verbose=False)
        result = results[0]

        # 提取检测结果
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()   # 像素坐标
        scores = result.boxes.conf.cpu().numpy()        # 置信度
        class_ids = result.boxes.cls.cpu().numpy()      # 类别 ID

        # 使用我们自己的 NMS 进行后处理
        keep = nms(boxes_xyxy, scores, iou_threshold=0.5)
        boxes_nms = boxes_xyxy[keep]
        scores_nms = scores[keep]
        class_ids_nms = class_ids[keep]
```

**为什么 YOLOv8n？** YOLOv8 有五个规模：nano(n)、small(s)、medium(m)、large(l)、xlarge(x)。Nano 版本约 3.2M 参数，CPU 上也能运行，适合演示。实际应用中可根据精度-速度需求选择。

**YOLOv8 的内部改进**（相对于 YOLOv1 在 index.md 中的描述）：
- **无锚框设计**：不再依赖预设 anchor boxes，直接预测框的边界
- **解耦头**：分类和回归分支分离，避免任务冲突
- **C2f 模块**：改进的特征提取模块，替代了 C3

### 第6步：可视化 —— 对比 NMS 前后

```python
def draw_detections(image, boxes, scores, class_ids, class_names, save_path, title):
    # 为每个类别分配颜色
    colors = generate_colors(len(class_names))

    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        color = colors[int(cls_id) % len(colors)]

        # 绘制边界框
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2,
                                  edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        # 绘制标签（类别名 + 置信度）
        label = f"{class_names[int(cls_id)]} {score:.2f}"
        ax.text(x1, y1-5, label, fontsize=9, color='white',
                bbox=dict(facecolor=color, alpha=0.7))
```

代码分别绘制 NMS 前后的检测结果，直观展示 NMS 如何消除对同一物体的重复检测。

### 第7步：IoU 示例可视化

代码创建三张子图展示 IoU 的三种典型情况：

| 场景 | 重叠程度 | IoU | 含义 |
|------|---------|-----|------|
| Nearly Perfect | 几乎完全重叠 | ~0.96 | 预测框非常接近真值 |
| Moderate Overlap | 中等重叠 | ~0.55 | 刚好超过 0.5 阈值（正样本） |
| Almost Disjoint | 几乎不重叠 | ~0.08 | 远低于 0.5 阈值（负样本） |

### 关键概念速查表

| 概念 | 公式 | 代码对应 |
|------|------|---------|
| IoU | $\frac{\|A \cap B\|}{\|A \cup B\|}$ | `compute_iou(box1, box2)` |
| NMS 算法 | 保留最高分框，移除高 IoU 冗余框 | `nms(boxes, scores, threshold)` |
| xyxy 格式 | $[x_1, y_1, x_2, y_2]$ | IoU 计算、NMS 输入 |
| xywh 格式 | $[c_x, c_y, w, h]$ | YOLO 输出 |
| YOLO 归一化 | 中心相对于网格单元，宽高相对于整图 | `yolo_output_to_pixel()` |
| mAP | PR 曲线下面积的多类别平均值 | 评估指标 |
| COCO 80 类 | person, bicycle, car, ... | `COCO_CLASSES` 列表 |
| YOLOv8 nano | ~3.2M 参数，CPU 可运行 | `YOLO("yolov8n.pt")` |

## 完整代码

<<< @/applied/cv/object-detection/code/demo.py
