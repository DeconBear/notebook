---
title: "s12 目标检测 — exercise.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# s12 目标检测 — exercise.py 练习指南

<a href="../code/s12_object_detection/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过手写目标检测的核心算法，深入理解 IoU、NMS、YOLO 坐标转换和 mAP 评估：

1. **实现 IoU 计算** —— 理解交并比的数学原理和代码表达
2. **实现 NMS 算法** —— 掌握去除冗余检测框的经典算法
3. **掌握 YOLO 坐标转换** —— 归一化坐标 $\leftrightarrow$ 像素坐标
4. **理解 mAP 计算流程** —— Precision、Recall 和 F1 在检测中的定义

## 预备知识

- **IoU**：$\text{IoU}(A, B) = |A \cap B| / |A \cup B|$，衡量两个框的重叠程度
- **NMS**：贪心算法，保留置信度最高的框，移除与它 IoU 过高的框
- **边界框格式**：
  - `[x1, y1, x2, y2]`：左上角 + 右下角（像素坐标）
  - `[cx, cy, w, h]`：中心 + 宽高（归一化坐标）
- **Precision**：$TP / (TP + FP)$ —— 预测的框中，有多少是对的
- **Recall**：$TP / (TP + FN)$ —— 真实框中，有多少被找到了
- **F1**：$2 \times P \times R / (P + R)$ —— P 和 R 的调和平均

## 任务清单

### 练习 1：实现 IoU（交并比）计算

**任务**：在 `compute_iou(box_a, box_b)` 中实现两个边界框的 IoU 计算。

**公式**：

$$
\text{IoU}(A, B) = \frac{\text{area}(A \cap B)}{\text{area}(A \cup B)}
$$

**步骤提示**：

```
1. 交集矩形的左上角 = (max(x1_a, x1_b), max(y1_a, y1_b))
2. 交集矩形的右下角 = (min(x2_a, x2_b), min(y2_a, y2_b))
3. 交集面积 = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
4. area_a = (x2_a - x1_a) * (y2_a - y1_a)
5. area_b = (x2_b - x1_b) * (y2_b - y1_b)
6. 并集面积 = area_a + area_b - 交集面积
7. IoU = 交集面积 / 并集面积（如果并集为0则返回0）
```

**代码框架**：

```python
def compute_iou(box_a, box_b):
    x1_inter = max(box_a[0], box_b[0])
    y1_inter = max(box_a[1], box_b[1])
    x2_inter = min(box_a[2], box_b[2])
    y2_inter = min(box_a[3], box_b[3])

    inter_w = max(0, x2_inter - x1_inter)
    inter_h = max(0, y2_inter - y1_inter)
    inter_area = inter_w * inter_h

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area
```

**测试用例**：

| box_a | box_b | 预期 IoU | 场景 |
|-------|-------|----------|------|
| [10,10,50,50] | [10,10,50,50] | 1.0 | 完全重叠 |
| [10,10,50,50] | [60,60,100,100] | 0.0 | 完全不重叠 |
| [10,10,50,50] | [30,30,70,70] | ~0.143 | 部分重叠 |

**几何直觉**：
- 交集面积 = $(50-30) \times (50-30) = 400$（红色和绿色重叠的中间区域）
- box1 面积 = $(50-10) \times (50-10) = 1600$
- box2 面积 = $(70-30) \times (70-30) = 1600$
- 并集面积 = $1600 + 1600 - 400 = 2800$
- IoU = $400 / 2800 \approx 0.143$

### 练习 2：实现 NMS（非极大值抑制）

**任务**：实现 NMS 算法。

**为什么这个算法叫"非极大值抑制"？** 在一个重叠框的集合中，我们只保留置信度"极大"的那个，抑制（丢弃）所有非极大的。这与图像处理中的非极大值抑制（如 Canny 边缘检测中只保留梯度方向上的局部最大值）理念一致。

**算法伪代码**：

```
Input:  B = {b1,...,bN}, S = {s1,...,sN}, Nt (IoU阈值)
Output: D (保留的框)

D ← {}
while B ≠ {}:
    m ← argmax(S)           # 选置信度最高的框
    M ← bm; D ← D ∪ {M}
    B ← B \ {M}             # 从待处理列表中移除
    for bi in B:
        if IoU(M, bi) ≥ Nt: # 如果与 M 重叠太多
            B ← B \ {bi}    # 移除
return D
```

**代码框架**：

```python
def nms(boxes, scores, iou_threshold=0.5):
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    order = scores.argsort()[::-1]  # 按置信度降序
    keep = []

    while len(order) > 0:
        idx = order[0]
        keep.append(idx)
        if len(order) == 1:
            break

        # 计算当前框与剩余框的 IoU
        current_box = boxes[idx]
        remaining_boxes = boxes[order[1:]]

        # 计算所有剩余框与当前框的 IoU
        ious = []
        for i in range(len(remaining_boxes)):
            ious.append(compute_iou(current_box, remaining_boxes[i]))
        ious = np.array(ious)

        # 保留 IoU ≤ threshold 的框
        remaining_indices = np.where(ious <= iou_threshold)[0]
        order = order[remaining_indices + 1]

    return np.array(keep, dtype=np.int64)
```

**测试用例**：

```
输入: 5 个框 + 置信度 [0.95, 0.82, 0.76, 0.88, 0.61]
  - 框0 (置信度0.95): [100,100,200,200]
  - 框1 (置信度0.82): [110,110,210,210]   # 与框0高度重叠
  - 框2 (置信度0.76): [105,105,195,195]   # 与框0高度重叠
  - 框3 (置信度0.88): [300,100,400,200]   # 位置独立
  - 框4 (置信度0.61): [115,115,205,205]   # 与框0高度重叠

期望输出: [0, 3]（保留框0和框3）
```

**NMS 的执行过程（逐步演示）**：

1. 排序后：`order = [0, 3, 1, 2, 4]`
2. 取框0 (0.95)，保留。移除与框0 IoU>0.5 的框1,2,4。剩余：`order = [3]`
3. 取框3 (0.88)，保留。只剩一个框，结束。
4. 输出：`[0, 3]`

### 练习 3：YOLO 输出格式转换为像素坐标

**任务**：将 YOLO 的归一化输出 `[cx, cy, w, h]` 转换为像素坐标 `[x1, y1, x2, y2]`。

**YOLO 坐标的含义**：
- `cx, cy`：中心点坐标，归一化到 $[0, 1]$（相对于图像宽高）
- `w, h`：框的宽高，归一化到 $[0, 1]$（相对于图像宽高）

**转换步骤**：

```
1. cx_pixel = cx * img_w    # 中心 x 转为像素
2. cy_pixel = cy * img_h    # 中心 y 转为像素
3. w_pixel = w * img_w      # 宽度转为像素
4. h_pixel = h * img_h      # 高度转为像素
5. x1 = cx_pixel - w_pixel/2  # 左上角
6. y1 = cy_pixel - h_pixel/2  # 左上角
7. x2 = cx_pixel + w_pixel/2  # 右下角
8. y2 = cy_pixel + h_pixel/2  # 右下角
```

**代码框架**：

```python
def yolo_to_pixel(boxes, img_w, img_h):
    cx = boxes[:, 0]
    cy = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    cx_pixel = cx * img_w
    cy_pixel = cy * img_h
    w_pixel = w * img_w
    h_pixel = h * img_h

    x1 = cx_pixel - w_pixel / 2
    y1 = cy_pixel - h_pixel / 2
    x2 = cx_pixel + w_pixel / 2
    y2 = cy_pixel + h_pixel / 2

    return np.stack([x1, y1, x2, y2], axis=1)
```

**测试用例**：

```python
# 输入: 两个 YOLO 归一化框
boxes = [[0.5, 0.5, 0.3, 0.4],   # 中心在图像正中
         [0.25, 0.75, 0.15, 0.2]] # 左下方小物体
img_w, img_h = 640, 480

# 期望输出（第一个框）:
# cx_pixel=0.5*640=320, cy_pixel=0.5*480=240
# w_pixel=0.3*640=192, h_pixel=0.4*480=192
# x1=320-96=224, y1=240-96=144, x2=320+96=416, y2=240+96=336
# 结果: [224, 144, 416, 336]
```

### 练习 4：评估检测器的 Precision 和 Recall

**任务**：理解并实现单类别的 Precision/Recall 计算。

**在目标检测中**：

| 符号 | 定义 | 判定条件 |
|------|------|---------|
| TP（True Positive） | 正确检测 | 预测框与 GT 框 IoU > threshold 且类别正确 |
| FP（False Positive） | 误检 | 预测框没有匹配的 GT（IoU 不足或类别错误） |
| FN（False Negative） | 漏检 | GT 框没有被任何预测匹配到 |

**关键规则**：一个 GT 框只能被匹配一次（"一个真值不能被重复算对"）。

$$
\text{Precision} = \frac{TP}{TP + FP}, \quad
\text{Recall} = \frac{TP}{TP + FN}, \quad
F_1 = \frac{2 \times P \times R}{P + R}
$$

**代码框架**：

```python
def compute_precision_recall(pred_boxes, pred_scores, pred_classes,
                               gt_boxes, gt_classes, iou_threshold=0.5):
    TP = FP = 0
    matched_gt = set()  # 已被匹配的 GT 框索引

    for i, (pred_box, pred_cls) in enumerate(zip(pred_boxes, pred_classes)):
        best_iou = 0
        best_gt_idx = -1
        # 找最佳匹配的 GT 框
        for j, (gt_box, gt_cls) in enumerate(zip(gt_boxes, gt_classes)):
            if j in matched_gt:
                continue  # 已被匹配，跳过
            if pred_cls != gt_cls:
                continue  # 类别不同，跳过
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        # 判断是否匹配
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            TP += 1
            matched_gt.add(best_gt_idx)
        else:
            FP += 1

    FN = len(gt_boxes) - len(matched_gt)  # 未被匹配的 GT 框

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1
```

> **注意**：本练习是简化的单类别单阈值版本。真正的 mAP 需要对每个类别、多个 IoU 阈值和多个置信度阈值分别计算，再取平均。COCO 标准的 mAP@0.5:0.95 需要在 10 个 IoU 阈值下各算一次 AP 再取平均。

## 完整代码

<<< @/snippets/s12_object_detection/exercise.py
