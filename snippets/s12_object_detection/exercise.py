# -*- coding: utf-8 -*-
"""
s12 目标检测 练习
==================
完成以下 TODO 练习来加深对目标检测核心算法的理解：
IoU 计算、NMS（非极大值抑制）、YOLO 输出格式转换。
"""

import numpy as np
from typing import List, Tuple


# ============================================================
# 练习 1：实现 IoU（交并比）计算
# ============================================================

def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    TODO: 计算两个边界框的 IoU

    边界框格式: [x1, y1, x2, y2] —— 左上角 + 右下角坐标

    参数:
        box_a: 第一个边界框，形状 (4,) → [x1, y1, x2, y2]
        box_b: 第二个边界框，形状 (4,) → [x1, y1, x2, y2]

    返回:
        iou: IoU 值，范围 [0, 1]

    公式:
        IoU = area(A ∩ B) / area(A ∪ B)

    提示:
    1. 交集矩形的左上角 = (max(x1_a, x1_b), max(y1_a, y1_b))
    2. 交集矩形的右下角 = (min(x2_a, x2_b), min(y2_a, y2_b))
    3. 交集面积 = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    4. 并集面积 = area_a + area_b - 交集面积
    5. 如果并集面积为 0，返回 0.0
    """
    # TODO: 计算交集区域
    # x1_inter = max(???, ???)  # 交集左上角 x
    # y1_inter = max(???, ???)  # 交集左上角 y
    # x2_inter = min(???, ???)  # 交集右下角 x
    # y2_inter = min(???, ???)  # 交集右下角 y
    # inter_area = ???

    # TODO: 计算各自的面积
    # area_a = ???
    # area_b = ???

    # TODO: 计算并集面积和 IoU
    # union_area = ???
    # iou = ??? if union_area > 0 else 0.0

    return 0.0  # 替换为你的实现


# ============================================================
# 练习 2：实现 NMS（非极大值抑制）
# ============================================================

def nms(boxes: np.ndarray, scores: np.ndarray,
        iou_threshold: float = 0.5) -> np.ndarray:
    """
    TODO: 实现非极大值抑制（NMS）算法

    参数:
        boxes: 边界框数组，形状 (N, 4)，格式 [x1, y1, x2, y2]
        scores: 置信度得分数组，形状 (N,)
        iou_threshold: IoU 阈值，高于此值的框被抑制

    返回:
        keep: 保留的框的索引数组

    算法伪代码:
        Input: B = {b1,...,bN}, S = {s1,...,sN}, Nt (IoU threshold)
        Output: D (selected boxes)

        D ← {}
        while B ≠ {}:
            m ← argmax(S)           # 选置信度最高的框
            M ← b_m                 # 该框
            D ← D ∪ {M}; B ← B - {M}  # 保存并从 B 中移除
            for b_i in B:
                if IoU(M, b_i) >= Nt:  # 如果与 M 重叠太多
                    B ← B - {b_i}      # 移除该框
        return D

    提示:
    1. 用 argsort()[::-1] 按置信度降序排列
    2. 用循环实现上述伪代码
    3. 使用练习 1 中的 compute_iou 计算重叠
    4. 注意处理边界情况（空输入）
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    # TODO: 按置信度降序排序
    # order = ???

    # TODO: 逐步选出最高分框并移除重叠框
    keep = []  # 存放保留的索引

    # while len(order) > 0:
    #     idx = order[0]  # 当前最高分框
    #     keep.append(idx)
    #     if len(order) == 1:
    #         break
    #     # 计算当前框与剩余框的 IoU
    #     # 保留 IoU <= threshold 的框
    #     order = ???

    return np.array(keep, dtype=np.int64)


# ============================================================
# 练习 3：YOLO 输出格式 → 像素坐标转换
# ============================================================

def yolo_to_pixel(boxes: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """
    TODO: 将 YOLO 的归一化输出转换为像素坐标

    YOLO 输出格式（归一化）:
        - (x, y, w, h)，其中:
          - x, y: 边界框中心坐标，归一化到 [0, 1]（相对于图像宽高）
          - w, h: 边界框宽高，归一化到 [0, 1]（相对于图像宽高）

    转换为像素格式:
        - [x1, y1, x2, y2]，其中:
          - (x1, y1): 左上角，像素坐标
          - (x2, y2): 右下角，像素坐标

    参数:
        boxes: YOLO 归一化框，形状 (N, 4)，格式 [cx, cy, w, h]（归一化）
        img_w: 图像宽度（像素）
        img_h: 图像高度（像素）

    返回:
        boxes_pixel: 像素坐标框，形状 (N, 4)，格式 [x1, y1, x2, y2]

    转换步骤:
    1. cx_pixel = cx * img_w      # 中心 x 转为像素
    2. cy_pixel = cy * img_h      # 中心 y 转为像素
    3. w_pixel = w * img_w         # 宽度转为像素
    4. h_pixel = h * img_h         # 高度转为像素
    5. x1 = cx_pixel - w_pixel/2   # 左上角 x
    6. y1 = cy_pixel - h_pixel/2   # 左上角 y
    7. x2 = cx_pixel + w_pixel/2   # 右下角 x
    8. y2 = cy_pixel + h_pixel/2   # 右下角 y
    """

    # TODO: 从输入中提取 cx, cy, w, h
    # cx = boxes[:, 0]
    # cy = boxes[:, 1]
    # w = boxes[:, 2]
    # h = boxes[:, 3]

    # TODO: 转换为像素坐标
    # cx_pixel = ???
    # cy_pixel = ???
    # w_pixel = ???
    # h_pixel = ???

    # TODO: 计算 x1, y1, x2, y2
    # x1 = ???
    # y1 = ???
    # x2 = ???
    # y2 = ???

    # TODO: 堆叠为 (N, 4) 格式
    # boxes_pixel = np.stack([x1, y1, x2, y2], axis=1)

    return None  # 替换为你的实现


# ============================================================
# 练习 4：评估检测器的 mAP（概念实现）
# ============================================================

def compute_precision_recall(pred_boxes: np.ndarray,
                              pred_scores: np.ndarray,
                              pred_classes: np.ndarray,
                              gt_boxes: np.ndarray,
                              gt_classes: np.ndarray,
                              iou_threshold: float = 0.5) -> Tuple[float, float, float]:
    """
    TODO: 计算目标检测的 Precision 和 Recall（简化版）

    这是理解 mAP 计算的基础。对于单个类别，给定一组预测和 ground truth：

    True Positive (TP): 预测框与某个 GT 框的 IoU > threshold，且类别正确
    False Positive (FP): 预测框没有匹配的 GT 框（IoU 不足或类别错误）
    False Negative (FN): GT 框没有被任何预测框匹配到

    Precision = TP / (TP + FP)  —— 预测的框中，有多少是对的
    Recall = TP / (TP + FN)     —— GT 框中，有多少被找到了
    F1 = 2 * P * R / (P + R)    —— 两者的调和平均

    参数:
        pred_boxes: 预测框，形状 (N_pred, 4), [x1, y1, x2, y2]
        pred_scores: 预测置信度，形状 (N_pred,)
        pred_classes: 预测类别，形状 (N_pred,)
        gt_boxes: 真实框，形状 (N_gt, 4), [x1, y1, x2, y2]
        gt_classes: 真实类别，形状 (N_gt,)
        iou_threshold: 匹配的 IoU 阈值

    返回:
        (precision, recall, f1_score)

    提示:
    1. 对每个预测框，检查它是否与某个 GT 框匹配（IoU > threshold 且类别相同）
    2. 一个 GT 框只能被匹配一次（需要标记已匹配的 GT 框）
    3. 按上述定义统计 TP, FP, FN
    """
    # TODO: 实现 Precision/Recall 计算
    # TP = 0  # 正确检测
    # FP = 0  # 误检
    # FN = 0  # 漏检

    # matched_gt = set()  # 已被匹配的 GT 框索引

    # for each prediction:
    #   找到最佳匹配的 GT 框
    #   如果 IoU > threshold 且类别正确且该 GT 未被匹配:
    #       TP += 1, 标记该 GT 为已匹配
    #   否则:
    #       FP += 1

    # FN = 未被匹配的 GT 框数量

    # precision = TP / (TP + FP) if TP + FP > 0 else 0
    # recall = TP / (TP + FN) if TP + FN > 0 else 0

    return 0.0, 0.0, 0.0  # 替换为你的实现


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("s12 目标检测 — 练习测试")
    print("=" * 50)

    # ---- 测试练习 1：IoU 计算 ----
    print("\n[练习 1] IoU 计算测试：")

    # 测试用例
    test_cases = [
        ([10, 10, 50, 50], [10, 10, 50, 50], 1.0, "完全重叠"),
        ([10, 10, 50, 50], [60, 60, 100, 100], 0.0, "完全不重叠"),
        ([10, 10, 50, 50], [30, 30, 70, 70], 0.1428, "部分重叠"),
    ]

    for box_a, box_b, expected, desc in test_cases:
        iou = compute_iou(np.array(box_a), np.array(box_b))
        status = "✓" if abs(iou - expected) < 0.01 else "✗"
        print(f"  {status} {desc}: IoU = {iou:.4f} (期望 {expected})")

    # ---- 测试练习 2：NMS ----
    print("\n[练习 2] NMS 测试：")

    boxes = np.array([
        [100, 100, 200, 200],
        [110, 110, 210, 210],
        [105, 105, 195, 195],
        [300, 100, 400, 200],
        [115, 115, 205, 205],
    ], dtype=np.float32)
    scores = np.array([0.95, 0.82, 0.76, 0.88, 0.61], dtype=np.float32)

    keep = nms(boxes, scores, iou_threshold=0.5)

    if keep is not None and len(keep) > 0:
        print(f"  保留的框索引: {keep}")
        print(f"  保留的置信度: {scores[keep]}")
        # 期望: [0, 3]（框0最高分 + 框3位置不同）
        if set(keep) == {0, 3}:
            print(f"  ✓ 结果正确！保留框0和框3")
        else:
            print(f"  期望保留索引 [0, 3]，但得到 {keep}")
    else:
        print("  请完成 NMS 实现")

    # ---- 测试练习 3：YOLO 格式转换 ----
    print("\n[练习 3] YOLO 格式转换测试：")

    # YOLO 归一化输出: [cx, cy, w, h]
    yolo_boxes = np.array([
        [0.5, 0.5, 0.3, 0.4],    # 中心在图像正中
        [0.25, 0.75, 0.15, 0.2], # 左下方小物体
    ])
    img_w, img_h = 640, 480

    result = yolo_to_pixel(yolo_boxes, img_w, img_h)

    if result is not None:
        print(f"  输入 (归一化):\n{yolo_boxes}")
        print(f"  输出 (像素):\n{result}")
        # 验证第一个框
        # cx=0.5*640=320, cy=0.5*480=240, w=0.3*640=192, h=0.4*480=192
        # x1=320-96=224, y1=240-96=144, x2=320+96=416, y2=240+96=336
        expected_first = [224, 144, 416, 336]
        if np.allclose(result[0], expected_first):
            print(f"  ✓ 第一个框转换正确: {result[0]}")
        else:
            print(f"  期望: {expected_first}")
            print(f"  实际: {result[0]}")
    else:
        print("  请完成 yolo_to_pixel 实现")

    print("\n" + "=" * 50)
    print("完成所有练习后，运行 demo.py 查看完整的检测演示。")
    print("=" * 50)
