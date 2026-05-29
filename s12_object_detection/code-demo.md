---
title: "s12 目标检测 — demo.py"
---

# s12 目标检测 — demo.py

<a href="../code/s12_object_detection/demo.py" target="_blank" download>Download demo.py</a>

## What This Code Does

从零实现 IoU（交并比）计算和 NMS（非极大值抑制）算法，并使用 YOLOv8 nano 模型进行目标检测推理，对比 NMS 处理前后的检测结果。通过 IoU 可视化示例展示三种重叠程度的场景，通过 NMS 测试用例验证算法正确性，完整展示目标检测流水线中两个核心后处理步骤。

## Run It

```bash
cd s12_object_detection/code
python demo.py
```

## Key Output

- IoU 测试：完全重叠 IoU=1.0，完全不重叠 IoU=0.0，部分重叠 IoU≈0.143
- NMS 测试：输入 5 个框（3 个重叠 + 1 个独立），NMS 后正确保留 2 个框
- YOLOv8 推理：使用 yolov8n.pt（nano 版本，约 3.2M 参数），COCO 80 类检测
- NMS 能有效去除同一物体上的重复检测框，显著提升检测结果的可读性

<<< @/snippets/s12_object_detection/demo.py
