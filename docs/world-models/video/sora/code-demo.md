---
title: "wm07 视频生成式世界模型 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 视频世界模型 — demo.py 代码详解

<a href="/notebook/code/world-models/video/sora/demo.py" target="_blank" download>Download demo.py</a>

```bash
cd docs/world-models/video/sora/code
python demo.py
```

本文件**没有**训练 CNN。两件事：架构框图；用 `np.roll` 做「估计位移后的朴素下一帧」。

## 代码逐段详解

光斑中心 `(8+t, 6+2t)`，圆盘 `ogrid` 距离小于 12。运动是每步 $\Delta=(1,2)$ 像素。

```python
pred = np.roll(frames[-2], shift=(1, 2), axis=(0, 1))
```

`roll`：数组循环平移。用 $t$ 帧平移 $(1,2)$ 去猜 $t+1$。真 $t+1$ 若还在画布内，应对得很齐；靠近边界会被卷到另一侧——这就是「没有世界模型、只做光流平移」的脆弱点。

`axis('off')` 去掉刻度。三张图：现在、真下一帧、朴素预测。

## 源码位置

`docs/world-models/video/sora/code/demo.py`
