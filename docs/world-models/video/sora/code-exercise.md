---
title: "wm07 视频生成式世界模型 — exercise.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# wm07 视频生成式世界模型 — exercise.py 练习指南

<a href="/notebook/code/world-models/video/sora/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

1. 实现高斯光斑渲染 `render_blob`
2. 实现视频切分 `build_supervised_pairs`
3. 实现开环滚动预测 `open_loop_rollout`

## 任务清单

### TODO 1：`render_blob`

$$
I(y,x) = \exp\left(-\frac{(y-r)^2+(x-c)^2}{2\sigma^2}\right)
$$

用 `np.meshgrid(..., indexing='ij')` 生成坐标网格后直接向量化计算。

### TODO 2：`build_supervised_pairs`

对每条轨迹每个合法 $t$，取 `videos[i, t:t+hist]` 为输入、`videos[i, t+hist]` 为目标，最后 `np.stack`。

### TODO 3：`open_loop_rollout`

用真实前 `hist` 帧启动；循环中把 `model` 的预测 `append` 回 `frames`，再取 `frames[-hist:]` 作为下一步输入。返回 `(n_roll, H, W)`。

**易错点**：若每步仍用 `video` 的真实未来帧，就变成闭环/教师强制，误差不会随时间明显增长。

## 完成后的验证

运行后应看到开环第 10 步 MSE **大于**第 1 步 MSE——误差累积是成功标志。

## 完整代码

<<< @/world-models/video/sora/code/exercise.py
