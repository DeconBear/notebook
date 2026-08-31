---
title: "wm05 JEPA — exercise.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# wm05 JEPA / V-JEPA — exercise.py 练习指南

<a href="/notebook/code/world-models/abstract/jepa/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

实现 JEPA 训练流程中三个核心构件——图像分块、目标编码器的 EMA 更新、以及组合掩码划分与表征空间损失的完整前向计算，从代码层面掌握"预测表征而非像素"这一核心思想的工程实现。

## 预备知识

- Vision Transformer 的标准分块操作（reshape + transpose）
- EMA 更新公式：$\theta_{\text{target}} \leftarrow m\cdot\theta_{\text{target}} + (1-m)\cdot\theta_{\text{context}}$
- `torch.gather` 的用法：按索引张量从源张量中"抽取"元素

## 任务清单

### 任务1：实现 `patchify(images, patch_size)`

- **实现步骤**：
  1. `n, H, W = images.shape`；`n_h, n_w = H // patch_size, W // patch_size`
  2. reshape 成 `(n, n_h, patch_size, n_w, patch_size)`
  3. `transpose(0, 1, 3, 2, 4)` 把两个 `patch_size` 维度换到相邻位置
  4. reshape 成 `(n, n_h*n_w, patch_size*patch_size)`
- **验证技巧**：`images.shape=(10,20,20)`，`patch_size=4` 时，输出应为 `(10, 25, 16)`

### 任务2：实现 `ema_update(target, context, momentum)`

- **实现步骤**：对 `target` 和 `context` 的每一对参数，执行 `p_t.data.mul_(momentum).add_(p_c.data, alpha=1-momentum)`
- **易错点**：必须用原地操作（`.mul_`/`.add_`），且整个函数要在 `@torch.no_grad()` 装饰器下运行——目标网络永远不接收梯度

### 任务3：实现 `compute_jepa_loss(...)`

- **实现步骤**：
  1. 用 `torch.gather` 从 `batch_patches` 中取出上下文 patch
  2. `ctx_repr = ctx_encoder(ctx_patches, context_idx)`
  3. 在 `torch.no_grad()` 下，用 `tgt_encoder` 编码全部 patch，再 `gather` 出目标位置的表征
  4. `pred_repr = predictor(ctx_repr, context_idx, target_idx)`
  5. `loss = F.mse_loss(pred_repr, tgt_repr)`
- **易错点**：步骤 3 必须包在 `torch.no_grad()` 内；`gather` 的索引张量维度要用 `.unsqueeze(-1).expand(...)` 广播到与源张量匹配

## 验证标准

运行 `python exercise.py`，`train_and_check()` 会自动检测：

1. TODO 1 未完成时会提示"无法继续训练"
2. TODO 2 未实现时会提示"目标网络参数未发生变化"（通过对比 EMA 更新前后的权重）
3. TODO 3 完成后，训练损失应从初始值下降到 1.5 倍以下

## 延伸思考

- 如果去掉 EMA、直接让目标编码器和上下文编码器共享同一套参数（不做任何非对称处理），你预期会发生什么？（提示：回顾 index.md 第二节"为什么不会表征坍缩"）
- 本练习的掩码策略是"随机选一段连续区间"，比真实 I-JEPA 的"多个矩形块掩码"更简单。你觉得这种简化会让任务变得更容易还是更难？
- 如果把 `momentum` 从 0.996 调到 0.9（更新更快），训练还能稳定收敛吗？动量太小会有什么风险？

## 完整代码

<<< @/world-models/abstract/jepa/code/exercise.py
