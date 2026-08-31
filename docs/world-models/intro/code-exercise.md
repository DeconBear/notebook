---
title: "wm01 世界模型导论 — exercise.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# wm01 世界模型导论与分类 — exercise.py 练习指南

<a href="/notebook/code/world-models/intro/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

把 index.md 中"六大技术路径分类逻辑"和"多维度打分归一化"这两个概念代码化，并实现 demo.py 中 rollout 误差累积仿真的核心递推公式，从代码层面巩固对世界模型分类体系的理解。

## 预备知识

- 六条技术路径的判定依据：是否用树搜索、是否重建像素、是否有显式动力学、是否用语言 token 等（参见 [index.md 第四节](../wm01_world_model_intro/#四-六条技术路径分类)）
- min-max 归一化公式：$\text{normalized} = (x - \min)/(\max - \min + \epsilon)$
- 误差累积递推：$e_t = e_{t-1}(1+\text{step\_error}) + |noise_t|$

## 任务清单

### 任务1：实现世界模型分类器 `classify_world_model(features)`

- **实现步骤**：按优先级依次检查 `uses_tree_search` → `MuZero`；`predicts_pixels=False and has_explicit_dynamics=True` → `JEPA`；`predicts_pixels=True and learns_action_labels=False` → `Genie`；`predicts_pixels=True and is_diffusion_or_autoregressive=True` → `VideoGen`；`uses_language_tokens=True` → `LLM`；否则 → `RSSM/Dreamer`
- **提示**：用 `features.get(key, False)` 安全读取字典，缺失的键默认视为 `False`

### 任务2：实现打分归一化 `normalize_scores(scores)`

- **实现步骤**：
  1. 把字典转成矩阵 `raw`，形状 `(n_methods, n_dims)`
  2. `col_min = raw.min(axis=0)`，`col_max = raw.max(axis=0)`
  3. `normalized_raw = (raw - col_min) / (col_max - col_min + 1e-8)`
- **直觉理解**：按"列"（维度）而不是"行"（方法）归一化，这样每个评价维度上表现最好的方法都会被拉到 1.0，方便跨维度公平比较

### 任务3（Bonus）：实现单条误差累积轨迹 `simulate_rollout_error_toy(horizon, step_error_rate, ...)`

- **实现步骤**：
  ```
  e = init_error
  for t in range(horizon):
      noise = rng.normal(0, noise_scale)
      e = e * (1 + step_error_rate) + abs(noise)
      errors[t] = e
  ```

## 验证标准

运行 `python exercise.py`：

1. `test_classify_world_model()`：6 个代表性方法（PlaNet/Dreamer, MuZero, V-JEPA, Genie, Sora, LLM-as-simulator）应全部分类正确
2. `test_normalize_scores()`：第 2 维（规划能力）归一化后应包含 1.0（最高分）和 0.0（最低分）
3. `test_simulate_rollout_error()`（Bonus）：`step_error_rate=0.045` 的最终累积误差应明显大于 `step_error_rate=0.018` 的情形

## 延伸思考

- 如果一个方法同时 `uses_tree_search=True` 又 `uses_language_tokens=True`（比如某些结合 LLM 与搜索的智能体），按本练习的优先级规则会被分到哪一类？你觉得这种优先级设计的合理性边界在哪里？
- min-max 归一化对异常值（某个方法在某维度上极端地高或低）很敏感。如果换成基于排名的归一化，结果会有什么不同？
- 尝试把 `noise_scale` 从 0.003 调到 0.02，观察误差累积曲线的方差如何变化，这对应真实世界模型训练中"环境随机性"的影响。

## 完整代码

<<< @/world-models/intro/code/exercise.py
