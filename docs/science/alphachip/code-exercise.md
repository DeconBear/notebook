---
title: "as07 AlphaChip — exercise.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# as07 AlphaChip — exercise.py 练习指南

<a href="/notebook/code/science/alphachip/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

实现芯片布局强化学习的三个核心计算——线长评估、带掩码的动作采样、REINFORCE 损失，从代码层面理解"如何把一个组合优化问题转化为可训练的强化学习环境"。

## 预备知识

- Manhattan 距离：$d = |r_1-r_2| + |c_1-c_2|$
- `torch.distributions.Categorical(logits=...)` 的采样与 `log_prob` 用法
- REINFORCE 公式：$L = -(R-b)\sum_t \log\pi_\theta(a_t\mid s_t)$

## 任务清单

### 任务1：实现 `wirelength(positions, edges)`

- **实现步骤**：对每条边 `(i, j, w)`，计算 `positions[i]` 与 `positions[j]` 的 Manhattan 距离 `d`，把 `w * d` 累加到 `total`

### 任务2：实现 `sample_masked_action(logits, occupied_mask)`

- **实现步骤**：
  1. `masked_logits = logits.masked_fill(occupied_mask.bool(), float('-inf'))`
  2. `dist = torch.distributions.Categorical(logits=masked_logits)`
  3. `cell = dist.sample()`
  4. `log_prob = dist.log_prob(cell)`
- **直觉理解**：`masked_fill` 把已占用位置的分数设为负无穷，softmax 后这些位置的概率精确为 0，保证永远不会采样到非法动作

### 任务3（Bonus）：实现 `compute_reinforce_loss(log_probs, reward, baseline)`

- **实现步骤**：
  1. `total_log_prob = torch.stack(log_probs).sum()`
  2. `advantage = reward - baseline`
  3. `loss = -advantage * total_log_prob`

## 验证标准

运行 `python exercise.py`：

1. `test_wirelength()`：给定 3 个点、2 条边的小例子，验证总线长精确等于 3.0
2. `test_sample_masked_action()`：给定 5 个格子中只有 1 个可用的掩码，连续采样 10 次都应落在唯一可用位置
3. `test_train_toy_placement()`（Bonus）：训练 500 轮后，后期平均线长应明显低于初期平均线长

## 延伸思考

- 如果把基线的滑动平均系数从 0.9 调到 0.99（更新更慢），训练的稳定性和收敛速度会如何变化？
- 本练习的放置顺序 `order` 是固定的（按宏单元编号从0到N-1）。demo.py 中用的是"按连接度从高到低"排序。你觉得放置顺序本身会影响最终布局质量吗？如果把顺序也变成可学习的（比如用另一个策略网络决定"下一步放哪个模块"），复杂度会如何变化？
- 如果奖励函数只用线长（不考虑任何拥塞/密度惩罚），你能想到 RL 策略可能学到的"钻空子"解法吗？（提示：想想现实中"线长最短"是否等价于"物理上可实现的布局"）

## 完整代码

<<< @/science/alphachip/code/exercise.py
