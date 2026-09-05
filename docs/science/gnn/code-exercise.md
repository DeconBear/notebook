---
title: "as05 科学计算中的 GNN — exercise.py"
---

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# as05 科学计算中的 GNN — exercise.py 练习指南

<a href="/notebook/code/science/gnn/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

1. 手写一层均值消息传递（网格平滑）  
2. 手写可学习消息的 `index_add_` 聚合  
3. 观察链图上脉冲扩散与方差下降

## 任务清单

### 任务1：`mean_message_passing`

对每个节点，取邻居特征均值，与自身做 $(1-\alpha)/\alpha$ 混合。完成后：左端脉冲应向右扩散，多步后方差下降。

### 任务2：`aggregate_messages`

`messages = W_msg(h[src])`，再 `index_add_` 到 `dst`，除以度数。`MPLayerExercise` 会调用它完成 `ReLU(W_self(h)+agg)`。

## 验证标准

```bash
cd docs/science/gnn/code
python exercise.py
```

看到「✓ 练习完成！」并生成 `exercise_message_passing.png`。


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/science/gnn/code/exercise.py`
