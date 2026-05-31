---
title: "ml13 概率图模型基础 — exercise.py"
---

# ml13 概率图模型基础 — exercise.py 练习指南

<a href="../code/ml13_probabilistic_graphical_models/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过实现联合概率计算、d-分离判断、BP 消息传递和变量消除，从代码层面深入理解概率图模型的核心推理机制。

## 预备知识

- 贝叶斯网络联合分布的因子分解：$P = \prod P(X_i \mid \text{Pa}(X_i))$
- d-分离三种结构：链式（B 观测阻塞）、分叉（B 观测阻塞）、汇合（B 观测反而激活）
- 信念传播消息公式：$\mu_{f \to x_j}(x_j) = \sum_{x_i} \psi(x_i, x_j) \cdot \mu_{x_i \to f}(x_i)$
- 变量消除 = 利用分配律改变求和顺序

## 任务清单

### 任务1：联合概率计算 `compute_joint_probability(cpts, assignment)`

- **实现**：遍历所有 CPT 函数，对给定的 assignment 查询每个 CPT 的值，连乘
- **验证**：对 $P(A) P(B|A)$ 网络，$P(A=0, B=0) = 0.4 \times 0.7 = 0.28$

### 任务2：d-分离判断 `is_d_separated(edges, X, Y, Z)`

- **简化版**：只处理通过单个中间节点 B 连接的情况
- **判断规则**：
  - 链式 X→B→Y 或 X←B→Y：B in Z 则阻塞
  - 汇合 X→B←Y：B not in Z 则阻塞
- **注意**：这只是 d-分离的简化近似，完整版本需要考虑多步路径和后代节点

### 任务3：BP 消息传递 `bp_message_from_factor(psi, msg_in)`

- **核心操作**：$\mu(x_j) = \sum_{x_i} \psi(x_i, x_j) \cdot \text{msg}(x_i)$
- **NumPy 实现**：`(msg_in[:, np.newaxis] * psi).sum(axis=0)`
  - `msg_in[:, np.newaxis]`：将 $(n_i,)$ 广播为 $(n_i, 1)$
  - 与 $\psi$（$(n_i, n_j)$）逐元素相乘
  - 沿 axis=0 求和消除 $x_i$

### 任务4：变量消除 `simple_variable_elimination(factors, query_var, evidence, elim_order)`

- **步骤**：
  1. 切片固定 evidence 变量的取值
  2. 按消除顺序逐个变量：收集含该变量的所有因子 → 乘积 → 求和消除
  3. 剩余因子相乘并归一化
- **注意 ordering**：消除顺序影响计算效率，但不影响最终结果

## 完整代码

<<< @/snippets/ml13_probabilistic_graphical_models/exercise.py
