---
title: "ml11 隐马尔可夫模型 — exercise.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# ml11 隐马尔可夫模型 — exercise.py 练习指南

<a href="/notebook/code/ml/advanced/hmm/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过实现前向算法的递推步骤、Viterbi 解码的回溯步骤、平稳分布计算和 HMM 序列生成，从代码层面深入理解 HMM 三大问题的数学原理。

## 预备知识

- 前向算法：$\alpha_t(j) = [\sum_i \alpha_{t-1}(i) \cdot a_{ij}] \cdot b_j(x_t)$
- Viterbi 算法：用 max 替代 sum，并记录回溯指针
- 平稳分布：$\pi = \pi \mathbf{A}$，可通过迭代或求解线性方程组得到
- HMM 采样：利用转移矩阵和发射矩阵按步生成状态和观测

## 任务清单

### 任务1：前向算法递推 `forward_algorithm(pi, A, B, observations)`

- **初始化**（$t=1$）：$\alpha_1(i) = \pi_i \cdot B[i, x_1]$
- **递推**（$t=2..T$）：$\alpha_t(j) = [\sum_i \alpha_{t-1}(i) \cdot A_{ij}] \cdot B[j, x_t]$
- **终止**：$P(X|\lambda) = \sum_i \alpha_T(i)$
- **实现提示**：
  - 内积求和：`np.dot(alphas[t-1, :], A[:, j])` 计算 $\sum_i \alpha_{t-1}(i) \cdot A_{ij}$
  - 注意：本练习**不要求**缩放，直接使用原始概率（可能在长序列上下溢）

### 任务2：Viterbi 解码的回溯 `viterbi_decoding(pi, A, B, observations)`

- **关键区别**：用 log 空间避免下溢，用 max 代替 sum
- **candidates 计算**：`delta[t-1, :] + np.log(A[:, j] + eps)`
- **回溯指针**：`psi[t, j] = np.argmax(candidates)`
- **回溯步骤**（最关键的部分）：
  ```python
  best_path[T-1] = best_final_state          # 从终点出发
  for t in range(T-2, -1, -1):
      best_path[t] = psi[t+1, best_path[t+1]]  # 沿着指针逆向走
  ```
- **为什么从 T-2 到 -1？** 回溯是逆向的——从最后时刻的最优状态开始，沿着 psi 指针一步步向前追溯。方向是 `T-1 → 0`。

### 任务3：平稳分布 `stationary_distribution(A)`

- **迭代方法**：$\pi^{(n+1)} = \pi^{(n)} \mathbf{A}$，收敛后 $\pi \approx \pi \mathbf{A}$
- **收敛判断**：$\|\pi_{\text{new}} - \pi_{\text{old}}\| < \text{tol}$
- **实现**：`pi_new = pi @ A`（矩阵乘法）

### 任务4：HMM 序列生成 `sample_hmm(pi, A, B, T)`

- **初始状态采样**：`z_1 = rng.choice(N, p=pi)`——从初始分布中选
- **初始观测采样**：`x_1 = rng.choice(M, p=B[z_1, :])`——从发射分布中选
- **递推**：对每个 $t$，从 $A[z_{t-1}, :]$ 采样 $z_t$，再从 $B[z_t, :]$ 采样 $x_t$
- **`np.random.choice` 的 `p` 参数**：指定每个元素的采样概率（必须和为 1）

## 完整代码

<<< @/ml/advanced/hmm/code/exercise.py
