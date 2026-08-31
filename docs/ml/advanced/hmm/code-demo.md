---
title: "ml11 隐马尔可夫模型 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# ml11 隐马尔可夫模型 — demo.py 代码详解

<a href="/notebook/code/ml/advanced/hmm/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/ml/advanced/hmm/code
python demo.py
```

## 代码逐段详解

### 第1步：HMM 类与前向算法

```python
class HMM:
    def __init__(self, A, B, pi):
        self.A = np.array(A)    # 转移矩阵: P(Z_{t+1}=j | Z_t=i)
        self.B = np.array(B)    # 发射矩阵: P(X_t=o | Z_t=i)
        self.pi = np.array(pi)  # 初始分布: P(Z_1=i)
```

HMM 的三个参数对应三个概率分布：初始状态分布 $\boldsymbol{\pi}$（$N$ 维向量）、状态转移矩阵 $\mathbf{A}$（$N \times N$）、发射概率矩阵 $\mathbf{B}$（$N \times M$）。

#### 前向算法（带缩放）

```python
def forward(self, observations):
    alphas = np.zeros((T, self.N))
    c = np.zeros(T)

    # 初始化: α_1(i) = π_i · B[i, x_1]
    for i in range(self.N):
        alphas[0, i] = self.pi[i] * self.B[i, observations[0]]
    c[0] = alphas[0].sum()
    alphas[0] /= c[0]  # 缩放防止下溢

    # 递推: α_t(j) = [Σ_i α_{t-1}(i) · A[i, j]] · B[j, x_t]
    for t in range(1, T):
        for j in range(self.N):
            alphas[t, j] = np.dot(alphas[t-1, :], self.A[:, j]) * self.B[j, observations[t]]
        c[t] = alphas[t].sum()
        alphas[t] /= c[t]

    log_prob = np.sum(np.log(c + 1e-300))
```

**为什么需要缩放（Scaling）？** 当序列长度 $T$ 较大时，$\alpha_t(i)$ 的值会急剧变小（每个因子 $\le 1$，连乘 $T$ 次）。不加缩放的 $\alpha_T$ 可能下溢到 0。缩放的技巧是每步除以 $c_t = \sum_i \alpha_t(i)$，使得每步的 $\tilde{\alpha}_t$ 之和为 1。观测概率的对数可以通过缩放因子恢复：$\log P(X | \lambda) = \sum_{t=1}^T \log c_t$。

**为什么这有效？** 设 $\tilde{\alpha}_t = \alpha_t / (\prod_{s=1}^t c_s)$，则 $\prod_{s=1}^T c_s = \sum_i \alpha_T(i) = P(X | \lambda)$，取对数即为 $\sum \log c_t$。

### 第2步：Viterbi 算法

```python
def viterbi(self, observations):
    # 使用对数空间防止下溢
    deltas[0, i] = np.log(self.pi[i] + eps) + np.log(self.B[i, o1] + eps)

    for t in range(1, T):
        for j in range(self.N):
            candidates = deltas[t-1, :] + np.log(self.A[:, j] + eps)
            best_i = np.argmax(candidates)
            deltas[t, j] = candidates[best_i] + np.log(self.B[j, ot] + eps)
            psi[t, j] = best_i  # 回溯指针
```

Viterbi 与前向算法的关键区别：**max 代替 sum**。这一步体现了不同的目标——前向算法问"所有路径的总概率是多少"，Viterbi 问"哪一条路径最可能"。

对数空间中使用加法律（$\log(ab) = \log a + \log b$）替代乘法律，数值更稳定。特殊地，$\max$ 操作在 log 空间保持不变。

**回溯**：
```python
best_path[T-1] = best_final_state
for t in range(T-2, -1, -1):
    best_path[t] = psi[t+1, best_path[t+1]]
```
从最后时刻的最优状态出发，利用 $\psi_t(j)$ 存储的回溯指针，逆向重构整个最优路径。这就是经典的动态规划回溯技术。

### 第3步：POS 标注格子图可视化

格子的 x 轴是时间（观测序列），y 轴是隐藏状态。两个热度图对比了：
- **左图（Forward $\alpha$）**：展示了在每个时刻处于每个状态的概率（边际化所有路径后）
- **右图（Viterbi $\delta$ + 路径）**：展示了最优路径的构建过程——蓝色箭头连接最优序列

### 第4步：马尔可夫链转移图

展示了三个状态的天气马尔可夫链（Sunny/Cloudy/Rainy），箭头粗细正比于转移概率。通过迭代 $\pi^{(n+1)} = \pi^{(n)} \mathbf{A}$ 或求解 $\pi \mathbf{A} = \pi$ 得到平稳分布。

### 第5步：转移矩阵对状态序列的影响

对比两种极端的转移矩阵：
- **平滑转移**：高自转移概率（0.7），状态倾向保持稳定
- **尖峰转移**：强制循环结构（S1→S2→S3→S1），状态快速切换

Viterbi 解码出的路径反映了转移矩阵的结构——HMM 不可能解码出不遵守转移规律的路径。

## 关键概念速查表

| 概念 | 数学形式 | 代码位置 | 关键说明 |
|------|---------|---------|---------|
| 前向 $\alpha$ | $\alpha_t(j) = [\sum \alpha_{t-1} a_{ij}] b_j(x_t)$ | `HMM.forward()` | 动态规划 O(N^2 T) |
| 缩放 | $\tilde{\alpha}_t = \alpha_t / c_t$ | `c[t]` 归一化 | 防止概率下溢 |
| Viterbi $\delta$ | $\delta_t(j) = \max_i [\delta_{t-1} a_{ij}] b_j(x_t)$ | `HMM.viterbi()` | max 代替 sum |
| 回溯 $\psi$ | $\psi_t(j) = \arg\max_i [\delta_{t-1}(i) a_{ij}]$ | `psi[t, j]` | 记录最优前驱 |
| 平稳分布 | $\pi = \pi \mathbf{A}$ | `np.linalg.eig(A.T)` | 链的平衡状态 |
| log 空间 | $\log(ab) = \log a + \log b$ | `np.log(...)` | 数值稳定替代乘法律 |

## 完整代码

<<< @/ml/advanced/hmm/code/demo.py
