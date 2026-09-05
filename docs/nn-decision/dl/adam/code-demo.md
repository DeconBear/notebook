---
title: "s09 Adam深度解析与训练实战 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# s09 Adam深度解析与训练实战 — demo.py 代码详解

<a href="/notebook/code/nn-decision/dl/adam/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/nn-decision/dl/adam/code
python demo.py
```

## 代码逐段详解

### 第1步：导入库 — 每个库是做什么的

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import os, time
```

- **`numpy`**：矩阵运算、随机数生成、数学函数。整个 MLP 的参数、梯度、一阶矩/二阶矩全用 NumPy 数组存储。
- **`matplotlib`**：绘制训练损失曲线、验证准确率、梯度范数变化、学习率调度曲线、偏差修正对比图、梯度裁剪效果图。
- **`time`**：测量每个 epoch 的训练耗时，帮助评估训练效率。
- **`Optional`**（typing）：标注可选参数类型（如 `clip_grad_norm: Optional[float]` 表示可以是 `float` 或 `None`）。

---

### 第2步：Adam 优化器 — 完整偏差修正实现

```python
class AdamOptimizer:
    def __init__(self, lr=0.001, betas=(0.9, 0.999), eps=1e-8,
                 use_bias_correction=True):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.use_bias_correction = use_bias_correction
        self.m: Dict[int, np.ndarray] = {}  # 一阶矩 {param_id: m_vector}
        self.v: Dict[int, np.ndarray] = {}  # 二阶矩 {param_id: v_vector}
        self.t = 0                          # 迭代步数

    def step(self, params, grads):
        self.t += 1
        for key in params:
            param = params[key]
            grad = grads.get(key)
            if grad is None: continue

            param_id = id(param)
            if param_id not in self.m:
                self.m[param_id] = np.zeros_like(param)
                self.v[param_id] = np.zeros_like(param)

            # 一阶矩（方向）
            self.m[param_id] = (self.beta1 * self.m[param_id]
                                + (1 - self.beta1) * grad)
            # 二阶矩（尺度）
            self.v[param_id] = (self.beta2 * self.v[param_id]
                                + (1 - self.beta2) * (grad ** 2))

            if self.use_bias_correction:
                m_hat = self.m[param_id] / (1 - self.beta1 ** self.t)
                v_hat = self.v[param_id] / (1 - self.beta2 ** self.t)
                param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            else:
                param -= self.lr * self.m[param_id] / (np.sqrt(self.v[param_id]) + self.eps)
```

**完整数学公式**（每一步的对应关系）：

| 步骤 | 公式 | 代码 |
|------|------|------|
| 1. 更新一阶矩 | $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$ | `self.m[param_id] = self.beta1 * self.m[param_id] + (1 - self.beta1) * grad` |
| 2. 更新二阶矩 | $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$ | `self.v[param_id] = self.beta2 * self.v[param_id] + (1 - self.beta2) * (grad ** 2)` |
| 3. 偏差修正 | $\hat{m}_t = m_t / (1 - \beta_1^t)$ | `m_hat = self.m[param_id] / (1 - self.beta1 ** self.t)` |
| 4. 偏差修正 | $\hat{v}_t = v_t / (1 - \beta_2^t)$ | `v_hat = self.v[param_id] / (1 - self.beta2 ** self.t)` |
| 5. 参数更新 | $\theta_{t+1} = \theta_t - \alpha \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$ | `param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)` |

**设计要点**：

- **为什么用 `id(param)` 作为 key？** 因为参数字典的 key 是字符串（如 "W1"），但当参数被复制或共享时，字符串 key 可能不是唯一的。Python 的 `id()` 返回对象的内存地址，确保每个参数有唯一的标识。
- **懒初始化（lazy init）**：$m$ 和 $v$ 在第一次遇到参数时才分配——避免在 `__init__` 时就需要知道所有参数的形状。
- **`use_bias_correction` 开关**：允许运行对比实验——有/无偏差修正的 Adam 在训练初期的表现差异。

---

### 第3步：AdamW — 解耦权重衰减

```python
class AdamWOptimizer:
    def step(self, params, grads):
        # ... (与 Adam 相同的 m_t, v_t, 偏差修正) ...

        # AdamW 关键：先做自适应更新，再独立应用权重衰减
        param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)   # 自适应更新
        param -= self.lr * self.weight_decay * param             # 独立权重衰减
```

**Adam vs AdamW 的区别**：

- **Adam + L2 正则**：把权重衰减混入梯度，$g \leftarrow g + \lambda \theta$，然后被 $\sqrt{\hat{v}}$ 自适应缩放——衰减效果不均匀。
- **AdamW**：将权重衰减从梯度解耦，$\theta \leftarrow \theta - \alpha \cdot \text{Adam\_update} - \alpha \lambda \theta$——衰减效果对所有参数一致。

数学公式：

$$
\theta_{t+1} = \theta_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \alpha \lambda \theta_t
$$

**为什么这很重要？** Loshchilov & Hutter (2019) 发现，Adam + L2 的正则化效果被自适应缩放严重扭曲——梯度大的参数受到的正则化较弱，梯度小的参数受到的正则化较强。AdamW 的独立权重衰减让正则化力量对所有参数公平施加，在大模型训练中表现更好。

---

### 第4步：SGD+Momentum — 对比基线

```python
class SGDMomentumOptimizer:
    def step(self, params, grads):
        for key in params:
            # ...
            self.m[param_id] = self.momentum * self.m[param_id] + grad
            param -= self.lr * self.m[param_id]
```

注意：SGD+Momentum 的动量更新公式与 Adam 的一阶矩不同——这里用的是 $m_t = \beta m_{t-1} + g_t$（没有 $1-\beta$ 因子），这是经典 Momentum 的标准写法。有无 $1-\beta$ 因子的区别只是对动量做一个缩放，本质上等价，可以通过调整学习率来补偿。

---

### 第5步：MLP 模型 — Softmax + 交叉熵

```python
class MLP:
    def forward(self, X):
        # 隐藏层：ReLU
        for l in range(1, self.L):
            Z = W @ A_prev + b
            A = np.maximum(0, Z)    # ReLU
        # 输出层：Softmax（数值稳定版本）
        Z_stable = Z - np.max(Z, axis=0, keepdims=True)
        exp_Z = np.exp(Z_stable)
        A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
```

**数值稳定的 Softmax**：

$$
\text{softmax}(z_i) = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}
$$

减去最大值不改变结果（分子分母同时乘以 $e^{-\max(z)}$），但能防止 $e^{z_i}$ 溢出。例如，如果 $z = [1000, 2000, 3000]$，直接算 $e^{3000}$ 会溢出；先减去 3000 变成 $[-2000, -1000, 0]$，不会溢出。

**Softmax + 交叉熵的组合梯度**：

```python
def backward(self, Y, caches):
    dZ = caches[-1]["A"] - Y  # δ[L] = A[L] - Y
```

这是 Softmax + 交叉熵组合最优雅的性质——输出层的 $\delta^{[L]}$ 直接是**预测概率减去 one-hot 标签**，不需要计算 Softmax 的复杂雅可比矩阵。数学上这是因为 Softmax 的导数和交叉熵的导数恰好约掉了复杂项。

**隐藏层的 ReLU 反向传播**：

```python
dZ_prev = (W_curr.T @ dZ) * (Z_prev > 0)
```

$\text{ReLU}'(Z) = \mathbb{1}[Z > 0]$，用 `(Z_prev > 0)` 实现布尔门控。

---

### 第6步：学习率调度 — Warmup + Cosine Decay

```python
class LRScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.01):
        self.base_lr = optimizer.lr
        self.min_lr = optimizer.lr * min_lr_ratio

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            # Warmup 阶段：线性从 0 增加到 base_lr
            progress = self.current_step / self.warmup_steps
            self.optimizer.lr = self.base_lr * progress
        else:
            # Cosine Decay：从 base_lr 余弦衰减到 min_lr
            progress = (self.current_step - self.warmup_steps) / \
                       (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            self.optimizer.lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * \
                                (1 + np.cos(np.pi * progress))
```

**Warmup**：训练初期线性增加学习率

$$
\alpha_t = \alpha_{\text{target}} \cdot \frac{t}{t_{\text{warmup}}}, \quad t \leq t_{\text{warmup}}
$$

**Cosine Decay**：训练后期按余弦曲线衰减

$$
\alpha_t = \alpha_{\text{min}} + \frac{1}{2}(\alpha_{\text{max}} - \alpha_{\text{min}})\left(1 + \cos\left(\pi \cdot \frac{t - t_{\text{warmup}}}{T - t_{\text{warmup}}}\right)\right)
$$

**为什么需要 warmup？** Adam 的 $m_t$ 和 $v_t$ 在训练开始时是从零初始化的，早期梯度估计极不靠谱。如果直接以全量学习率更新，很可能一步跳到不稳定区域。Warmup 给优化器留出几百到几千步的"预热"时间，让 $m_t$ 和 $v_t$ 逐步积累到稳定状态。

---

### 第7步：梯度裁剪 — 防止爆炸的最后防线

```python
def clip_gradients(grads, max_norm):
    total_norm_sq = 0.0
    for grad in grads.values():
        if grad is not None:
            total_norm_sq += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm_sq)

    if total_norm > max_norm:
        scale = max_norm / total_norm
        grads_clipped = {k: g * scale for k, g in grads.items()}
        return grads_clipped
    return grads
```

数学公式：

$$
\tilde{g} = \begin{cases}
g & \text{if } \|g\|_2 \leq \text{max\_norm} \\
g \cdot \dfrac{\text{max\_norm}}{\|g\|_2} & \text{if } \|g\|_2 > \text{max\_norm}
\end{cases}
$$

**关键点**：梯度裁剪**不改变梯度方向**，只限制长度。所有梯度被按相同比例缩放，因此更新方向不变，只是步长被限制。

---

### 第8步：训练循环 — 单 epoch 的完整流程

```python
def train_one_epoch(model, optimizer, X, Y_labels, batch_size,
                    clip_grad_norm=None, scheduler=None):
    for start in range(0, m, batch_size):
        X_batch = X[:, batch_idx]
        Y_batch = to_one_hot(Y_batch_labels)

        # ① 前向传播
        probs, caches = model.forward(X_batch)

        # ② 反向传播
        grads = model.backward(Y_batch, caches)

        # ③ 梯度裁剪
        if clip_grad_norm is not None:
            grads = clip_gradients(grads, clip_grad_norm)

        # ④ 记录梯度范数
        grad_norm = compute_gradient_norm(grads)

        # ⑤ 参数更新
        optimizer.step(model.params, grads)

        # ⑥ 学习率调度
        if scheduler is not None:
            scheduler.step()
```

六步训练流水线：
1. **前向**：计算预测和中间值
2. **反向**：计算所有参数的梯度（$\delta$ 递推）
3. **裁剪**：限制梯度最大范数（可选）
4. **记录**：保存梯度范数用于诊断
5. **更新**：优化器用梯度更新参数
6. **调度**：调整学习率（可选）

---

### 第9步：偏差修正对比实验

```python
for use_bc, label in [(True, "With Bias Correction"), (False, "Without Bias Correction")]:
    model = MLP([784, 128, 64, 10], seed=42)
    opt = AdamOptimizer(lr=0.001, use_bias_correction=use_bc)
    history = train_model(model, opt, X_train, Y_train, X_val, Y_val, n_epochs=5)
```

这个对比实验只在 5 个 epoch 上运行（偏差修正在早期最明显），对比：
- **有修正**：`m_hat = m / (1 - beta1^t)` → 早期步长正常
- **无修正**：直接用 `m` 和 `v` → 早期步长偏小，损失下降更慢

---

### 第10步：大学习率下的梯度裁剪演示

```python
# 故意用大学习率 lr=0.1 来制造梯度爆炸
opt_big_lr = AdamOptimizer(lr=0.1)
history_no_clip = train_model(..., clip_grad_norm=None)     # 无裁剪
history_with_clip = train_model(..., clip_grad_norm=1.0)   # 有裁剪
```

对比图表直观展示：无裁剪时梯度范数可能飙到几百甚至上千，训练不稳定；有裁剪时限死在 1.0，训练平稳运行。这证明了梯度裁剪在防止训练崩溃中的关键作用。

---

### 关键概念速查表

| 概念 | 数学公式 | 代码实现 |
|------|---------|---------|
| Adam 完整更新 | $\theta - \alpha \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$ | `param -= lr * m_hat / (np.sqrt(v_hat) + eps)` |
| 偏差修正 | $\hat{m}_t = m_t / (1 - \beta_1^t)$ | `m_hat = m / (1 - beta1 ** t)` |
| AdamW 权重衰减 | $\theta - \alpha \cdot \text{Adam} - \alpha \lambda \theta$ | 独立的一行 `param -= lr * wd * param` |
| Softmax + CE 梯度 | $\delta^{[L]} = A^{[L]} - Y$ | `dZ = caches[-1]["A"] - Y` |
| Warmup | $\alpha_t = \alpha_{\text{target}} \cdot t / t_w$ | `lr = base_lr * (step / warmup_steps)` |
| Cosine Decay | $\alpha_t = \alpha_{\min} + \frac{1}{2}(\alpha_{\max} - \alpha_{\min})(1 + \cos(\pi \cdot p))$ | `min_lr + 0.5 * (base - min) * (1 + cos(pi * progress))` |
| 梯度裁剪 | $\tilde{g} = g \cdot \min(1, \text{max\_norm} / \|g\|)$ | `scale = max_norm / total_norm` → `grad * scale` |
| 梯度范数 | $\|g\|_2 = \sqrt{\sum \|g_i\|^2}$ | `np.sqrt(sum(np.sum(g**2) for g in grads.values()))` |


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/nn-decision/dl/adam/code/demo.py`
