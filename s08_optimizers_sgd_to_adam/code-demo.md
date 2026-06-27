---
title: "s08 优化器：从SGD到Adam — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# s08 优化器：从SGD到Adam — demo.py 代码详解

<a href="../code/s08_optimizers_sgd_to_adam/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd s08_optimizers_sgd_to_adam/code
python demo.py
```

## 代码逐段详解

### 第1步：导入库 — 每个库是做什么的

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Callable
import os
```

- **`numpy`**：提供数组操作、随机数生成（`np.random.randn` 用于模拟梯度噪声）、数学运算（`np.sqrt`、`np.linalg.norm`）。
- **`matplotlib`**：绘制损失地形等高线图、优化器轨迹、损失曲线、超参数游乐场子图。
- **`typing`**：类型注解，标注函数签名中的参数和返回值类型。

---

### 第2步：损失地形 — 狭长峡谷形的二维二次型

```python
class LossLandscape:
    def __init__(self, a: float = 20.0, b: float = 1.0):
        self.a = a  # 陡峭方向曲率
        self.b = b  # 平缓方向曲率

    def __call__(self, theta: np.ndarray) -> float:
        theta1, theta2 = theta[0], theta[1]
        return 0.5 * (self.a * theta1 ** 2 + self.b * theta2 ** 2)

    def gradient(self, theta: np.ndarray) -> np.ndarray:
        theta1, theta2 = theta[0], theta[1]
        return np.array([self.a * theta1, self.b * theta2])
```

损失函数定义：

$$
L(\theta_1, \theta_2) = \frac{1}{2}(a \cdot \theta_1^2 + b \cdot \theta_2^2)
$$

梯度：

$$
\nabla L(\theta_1, \theta_2) = \begin{bmatrix} a \cdot \theta_1 \\ b \cdot \theta_2 \end{bmatrix}
$$

**条件数（Condition Number）** $\kappa = a / b = 20$ 决定了地形的"狭长度"。$\kappa \gg 1$ 意味着：
- $\theta_1$ 方向（系数 $a=20$）：**陡峭**，梯度 $= 20\theta_1$，稍微偏离原点就产生很大梯度
- $\theta_2$ 方向（系数 $b=1$）：**平缓**，梯度 $= \theta_2$，偏离较多才有较大梯度

全局最优解在原点 $(0, 0)$，最小损失 $= 0$。这个简单的二次型能清晰展示不同优化器在"狭长峡谷"中的表现差异。

---

### 第3步：SGD — 最朴素的优化器

```python
class SGDOptimizer:
    def __init__(self, lr: float = 0.02):
        self.lr = lr

    def step(self, theta: np.ndarray, grad: np.ndarray) -> np.ndarray:
        return theta - self.lr * grad
```

更新公式：

$$
\theta_{t+1} = \theta_t - \alpha \cdot g_t
$$

**特点**：不记忆任何历史信息。每一步只看当前梯度，直接往反方向走。这是最纯粹的梯度下降，也是所有改进的基准线（baseline）。

**SGD 存储开销**：0 个额外向量——只需要存储参数本身。

---

### 第4步：Momentum — 给优化器加"惯性"

```python
class MomentumOptimizer:
    def __init__(self, lr: float = 0.02, beta: float = 0.9):
        self.lr = lr
        self.beta = beta
        self.m = None  # 速度向量

    def step(self, theta: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(theta)  # m_0 = 0

        self.m = self.beta * self.m + (1 - self.beta) * grad
        return theta - self.lr * self.m
```

更新公式：

$$
m_t = \beta \cdot m_{t-1} + (1 - \beta) \cdot g_t
$$

$$
\theta_{t+1} = \theta_t - \alpha \cdot m_t
$$

**直觉**：$m_t$ 是梯度的**指数滑动平均（EMA）**。$\beta=0.9$ 意味着大约 90% 的权重来自历史梯度，10% 来自当前梯度。有效记忆长度约 $\frac{1}{1-\beta} \approx 10$ 步。

**Momentum 存储开销**：1 个额外向量（$m_t$，与参数同形）。相比 SGD 多了一倍的存储，但训练更平稳。

**为什么 `(1-beta)` 而非直接 `beta`**？这是"凸组合"的标准写法：确保所有权重之和为 1。展开后 $m_t = (1-\beta)(g_t + \beta g_{t-1} + \beta^2 g_{t-2} + \dots)$。

---

### 第5步：RMSProp — 给每个参数自适应步长

```python
class RMSPropOptimizer:
    def __init__(self, lr: float = 0.05, beta: float = 0.9, eps: float = 1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.v = None

    def step(self, theta: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self.v is None:
            self.v = np.zeros_like(theta)

        self.v = self.beta * self.v + (1 - self.beta) * (grad ** 2)
        return theta - self.lr * grad / (np.sqrt(self.v) + self.eps)
```

更新公式：

$$
v_t = \beta \cdot v_{t-1} + (1 - \beta) \cdot g_t \odot g_t
$$

$$
\theta_{t+1} = \theta_t - \alpha \cdot \frac{g_t}{\sqrt{v_t} + \epsilon}
$$

**直觉**：$v_t$ 是梯度**平方**的 EMA，估算每个参数的梯度方差。对于梯度大的参数（陡峭方向），$v_t$ 大 → 分母 $\sqrt{v_t}$ 大 → 有效步长自动变小。对于梯度小的参数（平缓方向），$v_t$ 小 → 分母小 → 有效步长相对更大。

**$\epsilon = 10^{-8}$ 的作用**：防止除以零。在训练的极早期，某些参数的 $v_t$ 可能仍接近 0，$\epsilon$ 提供了数值稳定性。

**RMSProp 存储开销**：1 个额外向量（$v_t$）。

---

### 第6步：Adam — Momentum + RMSProp + 偏差修正

```python
class AdamOptimizer:
    def __init__(self, lr: float = 0.1, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None   # 一阶矩
        self.v = None   # 二阶矩
        self.t = 0      # 迭代步数计数器

    def step(self, theta: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(theta)
            self.v = np.zeros_like(theta)

        self.t += 1

        # 一阶矩（Momentum 部分）
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        # 二阶矩（RMSProp 部分）
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)

        # 偏差修正
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # 参数更新
        return theta - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

**完整数学公式**：

一阶矩（方向）：
$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

二阶矩（尺度）：
$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t \odot g_t
$$

偏差修正：
$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

参数更新（Adam 核心公式）：
$$
\theta_{t+1} = \theta_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

**偏差修正为什么必要？** $m_t$ 和 $v_t$ 都从 0 初始化。在训练最初几步，它们的值系统性地偏小（如第一步 $m_1 = 0.1g_1$，只有真实梯度的 10%）。除以 $1-\beta_1^t$ 补偿了这个初始化偏差：第 1 步时 $1-0.9^1=0.1$，修正后 $\hat{m}_1 = m_1/0.1 = g_1$，完美补偿。

**Adam 存储开销**：2 个额外向量（$m_t$ 和 $v_t$），是四种优化器中存储需求最大的。对于大模型（如有 10 亿参数），这意味着额外需要约 8GB 显存（每个参数存储 2 个 float32 状态）。

---

### 第7步：运行优化器 — 记录完整轨迹

```python
def run_optimizer(optimizer, landscape, theta_init, n_steps=100,
                  add_noise=False, noise_std=0.0):
    theta = theta_init.copy()
    trajectory = [theta.copy()]
    losses = [landscape(theta)]

    for _ in range(n_steps):
        grad = landscape.gradient(theta)

        if add_noise:
            grad = grad + np.random.randn(*grad.shape) * noise_std

        theta = optimizer.step(theta, grad)
        trajectory.append(theta.copy())
        losses.append(landscape(theta))

    return np.array(trajectory), losses
```

`add_noise` 参数用于模拟 **mini-batch 梯度噪声**——真实训练中，我们只能用一个 mini-batch 估计梯度，估计值总是带有噪声。这个演示让你直观地看到：Adam 对噪声的鲁棒性远优于 SGD。

---

### 第8步：可视化 — 三种对比视角

#### 视角1：等高线图上的轨迹对比

```python
def plot_contour_comparison(landscape, all_trajectories, filename):
    # 生成等高线
    Z = 0.5 * (landscape.a * X**2 + landscape.b * Y**2)
    levels = np.logspace(-2, 2, 15)  # 对数间隔等高线
    ax.contour(X, Y, Z, levels=levels, cmap='Blues')
    ax.contourf(X, Y, Z, levels=levels, cmap='Blues', alpha=0.15)

    # 绘制每条优化器轨迹
    for name, traj in all_trajectories.items():
        ax.plot(traj[:, 0], traj[:, 1], '-', color=color, label=name)
        ax.plot(traj[0, 0], traj[0, 1], 'o')   # 起点
        ax.plot(traj[-1, 0], traj[-1, 1], 's')  # 终点
```

`np.logspace(-2, 2, 15)` 生成对数间隔的 15 个高度值——近距离处（接近原点）等高线密集，远处稀疏，配合狭长峡谷的损失地形。起点用小圆点标记，终点用方块标记，清晰展示每种优化器从 $(3.0, 3.0)$ 到达原点附近的路径。

#### 视角2：损失下降曲线

对数纵轴的折线图，直观对比收敛速度。Adam 的曲线通常下降最快，SGD 在陡峭方向上震荡导致损失下降缓慢。

#### 视角3：超参数游乐场

4 张子图分别对应 lr = 0.01, 0.05, 0.1, 0.5，每张绘制四种优化器的轨迹。展示**学习率对优化器的影响**：
- 小 lr = 0.01：所有优化器都收敛缓慢
- 大 lr = 0.5：SGD 剧烈震荡甚至发散，Adam 仍能稳定收敛

这直观展示了 Adam 对学习率的鲁棒性——在较宽的 lr 范围内都能稳定工作。

---

### 第9步：噪声鲁棒性对比

```python
noisy_sgd = SGDOptimizer(lr=0.02)
noisy_adam = AdamOptimizer(lr=0.1)

traj_sgd_noisy, loss_sgd_noisy = run_optimizer(
    noisy_sgd, landscape, theta_init, n_steps=100,
    add_noise=True, noise_std=1.0  # 标准差为 1.0 的高斯噪声
)
```

梯度中加入 `noise_std=1.0` 的噪声后，SGD 的路径剧烈抖动（每一步的梯度方向都受噪声影响），而 Adam 由于 $m_t$ 的平滑作用，路径相对平稳——这是 Adam 在实际训练中表现良好的关键原因之一。

---

### 关键概念速查表

| 优化器 | 核心记忆 | 更新公式 | 存储 | 解决的痛点 |
|--------|---------|---------|------|-----------|
| SGD | 无 | $\theta - \alpha g_t$ | 0 | —（基线） |
| Momentum | $m_t$（一阶矩） | $\theta - \alpha m_t$ | 1x | 方向抖动 |
| RMSProp | $v_t$（二阶矩） | $\theta - \alpha g_t / \sqrt{v_t}$ | 1x | 步长不统一 |
| Adam | $m_t$ + $v_t$ | $\theta - \alpha \hat{m}_t / \sqrt{\hat{v}_t}$ | 2x | 方向 + 步长 + 初始化偏差 |
| 条件数 | $\kappa = a/b$ | — | — | 衡量损失地形"狭长度" |
| 指数滑动平均 | $m_t = \beta m_{t-1} + (1-\beta)g_t$ | — | — | Adam 的基础运算 |

## 完整代码

<<< @/snippets/s08_optimizers_sgd_to_adam/demo.py
