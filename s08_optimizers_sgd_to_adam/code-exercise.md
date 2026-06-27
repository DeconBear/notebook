---
title: "s08 优化器：从SGD到Adam — exercise.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# s08 优化器：从SGD到Adam — exercise.py 练习指南

<a href="../code/s08_optimizers_sgd_to_adam/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

亲手实现 Momentum、RMSProp、NAG（Nesterov 加速梯度）三种优化器的更新规则，理解每种优化器"从 SGD 出发，一步步解决了什么问题"的设计演进脉络。

## 预备知识

建议先阅读 index.md 并运行 demo.py，确保理解：

| 概念 | 数学定义 |
|------|---------|
| SGD | $\theta_{t+1} = \theta_t - \alpha g_t$ |
| 指数滑动平均（EMA）| $m_t = \beta m_{t-1} + (1 - \beta) g_t$ |
| Momentum | $m_t$ 定方向，$\theta_{t+1} = \theta_t - \alpha m_t$ |
| RMSProp | $v_t$ 缩步长，$\theta_{t+1} = \theta_t - \alpha g_t / \sqrt{v_t + \epsilon}$ |
| 狭长峡谷 | 损失地形某些方向陡峭、某些平缓，条件数 $\kappa \gg 1$ |

---

## 任务清单

### 任务1：实现 Momentum 更新规则

**描述**：补全 `MomentumOptimizerExercise.step()` 方法的三个 TODO——初始化速度向量、更新速度、更新参数。

**公式回顾**：

$$
m_t = \beta \cdot m_{t-1} + (1 - \beta) \cdot g_t
$$

$$
\theta_{t+1} = \theta_t - \alpha \cdot m_t
$$

**提示**：
- 首次调用时 `self.m is None`，需要初始化为 `np.zeros_like(theta)`
- `self.m = self.beta * self.m + (1 - self.beta) * grad` 更新速度
- `return theta - self.lr * self.m` 更新参数
- 注意 `(1 - self.beta)` 不能省略——如果不乘，就变成了对梯度做纯滑动平均而没有衰减

**期望行为**：
- 从 $(3.0, 2.5)$ 出发，经过 5 步后动量向量应指向原点附近
- 每一步的动量 $m_t$ 是历史梯度的加权平均，不会像 SGD 那样每步方向剧烈变化

**为什么 Momentum 比 SGD 好？** 在狭长峡谷中，SGD 在陡峭方向上来回震荡（一步正向一步反向），而 Momentum 通过平滑历史方向，抵消了这种震荡——正负梯度互相抵消，让 $m_t$ 在震荡方向上接近 0。

---

### 任务2：实现 RMSProp 自适应步长

**描述**：补全 `RMSPropOptimizerExercise.step()` 方法的三个 TODO——初始化 $v$、更新梯度平方的 EMA、自适应步长更新。

**公式回顾**：

$$
v_t = \beta \cdot v_{t-1} + (1 - \beta) \cdot g_t \odot g_t
$$

$$
\theta_{t+1} = \theta_t - \alpha \cdot \frac{g_t}{\sqrt{v_t} + \epsilon}
$$

**提示**：
- `grad ** 2` 是逐元素平方，对应公式中的 $g_t \odot g_t$
- `np.sqrt(self.v)` 计算 $\sqrt{v_t}$
- `self.eps`（通常 $10^{-8}$）加到分母上防止除零
- 有效学习率 $= \alpha / (\sqrt{v_t} + \epsilon)$ —— 可以打印出来观察

**期望行为**：
- 陡峭方向（$\theta_1$，系数 $a=20$）的有效学习率应该**小于**平缓方向（$\theta_2$，系数 $b=1$）
- 这是因为 $\theta_1$ 方向梯度大 → $v_t$ 增长快 → 分母大 → 有效步长自动变小
- 最终输出应显示 `θ₁ < θ₂: True`

**为什么 RMSProp 解决了步长不统一问题？** SGD 对所有参数用同一个学习率，但不同参数梯度的尺度可能差几个数量级。RMSProp 让每个参数拥有自己的"自适应学习率"——梯度历史大的参数步长自动变小，历史小的步长自动放大。

---

### 任务3：实现 Nesterov 加速梯度 (NAG)

**描述**：补全 `NAGOptimizer.step()` 方法的五个 TODO。NAG 是 Momentum 的进阶版——它"先沿动量方向看一步，再在那个位置计算梯度"。

**与普通 Momentum 的区别**：
- **Momentum**：在当前点 $\theta_t$ 计算梯度，然后沿 $m_t$ 方向更新
- **NAG**：先沿 $m_{t-1}$ 方向走一步到"前瞻位置"，在那个位置计算梯度，再更新

**公式**：

前瞻位置：
$$
\theta_{\text{lookahead}} = \theta_t - \alpha \cdot \beta \cdot m_{t-1}
$$

在"前瞻"位置计算梯度并更新动量：
$$
m_t = \beta \cdot m_{t-1} + (1 - \beta) \cdot \nabla L(\theta_{\text{lookahead}})
$$

参数更新：
$$
\theta_{t+1} = \theta_t - \alpha \cdot m_t
$$

**提示**：
- 注意：NAG 的 `step()` 接收的是 `grad_fn`（一个函数），而不是 `grad`（一个值）！因为需要在 `lookahead` 位置重新计算梯度
- `theta_lookahead = theta - self.lr * self.beta * self.m`
- `grad_lookahead = grad_fn(theta_lookahead)` —— 在"前瞻"位置调梯度函数
- 动量更新与 Momentum 相同，只是用的是 `grad_lookahead` 而非当前位置的梯度

**直觉理解**：NAG 像是一个"会思考的滚球"。普通 Momentum 是"盲人下山"——沿着惯性方向走，走到哪算哪。NAG 是"睁眼下山"——先往前看一眼，发现前面是悬崖（梯度变大），就可以提前调整方向。在数学上这对应更紧的收敛界。

**期望行为**：从 $(3.0, 2.5)$ 出发，经过 10 步后，NAG 的损失通常小于普通 Momentum——因为它更"聪明"地选择了更新方向。

---

### 优化器演进总览

| 优化器 | 改进点 | 要记住的关键行 |
|--------|--------|--------------|
| SGD | 基线 | `theta - lr * grad` |
| Momentum | 加惯性，平滑方向 | `m = beta*m + (1-beta)*grad` |
| RMSProp | 自适应步长 | `v = beta*v + (1-beta)*(grad**2)` |
| Adam | 方向 + 步长 + 修正 | `m` + `v` + 偏差修正 |
| NAG | "会思考的"Momentum | 先在 lookahead 处算梯度 |

## 完整代码

<<< @/snippets/s08_optimizers_sgd_to_adam/exercise.py
