---
title: "s03 逻辑回归 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# s03 逻辑回归 — demo.py 代码详解

<a href="/notebook/code/ml/foundations/logistic-regression/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/ml/foundations/logistic-regression/code
python demo.py
```

## 代码逐段详解

### 第1步：导入库 — 每个库是做什么的

```python
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as SklearnLR
```

- **`os`**：操作文件路径，创建 `images/` 目录
- **`numpy`**：数值计算核心。关键用法：`np.exp()` 指数函数（Sigmoid 和 Softmax 的核心），`np.logaddexp()` 稳定计算 log-sum-exp，`np.log()` 对数函数（交叉熵损失），`np.argmax()` 取最大值索引，`np.unique()` 统计类别数
- **`matplotlib`**：绘图，包括函数曲线、散点图、决策边界热力图、等高线图
- **`load_iris`**（sklearn）：加载经典的 Iris（鸢尾花）数据集——150 个样本，4 个特征，3 个类别
- **`train_test_split`**（sklearn）：按比例（80/20）随机划分训练集和测试集
- **`sklearn.linear_model.LogisticRegression`**：sklearn 的实现，用作基准对比

### 第2步：Sigmoid 函数 — 从实数到概率

```python
def sigmoid(z: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -np.asarray(z, dtype=float)))
```

Sigmoid 函数的数学形式：

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

**Sigmoid 的核心性质**：
- **值域**：$\sigma(z) \in (0, 1)$，天然适合解释为概率。当 $z \to +\infty$ 时 $\sigma \to 1$，当 $z \to -\infty$ 时 $\sigma \to 0$
- **对称性**：$\sigma(-z) = 1 - \sigma(z)$，得分取反，概率互补
- **导数**：$\sigma'(z) = \sigma(z)(1 - \sigma(z))$，可以用自身表达——这让反向传播计算异常简便
- **中心点**：$\sigma(0) = 0.5$，对应决策边界 $w \cdot x + b = 0$

**数值稳定技巧**：令 $\operatorname{softplus}(u)=\log(1+e^u)$，则 $\sigma(z)=\exp[-\operatorname{softplus}(-z)]$。`np.logaddexp(0, -z)` 稳定计算 softplus，不需要先算可能溢出的 $e^{-z}$，也不需要裁剪 logit。数学上有限 $z$ 的输出严格在 $(0,1)$ 内；浮点数中极端输入仍可能舍入为 0 或 1，因此**计算损失不能再对概率直接取对数**。该实现针对有限实数输入；NaN、无穷值或矩阵乘法本身已溢出的输入不在保证范围内。

### 第3步：Softmax 函数 — 从得分到概率分布

```python
def softmax(z: np.ndarray) -> np.ndarray:
    z_stable = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_stable)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
```

Softmax 将 $K$ 个原始得分 $[z_1, \dots, z_K]$ 转换为概率分布：

$$
\text{softmax}(z_k) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}
$$

**三个关键性质**：
1. **非负性**：每个输出 $\in (0, 1)$
2. **归一性**：所有输出之和 $= 1$（构成一个合法的概率分布）
3. **保序性**：如果 $z_i > z_j$，则 $\text{softmax}(z_i) > \text{softmax}(z_j)$

**数值稳定技巧**：`z - np.max(z, axis=1, keepdims=True)` 在每行（每个样本）减去该行的最大值。这等价于分子分母同除 $e^{\max(z)}$，不改变概率值但避免了 $e^{z}$ 上溢出。`axis=1` 表示沿列方向（对每个样本的所有类别做 max），`keepdims=True` 保持维度以便广播。

### 第4步：二分类逻辑回归模型

#### 4.1 模型定义

逻辑回归 = 线性模型 + Sigmoid：

$$
P(y=1 \mid x) = \hat{y} = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}
$$

这个值被解释为"给定输入 $x$，样本属于正类的概率"。逻辑回归本质上就是一个**没有隐藏层的神经网络**。

**预测概率** `_predict_proba()`：

```python
def _predict_proba(self, X):
    z = X @ self.w + self.b    # 线性组合 (n,)
    return sigmoid(z)           # 通过 Sigmoid 得到概率 (n,)
```

`X @ self.w` 是矩阵乘法 $X_{n \times d} \cdot w_d$，结果是 $n$ 个得分值。然后逐元素通过 Sigmoid 转化为概率。

**类别预测** `predict()`：

```python
def predict(self, X, threshold=0.5):
    proba = self._predict_proba(X)
    return (proba >= threshold).astype(int)
```

当预测概率 $\geq 0.5$（即 $w^T x + b \geq 0$）时判为正类（1），否则判为负类（0）。

#### 4.2 损失函数：二元交叉熵

为什么不继续用 MSE？两个致命问题：
1. Sigmoid 的饱和特性导致梯度消失（当预测值接近 0 或 1 且标签相反时，MSE 的梯度极小）
2. MSE + Sigmoid 关于参数一般**不保证凸性**，不能沿用凸优化的全局最优保证；非凸并不意味着每个数据集都必然存在多个局部最小值

解决方案是**二元交叉熵（Binary Cross-Entropy）**：

$$
J(w, b) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
$$

这个公式设计得非常精妙：当 $y=1$ 时只有第一项起作用（$-\log(\hat{y})$），预测概率越高损失越小；当 $y=0$ 时只有第二项起作用（$-\log(1-\hat{y})$），预测概率越低损失越小。

代码实现：

```python
def _compute_loss(self, X, y):
    z = X @ self.w + self.b
    return np.mean(np.logaddexp(0.0, (1.0 - 2.0 * y) * z))
```

这里要求二分类标签 $y\in\{0,1\}$。从 $\log\sigma(z)=-\operatorname{softplus}(-z)$ 和 $\log(1-\sigma(z))=-\operatorname{softplus}(z)$ 得到单样本损失：

$$
\ell(z,y)=y\operatorname{softplus}(-z)+(1-y)\operatorname{softplus}(z)
=\operatorname{softplus}((1-2y)z).
$$

最后一个等号只适用于硬标签 0/1；软标签应使用前面的加权形式。按标签选择 softplus 的符号，也避免了 `softplus(z) - y*z` 在大正数且 $y=1$ 时相减丢失小损失。

**为什么不用概率裁剪？** 对 $z=1000,y=0$，真实损失约为 1000；把预测概率裁剪至 $1-10^{-15}$ 后，损失却只有约 34.54。裁剪会把极端错误预测的损失截平，且裁剪区间外的实际导数不再是后文使用的 $\hat y-y$。稳定 logit 公式保留了目标函数，也让损失与梯度一致。

#### 4.3 梯度：数学之美的巅峰

Sigmoid + 交叉熵组合的梯度有一个惊人简洁的形式。经过数学推导（链式法则 + Sigmoid 导数性质）：

$$
\frac{\partial \ell}{\partial z} = \hat{y} - y
$$

这里的 $\ell$ 是**单样本损失**：由 $\ell=\operatorname{softplus}(z)-yz$ 对 $z$ 求导即得 $\sigma(z)-y$。对平均损失 $J=\frac1n\sum_i\ell_i$，则 $\partial J/\partial z_i=(\hat y_i-y_i)/n$，不能漏掉 $1/n$。利用链式法则传播到参数：

$$
\frac{\partial J}{\partial w} = \frac{1}{n} X^T (\hat{y} - y)
$$

$$
\frac{\partial J}{\partial b} = \frac{1}{n} \sum (\hat{y}_i - y_i)
$$

代码实现：

```python
def _compute_gradients(self, X, y):
    y_pred = self._predict_proba(X)
    n = len(y)
    errors = y_pred - y                           # (n,) 预测误差
    dw = (1.0 / n) * (X.T @ errors)               # (d,) 权重梯度
    db = (1.0 / n) * np.sum(errors)               # 标量 偏置梯度
    return dw, db
```

`X.T @ errors` 是矩阵乘法：$X^T_{(d \times n)} \cdot \text{errors}_{(n)}$，每行是将所有样本的误差按对应特征加权求和。它与线性回归使用 $\frac{1}{2n}\sum_i(\hat y_i-y_i)^2$ 时的梯度形式一致；若 MSE 定义为 $\frac1n\sum_i(\hat y_i-y_i)^2$，则梯度多一个因子 2。逻辑回归的 $\hat y$ 还包含 Sigmoid。

#### 4.4 训练循环

```python
def fit(self, X, y, verbose=True):
    n_samples, n_features = X.shape
    self.w = np.random.randn(n_features) * 0.01
    self.b = 0.0
    for epoch in range(self.max_epochs):
        loss = self._compute_loss(X, y)       # 前向 + 损失
        dw, db = self._compute_gradients(X, y) # 反向传播
        self.w -= self.learning_rate * dw       # 参数更新
        self.b -= self.learning_rate * db
```

这是只展示核心步骤的节选；完整源码还记录更新前损失、检查收敛，并在结束时重算最终参数的损失。与线性回归的训练循环结构相同——这揭示了机器学习的一个统一框架：
1. **前向传播**：计算 $\hat{y} = f_\theta(x)$
2. **计算损失**：$J(\hat{y}, y)$
3. **反向传播**：求 $\nabla_\theta J$
4. **更新参数**：$\theta \leftarrow \theta - \eta \nabla_\theta J$

不同模型只是换用了不同的 $f_\theta$ 和 $J$，但四步循环的结构不变。

完整 Iris 演示固定随机种子 42，并使用学习率 0.04。原来的 0.5 会让未标准化特征上的 Softmax 训练振荡；步长上界与推导见[正文实战案例](./index.md)。损失曲线记录每轮更新前的值，末尾打印值则重新评价最终参数，两者并非同一个时刻。

### 第5步：多分类 Softmax 回归

二分类 $\to$ 多分类的推广：

- **权重**：从向量 $w_{(d,)}$ 变为矩阵 $W_{(d \times K)}$，每个类别有自己的权重向量
- **偏置**：从标量 $b$ 变为向量 $b_{(K,)}$，每个类别有自己的偏置
- **激活**：从 Sigmoid 变为 Softmax
- **损失**：从二元交叉熵变为多分类交叉熵

多分类交叉熵损失：

$$
J = -\frac{1}{n} \sum_{i=1}^{n} \sum_{k=1}^{K} y_{ik} \log(\hat{y}_{ik})
$$

由于真实标签是 one-hot 编码（只有正确类别的位置为 1，其余为 0），实际计算时只需取正确类别位置的 $\log$ 概率：

```python
def _compute_loss(self, X, y):
    z = X @ self.W + self.b
    z_stable = z - np.max(z, axis=1, keepdims=True)
    log_normalizer = np.log(np.sum(np.exp(z_stable), axis=1))
    return np.mean(log_normalizer - z_stable[np.arange(len(y)), y])
```

对每行令 $m_i=\max_k z_{ik}$、$s_{ik}=z_{ik}-m_i$，则

$$
-\log p_{i,y_i}=\log\sum_k e^{s_{ik}}-s_{i,y_i}.
$$

每行至少有一个 $s_{ik}=0$，所以指数和至少为 1，且每项不超过 1。直接计算 log-softmax 损失避免了先 Softmax 下溢为 0、再取对数或裁剪的错误。比如 logits 为 $[1000,-1000]$、真类为 1 时，损失约为 2000，而不是裁剪后的约 34.54。`z_stable[np.arange(len(y)), y]` 为每个样本选出真类得分。

本例要求有限且可表示的 logits，标签是连续的整数 $0,\ldots,K-1$；若使用字符串或不连续标签，应先编码。稳定变换不能修复输入已包含 NaN/无穷值，或极端数值相减本身溢出的情况。

**One-hot 编码**在梯度计算中扮演关键角色：

```python
n = len(y)
proba = self._predict_proba(X)
y_onehot = np.zeros((n, self.n_classes))
y_onehot[np.arange(n), y] = 1
errors = proba - y_onehot
```

`y_onehot` 中，每行只有一个位置为 1（真实类别），其余为 0。`errors = proba - y_onehot` 的结果是：正确类别位置为 $\hat{y}_k - 1$（负值），其他位置为 $\hat{y}_k$（正值）。梯度方向就是"增大正确类别的概率，减小错误类别的概率"。

### 第6步：可视化

#### 6.1 Sigmoid 函数曲线

在 $[-8, 8]$ 区间绘制 Sigmoid 曲线，关键标注：
- 红色虚线标注 $z=0$ 和 $\sigma=0.5$——这是决策边界的位置
- 灰色虚线标注渐近线 $y=0$ 和 $y=1$——Sigmoid 的值域下界和上界
- 绿色区域标注 $z>0$ 的"正类区域"，红色区域标注 $z<0$ 的"负类区域"

#### 6.2 决策边界与概率热力图

这是逻辑回归最直观的可视化。代码使用 `np.meshgrid()` 在二维平面上创建 $300 \times 300$ 的网格，对每个网格点计算 $P(y=1 \mid x)$，然后用 `contourf()` 填充颜色：

- **蓝色区域**：模型输出低概率，倾向于预测负类
- **红色区域**：模型输出高概率，倾向于预测正类
- **绿色轮廓线**：$\sigma = 0.5$ 的等概率线，即决策边界 $w^T x + b = 0$

颜色渐变展示了模型的"置信度"——远离决策边界的区域颜色更深（更确信），决策边界附近的区域颜色较浅（不确定）。

### 第7步：模型评估 — 混淆矩阵

对于分类问题，仅看准确率可能不够。代码计算了**混淆矩阵（Confusion Matrix）**的四个元素：

|               | 预测正类 (1) | 预测负类 (0) |
|---------------|:------------:|:------------:|
| **真实正类 (1)** | TP (真正例)   | FN (假负例)   |
| **真实负类 (0)** | FP (假正例)   | TN (真负例)   |

- **准确率**：$\frac{TP + TN}{TP + TN + FP + FN}$，所有预测中正确的比例
- 从混淆矩阵可以进一步计算精确率 $P = \frac{TP}{TP+FP}$、召回率 $R = \frac{TP}{TP+FN}$、F1 分数等更细致的指标

## 关键概念速查表

| 概念 | 数学形式 | 代码位置 | 关键说明 |
|------|---------|---------|---------|
| Sigmoid | $\sigma(z) = \frac{1}{1+e^{-z}}$ | `sigmoid()` | 实数→(0,1)概率映射 |
| Softmax | $\frac{e^{z_k}}{\sum e^{z_j}}$ | `softmax()` | 得分→概率分布 |
| 交叉熵 (二元) | $-[y\log\hat{y} + (1-y)\log(1-\hat{y})]$ | `_compute_loss()` | 分类问题的标准损失 |
| 黄金梯度 | $\partial \ell/\partial z = \hat{y} - y$ | `_compute_gradients()` | Sigmoid导数被约掉 |
| 决策边界 | $w^T x + b = 0$ | 可视化 | $\sigma = 0.5$ 的等概率线 |
| One-hot 编码 | 正确类别=1，其余=0 | `_compute_gradients()` | 多分类梯度的关键 |
| 混淆矩阵 | TP/TN/FP/FN | `main()` | 分类问题精细化评估 |


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/ml/foundations/logistic-regression/code/demo.py`
