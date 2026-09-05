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
- **`numpy`**：数值计算核心。关键用法：`np.exp()` 指数函数（Sigmoid 和 Softmax 的核心），`np.clip()` 裁剪数值范围防止溢出，`np.log()` 对数函数（交叉熵损失），`np.argmax()` 取最大值索引，`np.unique()` 统计类别数
- **`matplotlib`**：绘图，包括函数曲线、散点图、决策边界热力图、等高线图
- **`load_iris`**（sklearn）：加载经典的 Iris（鸢尾花）数据集——150 个样本，4 个特征，3 个类别
- **`train_test_split`**（sklearn）：按比例（80/20）随机划分训练集和测试集
- **`sklearn.linear_model.LogisticRegression`**：sklearn 的实现，用作基准对比

### 第2步：Sigmoid 函数 — 从实数到概率

```python
def sigmoid(z: np.ndarray) -> np.ndarray:
    z_clipped = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z_clipped))
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

**数值稳定技巧**：`np.clip(z, -500, 500)` 将 $z$ 限制在 $[-500, 500]$ 内。当 $z > 500$ 时 $e^{-z} \approx 0$（下溢出），当 $z < -500$ 时 $e^{-z} \approx \infty$（上溢出）。裁剪确保计算稳定，而 $[-500, 500]$ 已经足够覆盖所有实际应用场景（$\sigma(500) \approx 1$，$\sigma(-500) \approx 0$）。

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
2. MSE + Sigmoid 在参数空间中是**非凸**的，存在多个局部最小值

解决方案是**二元交叉熵（Binary Cross-Entropy）**：

$$
J(w, b) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
$$

这个公式设计得非常精妙：当 $y=1$ 时只有第一项起作用（$-\log(\hat{y})$），预测概率越高损失越小；当 $y=0$ 时只有第二项起作用（$-\log(1-\hat{y})$），预测概率越低损失越小。

代码实现：

```python
def _compute_loss(self, X, y):
    y_pred = self._predict_proba(X)
    n = len(y)
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)  # 防止 log(0)
    loss = -(1.0 / n) * np.sum(
        y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)
    )
    return loss
```

`np.clip(y_pred, eps, 1 - eps)` 将预测概率限制在 $[10^{-15}, 1 - 10^{-15}]$ 之间，防止 $\log(0) = -\infty$ 导致数值崩溃。$10^{-15}$ 已经足够小，对实际损失值几乎没有影响。

#### 4.3 梯度：数学之美的巅峰

Sigmoid + 交叉熵组合的梯度有一个惊人简洁的形式。经过数学推导（链式法则 + Sigmoid 导数性质）：

$$
\frac{\partial J}{\partial z} = \hat{y} - y
$$

这被称为"黄金组合"——损失对原始得分 $z$ 的梯度就等于预测误差！利用链式法则传播到参数：

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

`X.T @ errors` 是矩阵乘法：$X^T_{(d \times n)} \cdot \text{errors}_{(n)}$，每行是将所有样本的误差按对应特征加权求和。这个梯度形式与线性回归 + MSE 的梯度**完全一致**——唯一的区别在于 $\hat{y}$ 的计算方式不同（逻辑回归中多了 Sigmoid）。

#### 4.4 训练循环

```python
def fit(self, X, y, verbose=True):
    self.w = np.random.randn(n_features) * 0.01
    self.b = 0.0
    for epoch in range(self.max_epochs):
        loss = self._compute_loss(X, y)       # 前向 + 损失
        dw, db = self._compute_gradients(X, y) # 反向传播
        self.w -= self.learning_rate * dw       # 参数更新
        self.b -= self.learning_rate * db
```

与线性回归的训练循环结构完全相同——这揭示了机器学习的一个统一框架：
1. **前向传播**：计算 $\hat{y} = f_\theta(x)$
2. **计算损失**：$J(\hat{y}, y)$
3. **反向传播**：求 $\nabla_\theta J$
4. **更新参数**：$\theta \leftarrow \theta - \eta \nabla_\theta J$

不同模型只是换用了不同的 $f_\theta$ 和 $J$，但四步循环的结构不变。

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
    proba = self._predict_proba(X)          # (n, K)
    proba = np.clip(proba, eps, 1 - eps)
    loss = -(1.0 / n) * np.sum(np.log(proba[np.arange(n), y]))
    return loss
```

`proba[np.arange(n), y]` 是高级索引（fancy indexing）：对每个样本 $i$，取第 $i$ 行、第 $y_i$ 列的概率值——即模型对该样本正确类别的预测概率。

**One-hot 编码**在梯度计算中扮演关键角色：

```python
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
| 黄金梯度 | $\partial J/\partial z = \hat{y} - y$ | `_compute_gradients()` | Sigmoid导数被约掉 |
| 决策边界 | $w^T x + b = 0$ | 可视化 | $\sigma = 0.5$ 的等概率线 |
| One-hot 编码 | 正确类别=1，其余=0 | `_compute_gradients()` | 多分类梯度的关键 |
| 混淆矩阵 | TP/TN/FP/FN | `main()` | 分类问题精细化评估 |


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/ml/foundations/logistic-regression/code/demo.py`
