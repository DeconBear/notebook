---
title: "s02 线性回归 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# s02 线性回归 — demo.py 代码详解

<a href="/notebook/code/ml/foundations/linear-regression/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/ml/foundations/linear-regression/code
python demo.py
```

## 代码逐段详解

### 第1步：导入库 — 每个库是做什么的

```python
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression as SklearnLR
```

- **`os`**：操作文件路径，用于创建 `images/` 目录
- **`numpy`**：数值计算核心。关键用法：
  - `np.random.uniform()` 生成均匀分布的随机数
  - `np.mean()` 计算平均值（MSE 损失的核心）
  - `np.sum()` 求和（梯度计算中累加误差）
  - `np.linalg.inv()` 矩阵求逆（正规方程求解）
  - `np.column_stack()` 按列拼接（为正规方程构建增广矩阵）
  - `np.meshgrid()` 创建网格（绘制损失函数等高线）
- **`matplotlib`**：绘图，包括 2D 散点/线图、3D 曲面、等高线图
- **`Axes3D`**（来自 `mpl_toolkits.mplot3d`）：支持 3D 绘图，用于展示损失函数的 3D 曲面
- **`sklearn.linear_model.LinearRegression`**：scikit-learn 的标准线性回归实现，用作基准对比，验证我们从头实现的正确性

### 第2步：数据生成

```python
def generate_regression_data(n_samples=100, noise_std=3.0, true_w=2.0, true_b=5.0, random_seed=42):
```

生成回归任务的合成数据。真实函数为：

$$
y = 2x + 5 + \epsilon, \quad \epsilon \sim \mathcal{N}(0, 3^2)
$$

具体步骤：
1. `np.random.uniform(0, 10, n_samples)` 在 $[0, 10]$ 区间均匀采样 100 个 $x$ 值
2. `np.random.randn(n_samples) * noise_std` 生成标准差为 3.0 的高斯噪声
3. 按 $y = 2x + 5 + \text{noise}$ 计算目标值

噪声标准差 3.0 相对较大（真实值范围约 $[5, 25]$），这使得数据点较为分散，更接近真实场景中的数据。

### 第3步：线性回归模型 — 梯度下降法

#### 3.1 模型假设

线性回归假设输出是输入的线性函数：

$$
\hat{y} = w x + b
$$

其中 $w$ 是权重（斜率），$b$ 是偏置（截距）。这两个参数就是模型需要从数据中"学习"的内容。

#### 3.2 损失函数：均方误差 MSE

$$
J(w, b) = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2
$$

**为什么用平方而不是绝对值？** 四个关键原因：
1. **处处可导**：平方函数光滑，绝对值在 $x=0$ 处不可导——梯度下降需要处处有梯度
2. **对大误差更敏感**：误差为 10 的样本，MSE 中惩罚 100，MAE 中仅惩罚 10——这鼓励模型优先修正大偏差
3. **概率解释**：若假设误差服从正态分布 $\epsilon \sim \mathcal{N}(0, \sigma^2)$，最小化 MSE 等价于最大似然估计（MLE）
4. **凸函数**：MSE 关于 $(w, b)$ 是凸函数，只有一个全局最小值，不会被卡在局部最优

代码实现：

```python
def _compute_loss(self, X, y):
    y_pred = self.predict(X)            # 计算所有预测值
    return np.mean((y_pred - y) ** 2)   # 平均平方误差
```

`np.mean((y_pred - y) ** 2)` 是向量化操作：先逐元素计算平方差，再取平均。一条语句完成全部 $n$ 个样本的 MSE 计算。

#### 3.3 梯度推导与计算

MSE 损失对 $w$ 和 $b$ 的偏导数（通过链式法则推导）：

$$
\frac{\partial J}{\partial w} = \frac{2}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i) \cdot x_i
$$

$$
\frac{\partial J}{\partial b} = \frac{2}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)
$$

代码实现：

```python
def _compute_gradients(self, X, y):
    y_pred = self.predict(X)          # 计算预测值 (n,)
    n = len(y)
    errors = y_pred - y               # 误差向量 (n,)
    dw = (2.0 / n) * np.sum(errors * X)  # ∂J/∂w: 逐元素乘后求和
    db = (2.0 / n) * np.sum(errors)      # ∂J/∂b: 误差求和
    return dw, db
```

`errors * X` 是逐元素乘法：对每个样本 $i$，计算 $(\hat{y}_i - y_i) \cdot x_i$，然后用 `np.sum()` 累加。系数 $2/n$ 中，因子 2 来自平方函数的导数，$1/n$ 来自平均。

#### 3.4 梯度下降更新规则

$$
w \leftarrow w - \eta \cdot \frac{\partial J}{\partial w}
$$

$$
b \leftarrow b - \eta \cdot \frac{\partial J}{\partial b}
$$

代码：

```python
self.w -= self.learning_rate * dw
self.b -= self.learning_rate * db
```

注意这里是 `-=`（减等于），因为我们沿梯度的**反方向**走——梯度指向函数值上升最快的方向，我们要下降所以要反向。

**收敛条件**：当相邻两轮的损失变化小于 `tolerance`（默认 $10^{-6}$）时，认为已收敛，提前终止训练。

**参数初始化**：

```python
self.w = np.random.randn() * 0.1
self.b = np.random.randn() * 0.1
```

用标准正态分布的小随机数初始化。为什么不是全零？对于单变量线性回归，全零初始化也可以工作（因为只有一个 $w$，不存在对称性问题），但用随机初始化是更好的通用实践。乘 0.1 是为了让初始值接近 0 但不完全为 0。

### 第4步：正规方程 — 线性回归的解析解

对于线性回归，我们不仅能梯度下降，还能直接求出闭式解。这是少数能写出解析解的机器学习模型之一。

**推导**：将 MSE 写成矩阵形式 $J(\theta) = \frac{1}{n} \|X\theta - y\|^2$，令 $\nabla_\theta J = 0$，解得：

$$
\theta^* = (X^T X)^{-1} X^T y
$$

这就是**正规方程（Normal Equation）**。

代码实现：

```python
def normal_equation_solution(X, y):
    n = len(X)
    # 构建增广矩阵: X_aug = [x, 1]，第一列是x，第二列是全1（对应偏置）
    X_aug = np.column_stack([X, np.ones(n)])
    # θ = (X^T X)^{-1} X^T y
    theta = np.linalg.inv(X_aug.T @ X_aug) @ X_aug.T @ y
    w = theta[0]  # 权重
    b = theta[1]  # 偏置
    return w, b
```

关键细节：
- `np.column_stack([X, np.ones(n)])` 构建增广矩阵 $[x, \mathbf{1}]$，形状 $(n, 2)$。第二列全 1 对应偏置项——这样 $\theta = [w, b]^T$ 就能用一个矩阵方程求解。
- `@` 是 Python 3.5+ 的矩阵乘法运算符，等价于 `np.matmul()`。`X_aug.T @ X_aug` 计算 $X^T X$，形状 $(2, 2)$。
- `np.linalg.inv()` 计算矩阵的逆。对于 $2 \times 2$ 矩阵，求逆非常快。

**正规方程 vs 梯度下降**：
- 正规方程：一步到位得到精确解，无需选择学习率，无需迭代。但 $X^T X$ 求逆的复杂度是 $O(d^3)$，当特征维度 $d$ 很大时不可行。
- 梯度下降：需要选择学习率，需要多轮迭代，但复杂度是 $O(nd)$ 每轮，适合大规模数据和深度学习。

### 第5步：可视化 — 四合一分析图

代码生成了一个 $2 \times 2$ 的子图布局：

**子图 1：数据散点和拟合直线**

三条直线分别来自梯度下降法（红色实线）、正规方程（绿色虚线）和 sklearn（蓝色点划线）。理想情况下三条线几乎重合——三种方法给出的 $w, b$ 非常接近，互相验证了实现的正确性。

**子图 2：训练损失曲线**

横轴是 epoch，纵轴是 MSE 损失（对数刻度）。损失通常在前几十轮快速下降，之后趋于平稳。对数刻度让早期快速下降和后期精细调整都能看清楚。

**子图 3：损失函数等高线 + 梯度下降轨迹**

这是在 $(w, b)$ 参数空间中绘制 MSE 的等高线图。同心椭圆（或近似椭圆）是等损失线——同一椭圆上的 $(w, b)$ 组合给出相同的 MSE。

红色轨迹是梯度下降的优化路径：从蓝色起点出发，沿着局部梯度方向一步步走向红色星形的最优点。轨迹垂直于等高线（因为梯度方向垂直于等高线），且步长越来越小（接近最优点时梯度趋近于零）。

生成等高线用的 `np.meshgrid()` 在 $(w, b)$ 平面上创建了 $100 \times 100$ 的网格，对每个网格点计算 MSE。损失函数的等高线呈椭圆形状，说明 $w$ 和 $b$ 的最优值之间存在一定的耦合关系。

**子图 4：三种方法的参数对比**

柱状图直观比较梯度下降、正规方程和 sklearn 三种方法得到 $w$ 和 $b$ 值。灰色虚线标注了真实值 $w=2.0$ 和 $b=5.0$。

### 第6步：学习率对比实验

`compare_learning_rates()` 函数用三种不同的学习率（0.001、0.01、0.05）训练模型，在一张图上对比它们的损失曲线：

- **$\eta = 0.001$**：收敛最慢，200 轮后损失仍较高
- **$\eta = 0.01$**：适中，在大约 300 轮时收敛到较小的损失值
- **$\eta = 0.05$**：最快收敛，但如果太大可能震荡或发散

这个实验直观展示了学习率作为"步长"的含义：步长太小，下山太慢；步长适中，高效到达谷底；步长太大，可能在谷底来回跳跃甚至发散。

### 第7步：模型评估 — MSE 和 $R^2$

在 `main()` 中，通过计算 MSE 和 $R^2$ 来评估模型：

- **MSE（均方误差）**：$\text{MSE} = \frac{1}{n} \sum (\hat{y}_i - y_i)^2$，单位是目标值的平方。越小越好，但绝对值依赖于数据的尺度。
- **$R^2$（决定系数）**：$R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}$，衡量模型解释了数据中多少比例的方差。越接近 1 表示模型解释力越强。$SS_{\text{res}} = \sum (y_i - \hat{y}_i)^2$ 是残差平方和，$SS_{\text{tot}} = \sum (y_i - \bar{y})^2$ 是总平方和（$\bar{y}$ 是 $y$ 的均值）。

$R^2$ 可以被理解为"相比于只用均值来预测，模型减少了多少比例的误差"。$R^2 = 0$ 表示模型不比均值好，$R^2 = 1$ 表示完美预测。

## 关键概念速查表

| 概念 | 数学形式 | 代码位置 | 关键说明 |
|------|---------|---------|---------|
| 线性模型 | $\hat{y} = wx + b$ | `predict()` | 最简单的参数化模型 |
| MSE 损失 | $J = \frac{1}{n} \sum (\hat{y}_i - y_i)^2$ | `_compute_loss()` | 处处可导，对大误差惩罚重 |
| 梯度 $\partial J/\partial w$ | $\frac{2}{n} \sum (\hat{y}_i - y_i) x_i$ | `_compute_gradients()` | 链式法则推导 |
| 梯度 $\partial J/\partial b$ | $\frac{2}{n} \sum (\hat{y}_i - y_i)$ | `_compute_gradients()` | 比 $w$ 的梯度少乘 $x_i$ |
| 梯度下降更新 | $\theta \leftarrow \theta - \eta \nabla_\theta J$ | `fit()` | `self.w -= lr * dw` |
| 正规方程 | $\theta^* = (X^T X)^{-1} X^T y$ | `normal_equation_solution()` | 闭式解，$O(d^3)$ |
| 学习率 $\eta$ | 步长 | `__init__()` | 太小慢，太大震荡 |
| $R^2$ | $1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}$ | `main()` | 模型解释方差比例 |


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/ml/foundations/linear-regression/code/demo.py`
