---
title: "s04 偏差-方差权衡 — demo.py"
---

# s04 偏差-方差权衡 — demo.py 代码详解

<a href="../code/s04_bias_variance/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd s04_bias_variance/code
python demo.py
```

## 代码逐段详解

### 第1步：导入库 — 每个库是做什么的

```python
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
```

- **`os`**：文件路径操作，创建 `images/` 目录
- **`numpy`**：数值计算核心。关键用法：`np.sin()` 生成正弦波数据，`np.linalg.pinv()` 伪逆求解正规方程，`np.hstack()` 构建多项式特征矩阵，`np.sign()` 计算 L1 正则化梯度中的符号函数，`np.logspace()` 生成对数均匀的 $\lambda$ 序列，`np.argmin()` 找到最优模型复杂度
- **`matplotlib`**：绘图，包括多项式拟合对比图、Bias-Variance 曲线、正则化对比图、系数路径图、交叉验证图
- **`sklearn.preprocessing.PolynomialFeatures`**：sklearn 的多项式特征生成器（用于对比验证，demo 中同时使用了自己的手动实现）
- **`sklearn.linear_model.LinearRegression, Ridge, Lasso`**：sklearn 的线性和正则化回归模型，用作基准对比
- **`sklearn.model_selection.KFold`**：sklearn 的 K-Fold 交叉验证分割器
- **`sklearn.pipeline.make_pipeline`**：构建处理管道，将多项式特征生成 + 回归模型串联为一个整体

### 第2步：数据生成 — 正弦波 + 噪声

```python
def generate_sine_data(n_samples=80, noise_std=0.3, random_seed=42):
```

生成模拟数据，真实函数为：

$$
f(x) = \sin(2\pi x), \quad x \in [0, 1]
$$

加上高斯噪声 $\epsilon \sim \mathcal{N}(0, 0.3^2)$，得到观测值：

$$
y = \sin(2\pi x) + \epsilon
$$

关键步骤：
1. `np.random.uniform(0, 1, n_samples)` 在 $[0, 1]$ 均匀采样 80 个点
2. `np.sort(X)` 对 $x$ 排序——这纯粹是为了画曲线美观，不是训练必需
3. `np.sin(2 * np.pi * X)` 计算真实函数值
4. 加上 `np.random.randn(n_samples) * noise_std` 的高斯噪声

选择正弦函数是因为它是**非线性**的——低次多项式无法很好地拟合（欠拟合），而高次多项式会拟合到噪声（过拟合），这正好展示了模型复杂度的权衡。

### 第3步：多项式特征生成

```python
def polynomial_features(X, degree):
    X = X.reshape(-1, 1)
    return np.hstack([X ** d for d in range(degree + 1)])
```

这个函数将一维特征 $x$ 扩展为多项式特征向量：

$$
x \rightarrow [1, x, x^2, x^3, \dots, x^{\text{degree}}]
$$

- `X.reshape(-1, 1)` 确保 $X$ 是列向量 $(n, 1)$
- `[X ** d for d in range(degree + 1)]` 生成一个列表，每个元素是一列 $x^d$
- `np.hstack()` 将所有列横向拼接成 $(n, \text{degree}+1)$ 的矩阵

第一列全 1（$x^0$）对应偏置项。多项式回归本质上仍然是线性回归——它在特征空间中依然是线性的（$\hat{y} = \theta_0 + \theta_1 x + \theta_2 x^2 + \dots$），但通过对原始特征做了非线性变换（将 $x$ 映射到 $[1, x, x^2, \dots]$），使模型能够拟合非线性关系。

### 第4步：多项式拟合（正规方程）

```python
def fit_polynomial(X, y, degree):
    Phi = polynomial_features(X, degree)
    theta = np.linalg.pinv(Phi.T @ Phi) @ Phi.T @ y
    return theta
```

对于多项式回归，正规方程的形式完全一致：

$$
\theta = (\Phi^T \Phi)^{-1} \Phi^T y
$$

其中 $\Phi$ 是多项式特征矩阵（代替了原始特征矩阵 $X$），$\theta$ 是 $[\theta_0, \theta_1, \dots, \theta_d]^T$。

**关键细节**：使用 `np.linalg.pinv()`（伪逆/Moore-Penrose 逆）而不是 `np.linalg.inv()`。当多项式次数较高时，$\Phi^T \Phi$ 可能接近奇异（数值上不可逆），伪逆通过 SVD 分解给出一个稳定的近似解，避免程序崩溃。

### 第5步：带正则化的线性回归

#### 5.1 正则化的动机

当多项式次数过高时（如 degree=15），模型有过多的自由度来"记住"每个训练数据点的位置，包括噪声。曲线会剧烈震荡，在训练数据上误差很小但在测试数据上误差很大——这就是过拟合。

**正则化的核心思想**：在损失函数中添加对参数大小的惩罚项，约束模型复杂度。

#### 5.2 L2 正则化（Ridge 回归）

$$
J_{\text{Ridge}}(\theta) = \text{MSE} + \lambda \sum_{j=1}^{d} \theta_j^2
$$

- $\lambda$ 是正则化强度：$\lambda$ 越大，参数被压缩得越厉害
- 偏置项 $\theta_0$ 通常**不参与正则化**——因为偏置只影响截距，不反映模型复杂度

梯度（加入正则化项后）：

$$
\frac{\partial J}{\partial \theta} = \frac{\partial \text{MSE}}{\partial \theta} + 2\lambda \theta
$$

代码实现：

```python
if self.reg_type == 'l2':
    reg_loss = self.lambda_ * np.sum(weights_no_bias ** 2)
    dw_reg[1:] = 2.0 * self.lambda_ * weights_no_bias  # 偏置项不参与
```

$2\lambda \theta$ 这一项就是**权重衰减（weight decay）**——它在每次梯度下降更新中把权重向零的方向"拉"一把。$\lambda$ 越大，拉的力度越大。

#### 5.3 L1 正则化（Lasso 回归）

$$
J_{\text{Lasso}}(\theta) = \text{MSE} + \lambda \sum_{j=1}^{d} |\theta_j|
$$

L1 与 L2 的关键区别：
- L2 使权重趋近于 0 但通常不为 0
- L1 倾向于产生**稀疏解**——许多权重精确为 0，实现内置的特征选择

梯度的正则化部分：

```python
elif self.reg_type == 'l1':
    reg_loss = self.lambda_ * np.sum(np.abs(weights_no_bias))
    dw_reg[1:] = self.lambda_ * np.sign(weights_no_bias)
```

`np.sign(w)` 返回每个权重的符号（$+1$、$-1$ 或 $0$）。这意味着 L1 正则化给每个权重的梯度贡献是恒定的（$\pm\lambda$），不管权重多大——这导致小权重被"推"到精确的零。

**几何直觉**：L2 的约束区域是圆形（球面），L1 的约束区域是菱形。损失函数的等高线与菱形的**角**（坐标轴上）最先接触的概率最大，而落在坐标轴上意味着某些权重精确为 0。

### 第6步：K-Fold 交叉验证

```python
def kfold_cross_validation(X, y, k=5, degree=3, lambda_=0.0, reg_type='none'):
```

**为什么需要交叉验证？** 正则化强度 $\lambda$ 需要合理选择——太小无法抑制过拟合，太大导致欠拟合。交叉验证用数据本身来决定"度"在哪里。

**K-Fold 流程**：
1. 将数据分成 $K$ 等份（折/fold）
2. 对于每一折 $i = 0, 1, \dots, K-1$：
   - 第 $i$ 份作为验证集
   - 其余 $K-1$ 份作为训练集
   - 训练模型并在验证集上计算 MSE
3. 返回 $K$ 次验证 MSE 的平均值

代码实现中，`np.setdiff1d(np.arange(n), val_idx)` 用于从全量索引中排除验证集索引，得到训练集索引。这是一种"不重叠"的划分方式——每个样本恰好被用作验证一次。

### 第7步：可视化套件

#### 7.1 多项式拟合对比图

`plot_polynomial_fits()` 在 $3 \times 5$ 的网格中展示 degree=1 到 15 的拟合效果：

- **欠拟合**（degree=1-2）：曲线过于平滑，无法捕捉正弦波的起伏，训练 MSE 和测试 MSE 都高
- **拟合良好**（degree=3-7）：曲线贴合正弦波形状，训练和测试 MSE 都低
- **过拟合**（degree=12-15）：曲线剧烈震荡，每个训练点都被穿过，训练 MSE 极低但测试 MSE 高

代码通过比较训练和测试 MSE 的大小关系来自动判断拟合质量：
- `train_mse > 0.15`：欠拟合
- `deg > 12 and test_mse > 3 * train_mse`：过拟合
- 其余：拟合良好

#### 7.2 Bias-Variance U 形曲线

`plot_bias_variance_curve()` 绘制训练误差和验证误差随多项式次数的变化：

- **训练误差（蓝线）**：随次数增加单调递减——模型越复杂，越能"记住"训练数据
- **验证误差（红线）**：呈 U 形——先降后升。最低点对应最优模型复杂度

代码用 `np.argmin(val_errors)` 找到验证误差最小的多项式次数，用绿色虚线标注。U 形曲线的左侧是"欠拟合区域"，右侧是"过拟合区域"。

#### 7.3 正则化效果对比

`plot_regularization_effect()` 对 degree=15 的严重过拟合模型应用不同正则化策略：

- **无正则化**：曲线剧烈震荡，训练 MSE 很低但验证 MSE 很高
- **L2 ($\lambda=0.01$)**：曲线变得平滑，大幅降低了过拟合
- **L1 ($\lambda=0.01$)**：类似 L2，但部分系数被精确推到零
- **L2 ($\lambda=0.1$)**：更强的正则化，曲线非常平滑但可能开始欠拟合（过于平滑，跟不上正弦波的弯曲）

#### 7.4 系数路径图

`plot_coefficient_paths()` 展示回归系数如何随 $\lambda$ 的增大而变化：

- **L2（Ridge）**：所有系数从初始值连续平滑地衰减到接近零，但通常不精确为零
- **L1（Lasso）**：随着 $\lambda$ 增大，系数逐个被"推"到精确的零——展示了 L1 的稀疏性

横轴是 $\lambda$（对数刻度，从 $10^{-4}$ 到 $10^2$），纵轴是系数值。`np.logspace(-4, 2, 50)` 在对数尺度上均匀采样 50 个 $\lambda$ 值。

#### 7.5 交叉验证选择模型

`plot_cv_results()` 对每个多项式次数计算 5-Fold 交叉验证的平均误差，画出 CV 误差曲线。最优多项式次数是 CV 误差最小的那个。

`ax.fill_between()` 绘制了 $\pm 1$ 标准差的阴影区域——它反映了不同折之间验证误差的波动程度。如果标准差很大，说明模型在不同数据划分上表现不稳定。

## 关键概念速查表

| 概念 | 数学形式 | 代码位置 | 关键说明 |
|------|---------|---------|---------|
| 多项式特征 | $[1, x, x^2, \dots, x^d]$ | `polynomial_features()` | 将线性模型变为非线性 |
| 欠拟合 | 训练误差高 + 验证误差高 | 可视化判断 | 模型太简单，没学到规律 |
| 过拟合 | 训练误差低 + 验证误差高 | 可视化判断 | 模型记噪声，泛化差 |
| L2 正则化 | $J + \lambda \sum \theta_j^2$ | `RegularizedLinearRegression` | 权重衰减，平滑压缩 |
| L1 正则化 | $J + \lambda \sum \|\theta_j\|$ | `RegularizedLinearRegression` | 产生稀疏解，特征选择 |
| K-Fold CV | $\frac{1}{K}\sum_{i=1}^{K} \text{MSE}_i$ | `kfold_cross_validation()` | 数据驱动选择超参数 |
| Bias² | $(\mathbb{E}[\hat{f}] - f)^2$ | 理论概念 | 平均预测与真实的差距 |
| Variance | $\text{Var}(\hat{f})$ | 理论概念 | 预测在不同训练集上的波动 |
| 系数路径 | $\theta_j(\lambda)$ | `plot_coefficient_paths()` | 观察正则化强度的效果 |
| 伪逆 | `np.linalg.pinv()` | `fit_polynomial()` | 稳定求解近似奇异矩阵 |

## 完整代码

<<< @/snippets/s04_bias_variance/demo.py
