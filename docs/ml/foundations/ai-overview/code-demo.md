---
title: "s01 AI概述 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# s01 AI概述 — demo.py 代码详解

<a href="/notebook/code/ml/foundations/ai-overview/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/ml/foundations/ai-overview/code
python demo.py
```

## 代码逐段详解

### 第1步：导入库 — 每个库是做什么的

```python
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
```

- **`os`**：Python 标准库，用于操作文件路径。我们用它来创建 `images/` 目录并拼接文件路径，这样保存图片时不会因为目录不存在而报错。
- **`numpy`**（导入为 `np`）：科学计算的核心库。我们用它的功能包括：
  - `np.random.randn()` 生成服从标准正态分布 $N(0,1)$ 的随机数（用于生成数据和初始化权重）
  - `np.dot()` 计算向量/矩阵的点积（$w \cdot x$）
  - `np.where()` 实现向量化的条件判断（阶跃函数的实现）
  - `np.vstack()` / `np.hstack()` 沿行/列方向拼接数组
  - `np.mean()` 计算平均值（用于评估准确率）
- **`matplotlib.pyplot`**（导入为 `plt`）：Python 最常用的绘图库。我们用 `plt.subplots()` 创建子图布局，`scatter()` 画散点图，`plot()` 画决策边界线，`arrow()` 画法向量箭头。
- **`matplotlib`**：设置 `rcParams['axes.unicode_minus'] = False` 避免负号显示为方块（尤其在中文环境下）。

### 第2步：数据生成 — 数据从哪来，长什么样

```python
def generate_linearly_separable_data(n_samples: int = 100, random_seed: int = 42):
```

这个函数生成一个**线性可分**的二分类数据集。所谓"线性可分"，就是存在一条直线能将两类数据完全分开——这是感知机能够收敛的前提条件。

**数据生成逻辑**：

1. **正类数据**：从均值 $(2, 2)$ 的二维正态分布中采样 100 个点
   ```python
   X_pos = np.random.randn(n_samples, 2) + np.array([2.0, 2.0])
   ```
   `np.random.randn(n_samples, 2)` 生成形状为 $(100, 2)$ 的标准正态随机数，`+ np.array([2.0, 2.0])` 将均值平移到 $(2, 2)$。标签全部设为 $+1$。

2. **负类数据**：从均值 $(-2, -2)$ 的二维正态分布中采样 100 个点
   ```python
   X_neg = np.random.randn(n_samples, 2) + np.array([-2.0, -2.0])
   y_neg = -np.ones(n_samples)
   ```
   标签全部设为 $-1$。

3. **合并与打乱**：防止训练时先看到一类再看到另一类，使用 `np.random.permutation()` 生成随机排列的索引来打乱数据顺序。

最终输出：
- `X`：形状 $(200, 2)$，特征矩阵，每行是一个点的 $(x_1, x_2)$ 坐标
- `y`：形状 $(200,)$，标签向量，每个元素是 $+1$ 或 $-1$

### 第3步：感知机模型定义 — 为什么这样设计

感知机是神经网络的最基本单元，它的数学模型是：

$$
\hat{y} = \text{sign}(w \cdot x + b) = \text{sign}\left(\sum_{i=1}^{n} w_i x_i + b\right)
$$

其中 $\text{sign}(z)$ 是阶跃函数（Step Function）：

$$
\text{sign}(z) = \begin{cases} +1 & \text{if } z \geq 0 \\ -1 & \text{if } z < 0 \end{cases}
$$

**类初始化** `__init__`：
- `learning_rate`（$\eta$）：学习率，控制每次参数更新的步长。太大可能震荡不收敛，太小收敛太慢。
- `max_epochs`：最大训练轮数。即使数据线性可分理论上保证收敛，但实际需要设一个上限防止无限循环。
- `w`：权重向量，形状 $(n\_features,)$，训练时初始化为小随机数。为什么不是全零？全零初始化会导致所有神经元学到同样的特征，这称为"对称性问题"。用小随机数打破对称性。
- `b`：偏置标量，初始化为 0。
- `losses`：记录每轮训练后误分类的样本数（感知机的"损失"概念与其他模型不同——它不最小化连续损失函数，而是直接最小化误分类数）。

**激活函数** `_activation(z)`：

```python
def _activation(self, z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0, 1, -1)
```

`np.where(condition, x, y)` 是向量化条件判断：对数组 `z` 中的每个元素，如果 `>= 0` 则输出 $+1$，否则输出 $-1$。这是阶跃函数的向量化实现，可以一次性对多个样本做判断。

**训练方法** `fit(X, y)` — 感知机学习算法：

这是感知机最核心的部分。算法思想非常简单：遍历每个样本，如果被分错了，就调整权重。

```python
if y_i * z <= 0:
    self.w += self.learning_rate * y_i * x_i
    self.b += self.learning_rate * y_i
```

这短短两行是整个感知机的精髓。让我们逐行拆解：

- `y_i * z <= 0` 判断误分类的条件：正确分类时，真实标签 $y_i$ 和净输入 $z = w \cdot x_i + b$ 应该同号（都为正或都为负），乘积 $> 0$。如果乘积 $\leq 0$，说明预测和真实标签不一致，即误分类。
- **权重更新** $w \leftarrow w + \eta \cdot y_i \cdot x_i$：
  - 如果 $y_i = +1$：把 $w$ 往 $x_i$ 的方向推，让 $w \cdot x_i$ 变大（更容易输出正类）
  - 如果 $y_i = -1$：把 $w$ 往 $x_i$ 的反方向推，让 $w \cdot x_i$ 变小（更容易输出负类）
- **偏置更新** $b \leftarrow b + \eta \cdot y_i$：偏置跟着权重一起更新。

**为什么这样更新是对的？** 直觉上，当我们误分类一个正类样本（$y=+1$，但模型输出 $-1$），说明 $w \cdot x + b$ 太小了。把 $w$ 沿着 $x$ 方向推一把，下次再遇到类似的样本，$w \cdot x$ 就会更大，更可能正确分类。

**感知机收敛定理**：如果数据是线性可分的，感知机算法一定能在有限步内收敛（所有样本分类正确）。如果数据不可分，算法将永远震荡，所以代码中设了 `max_epochs` 上限。

**预测方法** `predict(X)`：

```python
z = np.dot(X, self.w) + self.b
return self._activation(z)
```

`np.dot(X, self.w)` 计算矩阵乘法 $X_{n \times d} \cdot w_{d}$，得到形状 $(n,)$ 的得分向量。加上偏置后通过阶跃函数得到最终类别。这与训练时的计算完全一致，只是用矩阵形式批量处理。

**决策函数** `decision_function(X)`：返回未经阶跃函数处理的原始得分 $w \cdot x + b$，用于判断点到决策边界的距离和方向。

### 第4步：可视化 — 结果怎么看

**子图 1：决策边界图**

在二维平面上绘制决策边界。关键计算：

```python
slope = -w1 / w2          # 决策边界的斜率
intercept = -b_val / w2   # 决策边界的截距
```

这是从 $w_1 x_1 + w_2 x_2 + b = 0$ 解出 $x_2 = -(w_1 / w_2) x_1 - (b / w_2)$ 得到的。

- 红色圆点（`c='red', marker='o'`）：正类样本（$y = +1$）
- 蓝色方块（`c='blue', marker='s'`）：负类样本（$y = -1$）
- 绿色直线：决策边界 $w \cdot x + b = 0$
- 紫色箭头：权重向量 $w$（垂直于决策边界，指向正类方向）

法向量箭头的起点取在决策边界的中点上，方向沿 $w$，长度按比例缩放。从图上可以直观看到：$w$ 确实垂直于决策边界，且指向正类区域。

**子图 2：训练损失曲线**

横轴是 epoch（训练轮数），纵轴是每轮中误分类的样本数。对于线性可分数据，这条曲线应该单调下降并最终降到 0——表示感知机成功收敛。如果曲线震荡不收敛，说明数据可能线性不可分，需要更多层的网络。

### 第5步：主程序流程

```python
def main():
    X, y = generate_linearly_separable_data(n_samples=100, random_seed=42)
    perceptron = Perceptron(learning_rate=0.1, max_epochs=500)
    perceptron.fit(X, y)
    y_pred = perceptron.predict(X)
    accuracy = np.mean(y_pred == y)
    plot_decision_boundary(perceptron, X, y)
```

1. **生成数据**：200 个样本（每类 100 个），2 个特征，线性可分
2. **创建模型**：学习率设为 0.1（比默认的 0.01 大，因为感知机的更新规则收敛速度依赖学习率）
3. **训练**：在全部数据上训练感知机
4. **评估**：用 `np.mean(y_pred == y)` 计算准确率——即预测正确的比例。对于线性可分数据，期望达到 100%
5. **可视化**：展示决策边界和收敛过程
6. **测试几个点**：手动选几个坐标点验证模型的预测是否合理。例如 $(2.0, 2.0)$ 应该被预测为正类，$(-2.0, -2.0)$ 应该被预测为负类

## 关键概念速查表

| 概念 | 数学形式 | 代码位置 | 关键说明 |
|------|---------|---------|---------|
| 感知机模型 | $\hat{y} = \text{sign}(w \cdot x + b)$ | `predict()` | 最简神经网络，线性分类器 |
| 阶跃函数 | $\text{sign}(z) = +1 (z \ge 0), -1 (z < 0)$ | `_activation()` | `np.where(z >= 0, 1, -1)` |
| 误分类判断 | $y_i \cdot (w \cdot x_i + b) \leq 0$ | `fit()` | 正确分类时同号，乘积为正 |
| 权重更新 | $w \leftarrow w + \eta \cdot y_i \cdot x_i$ | `fit()` | 朝正确方向微调权重 |
| 偏置更新 | $b \leftarrow b + \eta \cdot y_i$ | `fit()` | 与权重同步更新 |
| 决策边界 | $w \cdot x + b = 0$ | `plot_decision_boundary()` | 超平面，法向量为 $w$ |
| 收敛定理 | 线性可分则有限步收敛 | `fit()` break逻辑 | 数据不可分时算法震荡 |
| 准确率 | $\frac{\text{正确预测数}}{\text{总样本数}}$ | `main()` | `np.mean(y_pred == y)` |

## 完整代码

<<< @/ml/foundations/ai-overview/code/demo.py
