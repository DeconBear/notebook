---
title: "ml01 k-近邻与距离度量 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# ml01 k-近邻与距离度量 — demo.py 代码详解

<a href="/notebook/code/ml/classic/knn/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/ml/classic/knn/code
python demo.py
```

## 代码逐段详解

### 第1步：导入库 — 每个库是做什么的

```python
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_moons, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
```

- **`os`**：文件路径操作，用于创建 `images/` 目录和生成文件保存路径
- **`numpy`**：数值计算核心。关键用法：`np.sum()` 计算平方和、`np.argsort()` 对距离排序找最近邻、`np.sqrt()` 开平方、`np.linalg.norm()` 计算向量范数、`np.linalg.pinv()` 伪逆（马氏距离用）、`np.cov()` 协方差矩阵、`np.clip()` 限制数值范围、`np.maximum()` 元素级取最大值
- **`matplotlib`**：绘图，包括决策边界填充图、距离等距线图、维数灾难曲线图、准确率对比图
- **`sklearn.neighbors.KNeighborsClassifier`**：sklearn 的标准 k-NN 实现，用作基准对比
- **`sklearn.datasets.make_moons`**：生成半月形数据集（非线性可分，测试决策边界的好工具）
- **`sklearn.datasets.make_classification`**：生成一般分类数据集
- **`sklearn.preprocessing.StandardScaler`**：特征标准化（每个维度减去均值除以标准差）。**k-NN 对特征尺度极其敏感**——如果某维度的数值范围远大于其他维度，它会主导距离计算
- **`sklearn.metrics.accuracy_score`**：分类准确率

### 第2步：四种距离度量的向量化实现

#### 2.1 欧氏距离的展开公式

```python
def euclidean_distance(X_test, X_train):
    sq_test = np.sum(X_test ** 2, axis=1, keepdims=True)   # (m, 1)
    sq_train = np.sum(X_train ** 2, axis=1)                # (n,)
    cross = 2.0 * X_test @ X_train.T                       # (m, n)
    sq_dists = sq_test + sq_train - cross
    return np.sqrt(np.maximum(sq_dists, 0.0))
```

这个实现看起来比直观的方式复杂，但它避免了**显式广播大矩阵**。

直观写法（内存浪费）：
```python
diff = X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]  # (m, n, d) — 大!
distances = np.sqrt(np.sum(diff ** 2, axis=2))
```

展开公式写法利用了数学恒等式：

$$
\| \mathbf{a} - \mathbf{b} \|^2 = \|\mathbf{a}\|^2 + \|\mathbf{b}\|^2 - 2 \mathbf{a} \cdot \mathbf{b}
$$

这样只需要计算 `(m, d) @ (d, n) = (m, n)` 的矩阵乘法——非常高效。

`np.maximum(sq_dists, 0.0)` 是一个**数值稳定性保护**：浮点误差可能导致极小负值（如 `-1e-15`），传递给 `np.sqrt()` 会返回 `NaN`。用 `maximum` 把负值 clamp 到 0 即可避免此问题。

#### 2.2 曼哈顿距离（L1）

```python
def manhattan_distance(X_test, X_train):
    abs_diff = np.abs(X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :])
    return np.sum(abs_diff, axis=2)
```

曼哈顿距离是各维度差值的绝对值之和：

$$
d_{\text{Man}}(\mathbf{x}, \mathbf{y}) = \sum_{i=1}^{d} |x_i - y_i|
$$

与欧氏距离平方不一样，L1 不放大差异。如果数据中有异常值，异常值在某个维度上的大偏差对 L1 距离的影响远小于对 L2 距离的影响（因为 L2 会平方放大）。

这里用了 broadcasting：`X_test[:, np.newaxis, :]` 将形状 `(m, d)` 变为 `(m, 1, d)`，与 `(1, n, d)` 相减得到 `(m, n, d)`。

#### 2.3 余弦距离

```python
def cosine_distance(X_test, X_train):
    norm_test = np.linalg.norm(X_test, axis=1, keepdims=True)
    norm_train = np.linalg.norm(X_train, axis=1)
    dot = X_test @ X_train.T
    cos_sim = dot / (norm_test @ norm_train[np.newaxis, :] + 1e-10)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    return 1.0 - cos_sim
```

关键步骤详解：

1. `np.linalg.norm(X_test, axis=1, keepdims=True)`：沿特征维度（axis=1）计算 L2 范数，`keepdims=True` 保留列向量形状 `(m, 1)` 而非 `(m,)`——这对后续的矩阵乘法至关重要

2. `norm_test @ norm_train[np.newaxis, :]`：这里没有直接用点积，而是用**外积**来计算所有范数乘积的组合，得到 `(m, n)` 的矩阵

3. `np.clip(cos_sim, -1.0, 1.0)`：数值稳定性。理论上余弦相似度在 $[-1, 1]$ 之间，但浮点运算可能产生 `1.0000001` 或 `-1.0000001`。`np.clip` 将其裁剪回合理范围

4. 返回 `1.0 - cos_sim`：将余弦相似度转换为距离（相似度越高，距离越小）

#### 2.4 马氏距离

```python
def mahalanobis_distance(X_test, X_train):
    Sigma = np.cov(X_train.T)
    Sigma_inv = np.linalg.pinv(Sigma)
    ...
    for i in range(m):
        diff = X_test[i] - X_train
        distances[i] = np.sqrt(np.sum((diff @ Sigma_inv) * diff, axis=1))
```

马氏距离的公式：

$$
d_{\text{Mah}} = \sqrt{(\mathbf{x} - \mathbf{y})^T \Sigma^{-1} (\mathbf{x} - \mathbf{y})}
$$

`np.cov(X_train.T)` 计算协方差矩阵（注意转置：sklearn/NumPy 的数据形状是 `(n_samples, n_features)`，而 `np.cov` 期望 `(n_features, n_samples)`）

`np.linalg.pinv(Sigma)` 用伪逆代替标准逆。当样本数 < 特征数时，协方差矩阵是奇异的——伪逆通过 SVD 分解给出一个稳定近似。

`(diff @ Sigma_inv) * diff` 是一个小技巧：先计算 $\text{diff} \cdot \Sigma^{-1}$（得到一个 `(n_train, d)` 矩阵），再逐元素乘以 `diff`，最后 `np.sum(axis=1)` 得到每行的二次型结果。等价于逐行计算 $\text{diff}_j \Sigma^{-1} \text{diff}_j^T$，但利用了批量运算。

### 第3步：自定义 KNNClassifier 类

#### 3.1 fit() — 惰性学习的本质

```python
def fit(self, X, y):
    self.X_train = np.asarray(X, dtype=np.float64)
    self.y_train = np.asarray(y)
    return self
```

k-NN 的 `fit()` 只是存储训练数据——**没有权重更新、没有梯度下降、没有迭代**。这就是"A lazy learner 从不主动学习，只在被问到问题时才去翻课本"。

#### 3.2 predict() — 预测流程

```python
def predict(self, X_test):
    distances = METRIC_FUNCTIONS[self.metric](X_test, self.X_train)
    top_k_indices = np.argsort(distances, axis=1)[:, :self.k]
    top_k_labels = self.y_train[top_k_indices]
    ...
```

核心步骤：

1. **计算距离矩阵**：用选择的距离度量函数，得到 `(m_test, n_train)` 的距离矩阵
2. **排序找近邻**：`np.argsort(distances, axis=1)` 沿测试样本维度排序，返回从小到大的索引。`[:, :self.k]` 取前 k 个，即最近的 k 个邻居
3. **索引标签**：`self.y_train[top_k_indices]` 用 fancy indexing 获取所有近邻的标签

#### 3.3 均匀投票

```python
def _vote_uniform(self, top_k_labels):
    for i in range(m_test):
        counts = np.bincount(top_k_labels[i].astype(int))
        predictions[i] = np.argmax(counts)
```

`np.bincount()` 统计数组中每个非负整数出现的次数。例如：
- 输入 `[0, 1, 0, 2, 0]` → 输出 `[3, 1, 1]`（0 出现 3 次，1 出现 1 次，2 出现 1 次）

`np.argmax(counts)` 返回出现次数最多的类别索引。

#### 3.4 距离加权投票

```python
def _vote_distance_weighted(self, top_k_labels, top_k_distances):
    weights = 1.0 / (top_k_distances + 1e-6)
    for i in range(m_test):
        class_weights = np.zeros(n_classes)
        for j in range(self.k):
            label = int(top_k_labels[i, j])
            class_weights[label] += weights[i, j]
        predictions[i] = np.argmax(class_weights)
```

权重公式：

$$
w_{ij} = \frac{1}{d(\mathbf{x}_i^{\text{test}}, \mathbf{x}_j^{\text{train}}) + \varepsilon}
$$

`1e-6`（即 $10^{-6}$）防止除零。对于距离为 0 的训练样本（测试样本恰好与某个训练样本重合），其权重会非常大（$1/10^{-6} = 10^6$），几乎以一己之力决定投票结果——这符合直觉。

然后对每个类别累计权重，选权重最大的类别。这确保了"近邻"比"远邻"拥有更大的话语权。

### 第4步：决策边界可视化

```python
def plot_decision_boundary(ax, model, X, y, title, step=0.05):
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, step),
        np.arange(y_min, y_max, step)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_points)
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
```

这段代码的核心思想是**对空间中的每一个点做预测，然后用不同颜色填充**。步骤如下：

1. **`np.meshgrid`** 在特征空间打网格
2. **`np.c_[xx.ravel(), yy.ravel()]`** 将二维网格坐标展开为一维点列表，每个点 `(x_coord, y_coord)` 一行
3. 对**所有网格点**调用 `model.predict()`——这可能很慢（如果网格很密），`step=0.05` 控制分辨率
4. **`Z.reshape(xx.shape)`** 将预测结果变回二维
5. **`ax.contourf`** 用填充等高线绘制决策区域

### 第5步：维数灾难的数值实验

```python
def plot_curse_of_dimensionality():
    for d in dims:
        points = np.random.uniform(0, 1, (n_points, d))
        idx1 = np.random.choice(n_points, size=min(500, n_points), replace=False)
        idx2 = np.random.choice(n_points, size=min(500, n_points), replace=False)
        diffs = points[idx1] - points[idx2]
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        ratios.append(dists.min() / (dists.max() + 1e-10))
```

这段代码用**Mento Carlo 模拟**验证维数灾难：在 $d$ 维超立方体中均匀采样点，计算随机点对之间的距离，然后看 `min(dist) / max(dist)` 的比值。

关键发现：随着 $d$ 增大，这个比值趋近于 1。原因是在高维空间中，向量的平方和（即 $\| \mathbf{x} - \mathbf{y} \|^2$）近似于正态分布（中心极限定理），其方差随 $d$ 增大相对减小，使得所有距离都集中在均值附近。

### 第6步：与 sklearn 的对比验证

```python
def plot_sklearn_comparison():
    knn_custom = KNNClassifier(k=k, metric='euclidean', weights='uniform')
    knn_custom.fit(X_train, y_train)
    custom_acc.append(accuracy_score(y_test, knn_custom.predict(X_test)))

    knn_sk = KNeighborsClassifier(n_neighbors=k, weights='uniform', metric='euclidean')
    knn_sk.fit(X_train, y_train)
    sklearn_acc.append(accuracy_score(y_test, knn_sk.predict(X_test)))
```

用相同的数据、相同的 $k$ 值、相同的距离度量和投票策略，自定义实现应该和 sklearn 给出**完全相同的预测**。如果两条曲线在图中重合，就验证了自定义实现的正确性。

## 关键概念速查表

| 概念 | 数学形式 | 代码位置 | 关键说明 |
|------|---------|---------|---------|
| 欧氏距离 | $\|\mathbf{a} - \mathbf{b}\|_2$ | `euclidean_distance()` | 展开公式避免大矩阵广播 |
| 曼哈顿距离 | $\sum \|a_i - b_i\|$ | `manhattan_distance()` | $L_1$ 范数，对异常值鲁棒 |
| 余弦距离 | $1 - \frac{\mathbf{a}\cdot\mathbf{b}}{\|\mathbf{a}\|\|\mathbf{b}\|}$ | `cosine_distance()` | 度量方向差异 |
| 马氏距离 | $\sqrt{(\mathbf{x}-\mathbf{y})^T\Sigma^{-1}(\mathbf{x}-\mathbf{y})}$ | `mahalanobis_distance()` | 考虑特征相关性 |
| 惰性学习 | `fit()` 只存储数据 | `KNNClassifier.fit()` | 无训练过程 |
| 多数投票 | `np.bincount` + `np.argmax` | `_vote_uniform()` | 每个邻居一票 |
| 距离加权 | $w = 1/(d + \varepsilon)$ | `_vote_distance_weighted()` | 近邻权重大 |
| 决策边界 | 网格预测 + `contourf` | `plot_decision_boundary()` | Voronoi 图 |
| 维数灾难 | `min/max dist → 1` | `plot_curse_of_dimensionality()` | 高维中距离失效 |

## 完整代码

<<< @/ml/classic/knn/code/demo.py
