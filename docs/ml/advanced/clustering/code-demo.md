---
title: "ml08 聚类 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# ml08 聚类 — demo.py 代码详解

<a href="/notebook/code/ml/advanced/clustering/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/ml/advanced/clustering/code
python demo.py
```

## 代码逐段详解

### 第1步：K-Means 从零实现——Lloyd 算法

K-Means 的核心目标是最小化簇内平方误差（WCSS）：

$$
J = \sum_{k=1}^{K} \sum_{x_i \in C_k} \|x_i - \mu_k\|^2
$$

```python
class KMeans:
    def __init__(self, n_clusters=3, init='kmeans++', max_iter=300, tol=1e-4, random_state=None):
```

- **`n_clusters`**：簇的数量 K——K-Means 必须预先指定的唯一超参数
- **`init='kmeans++'`**：使用 K-Means++ 算法初始化质心，避免陷入糟糕的局部最优
- **`max_iter=300`**：最大迭代次数（Lloyd 算法通常很快收敛，300 很充裕）
- **`tol=1e-4`**：收敛容忍度——质心移动距离小于 tol 时提前停止

#### K-Means++ 初始化

```python
def _init_centroids(self, X):
    centroids[0] = X[rng.randint(n_samples)]  # 步骤1: 随机选第一个质心
    for k in range(1, self.n_clusters):
        dists = cdist(X, centroids[:k]) ** 2   # 到最近质心的距离平方
        min_dists = dists.min(axis=1)
        probs = min_dists / min_dists.sum()    # 归一化为采样概率
        centroids[k] = X[rng.choice(n_samples, p=probs)]  # 按概率选下一个
```

K-Means++ 的关键洞察：**距离越远的点，越应该成为新质心**。第 3-4 步骤通过概率采样实现这一点——$P(x_i) \propto D(x_i)^2$，其中 $D(x_i)$ 是到最近已有质心的距离。

数学上，K-Means++ 保证了 $\mathbb{E}[J] \le O(\log K) \cdot J_{\text{opt}}$，即期望目标值不超过最优解的 $O(\log K)$ 倍。

#### Lloyd 迭代

```python
def fit(self, X):
    self.centroids_ = self._init_centroids(X)
    for iteration in range(self.max_iter):
        # 分配步: label[i] = argmin_k ||x_i - mu_k||
        distances = cdist(X, self.centroids_)       # (n, K)
        labels = np.argmin(distances, axis=1)        # 每行找到最小距离的列索引

        # 更新步: mu_k = mean(x_i in C_k)
        new_centroids = np.zeros_like(self.centroids_)
        for k in range(self.n_clusters):
            mask = (labels == k)
            if mask.sum() > 0:
                new_centroids[k] = X[mask].mean(axis=0)
```

Lloyd 算法的两个步骤：
1. **分配步（E 步）**：固定质心，将每个样本分配给最近的质心。这等价于在给定参数下推断隐变量（簇归属）。
2. **更新步（M 步）**：固定分配，将每个质心移到其簇所有样本的均值。这等价于在给定隐变量下最大化似然。

$J$ 在这两步中都**单调不增**，加上 $J \ge 0$ 有下界，保证了收敛。（注意收敛到的是局部最优，而非全局最优。）

### 第2步：DBSCAN 从零实现

DBSCAN 用完全不同的哲学定义簇：**簇 = 密度相连的点的最大集合**。

```python
class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps            # 邻域半径 epsilon
        self.min_samples = min_samples  # 核心点的最小邻居数
```

两个参数的含义：
- **`eps`（$\varepsilon$）**：定义"附近"的距离阈值
- **`min_samples`（minPts）**：定义"密集"的计数阈值

一个点是核心点当且仅当其 $\varepsilon$-邻域内至少有 minPts 个点。

#### 区域查询

```python
def _region_query(self, X, point_idx):
    dists = np.linalg.norm(X - X[point_idx], axis=1)  # 欧氏距离
    return np.where(dists <= self.eps)[0]              # <= eps 的邻居
```

`np.linalg.norm(X - X[point_idx], axis=1)` 利用了 NumPy 的广播机制：`X` 的每一行减去 `X[point_idx]`，再沿列方向计算 L2 范数，得到每个点到查询点的距离向量。

#### 簇的扩展（类似 BFS）

```python
seeds = list(neighbors)
while j < len(seeds):
    current = seeds[j]
    j += 1
    # 如果是噪声点，重新标记为边界点
    if labels[current] == -1:
        labels[current] = cluster_id - 1
    # 如果是核心点，将其邻居加入种子集
    if len(current_neighbors) >= self.min_samples:
        for nb in current_neighbors:
            if nb not in seeds:
                seeds.append(nb)
```

这是 DBSCAN 最核心的逻辑——**区域扩张**：
1. 从一个核心点开始，将其 $\varepsilon$-邻域内的所有点加入"种子集"
2. 逐个处理种子集中的点：如果某个种子点也是核心点，则将其邻域也加入种子集
3. 直到种子集被耗尽——此时该簇中的所有点都已找到

这个过程类似于广度优先搜索（BFS）或洪水填充（flood fill）。

### 第3步：聚类结果对比可视化

```python
def plot_clustering_comparison(X_list, titles, figsize=(18, 5)):
```

在三个合成数据集上对比 K-Means 和 DBSCAN：
- **Blobs**（各向同性高斯团）：K-Means 的理想场景，两种算法都应表现良好
- **Moons**（月牙形）：K-Means 失败——用直线强行切分非凸形状
- **Circles**（同心圆）：K-Means 完全失败——用扇形切分嵌套结构，DBSCAN 完美分离

> **关键洞察**：K-Means 和 DBSCAN 的差异根源在于它们对"什么是簇"的哲学不同——K-Means 认为簇是"球形区域"，DBSCAN 认为簇是"密度相连的点集"。

### 第4步：轮廓系数分析

```python
def plot_silhouette_analysis(X, K_range=range(2, 7)):
    sample_sil_values = silhouette_samples(X, labels)
```

轮廓系数 $s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$ 衡量了每个样本的聚类质量。`silhouette_samples` 返回每个样本的轮廓系数，`silhouette_score` 返回平均值。

在轮廓图中需要关注：
- **红色虚线**：平均轮廓系数——越高越好
- **每个簇的"柱子"宽度**：反映了簇的大小
- **柱子伸出虚线的情况**：如果某簇的大部分样本在虚线左侧，说明聚类质量差

### 第5步：DBSCAN 参数影响分析

```python
def plot_dbscan_parameter_effect(X, param_list):
```

展示不同 `(eps, min_samples)` 组合对 DBSCAN 结果的影响：
- **eps 太小**：几乎所有点都是噪声（找不到足够的邻居）
- **eps 太大**：所有点合并为一个簇（丧失了区分能力）
- **minPts 太小**：过多的核心点，可能把噪声也聚类
- **minPts 太大**：过少的核心点，很多点被标记为噪声

> 参数选择的经验法则：$\text{minPts} \ge d + 1$（$d$ 为数据维度），eps 通过 k-距离图的"拐点"确定。

### 第6步：肘部法则

```python
def plot_inertia_vs_k(X, K_range=range(1, 11)):
```

肘部法则是选择 K-Means 最佳 K 的经典方法。原理：
- 随着 K 增大，inertia（WCSS）必然单调递减（每个点离其质心越来越近）
- 但"真正"的簇数之后，递减速率会显著变缓——在曲线上形成一个"肘部"
- 选择肘部对应的 K 作为最佳簇数

**数学直觉**：inertia 曲线类似于 PCA 的累计解释方差曲线——增加的簇在"真实簇数"之后只能解释噪声方差，因此边际收益骤降。

### 第7步：聚类评估指标汇总

K-Means 在 4 个真实簇的数据上，不同 K 值下的指标：
- **K=2**：轮廓系数较低（欠聚类——合并了应该分开的簇）
- **K=4**：轮廓系数最高（正确——数据真实有 4 个簇）
- **K=6**：轮廓系数下降（过聚类——拆分了一个自然簇）

### 关键概念速查表

| 概念 | 数学形式 | 代码位置 | 关键说明 |
|------|---------|---------|---------|
| WCSS | $J = \sum \|x - \mu_k\|^2$ | `KMeans.fit()` | K-Means 的最小化目标 |
| Lloyd 迭代 | 分配步 + 更新步 | `KMeans.fit()` for loop | 交替优化，保证单调收敛 |
| K-Means++ | $P(x) \propto D(x)^2$ | `_init_centroids()` | 概率采样选分散质心 |
| $\varepsilon$-邻域 | $\{q: \|p-q\| \le \varepsilon\}$ | `_region_query()` | DBSCAN 的密度定义基础 |
| 核心点 | $\|N_\varepsilon(p)\| \ge$ minPts | `fit()` 中判断 | 密集区域内部点 |
| 密度可达链 | 核心点 → 核心点 → ... → 任意点 | `seeds` 循环 | 簇的连通性定义 |
| 轮廓系数 | $s = (b-a)/\max(a,b)$ | `silhouette_score()` | 聚类质量内部评估 |
| 肘部法则 | inertia vs K 曲线 | `plot_inertia_vs_k()` | 选择最佳 K 值 |


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/ml/advanced/clustering/code/demo.py`
