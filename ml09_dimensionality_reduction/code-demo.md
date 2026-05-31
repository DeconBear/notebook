---
title: "ml09 降维与特征工程 — demo.py"
---

# ml09 降维与特征工程 — demo.py 代码详解

<a href="../code/ml09_dimensionality_reduction/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd ml09_dimensionality_reduction/code
python demo.py
```

## 代码逐段详解

### 第1步：PCA 从零实现（SVD 方法）

```python
class PCA:
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_          # (1) 数据中心化
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)  # (2) SVD
        self.components_ = Vt[:self.n_components]  # (3) 前 k 个右奇异向量
        eigenvalues = S ** 2 / (X.shape[0] - 1)    # (4) 奇异值 → 特征值
```

**数学推导**：

给定数据中心化后的矩阵 $\tilde{\mathbf{X}}_{n \times d}$，其 SVD 分解为：

$$
\tilde{\mathbf{X}} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T
$$

其中 $\mathbf{V}$ 的列是 $\tilde{\mathbf{X}}^T \tilde{\mathbf{X}}$ 的特征向量（即协方差矩阵的特征向量）。$\mathbf{\Sigma}$ 的对角元素（奇异值）$\sigma_j$ 与协方差矩阵的特征值 $\lambda_j$ 的关系为：

$$
\lambda_j = \frac{\sigma_j^2}{n - 1}
$$

**为什么用 SVD 而不是直接对角化协方差矩阵？**
- SVD 是数值上更稳定的算法
- 不需要显式构建 $d \times d$ 协方差矩阵（当 $d$ 很大时节省内存）
- sklearn 的 PCA 实现默认使用 SVD

**投影（transform）**：

```python
def transform(self, X):
    X_centered = X - self.mean_
    return X_centered @ self.components_.T  # (n, d) @ (d, k) = (n, k)
```

数学上：$\mathbf{Z} = \tilde{\mathbf{X}} \mathbf{W}_k$，其中 $\mathbf{W}_k$ 的列是前 $k$ 个主成分方向。

### 第2步：PCA 方差解释率图

方差解释率衡量每个主成分"解释"了数据中多少比例的总方差：

$$
\text{explained variance ratio}_j = \frac{\lambda_j}{\sum_i \lambda_i}
$$

图中的柱状图展示了每个主成分的单独解释率（快速递减——前几个主成分捕获了大部分方差）。红色折线展示了累积解释率。水平虚线标注了 90% 和 95% 阈值——这两条线告诉我们需要保留多少主成分才能保留相应比例的信息。

> **关键洞察**：在很多真实数据集中，前 10-20% 的主成分就能解释 90% 以上的方差——这就是降维有效性的数据基础。

### 第3步：PCA vs t-SNE vs LDA 对比

```python
def plot_pca_vs_tsne_vs_lda(X, y):
```

在同一个数据集上用三种方法降维到 2D：

- **PCA**：基于全局方差最大化，无监督。结果中不同类的点可能混在一起——因为 PCA 不关心类标签。
- **t-SNE**：基于局部邻域保持，无监督但对局部结构敏感。在可视化任务中几乎总是胜过 PCA——这得益于它的概率化邻居保持策略和 t 分布重尾对拥挤问题的处理。
- **LDA**：基于类可分性最大化，有监督。当数据有标签且目标是分类前置降维时，LDA 通常是最好的选择——因为它的优化目标直接就是最大化类间分离度。

### 第4步：特征工程 Pipeline

```python
def demo_feature_engineering():
```

展示了三个特征工程技巧在房价预测任务中的应用：

#### 分箱（Binning）
```python
binner = KBinsDiscretizer(n_bins=3, encode='onehot-dense', strategy='uniform')
age_binned = binner.fit_transform(df[['age']])
```
将连续房龄分为"新/中/旧"三个区间。为什么有效？线性模型只能拟合线性关系——但房价和房龄的关系可能是非线性的（新房贵、中龄房便宜、老房可能因历史价值又贵了）。分箱将这种非线性关系转化为可被线性模型处理的离散特征。

#### 交互特征
```python
df['area_x_rooms'] = df['area'] * df['rooms']
df['area_per_room'] = df['area'] / df['rooms']
```
线性模型的每个特征独立贡献于预测，但现实中特征之间存在交互效应——大面积 + 多房间的组合往往比两者各自贡献的加和有更大的影响（乘积项捕获了这种乘性关系）。

#### 目标编码
```python
target_encoding[d] = (n_d * cat_mean + alpha * global_mean) / (n_d + alpha)
```
对于高基数类别特征（如地区有几十个取值），One-Hot 会产生大量稀疏列。目标编码用一个数值替代每个类别——该类别对应的目标变量均值，但用平滑参数 $\alpha$ 向全局均值收缩来防止过拟合。

### 第5步：LDA 降维 + 分类实验

比较三种前置降维方案对分类性能的影响：
- **无降维（20D）**：直接使用 20 维特征
- **PCA (5D)**：投影到 5 个主成分
- **LDA (2D)**：投影到 $C-1=2$ 个判别方向

LDA 在很多情况下能以极低的维度（$C-1$）达到甚至超过全维度的分类性能——因为它的降维方向是专门为"区分不同类"而优化的。

## 关键概念速查表

| 概念 | 数学形式 | 代码位置 | 关键说明 |
|------|---------|---------|---------|
| PCA (SVD) | $\tilde{\mathbf{X}} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$ | `PCA.fit()` | 右奇异向量 = 主成分方向 |
| 方差解释率 | $\lambda_j / \sum \lambda_i$ | `explained_variance_ratio_` | 每个主成分的"重要性" |
| t-SNE | $\min KL(P\|Q)$ | `TSNE.fit_transform()` | 邻域概率分布的 KL 散度最小化 |
| LDA | $\max \frac{\mathbf{w}^T \mathbf{S}_B \mathbf{w}}{\mathbf{w}^T \mathbf{S}_W \mathbf{w}}$ | `LDA.fit_transform()` | 广义瑞利商问题 |
| 分箱 | 连续 → 离散区间 | `KBinsDiscretizer` | 赋予线性模型非线性能力 |
| 交互特征 | $x_i \times x_j$ | `df['area_x_rooms']` | 捕获特征联合效应 |
| 目标编码 | 贝叶斯收缩 | `target_encoding` | 类别 → 数值，平滑防过拟合 |

## 完整代码

<<< @/snippets/ml09_dimensionality_reduction/demo.py
