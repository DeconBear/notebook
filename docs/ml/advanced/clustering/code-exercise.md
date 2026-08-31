---
title: "ml08 聚类 — exercise.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# ml08 聚类 — exercise.py 练习指南

<a href="/notebook/code/ml/advanced/clustering/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过实现 K-Means 单次迭代、轮廓系数计算、DBSCAN 区域查询以及综合 K 选择，从代码层面深入理解三种聚类算法的核心机制。

## 预备知识

在开始练习前，确保你已经理解了以下概念：

- Lloyd 算法的两步交替过程：分配步（将每个点分配到最近质心）和更新步（将质心移到簇均值）
- 欧氏距离平方在聚类中的核心地位：$\|x - \mu\|^2 = \sum (x_d - \mu_d)^2$
- 轮廓系数的数学定义：$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$
- DBSCAN 的核心点判定逻辑：$\varepsilon$-邻域内至少有 minPts 个邻居
- 肘部法则和轮廓系数分析如何帮助选择 K

## 任务清单

### 任务1：实现 K-Means 的单次迭代 `kmeans_single_iteration(X, centroids)`

- **用到的公式**：
  - 分配：$\text{label}[i] = \arg\min_k \|x_i - \mu_k\|^2$
  - 更新：$\mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i$
  - 惯性：$J = \sum_{k=1}^{K} \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$
- **实现步骤**：
  1. 用 `cdist(X, centroids)` 计算距离矩阵
  2. 用 `np.argmin(distances, axis=1)` 得到标签
  3. 对每个 k，用 `X[labels == k].mean(axis=0)` 更新质心
  4. 计算惯性（每个点到其簇质心的距离平方和）
- **需要调用的函数**：`cdist()`, `np.argmin()`, `np.mean()`
- **返回**：`(labels, new_centroids, inertia)`
- **期望输出**：标签为 0/1/2 三类，新质心 3x2 矩阵，inertia 为正实数

### 任务2：实现轮廓系数计算 `silhouette_score_manual(X, labels)`

- **算法流程**：
  1. 预计算距离矩阵 `cdist(X, X)`——shape `(n, n)`
  2. 对每个样本 i：
     - $a(i)$：同簇其他样本的距离均值
     - $b(i)$：最近的其他簇的距离均值（取 min）
     - $s(i) = (b - a) / \max(a, b)$
  3. 返回所有 s(i) 的均值
- **实现提示**：
  - 同簇样本：`mask = (labels == labels[i])`，注意排除 $i$ 自身
  - 其他簇平均距离：遍历所有 `unique(labels)`，跳过 `labels[i]`
- **需要调用的函数**：`cdist()`, `np.unique()`, `np.mean()`
- **边界情况**：如果只有一个簇或每个样本各成一簇，返回 0
- **期望输出**：范围 $[-1, 1]$，对于分离良好的 blobs 数据应 $\in (0.6, 0.8)$

### 任务3：实现 DBSCAN 区域查询 `region_query(X, point_idx, eps)`

- **数学定义**：$N_\varepsilon(p) = \{ q : \|p - q\| \le \varepsilon \}$
- **实现步骤**：
  1. 计算 `X[point_idx]` 到 X 中所有点的距离
  2. 返回距离 $\le \varepsilon$ 的点的索引
- **需要调用的函数**：`np.linalg.norm()`, `np.where()`
- **验证**：对 `[[0,0], [0.3,0.3], [0.5,0.5], [2,2]]`，`eps=0.5`，查询点 0 应返回 `[0, 1]`

### 任务4：综合选择最佳 K `find_optimal_k(X, K_range)`

- **实现策略**：
  1. 对 K_range 中的每个 K，运行 sklearn 的 KMeans
  2. 记录 inertia 和轮廓系数
  3. 找轮廓系数最高的 K 作为最佳选择
  4. 打印所有结果表格
- **需要调用的函数**：`sklearn.cluster.KMeans`, `sklearn.metrics.silhouette_score`
- **思考题**：轮廓系数和肘部法则在某些情况下可能给出不同建议——为什么？应该更信任哪个？

## 完整代码

<<< @/ml/advanced/clustering/code/exercise.py
