---
title: "ml01 k-近邻与距离度量 — exercise.py"
---

# ml01 k-近邻与距离度量 — exercise.py 练习指南

<a href="../code/ml01_knn/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过补全余弦距离计算、距离加权投票和 k 值交叉验证选择三个模块，从代码层面理解 k-NN 算法的核心组件。

## 预备知识

在开始练习前，确保你已经理解了以下概念（参见 [demo.py 代码详解](./code-demo) 中的详细解释）：

- 余弦距离的公式：$d_{\text{Cos}} = 1 - \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|_2 \|\mathbf{b}\|_2}$
- 距离加权投票：$w_i = \frac{1}{d_i + \varepsilon}$，每个类别累计权重，选权重最大的类别
- k 值对决策边界的影响：$k$ 越小，决策边界越复杂（过拟合）；$k$ 越大，边界越平滑（欠拟合）
- 交叉验证是选择 $k$ 值的常用方法

## 任务清单

### 任务1：实现余弦距离计算 `cosine_distance(X_test, X_train)`

- **用到的公式**：
  - L2 范数：$\|\mathbf{x}\|_2 = \sqrt{\sum x_i^2}$
  - 余弦相似度：$\cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}$
  - 余弦距离：$d = 1 - \cos(\theta)$
- **实现步骤**：
  1. 用 `np.linalg.norm(X_test, axis=1, keepdims=True)` 计算测试样本的 L2 范数，shape 为 `(m, 1)`
  2. 用 `np.linalg.norm(X_train, axis=1)` 计算训练样本的 L2 范数，shape 为 `(n,)`
  3. 用 `X_test @ X_train.T` 计算点积矩阵，shape 为 `(m, n)`
  4. 计算余弦相似度：`cos_sim = dot / (norm_test @ norm_train[np.newaxis, :] + 1e-10)`（`1e-10` 防除零）
  5. 用 `np.clip(cos_sim, -1.0, 1.0)` 限制在 $[-1, 1]$ 范围内（数值稳定性）
  6. 返回 `1.0 - cos_sim`
- **需要调用的函数**：`np.linalg.norm()`、`@` 矩阵乘法、`np.clip()`
- **关键细节**：`norm_test @ norm_train[np.newaxis, :]` 产生的外积矩阵 shape 为 `(m, n)`，正是我们需要的

### 任务2：实现距离加权投票 `predict_weighted(top_k_labels, top_k_distances, n_classes)`

- **算法流程**：
  1. 计算权重矩阵：`weights = 1.0 / (top_k_distances + 1e-6)`（`1e-6` 防除零）
  2. 对每个测试样本 `i`：
     - 创建 `class_weights = np.zeros(n_classes)`
     - 对该样本的 `k` 个邻居 `j`：
       - `label = int(top_k_labels[i, j])`
       - `class_weights[label] += weights[i, j]`
     - `predictions[i] = np.argmax(class_weights)`
  3. 返回预测数组
- **需要调用的函数**：`np.zeros()`、`np.argmax()`、`int()`
- **直觉理解**：距离 $d$ 越小，权重 $1/d$ 越大，该类获得的累计权重也越大。一个特别近的邻居（$d \approx 0$）会获得极大的权重，几乎独立决定分类结果

### 任务3（Bonus）：实现 k 值选择的 K-Fold 交叉验证 `kfold_choose_k(X, y, k_values, n_folds, metric)`

- **算法流程**：
  1. 对每个候选的 $k$ 值：
     - 计算每折大小：`fold_size = n // n_folds`
     - 对每一折（`fold = 0, 1, ..., n_folds-1`）：
       - 确定验证集起止索引：`val_start = fold * fold_size`，`val_end = (fold+1) * fold_size`（最后一折到末尾）
       - 验证集索引：`val_idx = np.arange(val_start, val_end)`
       - 训练集索引：`train_idx = np.setdiff1d(np.arange(n), val_idx)`
       - 用 k-NN（均匀投票）在训练集上"训练"、在验证集上预测
       - 计算验证准确率
     - 记录该 $k$ 值的平均准确率
  2. 返回平均准确率最高的 $k$ 值
- **关键细节**：k-NN 的"训练"就是存储数据——`fit()` 只需赋值，无需计算

## 验证标准

运行 `python exercise.py`：

1. `test_cosine_distance()`：相同向量的余弦距离应为 0；正交向量的余弦距离应为 1
2. `test_weighted_voting()`：距离加权投票应正确选出权重最大的类别
3. `test_kfold()`：交叉验证应返回有效的 $k$ 值和与之对应的交叉验证分数

## 完整代码

<<< @/snippets/ml01_knn/exercise.py
