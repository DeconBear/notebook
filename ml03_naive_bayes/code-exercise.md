---
title: "ml03 朴素贝叶斯与贝叶斯网络 — exercise.py"
---

# ml03 朴素贝叶斯与贝叶斯网络 — exercise.py 练习指南

<a href="../code/ml03_naive_bayes/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过补全 GaussianNB 对数联合概率、MultinomialNB 特征概率估计和交叉验证选平滑参数三个模块，掌握朴素贝叶斯的核心计算。

## 预备知识

- 对数一维高斯密度：$\ln \mathcal{N}(x|\mu, \sigma^2) = -\frac{1}{2}[\ln(2\pi\sigma^2) + (x-\mu)^2/\sigma^2]$
- 条件独立下的联合对数似然：$\ln P(\mathbf{x}|\omega) = \sum_i \ln P(x_i|\omega)$
- 带平滑的多项分布估计：$P(w_i|c) = \frac{N_{ci} + \alpha}{N_c + \alpha \cdot |V|}$
- 零概率的危害：若 $\exists i \text{ s.t. } P(x_i|\omega) = 0$，则 $P(\mathbf{x}|\omega) = 0$

## 任务清单

### 任务1：实现 GaussianNB 的对数联合概率

- **步骤**：
  1. 对每个类别 $j$：计算 `log_lik = -0.5 * (np.log(2*pi*var[j]) + (X - mean[j])**2 / var[j])`
  2. 利用条件独立：`log_lik_sum = np.sum(log_lik, axis=1)`
  3. 加上先验：`log_joint[:, j] = log_lik_sum + np.log(priors[j])`
- **关键细节**：注意 `(X - mu)` 的 shape 为 `(n, d)`，`var` 的 shape 为 `(d,)`，NumPy 会自动 broadcasting

### 任务2：实现 MultinomialNB 的特征概率估计

- **步骤**：
  1. 对每个类别，计算 `count_c = np.sum(X_c, axis=0)`（各特征的计数）
  2. 计算 `total = np.sum(count_c)`（总计数）
  3. 带平滑的概率：`prob = (count_c + alpha) / (total + alpha * n_features)`
  4. 存储对数概率：`feature_log_prob[j] = np.log(prob)`
- **为什么取对数**：预测时用加法代替大量乘法，避免浮点下溢

### 任务3（Bonus）：交叉验证选最佳 smooth 参数

- **步骤**：
  1. 对每个候选 $\alpha$，做 K-Fold 交叉验证
  2. 用 `np.setdiff1d` 排除验证集得到训练集
  3. 训练 MultinomialNB 并在验证集上计算准确率
  4. 选平均准确率最高的 $\alpha$

## 验证标准

1. `test_gaussian_nb()`：GaussianNB 在简单分类数据集上准确率 > 50%
2. `test_multinomial_nb()`：Laplace 平滑后所有特征概率 > 0（无 `-inf` 值）
3. `test_alpha_selection()`：交叉验证选择出有效的平滑参数

## 完整代码

<<< @/snippets/ml03_naive_bayes/exercise.py
