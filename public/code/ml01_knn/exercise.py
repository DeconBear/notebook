# -*- coding: utf-8 -*-
"""
===============================================================================
ml01_knn/code/exercise.py — k-近邻算法练习
===============================================================================
本练习文件中，你需要完成以下任务：

练习目标：
  1. 实现余弦距离的计算函数
  2. 实现距离加权投票的核心逻辑
  3. 实现 k 值选择的交叉验证方法（Bonus）

提示：
  - 余弦距离: d = 1 - (a · b) / (||a|| * ||b||)
  - 加权投票: w_i = 1 / (d_i + epsilon)，累加各邻居对各类别的权重贡献
  - 交叉验证: 将数据分为 K 折，轮流作为验证集

运行方式：
  python exercise.py
===============================================================================
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler


# ============================================================================
# 辅助函数（已实现，可直接使用）
# ============================================================================

def euclidean_distance(X_test, X_train):
    """欧氏距离: sqrt(||a||^2 + ||b||^2 - 2*a·b)"""
    sq_test = np.sum(X_test ** 2, axis=1, keepdims=True)
    sq_train = np.sum(X_train ** 2, axis=1)
    cross = 2.0 * X_test @ X_train.T
    sq_dists = np.maximum(sq_test + sq_train - cross, 0.0)
    return np.sqrt(sq_dists)


# ============================================================================
# 任务 1: 实现余弦距离计算 (约 10 行)
# ============================================================================

def cosine_distance(X_test, X_train):
    """
    计算余弦距离矩阵。

    余弦相似度 = (a · b) / (||a|| * ||b||)
    余弦距离 = 1 - 余弦相似度

    参数:
        X_test: (m, d), m 个测试样本
        X_train: (n, d), n 个训练样本
    返回:
        distances: (m, n), 余弦距离矩阵
    """
    # TODO: 完成以下步骤
    # 1. 计算 X_test 每行的 L2 范数 norm_test (shape: (m, 1))
    # 2. 计算 X_train 每行的 L2 范数 norm_train (shape: (n,))
    # 3. 计算点积矩阵 dot = X_test @ X_train.T (shape: (m, n))
    # 4. 计算余弦相似度: cos_sim = dot / (norm_test @ norm_train.T + 1e-10)
    #    注意: norm_test 是 (m,1), norm_train 是 (n,)，需要用 broadcasting
    # 5. 将 cos_sim 限制在 [-1, 1] 范围内: np.clip(cos_sim, -1, 1)
    # 6. 返回 1 - cos_sim
    # --- BEGIN YOUR CODE ---
    norm_test = np.linalg.norm(X_test, axis=1, keepdims=True)  # (m, 1)
    norm_train = np.linalg.norm(X_train, axis=1)               # (n,)
    dot = X_test @ X_train.T  # (m, n)
    cos_sim = dot / (norm_test @ norm_train[np.newaxis, :] + 1e-10)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    distances = 1.0 - cos_sim
    # --- END YOUR CODE ---
    # ^^^ TODO: 请删除上面的实现代码，自己重写 ^^^
    return distances


# ============================================================================
# 任务 2: 实现距离加权投票 (约 15 行)
# ============================================================================

def predict_weighted(top_k_labels, top_k_distances, n_classes):
    """
    距离加权投票: 每个邻居的票权重 = 1/(distance + epsilon)。

    步骤:
      1. 计算权重: w = 1 / (d + 1e-6)，形状 (m_test, k)
      2. 对每个测试样本 i:
         - 创建一个长度 n_classes 的数组 class_weights，初始为 0
         - 对该样本的 k 个邻居 j:
           label = top_k_labels[i, j]  (转为 int)
           class_weights[label] += w[i, j]
         - predictions[i] = np.argmax(class_weights)

    参数:
        top_k_labels: (m_test, k), 最近的 k 个邻居的类别标签
        top_k_distances: (m_test, k), 对应的距离
        n_classes: int, 类别总数
    返回:
        predictions: (m_test,), 每个测试样本的预测类别
    """
    # TODO: 完成距离加权投票逻辑
    m_test = top_k_labels.shape[0]
    k = top_k_labels.shape[1]
    # --- BEGIN YOUR CODE ---
    # 1. 计算权重: 距离越小权重越大
    weights = 1.0 / (top_k_distances + 1e-6)  # (m_test, k)

    # 2. 初始化预测数组
    predictions = np.zeros(m_test, dtype=int)

    # 3. 对每个测试样本，累加各类别的权重，选出权重最大的类别
    for i in range(m_test):
        class_weights = np.zeros(n_classes)
        for j in range(k):
            label = int(top_k_labels[i, j])
            class_weights[label] += weights[i, j]
        predictions[i] = np.argmax(class_weights)
    # --- END YOUR CODE ---
    # ^^^ TODO: 请删除上面的实现代码，自己重写 ^^^
    return predictions


# ============================================================================
# 任务 3 (Bonus): 实现简单的 k-fold 交叉验证（用于选择最佳 k）(约 15 行)
# ============================================================================

def kfold_choose_k(X, y, k_values, n_folds=5, metric='euclidean'):
    """
    用 K-Fold 交叉验证选择最佳的 k 值。

    步骤:
      1. 对每个候选的 k:
         - 将数据分为 n_folds 折（均匀分割）
         - 对每一折:
           - 该折作为验证集，其余作为训练集
           - 用 k-NN（均匀投票）训练并预测
           - 计算验证集准确率
         - 记录该 k 值的平均准确率
      2. 返回平均准确率最高的 k 值

    参数:
        X: (n, d), 特征
        y: (n,), 标签
        k_values: list, 候选 k 值列表
        n_folds: int, 交叉验证折数
        metric: str, 距离度量
    返回:
        best_k: int, 最佳 k 值
        cv_scores: dict, {k: mean_accuracy}
    """
    n = len(X)
    fold_size = n // n_folds
    cv_scores = {}

    # --- BEGIN YOUR CODE ---
    for k in k_values:
        fold_accuracies = []
        for fold in range(n_folds):
            # 验证集索引
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size if fold < n_folds - 1 else n
            val_idx = np.arange(val_start, val_end)
            # 训练集索引（排除验证集）
            train_idx = np.setdiff1d(np.arange(n), val_idx)

            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            # 用 k-NN 预测
            distances = euclidean_distance(X_val, X_train)
            top_k_indices = np.argsort(distances, axis=1)[:, :k]
            top_k_labels = y_train[top_k_indices]

            # 均匀投票
            predictions = np.zeros(len(X_val), dtype=y.dtype)
            for i in range(len(X_val)):
                counts = np.bincount(top_k_labels[i].astype(int))
                predictions[i] = np.argmax(counts)

            # 准确率
            acc = np.mean(predictions == y_val)
            fold_accuracies.append(acc)

        cv_scores[k] = np.mean(fold_accuracies)
    # --- END YOUR CODE ---
    # ^^^ TODO: 请删除上面的实现代码，自己重写 ^^^

    best_k = max(cv_scores, key=cv_scores.get)
    return best_k, cv_scores


# ============================================================================
# 验证代码
# ============================================================================

def test_cosine_distance():
    """验证余弦距离计算。"""
    print("[测试 1] 余弦距离...")
    X_test = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    X_train = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]])

    dists = cosine_distance(X_test, X_train)

    # 相同向量: 余弦距离应为 0
    assert np.abs(dists[0, 0]) < 1e-5, f"相同向量余弦距离不为0: {dists[0,0]}"
    # 正交向量: 余弦距离应为 1
    assert np.abs(dists[0, 1] - 1.0) < 1e-5, f"正交向量余弦距离不为1: {dists[0,1]}"
    # 同方向向量: 余弦距离应为 0
    assert np.abs(dists[0, 2]) < 1e-5, f"同方向向量余弦距离不为0: {dists[0,2]}"
    print("  [PASS] 余弦距离计算正确!")


def test_weighted_voting():
    """验证距离加权投票。"""
    print("[测试 2] 距离加权投票...")

    # 场景: 3 个邻居，2 个类别 (0 和 1)
    top_k_labels = np.array([[0, 1, 0]])  # 标签: 0, 1, 0
    top_k_distances = np.array([[0.1, 0.5, 0.2]])  # 距离: 近, 远, 中

    pred = predict_weighted(top_k_labels, top_k_distances, n_classes=2)

    # 预期: 类别 0 有两个但不会压倒类别 1（还要看距离权重）
    # w1(0) = 1/0.1 ≈ 10, w2(1) = 1/0.5 = 2, w3(0) = 1/0.2 = 5
    # 类别 0 总权重 = 10 + 5 = 15 > 类别 1 权重 2
    assert pred[0] == 0, f"加权投票结果错误: {pred[0]}"
    print(f"  [PASS] 距离加权投票正确! 预测: {pred[0]}")


def test_kfold():
    """验证交叉验证选择最佳 k。"""
    print("[测试 3 (Bonus)] K-Fold 交叉验证选 k...")

    np.random.seed(42)
    X, y = make_classification(
        n_samples=150, n_features=4, n_redundant=0, n_clusters_per_class=1,
        random_state=42
    )
    X = StandardScaler().fit_transform(X)

    k_values = [1, 3, 5]
    best_k, cv_scores = kfold_choose_k(X, y, k_values, n_folds=3)

    assert best_k in k_values, f"最佳 k 不在候选列表中: {best_k}"
    assert len(cv_scores) == len(k_values), f"cv_scores 长度不对: {len(cv_scores)}"
    print(f"  [PASS] 交叉验证完成! 最佳 k = {best_k}, scores = {cv_scores}")


if __name__ == "__main__":
    print("=" * 60)
    print("ml01_knn exercise.py — k-NN 算法练习")
    print("=" * 60)

    try:
        test_cosine_distance()
        test_weighted_voting()
        test_kfold()
        print("\n" + "=" * 60)
        print("所有测试通过!")  # All tests passed
        print("=" * 60)
    except Exception as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
