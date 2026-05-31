# -*- coding: utf-8 -*-
"""
===============================================================================
ml06_random_forest/code/exercise.py — Bagging 与随机森林练习
===============================================================================
本练习文件中，你需要完成以下任务：

练习目标：
  1. 实现 Bootstrap 采样
  2. 实现 Bagging 的多数投票预测
  3. 实现 OOB 误差的计算（Bonus）

提示：
  - Bootstrap: np.random.choice(n, size=n, replace=True)
  - 多数投票: scipy.stats.mode 或 np.bincount + np.argmax
  - OOB: 对每个样本，找未使用该样本的树做预测

运行方式：
  python exercise.py
===============================================================================
"""

import numpy as np
from scipy import stats as scipy_stats
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# ============================================================================
# 任务 1: 实现 Bootstrap 采样 (约 5 行)
# ============================================================================

def bootstrap_sample(X, y):
    """
    从数据集中进行 Bootstrap 采样（有放回）。

    参数:
        X: (n, d), 特征矩阵
        y: (n,), 标签
    返回:
        X_boot: (n, d), Bootstrap 采样后的特征
        y_boot: (n,), Bootstrap 采样后的标签
        oob_idx: (n_oob,), 未被采样到的样本索引（OOB 样本）
    """
    n = len(y)
    # TODO:
    # 1. boot_idx = np.random.choice(n, size=n, replace=True)
    # 2. X_boot = X[boot_idx], y_boot = y[boot_idx]
    # 3. oob_idx = np.setdiff1d(np.arange(n), np.unique(boot_idx))
    # 4. 返回 X_boot, y_boot, oob_idx
    # --- BEGIN YOUR CODE ---
    boot_idx = np.random.choice(n, size=n, replace=True)
    X_boot = X[boot_idx]
    y_boot = y[boot_idx]
    oob_idx = np.setdiff1d(np.arange(n), np.unique(boot_idx))
    # --- END YOUR CODE ---
    return X_boot, y_boot, oob_idx


# ============================================================================
# 任务 2: 实现多数投票预测 (约 10 行)
# ============================================================================

def majority_vote_predict(trees, X):
    """
    对多棵树的预测结果进行多数投票。

    参数:
        trees: list, 已训练的决策树列表（每棵树有 predict 方法）
        X: (n, d), 测试样本
    返回:
        predictions: (n,), 投票后的预测结果
    """
    # TODO:
    # 1. all_preds = np.array([tree.predict(X) for tree in trees])
    #    这会得到 (n_trees, n_samples) 的预测矩阵
    # 2. 用 scipy_stats.mode(all_preds, axis=0) 获取每列的众数
    #    注意 mode 返回 (mode_values, counts)
    # 3. 返回 mode_values.ravel()
    # --- BEGIN YOUR CODE ---
    all_preds = np.array([tree.predict(X) for tree in trees])  # (n_trees, n_samples)
    predictions = scipy_stats.mode(all_preds, axis=0, keepdims=False)[0]
    predictions = predictions.ravel()
    # --- END YOUR CODE ---
    return predictions


# ============================================================================
# 任务 3 (Bonus): 实现 OOB 误差计算 (约 15 行)
# ============================================================================

def compute_oob_score(trees, bootstrap_indices, X, y):
    """
    计算 Out-of-Bag (OOB) 准确率。

    对每个样本 i:
      - 找到所有未使用样本 i 训练的树
      - 用这些树对该样本做多数投票预测
      - 与真实标签比较

    参数:
        trees: list, 决策树列表
        bootstrap_indices: list of np.ndarray, 每棵树的 Bootstrap 索引
        X: (n, d), 全部训练数据
        y: (n,), 全部训练标签
    返回:
        oob_accuracy: float
    """
    n_samples = len(y)
    oob_predictions = np.zeros(n_samples, dtype=y.dtype)
    oob_covered = np.zeros(n_samples, dtype=bool)

    # TODO:
    # 1. for i in range(n_samples):
    #    a. 找所有未使用样本 i 的树索引: [t for t in range(n_trees) if i not in bootstrap_indices[t]]
    #    b. if len(voting_trees) == 0: continue (该样本未被任何 OOB 树覆盖)
    #    c. oob_covered[i] = True
    #    d. 收集这些树的预测: votes = [trees[t].predict(X[i:i+1])[0] for t in voting_trees]
    #    e. oob_predictions[i] = scipy_stats.mode(votes)[0]
    # 2. 只对有 OOB 覆盖的样本计算准确率
    # 3. 返回准确率
    # --- BEGIN YOUR CODE ---
    n_trees = len(trees)
    for i in range(n_samples):
        voting_trees = [t for t in range(n_trees) if i not in bootstrap_indices[t]]
        if len(voting_trees) == 0:
            continue
        oob_covered[i] = True
        votes = [trees[t].predict(X[i:i+1])[0] for t in voting_trees]
        oob_predictions[i] = scipy_stats.mode(votes, keepdims=False)[0]

    if np.sum(oob_covered) == 0:
        return 0.0
    oob_accuracy = np.mean(oob_predictions[oob_covered] == y[oob_covered])
    # --- END YOUR CODE ---
    return oob_accuracy


# ============================================================================
# 验证代码
# ============================================================================

def test_bootstrap():
    """验证 Bootstrap 采样。"""
    print("[测试 1] Bootstrap 采样...")
    np.random.seed(42)
    X = np.arange(20).reshape(10, 2)
    y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1])

    X_boot, y_boot, oob_idx = bootstrap_sample(X, y)

    # 采样后的数据应与原始数据维度相同
    assert len(X_boot) == len(X), f"Bootstrap 样本数不对: {len(X_boot)}"
    assert len(y_boot) == len(y), f"Bootstrap 标签数不对: {len(y_boot)}"
    # OOB 样本应该存在（通常约 36.8%）
    assert len(oob_idx) > 0, "应该有 OOB 样本"
    # OOB 样本数应该接近 36.8%
    print(f"  [PASS] OOB 样本数: {len(oob_idx)}/{len(y)} = {len(oob_idx)/len(y)*100:.1f}% (~36.8%)")


def test_majority_vote():
    """验证多数投票。"""
    print("[测试 2] 多数投票...")

    # 模拟 3 棵树，2 个样本
    class MockTree:
        def __init__(self, preds):
            self.preds = preds
        def predict(self, X):
            return np.array(self.preds[:len(X)])

    trees = [
        MockTree([0, 1]),
        MockTree([0, 1]),
        MockTree([1, 0]),  # 第三棵树与前面不同
    ]
    X_test = np.array([[0.0, 0.0], [1.0, 1.0]])

    preds = majority_vote_predict(trees, X_test)
    # 样本 0: [0, 0, 1] → 众数 = 0
    assert preds[0] == 0, f"样本0投票错误: {preds[0]}"
    # 样本 1: [1, 1, 0] → 众数 = 1
    assert preds[1] == 1, f"样本1投票错误: {preds[1]}"
    print(f"  [PASS] 多数投票正确!")


def test_oob_score():
    """验证 OOB 误差计算。"""
    print("[测试 3 (Bonus)] OOB 误差...")

    np.random.seed(42)
    X, y = make_classification(n_samples=100, n_features=4,
                                n_informative=2, random_state=42)

    # 训练 10 棵简单的决策树
    trees = []
    bootstrap_indices = []
    n = len(y)

    for _ in range(10):
        X_boot, y_boot, oob_idx = bootstrap_sample(X, y)
        tree = DecisionTreeClassifier(max_depth=3, random_state=np.random.randint(1000))
        tree.fit(X_boot, y_boot)
        trees.append(tree)
        # 记录 Bootstrap 索引
        boot_idx = np.random.choice(n, size=n, replace=True)
        bootstrap_indices.append(boot_idx)
        tree.fit(X[boot_idx], y[boot_idx])

    # 重新正确地训练（避免两次 fit）
    trees = []
    bootstrap_indices = []
    for _ in range(10):
        boot_idx = np.random.choice(n, size=n, replace=True)
        tree = DecisionTreeClassifier(max_depth=3, random_state=np.random.randint(1000))
        tree.fit(X[boot_idx], y[boot_idx])
        trees.append(tree)
        bootstrap_indices.append(boot_idx)

    oob_acc = compute_oob_score(trees, bootstrap_indices, X, y)

    assert 0 <= oob_acc <= 1, f"OOB 准确率应在 [0, 1]: {oob_acc}"
    print(f"  [PASS] OOB 准确率: {oob_acc:.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("ml06_random_forest exercise.py — Bagging 与随机森林练习")
    print("=" * 60)

    try:
        test_bootstrap()
        test_majority_vote()
        test_oob_score()
        print("\n" + "=" * 60)
        print("所有测试通过!")
        print("=" * 60)
    except Exception as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
