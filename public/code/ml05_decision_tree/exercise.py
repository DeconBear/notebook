# -*- coding: utf-8 -*-
"""
===============================================================================
ml05_decision_tree/code/exercise.py — 决策树练习
===============================================================================
本练习文件中，你需要完成以下任务：

练习目标：
  1. 实现基尼指数的计算
  2. 实现信息增益的计算（用于分裂评估）
  3. 实现最佳分裂点的搜索（单特征版本）

提示：
  - Gini = 1 - sum(p_j^2)，其中 p_j = n_j / n
  - 信息增益 = H_parent - (n_left/n) * H_left - (n_right/n) * H_right
  - 连续特征的分裂点: 排序后相邻取值的 M=n/2 处
  - 熵: H = -sum(p_j * log2(p_j))

运行方式：
  python exercise.py
===============================================================================
"""

import numpy as np


# ============================================================================
# 任务 1: 实现基尼指数计算 (约 5 行)
# ============================================================================

def compute_gini(y):
    """
    计算数据集 y 的基尼指数。

    Gini = 1 - sum(p_j^2), 其中 p_j 是类别 j 的占比。

    参数:
        y: (n,), 标签数组
    返回:
        gini: float
    """
    # TODO:
    # 1. 用 np.unique(y, return_counts=True) 获取各类别的计数
    # 2. 计算 p_j = counts / len(y)
    # 3. 返回 1 - sum(p_j^2)
    # --- BEGIN YOUR CODE ---
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    gini_val = 1.0 - np.sum(probs ** 2)
    # --- END YOUR CODE ---
    return gini_val


# ============================================================================
# 任务 2: 实现熵的计算 (约 5 行)
# ============================================================================

def compute_entropy(y):
    """
    计算数据集 y 的信息熵。

    H = -sum(p_j * log2(p_j))

    参数:
        y: (n,), 标签数组
    返回:
        entropy: float
    """
    # TODO:
    # 1. 获取各类别的占比 p_j
    # 2. 计算 H = -sum(p_j * log2(p_j + 1e-10))  (加微小量防 log(0))
    # 3. 返回 H
    # --- BEGIN YOUR CODE ---
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    ent = -np.sum(probs * np.log2(probs + 1e-10))
    # --- END YOUR CODE ---
    return ent


# ============================================================================
# 任务 3: 实现信息增益计算（用于评估某个分裂） (约 8 行)
# ============================================================================

def information_gain(y_parent, y_left, y_right):
    """
    计算一次分裂的信息增益（基尼指数版本）。

    IG = Gini_parent - (n_left/n * Gini_left + n_right/n * Gini_right)

    参数:
        y_parent: (n,), 父节点标签
        y_left: (n_left,), 左子节点标签
        y_right: (n_right,), 右子节点标签
    返回:
        gain: float, 信息增益（基尼减少量）
    """
    # TODO:
    # 1. 计算 Gini_parent
    # 2. 计算 Gini_left 和 Gini_right
    # 3. 计算加权子节点不纯度: w_impurity = (n_left/n)*Gini_left + (n_right/n)*Gini_right
    # 4. 返回 Gini_parent - w_impurity
    # --- BEGIN YOUR CODE ---
    gini_parent = compute_gini(y_parent)
    gini_left = compute_gini(y_left)
    gini_right = compute_gini(y_right)
    w_impurity = (len(y_left)/len(y_parent)) * gini_left + \
                 (len(y_right)/len(y_parent)) * gini_right
    gain = gini_parent - w_impurity
    # --- END YOUR CODE ---
    return gain


# ============================================================================
# 任务 4 (Bonus): 搜索单特征的最佳分裂点 (约 12 行)
# ============================================================================

def best_split_single_feature(X_col, y):
    """
    对单个连续特征搜索最佳分裂阈值。

    步骤:
      1. 按特征值排序
      2. 在相邻取值的 M=n/2 处设候选阈值
      3. 对每个候选阈值，计算基尼减少量
      4. 返回基尼减少最大的阈值

    参数:
        X_col: (n,), 单特征的取值
        y: (n,), 标签
    返回:
        best_threshold: float or None
        best_gain: float
    """
    n = len(y)
    # 按特征值排序
    sorted_indices = np.argsort(X_col)
    sorted_X = X_col[sorted_indices]
    sorted_y = y[sorted_indices]

    best_gain = -np.inf
    best_threshold = None

    # TODO:
    # 1. 计算父节点基尼
    # 2. for i in range(n - 1):
    #    a. if sorted_X[i] == sorted_X[i+1]: continue (跳过重复值)
    #    b. threshold = (sorted_X[i] + sorted_X[i+1]) / 2
    #    c. y_left = sorted_y[:i+1], y_right = sorted_y[i+1:]
    #    d. gain = 父基尼 - 加权子基尼
    #    e. 若 gain > best_gain: 更新 best_threshold 和 best_gain
    # 3. 返回 best_threshold, best_gain
    # --- BEGIN YOUR CODE ---
    gini_parent = compute_gini(y)
    for i in range(n - 1):
        if sorted_X[i] == sorted_X[i + 1]:
            continue
        threshold = (sorted_X[i] + sorted_X[i + 1]) / 2.0
        y_left = sorted_y[:i + 1]
        y_right = sorted_y[i + 1:]
        gini_left = compute_gini(y_left)
        gini_right = compute_gini(y_right)
        w_impurity = (len(y_left)/n)*gini_left + (len(y_right)/n)*gini_right
        gain = gini_parent - w_impurity
        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold
    # --- END YOUR CODE ---
    return best_threshold, best_gain


# ============================================================================
# 验证代码
# ============================================================================

def test_gini():
    """验证基尼指数计算。"""
    print("[测试 1] 基尼指数...")
    # 纯净数据集
    y_pure = np.array([0, 0, 0, 0])
    assert np.abs(compute_gini(y_pure) - 0.0) < 1e-6, "纯净集 Gini 不为 0"

    # 各半数据集 (二分类)
    y_mixed = np.array([0, 0, 1, 1])
    assert np.abs(compute_gini(y_mixed) - 0.5) < 1e-6, f"各半集 Gini 不为 0.5: {compute_gini(y_mixed)}"

    # 三类均匀
    y_three = np.array([0, 1, 2])
    expected = 1 - (1/3)**2 - (1/3)**2 - (1/3)**2  # = 1 - 3/9 = 2/3
    assert np.abs(compute_gini(y_three) - expected) < 1e-6, f"三类 Gini error: {compute_gini(y_three)}"
    print(f"  [PASS] 基尼指数计算正确!")


def test_information_gain():
    """验证信息增益。"""
    print("[测试 2] 信息增益...")
    # 一次完美分裂: 分裂后左右完全纯净
    y_parent = np.array([0, 0, 1, 1])
    y_left = np.array([0, 0])
    y_right = np.array([1, 1])

    gain = information_gain(y_parent, y_left, y_right)
    # parent Gini = 0.5, 左=0, 右=0
    # weighted = (2/4)*0 + (2/4)*0 = 0
    # gain = 0.5 - 0 = 0.5
    assert np.abs(gain - 0.5) < 1e-6, f"完美分裂的 IG 不为 0.5: {gain}"
    print(f"  [PASS] 完美分裂的 IG = {gain:.4f}")


def test_best_split():
    """验证最佳分裂点搜索。"""
    print("[测试 3] 最佳分裂点...")
    # 构建一个简单场景：前半部分 class 0, 后半部分 class 1
    X_col = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([0, 0, 1, 1])

    thresh, gain = best_split_single_feature(X_col, y)

    assert thresh is not None, "未找到分裂点"
    # 最佳分裂应在 2.0 和 3.0 之间 = 2.5
    assert 2.0 < thresh < 3.0, f"分裂阈值错误: {thresh}"
    print(f"  [PASS] 最佳阈值 = {thresh:.4f}, gain = {gain:.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("ml05_decision_tree exercise.py — 决策树练习")
    print("=" * 60)

    try:
        test_gini()
        test_information_gain()
        test_best_split()
        print("\n" + "=" * 60)
        print("所有测试通过!")
        print("=" * 60)
    except Exception as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
