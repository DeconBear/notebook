# -*- coding: utf-8 -*-
"""
===============================================================================
ml07_boosting/code/exercise.py — Boosting 练习
===============================================================================
本练习文件中，你需要完成以下任务：

练习目标：
  1. 实现 AdaBoost 的学习器权重计算 (alpha)
  2. 实现 AdaBoost 的样本权重更新
  3. 实现 GBDT 回归的单轮迭代（拟合残差）（Bonus）

提示：
  - alpha = 0.5 * log((1 - err) / err)
  - 权重更新: w_i *= exp(alpha * I(pred_i != y_i))
  - GBDT (MSE): residual = y - F, 训练树拟合残差, F += lr * tree.predict(X)

运行方式：
  python exercise.py
===============================================================================
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor


# ============================================================================
# 任务 1: 实现 AdaBoost 的 alpha 计算 (约 4 行)
# ============================================================================

def compute_alpha(error_rate):
    """
    计算 AdaBoost 中弱学习器的权重 alpha。

    公式: alpha = 0.5 * log((1 - epsilon) / epsilon)

    参数:
        error_rate: float, 加权错误率 epsilon (0 < epsilon <= 0.5)
    返回:
        alpha: float
    """
    # TODO:
    # 1. 防止除零: error_rate = max(error_rate, 1e-10)
    # 2. alpha = 0.5 * np.log((1.0 - error_rate) / error_rate)
    # 3. 返回 alpha
    # --- BEGIN YOUR CODE ---
    error_rate = max(error_rate, 1e-10)
    alpha = 0.5 * np.log((1.0 - error_rate) / error_rate)
    # --- END YOUR CODE ---
    return alpha


# ============================================================================
# 任务 2: 实现 AdaBoost 的样本权重更新 (约 6 行)
# ============================================================================

def update_weights(weights, alpha, incorrect_mask):
    """
    更新 AdaBoost 的样本权重。

    公式: w_i_new = w_i * exp(alpha * I(error))
    然后归一化: w_i_new = w_i_new / sum(w_i_new)

    参数:
        weights: (n,), 当前样本权重
        alpha: float, 本轮学习器的权重
        incorrect_mask: (n,), bool 数组，True 表示预测错误的样本
    返回:
        new_weights: (n,), 更新并归一化后的权重
    """
    # TODO:
    # 1. incorrect_float = incorrect_mask.astype(float)  (True→1, False→0)
    # 2. weights = weights * np.exp(alpha * incorrect_float)
    # 3. 归一化: weights = weights / np.sum(weights)
    # 4. 返回 weights
    # --- BEGIN YOUR CODE ---
    incorrect_float = incorrect_mask.astype(float)
    weights = weights * np.exp(alpha * incorrect_float)
    weights = weights / np.sum(weights)
    # --- END YOUR CODE ---
    return weights


# ============================================================================
# 任务 3 (Bonus): 实现 GBDT 回归的单轮迭代 (约 8 行)
# ============================================================================

def gbdt_mse_step(X, y_current_pred, learning_rate, max_depth):
    """
    执行 GBDT（MSE 损失）的一轮迭代。

    使用平方损失时，负梯度 = 残差 = y - y_current_pred。
    因此每轮只需:
      1. 计算残差
      2. 训练树拟合残差
      3. 更新预测

    参数:
        X: (n, d), 特征
        y_current_pred: (n,), 当前集成的预测值
        learning_rate: float
        max_depth: int
    返回:
        new_pred: (n,), 更新后的预测值
        tree: 训练好的决策树
    """
    y = ...  # 需要外部传入真实标签
    # 实际上这个函数需要 y 作为参数

    # TODO:
    # 计算残差并训练树（这个需要修改函数签名以获得 y）
    # 但在本次练习中你只需要理解流程即可
    pass
    # --- BEGIN YOUR CODE ---
    # residuals = y - y_current_pred
    # tree = DecisionTreeRegressor(max_depth=max_depth)
    # tree.fit(X, residuals)
    # new_pred = y_current_pred + learning_rate * tree.predict(X)
    # return new_pred, tree
    # --- END YOUR CODE ---


# ============================================================================
# 验证代码
# ============================================================================

def test_compute_alpha():
    """验证 alpha 计算。"""
    print("[测试 1] Alpha 计算...")

    # error = 0.1: alpha = 0.5 * log(0.9/0.1) = 0.5 * log(9) ≈ 1.099
    alpha = compute_alpha(0.1)
    expected = 0.5 * np.log(9.0)
    assert np.abs(alpha - expected) < 1e-6, f"alpha error: {alpha} vs {expected}"

    # error = 0.5: alpha = 0.5 * log(0.5/0.5) = 0
    alpha = compute_alpha(0.5)
    assert np.abs(alpha - 0.0) < 1e-6, f"error=0.5 时 alpha 应为 0: {alpha}"

    # error = 0.3: alpha > 0（正确率高于随机）
    alpha = compute_alpha(0.3)
    assert alpha > 0, f"error=0.3 时 alpha 应为正: {alpha}"

    print(f"  [PASS] alpha(0.1)={alpha:.4f}, alpha(0.5)={compute_alpha(0.5):.4f}")


def test_update_weights():
    """验证权重更新。"""
    print("[测试 2] 权重更新...")
    np.random.seed(42)

    weights = np.ones(10) / 10.0
    alpha = 1.0
    incorrect = np.array([True, False, False, True, False, False, False, False, False, False])

    new_weights = update_weights(weights.copy(), alpha, incorrect)

    # 错误样本的权重应增大
    err_weight_before = weights[incorrect].sum()
    err_weight_after = new_weights[incorrect].sum()
    assert err_weight_after > err_weight_before, \
        f"错误样本权重应增大: {err_weight_before:.4f} → {err_weight_after:.4f}"

    # 权重应归一化
    assert np.abs(np.sum(new_weights) - 1.0) < 1e-6, f"权重未归一化: {np.sum(new_weights)}"

    # 错误样本的权重应大于正确样本的单个权重
    assert new_weights[0] > new_weights[1], "错误样本权重应 > 正确样本权重"

    print(f"  [PASS] 错误样本权重之和: {err_weight_before:.4f} → {err_weight_after:.4f}")


def test_adaboost_simple():
    """测试完整 AdaBoost 流程在一个简单数据集上。"""
    print("[测试 3] AdaBoost 简单流程...")

    # 构造一个容易的数据集
    X = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 1.0], [4.0, 1.0],
                  [5.0, 0.0], [6.0, 0.0], [7.0, 1.0], [8.0, 1.0]])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    n_samples = len(y)
    weights = np.ones(n_samples) / n_samples

    # 模拟一个简单分类器(只按特征 0 的阈值分类)
    def simple_predict(X, threshold):
        return (X[:, 0] > threshold).astype(int)

    n_rounds = 5
    alphas = []
    errors = []

    for m in range(n_rounds):
        # 搜索最佳阈值（简化：只搜索几个阈值）
        best_err = np.inf
        best_thresh = None
        for thresh in [2.5, 3.5, 4.5, 5.5]:
            pred = simple_predict(X, thresh)
            incorrect = (pred != y).astype(float)
            err = np.sum(weights * incorrect) / np.sum(weights)
            if err < best_err:
                best_err = err
                best_thresh = thresh

        if best_thresh is None:
            break

        alpha = compute_alpha(best_err)
        alphas.append(alpha)

        pred = simple_predict(X, best_thresh)
        incorrect_mask = (pred != y)
        weights = update_weights(weights, alpha, incorrect_mask)

        errors.append(best_err)

    assert len(alphas) == n_rounds, f"应运行 {n_rounds} 轮"
    assert errors[-1] <= errors[0], "错误率应总体下降（但权重重新分布后不一定单调）"

    print(f"  [PASS] {n_rounds} 轮完成! 错误率: {[f'{e:.3f}' for e in errors]}")


if __name__ == "__main__":
    print("=" * 60)
    print("ml07_boosting exercise.py — Boosting 练习")
    print("=" * 60)

    try:
        test_compute_alpha()
        test_update_weights()
        test_adaboost_simple()
        print("\n" + "=" * 60)
        print("所有测试通过!")
        print("=" * 60)
    except Exception as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
