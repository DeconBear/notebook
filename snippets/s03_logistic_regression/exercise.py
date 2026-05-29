# -*- coding: utf-8 -*-
"""
===============================================================================
s03_logistic_regression/code/exercise.py — 逻辑回归练习
===============================================================================
本练习文件中，逻辑回归的核心函数被替换为了 TODO 注释。
你的任务是完成以下内容：

练习目标：
  1. 实现 Sigmoid 函数及其导数性质的理解
  2. 计算交叉熵损失及其梯度
  3. 实现 Softmax 函数用于多分类扩展

提示：
  - Sigmoid: σ(z) = 1 / (1 + e^{-z})
  - 交叉熵: L = -(1/n) Σ [y log(ŷ) + (1-y) log(1-ŷ)]
  - Softmax: softmax(z_k) = e^{z_k} / Σ e^{z_j}
  - 关键梯度: ∂L/∂z = ŷ - y (Sigmoid + 交叉熵的优美结果)

运行方式：
  python exercise.py
===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# 中文字体配置
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 修复负号显示


# ======================================================================
# TODO 1: 实现 Sigmoid 函数
# ======================================================================
# Sigmoid 函数的定义为:
#   σ(z) = 1 / (1 + e^{-z})
#
# Sigmoid 将任意实数 z 映射到 (0, 1) 区间，可以解释为概率。
# 注意：当 z 非常大时 e^{-z} ≈ 0，当 z 非常小时 e^{-z} 可能溢出。
# 建议使用 np.clip(z, -500, 500) 来防止数值问题。
#
# 提示：直接使用 np.exp() 计算指数。
# ======================================================================
def sigmoid(z):
    """
    Sigmoid 激活函数。

    参数:
        z: np.ndarray, 输入值（标量、向量或矩阵）

    返回:
        np.ndarray, σ(z)，形状与 z 相同
    """
    # TODO: 实现 Sigmoid 函数 σ(z) = 1 / (1 + e^{-z})
    pass  # <-- 替换为你的代码


# ======================================================================
# TODO 2: 计算交叉熵损失
# ======================================================================
# 二元交叉熵损失的定义:
#   L = -(1/n) * Σ_{i=1}^{n} [y_i * log(ŷ_i) + (1-y_i) * log(1-ŷ_i)]
#
# 其中:
#   - ŷ_i = σ(w·x_i + b) 是第 i 个样本的预测概率
#   - y_i 是真实标签（0 或 1）
#   - n 是样本数
#
# 交叉熵衡量的是两个概率分布（预测 vs 真实）之间的差异。
# 当预测完全正确时（ŷ_i = y_i），损失为 0。
# 当预测完全错误时，损失接近无穷大。
#
# 注意：log(0) 是无穷大，需要对 ŷ 做裁剪（clip）。
# ======================================================================
def cross_entropy_loss(y_pred, y_true):
    """
    计算二元交叉熵损失。

    参数:
        y_pred: np.ndarray, 预测概率 ŷ，形状 (n,)
        y_true: np.ndarray, 真实标签 y，形状 (n,)，值为 0 或 1

    返回:
        float, 平均交叉熵损失
    """
    n = len(y_true)
    eps = 1e-15  # 防止 log(0)

    # TODO: 实现交叉熵损失计算
    # 步骤 1: 将 y_pred 裁剪到 [eps, 1-eps]（防止 log(0)）
    # 步骤 2: 使用公式 L = -(1/n) Σ [y*log(ŷ) + (1-y)*log(1-ŷ)]
    pass  # <-- 替换为你的代码


# ======================================================================
# TODO 3: 计算交叉熵损失的梯度
# ======================================================================
# 逻辑回归中，损失对原始得分 z 的梯度非常简洁:
#   ∂L/∂z = ŷ - y
#
# 然后利用链式法则:
#   ∂L/∂w = (1/n) * X^T @ (ŷ - y)   # 矩阵形式
#   ∂L/∂b = (1/n) * Σ (ŷ_i - y_i)   # 标量形式
#
# 这个简洁的形式是 Sigmoid + 交叉熵组合的数学之美：
# Sigmoid 的导数项 σ'(z) = σ(z)(1-σ(z)) 在链式法则中恰好被约掉。
# ======================================================================
def compute_gradients(X, y_true, y_pred):
    """
    计算交叉熵损失对 w 和 b 的梯度。

    参数:
        X: np.ndarray, 特征矩阵，形状 (n_samples, n_features)
        y_true: np.ndarray, 真实标签，形状 (n_samples,)
        y_pred: np.ndarray, 预测概率 ŷ，形状 (n_samples,)

    返回:
        dw: np.ndarray, ∂J/∂w，形状 (n_features,)
        db: float, ∂J/∂b
    """
    n = len(y_true)

    # TODO: 计算梯度
    # 步骤 1: 计算预测误差 errors = y_pred - y_true
    # 步骤 2: dw = (1/n) * X^T @ errors
    # 步骤 3: db = (1/n) * Σ errors
    pass  # <-- 替换为你的代码


# ======================================================================
# TODO 4 (Bonus): 实现 Softmax 函数
# ======================================================================
# Softmax 将 K 个原始得分转换为概率分布:
#   softmax(z)_k = e^{z_k} / Σ_{j=1}^{K} e^{z_j}
#
# 性质:
#   (1) 每个输出 ∈ (0, 1)
#   (2) 所有输出之和 = 1
#   (3) 保序: 如果 z_i > z_j，则 softmax(z_i) > softmax(z_j)
#
# 数值稳定技巧:
#   先减去最大值 z_stable = z - max(z)
#   这样最大的指数值为 e^0 = 1，不会溢出
#   注意 max 应该对每一行（每个样本）分别计算
# ======================================================================
def softmax(z):
    """
    Softmax 函数。

    参数:
        z: np.ndarray, 原始得分矩阵，形状 (n_samples, n_classes)

    返回:
        np.ndarray, 概率分布，形状 (n_samples, n_classes)
    """
    # TODO: 实现 Softmax
    # 步骤 1: 数值稳定——每行减去该行的最大值 z_stable = z - max(z, axis=1, keepdims=True)
    # 步骤 2: 计算指数 exp_z = exp(z_stable)
    # 步骤 3: 归一化——每行除以其总和
    pass  # <-- 替换为你的代码


# ============================================================================
# 测试代码——验证你的实现
# ============================================================================

def test_sigmoid():
    """测试 Sigmoid 函数的实现是否正确。"""
    print("--- 测试 Sigmoid ---")

    # 测试几个关键值
    z_vals = np.array([-10, -1, 0, 1, 10])
    try:
        s_vals = sigmoid(z_vals)
    except Exception as e:
        print(f"✗ Sigmoid 函数出错: {e}")
        return

    expected = np.array([4.54e-05, 0.2689, 0.5, 0.7311, 0.99995])
    if np.allclose(s_vals, expected, rtol=0.01):
        print("✓ Sigmoid 基本值正确！")
    else:
        print(f"✗ Sigmoid 输出: {s_vals}")
        print(f"  期望输出:    {expected}")

    # 测试边界条件：z=0 应该输出 0.5
    if abs(sigmoid(np.array([0.0]))[0] - 0.5) < 1e-6:
        print("✓ σ(0) = 0.5 正确！")
    else:
        print(f"✗ σ(0) = {sigmoid(np.array([0.0]))[0]}, 期望 0.5")


def test_cross_entropy():
    """测试交叉熵损失的实现是否正确。"""
    print("\n--- 测试交叉熵损失 ---")

    # 完美预测的情况
    y_pred_perfect = np.array([0.99, 0.01, 0.99])  # 几乎完美的预测
    y_true = np.array([1.0, 0.0, 1.0])
    try:
        loss_perfect = cross_entropy_loss(y_pred_perfect, y_true)
    except Exception as e:
        print(f"✗ 交叉熵函数出错: {e}")
        return

    print(f"✓ 完美预测时的损失: {loss_perfect:.6f} (应该很小)")

    # 完全错误预测的情况
    y_pred_bad = np.array([0.01, 0.99, 0.01])  # 完全错误的预测
    loss_bad = cross_entropy_loss(y_pred_bad, y_true)
    print(f"✓ 完全错误时的损失: {loss_bad:.6f} (应该很大)")

    if loss_bad > loss_perfect:
        print("✓ 错误预测的损失 > 正确预测的损失，逻辑正确！")
    else:
        print("✗ 损失比较不正常，请检查实现。")


def test_softmax():
    """测试 Softmax 函数的实现是否正确。"""
    print("\n--- 测试 Softmax ---")

    z = np.array([[2.0, 1.0, 0.1],   # 第一个样本：类别 0 得分最高
                   [0.1, 2.0, 1.0]])  # 第二个样本：类别 1 得分最高

    try:
        prob = softmax(z)
    except Exception as e:
        print(f"✗ Softmax 函数出错: {e}")
        return

    # 检查每行之和是否为 1
    row_sums = np.sum(prob, axis=1)
    if np.allclose(row_sums, 1.0):
        print(f"✓ 每行概率和为 1: {row_sums}")
    else:
        print(f"✗ 行和: {row_sums}, 期望: [1.0, 1.0]")

    # 检查概率值是否在 [0, 1] 之间
    if np.all(prob >= 0) and np.all(prob <= 1):
        print("✓ 所有概率值在 [0, 1] 区间内")
    else:
        print("✗ 概率值超出 [0, 1] 区间")

    # 检查对第一个样本，类别 0 的概率是否最高
    if np.argmax(prob[0]) == 0:
        print(f"✓ 样本 1 最大概率索引正确: {np.argmax(prob[0])}")
    else:
        print(f"✗ 样本 1 最大概率索引: {np.argmax(prob[0])}, 期望: 0")

    print(f"  样本 1 的概率分布: {prob[0]}")
    print(f"  样本 2 的概率分布: {prob[1]}")


def main():
    """主函数：运行所有测试。"""
    print("=" * 60)
    print("逻辑回归练习 — 请完成代码中的 TODO 标记")
    print("=" * 60)

    test_sigmoid()
    test_cross_entropy()
    test_softmax()

    print("\n" + "=" * 60)
    print("练习结束！如果所有测试通过，你的实现基本正确。")
    print("=" * 60)


if __name__ == '__main__':
    main()
