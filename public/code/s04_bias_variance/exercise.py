# -*- coding: utf-8 -*-
"""
===============================================================================
s04_bias_variance/code/exercise.py — 过拟合、正则化与 Bias-Variance 练习
===============================================================================
本练习文件中，你需要完成以下任务：

练习目标：
  1. 实现 L2 正则化的梯度计算
  2. 实现 K-Fold 交叉验证的数据划分
  3. 通过多次训练计算 Bias² 和 Variance 的分解（Bonus）

提示：
  - L2 正则化梯度: ∂(λ·Σwⱼ²)/∂wⱼ = 2λ·wⱼ（偏置项不参与正则化）
  - L1 正则化梯度: ∂(λ·Σ|wⱼ|)/∂wⱼ = λ·sign(wⱼ)
  - 总梯度 = MSE 梯度 + 正则化梯度
  - K-Fold: 将数据分成 K 等份，每次用 1 份验证，K-1 份训练

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


# ============================================================================
# 辅助函数（已实现，可直接使用）
# ============================================================================

def generate_sine_data(n_samples=80, noise_std=0.3, random_seed=42):
    """生成 y = sin(2πx) + noise 的数据。"""
    np.random.seed(random_seed)
    X = np.random.uniform(0, 1, n_samples)
    X = np.sort(X)
    y = np.sin(2 * np.pi * X) + np.random.randn(n_samples) * noise_std
    return X, y


def polynomial_features(X, degree):
    """将一维特征 X 扩展为多项式特征矩阵 [1, x, x², ..., x^{degree}]。"""
    X = X.reshape(-1, 1)
    return np.hstack([X ** d for d in range(degree + 1)])


# ======================================================================
# TODO 1: 实现 L2 正则化的梯度
# ======================================================================
# 带 L2 正则化的损失函数:
#   J(w) = MSE + λ * Σ wⱼ²
#
# 总梯度:
#   ∂J/∂w = ∂(MSE)/∂w + ∂(λ·Σ wⱼ²)/∂w
#         = (2/n) * X^T @ (Xw - y) + 2λ * w
#
# 注意：偏置项（w[0]）通常不参与正则化！
# 因为偏置不反映模型复杂度，只影响截距。
# 所以对于 w[0]，正则化梯度为 0。
#
# 提示：
#   - 先计算 MSE 部分的梯度: dw_mse = (2/n) * X.T @ (Xw - y)
#   - 再计算 L2 正则化梯度: dw_l2 = 2 * lambda_ * w
#   - 偏置项 dw_l2[0] = 0
#   - 返回 dw_mse + dw_l2
# ======================================================================

def compute_regularized_gradient(X, y, w, lambda_, reg_type='l2'):
    """
    计算带正则化的梯度。

    参数:
        X: np.ndarray, 特征矩阵 (n, d)，包含常数项列
        y: np.ndarray, 目标值 (n,)
        w: np.ndarray, 当前权重 (d,)
        lambda_: float, 正则化强度 λ
        reg_type: str, 'l2' 或 'l1'

    返回:
        dw: np.ndarray, 总梯度 (d,)
    """
    n = len(y)

    # TODO: 实现带正则化的梯度计算
    # 步骤 1: 计算 MSE 部分的梯度 dw_mse = (2/n) * X^T @ (X @ w - y)
    # 步骤 2: 创建正则化梯度数组 dw_reg，初始化为零
    # 步骤 3: 根据 reg_type 填充 dw_reg[1:]（跳过偏置项 w[0]）
    #   - 如果是 'l2': dw_reg[1:] = 2 * lambda_ * w[1:]
    #   - 如果是 'l1': dw_reg[1:] = lambda_ * sign(w[1:])
    # 步骤 4: 返回 dw_mse + dw_reg
    pass  # <-- 替换为你的代码


# ======================================================================
# TODO 2: 实现 K-Fold 交叉验证的数据划分
# ======================================================================
# K-Fold 交叉验证的流程:
#   1. 将数据分成 K 等份（折）
#   2. 对于每次迭代 i = 0, 1, ..., K-1:
#      a. 第 i 份作为验证集
#      b. 其余 K-1 份作为训练集
#      c. 用训练集训练模型，用验证集评估
#   3. 返回 K 次评估结果的平均值
#
# 例如：n=100, k=5，每份 20 个样本
#   - 迭代 1: 验证=索引 0-19,   训练=索引 20-99
#   - 迭代 2: 验证=索引 20-39,  训练=索引 0-19, 40-99
#   - ...
#   - 迭代 5: 验证=索引 80-99,  训练=索引 0-79
# ======================================================================

def kfold_cross_validation(X, y, k=5, degree=3):
    """
    执行 K-Fold 交叉验证，返回平均验证 MSE。

    参数:
        X: np.ndarray, 特征 (n,)
        y: np.ndarray, 目标值 (n,)
        k: int, 折数
        degree: int, 多项式次数

    返回:
        avg_val_mse: float, K 次验证的平均 MSE
        val_mses: list, 每次验证的 MSE
    """
    n = len(X)
    fold_size = n // k  # 每折的基础大小
    val_mses = []  # 存储每次验证的 MSE

    # TODO: 实现 K-Fold 交叉验证
    # 伪代码：
    # for i in range(k):
    #     # 确定验证集的起止索引
    #     val_start = i * fold_size
    #     val_end = (i + 1) * fold_size if i < k - 1 else n
    #
    #     # 创建验证集索引和训练集索引
    #     val_idx = np.arange(val_start, val_end)
    #     # 训练集索引：0 到 n-1 中排除 val_idx
    #     # 提示：使用 np.where 或 np.setdiff1d
    #
    #     # 划分数据
    #     X_train, y_train = X[train_idx], y[train_idx]
    #     X_val, y_val = X[val_idx], y[val_idx]
    #
    #     # 生成多项式特征
    #     Phi_train = polynomial_features(X_train, degree)
    #     Phi_val = polynomial_features(X_val, degree)
    #
    #     # 使用正规方程训练
    #     theta = np.linalg.inv(Phi_train.T @ Phi_train) @ Phi_train.T @ y_train
    #
    #     # 验证集预测并计算 MSE
    #     y_pred = Phi_val @ theta
    #     val_mse = np.mean((y_pred - y_val) ** 2)
    #     val_mses.append(val_mse)

    pass  # <-- 替换为你的代码

    # 返回平均 MSE 和所有验证 MSE
    # avg_val_mse = np.mean(val_mses)
    # return avg_val_mse, val_mses


# ======================================================================
# TODO 3 (Bonus): 计算 Bias² 和 Variance 的分解
# ======================================================================
# 通过对多个训练集训练多个模型，可以经验性地估计 Bias² 和 Variance。
#
# 算法（经验性估计）：
#   1. 生成 M 个不同的训练集（从真实分布中独立采样）
#   2. 对每个训练集训练一个模型 f̂_m(x)
#   3. 对每个测试点 x:
#      a. 计算 M 个模型的平均预测: E[f̂(x)] ≈ (1/M) Σ f̂_m(x)
#      b. Bias²(x) = (E[f̂(x)] - f(x))² — 平均预测与真实值的差距
#      c. Variance(x) = (1/M) Σ (f̂_m(x) - E[f̂(x)])² — 各模型预测的离散度
#   4. 对整个测试集取平均
#
# 这帮助我们理解: 高偏差 → 模型太简单, 高方差 → 模型太复杂
# ======================================================================

def compute_bias_variance(X_test, y_true, n_trials=100, degree=3,
                          noise_std=0.3):
    """
    通过多次训练来估计 Bias² 和 Variance。

    参数:
        X_test: np.ndarray, 测试集特征 (m,)
        y_true: np.ndarray, 测试集真实值 (m,)，无噪声的 f(x)
        n_trials: int, 重复训练的次数
        degree: int, 多项式次数
        noise_std: float, 每次重采样时添加的噪声标准差

    返回:
        avg_bias_sq: float, 平均 Bias²
        avg_variance: float, 平均 Variance
        predictions: np.ndarray, (n_trials, m)，所有模型的预测
    """
    n_test = len(X_test)
    predictions = np.zeros((n_trials, n_test))  # 存储 M 次训练的预测

    # TODO (Bonus): 实现 Bias-Variance 的经验估计
    # 伪代码：
    # for trial in range(n_trials):
    #     # 生成新的训练数据（基于真实函数 + 新噪声）
    #     y_noisy = y_true + noise_std * np.random.randn(n_test)
    #     # 训练模型
    #     Phi_test = polynomial_features(X_test, degree)
    #     theta = np.linalg.inv(Phi_test.T @ Phi_test) @ Phi_test.T @ y_noisy
    #     predictions[trial] = Phi_test @ theta
    #
    # # 计算每个测试点的平均预测
    # mean_preds = np.mean(predictions, axis=0)  # (m,)
    #
    # # 计算 Bias²: 平均预测与真实值的平方差
    # # bias_sq = np.mean((mean_preds - y_true) ** 2)
    #
    # # 计算 Variance: 各模型预测的方差
    # # variance = np.mean(np.var(predictions, axis=0))
    #
    # return bias_sq, variance, predictions

    pass  # <-- 替换为你的代码


# ============================================================================
# 测试代码
# ============================================================================

def test_l2_gradient():
    """测试 L2 正则化梯度的实现。"""
    print("--- 测试 L2 正则化梯度 ---")

    # 创建简单的测试数据
    X = np.array([[1, 1], [2, 1], [3, 1]])  # 含偏置列的全 1
    y = np.array([2, 4, 6])
    w = np.array([0.5, 0.5])  # [w1, bias]
    lambda_ = 0.1

    try:
        dw = compute_regularized_gradient(X, y, w, lambda_, reg_type='l2')
    except Exception as e:
        print(f"✗ 函数出错: {e}")
        return

    print(f"  总梯度 dw = {dw}")
    # 检查偏置项的梯度是否不包含正则化分量
    if abs(dw[1] - (-2.0)) < 1.0:
        print(f"  偏置梯度 dw[1] = {dw[1]:.4f} (L2 正则化不应影响偏置)")
    else:
        print(f"  偏置梯度 dw[1] = {dw[1]:.4f}，请确认偏置未被正则化")
    print("  ✓ 测试通过（请手动验证梯度值是否合理）")


def test_kfold():
    """测试 K-Fold 交叉验证的实现。"""
    print("\n--- 测试 K-Fold 交叉验证 ---")

    X, y = generate_sine_data(n_samples=60, random_seed=0)

    try:
        avg_mse, val_mses = kfold_cross_validation(X, y, k=5, degree=3)
    except Exception as e:
        print(f"✗ 函数出错: {e}")
        return

    if len(val_mses) == 5:
        print(f"  ✓ 正确生成了 5 个验证误差: {[f'{m:.4f}' for m in val_mses]}")
        print(f"  ✓ 平均验证 MSE = {avg_mse:.4f}")
    else:
        print(f"  ✗ 期望 5 个验证误差，但得到了 {len(val_mses)} 个")


def test_bias_variance():
    """测试 Bias-Variance 分解的实现。"""
    print("\n--- 测试 Bias-Variance 分解 (Bonus) ---")

    X_test = np.linspace(0, 1, 50)
    y_true = np.sin(2 * np.pi * X_test)

    try:
        bias_sq, variance, predictions = compute_bias_variance(
            X_test, y_true, n_trials=30, degree=3, noise_std=0.3
        )
    except Exception as e:
        print(f"  (Bonus 未完成或出错: {e})")
        return

    print(f"  Bias² = {bias_sq:.6f}")
    print(f"  Variance = {variance:.6f}")
    print(f"  预测形状: {predictions.shape}")
    print(f"  ✓ Bias-Variance 分解完成！")


def main():
    """主函数：运行所有测试。"""
    print("=" * 60)
    print("过拟合与正则化练习 — 请完成代码中的 TODO 标记")
    print("=" * 60)

    test_l2_gradient()
    test_kfold()
    test_bias_variance()

    print("\n" + "=" * 60)
    print("练习结束！如果测试通过，你的实现基本正确。")
    print("=" * 60)


if __name__ == '__main__':
    main()
