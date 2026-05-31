# -*- coding: utf-8 -*-
"""
ml09 降维与特征工程 — 练习代码
===============================
请完成以下 TODO 任务，巩固对 PCA、降维方法和特征工程的理解。

每个 TODO 都有详细的指示和预期输出描述。
建议先阅读 index.md 和 code-demo.md，再尝试独立补全代码。
"""

import numpy as np


# ============================================================================
# TODO 1: 实现 PCA 的协方差矩阵 + 特征值分解方法
# ============================================================================
def pca_eigen(X, n_components=2):
    """
    使用协方差矩阵的特征值分解实现 PCA。

    步骤:
        1. 数据中心化: X_c = X - mean(X)
        2. 计算协方差矩阵: S = (1/n) X_c^T X_c
        3. 特征分解: eigenvalues, eigenvectors = np.linalg.eigh(S)
        4. 取最大的 k 个特征值对应的特征向量
        5. 投影: X_c @ W

    参数:
        X: 数据矩阵 shape (n_samples, n_features)
        n_components: 保留的主成分数

    返回:
        X_transformed: 降维后的数据 shape (n_samples, n_components)
        components: 主成分方向 shape (n_components, n_features)
        explained_variance_ratio: 方差解释率 shape (n_components,)
    """
    # TODO: 步骤 1 —— 数据中心化
    X_c = None  # ← TODO: X - X.mean(axis=0)

    # TODO: 步骤 2 —— 计算协方差矩阵
    # 提示: S = (X_c.T @ X_c) / n_samples 或 np.cov(X_c.T) (后者会自动除以 n-1)
    n = X.shape[0]
    S = None  # ← TODO: (X_c.T @ X_c) / (n - 1)

    # TODO: 步骤 3 —— 特征值分解
    # 提示: 使用 np.linalg.eigh (专门用于对称矩阵，返回按升序排列的特征值)
    eigenvalues, eigenvectors = None, None  # ← TODO

    # TODO: 步骤 4 —— 取最大的 k 个特征值对应的特征向量（从后往前取）
    # 提示: eigh 返回升序排列，所以最大的在最后
    top_indices = None  # ← TODO: np.argsort(eigenvalues)[::-1][:n_components]
    components = None   # ← TODO: eigenvectors[:, top_indices].T  注意转置
    top_eigenvalues = None  # ← TODO: eigenvalues[top_indices]

    # TODO: 步骤 5 —— 计算方差解释率
    explained_variance_ratio = None  # ← TODO: top_eigenvalues / eigenvalues.sum()

    # TODO: 步骤 6 —— 投影
    X_transformed = None  # ← TODO: X_c @ components.T

    return X_transformed, components, explained_variance_ratio


def test_pca_eigen():
    """测试 PCA 协方差矩阵实现"""
    print("=" * 60)
    print("TODO 1 测试: PCA 协方差矩阵实现")
    print("=" * 60)
    np.random.seed(42)
    X = np.random.randn(100, 5)
    X_trans, comps, var_ratio = pca_eigen(X, n_components=2)

    if X_trans is None:
        print("  TODO 1 未完成，请补全 pca_eigen 函数")
    else:
        print(f"  降维后形状: {X_trans.shape} (期望: (100, 2))")
        print(f"  主成分方向形状: {comps.shape} (期望: (2, 5))")
        print(f"  方差解释率: {var_ratio} (和应为 ≤ 1.0)")
        # 验证：方差解释率非负且和为 ≤ 1
        assert np.all(var_ratio >= 0), "方差解释率必须非负"
        assert var_ratio.sum() <= 1.0 + 1e-10, "累积解释率不应超过 1.0"


# ============================================================================
# TODO 2: 实现 LDA 的散布矩阵计算
# ============================================================================
def compute_scatter_matrices(X, y):
    """
    计算 LDA 所需的类内散布矩阵 S_W 和类间散布矩阵 S_B。

    定义:
        S_W = sum_{c} sum_{i: y_i=c} (x_i - mu_c)(x_i - mu_c)^T
        S_B = sum_{c} N_c (mu_c - mu)(mu_c - mu)^T

    参数:
        X: 数据矩阵 shape (n_samples, d)
        y: 标签向量 shape (n_samples,)

    返回:
        S_W: 类内散布矩阵 (d, d)
        S_B: 类间散布矩阵 (d, d)
        mu: 全局均值 (d,)
    """
    n, d = X.shape
    classes = np.unique(y)
    mu = X.mean(axis=0)  # 全局均值

    # TODO: 初始化散布矩阵为零矩阵
    S_W = None  # ← TODO: np.zeros((d, d))
    S_B = None  # ← TODO: np.zeros((d, d))

    for c in classes:
        # TODO: 获取第 c 类的样本
        X_c = None  # ← TODO: X[y == c]
        n_c = X_c.shape[0]

        # TODO: 计算第 c 类的均值 mu_c
        mu_c = None  # ← TODO: X_c.mean(axis=0)

        # TODO: 累加到 S_W: sum((x - mu_c)(x - mu_c)^T)
        # 提示: 对 X_c 的每一行做外积然后求和
        # 高效实现: (X_c - mu_c).T @ (X_c - mu_c)
        S_W += None  # ← TODO

        # TODO: 累加到 S_B: n_c * (mu_c - mu)(mu_c - mu)^T
        diff = None  # ← TODO: (mu_c - mu).reshape(-1, 1)  # (d, 1)
        S_B += None  # ← TODO: n_c * (diff @ diff.T)

    return S_W, S_B, mu


def test_scatter_matrices():
    """测试散布矩阵计算"""
    print("\n" + "=" * 60)
    print("TODO 2 测试: LDA 散布矩阵")
    print("=" * 60)
    np.random.seed(42)
    X = np.vstack([
        np.random.randn(30, 3) + np.array([0, 0, 0]),
        np.random.randn(30, 3) + np.array([3, 3, 0]),
    ])
    y = np.array([0]*30 + [1]*30)
    S_W, S_B, mu = compute_scatter_matrices(X, y)

    if S_W is None:
        print("  TODO 2 未完成，请补全 compute_scatter_matrices 函数")
    else:
        print(f"  S_W shape: {S_W.shape} (期望: (3, 3))")
        print(f"  S_B shape: {S_B.shape} (期望: (3, 3))")
        print(f"  全局均值 mu: {mu}")
        print(f"  S_B 的迹 (类间总方差): {np.trace(S_B):.2f}")
        print(f"  S_W 的迹 (类内总方差): {np.trace(S_W):.2f}")


# ============================================================================
# TODO 3: 实现简单的分箱特征（等宽分箱）
# ============================================================================
def equal_width_binning(x, n_bins=4):
    """
    对连续特征 x 做等宽分箱，返回 bin 标签和 One-Hot 编码。

    参数:
        x: 一维数组 (n_samples,)
        n_bins: 箱子数量

    返回:
        bin_labels: 每个样本所属的箱子编号 (n_samples,)，范围 [0, n_bins-1]
        one_hot: One-Hot 编码矩阵 (n_samples, n_bins)
    """
    # TODO: 步骤 1 —— 计算每个箱子的边界
    x_min, x_max = x.min(), x.max()
    bins = None  # ← TODO: np.linspace(x_min, x_max, n_bins + 1)

    # TODO: 步骤 2 —— 分配每个样本到对应的箱子
    # 提示: 使用 np.digitize(x, bins[:-1]) - 1
    # 或者 np.searchsorted
    bin_labels = None  # ← TODO

    # 确保标签范围在 [0, n_bins-1]
    bin_labels = np.clip(bin_labels, 0, n_bins - 1)

    # TODO: 步骤 3 —— 转 One-Hot 编码
    # 提示: 创建一个全零矩阵，在每行对应标签位置置 1
    one_hot = None  # ← TODO: np.zeros((len(x), n_bins))
    # one_hot[np.arange(len(x)), bin_labels] = 1

    return bin_labels, one_hot


def test_binning():
    """测试分箱"""
    print("\n" + "=" * 60)
    print("TODO 3 测试: 等宽分箱")
    print("=" * 60)
    x = np.array([0.1, 2.3, 5.0, 7.8, 9.9])
    bin_labels, one_hot = equal_width_binning(x, n_bins=3)

    if bin_labels is None:
        print("  TODO 3 未完成，请补全 equal_width_binning 函数")
    else:
        print(f"  输入: {x}")
        print(f"  箱子标签: {bin_labels}")
        print(f"  One-Hot:\n{one_hot}")


# ============================================================================
# TODO 4: 实现交互特征生成
# ============================================================================
def generate_interaction_features(X, max_degree=2):
    """
    生成所有度 ≤ max_degree 的多项式交互特征。

    对于 d=3, degree=2 的特征: 1, x1, x2, x3, x1^2, x1x2, x1x3, x2^2, x2x3, x3^2

    参数:
        X: 数据矩阵 (n, d)
        max_degree: 最大交互度

    返回:
        X_poly: 多项式特征矩阵 (n, 1 + d + d(d+1)/2 + ...)
        feature_names: 每个特征的名字列表
    """
    from itertools import combinations_with_replacement

    n, d = X.shape
    feature_names = ['1']  # 偏置项（截距）
    X_poly_cols = [np.ones(n)]  # 全 1 列

    for degree in range(1, max_degree + 1):
        # TODO: 生成所有度 = degree 的特征组合
        # 提示: 使用 combinations_with_replacement(range(d), degree)
        for combo in None:  # ← TODO
            # combo 是 (i1, i2, ...)，计算 x_i1 * x_i2 * ...
            new_feature = None  # ← TODO: np.prod(X[:, combo], axis=1)
            X_poly_cols.append(new_feature)

            # 生成特征名
            name = None  # ← TODO: '*'.join([f'x{i}' for i in combo])
            feature_names.append(name)

    X_poly = np.column_stack(X_poly_cols)
    return X_poly, feature_names


def test_interaction():
    """测试交互特征"""
    print("\n" + "=" * 60)
    print("TODO 4 测试: 交互特征生成")
    print("=" * 60)
    X = np.array([[1., 2.], [3., 4.]])
    X_poly, names = generate_interaction_features(X, max_degree=2)

    if X_poly.shape[1] == 1:  # 只有偏置列
        print("  TODO 4 未完成，请补全 generate_interaction_features 函数")
    else:
        print(f"  输入: (2, 2)")
        print(f"  输出形状: {X_poly.shape} (期望: (2, 6))")
        print(f"  特征名: {names}")
        print(f"  X_poly:\n{X_poly}")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == '__main__':
    print("ml09 降维与特征工程 — 练习代码")
    print("请完成所有 TODO 标记的代码")
    print()

    test_pca_eigen()
    test_scatter_matrices()
    test_binning()
    test_interaction()

    print("\n全部测试完成！")
