# -*- coding: utf-8 -*-
"""
as06 蛋白质结构预测玩具演示 — 练习代码
=====================================================
请完成以下 TODO 任务，巩固对"共进化信号 → 接触图 → 3D 结构"这条
AlphaFold 简化流水线的理解。

每个 TODO 都有详细的指示和预期输出描述。
建议先阅读 index.md 和 demo.py，再尝试独立补全代码。
运行方式：在 as06_alphafold/ 目录下执行 python code/exercise.py
"""

import numpy as np

ALPHABET_SIZE = 8


# ============================================================================
# TODO 1: 从 3D 坐标计算真实接触图
# ============================================================================
def compute_contact_map(coords: np.ndarray, threshold: float = 8.0) -> np.ndarray:
    """
    TODO 1: 根据残基的 3D 坐标计算接触图。

    接触图定义: contact[i, j] = 1 如果残基 i, j 的欧氏距离 < threshold，否则为 0。
    序列上的近邻残基 (|i-j| <= 1) 天然就是"接触"的（共价相连），
    这类信息没有价值，因此约定将其置为 0，不计入后续评估。

    参数:
        coords: 残基坐标，shape (N, 3)
        threshold: 接触距离阈值

    返回:
        contact: 0/1 矩阵，shape (N, N)
    """
    n = coords.shape[0]
    # TODO: 计算所有残基对之间的欧氏距离矩阵
    # 提示: diff = coords[:, None, :] - coords[None, :, :]   # (N, N, 3)
    #       dist = np.linalg.norm(diff, axis=-1)              # (N, N)
    dist = None  # ← TODO

    # TODO: 根据阈值生成 0/1 接触矩阵
    contact = None  # ← TODO: (dist < threshold).astype(np.float32)

    # TODO: 将序列近邻 (|i-j| <= 1) 置为 0
    # 提示: 双重循环，或用 np.abs(np.arange(n)[:,None] - np.arange(n)[None,:]) <= 1
    # for i in range(n):
    #     for j in range(max(0, i-1), min(n, i+2)):
    #         contact[i, j] = 0

    return contact, dist


def test_contact_map():
    """测试接触图计算。用一个简单的正方形+额外点验证。"""
    print("=" * 60)
    print("TODO 1 测试: 接触图计算")
    print("=" * 60)

    # 构造 5 个点：0-1-2-3 是一条直线（间距 2），4 与 0 靠得很近（模拟长程接触）
    coords = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        [6.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],   # 与残基 0 距离很近，但序号相差 4（长程接触）
    ])

    result = compute_contact_map(coords, threshold=3.0)
    if result is None or result[0] is None:
        print("  TODO 未完成: compute_contact_map 未返回有效结果")
        return

    contact, dist = result
    print(f"  接触矩阵:\n{contact}")

    # 期望: contact[0, 4] = 1 (真实长程接触), contact[0,1] = 0 (近邻被屏蔽)
    if contact[0, 1] == 0 and contact[0, 4] == 1:
        print("  ✓ 正确: 序列近邻(0,1)被屏蔽为0，长程接触(0,4)被正确识别为1")
    else:
        print(f"  ✗ 结果不符合预期: contact[0,1]={contact[0,1]} (期望0), "
              f"contact[0,4]={contact[0,4]} (期望1)")
    print()


# ============================================================================
# TODO 2: 用 Outer Product Mean 思想提取共进化耦合矩阵
# ============================================================================
def outer_product_mean_coupling(msa: np.ndarray, n_residues: int) -> np.ndarray:
    """
    TODO 2: 从合成 MSA 中提取残基对的共进化耦合强度（AlphaFold Evoformer 中
    Outer Product Mean 操作的极简教学版）。

    核心思想:
        1. 把每个位置的字符做 one-hot 编码: one_hot[m, i, :] 是序列 m 在位置 i 的 one-hot 向量
        2. 单列频率 freq[i, :] = one_hot 在序列维度上的均值 (即 P(字符 | 位置 i))
        3. 对每一对位置 (i, j):
           - 联合频率 joint = (1/M) * sum_m outer(one_hot[m,i,:], one_hot[m,j,:])  ← 这就是"外积均值"
           - 若 i, j 独立，期望的联合频率应为 outer(freq[i], freq[j])
           - 耦合强度 = ||joint - 独立假设期望值||（Frobenius 范数），越大说明越不独立，
             暗示 i, j 可能在结构上耦合（接触）

    参数:
        msa: shape (M, N)，M 条序列，整数编码 (0 ~ ALPHABET_SIZE-1)
        n_residues: N

    返回:
        coupling: shape (N, N) 耦合强度矩阵
    """
    m, n = msa.shape

    # ---- 步骤 1: one-hot 编码 ----
    one_hot = np.zeros((m, n, ALPHABET_SIZE), dtype=np.float32)
    rows = np.arange(m)[:, None]
    cols = np.arange(n)[None, :]
    one_hot[rows, cols, msa] = 1.0

    # ---- 步骤 2: 单列频率 ----
    # TODO: 计算每个位置上各字符出现的频率，shape (N, ALPHABET_SIZE)
    # 提示: one_hot.mean(axis=0)
    freq = None  # ← TODO: one_hot.mean(axis=0)

    if freq is None:
        return None

    coupling = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            # ---- 步骤 3: 外积均值 (联合频率) ----
            # TODO: 计算 joint[a, b] = (1/M) * sum_m one_hot[m,i,a] * one_hot[m,j,b]
            # 提示: (one_hot[:, i, :, None] * one_hot[:, j, None, :]).mean(axis=0)  # (K, K)
            joint = None  # ← TODO

            # TODO: 若独立，期望的联合频率 = outer(freq[i], freq[j])
            indep = None  # ← TODO: np.outer(freq[i], freq[j])

            # TODO: 耦合强度 = Frobenius 范数 ||joint - indep||
            score = None  # ← TODO: np.linalg.norm(joint - indep)

            if joint is None or indep is None or score is None:
                return None
            coupling[i, j] = coupling[j, i] = score
    return coupling


def test_outer_product_mean():
    """
    测试 Outer Product Mean 耦合计算。
    构造一个"人为强耦合"的 MSA：位置 0 和位置 1 的字符完全同步变化，
    位置 2 完全独立随机——预期耦合分数 coupling[0,1] 应远大于 coupling[0,2]。
    """
    print("=" * 60)
    print("TODO 2 测试: Outer Product Mean 耦合矩阵")
    print("=" * 60)

    rng = np.random.RandomState(0)
    m_seqs, n_pos = 200, 3
    msa = np.zeros((m_seqs, n_pos), dtype=np.int64)
    shared = rng.randint(0, ALPHABET_SIZE, size=m_seqs)
    msa[:, 0] = shared             # 位置 0
    msa[:, 1] = shared             # 位置 1: 与位置 0 完全同步（强耦合）
    msa[:, 2] = rng.randint(0, ALPHABET_SIZE, size=m_seqs)  # 位置 2: 完全独立

    coupling = outer_product_mean_coupling(msa, n_pos)
    if coupling is None:
        print("  TODO 未完成: outer_product_mean_coupling 未正确实现")
        return
    if coupling[0, 1] == 0 and coupling[0, 2] == 0:
        print("  TODO 未完成: outer_product_mean_coupling 返回了全零矩阵")
        return

    print(f"  coupling[0,1] (强耦合对) = {coupling[0, 1]:.4f}")
    print(f"  coupling[0,2] (独立对)   = {coupling[0, 2]:.4f}")
    if coupling[0, 1] > 3 * coupling[0, 2]:
        print("  ✓ 正确: 强耦合对的分数显著高于独立对，符合共进化直觉")
    else:
        print("  ✗ 结果不符合预期: 强耦合对分数应显著高于独立对")
    print()


# ============================================================================
# TODO 3: 实现距离几何重建的梯度计算（简化结构模块）
# ============================================================================
def distance_geometry_gradient(
    coords: np.ndarray,
    target_dist: np.ndarray,
    weight: np.ndarray,
) -> np.ndarray:
    """
    TODO 3: 计算"用 3D 坐标满足目标距离约束"这一损失函数的梯度。

    损失函数:
        L(X) = sum_{i<j} w_ij * (||x_i - x_j|| - d_ij^target)^2

    梯度（对每个残基坐标 x_i）:
        dL/dx_i = sum_j 2 * w_ij * (||x_i - x_j|| - d_ij^target) * (x_i - x_j) / ||x_i - x_j||

    参数:
        coords: 当前坐标估计，shape (N, 3)
        target_dist: 目标距离矩阵，shape (N, N)
        weight: 权重矩阵，shape (N, N)

    返回:
        grad: 梯度，shape (N, 3)
    """
    # TODO: 计算两两差向量 diff[i,j] = coords[i] - coords[j]
    # 提示: coords[:, None, :] - coords[None, :, :]
    diff = None  # ← TODO, shape (N, N, 3)

    # TODO: 计算两两距离 (加一个小 epsilon=1e-8 避免除零)
    dist = None  # ← TODO: np.linalg.norm(diff, axis=-1) + 1e-8

    # TODO: 计算残差 err = dist - target_dist
    err = None  # ← TODO

    if diff is None or dist is None or err is None:
        return None

    # TODO: 计算系数 coef = 2 * weight * err / dist，并将对角线置零（自身不参与）
    coef = None  # ← TODO
    # np.fill_diagonal(coef, 0.0)

    # TODO: 梯度 = sum_j coef[i,j] * diff[i,j,:]，对 j 求和
    # 提示: (coef[:, :, None] * diff).sum(axis=1)
    grad = None  # ← TODO

    return grad


def test_distance_geometry_gradient():
    """
    测试距离几何梯度计算。
    构造一个简单的 2 点系统：当前距离为 4.0，目标距离为 2.0（需要拉近），
    验证梯度方向是否指向"让两点靠近"。
    """
    print("=" * 60)
    print("TODO 3 测试: 距离几何梯度计算")
    print("=" * 60)

    coords = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])   # 当前距离 = 4.0
    target_dist = np.array([[0.0, 2.0], [2.0, 0.0]])         # 目标距离 = 2.0（应该拉近）
    weight = np.array([[0.0, 1.0], [1.0, 0.0]])

    grad = distance_geometry_gradient(coords, target_dist, weight)
    if grad is None:
        print("  TODO 未完成，请补全 distance_geometry_gradient 函数")
        return

    print(f"  当前坐标: {coords.tolist()}")
    print(f"  梯度: {grad.tolist()}")
    # 当前距离(4.0) > 目标距离(2.0)，误差为正，梯度应该指向"减小距离"的方向：
    # 点0 的梯度 x 分量应为正（沿梯度下降会使点0向+x移动，即靠近点1？）
    # 实际上 dL/dx_0 = 2*w*err*(x0-x1)/dist = 2*1*(4-2)*(0-4)/4 = -4 (指向让x0增大，即靠近x1)
    # 梯度下降: x0_new = x0 - lr*grad[0] = 0 - lr*(-4) = 4*lr > 0，确实向x1靠近
    expected_sign = -1  # dL/dx_0 的 x 分量应为负（梯度下降会使坐标向正方向移动，即靠近点1）
    if grad[0, 0] < 0 and grad[1, 0] > 0:
        print("  ✓ 正确: 梯度方向表明梯度下降会让两点相互靠近（因为当前距离>目标距离）")
    else:
        print(f"  ✗ 梯度方向不符合预期: grad[0,0]={grad[0,0]:.3f} (期望<0), "
              f"grad[1,0]={grad[1,0]:.3f} (期望>0)")
    print()


# ============================================================================
# 主程序
# ============================================================================
if __name__ == "__main__":
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║   as06 蛋白质结构预测玩具演示 — 动手练习                     ║")
    print("║   请依次完成 TODO 1, 2, 3                                    ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    test_contact_map()
    test_outer_product_mean()
    test_distance_geometry_gradient()

    print("=" * 60)
    print("所有测试完成！请检查输出结果。")
    print("如有未通过的测试，请回到对应的 TODO 部分补全代码。")
    print("=" * 60)
    print()
    print("提示: 完成 TODO 后，运行 demo.py 查看完整的接触图预测")
    print("      与结构重建演示效果。")
    print("  python code/demo.py")
    print()
