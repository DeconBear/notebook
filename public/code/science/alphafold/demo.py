# -*- coding: utf-8 -*-
"""
===============================================================================
as06_alphafold/code/demo.py — 蛋白质结构预测玩具演示：从"共进化"到"3D 坐标"
===============================================================================
重要声明：这不是 AlphaFold！

真实的 AlphaFold2 用一个在几十万条已知结构上训练过的深度网络
（Evoformer + Structure Module，数千万参数）来做结构预测。
本演示只用 NumPy 从零搭建一个"教学玩具"，目的是让你直观理解
AlphaFold 整条流水线背后的几个关键物理/统计直觉：

  1. 共进化信号（Co-evolution）：如果两个残基在结构上互相接触，
     它们在演化过程中的突变往往是"配对"的（一个变了，另一个也要跟着变，
     否则结构会被破坏）。这是"多序列比对 MSA → 残基接触"的信息来源，
     也是 Evoformer 里 MSA 表示与 Pair 表示相互通信的物理基础。
  2. Outer Product Mean：AlphaFold2 用一个叫 Outer Product Mean 的操作，
     把 MSA 表示（每条序列每个位置的特征）转换成 Pair 表示（残基对的特征）。
     本演示用最朴素的版本重现这个思想：对 MSA 每一列做 one-hot 编码，
     计算列与列之间的"外积均值"，得到一个粗糸的共进化耦合矩阵。
  3. Structure Module 的几何直觉：给定一组残基对之间的"距离约束"，
     如何反解出一组 3D 坐标？真实的 Structure Module 用 Invariant Point
     Attention 直接回归坐标；本演示用经典的"距离几何 / 应力多维标度
     （stress majorization）"梯度下降来做同样的事——从残基对约束重建 3D 结构，
     这正是 AlphaFold pipeline 里"从 Pair 表示到 3D 坐标"这一步的极简类比。

流程（对照真实 AlphaFold）：
    合成"真实"蛋白骨架 (ground truth，仅用于生成教学数据和评估)
        │
        ▼
    合成多序列比对 MSA  ←→  真实 AlphaFold: 用 HMM/搜索数据库得到 MSA
        │  (Outer Product Mean 提取共进化耦合)
        ▼
    预测接触图 (Pair 表示的简化版)  ←→  真实 AlphaFold: Evoformer 输出的 pair repr.
        │  (precision@L/precision@L/5 等 CASP 式指标评估)
        ▼
    简化"结构模块"：距离几何梯度下降重建 3D 坐标  ←→  真实 AlphaFold: Structure Module (IPA)
        │
        ▼
    与真实骨架比较 (Kabsch 对齐 + RMSD)

运行方式：在 as06_alphafold/ 目录下执行 python code/demo.py
依赖: pip install numpy matplotlib
随机种子固定为 42，结果可复现。
===============================================================================
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)


def _save(fig, name):
    path = os.path.join(_IMAGES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[可视化] 已保存至 images/{name}")


def set_seed(seed: int = 42):
    """固定随机种子，保证实验可复现。"""
    np.random.seed(seed)


# ============================================================================
# 第一部分：合成一个"真实"蛋白骨架（仅用于教学，不代表真实蛋白物理）
# ============================================================================

def generate_synthetic_backbone(n_residues: int = 40, seed: int = 42) -> np.ndarray:
    """
    生成一条合成的三维"骨架"坐标，模拟一个含有 α-螺旋 + 转角 + 延伸链的
    简化蛋白结构。这不是分子动力学模拟，只是为了得到一组有"局部规律 +
    长程接触"的 3D 坐标，供后续演示使用。

    参数:
        n_residues: 残基数量
        seed: 随机种子

    返回:
        coords: shape (n_residues, 3)，每个残基的 Cα 坐标（任意长度单位）
    """
    rng = np.random.RandomState(seed)
    coords = np.zeros((n_residues, 3))

    # 用键长约 3.8（Cα-Cα 典型距离）+ 螺旋/延伸角度模式生成骨架
    bond_len = 3.8
    pos = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    coords[0] = pos

    # 将序列分成 3 段：螺旋段、转角、延伸段，分别用不同的角度增量
    seg1 = n_residues // 3            # α-螺旋段：小角度规则旋转
    seg2 = 2 * n_residues // 3        # 转角：较大随机扭转
    for i in range(1, n_residues):
        if i < seg1:
            # α-螺旋：每步绕固定轴旋转约 100°，并沿螺旋轴前进
            theta = np.deg2rad(100)
            axis = np.array([0.0, 0.0, 1.0])
            radius = 2.3
            rise = 1.5
            angle = i * theta
            pos = np.array([radius * np.cos(angle), radius * np.sin(angle), rise * i])
        elif i < seg2:
            # 转角区域：加入随机扰动，让链回折，制造长程接触
            direction = direction + rng.normal(scale=0.6, size=3)
            direction = direction / np.linalg.norm(direction)
            pos = coords[i - 1] + bond_len * direction
        else:
            # 延伸段：基本沿直线前进，带小噪声，向回折的区域靠拢制造 β-sheet 式接触
            target = coords[seg1 // 2]  # 朝螺旋中段靠拢，制造跨段接触
            direction = 0.7 * direction + 0.3 * (target - coords[i - 1])
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            pos = coords[i - 1] + bond_len * direction + rng.normal(scale=0.3, size=3)
        coords[i] = pos

    # 居中
    coords -= coords.mean(axis=0)
    return coords


def compute_contact_map(coords: np.ndarray, threshold: float = 8.0) -> np.ndarray:
    """
    根据 3D 坐标计算真实接触图：两个残基的 Cα 距离小于阈值即为"接触"。

    参数:
        coords: shape (N, 3)
        threshold: 接触距离阈值（单位与坐标一致，AlphaFold 论文常用 8 Å）

    返回:
        contact: shape (N, N) 的 0/1 矩阵，对角线及近邻(|i-j|<=1)设为 0（不计入评估）
    """
    diff = coords[:, None, :] - coords[None, :, :]        # (N, N, 3)
    dist = np.linalg.norm(diff, axis=-1)                    # (N, N) 距离矩阵
    contact = (dist < threshold).astype(np.float32)
    n = coords.shape[0]
    for i in range(n):
        for j in range(max(0, i - 1), min(n, i + 2)):
            contact[i, j] = 0                                # 序列上的近邻天然"接触"，不计入评估
    return contact, dist


# ============================================================================
# 第二部分：合成多序列比对 (MSA)——模拟共进化信号
# ============================================================================

ALPHABET_SIZE = 8  # 简化的"氨基酸字母表"大小（真实氨基酸有 20 种，这里简化以加快演示）


def generate_synthetic_msa(
    n_residues: int,
    contact_map: np.ndarray,
    n_sequences: int = 300,
    mutation_rate: float = 0.35,
    coupling_strength: float = 0.85,
    seed: int = 42,
) -> np.ndarray:
    """
    合成一个"多序列比对"（MSA），模拟真实蛋白家族中的共进化现象：

    直觉：如果残基 i 和 j 在结构上接触，那么演化压力会让它们的突变"配对"——
    比如 i 从疏水变成带电时，j 往往也要发生互补突变以维持结构稳定，
    否则该突变个体会被自然选择淘汰。这就是 DCA（Direct Coupling Analysis）
    和 AlphaFold 的 MSA 通道都在利用的信号。

    这里用一个简化的生成过程：
      1. 先随机生成一条"祖先序列"
      2. 对每条后代序列，独立地在每个位置做随机突变
      3. 对于处于接触状态的残基对 (i, j)，以 coupling_strength 的概率
         强制让突变"配对"（用同一个随机偏移量同时改变两个位置的字符），
         模拟共进化约束

    参数:
        n_residues: 序列长度
        contact_map: 真实接触图，用于注入共进化信号（真实场景中我们不知道
                      这个矩阵——这里只是用它来"制造"具有可学习共进化信号的数据）
        n_sequences: MSA 序列条数
        mutation_rate: 每个位置发生随机突变的概率
        coupling_strength: 接触残基对突变配对的概率
        seed: 随机种子

    返回:
        msa: shape (n_sequences, n_residues)，每个元素是 0~ALPHABET_SIZE-1 的整数
    """
    rng = np.random.RandomState(seed + 1)
    ancestor = rng.randint(0, ALPHABET_SIZE, size=n_residues)
    msa = np.tile(ancestor, (n_sequences, 1))

    contact_pairs = [(i, j) for i in range(n_residues) for j in range(i + 1, n_residues)
                      if contact_map[i, j] > 0]

    for s in range(n_sequences):
        seq = msa[s].copy()
        # 独立随机突变
        mask = rng.random(n_residues) < mutation_rate
        seq[mask] = rng.randint(0, ALPHABET_SIZE, size=mask.sum())
        # 共进化配对突变：接触残基对按耦合强度联合突变
        for (i, j) in contact_pairs:
            if rng.random() < coupling_strength * mutation_rate:
                shift = rng.randint(1, ALPHABET_SIZE)
                seq[i] = (ancestor[i] + shift) % ALPHABET_SIZE
                seq[j] = (ancestor[j] + shift) % ALPHABET_SIZE   # 配对：相同偏移量 = 相关突变
        msa[s] = seq
    return msa


def outer_product_mean_coupling(msa: np.ndarray, n_residues: int) -> np.ndarray:
    """
    用"外积均值"（Outer Product Mean）思想，从 MSA 中提取残基对的共进化耦合强度。

    这是 AlphaFold2 Evoformer 中 MSA→Pair 通信的核心操作的极简教学版：
        真实版本：对 MSA 表示的每一列做线性投影后，计算逐序列的外积再取均值，
                 得到的张量被加到 Pair 表示上，让"序列层面的共进化统计"
                 直接影响"残基对层面的几何推理"。
        教学版本：把每个位置的字符做 one-hot 编码，计算列 i 和列 j 的
                 one-hot 向量的外积在所有序列上的均值，再用其"偏离独立性
                 的程度"（类似互信息）作为耦合强度分数——分数越高，
                 说明两列的字符分布越不独立，暗示它们在结构上可能接触。

    参数:
        msa: shape (M, N)，M 条序列，每条长度 N
        n_residues: N

    返回:
        coupling: shape (N, N)，耦合强度矩阵（越大代表越可能接触）
    """
    m, n = msa.shape
    # one-hot 编码: (M, N, K)
    one_hot = np.zeros((m, n, ALPHABET_SIZE), dtype=np.float32)
    rows = np.arange(m)[:, None]
    cols = np.arange(n)[None, :]
    one_hot[rows, cols, msa] = 1.0

    freq = one_hot.mean(axis=0)                              # (N, K) 单列频率 P(a_i)
    coupling = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(i + 1, n):
            # 联合频率 P(a_i, a_j)：外积均值 = (1/M) * sum_m outer(onehot_i, onehot_j)
            joint = (one_hot[:, i, :, None] * one_hot[:, j, None, :]).mean(axis=0)  # (K, K)
            indep = np.outer(freq[i], freq[j])                # 若独立时的期望联合频率
            # Frobenius 范数衡量 "联合分布偏离独立假设" 的程度 —— 互信息的简化替代
            score = np.linalg.norm(joint - indep)
            coupling[i, j] = coupling[j, i] = score
    return coupling


# ============================================================================
# 第三部分：接触预测评估（CASP 风格指标）
# ============================================================================

def evaluate_contact_precision(pred_coupling: np.ndarray, true_contact: np.ndarray) -> dict:
    """
    用 CASP 竞赛中常见的 precision@L, precision@L/2, precision@L/5 指标
    评估接触图预测质量：取预测分数最高的 top-k 个残基对（k = L, L/2, L/5，
    L 为序列长度），检验其中有多少真的是接触对。

    参数:
        pred_coupling: 预测的耦合/接触分数矩阵，shape (N, N)
        true_contact: 真实接触图（0/1），shape (N, N)

    返回:
        dict，包含各 top-k 下的 precision
    """
    n = pred_coupling.shape[0]
    # 只取上三角（排除对角线和近邻，contact map 中已置 0）
    iu = np.triu_indices(n, k=2)
    scores = pred_coupling[iu]
    labels = true_contact[iu]

    order = np.argsort(-scores)                              # 分数从高到低排序
    sorted_labels = labels[order]

    results = {}
    for frac, name in [(1.0, 'L'), (0.5, 'L/2'), (0.2, 'L/5')]:
        k = max(1, int(round(n * frac)))
        k = min(k, len(sorted_labels))
        precision = sorted_labels[:k].mean() if k > 0 else 0.0
        results[name] = (precision, k)
    return results


# ============================================================================
# 第四部分：简化"结构模块"——用距离几何重建 3D 坐标
# ============================================================================

def reconstruct_structure(
    pred_coupling: np.ndarray,
    n_residues: int,
    n_iters: int = 4000,
    lr: float = 0.02,
    n_restarts: int = 4,
    seed: int = 42,
) -> np.ndarray:
    """
    简化版"结构模块"：把预测的接触/耦合信息转换为目标距离约束，
    再用梯度下降（应力多维标度, stress majorization 的简化版）求解一组
    3D 坐标，使残基对之间的实际距离尽量满足这些约束。

    真实 AlphaFold 的 Structure Module 用 Invariant Point Attention 端到端
    直接回归坐标（同时利用序列、MSA、pair 表示的全部信息，并对旋转平移
    等价性做了显式建模）。这里的简化版只用"耦合分数 → 目标距离"这一条
    线索，是对"pair 表示 → 3D 结构"这一步骤最朴素的几何类比。

    做法：
      1. 把耦合分数映射为目标距离：分数越高 → 目标距离越短（约 6~7Å 代表接触）；
         分数低 → 目标距距离拉大（代表"大概率不接触"，用较弱的权重）
      2. 序列上的近邻残基使用固定的共价键长目标（约 3.8Å），权重最高
      3. 随机初始化 3D 坐标，用梯度下降最小化
             L = Σ_{i<j} w_{ij} (||x_i - x_j|| - d_{ij}^{target})²

    为了缓解梯度下降陷入局部最优的问题，这里做 `n_restarts` 次随机初始化，
    每次用带动量的梯度下降优化，最终保留损失最低的一次结果——这与真实
    结构预测流程中"多次采样 + 按置信度/能量挑选最佳结构"的思路是一致的。

    参数:
        pred_coupling: 预测耦合分数矩阵 (N, N)
        n_residues: N
        n_iters: 梯度下降迭代步数
        lr: 学习率
        n_restarts: 随机重启次数，取损失最低的一次
        seed: 随机种子

    返回:
        coords: 重建出的 3D 坐标 (N, 3)
    """
    n = n_residues

    # ---- 构造目标距离矩阵与权重矩阵 ----
    target_dist = np.full((n, n), 20.0)                      # 默认：远距离（弱约束）
    weight = np.full((n, n), 0.02)                            # 默认：弱权重

    # 归一化耦合分数到 [0, 1]，用于插值目标距离
    c = pred_coupling.copy()
    if c.max() > c.min():
        c_norm = (c - c.min()) / (c.max() - c.min() + 1e-8)
    else:
        c_norm = np.zeros_like(c)

    for i in range(n):
        for j in range(i + 1, n):
            if abs(i - j) == 1:
                target_dist[i, j] = target_dist[j, i] = 3.8   # 共价键长约束
                weight[i, j] = weight[j, i] = 5.0
            elif abs(i - j) == 2:
                target_dist[i, j] = target_dist[j, i] = 5.5   # 骨架二级结构典型间距，弱约束
                weight[i, j] = weight[j, i] = 0.6
            else:
                # 分数越高 → 认为更可能接触 → 目标距离越接近 6~7Å
                d = 20.0 - 14.0 * c_norm[i, j]
                target_dist[i, j] = target_dist[j, i] = d
                weight[i, j] = weight[j, i] = 0.05 + 2.0 * c_norm[i, j] ** 2

    iu = np.triu_indices(n, k=1)
    best_coords, best_loss = None, np.inf

    for restart in range(n_restarts):
        rng = np.random.RandomState(seed + 100 * restart)
        coords = rng.normal(scale=5.0, size=(n, 3))
        velocity = np.zeros_like(coords)
        momentum = 0.9

        for it in range(n_iters):
            diff = coords[:, None, :] - coords[None, :, :]    # (N, N, 3)
            dist = np.linalg.norm(diff, axis=-1) + 1e-8        # (N, N)
            # 损失: sum w_ij (dist_ij - target_ij)^2 (仅上三角，避免重复计算)
            err = dist - target_dist                           # (N, N)
            # dL/dx_i = sum_j 2 * w_ij * err_ij * (x_i - x_j) / dist_ij
            coef = 2.0 * weight * err / dist                   # (N, N)
            np.fill_diagonal(coef, 0.0)
            grad = (coef[:, :, None] * diff).sum(axis=1)        # (N, 3)
            velocity = momentum * velocity - lr * grad          # 带动量的梯度下降
            coords = coords + velocity

        final_loss = (weight[iu] * (err[iu]) ** 2).sum()
        print(f"    [结构重建] 第 {restart+1}/{n_restarts} 次重启, "
              f"最终加权距离残差损失 = {final_loss:.2f}")
        if final_loss < best_loss:
            best_loss = final_loss
            best_coords = coords.copy()

    best_coords -= best_coords.mean(axis=0)
    return best_coords


def kabsch_align(mobile: np.ndarray, target: np.ndarray) -> tuple:
    """
    Kabsch 算法：找到最优旋转（和可选的镜像修正），把 mobile 对齐到 target，
    使对齐后两组点的均方根偏差 (RMSD) 最小。这是比较两个 3D 结构时的标准做法
    （结构预测本身不关心整体的平移/旋转，只关心相对几何）。

    参数:
        mobile: 待对齐坐标 (N, 3)
        target: 目标坐标 (N, 3)

    返回:
        aligned: 对齐后的 mobile 坐标 (N, 3)
        rmsd: 对齐后的均方根偏差
    """
    mobile_c = mobile - mobile.mean(axis=0)
    target_c = target - target.mean(axis=0)

    h = mobile_c.T @ target_c                                  # (3, 3) 协方差矩阵
    u, s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))                     # 修正镜像（防止反射）
    correction = np.diag([1, 1, d])
    r = vt.T @ correction @ u.T

    aligned = (r @ mobile_c.T).T
    rmsd = np.sqrt(((aligned - target_c) ** 2).sum(axis=1).mean())
    return aligned + target.mean(axis=0), rmsd


# ============================================================================
# 第五部分：可视化
# ============================================================================

def plot_pipeline_schematic():
    """
    绘制 AlphaFold 流水线示意图 (as06-01-alphafold-pipeline.png)。
    这是纯教学示意图，展示 序列 → MSA搜索 → Evoformer(MSA轨道+Pair轨道
    通过 attention 与 outer-product-mean 相互通信) → Structure Module(IPA)
    → 3D结构+置信度(pLDDT) 的整体流程，以及贯穿全程的 Recycling 循环。
    """
    fig, ax = plt.subplots(1, 1, figsize=(13, 7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.5)
    ax.axis('off')

    def box(x, y, w, h, text, color, fontsize=10):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                            linewidth=1.5, edgecolor='#333333', facecolor=color)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', wrap=True)

    def arrow(x1, y1, x2, y2, style='-|>', color='#333333', lw=1.8, connectionstyle="arc3,rad=0.0"):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                            mutation_scale=15, linewidth=lw, color=color,
                            connectionstyle=connectionstyle)
        ax.add_patch(a)

    # 输入序列
    box(0.3, 5.6, 1.9, 1.2, "输入\n氨基酸序列", '#E8E8E8')
    # MSA 搜索 + 模板搜索
    box(2.7, 5.6, 2.1, 1.2, "MSA 搜索 +\n模板检索\n(数据库比对)", '#CDE7F0')
    arrow(2.2, 6.2, 2.7, 6.2)

    # Evoformer 大框
    evo_x, evo_y, evo_w, evo_h = 5.3, 3.7, 4.3, 3.2
    evo_box = FancyBboxPatch((evo_x, evo_y), evo_w, evo_h, boxstyle="round,pad=0.08",
                            linewidth=2, edgecolor='#2E5266', facecolor='#F5FAFD')
    ax.add_patch(evo_box)
    ax.text(evo_x + evo_w / 2, evo_y + evo_h - 0.28, "Evoformer（迭代 N 次）",
            ha='center', fontsize=11, fontweight='bold', color='#2E5266')

    box(evo_x + 0.25, evo_y + 1.85, 1.85, 1.0, "MSA 表示\n(行注意力+\n列注意力)", '#FDE8D7', fontsize=8.5)
    box(evo_x + 2.25, evo_y + 1.85, 1.85, 1.0, "Pair 表示\n(三角注意力/\n乘法更新)", '#D9EAD3', fontsize=8.5)
    arrow(evo_x + 2.1, evo_y + 2.5, evo_x + 2.25, evo_y + 2.5)
    ax.text(evo_x + 2.18, evo_y + 2.65, "Outer Product Mean →", ha='center', fontsize=6, color='#555')
    arrow(evo_x + 2.25, evo_y + 2.1, evo_x + 2.1, evo_y + 2.1)
    ax.text(evo_x + 2.18, evo_y + 1.6, "← 三角乘法更新 MSA", ha='center', fontsize=6, color='#555')

    box(evo_x + 1.15, evo_y + 0.25, 2.0, 0.9, "置信度头\npLDDT / PAE 预测", '#EFE0F5', fontsize=8)

    arrow(2.7 + 2.1, 6.2, evo_x, evo_y + 2.05)

    # Structure Module
    box(10.15, 3.7, 2.2, 1.4, "Structure Module\n(Invariant Point\nAttention)", '#F9D5D3', fontsize=9)
    arrow(evo_x + evo_w, evo_y + 2.05, 10.15, 4.4)

    # Recycling 循环（结构模块输出反馈回 Evoformer 输入）
    arrow(10.15, 3.6, evo_x + evo_w - 0.3, evo_y - 0.15,
          connectionstyle="arc3,rad=0.35", color='#B23A48', lw=1.6)
    ax.text(8.3, 2.9, "Recycling：把结构预测反馈回 Evoformer，迭代 refine", fontsize=8, color='#B23A48')

    # 输出结构
    box(10.35, 0.9, 1.9, 1.4, "3D 结构\n+ 置信度着色\n(pLDDT)", '#FFF3B0', fontsize=9)
    arrow(11.25, 3.7, 11.25, 2.3)

    # 顶部标注：CASP 评测
    ax.text(6.5, 7.1, "AlphaFold 流水线（教学示意图，非原始架构图）",
            ha='center', fontsize=13, fontweight='bold')
    ax.text(6.5, 0.35,
            "AF1(2018): 接触图+梯度下降  →  AF2(2020): Evoformer+IPA, CASP14 中位 GDT>90  →  AF3(2024): 扩散模型, 扩展到复合物/配体/核酸",
            ha='center', fontsize=8.5, color='#555555')

    _save(fig, 'as06-01-alphafold-pipeline.png')


def plot_msa_and_contacts(msa, true_contact, pred_coupling, coords):
    """绘制 MSA 概览、真实接触图 vs 预测耦合矩阵的对比。"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    im0 = ax.imshow(msa[:80], aspect='auto', cmap='tab20', interpolation='nearest')
    ax.set_title('合成 MSA（前 80 条序列）', fontweight='bold')
    ax.set_xlabel('残基位置')
    ax.set_ylabel('序列编号')

    ax = axes[1]
    im1 = ax.imshow(true_contact, cmap='Greys', vmin=0, vmax=1)
    ax.set_title('真实接触图 (Ground Truth)', fontweight='bold')
    ax.set_xlabel('残基 j')
    ax.set_ylabel('残基 i')

    ax = axes[2]
    im2 = ax.imshow(pred_coupling, cmap='viridis')
    ax.set_title('预测耦合矩阵\n(Outer Product Mean)', fontweight='bold')
    ax.set_xlabel('残基 j')
    ax.set_ylabel('残基 i')
    plt.colorbar(im2, ax=ax, fraction=0.046)

    plt.tight_layout()
    _save(fig, 'contact_map_prediction.png')


def plot_precision_bars(precision_results: dict):
    """绘制 CASP 风格的 precision@L / L2 / L5 柱状图。"""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
    names = list(precision_results.keys())
    values = [precision_results[n][0] for n in names]
    ks = [precision_results[n][1] for n in names]

    bars = ax.bar(names, values, color=['#4C72B0', '#DD8452', '#55A868'])
    for bar, v, k in zip(bars, values, ks):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                f"{v:.2f}\n(top {k})", ha='center', fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Precision（预测接触中真接触的比例）')
    ax.set_title('接触预测精度（CASP 式指标）', fontweight='bold')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4)
    plt.tight_layout()
    _save(fig, 'precision_at_l.png')


def plot_structure_comparison(true_coords, pred_coords_aligned, rmsd):
    """对比重建结构与真实结构的 3D 折线图。"""
    fig = plt.figure(figsize=(11, 5.5))

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot(true_coords[:, 0], true_coords[:, 1], true_coords[:, 2],
              '-o', color='#2E5266', markersize=3, linewidth=1.5, label='真实骨架')
    ax1.set_title('真实骨架 (Ground Truth)', fontweight='bold')
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot(true_coords[:, 0], true_coords[:, 1], true_coords[:, 2],
              '-o', color='#2E5266', markersize=3, linewidth=1.2, alpha=0.5, label='真实骨架')
    ax2.plot(pred_coords_aligned[:, 0], pred_coords_aligned[:, 1], pred_coords_aligned[:, 2],
              '-o', color='#B23A48', markersize=3, linewidth=1.2, label='重建结构 (对齐后)')
    ax2.set_title(f'重建结构 vs 真实结构\n(RMSD = {rmsd:.2f})', fontweight='bold')
    ax2.legend()

    plt.tight_layout()
    _save(fig, 'structure_reconstruction.png')


# ============================================================================
# 第六部分：主程序
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("    as06 蛋白质结构预测玩具演示：共进化信号 → 3D 结构")
    print("    （教学玩具，非 AlphaFold 真实推理）")
    print("=" * 70)

    set_seed(42)
    N_RESIDUES = 40

    # ---- 1. 生成"真实"骨架与接触图 ----
    print("\n[步骤 1] 生成合成蛋白骨架与真实接触图...")
    true_coords = generate_synthetic_backbone(N_RESIDUES, seed=42)
    true_contact, true_dist = compute_contact_map(true_coords, threshold=8.0)
    n_contacts = int(true_contact.sum() / 2)
    print(f"  残基数: {N_RESIDUES}, 真实长程接触对数: {n_contacts}")

    # ---- 2. 合成 MSA（共进化信号） ----
    print("\n[步骤 2] 合成多序列比对 (MSA)，注入共进化信号...")
    msa = generate_synthetic_msa(N_RESIDUES, true_contact, n_sequences=300,
                                  mutation_rate=0.35, coupling_strength=0.85, seed=42)
    print(f"  MSA 规模: {msa.shape[0]} 条序列 x {msa.shape[1]} 个位置")

    # ---- 3. Outer Product Mean 提取耦合矩阵（预测接触图的简化版） ----
    print("\n[步骤 3] 用 Outer Product Mean 从 MSA 提取残基对耦合强度...")
    pred_coupling = outer_product_mean_coupling(msa, N_RESIDUES)

    # ---- 4. 评估接触预测精度 ----
    print("\n[步骤 4] 评估接触预测精度（CASP 式 precision@L 指标）...")
    precision_results = evaluate_contact_precision(pred_coupling, true_contact)
    for name, (p, k) in precision_results.items():
        print(f"  precision@{name:4s} (top {k:3d} 对): {p:.3f}")

    # ---- 5. 简化结构模块：从耦合矩阵重建 3D 坐标 ----
    print("\n[步骤 5] 简化版结构模块：距离几何梯度下降重建 3D 坐标...")
    pred_coords = reconstruct_structure(pred_coupling, N_RESIDUES,
                                         n_iters=1500, lr=0.02, n_restarts=4, seed=42)
    pred_aligned, rmsd = kabsch_align(pred_coords, true_coords)
    print(f"  Kabsch 对齐后 RMSD = {rmsd:.2f} （坐标单位与真实骨架一致）")

    # ---- 6. 可视化 ----
    print("\n[步骤 6] 生成可视化图表...")
    plot_pipeline_schematic()
    plot_msa_and_contacts(msa, true_contact, pred_coupling, true_coords)
    plot_precision_bars(precision_results)
    plot_structure_comparison(true_coords, pred_aligned, rmsd)

    # ---- 总结 ----
    print("\n" + "=" * 70)
    print("【总结】")
    print("=" * 70)
    print(f"  1. 共进化信号确实携带残基接触信息：")
    print(f"     precision@L/5 = {precision_results['L/5'][0]:.2f}"
          f"（远高于随机猜测的基线水平）")
    print(f"  2. 仅凭粗糙的接触约束也能大致重建整体折叠形态：")
    print(f"     RMSD = {rmsd:.2f}（越接近真实骨架，数值越小）")
    print(f"  3. 但请注意——这只是一个几百行代码的教学玩具：")
    print(f"     - 真实 AlphaFold2 用深度 Evoformer（48 个 block）+ IPA，")
    print(f"       在数十万条真实晶体结构上训练")
    print(f"     - 真实蛋白折叠涉及氢键、疏水核心、二级结构传播等复杂物理")
    print(f"     - pLDDT/PAE 等置信度指标本身也需要模型校准，不是绝对正确性保证")
    print(f"     - 蛋白质结构预测≠蛋白质折叠动力学，也≠功能预测，")
    print(f"       AlphaFold 解决的是'静态结构预测'这一个子问题，")
    print(f"       结构生物学、功能机制研究仍然需要大量实验验证")
    print(f"\n  所有图片已保存至 images/ 目录")
    print("=" * 70)
    print("\n  运行完成！\n")


if __name__ == "__main__":
    main()
