# -*- coding: utf-8 -*-
"""
ml11 隐马尔可夫模型 (HMM) — 演示代码
=====================================
功能：从零实现 HMM 的前向算法（评估）、Viterbi 算法（解码），
      展示 POS 词性标注的格子图（trellis）和状态序列。

每个函数都有中文 docstring，每行逻辑代码都有中文注释。
运行方式：在 ml11_hmm/ 目录下执行 python code/demo.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
from matplotlib.patches import FancyArrowPatch

_HERE = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_HERE, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)


# ============================================================================
# 第一部分：HMM 类定义
# ============================================================================

class HMM:
    """
    隐马尔可夫模型（离散观测）。

    参数 λ = (A, B, pi):
        A[i, j] = P(Z_{t+1}=j | Z_t=i)  — 状态转移矩阵 (N, N)
        B[i, o] = P(X_t=o | Z_t=i)       — 发射概率矩阵 (N, M)
        pi[i]   = P(Z_1=i)                — 初始状态分布 (N,)

    属性:
        N: 隐藏状态数
        M: 观测符号数（词汇量）
    """

    def __init__(self, A, B, pi):
        self.A = np.array(A)    # (N, N)
        self.B = np.array(B)    # (N, M)
        self.pi = np.array(pi)  # (N,)
        self.N = len(pi)
        self.M = B.shape[1]

    def forward(self, observations):
        """
        前向算法：计算 P(X | λ)。

        递推公式:
            α_1(i) = π_i · B[i, x_1]
            α_t(j) = [Σ_i α_{t-1}(i) · A[i, j]] · B[j, x_t]
            P(X | λ) = Σ_i α_T(i)

        使用缩放（scaling）防止下溢：
            c_t = Σ_i α_t(i)
            α̃_t(i) = α_t(i) / c_t
            log P = Σ_t log c_t

        参数:
            observations: 观测序列（索引列表），长度 T

        返回:
            log_prob: log P(X | λ)
            alphas: 缩放后的 α 矩阵 (T, N)，α[t, i] = α̃_{t+1}(i)
            c: 每步的缩放因子 (T,)
        """
        T = len(observations)
        alphas = np.zeros((T, self.N))  # (T, N)
        c = np.zeros(T)                  # 缩放因子

        # 初始化 t=1
        o1 = observations[0]
        for i in range(self.N):
            alphas[0, i] = self.pi[i] * self.B[i, o1]
        c[0] = alphas[0].sum()
        if c[0] > 0:
            alphas[0] /= c[0]

        # 递推 t = 2, ..., T
        for t in range(1, T):
            ot = observations[t]
            for j in range(self.N):
                # α_t(j) = [Σ_i α_{t-1}(i) * A[i, j]] * B[j, o_t]
                alphas[t, j] = np.dot(alphas[t-1, :], self.A[:, j]) * self.B[j, ot]
            c[t] = alphas[t].sum()
            if c[t] > 0:
                alphas[t] /= c[t]

        log_prob = np.sum(np.log(c + 1e-300))  # 防止 log(0)
        return log_prob, alphas, c

    def viterbi(self, observations):
        """
        Viterbi 算法：找到最可能的状态序列。

        递推:
            δ_1(i) = π_i · B[i, x_1]
            δ_t(j) = max_i [δ_{t-1}(i) · A[i, j]] · B[j, x_t]
            ψ_t(j) = argmax_i [δ_{t-1}(i) · A[i, j]]

        使用对数防止下溢：
            log δ_t(j) = max_i [log δ_{t-1}(i) + log A[i, j]] + log B[j, x_t]

        参数:
            observations: 观测序列，长度 T

        返回:
            log_prob: 最优路径的对数概率
            best_path: 最优状态序列（索引列表），长度 T
            deltas: log δ 矩阵 (T, N)
        """
        T = len(observations)
        deltas = np.zeros((T, self.N))  # log δ_t(i)
        psi = np.zeros((T, self.N), dtype=int)  # 回溯指针 ψ_t(i)

        # 初始化 t=1
        o1 = observations[0]
        for i in range(self.N):
            deltas[0, i] = np.log(self.pi[i] + 1e-300) + np.log(self.B[i, o1] + 1e-300)

        # 递推 t = 2, ..., T
        for t in range(1, T):
            ot = observations[t]
            for j in range(self.N):
                # log δ_t(j) = max_i [log δ_{t-1}(i) + log A[i, j]] + log B[j, ot]
                candidates = deltas[t-1, :] + np.log(self.A[:, j] + 1e-300)
                best_i = np.argmax(candidates)
                deltas[t, j] = candidates[best_i] + np.log(self.B[j, ot] + 1e-300)
                psi[t, j] = best_i

        # 终止：找到最优路径的终点
        best_final_state = np.argmax(deltas[T-1, :])
        log_prob = deltas[T-1, best_final_state]

        # 回溯
        best_path = np.zeros(T, dtype=int)
        best_path[T-1] = best_final_state
        for t in range(T-2, -1, -1):
            best_path[t] = psi[t+1, best_path[t+1]]

        return log_prob, best_path, deltas


# ============================================================================
# 第二部分：POS Tagging 示例
# ============================================================================

def pos_tagging_demo():
    """
    用简化的 POS 标注示例演示 HMM 的 Forward 和 Viterbi 算法。

    状态 = 词性标签: [DET(0=限定词), NOUN(1=名词), VERB(2=动词)]
    观测 = 词汇:     [0=the, 1=cat, 2=sat, 3=mat]
    """
    # 定义 HMM 参数
    N = 3  # 状态数: DET, NOUN, VERB
    M = 4  # 观测数: the(0), cat(1), sat(2), mat(3)

    # 转移矩阵 A[i, j] = P(Z_{t+1}=j | Z_t=i)
    #        DET   NOUN  VERB
    A = np.array([
        [0.1, 0.8, 0.1],  # DET → 通常后接 NOUN
        [0.1, 0.3, 0.6],  # NOUN → 通常后接 VERB
        [0.4, 0.4, 0.2],  # VERB → 可接 DET 或 NOUN
    ])

    # 发射矩阵 B[i, o] = P(X_t=o | Z_t=i)
    #        the  cat  sat  mat
    B = np.array([
        [0.8, 0.05, 0.1, 0.05],  # DET  → 通常产生 "the"
        [0.05, 0.6, 0.1, 0.25],  # NOUN → 通常产生 "cat"/"mat"
        [0.05, 0.1, 0.8, 0.05],  # VERB → 通常产生 "sat"
    ])

    # 初始分布 π[i] = P(Z_1=i)
    pi = np.array([0.5, 0.3, 0.2])  # 句子大概率以 DET 开头

    hmm = HMM(A, B, pi)

    # 观测: "the cat sat on the mat" → 只取 the(0), cat(1), sat(2), the(0), mat(3)
    observations = [0, 1, 2, 0, 3]  # the, cat, sat, the, mat

    # 前向算法
    log_prob, alphas, c = hmm.forward(observations)
    print(f'  观测序列: the cat sat the mat')
    print(f'  log P(X | λ) = {log_prob:.4f}')

    # Viterbi 解码
    log_prob_v, best_path, deltas = hmm.viterbi(observations)
    state_names = ['DET', 'NOUN', 'VERB']
    obs_names = ['the', 'cat', 'sat', 'the', 'mat']
    print(f'  Viterbi 最优路径: {" ".join(state_names[s] for s in best_path)}')
    print(f'  log P*(X, Z) = {log_prob_v:.4f}')

    # 可视化格子图
    plot_pos_trellis(alphas, deltas, best_path, state_names, obs_names,
                     observations)

    return alphas, best_path


def plot_pos_trellis(alphas, deltas, best_path, state_names, obs_names, observations):
    """
    绘制 POS tagging 的格子图（trellis），展示 Forward 和 Viterbi 权重。
    """
    T = len(observations)
    N = len(state_names)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # 左图: Forward α 热度图
    ax1 = axes[0]
    im1 = ax1.imshow(alphas.T, aspect='auto', cmap='YlOrRd', origin='upper')
    ax1.set_xticks(range(T))
    ax1.set_xticklabels([obs_names[o] for o in observations], fontsize=10)
    ax1.set_yticks(range(N))
    ax1.set_yticklabels(state_names, fontsize=10)
    ax1.set_xlabel('Time (Observation)')
    ax1.set_ylabel('Hidden State')
    ax1.set_title('Forward α (scaled)')

    # 在格子上标注数值
    for i in range(N):
        for t in range(T):
            ax1.text(t, i, f'{alphas[t, i]:.3f}', ha='center', va='center',
                     fontsize=8, color='black' if alphas[t, i] < 0.6 else 'white')

    plt.colorbar(im1, ax=ax1, shrink=0.8)

    # 右图: Viterbi 路径
    ax2 = axes[1]
    im2 = ax2.imshow(np.exp(deltas).T, aspect='auto', cmap='YlOrRd', origin='upper')
    ax2.set_xticks(range(T))
    ax2.set_xticklabels([obs_names[o] for o in observations], fontsize=10)
    ax2.set_yticks(range(N))
    ax2.set_yticklabels(state_names, fontsize=10)
    ax2.set_xlabel('Time (Observation)')
    ax2.set_ylabel('Hidden State')
    ax2.set_title('Viterbi δ (exp) with Best Path')

    # 画最优路径
    for t in range(T - 1):
        ax2.annotate('', xy=(t + 1, best_path[t + 1]), xytext=(t, best_path[t]),
                     arrowprops=dict(arrowstyle='->', color='blue', lw=2.5,
                                     connectionstyle='arc3,rad=0'))
    # 标注路径节点
    for t in range(T):
        ax2.plot(t, best_path[t], 'o', color='blue', markersize=10, markeredgecolor='white')

    plt.colorbar(im2, ax=ax2, shrink=0.8)
    plt.tight_layout()
    path = os.path.join(_IMAGES_DIR, 'ml11-04-pos-trellis.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[保存] {path}')


# ============================================================================
# 第三部分：马尔可夫链可视化
# ============================================================================

def plot_markov_chain_example():
    """绘制一个简单的 3 状态马尔可夫链及其平稳分布"""
    # 转移矩阵
    A = np.array([
        [0.5, 0.3, 0.2],
        [0.2, 0.6, 0.2],
        [0.1, 0.3, 0.6],
    ])

    # 计算平稳分布：解 πA = π，即 π(A - I) = 0，且 Σπ = 1
    # 等价于求 A^T 的特征值 1 对应的特征向量
    eigenvalues, eigenvectors = np.linalg.eig(A.T)
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    pi = np.real(eigenvectors[:, idx])
    pi = pi / pi.sum()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：转移图
    ax1 = axes[0]
    # 三个状态节点的坐标（三角形排列）
    positions = {0: (0, 1), 1: (1, 0), 2: (2, 1)}
    state_labels = ['State 1\n(Sunny)', 'State 2\n(Cloudy)', 'State 3\n(Rainy)']
    colors = ['#FFD700', '#B0C4DE', '#6B8E23']

    # 画节点
    for s in range(3):
        x, y = positions[s]
        circle = plt.Circle((x, y), 0.15, color=colors[s], ec='black', linewidth=1.5,
                            zorder=2)
        ax1.add_patch(circle)
        ax1.text(x, y, str(s+1), ha='center', va='center', fontsize=12, fontweight='bold')
        ax1.text(x, y - 0.3, state_labels[s], ha='center', va='top', fontsize=9)

    # 画转移箭头
    for i in range(3):
        for j in range(3):
            if A[i, j] > 0.01:
                xi, yi = positions[i]
                xj, yj = positions[j]

                if i == j:
                    # 自环
                    ax1.annotate('', xy=(xi + 0.08, yi + 0.12), xytext=(xi - 0.08, yi + 0.12),
                                 arrowprops=dict(arrowstyle='->', color='gray',
                                                 connectionstyle='arc3,rad=0.5', lw=1.5))
                    ax1.text(xi, yi + 0.28, f'{A[i,j]:.1f}', ha='center', fontsize=8, color='gray')
                else:
                    ax1.annotate('', xy=(xj, yj), xytext=(xi, yi),
                                 arrowprops=dict(arrowstyle='->', color='gray',
                                                 connectionstyle='arc3,rad=0.2',
                                                 lw=A[i,j]*4, alpha=0.7))
                    mx, my = (xi + xj) / 2, (yi + yj) / 2
                    offset = 0.08 if abs(xi - xj) < 0.1 else 0
                    ax1.text(mx, my + 0.1, f'{A[i,j]:.1f}', ha='center', fontsize=9, color='navy')

    ax1.set_xlim(-0.5, 2.5)
    ax1.set_ylim(-0.3, 1.5)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('Markov Chain Transition Diagram\nA = Transition Matrix')

    # 右图：平稳分布
    ax2 = axes[1]
    bars = ax2.bar(range(3), pi, color=colors, edgecolor='black', linewidth=1)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(['Sunny', 'Cloudy', 'Rainy'])
    ax2.set_ylabel('Stationary Probability')
    ax2.set_title('Stationary Distribution π')
    for bar, val in zip(bars, pi):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', fontsize=10)
    ax2.set_ylim(0, max(pi) * 1.2)

    plt.tight_layout()
    path = os.path.join(_IMAGES_DIR, 'ml11-05-markov-chain.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[保存] {path}')


# ============================================================================
# 第四部分：HMM 在不同转移矩阵下的行为
# ============================================================================

def compare_hmm_behaviors():
    """对比不同转移矩阵下 HMM 的状态序列特性"""
    observations = [0, 1, 2, 0, 3, 1, 2]  # 7 个观测

    # 场景 1：平滑转移（相邻状态间高转移概率）
    A_smooth = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.7, 0.2],
        [0.1, 0.2, 0.7],
    ])

    # 场景 2：尖峰转移（只向特定状态转移）
    A_peaky = np.array([
        [0.1, 0.85, 0.05],
        [0.05, 0.1, 0.85],
        [0.85, 0.05, 0.1],
    ])

    B = np.array([
        [0.8, 0.1, 0.1, 0.0],
        [0.1, 0.6, 0.2, 0.1],
        [0.05, 0.15, 0.7, 0.1],
    ])
    pi = np.array([0.33, 0.34, 0.33])

    hmm1 = HMM(A_smooth, B, pi)
    hmm2 = HMM(A_peaky, B, pi)

    _, path1, _ = hmm1.viterbi(observations)
    _, path2, _ = hmm2.viterbi(observations)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    state_names = ['S1', 'S2', 'S3']
    T = len(observations)

    for idx, (ax, path, A, title) in enumerate([
        (axes[0], path1, A_smooth, 'Smooth Transitions\n(high self-transition)'),
        (axes[1], path2, A_peaky, 'Peaky Transitions\n(forced cycle S1→S2→S3→S1)'),
    ]):
        # 画转移矩阵热力图
        im = ax.imshow(A, cmap='YlOrRd', vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{A[i,j]:.2f}', ha='center', va='center', fontsize=9,
                        color='black' if A[i, j] < 0.6 else 'white')
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(state_names)
        ax.set_yticklabels(state_names)
        ax.set_xlabel('To State')
        ax.set_ylabel('From State')
        ax.set_title(title)

        # 标注 Viterbi 路径
        path_str = ' → '.join([state_names[p] for p in path])
        ax.text(1.5, -0.4, f'Viterbi: {path_str}', ha='center', fontsize=11,
                transform=ax.transData, color='blue',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    path = os.path.join(_IMAGES_DIR, 'ml11-06-hmm-behaviors.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[保存] {path}')


# ============================================================================
# 第五部分：主程序
# ============================================================================

def main():
    print('=' * 60)
    print('ml11 隐马尔可夫模型 (HMM) — 演示代码')
    print('=' * 60)

    # 1. POS 标注 Demo
    print('\n[1/3] POS Tagging Demo (Forward + Viterbi)...')
    pos_tagging_demo()

    # 2. 马尔可夫链可视化
    print('[2/3] 马尔可夫链与平稳分布...')
    plot_markov_chain_example()

    # 3. 不同转移矩阵下的行为对比
    print('[3/3] 转移矩阵对状态序列的影响...')
    compare_hmm_behaviors()

    print('\n完成！所有图表已保存到 images/ 目录。')


if __name__ == '__main__':
    main()
