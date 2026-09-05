# -*- coding: utf-8 -*-
"""
=== AlphaGo 最小演示：井字棋上的 MCTS（PUCT）===
不训练深度网络。用均匀先验 + 随机滚出，对比「纯贪心 / 随机」与 MCTS。
画出根节点访问次数，对应正文里「按 N 落子」。
运行: python demo.py
"""
import os
import math
import random
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)
np.random.seed(42)
random.seed(42)

N_SIM = 200
N_GAMES = 80
C_PUCT = 1.5
EMPTY, X, O = 0, 1, -1


def winner(b):
    lines = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6),
    ]
    for i, j, k in lines:
        s = b[i] + b[j] + b[k]
        if s == 3:
            return X
        if s == -3:
            return O
    if all(v != EMPTY for v in b):
        return 0
    return None


def legal(b):
    return [i for i, v in enumerate(b) if v == EMPTY]


def copy_play(b, a, p):
    nb = list(b)
    nb[a] = p
    return nb


def rollout(b, to_move, root_player):
    """从 to_move 开始随机走完，返回根玩家视角的 z ∈ {+1, 0, -1}。"""
    player = to_move
    board = list(b)
    while True:
        w = winner(board)
        if w is not None:
            if w == 0:
                return 0.0
            return 1.0 if w == root_player else -1.0
        board = copy_play(board, random.choice(legal(board)), player)
        player = -player


class Node:
    def __init__(self, board, player, parent=None, action=None, prior=1.0):
        self.board = board
        self.player = player
        self.parent = parent
        self.action = action
        self.prior = prior
        self.children = {}
        self.n = 0
        self.w = 0.0
        self.expanded = False

    def q(self):
        return 0.0 if self.n == 0 else self.w / self.n

    def puct(self, c):
        parent_n = self.parent.n if self.parent else 1
        u = c * self.prior * math.sqrt(parent_n) / (1 + self.n)
        return self.q() + u


def expand(node):
    acts = legal(node.board)
    if not acts:
        return
    prior = 1.0 / len(acts)
    for a in acts:
        child_board = copy_play(node.board, a, node.player)
        node.children[a] = Node(child_board, -node.player, node, a, prior)
    node.expanded = True


def select(node, c):
    while node.expanded and node.children:
        node = max(node.children.values(), key=lambda ch: ch.puct(c))
    return node


def backup(node, z_root, root_player):
    """边上的 Q 存『走进这个节点的棋手』的价值，父节点才能 argmax Q。"""
    while node is not None:
        node.n += 1
        if node.parent is None:
            node.w += z_root
        else:
            mover = node.parent.player
            node.w += z_root if mover == root_player else -z_root
        node = node.parent


def mcts_move(board, player, n_sim=N_SIM, c=C_PUCT):
    root = Node(board, player)
    expand(root)
    # 根上混一点 Dirichlet，避免第一次模拟把 N 锁死在同一手（Zero 也这么干）
    acts = list(root.children.keys())
    if acts:
        noise = np.random.dirichlet([0.3] * len(acts))
        for a, n in zip(acts, noise):
            ch = root.children[a]
            ch.prior = 0.75 * ch.prior + 0.25 * float(n)
    for _ in range(n_sim):
        leaf = select(root, c)
        w = winner(leaf.board)
        if w is None:
            if not leaf.expanded:
                expand(leaf)
            z = rollout(leaf.board, leaf.player, root.player)
            backup(leaf, z, root.player)
        else:
            z = 0.0 if w == 0 else (1.0 if w == root.player else -1.0)
            backup(leaf, z, root.player)
    visits = {a: ch.n for a, ch in root.children.items()}
    best = max(visits, key=visits.get)
    return best, visits, root


def random_move(board, player):
    return random.choice(legal(board))


def greedy_center(board, player):
    acts = legal(board)
    for pref in (4, 0, 2, 6, 8, 1, 3, 5, 7):
        if pref in acts:
            return pref
    return acts[0]


def play_game(x_policy, o_policy):
    b = [EMPTY] * 9
    p = X
    while True:
        w = winner(b)
        if w is not None:
            return w
        a = x_policy(b, p) if p == X else o_policy(b, p)
        b = copy_play(b, a, p)
        p = -p


def mcts_policy(b, p):
    a, _, _ = mcts_move(b, p)
    return a


def _save(fig, name):
    path = os.path.join(_IMAGES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'图已保存: {path}')


def _box(ax, x, y, w, h, text, color, fs=10):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.04',
        linewidth=1.4, edgecolor='#333333', facecolor=color,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fs, fontweight='bold', color='#222222')


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=14,
        linewidth=1.5, color='#333333',
    ))


def draw_three_networks():
    fig, ax = plt.subplots(figsize=(11.6, 5.4))
    ax.set_xlim(0, 11.6)
    ax.set_ylim(0, 5.6)
    ax.axis('off')
    ax.text(5.8, 5.2, 'AlphaGo 三件套：棋感交给网络，算清交给搜索',
            ha='center', fontsize=14, fontweight='bold')
    _box(ax, 0.3, 2.7, 2.6, 1.8, 'SL 策略 $p_\\sigma$\n模仿人类棋谱\n给 MCTS 当先验 $P$', '#CDE7F0', 10)
    _box(ax, 3.2, 2.7, 2.6, 1.8, 'RL 策略 $p_\\rho$\n自我对弈 + 策略梯度\n学会赢而不只是模仿', '#D9EAD3', 10)
    _box(ax, 6.1, 2.7, 2.6, 1.8, '价值网络 $v_\\theta$\n局面胜率\n叶子上少做完整滚出', '#FDE8D7', 10)
    _box(ax, 9.0, 2.7, 2.3, 1.8, 'MCTS\nPUCT 选边\n按访问次数落子', '#E8D5F2', 10)
    _arrow(ax, 2.95, 3.6, 3.15, 3.6)
    _arrow(ax, 5.85, 3.6, 6.05, 3.6)
    _arrow(ax, 8.75, 3.6, 8.95, 3.6)
    ax.text(5.8, 1.6, '2016 年那一版是三条网；AlphaGo Zero 收成一张双头网（$\\pi$ 与 $v$）。',
            ha='center', fontsize=10, color='#444444')
    ax.text(5.8, 0.7, '后面 PPO / GRPO / RLHF 只优化「棋感」这一层；这一章把「棋感 + 搜索」讲完。',
            ha='center', fontsize=10, color='#444444')
    _save(fig, 'ag-01-three-networks.png')


def draw_self_play():
    fig, ax = plt.subplots(figsize=(11.4, 4.8))
    ax.set_xlim(0, 11.4)
    ax.set_ylim(0, 5.0)
    ax.axis('off')
    ax.text(5.7, 4.6, '自我对弈：旧策略当对手，终局胜负回标每一手',
            ha='center', fontsize=13, fontweight='bold')
    _box(ax, 0.35, 2.3, 2.5, 1.5, '当前 $p_\\rho$\n执黑', '#CDE7F0', 10)
    _box(ax, 3.2, 2.3, 2.5, 1.5, '旧拷贝 $p_{\\rho}^{-}$\n执白', '#E8E8E8', 10)
    _box(ax, 6.05, 2.3, 2.3, 1.5, '终局 $z$\n$\\in\\{+1,-1\\}$', '#FDE8D7', 10)
    _box(ax, 8.7, 2.3, 2.4, 1.5, 'REINFORCE\n$z\\,\\nabla\\log p_\\rho$', '#D9EAD3', 10)
    _arrow(ax, 2.9, 3.05, 3.15, 3.05)
    _arrow(ax, 5.75, 3.05, 6.0, 3.05)
    _arrow(ax, 8.4, 3.05, 8.65, 3.05)
    ax.text(5.7, 1.15, '对手用自己的旧拷贝，避免策略对着一个固定靶子过拟合。',
            ha='center', fontsize=10, color='#444444')
    ax.text(5.7, 0.4, '这和 s20 的策略梯度是同一类东西；棋上还可以再套 MCTS 把「这一手」算稳。',
            ha='center', fontsize=10, color='#444444')
    _save(fig, 'ag-02-self-play.png')


def draw_mcts_cycle():
    fig, ax = plt.subplots(figsize=(11.4, 5.2))
    ax.set_xlim(0, 11.4)
    ax.set_ylim(0, 5.4)
    ax.axis('off')
    ax.text(5.7, 5.05, 'MCTS 一轮：选择、扩展、评估、回传',
            ha='center', fontsize=14, fontweight='bold')
    _box(ax, 0.4, 2.6, 2.3, 1.6, '选择\n沿 PUCT 走到叶子', '#CDE7F0', 10)
    _box(ax, 3.1, 2.6, 2.3, 1.6, '扩展\n写入先验 $P(s,a)$', '#D9EAD3', 10)
    _box(ax, 5.8, 2.6, 2.4, 1.6, '评估\n$v_\\theta$（+ 可选滚出）', '#FDE8D7', 10)
    _box(ax, 8.6, 2.6, 2.4, 1.6, '回传\n路径上累加 $W,N$', '#E8D5F2', 10)
    _arrow(ax, 2.75, 3.4, 3.05, 3.4)
    _arrow(ax, 5.45, 3.4, 5.75, 3.4)
    _arrow(ax, 8.25, 3.4, 8.55, 3.4)
    ax.annotate('', xy=(1.55, 2.55), xytext=(9.8, 2.55),
                arrowprops=dict(arrowstyle='-|>', color='#888888',
                                connectionstyle='arc3,rad=-0.32', lw=1.4))
    ax.text(5.7, 1.15,
            r'PUCT $= Q + c_{\mathrm{puct}}\,P\,\sqrt{\sum N}/(1+N)$：$Q$ 利用，$P/\sqrt{N}$ 探索。',
            ha='center', fontsize=10)
    ax.text(5.7, 0.4, '对局时按根节点访问次数 $N$ 落子，而不是网络瞬时 $\\arg\\max$。',
            ha='center', fontsize=10, color='#444444')
    _save(fig, 'ag-03-mcts-cycle.png')


def draw_search_vs_policy():
    fig, ax = plt.subplots(figsize=(11.4, 4.8))
    ax.set_xlim(0, 11.4)
    ax.set_ylim(0, 5.0)
    ax.axis('off')
    ax.text(5.7, 4.6, '从棋感到对局：MCTS 把先验炼成访问次数',
            ha='center', fontsize=13, fontweight='bold')
    _box(ax, 0.4, 2.2, 3.2, 1.7, '网络先验 $\\pi(a\\mid s)$\n「感觉这手像」', '#CDE7F0', 11)
    _box(ax, 4.1, 2.2, 3.2, 1.7, 'MCTS 搜索\n把算力砸在争议分叉', '#FDE8D7', 11)
    _box(ax, 7.8, 2.2, 3.2, 1.7, '访问分布 $\\pi_{MCTS}$\n「算完真该下这手」', '#D9EAD3', 11)
    _arrow(ax, 3.65, 3.05, 4.05, 3.05)
    _arrow(ax, 7.35, 3.05, 7.75, 3.05)
    ax.text(5.7, 1.15, 'AlphaGo Zero：用 $\\pi_{MCTS}$ 当策略标签、终局 $z$ 当价值标签，搜索本身就是策略改进算子。',
            ha='center', fontsize=10, color='#444444')
    ax.text(5.7, 0.4, 'LLM 对齐通常不长这棵树，但「先验 + 评估」的分工还在（RM / 验证器）。',
            ha='center', fontsize=10, color='#444444')
    _save(fig, 'ag-04-search-vs-policy.png')


def eval_match(n=N_GAMES):
    stats = {'MCTS vs 随机': [0, 0, 0], 'MCTS vs 中心贪心': [0, 0, 0]}
    for _ in range(n):
        w = play_game(mcts_policy, random_move)
        stats['MCTS vs 随机'][{X: 0, 0: 1, O: 2}[w]] += 1
        w = play_game(mcts_policy, greedy_center)
        stats['MCTS vs 中心贪心'][{X: 0, 0: 1, O: 2}[w]] += 1
    return stats


def main():
    print('绘制正文示意图…')
    draw_three_networks()
    draw_self_play()
    draw_mcts_cycle()
    draw_search_vs_policy()

    board = [EMPTY] * 9
    _, visits, _ = mcts_move(board, X, n_sim=400)
    print('空棋盘、先手 X，400 次模拟后的根访问：')
    for a in range(9):
        print(f'  格 {a}: N={visits.get(a, 0)}')

    stats = eval_match()
    print(f'\n各 {N_GAMES} 局（MCTS 执 X）：')
    for k, (win, draw, lose) in stats.items():
        print(f'  {k}: 胜 {win} 和 {draw} 负 {lose}')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = [f'{i}\n{"中心" if i == 4 else ""}' for i in range(9)]
    axes[0].bar(range(9), [visits.get(i, 0) for i in range(9)], color='#3b82f6')
    axes[0].set_xticks(range(9))
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_title('根节点访问次数 N（空盘）')
    axes[0].set_ylabel('N')

    names = list(stats.keys())
    wins = [stats[k][0] / N_GAMES for k in names]
    axes[1].bar(names, wins, color='#ef4444')
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel('胜率（执 X）')
    axes[1].set_title(f'MCTS ({N_SIM} 模拟) vs 弱对手')
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'mcts_tic_tac_toe.png')
    fig.savefig(out, dpi=140)
    print(f'\n图已保存: {out}')


if __name__ == '__main__':
    main()
