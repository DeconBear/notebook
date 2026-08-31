# -*- coding: utf-8 -*-
"""
===============================================================================
as07_alphachip_eda/code/demo.py — AlphaChip 玩具版：用强化学习做芯片布局
===============================================================================
芯片布局规划（floorplanning/placement）：给定一组功能模块（宏单元，macro）
及它们之间的连接关系（netlist），把每个模块摆放到芯片版图的某个位置，
使总连线长度（wirelength）尽量短——这直接影响芯片的功耗、时序和面积（PPA）。

AlphaChip（脱胎自 Mirhoseini et al. 2021《A graph placement methodology for
fast chip design》）把这个问题表述为一个序贯决策过程：
    1. 用图神经网络（呼应 as05）编码 netlist 的连接结构，得到每个宏单元的表征
    2. 智能体依次为每个宏单元选择摆放位置（网格上的一个格子）
    3. 用强化学习（策略梯度）训练策略网络，最小化最终布局的总线长

本 demo 实现一个简化但完整的教学版本：
    - 合成一个带"簇状"连接结构的 netlist（模拟真实芯片中功能相关的模块
      连接更紧密的现象）
    - 用消息传递 GNN 编码 netlist，得到每个宏单元的表征
    - 用策略网络 + REINFORCE（配合滑动平均基线降方差）训练一个序贯放置策略
    - 对比三种放置策略：随机放置 / 贪心启发式（按连接度排序+螺旋摆放）/
      RL 学到的策略，比较最终总线长

运行方式：cd docs/science/alphachip/code && python demo.py
依赖：numpy, torch, matplotlib
===============================================================================
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)


def _save_path(name):
    return os.path.join(_IMAGES_DIR, name)


# ============================================================================
# 第一部分：合成 netlist（带簇状结构，模拟真实芯片模块连接模式）
# ============================================================================

N_MACROS = 16          # 宏单元数量
GRID_SIZE = 4           # 放置网格 GRID_SIZE x GRID_SIZE（= N_MACROS，每格恰好放一个宏单元）
N_CLUSTERS = 4          # 功能簇数量（如 CPU核/缓存/IO/电源域 的类比）


def generate_netlist(n_macros=N_MACROS, n_clusters=N_CLUSTERS, seed=42):
    """
    生成一个带簇状结构的合成 netlist：
      1. 把 n_macros 个宏单元随机分配到 n_clusters 个簇中
      2. 簇内的宏单元之间以较高概率连接（模拟同一功能模块内部信号密集）
      3. 簇间也有一些连接（模拟模块间必要的通信，如总线/控制信号）

    返回:
        edges: list of (i, j, weight)，weight 模拟连线的"关键性"（如信号频率/位宽）
        cluster_id: (n_macros,) 每个宏单元所属簇
    """
    rng = np.random.RandomState(seed)
    cluster_id = rng.randint(0, n_clusters, size=n_macros)

    edges = []
    for i in range(n_macros):
        for j in range(i + 1, n_macros):
            same_cluster = cluster_id[i] == cluster_id[j]
            prob = 0.55 if same_cluster else 0.08
            if rng.random() < prob:
                weight = rng.uniform(2.0, 5.0) if same_cluster else rng.uniform(0.5, 1.5)
                edges.append((i, j, weight))
    return edges, cluster_id


EDGES, CLUSTER_ID = generate_netlist()
N_EDGES = len(EDGES)
print(f"[初始化] 合成 netlist: {N_MACROS} 个宏单元, {N_EDGES} 条连线, {N_CLUSTERS} 个功能簇")


def wirelength(positions, edges):
    """
    计算总线长（半周长模型 HPWL 的简化版：用 Manhattan 距离近似两点间连线长度）。

    参数:
        positions: (n_macros, 2) 每个宏单元的 (row, col) 网格坐标
        edges: list of (i, j, weight)
    返回:
        total: float，加权总线长
    """
    total = 0.0
    for i, j, w in edges:
        d = abs(positions[i][0] - positions[j][0]) + abs(positions[i][1] - positions[j][1])
        total += w * d
    return total


# ============================================================================
# 第二部分：GNN 编码 netlist（呼应 as05 的消息传递机制）
# ============================================================================

NODE_DIM = 16
HIDDEN_DIM = 32


class NetlistGNN(nn.Module):
    """
    用消息传递编码 netlist 连接结构，得到每个宏单元的表征。
    每个宏单元的初始特征 = 它的连接度（度数）+ 所属簇的 one-hot（模拟真实系统
    中宏单元自带的类型/尺寸/端口数量等属性特征）。
    """

    def __init__(self, n_clusters, node_dim=NODE_DIM, hidden_dim=HIDDEN_DIM, n_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(1 + n_clusters, node_dim)
        self.message_fns = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * node_dim + 1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, node_dim))
            for _ in range(n_layers)
        ])
        self.update_fns = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * node_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, node_dim))
            for _ in range(n_layers)
        ])

    def forward(self, node_features, edge_index, edge_weight):
        h = self.input_proj(node_features)
        src, dst = edge_index[0], edge_index[1]
        for msg_fn, upd_fn in zip(self.message_fns, self.update_fns):
            m = msg_fn(torch.cat([h[src], h[dst], edge_weight.unsqueeze(-1)], dim=-1))
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, m)
            h = upd_fn(torch.cat([h, agg], dim=-1))
        return h  # (n_macros, node_dim)


def build_gnn_inputs(edges, cluster_id, n_macros, n_clusters):
    """把 netlist 转换成 GNN 需要的张量（度数特征、簇 one-hot、双向边索引）。"""
    degree = np.zeros(n_macros)
    for i, j, w in edges:
        degree[i] += w
        degree[j] += w
    degree = degree / (degree.max() + 1e-8)

    cluster_onehot = np.eye(n_clusters)[cluster_id]
    node_features = torch.tensor(np.concatenate([degree[:, None], cluster_onehot], axis=1), dtype=torch.float32)

    srcs, dsts, weights = [], [], []
    for i, j, w in edges:
        srcs += [i, j]
        dsts += [j, i]
        weights += [w, w]
    edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    return node_features, edge_index, edge_weight


# ============================================================================
# 第三部分：序贯放置策略网络 + REINFORCE 训练
# ============================================================================

class PlacementPolicy(nn.Module):
    """
    序贯放置策略：
      1. GNN 编码全部宏单元得到静态表征 node_repr (n_macros, node_dim)
      2. 按固定顺序（这里简单地按连接度从高到低排序，模拟"先放关键模块"的
         启发式顺序——真实 AlphaChip 用更复杂的顺序策略，这里简化处理）
         依次为每个宏单元选择网格位置
      3. 每一步的输入 = 当前待放置宏单元的 GNN 表征 + 当前网格占用掩码，
         输出 = 在剩余可用格子上的放置概率分布
    """

    def __init__(self, gnn, n_macros, grid_size, node_dim=NODE_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.gnn = gnn
        self.n_macros = n_macros
        self.grid_size = grid_size
        self.n_cells = grid_size * grid_size
        self.place_head = nn.Sequential(
            nn.Linear(node_dim + self.n_cells, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, self.n_cells),
        )

    def forward(self, node_features, edge_index, edge_weight, order, greedy=False):
        """
        依次放置 order 中的每个宏单元，返回:
            positions: (n_macros, 2) 网格坐标
            log_probs: list of log_prob (每一步动作的对数概率，用于 REINFORCE)
        """
        node_repr = self.gnn(node_features, edge_index, edge_weight)  # (n_macros, node_dim)
        occupied_mask = torch.zeros(self.n_cells)                    # 1 表示已占用
        positions = np.zeros((self.n_macros, 2), dtype=int)
        log_probs = []

        for macro_id in order:
            feat = torch.cat([node_repr[macro_id], occupied_mask])
            logits = self.place_head(feat)
            logits = logits.masked_fill(occupied_mask.bool(), float('-inf'))  # 屏蔽已占用格子
            dist = torch.distributions.Categorical(logits=logits)
            if greedy:
                cell = torch.argmax(logits).item()
            else:
                cell = dist.sample().item()
            log_probs.append(dist.log_prob(torch.tensor(cell)))
            occupied_mask[cell] = 1.0
            positions[macro_id] = [cell // self.grid_size, cell % self.grid_size]

        return positions, log_probs


def get_placement_order(cluster_id, edges, n_macros):
    """按连接度从高到低排序，模拟"先放关键模块"的启发式顺序（已实现，供策略网络使用）。"""
    degree = np.zeros(n_macros)
    for i, j, w in edges:
        degree[i] += w
        degree[j] += w
    return list(np.argsort(-degree))


def train_rl_placement(n_episodes=800, lr=3e-3):
    """
    用 REINFORCE 训练放置策略。

    损失: L = -E[(R - baseline) * sum(log_prob)]
    其中 R = -wirelength(最终布局)（越短线长，奖励越高），
    baseline 用滑动平均奖励，降低方差、加速收敛（标准的 REINFORCE-with-baseline 技巧）。
    """
    node_features, edge_index, edge_weight = build_gnn_inputs(EDGES, CLUSTER_ID, N_MACROS, N_CLUSTERS)
    order = get_placement_order(CLUSTER_ID, EDGES, N_MACROS)

    gnn = NetlistGNN(N_CLUSTERS)
    policy = PlacementPolicy(gnn, N_MACROS, GRID_SIZE)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    baseline = None
    reward_history = []
    wl_history = []

    for ep in range(n_episodes):
        positions, log_probs = policy(node_features, edge_index, edge_weight, order, greedy=False)
        wl = wirelength(positions, EDGES)
        reward = -wl

        if baseline is None:
            baseline = reward
        else:
            baseline = 0.95 * baseline + 0.05 * reward

        advantage = reward - baseline
        loss = -advantage * torch.stack(log_probs).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        reward_history.append(reward)
        wl_history.append(wl)
        if (ep + 1) % 100 == 0:
            print(f"  episode {ep+1:4d}/{n_episodes} | wirelength={wl:.2f} | baseline={-baseline:.2f}")

    return policy, node_features, edge_index, edge_weight, order, reward_history, wl_history


# ============================================================================
# 第四部分：基线放置策略 —— 随机 / 贪心启发式
# ============================================================================

def random_placement(n_macros, grid_size, seed=0):
    rng = np.random.RandomState(seed)
    cells = rng.permutation(grid_size * grid_size)[:n_macros]
    positions = np.array([[c // grid_size, c % grid_size] for c in cells])
    return positions


def greedy_heuristic_placement(cluster_id, edges, n_macros, grid_size):
    """
    贪心启发式：按簇分组，把同一簇的宏单元尽量摆在网格上相邻的位置
    （模拟传统 EDA 工具中"先分区再摆放"的思路，不使用任何学习）。
    """
    order = np.argsort(cluster_id)  # 同簇的宏单元编号相邻排列
    positions = np.zeros((n_macros, 2), dtype=int)
    for idx, macro_id in enumerate(order):
        positions[macro_id] = [idx // grid_size, idx % grid_size]
    return positions


# ============================================================================
# 第五部分：可视化
# ============================================================================

CLUSTER_COLORS = ['#E74C3C', '#3498DB', '#2ECC71', '#F1C40F', '#9B59B6', '#1ABC9C']


def plot_netlist_graph(save_name='alphachip_netlist.png'):
    """绘制合成 netlist 的图结构（用簡易弹簧布局近似，仅用于展示连接模式）。"""
    rng = np.random.RandomState(1)
    pos2d = {i: rng.uniform(-1, 1, size=2) for i in range(N_MACROS)}
    # 简单的力引导迭代，让同簇节点更靠近
    for _ in range(200):
        forces = {i: np.zeros(2) for i in range(N_MACROS)}
        for i, j, w in EDGES:
            diff = pos2d[j] - pos2d[i]
            dist = np.linalg.norm(diff) + 1e-6
            f = 0.02 * w * diff / dist
            forces[i] += f
            forces[j] -= f
        for i in range(N_MACROS):
            for j in range(N_MACROS):
                if i != j:
                    diff = pos2d[i] - pos2d[j]
                    dist = np.linalg.norm(diff) + 1e-6
                    forces[i] += 0.001 * diff / (dist ** 2)
        for i in range(N_MACROS):
            pos2d[i] += 0.05 * forces[i]

    fig, ax = plt.subplots(figsize=(7, 7))
    for i, j, w in EDGES:
        ax.plot([pos2d[i][0], pos2d[j][0]], [pos2d[i][1], pos2d[j][1]],
                color='gray', alpha=0.4, linewidth=0.5 + w * 0.3, zorder=1)
    for i in range(N_MACROS):
        ax.scatter(*pos2d[i], s=350, color=CLUSTER_COLORS[CLUSTER_ID[i] % len(CLUSTER_COLORS)],
                   edgecolors='black', zorder=2)
        ax.text(*pos2d[i], str(i), ha='center', va='center', fontsize=8, zorder=3)
    ax.set_title('合成 Netlist：16个宏单元，4个功能簇（颜色区分）', fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    fig.savefig(_save_path(save_name), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[可视化] netlist 图结构已保存至 images/{save_name}')


def plot_placement(ax, positions, title, grid_size=GRID_SIZE):
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.invert_yaxis()
    for r in range(grid_size + 1):
        ax.axhline(r - 0.5, color='lightgray', linewidth=0.8)
    for c in range(grid_size + 1):
        ax.axvline(c - 0.5, color='lightgray', linewidth=0.8)
    for i, j, w in EDGES:
        r1, c1 = positions[i]
        r2, c2 = positions[j]
        ax.plot([c1, c2], [r1, r2], color='gray', alpha=0.35, linewidth=0.4 + w * 0.25, zorder=1)
    for i in range(N_MACROS):
        r, c = positions[i]
        ax.scatter(c, r, s=420, color=CLUSTER_COLORS[CLUSTER_ID[i] % len(CLUSTER_COLORS)],
                   edgecolors='black', zorder=2)
        ax.text(c, r, str(i), ha='center', va='center', fontsize=8, zorder=3)
    wl = wirelength(positions, EDGES)
    ax.set_title(f'{title}\n总线长(HPWL近似) = {wl:.1f}', fontsize=11, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    return wl


def plot_placement_comparison(pos_random, pos_greedy, pos_rl, save_name='alphachip_placement_comparison.png'):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    wl_r = plot_placement(axes[0], pos_random, '随机放置')
    wl_g = plot_placement(axes[1], pos_greedy, '贪心启发式\n(按簇分组摆放)')
    wl_rl = plot_placement(axes[2], pos_rl, 'RL学到的策略\n(GNN编码 + 策略梯度)')
    plt.tight_layout()
    fig.savefig(_save_path(save_name), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[可视化] 三种放置方案对比已保存至 images/{save_name}')
    return wl_r, wl_g, wl_rl


def plot_training_curve(wl_history, save_name='alphachip_training_curve.png'):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(wl_history, color='#2E86AB', alpha=0.35, linewidth=1, label='单轮线长')
    window = 30
    if len(wl_history) > window:
        smooth = np.convolve(wl_history, np.ones(window) / window, mode='valid')
        ax.plot(np.arange(window - 1, len(wl_history)), smooth, color='#C0392B', linewidth=2,
                 label=f'滑动平均(window={window})')
    ax.set_xlabel('训练轮数(episode)')
    ax.set_ylabel('总线长(越低越好)')
    ax.set_title('RL 放置策略训练曲线', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(_save_path(save_name), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[可视化] 训练曲线已保存至 images/{save_name}')


def draw_alphachip_pipeline_diagram(save_name='as07-01-alphachip-pipeline.png'):
    """手绘 AlphaChip 流水线概念图：netlist -> GNN编码 -> 序贯放置策略 -> RL训练。"""
    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 6.5)
    ax.axis('off')

    def box(x, y, w, h, text, color, fontsize=10):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", linewidth=1.5,
                            edgecolor='#333333', facecolor=color)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fontsize, fontweight='bold')

    def arrow(x1, y1, x2, y2):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=15,
                            linewidth=1.6, color='#333333')
        ax.add_patch(a)

    ax.text(6.25, 6.15, 'AlphaChip 玩具版流水线（教学示意图）', ha='center', fontsize=13.5, fontweight='bold')

    box(0.3, 4.0, 2.1, 1.3, 'Netlist\n(宏单元+连接关系)', '#E8E8E8')
    box(2.9, 4.0, 2.3, 1.3, 'GNN 编码\n(消息传递,呼应as05)', '#CDE7F0', fontsize=9.5)
    box(5.7, 4.0, 2.6, 1.3, '策略网络\n依次选择放置位置\n(带占用掩码)', '#D9EAD3', fontsize=9)
    box(8.8, 4.0, 2.4, 1.3, '完整布局\n计算总线长 -> 奖励R', '#FDE8D7', fontsize=9.5)
    arrow(2.4, 4.65, 2.9, 4.65)
    arrow(5.2, 4.65, 5.7, 4.65)
    arrow(8.3, 4.65, 8.8, 4.65)

    box(4.4, 1.0, 4.0, 1.3, 'REINFORCE:\nL = -(R - baseline)*sum(log_prob)\n反向传播更新策略网络+GNN', '#F9D5D3', fontsize=9)
    arrow(10.0, 4.0, 6.5, 2.3)
    ax.text(8.5, 3.0, 'reward = -wirelength', fontsize=8.5, color='#7F2704')

    plt.tight_layout()
    fig.savefig(_save_path(save_name), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[可视化] AlphaChip 流水线示意图已保存至 images/{save_name}')


# ============================================================================
# 主程序
# ============================================================================

def main():
    print('=' * 70)
    print('as07 AlphaChip 玩具版：用强化学习做芯片布局')
    print('=' * 70)

    print('\n[1/5] 生成 netlist 可视化...')
    plot_netlist_graph()
    draw_alphachip_pipeline_diagram()

    print('\n[2/5] 训练 RL 放置策略 (REINFORCE + GNN编码)...')
    policy, node_features, edge_index, edge_weight, order, reward_hist, wl_hist = train_rl_placement(
        n_episodes=800, lr=3e-3)

    print('\n[3/5] 用训练好的策略做贪婪推理(取概率最高的位置)...')
    pos_rl, _ = policy(node_features, edge_index, edge_weight, order, greedy=True)

    print('\n[4/5] 计算基线方案: 随机放置 / 贪心启发式...')
    pos_random = random_placement(N_MACROS, GRID_SIZE, seed=7)
    pos_greedy = greedy_heuristic_placement(CLUSTER_ID, EDGES, N_MACROS, GRID_SIZE)

    print('\n[5/5] 生成对比可视化...')
    wl_r, wl_g, wl_rl = plot_placement_comparison(pos_random, pos_greedy, pos_rl)
    plot_training_curve(wl_hist)

    print('\n' + '=' * 70)
    print('【总结】总线长对比 (越低越好)')
    print('=' * 70)
    print(f'  随机放置:       {wl_r:.2f}')
    print(f'  贪心启发式:     {wl_g:.2f}  (改善 {(1 - wl_g/wl_r)*100:.1f}% vs 随机)')
    print(f'  RL学到的策略:   {wl_rl:.2f}  (改善 {(1 - wl_rl/wl_r)*100:.1f}% vs 随机, '
          f'{(1 - wl_rl/wl_g)*100:+.1f}% vs 贪心)')
    print('\n  核心结论: RL 策略通过与环境（这里是"计算线长"这个简单模拟器）反复试错，')
    print('  学到了比人工启发式更好（或至少相当）的放置方案，且不需要为每个新 netlist')
    print('  手工设计新的启发式规则——这正是 AlphaChip 论文的核心卖点：把布局问题的')
    print('  "专家经验"替换为"从数据/试错中学习的策略"。')
    print(f'\n  所有图片已保存至 {_IMAGES_DIR}')
    print('=' * 70)
    print('\n  运行完成！\n')


if __name__ == '__main__':
    main()
