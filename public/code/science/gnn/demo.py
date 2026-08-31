# -*- coding: utf-8 -*-
"""
===============================================================================
as05_gnn_science/code/demo.py — 科学计算中的 GNN：消息传递演示
===============================================================================
本 demo 用一个「玩具网格平滑 + 节点属性预测」问题展示科学计算里的图神经网络：

场景 A（网格平滑 / 热扩散）：
    把一维/二维网格看成图：节点 = 网格点，边 = 相邻关系。
    消息传递的一层等价于一次离散拉普拉斯平滑：
        h_i^{t+1} = (1-α) h_i^t + α * mean_{j∈N(i)} h_j^t
    这正是 PDE 数值方法（有限差分 / 有限元）与 GNN 之间最直观的桥梁。

场景 B（分子图：节点属性预测）：
    构造一个小分子图（原子=节点，化学键=边），用 2 层消息传递
    聚合邻居信息，预测每个原子的「局部环境特征」（玩具标签）。
    这条线通向 AlphaFold / 分子力场 / 材料性质预测。

输出：
  - as05-01-message-passing.png  消息传递概念图
  - mesh_smoothing.png           网格平滑前后对比
  - molecule_mp_result.png       分子图 + 消息传递前后节点特征

运行方式：cd docs/science/gnn/code && python demo.py
依赖：numpy, torch, matplotlib
===============================================================================
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)


# ============================================================================
# 第 1 部分：图的基本数据结构（邻接列表）
# ============================================================================

def build_grid_graph_1d(n: int):
    """
    构造一维链图：节点 0—1—2—...—(n-1)，返回边列表 (src, dst) 双向。
    """
    edges = []
    for i in range(n - 1):
        edges.append((i, i + 1))
        edges.append((i + 1, i))
    return edges


def build_grid_graph_2d(h: int, w: int):
    """
    构造二维网格图（4-邻接），节点编号 row-major：i = r*w + c。
    """
    edges = []
    for r in range(h):
        for c in range(w):
            i = r * w + c
            if c + 1 < w:
                j = r * w + (c + 1)
                edges.append((i, j))
                edges.append((j, i))
            if r + 1 < h:
                j = (r + 1) * w + c
                edges.append((i, j))
                edges.append((j, i))
    return edges


def edges_to_adj_list(n: int, edges):
    """边列表 -> 邻接表 dict[int, list[int]]"""
    adj = {i: [] for i in range(n)}
    for s, d in edges:
        adj[s].append(d)
    return adj


# ============================================================================
# 第 2 部分：消息传递的两种实现（NumPy 平滑 + PyTorch GNN 层）
# ============================================================================

def message_passing_smooth(h: np.ndarray, adj: dict, alpha: float = 0.5) -> np.ndarray:
    """
    一层「均值聚合」消息传递 —— 网格平滑 / 热扩散的离散版本。

        h_i' = (1-α) * h_i + α * mean_{j ∈ N(i)} h_j

    当图是规则网格、α=1 时，这正是离散拉普拉斯平滑算子的一步。
    """
    h_new = h.copy()
    for i, neighbors in adj.items():
        if not neighbors:
            continue
        msg = np.mean([h[j] for j in neighbors], axis=0)
        h_new[i] = (1 - alpha) * h[i] + alpha * msg
    return h_new


class MessagePassingLayer(nn.Module):
    """
    可学习的消息传递层（Mean Aggregation + 线性变换）。

    消息函数：m_{j→i} = W_msg * h_j
    聚合函数：m_i = mean_{j∈N(i)} m_{j→i}
    更新函数：h_i' = σ( W_self * h_i + m_i )
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W_msg = nn.Linear(in_dim, out_dim, bias=False)
        self.W_self = nn.Linear(in_dim, out_dim, bias=True)
        self.act = nn.ReLU()

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # h: (N, F_in), edge_index: (2, E) 其中 edge_index[0]=src, [1]=dst
        src, dst = edge_index[0], edge_index[1]
        messages = self.W_msg(h[src])                       # (E, F_out)

        # 按目标节点聚合（均值）
        n = h.shape[0]
        agg = torch.zeros(n, messages.shape[1], dtype=h.dtype)
        count = torch.zeros(n, 1, dtype=h.dtype)
        agg.index_add_(0, dst, messages)
        count.index_add_(0, dst, torch.ones(dst.shape[0], 1))
        count = count.clamp(min=1.0)
        agg = agg / count

        return self.act(self.W_self(h) + agg)


class TinyGNN(nn.Module):
    """两层消息传递 + 读出头，用于节点属性预测。"""

    def __init__(self, in_dim: int, hidden: int = 16, out_dim: int = 1):
        super().__init__()
        self.mp1 = MessagePassingLayer(in_dim, hidden)
        self.mp2 = MessagePassingLayer(hidden, hidden)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, h, edge_index):
        h = self.mp1(h, edge_index)
        h = self.mp2(h, edge_index)
        return self.head(h)


# ============================================================================
# 第 3 部分：概念图 — 消息传递
# ============================================================================

def draw_message_passing_diagram(save_path: str):
    """手绘风格：中心节点收集邻居消息，再更新自身表示。"""
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')

    ax.text(5, 6.6, '科学计算中的消息传递（Message Passing）',
            ha='center', fontsize=14, weight='bold')

    # 左：图结构
    ax.text(2.5, 5.9, '① 把物理对象建成图', ha='center', fontsize=11, weight='bold', color='#1F4E79')
    # 分子示意
    mol_nodes = [(1.2, 4.6), (2.0, 5.2), (2.8, 4.6), (2.0, 3.9), (3.4, 5.2)]
    mol_edges = [(0, 1), (1, 2), (1, 3), (2, 4)]
    for i, j in mol_edges:
        ax.plot([mol_nodes[i][0], mol_nodes[j][0]],
                [mol_nodes[i][1], mol_nodes[j][1]], 'k-', lw=1.5, zorder=1)
    colors = ['#E74C3C', '#3498DB', '#E74C3C', '#2ECC71', '#3498DB']
    labels = ['O', 'C', 'O', 'H', 'N']
    for (x, y), c, lab in zip(mol_nodes, colors, labels):
        circ = Circle((x, y), 0.22, facecolor=c, edgecolor='black', zorder=2)
        ax.add_patch(circ)
        ax.text(x, y, lab, ha='center', va='center', color='white', fontsize=9, weight='bold', zorder=3)
    ax.text(2.2, 3.35, '分子图 / 网格图 / 粒子图', ha='center', fontsize=9, color='#555')

    # 中：消息传递公式
    ax.text(5.5, 5.9, '② 一层消息传递', ha='center', fontsize=11, weight='bold', color='#1F4E79')
    box = FancyBboxPatch((3.8, 3.5), 3.4, 2.1, boxstyle='round,pad=0.05',
                          facecolor='#FFF2CC', edgecolor='#BF8F00', lw=1.5)
    ax.add_patch(box)
    ax.text(5.5, 5.2, r'$m_{j \to i} = \phi(h_j, h_i, e_{ij})$', ha='center', fontsize=11)
    ax.text(5.5, 4.55, r'$m_i = \mathrm{AGG}_{j \in \mathcal{N}(i)} m_{j\to i}$', ha='center', fontsize=11)
    ax.text(5.5, 3.9, r"$h_i' = \psi(h_i, m_i)$", ha='center', fontsize=11)

    # 右：应用
    ax.text(8.3, 5.9, '③ 科学应用', ha='center', fontsize=11, weight='bold', color='#1F4E79')
    apps = [
        (8.3, 5.15, '网格 PDE / 流体', '#DDEBF7'),
        (8.3, 4.45, '分子性质预测', '#E2F0D9'),
        (8.3, 3.75, '蛋白质 / AlphaFold', '#FDE2E2'),
    ]
    for x, y, text, color in apps:
        box = FancyBboxPatch((x - 1.1, y - 0.28), 2.2, 0.55, boxstyle='round,pad=0.03',
                              facecolor=color, edgecolor='black', lw=1.0)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=10)

    # 底部：中心节点收集邻居
    ax.text(5, 2.85, '④ 直观：中心节点 i 收集邻居消息再更新',
            ha='center', fontsize=11, weight='bold', color='#1F4E79')
    center = (5.0, 1.35)
    neighbors = [(3.2, 1.9), (3.4, 0.7), (6.8, 1.9), (6.6, 0.7), (5.0, 0.35)]
    for nx, ny in neighbors:
        ax.annotate('', xy=center, xytext=(nx, ny),
                    arrowprops=dict(arrowstyle='-|>', color='#C0392B', lw=1.8,
                                    connectionstyle='arc3,rad=0.05'))
        circ = Circle((nx, ny), 0.22, facecolor='#85C1E9', edgecolor='black', zorder=2)
        ax.add_patch(circ)
        ax.text(nx, ny, 'j', ha='center', va='center', fontsize=9, zorder=3)
    circ = Circle(center, 0.32, facecolor='#F5B041', edgecolor='black', lw=1.5, zorder=2)
    ax.add_patch(circ)
    ax.text(center[0], center[1], 'i', ha='center', va='center', fontsize=12, weight='bold', zorder=3)
    ax.text(5, 2.35, '消息沿边流入 → 聚合 → 更新表示', ha='center', fontsize=9, color='#555')

    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close()
    print(f'  [概念图] 已保存到 {save_path}')


# ============================================================================
# 第 4 部分：场景 A — 2D 网格热扩散 / 平滑
# ============================================================================

def demo_mesh_smoothing():
    """在 16×16 网格上放一个热斑，做若干步消息传递平滑，可视化演化。"""
    print('\n[场景 A] 2D 网格消息传递 = 热扩散平滑')
    H, W = 16, 16
    n = H * W
    edges = build_grid_graph_2d(H, W)
    adj = edges_to_adj_list(n, edges)

    # 初始：中心一块高温，其余接近 0
    field = np.zeros(n)
    for r in range(H):
        for c in range(W):
            if 5 <= r <= 10 and 5 <= c <= 10:
                field[r * W + c] = 1.0
            else:
                field[r * W + c] = 0.05 * np.random.randn()

    snapshots = [field.reshape(H, W).copy()]
    h = field.copy()
    for step in range(12):
        h = message_passing_smooth(h, adj, alpha=0.6)
        if step in (2, 5, 11):
            snapshots.append(h.reshape(H, W).copy())

    titles = ['初始热斑 t=0', '平滑后 t=3', '平滑后 t=6', '平滑后 t=12']
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, snap, title in zip(axes, snapshots, titles):
        im = ax.imshow(snap, cmap='inferno', vmin=0, vmax=1)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label='温度 / 节点特征')
    plt.suptitle('消息传递 = 离散热扩散：网格图上的均值聚合平滑', fontsize=12, y=1.02)
    path = os.path.join(_IMAGES_DIR, 'mesh_smoothing.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  [可视化] 网格平滑已保存到 {path}')
    return snapshots


# ============================================================================
# 第 5 部分：场景 B — 玩具分子图节点属性预测
# ============================================================================

def make_toy_molecule():
    """
    构造一个玩具「乙醇」风格小分子：
      原子：C-C-O + H 们（简化，只放 6 个节点）
      节点特征：原子类型 one-hot（C/O/H）+ 度数
      标签：每个原子的「局部环境分数」= 邻居原子序数之和（玩具监督信号）
    """
    # 节点：0:C, 1:C, 2:O, 3:H, 4:H, 5:H
    atom_type = np.array([0, 0, 1, 2, 2, 2])  # 0=C, 1=O, 2=H
    # 边（无向变双向）
    undirected = [(0, 1), (1, 2), (0, 3), (0, 4), (1, 5)]
    edges = []
    for s, d in undirected:
        edges.append((s, d))
        edges.append((d, s))

    n = 6
    # one-hot 原子类型 (N, 3) + 度数 (N, 1)
    onehot = np.eye(3)[atom_type]
    degree = np.zeros((n, 1))
    for s, d in edges:
        degree[s] += 1
    degree = degree / 2.0  # 因为双向边，度数算了两次
    features = np.concatenate([onehot, degree], axis=1).astype(np.float32)

    # 玩具标签：邻居原子类型编号之和
    Z = {0: 6, 1: 8, 2: 1}  # C=6, O=8, H=1
    adj = edges_to_adj_list(n, edges)
    labels = np.zeros((n, 1), dtype=np.float32)
    for i in range(n):
        labels[i, 0] = sum(Z[atom_type[j]] for j in adj[i]) / max(len(adj[i]), 1)

    # 2D 布局（手摆，方便画图）
    pos = np.array([
        [0.0, 0.0],   # C0
        [1.2, 0.0],   # C1
        [2.2, 0.6],   # O2
        [-0.6, 0.7],  # H3
        [-0.6, -0.7], # H4
        [1.2, -0.9],  # H5
    ], dtype=np.float32)

    atom_names = ['C', 'C', 'O', 'H', 'H', 'H']
    return features, edges, labels, pos, atom_names, adj


def demo_molecule_gnn():
    """训练 TinyGNN 预测玩具分子的节点属性，可视化前后特征。"""
    print('\n[场景 B] 玩具分子图：消息传递预测节点属性')
    features, edges, labels, pos, names, adj = make_toy_molecule()
    n = features.shape[0]

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # (2, E)
    x = torch.tensor(features)
    y = torch.tensor(labels)

    model = TinyGNN(in_dim=features.shape[1], hidden=16, out_dim=1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    losses = []
    for epoch in range(300):
        opt.zero_grad()
        pred = model(x, edge_index)
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    with torch.no_grad():
        pred = model(x, edge_index).numpy().flatten()
        true = labels.flatten()
        # 取第一层消息传递后的隐藏表示做可视化
        h1 = model.mp1(x, edge_index).numpy()

    mse = float(np.mean((pred - true) ** 2))
    print(f'  训练后 MSE: {mse:.6f}  最终 loss: {losses[-1]:.6f}')
    print(f'  真实标签: {np.round(true, 3)}')
    print(f'  模型预测: {np.round(pred, 3)}')

    # ---- 可视化：左图分子 + 标签/预测，右图损失 ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    color_map = {'C': '#3498DB', 'O': '#E74C3C', 'H': '#95A5A6'}
    for s, d in edges:
        if s < d:  # 只画一次
            ax.plot([pos[s, 0], pos[d, 0]], [pos[s, 1], pos[d, 1]],
                    color='#7F8C8D', lw=2, zorder=1)
    for i in range(n):
        circ = Circle((pos[i, 0], pos[i, 1]), 0.22,
                       facecolor=color_map[names[i]], edgecolor='black', zorder=2)
        ax.add_patch(circ)
        ax.text(pos[i, 0], pos[i, 1], names[i], ha='center', va='center',
                color='white', fontsize=11, weight='bold', zorder=3)
        ax.text(pos[i, 0], pos[i, 1] + 0.38,
                f'y={true[i]:.1f}\nŷ={pred[i]:.1f}',
                ha='center', va='bottom', fontsize=8, color='#2C3E50')
    ax.set_xlim(-1.3, 2.8)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('玩具分子图：真实标签 y vs 预测 ŷ', fontsize=12)

    ax = axes[1]
    ax.plot(losses, color='#27AE60', lw=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('GNN 训练损失', fontsize=12)
    ax.set_yscale('log')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(_IMAGES_DIR, 'molecule_mp_result.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  [可视化] 分子消息传递结果已保存到 {path}')

    # 额外：可视化第一层消息传递前后节点特征的 PCA 投影（或直接用前两维）
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    ax = axes[0]
    for i in range(n):
        ax.scatter(features[i, 0], features[i, 3], s=200, c=color_map[names[i]],
                   edgecolors='k', zorder=2)
        ax.text(features[i, 0] + 0.03, features[i, 3] + 0.03, f'{names[i]}{i}', fontsize=9)
    ax.set_xlabel('原子类型 one-hot[C]')
    ax.set_ylabel('度数特征')
    ax.set_title('消息传递前：原始节点特征', fontsize=11)
    ax.grid(alpha=0.3)

    ax = axes[1]
    # 用前两维隐藏特征
    for i in range(n):
        ax.scatter(h1[i, 0], h1[i, 1], s=200, c=color_map[names[i]],
                   edgecolors='k', zorder=2)
        ax.text(h1[i, 0] + 0.02, h1[i, 1] + 0.02, f'{names[i]}{i}', fontsize=9)
    ax.set_xlabel('隐藏维 0')
    ax.set_ylabel('隐藏维 1')
    ax.set_title('1 层消息传递后：邻居信息已混入', fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path2 = os.path.join(_IMAGES_DIR, 'node_features_before_after.png')
    plt.savefig(path2, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  [可视化] 节点特征前后对比已保存到 {path2}')
    return mse


# ============================================================================
# 主流程
# ============================================================================

def main():
    print('=' * 70)
    print('as05 科学计算中的 GNN Demo：消息传递 / 网格平滑 / 分子图')
    print('=' * 70)

    print('\n[0/2] 生成消息传递概念图...')
    draw_message_passing_diagram(os.path.join(_IMAGES_DIR, 'as05-01-message-passing.png'))

    demo_mesh_smoothing()
    mse = demo_molecule_gnn()

    print('\n' + '=' * 70)
    print('结论：')
    print('  - 网格上的均值消息传递 ≡ 离散热扩散 / 拉普拉斯平滑')
    print('  - 分子图上的可学习消息传递能聚合邻居化学环境，预测节点属性')
    print('  - 同一套「消息 → 聚合 → 更新」框架，覆盖网格 PDE、分子、蛋白质')
    print(f'  - 本 demo 分子节点预测最终 MSE ≈ {mse:.4f}')
    print('=' * 70)
    print(f'\nDemo 完成！查看 {_IMAGES_DIR} 目录下的可视化结果。')


if __name__ == '__main__':
    main()
