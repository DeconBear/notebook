# -*- coding: utf-8 -*-
"""
=== 图神经网络 GNN ===
两社团玩具图上从零实现 GCN 与 GAT（节点分类），不依赖 PyG。
运行: python demo.py
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)
torch.manual_seed(42)
np.random.seed(42)


def make_two_community_graph(n_each=10, p_in=0.55, p_out=0.08):
    """团内边密、团间边稀。特征：度数 + 微弱社团指示（可被噪声盖住）。"""
    n = n_each * 2
    y = np.array([0] * n_each + [1] * n_each)
    A = np.zeros((n, n), dtype=np.float32)
    rng = np.random.RandomState(42)
    for i in range(n):
        for j in range(i + 1, n):
            same = y[i] == y[j]
            if rng.rand() < (p_in if same else p_out):
                A[i, j] = A[j, i] = 1.0
    deg = A.sum(axis=1, keepdims=True)
    flag = y.reshape(-1, 1).astype(np.float32) + 0.4 * rng.randn(n, 1)
    x = np.concatenate([deg / deg.max(), flag], axis=1).astype(np.float32)
    src, dst = np.where(A > 0)
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
    return (torch.from_numpy(x), edge_index, torch.from_numpy(A),
            torch.from_numpy(y.astype(np.int64)), n_each)


def gcn_norm_adj(A: torch.Tensor) -> torch.Tensor:
    """D^{-1/2} (A+I) D^{-1/2}"""
    n = A.shape[0]
    A_hat = A + torch.eye(n)
    deg = A_hat.sum(dim=1).clamp(min=1.0)
    d_inv = deg.pow(-0.5)
    return d_inv.unsqueeze(1) * A_hat * d_inv.unsqueeze(0)


class GCN(nn.Module):
    def __init__(self, in_dim, hid=16, n_class=2):
        super().__init__()
        self.w1 = nn.Linear(in_dim, hid)
        self.w2 = nn.Linear(hid, n_class)

    def forward(self, x, A_norm):
        h = F.relu(A_norm @ self.w1(x))
        return self.w2(A_norm @ h)


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)

    def forward(self, h, edge_index, n, activate=True):
        src, dst = edge_index
        wh = self.W(h)
        e = F.leaky_relu(self.a(torch.cat([wh[dst], wh[src]], dim=-1)), 0.2).squeeze(-1)
        alpha = torch.zeros_like(e)
        for i in range(n):
            mask = dst == i
            if mask.any():
                alpha[mask] = torch.softmax(e[mask], dim=0)
        out = torch.zeros(n, wh.shape[1], device=h.device)
        out.index_add_(0, dst, alpha.unsqueeze(-1) * wh[src])
        return F.elu(out) if activate else out


class GAT(nn.Module):
    def __init__(self, in_dim, hid=16, n_class=2):
        super().__init__()
        self.g1 = GATLayer(in_dim, hid)
        self.g2 = GATLayer(hid, n_class)

    def forward(self, x, edge_index):
        n = x.shape[0]
        h = self.g1(x, edge_index, n, activate=True)
        return self.g2(h, edge_index, n, activate=False)


def train_node_clf(model, x, extra, y, kind='gcn', steps=200, lr=0.05):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses, accs = [], []
    for _ in range(steps):
        model.train()
        logit = model(x, extra)
        loss = F.cross_entropy(logit, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        pred = logit.argmax(dim=-1)
        accs.append(float((pred == y).float().mean()))
        losses.append(float(loss.detach()))
    with torch.no_grad():
        logit = model(x, extra)
        emb = logit  # 2 维 logit 当可视化嵌入
        pred = logit.argmax(dim=-1)
    return losses, accs, emb, pred


def draw_results(x, A, y, n_each, gcn_emb, gat_emb, gcn_acc, gat_acc):
    cream, navy = '#FBF7F0', '#1F3A5F'
    n = A.shape[0]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.0))
    fig.patch.set_facecolor(cream)

    # 左：图
    ax = axes[0]
    ax.set_facecolor(cream)
    pos = np.column_stack([
        np.concatenate([np.full(n_each, 0.25), np.full(n_each, 0.75)]),
        np.concatenate([np.linspace(0.12, 0.88, n_each), np.linspace(0.12, 0.88, n_each)]),
    ])
    rng = np.random.RandomState(0)
    pos = pos + 0.03 * rng.randn(n, 2)
    src, dst = np.where(np.triu(A.numpy(), 1))
    for i, j in zip(src, dst):
        ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], color='#BBB', lw=0.7, zorder=0)
    ax.scatter(pos[:, 0], pos[:, 1], c=y.numpy(), cmap='coolwarm', s=70, zorder=2, edgecolors='#333')
    ax.set_title('数据：两社团图', color=navy)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    def scatter_emb(ax, emb, acc, title):
        ax.set_facecolor(cream)
        z = emb.detach().numpy()
        ax.scatter(z[:, 0], z[:, 1], c=y.numpy(), cmap='coolwarm', s=70, edgecolors='#333')
        ax.set_title(f'{title}\n准确率 {acc[-1]*100:.0f}%', color=navy)
        ax.set_xticks([]); ax.set_yticks([])

    scatter_emb(axes[1], gcn_emb, gcn_acc, 'GCN 输出 logit')
    scatter_emb(axes[2], gat_emb, gat_acc, 'GAT 输出 logit')
    fig.suptitle('同一张图：GCN 均匀混邻居，GAT 学边权', color=navy, fontsize=13, fontweight='bold')
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'gnn_gcn_gat.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=cream)
    plt.close(fig)
    print('保存', out)


if __name__ == '__main__':
    x, edge_index, A, y, n_each = make_two_community_graph()
    A_norm = gcn_norm_adj(A)
    gcn = GCN(x.shape[1])
    gat = GAT(x.shape[1])
    _, acc_g, emb_g, _ = train_node_clf(gcn, x, A_norm, y, 'gcn')
    _, acc_a, emb_a, _ = train_node_clf(gat, x, edge_index, y, 'gat')
    print(f'GCN acc={acc_g[-1]:.3f}  GAT acc={acc_a[-1]:.3f}')
    draw_results(x, A, y, n_each, emb_g, emb_a, acc_g, acc_a)
