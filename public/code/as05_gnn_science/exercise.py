# -*- coding: utf-8 -*-
"""
===============================================================================
as05_gnn_science/code/exercise.py — 消息传递动手练习
===============================================================================
练习目标：
  1. 实现均值聚合消息传递（网格平滑的一步）
  2. 实现可学习消息传递层的前向（聚合邻居线性变换）
  3. 在小图上验证：平滑后方差下降；GNN 层能区分中心/边缘节点

运行方式：
  cd docs/science/gnn/code
  python exercise.py
===============================================================================
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)


def build_path_graph(n=8):
    """一维链：0—1—2—...—(n-1)，返回邻接表与边列表。"""
    edges = []
    for i in range(n - 1):
        edges.append((i, i + 1))
        edges.append((i + 1, i))
    adj = {i: [] for i in range(n)}
    for s, d in edges:
        adj[s].append(d)
    return adj, edges


# ============================================================================
# TODO 1：实现均值消息传递平滑
# ============================================================================
def mean_message_passing(h: np.ndarray, adj: dict, alpha: float = 0.5) -> np.ndarray:
    """
    一层均值消息传递：
        h_i' = (1-α) * h_i + α * mean_{j∈N(i)} h_j

    参数:
        h:     (N,) 或 (N, F) 节点特征
        adj:   dict，邻接表
        alpha: 邻居混合系数 ∈ [0, 1]

    返回:
        h_new: 与 h 同形状

    提示：
      for i, neighbors in adj.items():
          if neighbors:
              msg = np.mean([h[j] for j in neighbors], axis=0)
              h_new[i] = (1 - alpha) * h[i] + alpha * msg
    """
    h_new = h.copy()
    # TODO: 实现均值聚合更新
    pass  # <-- 替换为你的代码（别忘了 return h_new）
    return h_new


# ============================================================================
# TODO 2：实现可学习消息传递层的聚合
# ============================================================================
def aggregate_messages(h: torch.Tensor, edge_index: torch.Tensor,
                       W_msg: nn.Linear) -> torch.Tensor:
    """
    对每条边计算消息 W_msg(h_src)，再按目标节点做均值聚合。

    参数:
        h:          (N, F_in)
        edge_index: (2, E)，edge_index[0]=src, edge_index[1]=dst
        W_msg:      nn.Linear(F_in, F_out)

    返回:
        agg: (N, F_out)，每个节点收到的平均消息

    提示：
      src, dst = edge_index[0], edge_index[1]
      messages = W_msg(h[src])                    # (E, F_out)
      agg = zeros(N, F_out); count = zeros(N, 1)
      agg.index_add_(0, dst, messages)
      count.index_add_(0, dst, ones(E, 1))
      return agg / count.clamp(min=1)
    """
    # TODO: 实现消息聚合
    pass  # <-- 替换为你的代码


class MPLayerExercise(nn.Module):
    """练习用消息传递层：self 变换 + 邻居聚合。"""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W_msg = nn.Linear(in_dim, out_dim, bias=False)
        self.W_self = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, h, edge_index):
        agg = aggregate_messages(h, edge_index, self.W_msg)
        if agg is None:
            raise RuntimeError('TODO 2 尚未实现：aggregate_messages 返回了 None')
        return torch.relu(self.W_self(h) + agg)


def main():
    print('=' * 60)
    print('as05 GNN 练习 — 请完成 TODO 1 / TODO 2')
    print('=' * 60)

    n = 8
    adj, edges = build_path_graph(n)

    # ---- 测试 TODO 1 ----
    print('\n[1] 测试均值消息传递...')
    h0 = np.zeros(n)
    h0[0] = 1.0  # 左端一个脉冲
    h1 = mean_message_passing(h0, adj, alpha=0.5)
    # 正确实现时：节点0应变小，节点1应获得一些质量
    if h1 is None or np.allclose(h1, h0):
        print('  ✗ TODO 1 未正确实现（特征没有变化）。')
        return
    if h1[1] <= 0 or h1[0] >= 1.0:
        print(f'  ✗ TODO 1 结果不合理：h={np.round(h1, 3)}')
        print('  期望：脉冲从节点0向节点1扩散，h[0]<1 且 h[1]>0')
        return
    print(f'  ✓ 平滑一步后 h={np.round(h1, 3)}')

    # 多步平滑，方差应下降
    h = h0.copy()
    vars_ = [h.var()]
    for _ in range(20):
        h = mean_message_passing(h, adj, alpha=0.5)
        vars_.append(h.var())
    if vars_[-1] < vars_[0]:
        print(f'  ✓ 20 步后方差下降：{vars_[0]:.4f} → {vars_[-1]:.4f}')
    else:
        print('  ⚠ 方差未下降，请检查 alpha 混合是否正确')

    # ---- 测试 TODO 2 ----
    print('\n[2] 测试可学习消息聚合...')
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    # 节点特征：位置编码
    x = torch.tensor(np.linspace(0, 1, n).reshape(-1, 1), dtype=torch.float32)
    layer = MPLayerExercise(1, 4)
    try:
        out = layer(x, edge_index)
    except RuntimeError as e:
        print(f'  ✗ {e}')
        return
    if out.shape != (n, 4):
        print(f'  ✗ 输出形状错误：{out.shape}，期望 {(n, 4)}')
        return
    # 端点与中间节点的表示应不同（度数不同）
    diff = (out[0] - out[n // 2]).norm().item()
    if diff < 1e-6:
        print('  ⚠ 端点与中间节点表示几乎相同，聚合可能没生效')
    else:
        print(f'  ✓ 聚合输出形状 {tuple(out.shape)}，端点/中点差异={diff:.4f}')

    # ---- 可视化 ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.plot(h0, 'o-', label='初始脉冲', color='#C0392B')
    h_show = h0.copy()
    for step, color in [(1, '#E67E22'), (5, '#27AE60'), (20, '#2980B9')]:
        for _ in range(step if step == 1 else step - (1 if step == 5 else 5)):
            # 重新从上次状态走太麻烦，直接重算
            pass
    # 重新生成几条曲线
    curves = {'t=0': h0.copy()}
    h = h0.copy()
    for t in range(1, 21):
        h = mean_message_passing(h, adj, alpha=0.5)
        if t in (1, 5, 20):
            curves[f't={t}'] = h.copy()
    for name, arr in curves.items():
        ax.plot(arr, 'o-', label=name)
    ax.set_xlabel('节点编号')
    ax.set_ylabel('特征值')
    ax.set_title('链图上的消息传递扩散')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(vars_, color='#8E44AD')
    ax.set_xlabel('平滑步数')
    ax.set_ylabel('特征方差')
    ax.set_title('扩散使方差单调下降')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(_IMAGES_DIR, 'exercise_message_passing.png')
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'\n图片已保存: {out_path}')
    print('✓ 练习完成！')
    print('=' * 60)


if __name__ == '__main__':
    main()
