# -*- coding: utf-8 -*-
"""
===============================================================================
algo08_mst_networkflow/code/demo.py — 最小生成树与网络流从零实现
===============================================================================
本演示从零实现：
  1. Prim 算法（最小生成树，堆优化）
  2. Kruskal 算法（最小生成树，并查集）
  3. Dinic 最大流算法（BFS 层图 + DFS 阻塞流）
  4. 二分图最大匹配（通过最大流转化）

运行方式：python demo.py
依赖：matplotlib（仅用于可视化）
===============================================================================
"""

import os
import heapq
from collections import deque
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)

INF = float('inf')


# ============================================================================
# 第一部分：Prim 最小生成树算法
# ============================================================================

def prim(adj, n):
    """Prim 最小生成树 —— 堆优化版 O((V+E) log V)

    参数:
      adj: 邻接表 {u: [(v, weight), ...]}
      n: 顶点数量
    返回:
      (mst_weight, mst_edges) — MST 的总权重和边列表
    """
    visited = [False] * n
    mst_weight = 0
    mst_edges = []
    # 优先队列：(weight, from_vertex, to_vertex)
    # from_vertex 为 None 表示从已选集合出发
    pq = [(0, -1, 0)]  # 从顶点 0 开始

    while pq:
        w, fr, u = heapq.heappop(pq)
        if visited[u]:
            continue
        visited[u] = True
        mst_weight += w
        if fr != -1:
            mst_edges.append((fr, u, w))

        for v, weight in adj.get(u, []):
            if not visited[v]:
                heapq.heappush(pq, (weight, u, v))

    return mst_weight, mst_edges


# ============================================================================
# 第二部分：Kruskal 最小生成树算法
# ============================================================================

class UnionFind:
    """并查集（Kruskal 的核心组件）"""
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1
        return True


def kruskal(n, edges):
    """Kruskal 最小生成树 —— O(E log E)

    参数:
      n: 顶点数量
      edges: [(weight, u, v), ...] 边列表
    返回:
      (mst_weight, mst_edges)
    """
    edges.sort()  # 按权重升序排序
    uf = UnionFind(n)
    mst_weight = 0
    mst_edges = []

    for w, u, v in edges:
        if uf.union(u, v):
            mst_weight += w
            mst_edges.append((u, v, w))
            if len(mst_edges) == n - 1:
                break

    return mst_weight, mst_edges


# ============================================================================
# 第三部分：Dinic 最大流算法
# ============================================================================

class Dinic:
    """Dinic 最大流算法 —— O(V²E)

    核心：BFS 构建层图 + DFS 找阻塞流
    """

    def __init__(self, n):
        self.n = n
        # 邻接表存储：adj[u] = [(v, cap, rev_index), ...]
        # rev_index 是反向边在 adj[v] 中的索引
        self.adj = [[] for _ in range(n)]

    def add_edge(self, u, v, cap):
        """添加有向边，同时自动添加反向边（容量 0）"""
        # 正向边
        self.adj[u].append([v, cap, len(self.adj[v])])
        # 反向边
        self.adj[v].append([u, 0, len(self.adj[u]) - 1])

    def _bfs(self, s, t):
        """BFS 构建层图 —— 返回 True 表示还存在增广路径"""
        self.level = [-1] * self.n
        q = deque([s])
        self.level[s] = 0
        while q:
            u = q.popleft()
            for v, cap, rev in self.adj[u]:
                if cap > 0 and self.level[v] < 0:
                    self.level[v] = self.level[u] + 1
                    q.append(v)
        return self.level[t] >= 0

    def _dfs(self, u, t, f):
        """DFS 在层图中找阻塞流"""
        if u == t:
            return f

        for i in range(self.it[u], len(self.adj[u])):
            self.it[u] = i
            v, cap, rev = self.adj[u][i]
            if cap > 0 and self.level[u] < self.level[v]:
                d = self._dfs(v, t, min(f, cap))
                if d > 0:
                    # 更新残量
                    self.adj[u][i][1] -= d
                    self.adj[v][rev][1] += d
                    return d
        return 0

    def max_flow(self, s, t):
        """计算从 s 到 t 的最大流"""
        flow = 0
        INF_FLOW = 10**18
        while self._bfs(s, t):
            self.it = [0] * self.n  # 当前弧优化
            while True:
                f = self._dfs(s, t, INF_FLOW)
                if f == 0:
                    break
                flow += f
        return flow


# ============================================================================
# 第四部分：二分图最大匹配（通过最大流）
# ============================================================================

def bipartite_max_matching(left_size, right_size, edges):
    """二分图最大匹配 —— 转化为最大流求解

    参数:
      left_size: 左侧顶点数
      right_size: 右侧顶点数
      edges: [(left_node, right_node), ...] 左右节点的编号从 0 开始
    返回:
      最大匹配数
    """
    total_nodes = left_size + right_size + 2  # +2 for source and sink
    s = total_nodes - 2  # 源点
    t = total_nodes - 1  # 汇点

    dinic = Dinic(total_nodes)

    # 源点到左侧顶点（容量 1）
    for u in range(left_size):
        dinic.add_edge(s, u, 1)

    # 左侧到右侧的边（容量 1）
    for u, v in edges:
        dinic.add_edge(u, left_size + v, 1)

    # 右侧顶点到汇点（容量 1）
    for v in range(right_size):
        dinic.add_edge(left_size + v, t, 1)

    return dinic.max_flow(s, t)


# ============================================================================
# 第五部分：可视化
# ============================================================================

def plot_mst_networkflow_demos():
    """可视化 MST 和网络流算法"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 构建测试图
    adj = {
        0: [(1, 2), (3, 4)],
        1: [(0, 2), (2, 3), (3, 5)],
        2: [(1, 3), (3, 1)],
        3: [(0, 4), (1, 5), (2, 1)]
    }
    edges = [(2, 0, 1), (4, 0, 3), (3, 1, 2), (5, 1, 3), (1, 2, 3)]

    # 左上：Prim vs Kruskal 对比
    ax1 = axes[0, 0]
    prim_w, prim_e = prim(adj, 4)
    kruskal_w, kruskal_e = kruskal(4, edges)

    ax1.text(0.1, 0.9, f'Prim MST: 权重={prim_w}', fontsize=11, fontweight='bold')
    for i, (u, v, w) in enumerate(prim_e):
        ax1.text(0.15, 0.75 - i * 0.1, f'  {u}-{v} (w={w})', fontsize=9, fontfamily='monospace')

    ax1.text(0.1, 0.45, f'Kruskal MST: 权重={kruskal_w}', fontsize=11, fontweight='bold')
    for i, (u, v, w) in enumerate(kruskal_e):
        ax1.text(0.15, 0.3 - i * 0.1, f'  {u}-{v} (w={w})', fontsize=9, fontfamily='monospace')

    ax1.axis('off')
    ax1.set_title('Prim vs Kruskal', fontsize=14)

    # 右上：最大流演示
    ax2 = axes[0, 1]
    dinic = Dinic(4)
    dinic.add_edge(0, 1, 10)
    dinic.add_edge(0, 2, 5)
    dinic.add_edge(1, 2, 15)
    dinic.add_edge(1, 3, 10)
    dinic.add_edge(2, 3, 10)
    maxf = dinic.max_flow(0, 3)

    ax2.text(0.1, 0.9, f'流网络: 4 个顶点, 5 条边', fontsize=10)
    ax2.text(0.1, 0.75, f'边: 0→1(10), 0→2(5)', fontsize=9, fontfamily='monospace')
    ax2.text(0.1, 0.65, f'     1→2(15), 1→3(10)', fontsize=9, fontfamily='monospace')
    ax2.text(0.1, 0.55, f'     2→3(10)', fontsize=9, fontfamily='monospace')
    ax2.text(0.1, 0.35, f'Dinic 最大流: {maxf}', fontsize=14, fontweight='bold', color='#E91E63')
    ax2.text(0.1, 0.2, f'最小割容量 = 最大流 = {maxf}', fontsize=9)
    ax2.axis('off')
    ax2.set_title('Dinic 最大流', fontsize=14)

    # 左下：二分图匹配
    ax3 = axes[1, 0]
    left_nodes = ['A', 'B', 'C']
    right_nodes = ['X', 'Y', 'Z']
    match_edges = [(0, 0), (0, 1), (1, 0), (1, 2), (2, 1)]
    match_size = bipartite_max_matching(3, 3, match_edges)

    ax3.text(0.1, 0.9, f'左侧: {left_nodes}', fontsize=10)
    ax3.text(0.1, 0.8, f'右侧: {right_nodes}', fontsize=10)
    ax3.text(0.1, 0.7, f'边: A-X, A-Y, B-X, B-Z, C-Y', fontsize=9)
    ax3.text(0.1, 0.5, f'最大匹配数: {match_size}', fontsize=14, fontweight='bold', color='#4CAF50')
    ax3.text(0.1, 0.35, f'可能匹配: A-Y, B-Z, C-? (size=2)', fontsize=9)
    ax3.text(0.1, 0.25, f'或 A-X, B-Z, C-Y (size=3)', fontsize=9)
    ax3.axis('off')
    ax3.set_title('二分图最大匹配', fontsize=14)

    # 右下：算法复杂度对比表
    ax4 = axes[1, 1]
    ax4.axis('off')
    table_data = [
        ['算法', '问题', '复杂度'],
        ['Prim (堆)', 'MST', 'O((V+E)log V)'],
        ['Kruskal', 'MST', 'O(E log E)'],
        ['Dinic', '最大流', 'O(V²E)'],
        ['Dinic (二分图)', '最大匹配', 'O(E√V)'],
    ]
    table = ax4.table(cellText=table_data, cellLoc='center',
                       colWidths=[0.3, 0.3, 0.4], loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_facecolor('#1565C0')
            cell.set_text_props(color='white', fontweight='bold')
    ax4.set_title('算法复杂度速查', fontsize=14)

    plt.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'mst_networkflow_demo.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[图片保存] {out}')


# ============================================================================
# 主函数
# ============================================================================

def main():
    print('=' * 60)
    print('  最小生成树与网络流 — 演示程序')
    print('=' * 60)

    # 构建测试图
    adj = {
        0: [(1, 2), (3, 4)],
        1: [(0, 2), (2, 3), (3, 5)],
        2: [(1, 3), (3, 1)],
        3: [(0, 4), (1, 5), (2, 1)]
    }
    edges = [(2, 0, 1), (4, 0, 3), (3, 1, 2), (5, 1, 3), (1, 2, 3)]
    n = 4

    # 1. Prim
    print('\n--- Prim MST ---')
    prim_w, prim_edges = prim(adj, n)
    print(f'总权重: {prim_w}')
    for u, v, w in prim_edges:
        print(f'  {u} - {v}: {w}')

    # 2. Kruskal
    print('\n--- Kruskal MST ---')
    kruskal_w, kruskal_edges = kruskal(n, edges)
    print(f'总权重: {kruskal_w}')
    for u, v, w in kruskal_edges:
        print(f'  {u} - {v}: {w}')

    # 3. Dinic 最大流
    print('\n--- Dinic 最大流 ---')
    dinic = Dinic(4)
    dinic.add_edge(0, 1, 10)
    dinic.add_edge(0, 2, 5)
    dinic.add_edge(1, 2, 15)
    dinic.add_edge(1, 3, 10)
    dinic.add_edge(2, 3, 10)
    max_flow = dinic.max_flow(0, 3)
    print(f'源点 0 → 汇点 3 最大流: {max_flow}')

    # 4. 二分图匹配
    print('\n--- 二分图最大匹配 ---')
    match_edges = [(0, 0), (0, 1), (1, 0), (1, 2), (2, 1)]
    match = bipartite_max_matching(3, 3, match_edges)
    print(f'最大匹配数: {match}')

    # 5. 可视化
    plot_mst_networkflow_demos()

    print('\n✅ 所有演示已完成！图片保存在 images/ 目录中。')


if __name__ == '__main__':
    main()
