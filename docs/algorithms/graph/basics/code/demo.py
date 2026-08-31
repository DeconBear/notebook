# -*- coding: utf-8 -*-
"""
===============================================================================
algo06_graph_basics/code/demo.py — 图论基础从零实现
===============================================================================
本演示实现图的基本表示与核心算法：
  1. 图的多种表示法（邻接矩阵、邻接表、边列表）
  2. BFS 和 DFS 遍历
  3. 拓扑排序（Kahn 算法 + DFS 版）
  4. 二分图检测与环检测

运行方式：python demo.py
依赖：matplotlib（仅用于可视化图结构）
===============================================================================
"""

import os
from collections import deque
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)


# ============================================================================
# 第一部分：图类（支持邻接表 + 邻接矩阵 + 边列表）
# ============================================================================

class Graph:
    """通用的图类，同时维护邻接表和邻接矩阵"""

    def __init__(self, n_vertices, directed=False):
        self.V = n_vertices
        self.directed = directed
        # 邻接表
        self.adj_list = {i: [] for i in range(n_vertices)}
        # 邻接矩阵
        self.adj_matrix = [[0] * n_vertices for _ in range(n_vertices)]
        # 边列表
        self.edge_list = []

    def add_edge(self, u, v, weight=1):
        """添加边"""
        self.adj_list[u].append(v)
        self.adj_matrix[u][v] = weight
        self.edge_list.append((u, v, weight))
        if not self.directed:
            self.adj_list[v].append(u)
            self.adj_matrix[v][u] = weight

    def get_neighbors(self, u):
        return self.adj_list[u]

    def __repr__(self):
        return f'Graph(V={self.V}, E={len(self.edge_list)}, directed={self.directed})'


# ============================================================================
# 第二部分：BFS 与 DFS
# ============================================================================

def bfs(graph, start):
    """广度优先搜索 —— 返回遍历顺序和距离

    BFS 使用队列，按层遍历所有可达节点。
    时间复杂度 O(V + E)
    """
    visited = set()
    order = []
    distance = {}
    q = deque([start])
    visited.add(start)
    distance[start] = 0

    while q:
        v = q.popleft()
        order.append(v)
        for neighbor in graph.get_neighbors(v):
            if neighbor not in visited:
                visited.add(neighbor)
                distance[neighbor] = distance[v] + 1
                q.append(neighbor)

    return order, distance


def dfs(graph, start):
    """深度优先搜索 —— 返回遍历顺序

    DFS 使用递归，沿路径走到底后回溯。
    时间复杂度 O(V + E)
    """
    visited = set()
    order = []

    def _dfs(v):
        visited.add(v)
        order.append(v)
        for neighbor in graph.get_neighbors(v):
            if neighbor not in visited:
                _dfs(neighbor)

    _dfs(start)
    return order


# ============================================================================
# 第三部分：拓扑排序
# ============================================================================

def topological_sort_kahn(graph):
    """Kahn 算法 —— 基于 BFS 和入度的拓扑排序

    适合有向无环图（DAG）
    时间复杂度 O(V + E)
    """
    # 计算所有顶点的入度
    indegree = {i: 0 for i in range(graph.V)}
    for u in range(graph.V):
        for v in graph.adj_list[u]:
            indegree[v] += 1

    # 入度为 0 的顶点入队
    q = deque([v for v in range(graph.V) if indegree[v] == 0])
    result = []

    while q:
        u = q.popleft()
        result.append(u)
        for v in graph.adj_list[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                q.append(v)

    # 如果结果不包含所有顶点，说明存在环
    if len(result) != graph.V:
        return None  # 图中存在环，无法拓扑排序
    return result


def topological_sort_dfs(graph):
    """DFS 版拓扑排序 —— 按完成时间逆序

    后序遍历的逆序即为拓扑序
    """
    visited = set()
    result = []

    def _dfs(v):
        visited.add(v)
        for neighbor in graph.adj_list[v]:
            if neighbor not in visited:
                _dfs(neighbor)
        result.append(v)  # 后序：所有邻居处理完后才加入

    for v in range(graph.V):
        if v not in visited:
            _dfs(v)

    return result[::-1]  # 逆序 = 拓扑序


# ============================================================================
# 第四部分：环检测
# ============================================================================

def has_cycle_undirected(graph):
    """无向图环检测 —— DFS 检测回边

    在无向图中，如果 DFS 遇到一个已访问的邻居且不是父节点，则存在环。
    """
    visited = set()

    def _dfs(v, parent):
        visited.add(v)
        for neighbor in graph.adj_list[v]:
            if neighbor not in visited:
                if _dfs(neighbor, v):
                    return True
            elif neighbor != parent:
                # 遇到已访问的非父节点 → 环
                return True
        return False

    for v in range(graph.V):
        if v not in visited:
            if _dfs(v, -1):
                return True
    return False


def has_cycle_directed(graph):
    """有向图环检测 —— 三色标记法 (0=白/未访问, 1=灰/探索中, 2=黑/已完成)

    在 DFS 过程中遇到「灰色」节点 → 存在回边 → 有环。
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * graph.V

    def _dfs(v):
        color[v] = GRAY  # 开始探索
        for neighbor in graph.adj_list[v]:
            if color[neighbor] == GRAY:
                return True  # 回边！找到环
            if color[neighbor] == WHITE:
                if _dfs(neighbor):
                    return True
        color[v] = BLACK  # 探索完成
        return False

    for v in range(graph.V):
        if color[v] == WHITE:
            if _dfs(v):
                return True
    return False


# ============================================================================
# 第五部分：二分图检测
# ============================================================================

def is_bipartite(graph):
    """二分图检测 —— BFS 染色法

    将顶点染为 0/1 两种颜色，相邻顶点必须不同色。
    时间复杂度 O(V + E)
    """
    color = [-1] * graph.V  # -1 表示未染色

    for start in range(graph.V):
        if color[start] != -1:
            continue
        color[start] = 0
        q = deque([start])
        while q:
            v = q.popleft()
            for neighbor in graph.get_neighbors(v):
                if color[neighbor] == -1:
                    color[neighbor] = 1 - color[v]  # 染不同颜色
                    q.append(neighbor)
                elif color[neighbor] == color[v]:
                    return False  # 相邻同色 → 不是二分图
    return True


# ============================================================================
# 第六部分：连通分量
# ============================================================================

def connected_components(graph):
    """找出无向图的所有连通分量

    返回列表的列表，每个子列表是一个连通分量的顶点集合
    """
    visited = set()
    components = []

    for v in range(graph.V):
        if v not in visited:
            # BFS 探索这个连通分量
            component = []
            q = deque([v])
            visited.add(v)
            while q:
                u = q.popleft()
                component.append(u)
                for neighbor in graph.get_neighbors(u):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append(neighbor)
            components.append(component)

    return components


# ============================================================================
# 第七部分：可视化
# ============================================================================

def plot_graph_demos():
    """可视化图算法的结果"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 左上：BFS vs DFS 遍历顺序
    ax1 = axes[0, 0]
    g = Graph(6)
    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 4), (2, 5)]
    for u, v in edges:
        g.add_edge(u, v)
    bfs_order, _ = bfs(g, 0)
    dfs_order = dfs(g, 0)
    ax1.text(0.1, 0.8, f'BFS 遍历: {bfs_order}', fontsize=12, fontfamily='monospace')
    ax1.text(0.1, 0.5, f'DFS 遍历: {dfs_order}', fontsize=12, fontfamily='monospace')
    ax1.text(0.1, 0.2, f'连通分量: {connected_components(g)}', fontsize=10, fontfamily='monospace')
    ax1.axis('off')
    ax1.set_title('BFS vs DFS 遍历结果', fontsize=14)

    # 右上：拓扑排序
    ax2 = axes[0, 1]
    dag = Graph(6, directed=True)
    dag_edges = [(5, 2), (5, 0), (4, 0), (4, 1), (2, 3), (3, 1)]
    for u, v in dag_edges:
        dag.add_edge(u, v)
    topo_kahn = topological_sort_kahn(dag)
    topo_dfs = topological_sort_dfs(dag)
    ax2.text(0.1, 0.7, f'Kahn: {topo_kahn}', fontsize=12, fontfamily='monospace')
    ax2.text(0.1, 0.4, f'DFS:  {topo_dfs}', fontsize=12, fontfamily='monospace')
    ax2.text(0.1, 0.1, f'有环? {has_cycle_directed(dag)} (应 False)', fontsize=11)
    ax2.axis('off')
    ax2.set_title('拓扑排序 (DAG)', fontsize=14)

    # 左下：二分图检测
    ax3 = axes[1, 0]
    # 二分图示例：两个集合 {0,2,4} 和 {1,3,5}
    bg = Graph(6)
    bipartite_edges = [(0, 1), (0, 3), (2, 1), (2, 5), (4, 3), (4, 5)]
    for u, v in bipartite_edges:
        bg.add_edge(u, v)
    ax3.text(0.1, 0.7, f'二分图? {is_bipartite(bg)}', fontsize=14, fontfamily='monospace')

    # 非二分图示例：三角形 {0-1, 1-2, 2-0}
    nbg = Graph(3)
    for u, v in [(0, 1), (1, 2), (2, 0)]:
        nbg.add_edge(u, v)
    ax3.text(0.1, 0.4, f'三角形(奇环)? {is_bipartite(nbg)}', fontsize=14, fontfamily='monospace')
    ax3.text(0.1, 0.1, '含奇环的图不是二分图', fontsize=10)
    ax3.axis('off')
    ax3.set_title('二分图检测', fontsize=14)

    # 右下：图表示法对比
    ax4 = axes[1, 1]
    g2 = Graph(4)
    for u, v in [(0, 1), (0, 2), (1, 2), (2, 3)]:
        g2.add_edge(u, v)
    ax4.text(0.05, 0.9, f'邻接表:', fontsize=10, fontweight='bold')
    for i in range(g2.V):
        ax4.text(0.1, 0.8 - i * 0.12, f'  {i}: {g2.adj_list[i]}', fontsize=9, fontfamily='monospace')
    ax4.text(0.05, 0.35, f'边列表: {[(e[0],e[1]) for e in g2.edge_list]}', fontsize=9)
    ax4.text(0.05, 0.25, f'空间: O(V+E)={g2.V}+{len(g2.edge_list)}', fontsize=9)
    ax4.axis('off')
    ax4.set_title('图的三种表示法', fontsize=14)

    plt.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'graph_basics_demo.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[图片保存] {out}')


# ============================================================================
# 主函数
# ============================================================================

def main():
    print('=' * 60)
    print('  图论基础 — 演示程序')
    print('=' * 60)

    # 1. 图的构建与遍历
    print('\n--- 图的构建与遍历 ---')
    g = Graph(6)
    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 4), (2, 5)]
    for u, v in edges:
        g.add_edge(u, v)
    print(f'图: {g}')

    bfs_order, dist = bfs(g, 0)
    print(f'BFS (从 0 开始): {bfs_order}')
    print(f'距 0 的距离: {dist}')

    dfs_order = dfs(g, 0)
    print(f'DFS (从 0 开始): {dfs_order}')

    # 2. 连通分量
    print(f'\n连通分量: {connected_components(g)}')

    # 3. 环检测
    print(f'无向图有环? {has_cycle_undirected(g)}')

    # 4. 拓扑排序
    print('\n--- 拓扑排序 ---')
    dag = Graph(6, directed=True)
    dag_edges = [(5, 2), (5, 0), (4, 0), (4, 1), (2, 3), (3, 1)]
    for u, v in dag_edges:
        dag.add_edge(u, v)
    print(f'Kahn 拓扑序: {topological_sort_kahn(dag)}')
    print(f'DFS 拓扑序:  {topological_sort_dfs(dag)}')
    print(f'有向图有环? {has_cycle_directed(dag)}')

    # 5. 二分图检测
    print('\n--- 二分图检测 ---')
    bg = Graph(6)
    for u, v in [(0, 1), (0, 3), (2, 1), (2, 5), (4, 3), (4, 5)]:
        bg.add_edge(u, v)
    print(f'二分图? {is_bipartite(bg)}')

    nbg = Graph(3)
    for u, v in [(0, 1), (1, 2), (2, 0)]:
        nbg.add_edge(u, v)
    print(f'三角形(奇环) 二分图? {is_bipartite(nbg)}')

    # 6. 可视化
    plot_graph_demos()

    print('\n✅ 所有演示已完成！图片保存在 images/ 目录中。')


if __name__ == '__main__':
    main()
