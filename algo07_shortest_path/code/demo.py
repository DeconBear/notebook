# -*- coding: utf-8 -*-
"""
===============================================================================
algo07_shortest_path/code/demo.py — 最短路径算法从零实现
===============================================================================
本演示从零实现五种最短路径算法：
  1. Dijkstra（堆优化）—— 非负权图单源最短路径
  2. Bellman-Ford —— 支持负权边，检测负环
  3. SPFA —— 队列优化的 Bellman-Ford
  4. Floyd-Warshall —— 全源最短路径（DP）
  5. A* 搜索 —— 启发式搜索

运行方式：python demo.py
依赖：matplotlib（仅用于可视化）
===============================================================================
"""

import os
import math
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
# 第一部分：Dijkstra 算法（堆优化）
# ============================================================================

def dijkstra(adj, start):
    """Dijkstra 最短路径 —— 堆优化版 O((V+E) log V)

    参数:
      adj: 邻接表，格式 {u: [(v, weight), ...]}
      start: 起点
    返回:
      (dist, prev) — dist 是距离字典，prev 记录路径上的前驱
    """
    n = len(adj)
    dist = {i: INF for i in range(n)}
    dist[start] = 0
    prev = {i: None for i in range(n)}
    pq = [(0, start)]  # (距离, 顶点)

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:  # 过时的条目，跳过
            continue
        for v, w in adj.get(u, []):
            new_dist = dist[u] + w
            if new_dist < dist[v]:
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(pq, (new_dist, v))

    return dist, prev


def reconstruct_path(prev, target):
    """根据 prev 数组重建从起点到 target 的路径"""
    path = []
    cur = target
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return path[::-1]


# ============================================================================
# 第二部分：Bellman-Ford 算法
# ============================================================================

def bellman_ford(n, edges, start):
    """Bellman-Ford 算法 —— O(VE)

    参数:
      n: 顶点数量
      edges: [(u, v, weight), ...] 边列表
      start: 起点
    返回:
      (dist, prev, has_negative_cycle)
    """
    dist = [INF] * n
    dist[start] = 0
    prev = [None] * n

    # 第 1 到 V-1 轮松弛
    for _ in range(n - 1):
        updated = False
        for u, v, w in edges:
            if dist[u] != INF and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u
                updated = True
        if not updated:
            break  # 提前收敛

    # 第 V 轮：检测负环
    has_negative_cycle = False
    for u, v, w in edges:
        if dist[u] != INF and dist[u] + w < dist[v]:
            has_negative_cycle = True
            break

    return dist, prev, has_negative_cycle


# ============================================================================
# 第三部分：SPFA（队列优化的 Bellman-Ford）
# ============================================================================

def spfa(adj, start):
    """SPFA 算法 —— 队列优化的 Bellman-Ford，平均 O(E)

    返回: (dist, prev, has_negative_cycle)
    """
    n = len(adj)
    dist = [INF] * n
    dist[start] = 0
    prev = [None] * n
    in_queue = [False] * n
    count = [0] * n  # 记录每个顶点入队次数（检测负环）

    q = deque([start])
    in_queue[start] = True
    count[start] = 1

    while q:
        u = q.popleft()
        in_queue[u] = False
        for v, w in adj.get(u, []):
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u
                if not in_queue[v]:
                    q.append(v)
                    in_queue[v] = True
                    count[v] += 1
                    if count[v] >= n:
                        return dist, prev, True  # 负环

    return dist, prev, False


# ============================================================================
# 第四部分：Floyd-Warshall 算法
# ============================================================================

def floyd_warshall(n, edges):
    """Floyd-Warshall 全源最短路径 —— O(V³)

    参数:
      n: 顶点数量
      edges: [(u, v, weight), ...]
    返回:
      dist: n×n 矩阵，dist[i][j] 是 i 到 j 的最短距离
    """
    # 初始化距离矩阵
    dist = [[INF] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
    for u, v, w in edges:
        dist[u][v] = min(dist[u][v], w)  # 处理平行边

    # DP：k 必须在最外层！
    for k in range(n):
        for i in range(n):
            if dist[i][k] == INF:
                continue
            for j in range(n):
                if dist[k][j] == INF:
                    continue
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    return dist


# ============================================================================
# 第五部分：A* 搜索（简化版）
# ============================================================================

def a_star(adj, coords, start, goal):
    """A* 搜索 —— 启发式最短路径

    参数:
      adj: 邻接表 {u: [(v, w), ...]}
      coords: {vertex: (x, y)} 顶点的坐标（用于启发式函数）
      start, goal: 起点和终点
    返回:
      (path, cost) 或 (None, INF)
    """
    def heuristic(u):
        """欧几里得距离启发式函数"""
        if u not in coords or goal not in coords:
            return 0
        ux, uy = coords[u]
        gx, gy = coords[goal]
        return math.sqrt((ux - gx)**2 + (uy - gy)**2)

    # f_score = g_score + h_score
    g_score = {u: INF for u in adj}
    g_score[start] = 0
    f_score = {u: INF for u in adj}
    f_score[start] = heuristic(start)

    prev = {u: None for u in adj}
    open_set = [(f_score[start], start)]  # (f_score, vertex)

    while open_set:
        _, u = heapq.heappop(open_set)
        if u == goal:
            # 重建路径
            path = []
            cur = goal
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            return path[::-1], g_score[goal]

        for v, w in adj.get(u, []):
            tentative_g = g_score[u] + w
            if tentative_g < g_score[v]:
                g_score[v] = tentative_g
                f_score[v] = tentative_g + heuristic(v)
                prev[v] = u
                heapq.heappush(open_set, (f_score[v], v))

    return None, INF


# ============================================================================
# 第六部分：可视化
# ============================================================================

def plot_shortest_path_demos():
    """可视化各种最短路径算法的结果"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 构建测试图
    adj = {
        0: [(1, 2), (2, 4)],
        1: [(0, 2), (2, 1), (3, 7)],
        2: [(0, 4), (1, 1), (3, 3), (4, 5)],
        3: [(1, 7), (2, 3), (4, 1)],
        4: [(2, 5), (3, 1)]
    }

    # 左上：Dijkstra
    ax1 = axes[0, 0]
    dist, prev = dijkstra(adj, 0)
    for v in range(5):
        path = reconstruct_path(prev, v)
        ax1.text(0.1, 0.9 - v * 0.16,
                f'0→{v}: {"→".join(map(str, path))} (dist={dist[v]})',
                fontsize=9, fontfamily='monospace')
    ax1.axis('off')
    ax1.set_title('Dijkstra 最短路径 (从 0 出发)', fontsize=14)

    # 右上：Bellman-Ford 与 SPFA 对比
    ax2 = axes[0, 1]
    n = 5
    edges = [(0, 1, 2), (0, 2, 4), (1, 2, 1), (1, 3, 7),
             (2, 3, 3), (2, 4, 5), (3, 4, 1)]
    bf_dist, bf_prev, bf_neg = bellman_ford(n, edges, 0)
    spfa_dist, spfa_prev, spfa_neg = spfa(adj, 0)

    ax2.text(0.1, 0.8, f'Bellman-Ford 距离: {bf_dist}', fontsize=9, fontfamily='monospace')
    ax2.text(0.1, 0.6, f'SPFA 距离: {spfa_dist}', fontsize=9, fontfamily='monospace')
    ax2.text(0.1, 0.4, f'B-F 负环? {bf_neg}, SPFA 负环? {spfa_neg}', fontsize=9)
    ax2.text(0.1, 0.2, f'B-F O(VE)={n}*{len(edges)}, SPFA 平均 O(E)', fontsize=9)
    ax2.axis('off')
    ax2.set_title('Bellman-Ford vs SPFA', fontsize=14)

    # 左下：Floyd-Warshall 距离矩阵
    ax3 = axes[1, 0]
    fw_dist = floyd_warshall(n, edges)
    ax3.text(0.05, 0.95, '全源最短距离矩阵:', fontsize=10, fontweight='bold')
    for i in range(n):
        row_str = '  '.join(f'{fw_dist[i][j]:>3}' if fw_dist[i][j] != INF else 'INF'
                            for j in range(n))
        ax3.text(0.1, 0.8 - i * 0.12, f'  [{row_str}]', fontsize=9, fontfamily='monospace')
    ax3.text(0.05, 0.2, f'O(V³) = O({n}³) = {n**3} 次操作', fontsize=9)
    ax3.axis('off')
    ax3.set_title('Floyd-Warshall 全源最短路径', fontsize=14)

    # 右下：A* 搜索
    ax4 = axes[1, 1]
    # 简化二维网格
    coords = {0: (0, 0), 1: (2, 0), 2: (0, 1), 3: (2, 1), 4: (3, 2)}
    a_path, a_cost = a_star(adj, coords, 0, 4)
    if a_path:
        ax4.text(0.1, 0.7, f'A* 路径: {"→".join(map(str, a_path))}', fontsize=10, fontfamily='monospace')
        ax4.text(0.1, 0.5, f'A* 代价: {a_cost}', fontsize=10, fontfamily='monospace')
    # Dijkstra 到 4
    d_path = reconstruct_path(prev, 4)
    ax4.text(0.1, 0.3, f'Dijkstra 路径: {"→".join(map(str, d_path))}', fontsize=10, fontfamily='monospace')
    ax4.text(0.1, 0.1, 'A* 使用启发式减少探索节点', fontsize=9)
    ax4.axis('off')
    ax4.set_title('A* 搜索 (vs Dijkstra)', fontsize=14)

    plt.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'shortest_path_demo.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[图片保存] {out}')


# ============================================================================
# 主函数
# ============================================================================

def main():
    print('=' * 60)
    print('  最短路径算法 — 演示程序')
    print('=' * 60)

    # 构建测试图
    adj = {
        0: [(1, 2), (2, 4)],
        1: [(0, 2), (2, 1), (3, 7)],
        2: [(0, 4), (1, 1), (3, 3), (4, 5)],
        3: [(1, 7), (2, 3), (4, 1)],
        4: [(2, 5), (3, 1)]
    }
    edges = [(0, 1, 2), (0, 2, 4), (1, 2, 1), (1, 3, 7),
             (2, 3, 3), (2, 4, 5), (3, 4, 1)]

    # 1. Dijkstra
    print('\n--- Dijkstra ---')
    dist, prev = dijkstra(adj, 0)
    print(f'从 0 出发的最短距离: {dist}')
    for v in range(5):
        path = reconstruct_path(prev, v)
        print(f'  0→{v}: {"→".join(map(str, path))}, 距离={dist[v]}')

    # 2. Bellman-Ford
    print('\n--- Bellman-Ford ---')
    bf_dist, bf_prev, bf_neg = bellman_ford(5, edges, 0)
    print(f'距离: {bf_dist}')
    print(f'负环? {bf_neg}')

    # 3. SPFA
    print('\n--- SPFA ---')
    spfa_dist, spfa_prev, spfa_neg = spfa(adj, 0)
    print(f'距离: {spfa_dist}')
    print(f'负环? {spfa_neg}')

    # 4. Floyd-Warshall
    print('\n--- Floyd-Warshall ---')
    fw_dist = floyd_warshall(5, edges)
    print('全源最短距离矩阵:')
    for i in range(5):
        row = [fw_dist[i][j] if fw_dist[i][j] != INF else '∞' for j in range(5)]
        print(f'  {row}')

    # 5. A*
    print('\n--- A* ---')
    coords = {0: (0, 0), 1: (2, 0), 2: (0, 1), 3: (2, 1), 4: (3, 2)}
    path, cost = a_star(adj, coords, 0, 4)
    print(f'路径: {"→".join(map(str, path))}, 代价={cost}')

    # 6. 可视化
    plot_shortest_path_demos()

    print('\n✅ 所有演示已完成！图片保存在 images/ 目录中。')


if __name__ == '__main__':
    main()
