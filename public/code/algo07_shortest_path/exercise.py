# -*- coding: utf-8 -*-
"""
===============================================================================
algo07_shortest_path/code/exercise.py — 最短路径算法练习
===============================================================================
练习目标：
  1. 实现 Dijkstra 算法（朴素版，用数组扫描找最小值）
  2. 实现 Bellman-Ford 的松弛操作和负环检测
  3. 实现 Floyd-Warshall 算法
  4. 实现路径重建

运行方式：python exercise.py
===============================================================================
"""

INF = float('inf')
import heapq


# ============================================================================
# 任务 1：朴素版 Dijkstra（O(V²)）
# ============================================================================

def dijkstra_naive(adj_matrix, start):
    """朴素 Dijkstra —— O(V²)，适用于稠密图

    adj_matrix: n×n 邻接矩阵，adj_matrix[i][j] = 边权重（INF 表示无边）
    start: 起点
    返回: (dist, prev)
    """
    n = len(adj_matrix)
    dist = [INF] * n
    dist[start] = 0
    visited = [False] * n
    prev = [None] * n

    # TODO: 你的代码
    for _ in range(n):
        # 找到未访问节点中 dist 最小的
        u = -1
        min_dist = INF
        for i in range(n):
            if not visited[i] and dist[i] < min_dist:
                min_dist = dist[i]
                u = i

        if u == -1:
            break

        visited[u] = True
        # 松弛 u 的所有邻居
        for v in range(n):
            if adj_matrix[u][v] != INF and not visited[v]:
                new_dist = dist[u] + adj_matrix[u][v]
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    prev[v] = u

    return dist, prev


# ============================================================================
# 任务 2：Bellman-Ford 负环检测
# ============================================================================

def bellman_ford_detect_negative_cycle(n, edges):
    """使用 Bellman-Ford 检测图中是否存在负环

    n: 顶点数
    edges: [(u, v, weight), ...]
    返回: True（存在负环）/ False
    """
    # 需要从所有顶点出发，因此添加一个虚拟源点
    # 或者对每个未到达的连通分量单独检测
    # TODO: 你的代码
    dist = [0] * n  # 初始化为 0，相当于从虚拟源点出发（边权为 0）

    for i in range(n):
        updated = False
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                updated = True
        if not updated:
            break
        if i == n - 1 and updated:
            return True  # 第 n 轮仍能松弛 → 负环

    return False


# ============================================================================
# 任务 3：Floyd-Warshall 实现
# ============================================================================

def floyd_warshall(n, edges):
    """Floyd-Warshall 全源最短路径

    n: 顶点数量
    edges: [(u, v, w), ...]
    返回: n×n 距离矩阵
    """
    # 初始化
    dist = [[INF] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
    for u, v, w in edges:
        dist[u][v] = min(dist[u][v], w)

    # TODO: 你的代码 —— 三重循环：k 在最外层！
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
# 任务 4：路径重建
# ============================================================================

def reconstruct_shortest_path(prev, target):
    """根据 prev 数组重建从起点到 target 的最短路径

    prev[i] = 在最短路径中 i 的前驱节点，prev[start] = None
    返回: [start, ..., target] 路径列表
    """
    # TODO: 你的代码
    path = []
    cur = target
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return path[::-1]


# ============================================================================
# 任务 5（Bonus）：实现 Dijkstra 带路径重建
# ============================================================================

def dijkstra_with_path(adj, start, target):
    """Dijkstra + 路径重建（堆优化版）

    返回: (path, distance) 或 (None, INF)
    """
    # TODO: 你的代码
    dist = {u: INF for u in adj}
    dist[start] = 0
    prev = {u: None for u in adj}
    pq = [(0, start)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if u == target:
            path = reconstruct_shortest_path(prev, target)
            return path, dist[target]
        for v, w in adj.get(u, []):
            new_dist = dist[u] + w
            if new_dist < dist[v]:
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(pq, (new_dist, v))

    return None, INF


# ============================================================================
# 主函数
# ============================================================================

def main():
    print('=' * 60)
    print('  最短路径算法 — 练习程序')
    print('=' * 60)

    # 测试任务 1
    print('\n--- 任务 1：朴素 Dijkstra ---')
    adj_matrix = [
        [0, 2, 4, INF, INF],
        [2, 0, 1, 7, INF],
        [4, 1, 0, 3, 5],
        [INF, 7, 3, 0, 1],
        [INF, INF, 5, 1, 0],
    ]
    dist, prev = dijkstra_naive(adj_matrix, 0)
    print(f'从 0 出发距离: {dist}')
    assert dist[4] == 6, f'期望 dist[4]=6, 实际={dist[4]}'
    print('✅ 任务 1 通过！')

    # 测试任务 2
    print('\n--- 任务 2：负环检测 ---')
    edges_no_cycle = [(0, 1, 1), (1, 2, -2), (2, 0, 0.5)]
    edges_cycle = [(0, 1, 1), (1, 2, -2), (2, 0, -0.5)]  # 0→1→2→0 = -1.5
    print(f'无负环图: {bellman_ford_detect_negative_cycle(3, edges_no_cycle)} (期望 False)')
    print(f'有负环图: {bellman_ford_detect_negative_cycle(3, edges_cycle)} (期望 True)')
    assert not bellman_ford_detect_negative_cycle(3, edges_no_cycle)
    assert bellman_ford_detect_negative_cycle(3, edges_cycle)
    print('✅ 任务 2 通过！')

    # 测试任务 3
    print('\n--- 任务 3：Floyd-Warshall ---')
    edges = [(0, 1, 2), (0, 2, 4), (1, 2, 1), (1, 3, 7), (2, 3, 3), (2, 4, 5), (3, 4, 1)]
    fw = floyd_warshall(5, edges)
    print(f'0→4 最短距离: {fw[0][4]} (期望 6)')
    assert fw[0][4] == 6
    print('✅ 任务 3 通过！')

    # 测试任务 4
    print('\n--- 任务 4：路径重建 ---')
    prev = {0: None, 1: 0, 2: 1, 3: 2, 4: 2}
    path = reconstruct_shortest_path(prev, 4)
    print(f'从 0 到 4 的路径: {path} (期望 [0, 1, 2, 4])')
    assert path == [0, 1, 2, 4]
    print('✅ 任务 4 通过！')

    # 测试任务 5
    print('\n--- 任务 5：Dijkstra + 路径 ---')
    adj = {
        0: [(1, 2), (2, 4)],
        1: [(0, 2), (2, 1), (3, 7)],
        2: [(0, 4), (1, 1), (3, 3), (4, 5)],
        3: [(1, 7), (2, 3), (4, 1)],
        4: [(2, 5), (3, 1)]
    }
    path, cost = dijkstra_with_path(adj, 0, 4)
    print(f'路径: {"→".join(map(str, path))}, 距离={cost}')
    assert cost == 6
    print('✅ 任务 5 通过！')

    print('\n🎉 所有练习已完成！')


if __name__ == '__main__':
    main()
