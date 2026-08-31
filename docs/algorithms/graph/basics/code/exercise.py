# -*- coding: utf-8 -*-
"""
===============================================================================
algo06_graph_basics/code/exercise.py — 图论基础练习
===============================================================================
练习目标：
  1. 实现邻接表的 BFS 求最短路径距离
  2. 实现 DFS 检测无向图的环
  3. 实现 Kahn 拓扑排序算法
  4. 实现二分图检测（BFS 染色）

运行方式：python exercise.py
===============================================================================
"""

from collections import deque


# ============================================================================
# 任务 1：BFS 最短路径
# ============================================================================

def bfs_shortest_path(graph, start, target):
    """使用 BFS 求从 start 到 target 的最短路径距离

    graph: 邻接表格式的 dict {node: [neighbors]}
    返回: 最短距离 (int) 或 -1（不连通）
    """
    # TODO: 你的代码
    if start == target:
        return 0

    visited = {start}
    q = deque([(start, 0)])  # (node, distance)

    while q:
        node, dist = q.popleft()
        for neighbor in graph.get(node, []):
            if neighbor == target:
                return dist + 1
            if neighbor not in visited:
                visited.add(neighbor)
                q.append((neighbor, dist + 1))
    return -1


# ============================================================================
# 任务 2：DFS 环检测
# ============================================================================

def has_cycle_undirected(graph):
    """使用 DFS 检测无向图中是否存在环

    graph: 邻接表格式 {node: [neighbors]}
    返回: True/False
    """
    # TODO: 你的代码
    visited = set()

    def dfs(node, parent):
        visited.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                if dfs(neighbor, node):
                    return True
            elif neighbor != parent:
                return True  # 回边 = 环
        return False

    for node in graph:
        if node not in visited:
            if dfs(node, -1):
                return True
    return False


# ============================================================================
# 任务 3：Kahn 拓扑排序
# ============================================================================

def topological_sort(n, edges):
    """使用 Kahn 算法对有向无环图进行拓扑排序

    n: 顶点数量 (0 到 n-1)
    edges: [(u, v), ...] 有向边列表，表示 u → v
    返回: 拓扑排序列表，若存在环则返回 None
    """
    # 构建邻接表和入度数组
    adj = {i: [] for i in range(n)}
    indegree = [0] * n

    # TODO: 你的代码 —— 构建邻接表和计算入度
    for u, v in edges:
        adj[u].append(v)
        indegree[v] += 1

    # 入度为 0 的顶点入队
    q = deque([i for i in range(n) if indegree[i] == 0])
    result = []

    # TODO: 你的代码 —— BFS 拓扑排序
    while q:
        u = q.popleft()
        result.append(u)
        for v in adj[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                q.append(v)

    return result if len(result) == n else None


# ============================================================================
# 任务 4：二分图检测（BFS 染色）
# ============================================================================

def is_bipartite(graph):
    """判断无向图是否为二分图

    graph: 邻接表格式 {node: [neighbors]}
    返回: True/False
    """
    # TODO: 你的代码
    color = {}  # node → 0 or 1

    for start in graph:
        if start in color:
            continue
        color[start] = 0
        q = deque([start])
        while q:
            node = q.popleft()
            for neighbor in graph.get(node, []):
                if neighbor not in color:
                    color[neighbor] = 1 - color[node]
                    q.append(neighbor)
                elif color[neighbor] == color[node]:
                    return False
    return True


# ============================================================================
# 主函数
# ============================================================================

def main():
    print('=' * 60)
    print('  图论基础 — 练习程序')
    print('=' * 60)

    # 测试任务 1
    print('\n--- 任务 1：BFS 最短路径 ---')
    g = {0: [1, 2], 1: [0, 3, 4], 2: [0, 4, 5], 3: [1], 4: [1, 2], 5: [2]}
    print(f'0→5 最短距离: {bfs_shortest_path(g, 0, 5)} (期望 2)')
    print(f'0→3 最短距离: {bfs_shortest_path(g, 0, 3)} (期望 2)')
    assert bfs_shortest_path(g, 0, 5) == 2
    assert bfs_shortest_path(g, 0, 3) == 2
    print('✅ 任务 1 通过！')

    # 测试任务 2
    print('\n--- 任务 2：环检测 ---')
    g1 = {0: [1, 2], 1: [0], 2: [0]}  # 无环（树）
    g2 = {0: [1, 2], 1: [0, 2], 2: [0, 1]}  # 三角形（有环）
    print(f'树有环? {has_cycle_undirected(g1)} (期望 False)')
    print(f'三角形有环? {has_cycle_undirected(g2)} (期望 True)')
    assert not has_cycle_undirected(g1)
    assert has_cycle_undirected(g2)
    print('✅ 任务 2 通过！')

    # 测试任务 3
    print('\n--- 任务 3：拓扑排序 ---')
    edges = [(5, 2), (5, 0), (4, 0), (4, 1), (2, 3), (3, 1)]
    result = topological_sort(6, edges)
    print(f'拓扑序: {result}')
    # 验证：对每条边 u→v，u 在 result 中的位置必须在 v 之前
    pos = {v: i for i, v in enumerate(result)} if result else {}
    for u, v in edges:
        assert pos[u] < pos[v], f'边 {u}→{v} 不满足拓扑序'
    print('✅ 任务 3 通过！')

    # 测试任务 4
    print('\n--- 任务 4：二分图检测 ---')
    bg = {0: [1, 3], 1: [0, 2], 2: [1, 5], 3: [0, 4], 4: [3, 5], 5: [2, 4]}
    tri = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    print(f'二分图? {is_bipartite(bg)} (期望 True)')
    print(f'三角形? {is_bipartite(tri)} (期望 False)')
    assert is_bipartite(bg) and not is_bipartite(tri)
    print('✅ 任务 4 通过！')

    print('\n🎉 所有练习已完成！')


if __name__ == '__main__':
    main()
