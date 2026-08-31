# -*- coding: utf-8 -*-
"""
===============================================================================
algo08_mst_networkflow/code/exercise.py — 最小生成树与网络流练习
===============================================================================
练习目标：
  1. 实现 Prim 算法的核心循环（不用堆，朴素版）
  2. 实现 Kruskal 算法的并查集判断部分
  3. 实现 Ford-Fulkerson 方法（DFS 找增广路径）
  4. 实现 Dinic 的 BFS 层图构建

运行方式：python exercise.py
===============================================================================
"""

INF = float('inf')
from collections import deque


# ============================================================================
# 任务 1：朴素 Prim 算法
# ============================================================================

def prim_naive(adj_matrix, n):
    """朴素 Prim MST —— O(V²)

    adj_matrix: n×n 邻接矩阵
    返回: (mst_weight, parent)
    """
    # key[v] = 连接已选集合到 v 的最小边权重
    key = [INF] * n
    key[0] = 0
    mst_set = [False] * n
    parent = [-1] * n
    mst_weight = 0

    # TODO: 你的代码 —— O(V²) 循环
    for _ in range(n):
        # 找到 key 最小且不在 MST 中的顶点
        u = -1
        min_key = INF
        for i in range(n):
            if not mst_set[i] and key[i] < min_key:
                min_key = key[i]
                u = i

        if u == -1:
            break

        mst_set[u] = True
        mst_weight += key[u] if parent[u] != -1 else 0

        # 更新 u 的邻居的 key 值
        for v in range(n):
            if adj_matrix[u][v] != INF and not mst_set[v] and adj_matrix[u][v] < key[v]:
                key[v] = adj_matrix[u][v]
                parent[v] = u

    return mst_weight, parent


# ============================================================================
# 任务 2：Kruskal 核心 —— 并查集判断环
# ============================================================================

def kruskal_mst(n, edges):
    """Kruskal 算法

    edges: [(w, u, v), ...]
    返回: (mst_weight, mst_edges)
    """
    # 并查集
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        # TODO: 实现带路径压缩的 find
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        # TODO: 实现按秩合并
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        if rank[rx] < rank[ry]:
            parent[rx] = ry
        elif rank[rx] > rank[ry]:
            parent[ry] = rx
        else:
            parent[ry] = rx
            rank[rx] += 1
        return True

    # 排序边
    edges.sort()
    mst_weight = 0
    mst_edges = []

    # TODO: 遍历排序后的边，用并查集判断是否加入
    for w, u, v in edges:
        if union(u, v):
            mst_weight += w
            mst_edges.append((u, v, w))
            if len(mst_edges) == n - 1:
                break

    return mst_weight, mst_edges


# ============================================================================
# 任务 3：Ford-Fulkerson（DFS 找增广路径）
# ============================================================================

def ford_fulkerson(graph, s, t):
    """Ford-Fulkerson 最大流 —— 用 DFS 找增广路径

    graph: 邻接矩阵形式的残量网络 graph[u][v] = 残量容量
    返回: 最大流值
    """
    n = len(graph)
    # 复制图以避免修改原图
    residual = [row[:] for row in graph]

    def dfs(u, flow, visited):
        """在残量网络中 DFS 找 s→t 增广路径"""
        if u == t:
            return flow
        visited[u] = True
        for v in range(n):
            if not visited[v] and residual[u][v] > 0:
                # TODO: 递归搜索，取瓶颈容量
                bottleneck = dfs(v, min(flow, residual[u][v]), visited)
                if bottleneck > 0:
                    # TODO: 更新残量网络
                    residual[u][v] -= bottleneck
                    residual[v][u] += bottleneck
                    return bottleneck
        return 0

    max_flow = 0
    while True:
        visited = [False] * n
        flow = dfs(s, INF, visited)
        if flow == 0:
            break
        max_flow += flow

    return max_flow


# ============================================================================
# 任务 4（Bonus）：Dinic BFS 层图
# ============================================================================

def dinic_bfs_level_graph(adj, s, t, n):
    """BFS 构建层图 —— Dinic 算法的第一步

    adj: 邻接表 adj[u] = [(v, cap, rev_idx), ...]
    返回: level 数组（-1 表示不可达），以及是否还存在增广路径
    """
    # TODO: 你的代码 —— BFS 从 s 出发
    level = [-1] * n
    q = deque([s])
    level[s] = 0

    while q:
        u = q.popleft()
        for v, cap, rev in adj[u]:
            if cap > 0 and level[v] < 0:
                level[v] = level[u] + 1
                q.append(v)

    return level, level[t] >= 0


# ============================================================================
# 主函数
# ============================================================================

def main():
    print('=' * 60)
    print('  最小生成树与网络流 — 练习程序')
    print('=' * 60)

    # 测试任务 1
    print('\n--- 任务 1：朴素 Prim ---')
    adj_matrix = [
        [0, 2, INF, 4],
        [2, 0, 3, 5],
        [INF, 3, 0, 1],
        [4, 5, 1, 0],
    ]
    w, parent = prim_naive(adj_matrix, 4)
    print(f'MST 权重: {w} (期望 6)')
    print(f'parent 数组: {parent}')
    assert w == 6, f'期望 6, 实际 {w}'
    print('✅ 任务 1 通过！')

    # 测试任务 2
    print('\n--- 任务 2：Kruskal ---')
    edges = [(2, 0, 1), (4, 0, 3), (3, 1, 2), (5, 1, 3), (1, 2, 3)]
    w, mst = kruskal_mst(4, edges)
    print(f'MST 权重: {w}, 边: {mst}')
    assert w == 6
    print('✅ 任务 2 通过！')

    # 测试任务 3
    print('\n--- 任务 3：Ford-Fulkerson ---')
    # 构建残量网络（邻接矩阵）
    n = 4
    graph = [[0]*n for _ in range(n)]
    graph[0][1] = 10; graph[0][2] = 5
    graph[1][2] = 15; graph[1][3] = 10
    graph[2][3] = 10
    flow = ford_fulkerson(graph, 0, 3)
    print(f'最大流: {flow} (期望 15)')
    assert flow == 15
    print('✅ 任务 3 通过！')

    # 测试任务 4
    print('\n--- 任务 4：Dinic BFS 层图 ---')
    # 用 Dinic 类构建图（类似 demo.py）
    adj = [[] for _ in range(4)]
    def add_edge(u, v, cap):
        adj[u].append([v, cap, len(adj[v])])
        adj[v].append([u, 0, len(adj[u]) - 1])
    add_edge(0, 1, 10); add_edge(0, 2, 5)
    add_edge(1, 2, 15); add_edge(1, 3, 10)
    add_edge(2, 3, 10)

    level, has_path = dinic_bfs_level_graph(adj, 0, 3, 4)
    print(f'层图: {level} (期望: 0 在 level 0, 1/2 在 level 1, 3 在 level 2)')
    print(f'存在路径? {has_path}')
    assert level[3] == 2 and has_path
    print('✅ 任务 4 通过！')

    print('\n🎉 所有练习已完成！')


if __name__ == '__main__':
    main()
