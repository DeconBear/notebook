# -*- coding: utf-8 -*-
"""
algo12 动态规划（下）— 演示代码
===============================
功能：区间 DP（矩阵链乘法、石子合并）、树形 DP（最大独立集、直径）、
      状态压缩 DP（TSP Held-Karp）、数位 DP（数字和统计）。

每个函数都有中文 docstring，每行逻辑代码都有中文注释。
运行方式：在 algo12_dp_2/ 目录下执行 python code/demo.py
"""

import os
import sys
import math
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False

_HERE = os.path.dirname(os.path.abspath(__file__))
_IMAGES = os.path.join(_HERE, '..', 'images')
os.makedirs(_IMAGES, exist_ok=True)


# ============================================================================
# 第一部分：区间 DP — 矩阵链乘法
# ============================================================================

def matrix_chain_order(p):
    """
    矩阵链乘法：求最小标量乘法次数。
    p = [p0, p1, ..., pn]，矩阵 A_i 的维度为 p_{i-1} x p_i。

    dp[i][j] = 计算 A_i A_{i+1} ... A_j 的最少乘法次数
    dp[i][j] = min_{i<=k<j} (dp[i][k] + dp[k+1][j] + p_{i-1} * p_k * p_j)
    """
    n = len(p) - 1  # 矩阵数量
    INF = float('inf')
    dp = [[0 if i == j else INF for j in range(n)]
          for i in range(n)]
    # split[i][j] 记录最优分割点，用于回溯括号化方案
    split = [[0] * n for _ in range(n)]

    # 按链长递增枚举
    for length in range(2, n + 1):          # 链长从 2 到 n
        for i in range(n - length + 1):      # 左端点
            j = i + length - 1                # 右端点
            for k in range(i, j):             # 枚举分割点
                # 合并代价：左边 + 右边 + 本次合并
                cost = (dp[i][k] + dp[k + 1][j] +
                        p[i] * p[k + 1] * p[j + 1])
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    split[i][j] = k

    return dp[0][n - 1], split


def print_mcm_parenthesization(split, i, j):
    """递归打印矩阵链乘法的括号化方案。"""
    if i == j:
        return f'A{i+1}'
    k = split[i][j]
    left = print_mcm_parenthesization(split, i, k)
    right = print_mcm_parenthesization(split, k + 1, j)
    return f'({left} x {right})'


# ============================================================================
# 第二部分：区间 DP — 石子合并
# ============================================================================

def stone_merge(stones):
    """
    石子合并问题：n 堆石子排成一行，每次合并相邻两堆，代价为两堆之和。
    求将所有石子合并为一堆的最小总代价。

    dp[i][j] = 合并第 i 到第 j 堆的最小代价
    dp[i][j] = min_{i<=k<j} (dp[i][k] + dp[k+1][j]) + sum(stones[i:j+1])
    """
    n = len(stones)
    if n == 1:
        return 0

    # 前缀和，用于 O(1) 计算区间总和
    prefix = [0] * (n + 1)
    for i in range(1, n + 1):
        prefix[i] = prefix[i - 1] + stones[i - 1]

    def range_sum(i, j):
        """返回 stones[i:j+1] 的总和（i, j 是 0-based 索引）"""
        return prefix[j + 1] - prefix[i]

    INF = float('inf')
    dp = [[0] * n for _ in range(n)]

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = INF
            for k in range(i, j):
                cost = dp[i][k] + dp[k + 1][j] + range_sum(i, j)
                if cost < dp[i][j]:
                    dp[i][j] = cost

    return dp[0][n - 1], dp


# ============================================================================
# 第三部分：树形 DP — 最大独立集 + 树直径
# ============================================================================

def tree_max_independent_set(n, edges):
    """
    树的最大独立集：在树中选最多的节点，使任意两个不相邻。
    dp[u][0] = 不选 u 时子树能选的最大节点数
    dp[u][1] = 选 u 时子树能选的最大节点数

    参数:
        n: 节点数量（0 ~ n-1）
        edges: list of (u, v)，无向边
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    dp = [[0, 0] for _ in range(n)]  # dp[u][0], dp[u][1]
    visited = [False] * n

    def dfs(u):
        visited[u] = True
        dp[u][0] = 0  # 不选 u
        dp[u][1] = 1  # 选 u（至少是 1，表示选了 u 自身）
        for v in adj[u]:
            if not visited[v]:
                dfs(v)
                # 选了 u → 子节点 v 不能选
                dp[u][1] += dp[v][0]
                # 不选 u → 子节点 v 可选可不选
                dp[u][0] += max(dp[v][0], dp[v][1])

    dfs(0)  # 以节点 0 为根进行 DFS
    return max(dp[0][0], dp[0][1]), dp


def tree_diameter(n, edges):
    """
    树的直径（树形 DP 解法）。
    使用两次 DFS：从任意点找最远点 A，从 A 找最远点 B。
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # 第一次 DFS：从节点 0 出发，找最远节点
    def dfs_farthest(start):
        dist = [-1] * n
        dist[start] = 0
        stack = [start]
        farthest_node = start
        max_dist = 0
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    stack.append(v)
                    if dist[v] > max_dist:
                        max_dist = dist[v]
                        farthest_node = v
        return farthest_node, max_dist

    A, _ = dfs_farthest(0)
    B, diameter = dfs_farthest(A)
    return diameter, (A, B)


# ============================================================================
# 第四部分：状态压缩 DP — TSP (Held-Karp)
# ============================================================================

def tsp_held_karp(dist):
    """
    旅行商问题 (TSP) — Held-Karp 算法 O(n^2 * 2^n)。
    dp[mask][i] = 当前已访问城市集合 mask，最后在城市 i 的最短距离。

    参数:
        dist: n x n 距离矩阵
    返回:
        最短回路长度，以及访问顺序
    """
    n = len(dist)
    INF = float('inf')
    total_masks = 1 << n
    dp = [[INF] * n for _ in range(total_masks)]
    # parent[mask][i] 用于回溯路径
    parent = [[-1] * n for _ in range(total_masks)]

    # 初始化：从城市 0 出发
    dp[1][0] = 0  # mask=1 表示只访问了城市 0

    for mask in range(1, total_masks):
        # 检查城市 0 是否在 mask 中（路径必须从 0 开始）
        if not (mask & 1):
            continue
        for i in range(n):
            if not (mask & (1 << i)):  # i 不在 mask 中 → 无法作为"最后城市"
                continue
            if dp[mask][i] == INF:
                continue
            # 尝试从 i 走到下一个未访问的城市 j
            for j in range(n):
                if mask & (1 << j):  # j 已经访问过
                    continue
                new_mask = mask | (1 << j)
                new_cost = dp[mask][i] + dist[i][j]
                if new_cost < dp[new_mask][j]:
                    dp[new_mask][j] = new_cost
                    parent[new_mask][j] = i

    # 闭合回路：从最后城市回到城市 0
    full_mask = total_masks - 1  # 所有城市都访问过
    best = INF
    last_city = -1
    for i in range(n):
        if dp[full_mask][i] == INF:
            continue
        total = dp[full_mask][i] + dist[i][0]  # 回到起点
        if total < best:
            best = total
            last_city = i

    # 回溯访问顺序
    path = []
    mask = full_mask
    city = last_city
    while city != -1:
        path.append(city)
        prev = parent[mask][city]
        mask = mask & ~(1 << city)
        city = prev
    path.reverse()
    path.append(0)  # 回到起点

    return best, path


# ============================================================================
# 第五部分：数位 DP — 统计数字和
# ============================================================================

def digit_dp_sum_of_digits(N):
    """
    数位 DP：计算 [1, N] 中所有数字的各位数字之和。
    例如 N=15: 1+2+3+...+9+1+0+1+1+1+2+1+3+1+4+1+5 = 66

    方法：统计每一位上每个数字出现的次数。
    """
    if N <= 0:
        return 0

    digits = list(map(int, str(N)))
    n = len(digits)
    total = 0

    # 对每一位，统计该位上所有数字的贡献
    for pos in range(n):
        # 当前位的权重（个位=1, 十位=10, ...）
        weight = 10 ** (n - pos - 1)

        # 高位部分（当前位前面的数字）
        prefix = 0
        for k in range(pos):
            prefix = prefix * 10 + digits[k]

        # 低位部分（当前位后面的数字）
        suffix = 0
        for k in range(pos + 1, n):
            suffix = suffix * 10 + digits[k]

        current_digit = digits[pos]

        for d in range(10):
            count = 0
            if d < current_digit:
                # 该位选 d < current_digit 时，后面的位可以自由选择
                count = (prefix + 1) * weight
            elif d == current_digit:
                # 该位选 d == current_digit 时，后面受限制
                count = prefix * weight + suffix + 1
            else:
                # d > current_digit，当前位不能选 d
                count = prefix * weight

            total += d * count

    return total


def digit_dp_count_no_digit_4(N):
    """
    数位 DP 示例2：统计 [1, N] 中不包含数字 4 的数的个数。
    使用记忆化搜索。
    """
    if N <= 0:
        return 0

    digits = list(map(int, str(N)))
    n = len(digits)

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def dfs(pos, tight, has_leading_zero):
        """
        pos: 当前处理到的位数索引（0-based，从高位开始）
        tight: 是否受上界约束（1=受约束, 0=自由）
        has_leading_zero: 前面是否都是前导零（用于处理位数不足的情况）
        """
        if pos == n:
            return 1 if not has_leading_zero else 0  # 非全零才算一个有效数字
        limit = digits[pos] if tight else 9
        count = 0
        for d in range(limit + 1):
            if d == 4:  # 跳过包含 4 的情况
                continue
            next_tight = tight and (d == limit)
            next_zero = has_leading_zero and (d == 0)
            count += dfs(pos + 1, next_tight, next_zero)
        return count

    # 减去全零（0 不在 [1, N] 范围内）
    result = dfs(0, True, True)
    # 由于 has_leading_zero 的设定包含了 0，需要额外处理
    return result


# ============================================================================
# 第六部分：可视化
# ============================================================================

def visualize_matrix_chain(dp, split, n, p):
    """可视化矩阵链乘法的 DP 表。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # DP 值表
    im1 = ax1.imshow(dp, cmap='YlOrRd', aspect='auto', origin='upper')
    plt.colorbar(im1, ax=ax1, label='Min Multiplications')
    for i in range(n):
        for j in range(n):
            if i <= j:
                ax1.text(j, i, f'{dp[i][j]}', ha='center', va='center', fontsize=8)
    ax1.set_title('DP Table (Min Cost)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('j (right index)')
    ax1.set_ylabel('i (left index)')
    labels = [f'A{k+1}' for k in range(n)]
    ax1.set_xticks(range(n))
    ax1.set_xticklabels(labels)
    ax1.set_yticks(range(n))
    ax1.set_yticklabels(labels)

    # 分割点表
    im2 = ax2.imshow(split, cmap='viridis', aspect='auto', origin='upper')
    plt.colorbar(im2, ax=ax2, label='Split Point k')
    for i in range(n):
        for j in range(n):
            if i < j:
                ax2.text(j, i, f'{split[i][j]}', ha='center', va='center',
                         fontsize=8, color='white')
    ax2.set_title('Split Points', fontsize=13, fontweight='bold')
    ax2.set_xlabel('j')
    ax2.set_ylabel('i')
    ax2.set_xticks(range(n))
    ax2.set_xticklabels(labels)
    ax2.set_yticks(range(n))
    ax2.set_yticklabels(labels)

    plt.tight_layout()
    path = os.path.join(_IMAGES, 'algo12_matrix_chain_dp.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[可视化] 矩阵链乘 DP 表已保存: {path}')


# ============================================================================
# 主程序
# ============================================================================

def main():
    print('=' * 60)
    print('algo12 动态规划（下）— 演示')
    print('=' * 60)

    # --- 1. 矩阵链乘法 ---
    print('\n### 1. 区间 DP — 矩阵链乘法 ###')
    p = [30, 35, 15, 5, 10, 20, 25]  # 6 个矩阵的维度
    n = len(p) - 1
    min_cost, split = matrix_chain_order(p)
    print(f'矩阵维度: {p}')
    print(f'最小乘法次数: {min_cost}')
    parenthesization = print_mcm_parenthesization(split, 0, n - 1)
    print(f'最优括号化: {parenthesization}')
    visualize_matrix_chain(
        [[split[i][j] if i < j else 0 for j in range(n)] for i in range(n)],
        split, n, p)

    if n <= 6:
        dp_matrix = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i <= j:
                    INF = float('inf')
                    temp = [[0 if a == b else INF for b in range(n)] for a in range(n)]
                    for length in range(2, n + 1):
                        for a in range(n - length + 1):
                            b = a + length - 1
                            for k in range(a, b):
                                cost = (temp[a][k] + temp[k+1][b] +
                                        p[a] * p[k+1] * p[b+1])
                                if cost < temp[a][b]:
                                    temp[a][b] = cost
                    dp_matrix[i][j] = temp[i][j] if i < j else 0
        # Use the computed dp matrix
        dp_mat = [[0 if i == j else float('inf') for j in range(n)] for i in range(n)]
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                for k in range(i, j):
                    cost = (dp_mat[i][k] + dp_mat[k+1][j] +
                            p[i] * p[k+1] * p[j+1])
                    if cost < dp_mat[i][j]:
                        dp_mat[i][j] = cost
        min_final = dp_mat[0][n - 1]

    # --- 2. 石子合并 ---
    print('\n### 2. 区间 DP — 石子合并 ###')
    stones = [4, 3, 2, 6]
    merge_cost, dp_merge = stone_merge(stones)
    print(f'石子堆: {stones}')
    print(f'最小合并代价: {merge_cost}')

    # --- 3. 树形 DP — 最大独立集 ---
    print('\n### 3. 树形 DP — 最大独立集 ###')
    # 构造一棵示例树：星形结构
    n_nodes = 6
    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5)]
    print(f'树边: {edges}')
    mis, dp_tree = tree_max_independent_set(n_nodes, edges)
    print(f'最大独立集大小: {mis}')
    for u in range(n_nodes):
        print(f'  节点 {u}: dp[选]={dp_tree[u][1]}, dp[不选]={dp_tree[u][0]}')

    # --- 4. 树的直径 ---
    print('\n### 4. 树的直径 ###')
    edges2 = [(0, 1), (0, 2), (1, 3), (1, 4), (3, 5)]
    diameter, (A, B) = tree_diameter(6, edges2)
    print(f'树边: {edges2}')
    print(f'直径长度: {diameter} (从节点 {A} 到 {B})')

    # --- 5. 状态压缩 DP — TSP ---
    print('\n### 5. 状态压缩 DP — TSP ###')
    # 小规模 TSP: 4 个城市
    dist_matrix = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]
    print('距离矩阵:')
    for row in dist_matrix:
        print(f'  {row}')
    best_cost, best_path = tsp_held_karp(dist_matrix)
    print(f'最短回路长度: {best_cost}')
    print(f'访问顺序: {" → ".join(str(c) for c in best_path)}')

    # --- 6. 数位 DP — 数字和 ---
    print('\n### 6. 数位 DP — 数字和统计 ###')
    test_n = [15, 100, 999]
    for n_val in test_n:
        print(f'  [1, {n_val}] 所有数字的各位和: {digit_dp_sum_of_digits(n_val)}')

    # 数位 DP — 不含 4 的数字个数
    print('\n### 7. 数位 DP — 不含数字 4 的个数 ###')
    for n_val in [10, 50, 100]:
        print(f'  [1, {n_val}] 不含 4 的数字个数: '
              f'{digit_dp_count_no_digit_4(n_val)}')

    print('\n' + '=' * 60)
    print('总结：进阶 DP 四大专题 — 区间 DP / 树形 DP / 状态压缩 / 数位 DP')
    print('=' * 60)


if __name__ == '__main__':
    main()
