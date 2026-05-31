# -*- coding: utf-8 -*-
"""
algo11 动态规划（上）— 练习代码
===============================
请完成以下 TODO 任务，巩固对 DP 五步设计法的理解。
"""


# ============================================================================
# TODO 1: 完全背包问题（Unbounded Knapsack）
# ============================================================================
def unbounded_knapsack(weights, values, capacity):
    """
    完全背包：每种物品无限量。
    与 0-1 背包的区别：j 的遍历方向从逆序变为正序！

    状态定义：dp[j] = 容量为 j 时的最大价值
    转移方程：dp[j] = max(dp[j], dp[j - w_i] + v_i)

    参数:
        weights: list，各物品重量
        values: list，各物品价值
        capacity: 背包容量
    返回:
        最大总价值，以及每种物品的使用数量
    """
    # TODO: 补全以下代码
    n = len(weights)
    dp = [0] * (capacity + 1)

    # TODO: 外层遍历物品，内层遍历容量（正序！）
    # for i in range(n):
    #     w, v = weights[i], values[i]
    #     for j in range(w, capacity + 1):  ← 注意：正序！
    #         dp[j] = max(dp[j], dp[j - w] + v)

    # TODO: 回溯每种物品使用的数量
    # 提示：使用二维 dp 表 dp2d[i][j] 来记录"考虑前 i 件物品"的状态
    # 然后从 dp2d[n][capacity] 回溯

    # 临时返回
    return 0, {}  # TODO: 替换为正确返回值


# ============================================================================
# TODO 2: 不同子序列计数（Distinct Subsequences）
# ============================================================================
def num_distinct_subsequences(s, t):
    """
    计算字符串 t 在字符串 s 中作为子序列出现的不同方式数。

    dp[i][j] = s[:i] 中 t[:j] 出现的次数
    转移: s[i-1]==t[j-1] → dp[i][j] = dp[i-1][j-1] + dp[i-1][j]
          否则 → dp[i][j] = dp[i-1][j]

    例: s="rabbbit", t="rabbit" → 3

    参数:
        s: 源字符串
        t: 目标子序列
    返回:
        不同子序列的数量
    """
    # TODO: 补全实现
    m, n = len(s), len(t)
    # dp = [[0] * (n + 1) for _ in range(m + 1)]
    # 初始化: dp[i][0] = 1 (空串在任何字符串中出现 1 次)
    # dp[0][j] = 0 for j > 0

    # 填表
    # for i in range(1, m + 1):
    #     for j in range(1, n + 1):
    #         if s[i-1] == t[j-1]:
    #             dp[i][j] = dp[i-1][j-1] + dp[i-1][j]
    #         else:
    #             dp[i][j] = dp[i-1][j]
    # return dp[m][n]
    return 0  # TODO


# ============================================================================
# TODO 3: 最短公共超序列（Shortest Common Supersequence）
# ============================================================================
def shortest_common_supersequence(s1, s2):
    """
    最短公共超序列：包含 s1 和 s2 作为子序列的最短字符串。
    利用 LCS：SCS 长度 = len(s1) + len(s2) - LCS 长度

    参数:
        s1, s2: 两个输入字符串
    返回:
        scs 字符串
    """
    # TODO: 先计算 LCS 表
    # m, n = len(s1), len(s2)
    # dp = [[0] * (n + 1) for _ in range(m + 1)]
    # ... (LCS 填表)

    # TODO: 从 dp 表回溯构建 SCS
    # i, j = m, n
    # scs = []
    # while i > 0 and j > 0:
    #     if s1[i-1] == s2[j-1]:
    #         scs.append(s1[i-1])
    #         i -= 1; j -= 1
    #     elif dp[i-1][j] >= dp[i][j-1]:
    #         scs.append(s1[i-1])  # 保留了 s1 但没保留 s2 → 加入 s1
    #         i -= 1
    #     else:
    #         scs.append(s2[j-1])
    #         j -= 1
    # # 追加剩余字符
    # while i > 0: scs.append(s1[i-1]); i -= 1
    # while j > 0: scs.append(s2[j-1]); j -= 1
    # return ''.join(reversed(scs))
    return ''  # TODO


# ============================================================================
# TODO 4: 分割等和子集（Partition Equal Subset Sum）
# ============================================================================
def can_partition(nums):
    """
    判断能否将数组分成两个和相等的子集。
    等价于：是否存在子集，其和为 sum(nums)/2。
    所以：target = sum(nums)/2，调用子集和 DP。

    参数:
        nums: list of int，整数数组
    返回:
        (can_partition, subset1, subset2)
    """
    # TODO: 实现
    total = sum(nums)
    # 如果总和为奇数，不可能等分
    # if total % 2 != 0: return False, [], []
    # target = total // 2
    # 使用一维布尔 DP（空间优化版）
    # dp = [False] * (target + 1)
    # dp[0] = True
    # for num in nums:
    #     for j in range(target, num - 1, -1):  # 逆序！
    #         dp[j] = dp[j] or dp[j - num]
    # 回溯分拆两个子集...
    return False, [], []  # TODO


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == '__main__':
    print('=' * 50)
    print('algo11 动态规划（上）— 练习')
    print('=' * 50)

    # 测试 TODO 1: 完全背包
    print('\n--- TODO 1: 完全背包 ---')
    w, v = [2, 3, 4], [4, 5, 6]
    cap = 8
    max_val, counts = unbounded_knapsack(w, v, cap)
    print(f'  物品 (w={w}, v={v}), 容量={cap}')
    print(f'  最大价值={max_val}, 使用数量={counts}')

    # 测试 TODO 2: 不同子序列
    print('\n--- TODO 2: 不同子序列计数 ---')
    print(f'  s="rabbbit", t="rabbit" → {num_distinct_subsequences("rabbbit", "rabbit")} (期望: 3)')
    print(f'  s="babgbag", t="bag" → {num_distinct_subsequences("babgbag", "bag")} (期望: 5)')

    # 测试 TODO 3: 最短公共超序列
    print('\n--- TODO 3: 最短公共超序列 ---')
    print(f'  "abac", "cab" → SCS = "{shortest_common_supersequence("abac", "cab")}"')

    # 测试 TODO 4: 分割等和子集
    print('\n--- TODO 4: 分割等和子集 ---')
    for test in [[1, 5, 11, 5], [1, 2, 3, 5], [2, 2, 3, 5]]:
        ok, s1, s2 = can_partition(test)
        print(f'  {test} → {ok}', end='')
        if ok:
            print(f'  ({s1} | {s2})')
        else:
            print()

    print('\n提示: 请补全上方所有 TODO 函数后再运行测试。')
