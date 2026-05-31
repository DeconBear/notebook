# -*- coding: utf-8 -*-
"""
algo11 动态规划（上）— 演示代码
===============================
功能：用 Top-Down（记忆化递归）和 Bottom-Up（迭代制表）两种方式实现
      六大经典 DP 问题——0-1 背包、子集和、LCS、LIS、编辑距离、找零。
      + DP 表可视化（背包表、LCS 表、编辑距离表）。

每个函数都有中文 docstring，每行逻辑代码都有中文注释。
运行方式：在 algo11_dp_1/ 目录下执行 python code/demo.py
"""

import os
import sys
import bisect
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.patches as mpatches

_HERE = os.path.dirname(os.path.abspath(__file__))
_IMAGES = os.path.join(_HERE, '..', 'images')
os.makedirs(_IMAGES, exist_ok=True)


# ============================================================================
# 工具函数：打印二维 DP 表
# ============================================================================

def print_dp_table(dp, row_labels=None, col_labels=None, title='DP Table'):
    """漂亮地打印二维 DP 表。"""
    print(f'\n{title}:')
    m, n = len(dp), len(dp[0])
    # 表头
    if col_labels:
        header = ' ' * 6
        for j in range(n):
            label = col_labels[j] if j < len(col_labels) else str(j)
            header += f'{label:>5}'
        print(header)
    for i in range(m):
        prefix = ''
        if row_labels and i < len(row_labels):
            prefix = f'{row_labels[i]:>5} '
        elif row_labels is None:
            prefix = f'{i:>5} '
        row_str = prefix + ' '.join(f'{dp[i][j]:>4}' for j in range(n))
        print(row_str)


# ============================================================================
# 第一部分：0-1 背包问题（两种实现）
# ============================================================================

def knapsack_01_topdown(weights, values, capacity):
    """
    0-1 背包 — Top-Down（记忆化递归）。
    dp[i][cap] = 考虑前 i 件物品，容量为 cap 时的最大价值
    """
    n = len(weights)
    memo = [[-1] * (capacity + 1) for _ in range(n + 1)]

    def solve(i, cap):
        """递归求解：前 i 件物品，剩余容量 cap"""
        if i == 0 or cap == 0:  # 基线条件：无物品或无容量
            return 0
        if memo[i][cap] != -1:  # 已计算过 → 直接返回
            return memo[i][cap]
        # 不选第 i 件（索引为 i-1）
        result = solve(i - 1, cap)
        # 如果容量够，考虑选第 i 件
        if cap >= weights[i - 1]:
            result = max(result,
                         solve(i - 1, cap - weights[i - 1]) + values[i - 1])
        memo[i][cap] = result  # 存入备忘录
        return result

    max_value = solve(n, capacity)
    # 回溯：重建选择了哪些物品
    selected = []
    cap = capacity
    for i in range(n, 0, -1):
        if (cap >= weights[i - 1] and
                memo[i][cap] != memo[i - 1][cap]):  # 选了第 i 件
            selected.append(i - 1)
            cap -= weights[i - 1]

    return max_value, selected[::-1], memo


def knapsack_01_bottomup(weights, values, capacity):
    """
    0-1 背包 — Bottom-Up（迭代制表）。
    空间优化版：一维滚动数组，j 逆序遍历。
    """
    n = len(weights)
    dp = [0] * (capacity + 1)  # 一维数组，初始化为 0

    for i in range(n):
        # 注意：j 从 capacity 往下遍历到 weights[i]，这是关键！
        # 正序会导致一件物品被多次使用（完全背包）
        for j in range(capacity, weights[i] - 1, -1):
            # 状态转移：max(不选, 选)
            dp[j] = max(dp[j], dp[j - weights[i]] + values[i])

    # 回溯：需要完整的二维表信息 → 调用二维版本
    dp2d = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        w, v = weights[i - 1], values[i - 1]
        for j in range(capacity + 1):
            dp2d[i][j] = dp2d[i - 1][j]
            if j >= w:
                dp2d[i][j] = max(dp2d[i][j], dp2d[i - 1][j - w] + v)

    # 回溯选择
    selected = []
    j = capacity
    for i in range(n, 0, -1):
        if j >= weights[i - 1] and dp2d[i][j] != dp2d[i - 1][j]:
            selected.append(i - 1)
            j -= weights[i - 1]

    return dp[capacity], selected[::-1], dp2d


# ============================================================================
# 第二部分：子集和问题（0-1 背包的布尔变体）
# ============================================================================

def subset_sum(nums, target):
    """
    子集和问题：判断能否选出若干个数使其和为 target。
    这是 0-1 背包的布尔值版本。
    dp[i][j] = 前 i 个数能否凑出和 j
    """
    n = len(nums)
    dp = [[False] * (target + 1) for _ in range(n + 1)]
    dp[0][0] = True  # 0 个数凑出和 0

    for i in range(1, n + 1):
        for j in range(target + 1):
            # 不选 nums[i-1]
            dp[i][j] = dp[i - 1][j]
            # 选 nums[i-1]（如果够）
            if j >= nums[i - 1]:
                dp[i][j] = dp[i][j] or dp[i - 1][j - nums[i - 1]]

    # 回溯子集
    subset = []
    if dp[n][target]:
        j = target
        for i in range(n, 0, -1):
            if j >= nums[i - 1] and dp[i - 1][j - nums[i - 1]]:
                subset.append(nums[i - 1])
                j -= nums[i - 1]

    return dp[n][target], subset[::-1]


# ============================================================================
# 第三部分：最长公共子序列（LCS）
# ============================================================================

def lcs(s1, s2):
    """
    最长公共子序列（LCS）。
    dp[i][j] = s1[:i] 和 s2[:j] 的 LCS 长度。
    转移: s1[i-1]==s2[j-1] → dp[i-1][j-1]+1
          否则 → max(dp[i-1][j], dp[i][j-1])
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1  # 字符匹配 → 加长 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])  # 跳过某字符

    # 回溯 LCS 字符串
    i, j = m, n
    lcs_chars = []
    while i > 0 and j > 0:
        if s1[i - 1] == s2[j - 1]:
            lcs_chars.append(s1[i - 1])  # 匹配字符属于 LCS
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1  # 向上移动
        else:
            j -= 1  # 向左移动

    return dp[m][n], ''.join(reversed(lcs_chars)), dp


# ============================================================================
# 第四部分：最长递增子序列（LIS）
# ============================================================================

def lis_dp(arr):
    """
    LIS — O(n^2) DP 解法。
    dp[i] = 以 arr[i] 结尾的最长递增子序列长度。
    """
    n = len(arr)
    if n == 0:
        return 0, []
    dp = [1] * n      # 每个元素自身是长度为 1 的 LIS
    prev = [-1] * n   # 用于回溯 LIS 序列

    for i in range(n):
        for j in range(i):
            if arr[j] < arr[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                prev[i] = j

    # 找最大 dp 值和对应的终点
    max_len = max(dp)
    end_idx = dp.index(max_len)

    # 回溯 LIS 序列
    seq = []
    idx = end_idx
    while idx != -1:
        seq.append(arr[idx])
        idx = prev[idx]

    return max_len, seq[::-1]


def lis_patience_sort(arr):
    """
    LIS — O(n log n) 解法（Patience Sorting + 二分）。
    tails[k] = 长度为 k+1 的 LIS 的最小可能末尾元素。
    """
    if not arr:
        return 0, []

    tails = []         # tails[k] = 长度为 k+1 的 LIS 的最小末尾值
    # tails_idx[k] = tails[k] 在原始数组中的索引
    tails_idx = []
    prev = [-1] * len(arr)  # prev[i] = LIS 中 arr[i] 的前一个元素的索引

    for i, x in enumerate(arr):
        # 二分查找 tails 中第一个 >= x 的位置
        pos = bisect.bisect_left(tails, x)
        if pos == len(tails):
            tails.append(x)
            tails_idx.append(i)
        else:
            tails[pos] = x
            tails_idx[pos] = i
        # 记录 prev
        if pos > 0:
            prev[i] = tails_idx[pos - 1]

    # 回溯 LIS 序列
    seq = []
    idx = tails_idx[-1]
    while idx != -1:
        seq.append(arr[idx])
        idx = prev[idx]

    return len(tails), seq[::-1]


# ============================================================================
# 第五部分：编辑距离
# ============================================================================

def edit_distance(s1, s2):
    """
    编辑距离（Levenshtein Distance）。
    dp[i][j] = 将 s1[:i] 转换为 s2[:j] 的最少操作次数。
    操作：插入、删除、替换。
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 初始化：空串到另一字符串需要全部插入/删除
    for i in range(m + 1):
        dp[i][0] = i  # 删除 i 个字符
    for j in range(n + 1):
        dp[0][j] = j  # 插入 j 个字符

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # 字符相同，无需操作
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # 删除 s1[i-1]
                    dp[i][j - 1],      # 插入 s2[j-1]
                    dp[i - 1][j - 1]   # 替换 s1[i-1] → s2[j-1]
                )

    # 回溯编辑操作序列
    ops = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and s1[i - 1] == s2[j - 1]:
            ops.append(f'keep  {s1[i-1]}')
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(f'repl  {s1[i-1]}→{s2[j-1]}')
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(f'del   {s1[i-1]}')
            i -= 1
        else:
            ops.append(f'ins   {s2[j-1]}')
            j -= 1

    return dp[m][n], list(reversed(ops)), dp


# ============================================================================
# 第六部分：找零问题（最少硬币，DP 版本）
# ============================================================================

def coin_change_min(coins, amount):
    """
    找零问题（最少硬币数）— DP 解法。
    dp[i] = 凑出金额 i 所需的最少硬币数。
    任意面额系统都正确。
    """
    INF = float('inf')
    dp = [INF] * (amount + 1)
    dp[0] = 0  # 0 元需要 0 枚硬币
    # 记录每个金额最后使用的那枚硬币的面额
    coin_used = [-1] * (amount + 1)

    for i in range(1, amount + 1):
        for coin in coins:
            if i >= coin and dp[i - coin] + 1 < dp[i]:
                dp[i] = dp[i - coin] + 1
                coin_used[i] = coin

    if dp[amount] == INF:
        return -1, None, None  # 无法凑出

    # 回溯具体硬币组合
    result = {}
    remaining = amount
    while remaining > 0:
        c = coin_used[remaining]
        result[c] = result.get(c, 0) + 1
        remaining -= c

    return dp[amount], result, coin_used


# ============================================================================
# 第七部分：可视化 DP 表
# ============================================================================

def visualize_knapsack(dp2d, weights, values, capacity):
    """可视化 0-1 背包的 DP 表。"""
    m, n = len(dp2d), len(dp2d[0])
    fig, ax = plt.subplots(figsize=(max(n // 3, 10), max(m // 2, 6)))

    # 绘制热力图
    im = ax.imshow(dp2d, cmap='YlOrRd', aspect='auto', origin='upper')
    plt.colorbar(im, ax=ax, label='Max Value')

    # 标注数值
    for i in range(m):
        for j in range(n):
            ax.text(j, i, f'{dp2d[i][j]}', ha='center', va='center',
                    fontsize=7, color='black' if dp2d[i][j] < max(max(r) for r in dp2d) * 0.6 else 'white')

    ax.set_xlabel('Capacity', fontsize=12)
    ax.set_ylabel('Items Considered (first k)', fontsize=12)
    ax.set_title('0-1 Knapsack DP Table', fontsize=14, fontweight='bold')
    ax.set_xticks(range(n))
    ax.set_yticks(range(m))

    path = os.path.join(_IMAGES, 'algo11_knapsack_dp_table.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[可视化] 0-1 背包 DP 表已保存: {path}')


def visualize_edit_distance(dp, s1, s2):
    """可视化编辑距离的 DP 表。"""
    m, n = len(dp), len(dp[0])
    fig, ax = plt.subplots(figsize=(max(n // 2, 8), max(m // 2, 6)))

    im = ax.imshow(dp, cmap='Blues', aspect='auto', origin='upper')
    plt.colorbar(im, ax=ax, label='Edit Distance')

    for i in range(m):
        for j in range(n):
            ax.text(j, i, f'{dp[i][j]}', ha='center', va='center',
                    fontsize=9, fontweight='bold',
                    color='white' if dp[i][j] > 5 else 'black')

    # 列标签
    ax.set_xticks(range(n))
    ax.set_xticklabels([''] + list(s2))
    ax.set_yticks(range(m))
    ax.set_yticklabels([''] + list(s1))
    ax.set_xlabel(f'String 2: "{s2}"', fontsize=12)
    ax.set_ylabel(f'String 1: "{s1}"', fontsize=12)
    ax.set_title(f'Edit Distance DP Table (result = {dp[m-1][n-1]})',
                 fontsize=14, fontweight='bold')

    path = os.path.join(_IMAGES, 'algo11_edit_distance_table.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[可视化] 编辑距离 DP 表已保存: {path}')


# ============================================================================
# 主程序
# ============================================================================

def main():
    print('=' * 60)
    print('algo11 动态规划（上）— 演示')
    print('=' * 60)

    # --- 1. 0-1 背包 ---
    print('\n### 1. 0-1 背包问题 ###')
    weights = [2, 3, 4, 5]
    values = [3, 4, 5, 6]
    capacity = 8
    print(f'物品重量: {weights}, 价值: {values}, 容量: {capacity}')

    max_v_td, sel_td, memo = knapsack_01_topdown(weights, values, capacity)
    print(f'  Top-Down: 最大价值={max_v_td}, 选择物品索引={sel_td}')

    max_v_bu, sel_bu, dp2d = knapsack_01_bottomup(weights, values, capacity)
    print(f'  Bottom-Up: 最大价值={max_v_bu}, 选择物品索引={sel_bu}')
    visualize_knapsack(dp2d, weights, values, capacity)

    # --- 2. 子集和 ---
    print('\n### 2. 子集和问题 ###')
    nums = [3, 34, 4, 12, 5, 2]
    target = 9
    possible, subset = subset_sum(nums, target)
    print(f'数组: {nums}, 目标和: {target}')
    print(f'  能否凑出? {possible} → 子集: {subset if possible else "N/A"}')

    target2 = 30
    possible2, subset2 = subset_sum(nums, target2)
    print(f'  目标和: {target2} → 能否凑出? {possible2}')

    # --- 3. LCS ---
    print('\n### 3. 最长公共子序列（LCS）###')
    s1, s2 = 'ABCDAB', 'BDCABA'
    lcs_len, lcs_str, _ = lcs(s1, s2)
    print(f'  A="{s1}", B="{s2}" → LCS="{lcs_str}", 长度={lcs_len}')

    # --- 4. LIS ---
    print('\n### 4. 最长递增子序列（LIS）###')
    test_arrays = [
        [10, 9, 2, 5, 3, 7, 101, 18],
        [0, 1, 0, 3, 2, 3],
        [7, 7, 7, 7]
    ]
    for arr in test_arrays:
        len1, seq1 = lis_dp(arr)
        len2, seq2 = lis_patience_sort(arr)
        print(f'  {arr}')
        print(f'    DP O(n^2): 长度={len1}, 序列={seq1}')
        print(f'    Patience O(n log n): 长度={len2}, 序列={seq2}')

    # --- 5. 编辑距离 ---
    print('\n### 5. 编辑距离 ###')
    test_pairs = [('horse', 'ros'), ('intention', 'execution'), ('abc', 'abc')]
    for w1, w2 in test_pairs:
        dist, ops, ed_dp = edit_distance(w1, w2)
        print(f'  "{w1}" → "{w2}": 编辑距离={dist}')
        for op in ops:
            print(f'    {op}')

    # 可视化编辑距离
    dist, ops, ed_dp = edit_distance('horse', 'ros')
    visualize_edit_distance(ed_dp, 'horse', 'ros')

    # --- 6. 找零问题（DP） ---
    print('\n### 6. 找零问题（最少硬币数，DP） ###')
    coins_list = [[1, 5, 10, 25], [1, 3, 4]]
    for coins in coins_list:
        for amt in [6, 18, 37]:
            min_coins, combination, _ = coin_change_min(coins, amt)
            print(f'  面额 {coins}, 金额 {amt}: 最少 {min_coins} 枚 → {combination}')

    print('\n' + '=' * 60)
    print('总结：DP 五步法 — 状态定义 → 转移方程 → 初始条件 → 计算顺序 → 提取答案')
    print('=' * 60)


if __name__ == '__main__':
    main()
