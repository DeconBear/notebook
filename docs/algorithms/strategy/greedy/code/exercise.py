# -*- coding: utf-8 -*-
"""
algo09 贪心算法 — 练习代码
===========================
请完成以下 TODO 任务，巩固对贪心算法的理解。

每个 TODO 都有详细的提示。建议先阅读 index.md 和 demo.py，再尝试独立补全代码。
"""

import heapq


# ============================================================================
# TODO 1: 完成"最多合并次数"贪心算法
# ============================================================================
def min_merge_cost(arr):
    """
    问题：将 n 堆石子合并为一堆，每次只能合并相邻的两堆，合并的代价为两堆之和。
    求最小总合并代价。（类似 Huffman 编码的构造过程）

    贪心策略：每次选出最小的两个数进行合并。

    参数:
        arr: list，每堆石子的数量，例如 [4, 3, 2, 6]
    返回:
        total_cost: 最小总合并代价
        合并过程的日志列表
    """
    # TODO: 补全以下代码
    heap = arr.copy()
    # 提示: 使用 heapq.heapify(heap) 将列表转为最小堆
    # TODO: 初始化最小堆

    total_cost = 0
    log = []

    # 当堆中还有多于1堆时，继续合并
    while len(heap) > 1:
        # TODO: 取出最小的两堆
        a = None  # ← TODO: heapq.heappop(heap)
        b = None  # ← TODO: heapq.heappop(heap)

        # TODO: 合并代价 = a + b
        cost = None  # ← TODO

        # TODO: 累加到总代价
        # total_cost = ???

        # 将合并后的新堆放回堆中
        # TODO: heapq.heappush(heap, cost)

        log.append(f'合并 {a} 和 {b} → {cost}，累计 {total_cost}')

    return total_cost, log


# ============================================================================
# TODO 2: 区间覆盖问题
# ============================================================================
def min_intervals_to_cover(intervals, target_range):
    """
    问题：给定若干区间 [l, r] 和一个目标区间 [L, R]，
    求最少需要多少个区间才能完全覆盖目标区间。

    贪心策略：每次在已覆盖的范围内，选择右端点最远的区间。

    参数:
        intervals: list of (l, r)，可用的区间列表
        target_range: (L, R)，需要覆盖的目标区间
    返回:
        count: 最少区间数（-1 表示无法覆盖）
        chosen: 被选中的区间列表（按顺序）
    """
    L, R = target_range
    # TODO: 按左端点升序排列区间
    # sorted_intervals = ???

    count = 0
    chosen = []
    current_end = L   # 当前已覆盖到的右边界
    i = 0             # 当前遍历到的区间索引
    n = len(intervals)

    while current_end < R:
        best_end = current_end  # 本次迭代能找到的最远右端点
        best_idx = -1          # 对应的区间索引

        # 在所有左端点 <= current_end 的区间中，选右端点最大的
        while i < n and intervals[i][0] <= current_end:
            # TODO: 更新 best_end 和 best_idx
            # if intervals[i][1] > best_end:
            #     best_end = intervals[i][1]
            #     best_idx = i
            i += 1  # ← 这里有 bug！应该在外面遍历时递增
            pass

        # TODO: 如果 best_end 没有推进，说明无法继续覆盖
        # if best_end == current_end:
        #     return -1, []

        # TODO: 选择该区间，更新 current_end
        # chosen.append(intervals[best_idx])
        # current_end = best_end
        # count += 1

        break  # ← TODO: 删除这行，这是一个临时防护

    return count, chosen


# ============================================================================
# TODO 3: 判断一个面额系统是否为"规范性"（贪心找零正确性验证）
# ============================================================================
def verify_greedy_coin_system(coins, max_amount=100):
    """
    验证贪心找零算法在给定面额系统下，对 0 到 max_amount 的所有金额是否都最优。

    参数:
        coins: list，面额列表
        max_amount: 测试的最大金额
    返回:
        is_valid: bool，贪心是否对所有测试金额都正确
        failures: list of (amount, greedy_count, optimal_count)，失败案例
    """
    # TODO: 使用动态规划计算 0~max_amount 的最少硬币数作为基准
    # dp[0] = 0
    # for i in range(1, max_amount+1):
    #     for coin in coins:
    #         if i >= coin: dp[i] = min(dp[i], dp[i-coin] + 1)

    # 然后用贪心算法计算每个金额的硬币数
    # 比较两者是否一致，记录所有不一致的金额

    # TODO: 实现完整代码
    pass


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == '__main__':
    print('=' * 50)
    print('algo09 贪心算法 — 练习')
    print('=' * 50)

    # 测试 TODO 1: 石子合并
    print('\n--- TODO 1: 石子合并最小代价 ---')
    test_arr = [4, 3, 2, 6]
    # 期望: 合并 2+3=5 (累计5) → 合并 4+5=9 (累计14) → 合并 6+9=15 (累计29)
    cost, log = min_merge_cost(test_arr)
    print(f'数组: {test_arr}')
    print(f'最小总代价: {cost}')
    for step in log:
        print(f'  {step}')

    # 测试 TODO 2: 区间覆盖
    print('\n--- TODO 2: 区间覆盖最小数量 ---')
    test_intervals = [(1, 3), (2, 5), (3, 7), (4, 6), (6, 9), (7, 10)]
    target = (1, 10)
    count, chosen = min_intervals_to_cover(test_intervals, target)
    print(f'区间: {test_intervals}')
    print(f'目标: {target}')
    print(f'最少区间数: {count}')
    print(f'选择的区间: {chosen}')
    # 期望: 3个区间，如 [(1,3), (2,5), (7,10)] 或 [(1,3), (3,7), (7,10)]

    # 测试 TODO 3: 验证贪心找零正确性
    print('\n--- TODO 3: 验证面额系统的贪心正确性 ---')
    # 规范面额（人民币）应返回 True
    # validate, fails = verify_greedy_coin_system([1, 5, 10, 20, 50, 100])
    # print(f'人民币面额: {"贪心总是正确的" if validate else f"失败案例: {fails}"}')
    # 非规范面额 {1, 3, 4} 应返回 False
    # validate, fails = verify_greedy_coin_system([1, 3, 4])
    # print(f'{{1,3,4}}面额: {"贪心总是正确的" if validate else f"失败案例: {fails}"}')

    print('\n提示: 请补全上方所有 TODO 函数后再运行测试。')
