# -*- coding: utf-8 -*-
"""
algo10 递归、分治与二分 — 练习代码
==================================
请完成以下 TODO 任务，巩固对分治法和二分搜索的理解。
"""


# ============================================================================
# TODO 1: 实现快速幂（Binary Exponentiation / Fast Pow）
# ============================================================================
def fast_pow(x, n):
    """
    使用分治思想计算 x^n。
    核心思路：x^n = (x^(n/2))^2（当 n 为偶数），x^n = x * (x^(n/2))^2（当 n 为奇数）

    例如计算 x^13:
    x^13 = x * (x^6)^2
    x^6 = (x^3)^2
    x^3 = x * (x^1)^2
    x^1 = x * (x^0)^2 = x

    参数:
        x: 底数（float 或 int）
        n: 指数（非负整数）
    返回:
        x^n 的值
    """
    # TODO: 补全以下代码
    # 基线条件
    if n == 0:
        return 1.0  # 任何数的 0 次方 = 1

    # 递归条件：分治思想
    # half = fast_pow(x, n // 2)  ← 计算 x^(n/2)
    # if n % 2 == 0:
    #     return half * half      ← 偶数：x^n = (x^(n/2))^2
    # else:
    #     return _______________  ← 奇数：x^n = x * (x^(n/2))^2

    # TODO: 补全上面的递归实现
    return x ** n  # 临时占位


# ============================================================================
# TODO 2: 实现"在旋转排序数组中搜索"的二分搜索变体
# ============================================================================
def search_rotated_array(nums, target):
    """
    在旋转排序数组中搜索 target 的索引。
    旋转排序数组：一个有序数组在某点被旋转，如 [4,5,6,7,0,1,2]。

    思路：虽然整体不是有序的，但每次二分后，至少有一半是有序的。
    利用有序那一半判断 target 是否在其中。

    参数:
        nums: list，旋转排序数组
        target: 目标值
    返回:
        索引，-1 表示未找到
    """
    # TODO: 补全以下代码
    if not nums:
        return -1

    lo, hi = 0, len(nums) - 1

    while lo <= hi:
        mid = (lo + hi) // 2

        if nums[mid] == target:
            return mid

        # 判断哪半边是有序的
        # TODO: 如果左半边有序 (nums[lo] <= nums[mid])
        #   TODO: 如果 target 在左半边范围内，hi = mid - 1
        #   TODO: 否则 lo = mid + 1
        # TODO: 否则右半边有序
        #   TODO: 如果 target 在右半边范围内，lo = mid + 1
        #   TODO: 否则 hi = mid - 1
        pass  # ← TODO: 替换为完整实现

    return -1


# ============================================================================
# TODO 3: 实现分治法求最大子数组和（Maximum Subarray）
# ============================================================================
def max_subarray_divide_conquer(nums):
    """
    用分治法求解最大子数组和问题（Kadane DP O(n) 的替代方案）。
    分治思路：
    1. 数组从中点分成两半
    2. 最大子数组要么在左半边，要么在右半边，要么横跨中点
    3. 横跨中点的：从中点向左找最大后缀和 + 从中点向右找最大前缀和

    参数:
        nums: list，整数数组（可含负数）
    返回:
        最大子数组的和
    """
    def helper(lo, hi):
        # 基线条件：单个元素
        if lo == hi:
            return nums[lo]

        mid = (lo + hi) // 2

        # 递归求左右两边的最优解
        left_max = helper(lo, mid)
        right_max = helper(mid + 1, hi)

        # TODO: 计算横跨中点的最大子数组和
        # 从中点向左扫描，累加求和，记录最大后缀和
        # left_sum = float('-inf')
        # curr = 0
        # for i in range(mid, lo - 1, -1):
        #     curr += nums[i]
        #     left_sum = max(left_sum, curr)

        # 从中点向右扫描，累加求和，记录最大前缀和
        # right_sum = float('-inf')
        # curr = 0
        # for i in range(mid + 1, hi + 1):
        #     curr += nums[i]
        #     right_sum = max(right_sum, curr)

        # 横跨和 = left_sum + right_sum
        cross_max = float('-inf')  # TODO: 替换为正确值

        # 返回三者中的最大值
        return max(left_max, right_max, cross_max)

    if not nums:
        return 0
    return helper(0, len(nums) - 1)


# ============================================================================
# TODO 4: 实现二分答案：在 n 个木板间选 k 段，使最小段长度最大化
# ============================================================================
def maximize_min_segment_length(positions, k):
    """
    给定 n 个木板的坐标（已排序），需要放置 k-1 个切割点将木板分成 k 段。
    求使最小段长度最大化的切割方案，返回该最大化的最小段长度。

    思路（二分答案）：
    设答案为 x，检查能否将木板分成 k 段，每段长度 ≥ x。
    贪心扫描：从第一个位置开始，每次尽可能延申直到段长 ≥ x，计数段数。
    如果段数 ≥ k，则 x 可行。

    参数:
        positions: list of int，已排序的位置坐标
        k: 需要分成的段数
    返回:
        最大化的最小段长度（int）
    """
    def can_divide(min_len):
        """检查能否将 positions 分成 k 段，每段长度 >= min_len"""
        # TODO: 实现检查逻辑
        # segments = 0
        # last_pos = positions[0]
        # for pos in positions[1:]:
        #     if pos - last_pos >= min_len:
        #         segments += 1
        #         last_pos = pos
        # return segments + 1 >= k  (包括最后一段)
        return False  # TODO

    if not positions or k <= 1:
        return 0
    if k > len(positions):
        return 0

    # TODO: 二分搜索最大可行的最小段长度
    # lo = 0
    # hi = positions[-1] - positions[0]
    # while lo < hi:
    #     mid = (lo + hi + 1) // 2  # 上取整，避免死循环
    #     if can_divide(mid):
    #         lo = mid
    #     else:
    #         hi = mid - 1
    # return lo
    return 0  # TODO


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == '__main__':
    print('=' * 50)
    print('algo10 递归、分治与二分 — 练习')
    print('=' * 50)

    # 测试 TODO 1: 快速幂
    print('\n--- TODO 1: 快速幂 ---')
    print(f'  2^10 = {fast_pow(2, 10)}  (期望: 1024)')
    print(f'  3^5  = {fast_pow(3, 5)}   (期望: 243)')
    print(f'  2^0  = {fast_pow(2, 0)}   (期望: 1)')

    # 测试 TODO 2: 旋转数组搜索
    print('\n--- TODO 2: 旋转排序数组搜索 ---')
    arr2 = [4, 5, 6, 7, 0, 1, 2]
    for t in [0, 3, 6]:
        print(f'  search_rotated_array({arr2}, {t}) = {search_rotated_array(arr2, t)}')

    # 测试 TODO 3: 分治法最大子数组和
    print('\n--- TODO 3: 分治法最大子数组和 ---')
    test_cases = [
        ([-2, 1, -3, 4, -1, 2, 1, -5, 4], 6),        # 标准测试
        ([1, 2, 3, 4], 10),                            # 全正
        ([-1, -2, -3, -4], -1),                        # 全负
    ]
    for nums, expected in test_cases:
        result = max_subarray_divide_conquer(nums)
        status = '✓' if result == expected else f'✗ (got {result})'
        print(f'  {nums} → {result} {status}')

    # 测试 TODO 4: 二分答案
    print('\n--- TODO 4: 二分答案（最大化最小段长）---')
    positions = [0, 3, 4, 8, 9, 10]
    k = 3
    result = maximize_min_segment_length(positions, k)
    print(f'  位置: {positions}, k={k} → 最大最小段长 = {result}')

    print('\n提示: 请补全上方所有 TODO 函数后再运行测试。')
