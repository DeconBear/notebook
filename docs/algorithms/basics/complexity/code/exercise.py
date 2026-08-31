# -*- coding: utf-8 -*-
"""
===============================================================================
algo01_complexity/code/exercise.py — 复杂度分析练习
===============================================================================
本练习文件中，你需要完成以下任务：
  1. 分析给定代码片段的复杂度（大 O 记号）
  2. 实现动态数组的均摊分析模拟
  3. 使用主定理分析分治算法的递推式
  4. 设计一个"势能法"均摊分析

运行方式：python exercise.py
===============================================================================
"""

import math
import time
import random


# ============================================================================
# 任务 1：分析代码复杂度（填写大 O 记号）
# ============================================================================

def analyze_complexity():
    """分析以下代码片段的复杂度，将你的答案填在注释中。"""
    print('========== 任务 1：复杂度分析 ==========')

    # TODO: 分析下面函数的复杂度，填入你的答案
    # 复杂度：O(____)
    def func_a(n):
        s = 0
        for i in range(n):
            s += i
        for j in range(n):
            s -= j
        return s

    # TODO: 复杂度：O(____)
    def func_b(n):
        s = 0
        for i in range(n):
            j = 1
            while j < n:
                s += 1
                j *= 2  # j 每次翻倍
        return s

    # TODO: 复杂度：O(____)
    def func_c(n):
        if n <= 1:
            return 1
        return func_c(n // 2) + func_c(n // 2)  # 两个递归调用

    # TODO: 复杂度：O(____)
    def func_d(n):
        s = 0
        i = n
        while i > 1:
            for j in range(n):
                s += 1
            i //= 2
        return s

    # TODO: 复杂度：O(____)
    def func_e(arr):
        # arr 是已排序的数组
        n = len(arr)
        for i in range(n):
            # 使用二分查找在 arr[i+1:] 中查找 -arr[i]
            lo, hi = i + 1, n - 1
            while lo <= hi:
                mid = (lo + hi) // 2
                if arr[mid] == -arr[i]:
                    return True
                elif arr[mid] < -arr[i]:
                    lo = mid + 1
                else:
                    hi = mid - 1
        return False

    print('请将以上五个函数的大 O 复杂度填入 TODO 注释中。')
    print('提示：')
    print('  func_a: 两个独立的 O(n) 循环')
    print('  func_b: 外层 O(n)，内层 j 每次翻倍...')
    print('  func_c: T(n) = 2T(n/2) + O(1)，主定理...')
    print('  func_d: 外层 while 除以 2，内层 O(n)...')
    print('  func_e: 外层 O(n)，内层二分查找...')


# ============================================================================
# 任务 2：模拟动态数组的均摊分析
# ============================================================================

class SimulatedDynamicArray:
    """模拟动态数组，用于均摊分析。

    你需要完成 append 方法中的扩容逻辑。
    策略：当数组满时，扩容为原来的 2 倍。
    """

    def __init__(self):
        self.capacity = 1
        self.size = 0
        self.total_copies = 0  # 累积复制次数

    def append(self, value):
        """
        TODO: 完成动态数组的 append 操作
        1. 如果 size == capacity，需要扩容（capacity *= 2）
        2. 扩容时，需要复制旧元素到新数组（计入 total_copies）
        3. 将新元素放入合适位置，size += 1
        """
        # TODO: 你的代码从这里开始
        if self.size == self.capacity:
            # 扩容
            self.capacity *= 2
            # 每次扩容需要复制 size 个旧元素
            self.total_copies += self.size

        # 追加元素（不需要真的存储值，只追踪 size 和 copies 即可）
        self.size += 1

    def amortized_cost(self):
        """返回均摊每次 append 的复制次数"""
        if self.size == 0:
            return 0
        return self.total_copies / self.size


def task2_simulate():
    """运行动态数组均摊分析模拟"""
    print('\n========== 任务 2：动态数组均摊分析 ==========')
    da = SimulatedDynamicArray()
    log = []

    for i in range(1, 65):
        da.append(i)
        if i % 8 == 0 or i < 4 or i == 64:
            log.append((i, da.capacity, da.total_copies, da.amortized_cost()))

    print(f'{"n":>4}  {"容量":>6}  {"总复制":>6}  {"均摊":>6}')
    for size, cap, copies, amort in log:
        print(f'{size:4d}  {cap:6d}  {copies:6d}  {amort:6.2f}')

    print(f'\n最终均摊代价: {da.amortized_cost():.4f} (应该 <= 2.0)')

    # TODO: 验证你的结果——均摊代价是否 <= 2？
    assert da.amortized_cost() <= 2.0, '均摊代价应 <= 2'
    print('✅ 均摊代价验证通过！')


# ============================================================================
# 任务 3：主定理练习
# ============================================================================

def task3_master_theorem():
    """使用主定理分析以下递推式，填写你的答案。

    递推式：T(n) = a * T(n/b) + f(n)

    TODO: 对每个递推式，确定 a, b, log_b(a)，以及属于哪种情形（1/2/3）
    """
    print('\n========== 任务 3：主定理练习 ==========')

    problems = [
        # (描述, a, b, f_n_description)
        ('归并排序: T(n) = 2T(n/2) + n', None, None, None),
        ('二分查找: T(n) = T(n/2) + 1', None, None, None),
        ('T(n) = 3T(n/2) + n²', None, None, None),
        ('T(n) = 4T(n/2) + n', None, None, None),
        ('T(n) = T(2n/3) + 1', None, None, None),
    ]

    print('请分析以下递推式（填写 a, b, log_b(a), 所属情形）：')
    for desc, _, _, _ in problems:
        print(f'  {desc}')
    print('\n提示：')
    print('  归并排序: a=2, b=2, log₂2=1, f(n)=n → 情形 2 → Θ(n log n)')
    print('  二分查找: a=1, b=2, log₂1=0, f(n)=1 → 情形 2 → Θ(log n)')
    print('  T(n)=3T(n/2)+n²: a=3, b=2, log₂3≈1.585, f(n)=n² → 情形 3 → Θ(n²)')
    print('  T(n)=4T(n/2)+n: a=4, b=2, log₂4=2, f(n)=n → 情形 1 → Θ(n²)')
    print('  T(n)=T(2n/3)+1: a=1, b=3/2, log_{3/2}1=0, f(n)=1 → 情形 2 → Θ(log n)')


# ============================================================================
# 任务 4（Bonus）：实现势能法分析
# ============================================================================

def task4_potential_method():
    """使用势能法分析动态数组（Bonus 任务）

    势能函数定义：Φ = 2 * size - capacity
    均摊代价：ĉ_i = c_i + Φ(D_i) - Φ(D_{i-1})

    你需要验证：当数组不满时，ĉ_i = 3（常数）
                 当数组满时（需要扩容），ĉ_i = 3（摊销后）
    """
    print('\n========== 任务 4（Bonus）：势能法分析 ==========')

    def potential(size, capacity):
        """TODO: 计算势能函数 Φ = 2 * size - capacity"""
        return 2 * size - capacity

    # 模拟 append 序列并计算势能法下的均摊代价
    capacity = 1
    size = 0
    amortized_costs = []

    for i in range(1, 33):
        phi_before = potential(size, capacity)
        actual_cost = 1  # 插入的基本代价

        if size == capacity:
            # 需要扩容
            actual_cost += size  # 复制 size 个旧元素
            capacity *= 2

        size += 1
        phi_after = potential(size, capacity)

        # TODO: 计算均摊代价 ĉ_i = actual_cost + phi_after - phi_before
        amortized_cost = actual_cost + phi_after - phi_before
        amortized_costs.append(amortized_cost)

    print(f'势能法分析结果（前 32 次 append）：')
    print(f'  均摊代价列表: {amortized_costs[:8]}...')
    print(f'  所有均摊代价: {set(amortized_costs)}')
    print(f'  最大均摊代价: {max(amortized_costs)}')
    print(f'  最小均摊代价: {min(amortized_costs)}')

    # 验证：均摊代价应该始终是常数（3）
    assert all(abs(c - 3.0) < 0.01 for c in amortized_costs), \
        f'均摊代价应为 3，实际值: {set(amortized_costs)}'
    print('✅ 势能法验证通过：每次 append 的均摊代价恒为 3！')


# ============================================================================
# 主函数
# ============================================================================

def main():
    print('=' * 60)
    print('  复杂度分析 — 练习程序')
    print('=' * 60)

    analyze_complexity()
    task2_simulate()

    # 检查均摊代价（你的实现是否正确？）
    da = SimulatedDynamicArray()
    for i in range(1000):
        da.append(i)
    expected_amortized = da.total_copies / 1000.0
    print(f'\n1000 次 append 后的均摊代价: {expected_amortized:.6f}')

    task3_master_theorem()
    task4_potential_method()

    print('\n✅ 所有练习已完成！')


if __name__ == '__main__':
    main()
