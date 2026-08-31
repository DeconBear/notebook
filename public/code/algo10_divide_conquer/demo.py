# -*- coding: utf-8 -*-
"""
algo10 递归、分治与二分 — 演示代码
==================================
功能：归并排序 + 逆序对计数、快速排序（三种 pivot 策略对比）、
      二分搜索（含 lower/upper bound）、三分搜索、牛顿法求根。

每个函数都有中文 docstring，每行逻辑代码都有中文注释。
运行方式：在 algo10_divide_conquer/ 目录下执行 python code/demo.py
"""

import os
import random
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False

_HERE = os.path.dirname(os.path.abspath(__file__))
_IMAGES = os.path.join(_HERE, '..', 'images')
os.makedirs(_IMAGES, exist_ok=True)


# ============================================================================
# 第一部分：归并排序 + 逆序对计数
# ============================================================================

def merge_sort(arr, left=0, right=None):
    """
    归并排序（不修改原数组，返回新数组）。
    分治步骤：分解(二分) → 解决(递归排序子数组) → 合并(merge)
    """
    if right is None:
        right = len(arr) - 1
    if left >= right:
        return [arr[left]]

    mid = (left + right) // 2
    # 递归排序左右两部分
    left_sorted = merge_sort(arr, left, mid)
    right_sorted = merge_sort(arr, mid + 1, right)
    # 合并两个有序数组
    return merge(left_sorted, right_sorted)


def merge(left, right):
    """合并两个有序数组。两个指针分别遍历，取较小的放入结果。"""
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    # 将剩余元素追加到结果
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def count_inversions(arr):
    """
    使用分治思想统计逆序对数量。
    返回: (sorted_array, inversion_count)
    思路：逆序对 = 左半边逆序对 + 右半边逆序对 + 横跨逆序对
    横跨逆序对在 merge 过程中计算
    """
    n = len(arr)
    if n <= 1:
        return arr[:], 0

    mid = n // 2
    # 递归计算左右半边的逆序对
    left, inv_left = count_inversions(arr[:mid])
    right, inv_right = count_inversions(arr[mid:])
    # 合并并计算横跨逆序对
    merged, inv_cross = merge_and_count(left, right)

    return merged, inv_left + inv_right + inv_cross


def merge_and_count(left, right):
    """
    合并两个有序数组，同时统计横跨逆序对。
    当 right[j] 被放入结果时，left 中剩余的元素都大于 right[j]，
    它们都与 right[j] 构成逆序对。
    """
    result = []
    i = j = 0
    inv_count = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            # 关键：left[i:] 中的所有元素都 > right[j]
            inv_count += len(left) - i
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result, inv_count


# ============================================================================
# 第二部分：快速排序（三种 pivot 策略）
# ============================================================================

def quicksort_lomuto(arr, lo, hi):
    """Lomuto 划分方案的快速排序。pivot = arr[hi]（最后一个元素）。"""
    if lo >= hi:
        return
    p = partition_lomuto(arr, lo, hi)
    quicksort_lomuto(arr, lo, p - 1)
    quicksort_lomuto(arr, p + 1, hi)


def partition_lomuto(arr, lo, hi):
    """
    Lomuto 划分：以 arr[hi] 为 pivot。
    维护 i 指针表示"小元素边界"，j 扫描整个区间。
    最终 pivot 位于 i+1。
    """
    pivot = arr[hi]
    i = lo - 1  # i 指向最后一个 ≤ pivot 的元素
    for j in range(lo, hi):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]  # 交换
    # 将 pivot 放到正确位置
    arr[i + 1], arr[hi] = arr[hi], arr[i + 1]
    return i + 1


def quicksort_randomized(arr, lo, hi):
    """随机化快速排序：随机选择 pivot 避免最坏情况。"""
    if lo >= hi:
        return
    # 随机选一个元素与最后一个交换作为 pivot
    pivot_idx = random.randint(lo, hi)
    arr[pivot_idx], arr[hi] = arr[hi], arr[pivot_idx]
    p = partition_lomuto(arr, lo, hi)
    quicksort_randomized(arr, lo, p - 1)
    quicksort_randomized(arr, p + 1, hi)


def quicksort_hoare(arr, lo, hi):
    """Hoare 划分方案的快速排序（更高效）。"""
    if lo >= hi:
        return
    p = partition_hoare(arr, lo, hi)
    quicksort_hoare(arr, lo, p)
    quicksort_hoare(arr, p + 1, hi)


def partition_hoare(arr, lo, hi):
    """
    Hoare 划分：从两端向中间扫描，交换顺序错误的元素。
    pivot = arr[(lo+hi)//2]（中间元素）。
    """
    pivot = arr[(lo + hi) // 2]
    i = lo - 1
    j = hi + 1
    while True:
        # 从左找第一个 ≥ pivot 的元素
        i += 1
        while arr[i] < pivot:
            i += 1
        # 从右找第一个 ≤ pivot 的元素
        j -= 1
        while arr[j] > pivot:
            j -= 1
        if i >= j:
            return j  # 注意：Hoare 返回 j，不是 pivot 的最终位置
        arr[i], arr[j] = arr[j], arr[i]


# ============================================================================
# 第三部分：二分搜索及其变体
# ============================================================================

def binary_search(arr, target):
    """标准二分搜索：在有序数组中查找 target 的索引，找不到返回 -1。"""
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def lower_bound(arr, target):
    """
    二分下界：返回第一个 >= target 的位置。
    如果所有元素都 < target，返回 len(arr)。
    """
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def upper_bound(arr, target):
    """
    二分上界：返回第一个 > target 的位置。
    如果所有元素都 <= target，返回 len(arr)。
    """
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] <= target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def equal_range(arr, target):
    """返回 target 在有序数组中的起止范围 [first, last)。"""
    lo = lower_bound(arr, target)
    hi = upper_bound(arr, target)
    return lo, hi


def binary_search_answer(check_func, lo, hi, eps=1e-9):
    """
    二分答案：在 [lo, hi] 上二分搜索，找到满足 check_func(mid) 的最小值。
    适用于答案具有单调性的场景（如最大化最小值）。
    check_func(x) 返回 True 表示 x 可行。
    """
    while hi - lo > eps:
        mid = (lo + hi) / 2.0
        if check_func(mid):
            hi = mid  # mid 可行 → 尝试更小的值
        else:
            lo = mid  # mid 不可行 → 需要更大的值
    return lo


# ============================================================================
# 第四部分：三分搜索与牛顿法
# ============================================================================

def ternary_search(func, lo, hi, eps=1e-9):
    """
    三分搜索：在单峰函数 func 的 [lo, hi] 区间找最小值点。
    每次取两个三等分点 m1, m2，比较函数值来缩小区间。
    """
    while hi - lo > eps:
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if func(m1) < func(m2):
            hi = m2  # 极小值在左半区
        elif func(m1) > func(m2):
            lo = m1  # 极小值在右半区
        else:
            lo, hi = m1, m2  # 相等，极值在中间
    return (lo + hi) / 2.0


def newtons_method(f, f_prime, x0, eps=1e-9, max_iter=100):
    """
    牛顿迭代法求 f(x)=0 的根。
    x_{n+1} = x_n - f(x_n) / f'(x_n)
    每次在 (x_n, f(x_n)) 处作切线，切线与 x 轴交点即为 x_{n+1}。
    """
    x = x0
    history = [x]  # 记录迭代轨迹
    for _ in range(max_iter):
        fx = f(x)
        fpx = f_prime(x)
        if abs(fx) < eps:
            break
        if fpx == 0:
            print('  警告: 导数为 0，牛顿法停滞')
            break
        x = x - fx / fpx  # 牛顿迭代公式
        history.append(x)
    return x, history


# ============================================================================
# 第五部分：可视化与基准测试
# ============================================================================

def benchmark_sorting():
    """对比三种排序算法在不同输入规模下的运行时间。"""
    sizes = [100, 500, 1000, 2000, 5000]
    merge_times = []
    quick_times = []
    builtin_times = []

    for n in sizes:
        arr = [random.randint(0, 100000) for _ in range(n)]
        # 归并排序
        arr_copy = arr[:]
        t0 = time.perf_counter()
        merge_sort(arr_copy)
        merge_times.append(time.perf_counter() - t0)

        # 随机化快排
        arr_copy = arr[:]
        t0 = time.perf_counter()
        quicksort_randomized(arr_copy, 0, n - 1)
        quick_times.append(time.perf_counter() - t0)

        # Python 内置排序（Timsort）
        arr_copy = arr[:]
        t0 = time.perf_counter()
        arr_copy.sort()
        builtin_times.append(time.perf_counter() - t0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sizes, merge_times, 'o-', label='Merge Sort', linewidth=2)
    ax.plot(sizes, quick_times, 's-', label='Randomized QuickSort', linewidth=2)
    ax.plot(sizes, builtin_times, '^-', label='Python Timsort', linewidth=2)
    ax.set_xlabel('Input Size (n)', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Sorting Algorithm Benchmark', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(_IMAGES, 'algo10_sort_benchmark.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[可视化] 排序基准测试图已保存: {path}')


# ============================================================================
# 主程序
# ============================================================================

def main():
    print('=' * 60)
    print('algo10 递归、分治与二分 — 演示')
    print('=' * 60)

    # --- 1. 归并排序 ---
    print('\n### 1. 归并排序 ###')
    arr1 = [38, 27, 43, 3, 9, 82, 10]
    print(f'原数组: {arr1}')
    sorted_arr = merge_sort(arr1)
    print(f'排序后: {sorted_arr}')

    # --- 2. 逆序对计数 ---
    print('\n### 2. 逆序对计数 ###')
    test_arrays = [
        [2, 4, 1, 3, 5],      # 逆序对: (2,1), (4,1), (4,3) → 3
        [5, 4, 3, 2, 1],      # 逆序对: C(5,2) = 10
        [1, 2, 3, 4, 5],      # 逆序对: 0（已排序）
    ]
    for arr in test_arrays:
        _, inv = count_inversions(arr)
        print(f'  {arr} → 逆序对数: {inv}')

    # --- 3. 快速排序对比 ---
    print('\n### 3. 快速排序（对比三种 pivot 策略） ###')
    test_arr = [3, 6, 8, 10, 1, 2, 1]
    print(f'原数组: {test_arr}')

    for name, sort_func in [('Lomuto', quicksort_lomuto),
                              ('Randomized', quicksort_randomized),
                              ('Hoare', quicksort_hoare)]:
        arr_copy = test_arr[:]
        sort_func(arr_copy, 0, len(arr_copy) - 1)
        print(f'  {name} 快排结果: {arr_copy}')

    # --- 4. 二分搜索 ---
    print('\n### 4. 二分搜索及其变体 ###')
    sorted_test = [1, 2, 4, 4, 4, 6, 8, 8, 10]
    targets = [4, 5, 8]
    print(f'有序数组: {sorted_test}')
    for t in targets:
        idx = binary_search(sorted_test, t)
        lo = lower_bound(sorted_test, t)
        hi = upper_bound(sorted_test, t)
        rng = equal_range(sorted_test, t)
        print(f'  target={t}: 精确查找={idx}, lower_bound={lo}, '
              f'upper_bound={hi}, range={rng}, 出现次数={rng[1]-rng[0]}')

    # --- 5. 二分答案示例 ---
    print('\n### 5. 二分答案示例：求 √2 ###')
    def can_square_less_than_2(x):
        return x * x >= 2.0  # 检查 x^2 是否 >= 2

    sqrt2 = binary_search_answer(can_square_less_than_2, 0, 2.0)
    print(f'  二分答案求 √2 ≈ {sqrt2:.10f} (真实值: {2.0**0.5:.10f})')

    # --- 6. 三分搜索 ---
    print('\n### 6. 三分搜索：求单峰函数极小值 ###')
    # 示例函数：f(x) = (x-3)^2 + 2，最小值在 x=3 处
    def f_quadratic(x):
        return (x - 3.0) ** 2 + 2.0

    min_x = ternary_search(f_quadratic, -10, 10)
    print(f'  f(x) = (x-3)^2 + 2 在 [-10, 10] 上的最小值点: x={min_x:.6f}')
    print(f'  f({min_x:.6f}) = {f_quadratic(min_x):.10f}')

    # --- 7. 牛顿法 ---
    print('\n### 7. 牛顿迭代法求根 ###')
    # 目标：求 x^3 - 2x - 5 = 0 的根（约 2.09455...）
    def f_newton(x):
        return x**3 - 2*x - 5

    def f_prime_newton(x):
        return 3*x**2 - 2

    root, history = newtons_method(f_newton, f_prime_newton, x0=2.5)
    print(f'  方程: x^3 - 2x - 5 = 0')
    print(f'  初始猜测 x0 = 2.5')
    print(f'  求得的根: x = {root:.10f}')
    print(f'  验证 f(root) = {f_newton(root):.2e}')
    print(f'  迭代历史 ({len(history)} 步):')
    for i, xi in enumerate(history):
        print(f'    x_{i} = {xi:.10f}')

    # --- 8. 基准测试 ---
    print('\n### 8. 排序算法基准测试 ###')
    benchmark_sorting()


if __name__ == '__main__':
    main()
