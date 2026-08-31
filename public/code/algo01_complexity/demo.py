# -*- coding: utf-8 -*-
"""
===============================================================================
algo01_complexity/code/demo.py — 复杂度分析与渐进记号演示
===============================================================================
本程序演示复杂度分析的核心概念，包括：
  1. 常见复杂度等级的代码实现与计时对比
  2. 复杂度增长曲线的可视化
  3. 动态数组的均摊分析
  4. 主定理的数值验证

通过本演示，你将理解：
  - O(1)、O(log n)、O(n)、O(n log n)、O(n²) 在实际运行时间上的差异
  - 动态数组扩容为什么均摊 O(1)
  - 主定理如何预测分治算法的复杂度

运行方式：python demo.py
依赖：matplotlib（仅用于可视化）
===============================================================================
"""

import os
import time
import math
import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False

# 图片保存目录
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)


# ============================================================================
# 第一部分：常见复杂度等级的算法实现
# ============================================================================

def benchmark_constant(n):
    """O(1) — 常数时间：数组索引"""
    arr = list(range(n))
    start = time.perf_counter()
    _ = arr[n // 2]       # 一次索引操作
    return time.perf_counter() - start


def benchmark_logarithmic(n):
    """O(log n) — 对数时间：二分查找"""
    arr = list(range(n))
    target = n - 1
    start = time.perf_counter()
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            break
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return time.perf_counter() - start


def benchmark_linear(n):
    """O(n) — 线性时间：遍历求和"""
    arr = list(range(n))
    start = time.perf_counter()
    total = 0
    for x in arr:
        total += x
    return time.perf_counter() - start


def benchmark_n_log_n(n):
    """O(n log n) — 线性对数时间：归并排序"""
    arr = [random.randint(0, n) for _ in range(n)]
    start = time.perf_counter()

    def merge_sort(a):
        if len(a) <= 1:
            return a
        mid = len(a) // 2
        left = merge_sort(a[:mid])
        right = merge_sort(a[mid:])
        # 归并两个有序数组
        result = []
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i]); i += 1
            else:
                result.append(right[j]); j += 1
        result.extend(left[i:])
        result.extend(right[j:])
        return result

    _ = merge_sort(arr)
    return time.perf_counter() - start


def benchmark_quadratic(n):
    """O(n²) — 平方时间：冒泡排序"""
    arr = [random.randint(0, n) for _ in range(n)]
    start = time.perf_counter()
    for i in range(len(arr)):
        swapped = False
        for j in range(len(arr) - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    return time.perf_counter() - start


# ============================================================================
# 第二部分：复杂度增长曲线可视化
# ============================================================================

def plot_complexity_curves():
    """绘制常见复杂度函数的理论增长曲线（对数坐标）"""
    n = list(range(1, 101))
    functions = {
        'O(1)':      [1 for _ in n],
        'O(log n)':  [math.log2(x) for x in n],
        'O(n)':      n,
        'O(n log n)': [x * math.log2(x) if x > 0 else 0 for x in n],
        'O(n²)':     [x**2 for x in n],
        'O(2ⁿ)':     [2**min(x, 20) for x in n],  # 截断避免溢出
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：线性坐标
    for label, values in functions.items():
        ax1.plot(n, values, label=label, linewidth=2)
    ax1.set_xlabel('输入规模 n', fontsize=12)
    ax1.set_ylabel('操作次数 (理论值)', fontsize=12)
    ax1.set_title('复杂度增长曲线（线性坐标）', fontsize=14)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 5000)

    # 右图：对数坐标
    for label, values in functions.items():
        ax2.loglog(n, values, label=label, linewidth=2)
    ax2.set_xlabel('输入规模 n (log)', fontsize=12)
    ax2.set_ylabel('操作次数 (log)', fontsize=12)
    ax2.set_title('复杂度增长曲线（双对数坐标）', fontsize=14)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'complexity_growth_curves.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[图片保存] {out}')
    return out


# ============================================================================
# 第三部分：实测时间对比
# ============================================================================

def plot_benchmark_results():
    """对不同复杂度的算法进行实际计时，并绘制对比图"""
    sizes = [100, 200, 500, 1000, 2000, 5000]
    benchmarks = {
        'O(log n) - 二分查找': benchmark_logarithmic,
        'O(n) - 线性求和': benchmark_linear,
        'O(n log n) - 归并排序': benchmark_n_log_n,
        'O(n²) - 冒泡排序': benchmark_quadratic,
    }

    results = {}
    for name, func in benchmarks.items():
        times = []
        for n in sizes:
            # 多次运行取平均以减小噪声
            runs = [func(n) for _ in range(3)]
            times.append(sum(runs) / len(runs))
        results[name] = times
        print(f'[计时] {name}:', {s: f'{t*1000:.3f}ms' for s, t in zip(sizes, times)})

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, times in results.items():
        ax.plot(sizes, times, 'o-', label=name, linewidth=2, markersize=6)

    ax.set_xlabel('输入规模 n', fontsize=12)
    ax.set_ylabel('运行时间 (秒)', fontsize=12)
    ax.set_title('不同复杂度算法的实际运行时间对比', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'benchmark_comparison.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[图片保存] {out}')
    return out


# ============================================================================
# 第四部分：动态数组均摊分析
# ============================================================================

class DynamicArray:
    """从零实现动态数组（类似 Python list），展示均摊 O(1) 的追加操作"""

    def __init__(self):
        self._capacity = 1          # 当前容量
        self._size = 0              # 当前元素个数
        self._data = [None] * 1    # 底层定长数组
        self._copy_count = 0       # 统计元素复制次数（用于均摊分析）

    def append(self, value):
        """追加元素，容量不足时自动扩容为原来的 2 倍"""
        if self._size == self._capacity:
            # 扩容：分配新数组，复制旧元素
            self._resize(self._capacity * 2)
        self._data[self._size] = value
        self._size += 1

    def _resize(self, new_capacity):
        """扩容操作：复制到新数组（这是 O(n) 的"贵"操作）"""
        new_data = [None] * new_capacity
        for i in range(self._size):
            new_data[i] = self._data[i]
            self._copy_count += 1  # 统计每次复制
        self._data = new_data
        self._capacity = new_capacity

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        if index < 0 or index >= self._size:
            raise IndexError('索引越界')
        return self._data[index]


def demonstrate_amortized_analysis():
    """演示动态数组的均摊分析"""
    print('\n========== 动态数组均摊分析 ==========')

    da = DynamicArray()
    copy_log = []  # 记录每次 append 后累积的复制次数

    for i in range(1, 33):
        da.append(i)
        copy_log.append((i, da._copy_count, da._capacity))

    print(f'追加 {len(da)} 个元素后的统计：')
    print(f'  总复制次数: {da._copy_count}')
    print(f'  均摊每次 append 复制次数: {da._copy_count / len(da):.2f}')
    print(f'  最终容量: {da._capacity}')
    print(f'  理论预估值: 2 * 32 = 64，实际 {da._copy_count} <= 2n')

    # 可视化均摊分析
    sizes = [entry[0] for entry in copy_log]
    copies = [entry[1] for entry in copy_log]
    capacities = [entry[2] for entry in copy_log]
    amortized = [c / s for s, c in zip(sizes, copies)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：容量呈阶梯状增长（2 的幂次）
    ax1.step(sizes, capacities, where='post', linewidth=2, color='#2196F3')
    ax1.plot(sizes, sizes, '--', color='gray', alpha=0.5, label='n (对角线)')
    ax1.set_xlabel('append 次数 n', fontsize=12)
    ax1.set_ylabel('容量', fontsize=12)
    ax1.set_title('动态数组容量增长（每次扩容 ×2）', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 右图：均摊复制次数趋近于 2（常数）
    ax2.plot(sizes, amortized, 'o-', linewidth=2, color='#4CAF50')
    ax2.axhline(y=2, color='red', linestyle='--', alpha=0.7, label='理论上限 ≈ 2')
    ax2.set_xlabel('append 次数 n', fontsize=12)
    ax2.set_ylabel('均摊复制次数 (累积复制 / n)', fontsize=12)
    ax2.set_title('均摊分析：每次 append 均摊 O(1)', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'amortized_dynamic_array.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[图片保存] {out}')


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主流程：运行所有演示"""
    print('=' * 60)
    print('  复杂度分析与渐进记号 — 演示程序')
    print('=' * 60)

    # 1. 复杂度增长曲线
    print('\n--- 绘制复杂度理论增长曲线 ---')
    plot_complexity_curves()

    # 2. 实测计时对比
    print('\n--- 实际运行时间计时 ---')
    plot_benchmark_results()

    # 3. 均摊分析
    demonstrate_amortized_analysis()

    print('\n✅ 所有演示已完成！生成的图片保存在 images/ 目录中。')


if __name__ == '__main__':
    main()
