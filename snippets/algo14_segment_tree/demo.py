# -*- coding: utf-8 -*-
"""
algo14 线段树与树状数组 — 演示代码
==================================
功能：Fenwick Tree (BIT) — 点更新 + 前缀查询，差分数组区间更新；
      线段树 — 区间求和 + 惰性传播区间更新；
      持久化线段树（概念演示）、权值线段树（顺序统计）。

每个函数都有中文 docstring，每行逻辑代码都有中文注释。
运行方式：在 algo14_segment_tree/ 目录下执行 python code/demo.py
"""

import os
import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False

_HERE = os.path.dirname(os.path.abspath(__file__))
_IMAGES = os.path.join(_HERE, '..', 'images')
os.makedirs(_IMAGES, exist_ok=True)


# ============================================================================
# 第一部分：树状数组（Fenwick Tree / BIT）
# ============================================================================

class FenwickTree:
    """
    树状数组（Binary Indexed Tree）。
    支持：点更新（add）+ 前缀查询（sum），均为 O(log n)。
    """

    def __init__(self, n):
        """初始化大小为 n 的 BIT（索引从 1 开始）。"""
        self.n = n
        self.tree = [0] * (n + 1)

    def add(self, i, delta):
        """
        点更新：在位置 i 加上 delta。
        i += i & -i 沿着"lowbit"路径向上更新所有受影响的区间。
        """
        while i <= self.n:
            self.tree[i] += delta
            i += i & -i  # lowbit: 跳到下一个受影响的节点

    def query(self, i):
        """
        前缀查询：查询 [1, i] 的和。
        i -= i & -i 沿着"lowbit"路径向下收集。
        """
        total = 0
        while i > 0:
            total += self.tree[i]
            i -= i & -i  # lowbit: 跳到下一个不相交的前缀区间
        return total

    def range_sum(self, l, r):
        """区间查询：sum(arr[l..r]) = query(r) - query(l-1)"""
        return self.query(r) - self.query(l - 1)

    @classmethod
    def from_array(cls, arr):
        """从数组构建 BIT，O(n log n)。"""
        n = len(arr)
        bit = cls(n)
        for i, val in enumerate(arr, start=1):
            bit.add(i, val)
        return bit


def demonstrate_fenwick():
    """演示 BIT 的基本操作。"""
    print('\n### BIT 树状数组演示 ###')
    arr = [3, 1, 4, 1, 5, 9, 2, 6]
    bit = FenwickTree.from_array(arr)
    print(f'原始数组: {arr}')

    # 前缀和查询
    for i in range(1, len(arr) + 1):
        print(f'  BIT.query({i}) = {bit.query(i)} '
              f'(真实 prefix sum: {sum(arr[:i])})')

    # 点更新
    bit.add(4, 10)  # 位置 4 加 10
    print(f'  更新: arr[4] += 10')
    print(f'  更新后 BIT.query(5) = {bit.query(5)} '
          f'(期望: {sum(arr[:5]) + 10})')

    # 区间和
    print(f'  sum[2..5] = {bit.range_sum(2, 5)}')


# ============================================================================
# 第二部分：线段树（区间求和 + 惰性传播）
# ============================================================================

class SegmentTree:
    """
    线段树：支持区间查询 + 区间更新（惰性传播）。
    这里以区间求和为例。
    """

    def __init__(self, arr):
        """用数组构建线段树。"""
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)   # 4n 空间足够
        self.lazy = [0] * (4 * self.n)   # 惰性标记数组
        if self.n > 0:
            self._build(1, 0, self.n - 1, arr)

    def _build(self, p, l, r, arr):
        """递归构建线段树。p=节点索引, [l,r]=该节点代表的区间。"""
        if l == r:
            self.tree[p] = arr[l]  # 叶子节点：直接赋值
            return
        mid = (l + r) // 2
        self._build(p * 2, l, mid, arr)       # 递归构建左子树
        self._build(p * 2 + 1, mid + 1, r, arr)  # 递归构建右子树
        self.tree[p] = self.tree[p * 2] + self.tree[p * 2 + 1]  # 聚合

    def _push_down(self, p, l, r):
        """将节点 p 的惰性标记传递给子节点。"""
        if self.lazy[p] != 0:
            mid = (l + r) // 2
            val = self.lazy[p]
            # 更新左子树
            self.tree[p * 2] += val * (mid - l + 1)
            self.lazy[p * 2] += val
            # 更新右子树
            self.tree[p * 2 + 1] += val * (r - mid)
            self.lazy[p * 2 + 1] += val
            # 清除当前节点的惰性标记
            self.lazy[p] = 0

    def update_range(self, ql, qr, val):
        """区间更新：将 [ql, qr] 内所有元素 + val。"""
        self._update_range(1, 0, self.n - 1, ql, qr, val)

    def _update_range(self, p, l, r, ql, qr, val):
        """区间更新的递归实现。"""
        # 当前区间完全在查询区间内 → 直接更新并设置惰性标记
        if ql <= l and r <= qr:
            self.tree[p] += val * (r - l + 1)
            self.lazy[p] += val
            return

        # 先下传惰性标记
        self._push_down(p, l, r)

        mid = (l + r) // 2
        if ql <= mid:
            self._update_range(p * 2, l, mid, ql, qr, val)
        if qr > mid:
            self._update_range(p * 2 + 1, mid + 1, r, ql, qr, val)

        # 更新当前节点的值
        self.tree[p] = self.tree[p * 2] + self.tree[p * 2 + 1]

    def query_range(self, ql, qr):
        """区间查询：查询 [ql, qr] 的和。"""
        return self._query_range(1, 0, self.n - 1, ql, qr)

    def _query_range(self, p, l, r, ql, qr):
        """区间查询的递归实现。"""
        # 完全包含
        if ql <= l and r <= qr:
            return self.tree[p]
        # 不相交
        if qr < l or r < ql:
            return 0

        # 先下传惰性标记
        self._push_down(p, l, r)

        mid = (l + r) // 2
        left_sum = self._query_range(p * 2, l, mid, ql, qr)
        right_sum = self._query_range(p * 2 + 1, mid + 1, r, ql, qr)
        return left_sum + right_sum

    def point_update(self, idx, val):
        """单点更新：设置 arr[idx] = val。"""
        self._point_update(1, 0, self.n - 1, idx, val)

    def _point_update(self, p, l, r, idx, val):
        if l == r:
            self.tree[p] = val
            return
        self._push_down(p, l, r)
        mid = (l + r) // 2
        if idx <= mid:
            self._point_update(p * 2, l, mid, idx, val)
        else:
            self._point_update(p * 2 + 1, mid + 1, r, idx, val)
        self.tree[p] = self.tree[p * 2] + self.tree[p * 2 + 1]


# ============================================================================
# 第三部分：持久化线段树（概念演示 — 版本管理）
# ============================================================================

class PersistentSegTree:
    """
    持久化线段树（简版概念演示）。
    每更新一次，创建一条新路径，保留所有历史版本。
    """

    class Node:
        __slots__ = ('val', 'left', 'right')
        def __init__(self, val=0, left=None, right=None):
            self.val = val
            self.left = left
            self.right = right

    def __init__(self, arr):
        self.roots = []  # 存储每个版本的根节点
        self.n = len(arr)
        root = self._build(0, self.n - 1, arr)
        self.roots.append(root)

    def _build(self, l, r, arr):
        """构建初始版本的线段树。"""
        if l == r:
            return self.Node(val=arr[l])
        mid = (l + r) // 2
        left = self._build(l, mid, arr)
        right = self._build(mid + 1, r, arr)
        return self.Node(val=left.val + right.val, left=left, right=right)

    def update(self, idx, val):
        """创建一个新版本，将 arr[idx] 改为 val。"""
        old_root = self.roots[-1]
        new_root = self._update(old_root, 0, self.n - 1, idx, val)
        self.roots.append(new_root)

    def _update(self, node, l, r, idx, val):
        """递归更新，沿途创建新节点。"""
        if l == r:
            return self.Node(val=val)
        mid = (l + r) // 2
        if idx <= mid:
            new_left = self._update(node.left, l, mid, idx, val)
            return self.Node(val=new_left.val + node.right.val,
                             left=new_left, right=node.right)
        else:
            new_right = self._update(node.right, mid + 1, r, idx, val)
            return self.Node(val=node.left.val + new_right.val,
                             left=node.left, right=new_right)

    def query_version(self, version, ql, qr):
        """查询第 version 个版本的区间和。"""
        return self._query(self.roots[version], 0, self.n - 1, ql, qr)

    def _query(self, node, l, r, ql, qr):
        if node is None:
            return 0
        if ql <= l and r <= qr:
            return node.val
        if qr < l or r < ql:
            return 0
        mid = (l + r) // 2
        return (self._query(node.left, l, mid, ql, qr) +
                self._query(node.right, mid + 1, r, ql, qr))


# ============================================================================
# 第四部分：顺序统计（Order Statistics via BIT）
# ============================================================================

class OrderStatistics:
    """
    使用 BIT 实现顺序统计操作（值域为 [1, max_val]）。
    支持：插入、删除、查询排名、查询第 k 小。
    """

    def __init__(self, max_val):
        self.max_val = max_val
        self.bit = FenwickTree(max_val)

    def insert(self, x):
        """插入一个值为 x 的元素。"""
        self.bit.add(x, 1)

    def remove(self, x):
        """删除一个值为 x 的元素（假设存在）。"""
        self.bit.add(x, -1)

    def rank(self, x):
        """查询小于 x 的元素个数（x 的排名）。"""
        return self.bit.query(x - 1)

    def kth_smallest(self, k):
        """
        查询第 k 小（1-based）的元素。
        使用 BIT 上的二分查找：在值域上二分，用 BIT.query(mid) 判断。
        """
        lo, hi = 1, self.max_val
        while lo < hi:
            mid = (lo + hi) // 2
            if self.bit.query(mid) >= k:
                hi = mid
            else:
                lo = mid + 1
        return lo if self.bit.query(lo) >= k else -1


# ============================================================================
# 第五部分：可视化 BIT 结构
# ============================================================================

def visualize_bit_structure(n=8):
    """可视化 BIT 的树形结构。"""
    fig, ax = plt.subplots(figsize=(10, 5))

    # 绘制 BIT 的每个节点所覆盖的区间
    for i in range(1, n + 1):
        low = i - (i & -i) + 1
        high = i
        ax.barh(i, high - low + 1, left=low - 1, height=0.6,
                color='#42A5F5', edgecolor='white', linewidth=1.5)
        ax.text(low + (high - low) / 2 - 0.5, i,
                f'tree[{i}]=sum[{low}..{high}]',
                ha='center', va='center', fontsize=9, color='white',
                fontweight='bold')

    ax.set_xlabel('Array Index', fontsize=12)
    ax.set_ylabel('BIT Node', fontsize=12)
    ax.set_title('Fenwick Tree (BIT) Structure — Each Node Covers an Interval',
                 fontsize=14, fontweight='bold')
    ax.set_yticks(range(1, n + 1))
    ax.invert_yaxis()
    ax.set_xlim(0, n + 0.5)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(_IMAGES, 'algo14_bit_structure.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[可视化] BIT 结构图已保存: {path}')


# ============================================================================
# 主程序
# ============================================================================

def main():
    print('=' * 60)
    print('algo14 线段树与树状数组 — 演示')
    print('=' * 60)

    # --- 1. BIT (Fenwick Tree) ---
    demonstrate_fenwick()
    visualize_bit_structure(8)

    # --- 2. 线段树（区间更新 + 惰性传播） ---
    print('\n### 线段树 — 区间求和 + 惰性传播 ###')
    arr = [1, 3, 5, 7, 9, 11]
    seg = SegmentTree(arr)
    print(f'原始数组: {arr}')

    print(f'  sum[1..4] = {seg.query_range(1, 4)} (期望: 3+5+7+9=24)')

    # 区间更新：位置 1-3 全部 +2
    seg.update_range(1, 3, 2)
    print(f'  区间[1..3] += 2 后')
    print(f'  sum[1..4] = {seg.query_range(1, 4)}')

    # 区间更新：位置 2-5 全部 +3
    seg.update_range(2, 5, 3)
    print(f'  区间[2..5] += 3 后')
    print(f'  sum[0..5] = {seg.query_range(0, 5)}')

    # --- 3. 持久化线段树 ---
    print('\n### 持久化线段树 — 版本管理 ###')
    arr2 = [1, 2, 3, 4, 5]
    pst = PersistentSegTree(arr2)
    print(f'原始数组: {arr2}')
    print(f'  Version 0, sum[1..3] = {pst.query_version(0, 1, 3)} (期望: 6)')

    pst.update(2, 10)  # arr[2]=10，创建 version 1
    print(f'  更新 arr[2]=10 → Version 1')
    print(f'  Version 0, sum[0..4] = {pst.query_version(0, 0, 4)} (期望: 15)')
    print(f'  Version 1, sum[0..4] = {pst.query_version(1, 0, 4)} (期望: 22)')

    pst.update(4, 20)  # arr[4]=20，创建 version 2
    print(f'  更新 arr[4]=20 → Version 2')
    print(f'  Version 2, sum[0..4] = {pst.query_version(2, 0, 4)} (期望: 37)')
    print(f'  历史 Version 1 不受影响: sum={pst.query_version(1, 0, 4)}')

    # --- 4. 顺序统计 ---
    print('\n### 顺序统计（Order Statistics via BIT） ###')
    os_tree = OrderStatistics(100)
    data = [5, 2, 8, 1, 9, 3]
    for x in data:
        os_tree.insert(x)
    print(f'插入: {data}')
    print(f'  第 1 小: {os_tree.kth_smallest(1)} (期望: 1)')
    print(f'  第 3 小: {os_tree.kth_smallest(3)} (期望: 3)')
    print(f'  第 6 小: {os_tree.kth_smallest(6)} (期望: 9)')
    print(f'  5 的排名: {os_tree.rank(5)} (小于 5 的元素个数, 期望: 3)')

    print('\n' + '=' * 60)
    print('总结：BIT (O(log n) 点更新+前缀和) + 线段树 (区间更新+惰性传播)')
    print('=' * 60)


if __name__ == '__main__':
    main()
