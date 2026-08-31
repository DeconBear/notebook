# -*- coding: utf-8 -*-
"""
algo14 线段树与树状数组 — 练习代码
==================================
请完成以下 TODO 任务，巩固 BIT 和线段树的理解。
"""


# ============================================================================
# TODO 1: 实现差分 BIT 支持区间更新 + 点查询
# ============================================================================
class DiffBIT:
    """
    差分 BIT：维护差分数组的 BIT，支持区间更新 + 点查询。
    区间 [l, r] + val → diff[l] += val, diff[r+1] -= val
    点查询 arr[i] = sum(diff[1..i])
    """
    def __init__(self, n):
        self.n = n
        self.tree = [0] * (n + 2)  # +2 为了 diff[r+1] 不越界

    # TODO: 实现 add 和 query 方法
    # def _add(self, i, delta):
    #     while i <= self.n:
    #         self.tree[i] += delta
    #         i += i & -i

    # def _query(self, i):
    #     total = 0
    #     while i > 0:
    #         total += self.tree[i]
    #         i -= i & -i
    #     return total

    def range_add(self, l, r, val):
        """区间 [l, r] 所有元素 +val"""
        # TODO: self._add(l, val)
        # TODO: self._add(r + 1, -val)
        pass

    def point_query(self, i):
        """查询 arr[i] 的值"""
        # TODO: return self._query(i)
        pass


# ============================================================================
# TODO 2: 线段树 — 区间最大值（带单点更新）
# ============================================================================
class MaxSegmentTree:
    """
    线段树维护区间最大值（不需要惰性传播，因为最大值不可差分）。
    支持：单点更新、区间最大值查询。
    """
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)
        if self.n > 0:
            self._build(1, 0, self.n - 1, arr)

    def _build(self, p, l, r, arr):
        """构建线段树。"""
        # TODO: 实现
        # if l == r:
        #     self.tree[p] = arr[l]
        #     return
        # mid = (l + r) // 2
        # self._build(p*2, l, mid, arr)
        # self._build(p*2+1, mid+1, r, arr)
        # self.tree[p] = max(self.tree[p*2], self.tree[p*2+1])
        pass

    def update(self, idx, val):
        """单点更新：arr[idx] = val"""
        self._update(1, 0, self.n - 1, idx, val)

    def _update(self, p, l, r, idx, val):
        # TODO: 实现
        pass

    def query_max(self, ql, qr):
        """区间最大值查询"""
        return self._query_max(1, 0, self.n - 1, ql, qr)

    def _query_max(self, p, l, r, ql, qr):
        # TODO: 实现
        pass


# ============================================================================
# TODO 3: BIT 上的逆序对计数
# ============================================================================
def count_inversions_bit(arr):
    """
    使用 BIT 统计逆序对。遍历数组，对每个元素 arr[i]：
    1. 查询 BIT 中有多少元素 > arr[i]（即 i - rank(arr[i])）
    2. 将 arr[i] 插入 BIT
    需要先对值进行离散化（坐标压缩）。

    参数:
        arr: 整数数组
    返回:
        逆序对总数
    """
    # TODO: 1. 对 arr 进行离散化（sorted(set(arr)) → 排名）
    # TODO: 2. 从右到左遍历（或从左到右，使用不同的计数方式）
    # TODO: 3. 使用 BIT 查询 + 插入
    return 0


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == '__main__':
    print('=' * 50)
    print('algo14 线段树与树状数组 — 练习')
    print('=' * 50)

    # 测试 TODO 1: 差分 BIT
    print('\n--- TODO 1: 差分 BIT — 区间更新 + 点查询 ---')
    db = DiffBIT(10)
    db.range_add(2, 5, 3)
    db.range_add(4, 7, 2)
    for i in range(1, 9):
        print(f'  arr[{i}] = {db.point_query(i)}')
    # 期望: arr[1]=0, arr[2]=3, arr[3]=3, arr[4]=5, arr[5]=5, arr[6]=2, arr[7]=2, arr[8]=0

    # 测试 TODO 2: 区间最大值线段树
    print('\n--- TODO 2: 线段树 — 区间最大值 ---')
    arr = [4, 2, 7, 1, 9, 3, 6]
    mst = MaxSegmentTree(arr)
    print(f'  数组: {arr}')
    print(f'  max[1..4] = {mst.query_max(1, 4)} (期望: 9)')
    mst.update(3, 10)
    print(f'  更新 arr[3]=10 后 max[1..4] = {mst.query_max(1, 4)} (期望: 10)')

    # 测试 TODO 3: BIT 逆序对
    print('\n--- TODO 3: BIT 逆序对计数 ---')
    test_arrs = [[2, 4, 1, 3, 5], [5, 4, 3, 2, 1], [1, 2, 3, 4, 5]]
    for a in test_arrs:
        print(f'  {a} → 逆序对: {count_inversions_bit(a)}')

    print('\n提示: 请补全上方所有 TODO 函数后再运行测试。')
