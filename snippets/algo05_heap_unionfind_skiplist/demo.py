# -*- coding: utf-8 -*-
"""
===============================================================================
algo05_heap_unionfind_skiplist/code/demo.py — 堆、并查集与跳跃表从零实现
===============================================================================
本演示从零实现：
  1. 最小堆（MinHeap）—— push/pop/heapify/heap_sort
  2. 并查集（Union-Find）—— 路径压缩 + 按秩合并
  3. 跳跃表（Skip List）—— 概率平衡的多层链表

运行方式：python demo.py
依赖：matplotlib（仅用于可视化）
===============================================================================
"""

import os
import math
import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)


# ============================================================================
# 第一部分：最小堆（MinHeap）
# ============================================================================

class MinHeap:
    """最小堆 —— 基于数组的完全二叉树"""

    def __init__(self, arr=None):
        if arr is None:
            self._data = []
        else:
            self._data = list(arr)
            self._heapify()

    def push(self, item):
        """插入 —— O(log n)"""
        self._data.append(item)
        self._sift_up(len(self._data) - 1)

    def pop(self):
        """弹出最小值 —— O(log n)"""
        if not self._data:
            raise IndexError('堆空')
        if len(self._data) == 1:
            return self._data.pop()
        root = self._data[0]
        self._data[0] = self._data.pop()
        self._sift_down(0)
        return root

    def peek(self):
        """查看最小值 —— O(1)"""
        if not self._data:
            raise IndexError('堆空')
        return self._data[0]

    def _sift_up(self, idx):
        """上浮：当前节点比父节点小时交换"""
        parent = (idx - 1) // 2
        while idx > 0 and self._data[idx] < self._data[parent]:
            self._data[idx], self._data[parent] = self._data[parent], self._data[idx]
            idx = parent
            parent = (idx - 1) // 2

    def _sift_down(self, idx):
        """下沉：当前节点比子节点大时，与较小子节点交换"""
        n = len(self._data)
        while True:
            smallest = idx
            left = 2 * idx + 1
            right = 2 * idx + 2
            if left < n and self._data[left] < self._data[smallest]:
                smallest = left
            if right < n and self._data[right] < self._data[smallest]:
                smallest = right
            if smallest == idx:
                break
            self._data[idx], self._data[smallest] = self._data[smallest], self._data[idx]
            idx = smallest

    def _heapify(self):
        """建堆 —— O(n)：从最后一个非叶节点开始下沉"""
        for i in range(len(self._data) // 2 - 1, -1, -1):
            self._sift_down(i)

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f'MinHeap({self._data})'


def heap_sort(arr):
    """堆排序 —— O(n log n)，原地排序（使用最大堆思想模拟）"""
    heap = MinHeap(arr)
    result = []
    while len(heap) > 0:
        result.append(heap.pop())
    return result


# ============================================================================
# 第二部分：并查集（Union-Find）
# ============================================================================

class UnionFind:
    """并查集 —— 路径压缩 + 按秩合并"""

    def __init__(self, n):
        self.parent = list(range(n))  # 父节点数组
        self.rank = [0] * n           # 秩（树高的上界）
        self._set_count = n           # 集合数量

    def find(self, x):
        """查找 x 的根（带路径压缩）"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        """合并 x 和 y 所在的集合（按秩合并）"""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False  # 已在同一集合

        # 将秩较小的树合并到秩较大的树下
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1

        self._set_count -= 1
        return True

    def connected(self, x, y):
        """判断 x 和 y 是否在同一集合"""
        return self.find(x) == self.find(y)

    @property
    def set_count(self):
        return self._set_count


# ============================================================================
# 第三部分：跳跃表（Skip List）
# ============================================================================

class SkipNode:
    """跳跃表节点"""
    def __init__(self, value, level):
        self.value = value
        # forward[i] = 第 i 层（i=0 是最底层）的前向指针
        self.forward = [None] * (level + 1)


class SkipList:
    """跳跃表 —— 概率平衡的多层链表

    每层约有一半的概率向上"晋升"。
    最大层数: MAX_LEVEL = 16（可容纳 2^16 ≈ 65000 个元素）
    """

    MAX_LEVEL = 16
    P = 0.5  # 晋升概率

    def __init__(self):
        # 哨兵头节点（值不影响比较）
        self.head = SkipNode(float('-inf'), self.MAX_LEVEL)
        self.level = 0  # 当前最大层数
        self._size = 0

    def _random_level(self):
        """随机生成节点的层数（几何分布：P(level=k) = P^k * (1-P)）"""
        level = 0
        while random.random() < self.P and level < self.MAX_LEVEL:
            level += 1
        return level

    def search(self, target):
        """查找 —— O(log n) 期望"""
        cur = self.head
        # 从最高层向下搜索
        for i in range(self.level, -1, -1):
            while cur.forward[i] and cur.forward[i].value < target:
                cur = cur.forward[i]
        # 到达第 0 层后检查下一个节点
        cur = cur.forward[0]
        if cur and cur.value == target:
            return cur
        return None

    def insert(self, value):
        """插入 —— O(log n) 期望"""
        update = [None] * (self.MAX_LEVEL + 1)
        cur = self.head

        # 从高层向下，记录每层的前驱节点
        for i in range(self.level, -1, -1):
            while cur.forward[i] and cur.forward[i].value < value:
                cur = cur.forward[i]
            update[i] = cur

        # 检查是否已存在
        cur = cur.forward[0]
        if cur and cur.value == value:
            return  # 已存在，不重复插入

        # 随机生成新节点的层数
        new_level = self._random_level()
        if new_level > self.level:
            # 新层数超过当前最大层，需要更新 update[level+1 .. new_level] 为 head
            for i in range(self.level + 1, new_level + 1):
                update[i] = self.head
            self.level = new_level

        # 创建新节点并插入
        new_node = SkipNode(value, new_level)
        for i in range(new_level + 1):
            new_node.forward[i] = update[i].forward[i]
            update[i].forward[i] = new_node

        self._size += 1

    def delete(self, value):
        """删除 —— O(log n) 期望"""
        update = [None] * (self.MAX_LEVEL + 1)
        cur = self.head

        for i in range(self.level, -1, -1):
            while cur.forward[i] and cur.forward[i].value < value:
                cur = cur.forward[i]
            update[i] = cur

        cur = cur.forward[0]
        if cur and cur.value == value:
            # 从各层移除节点
            for i in range(self.level + 1):
                if update[i].forward[i] != cur:
                    break
                update[i].forward[i] = cur.forward[i]
            # 更新最大层数
            while self.level > 0 and self.head.forward[self.level] is None:
                self.level -= 1
            self._size -= 1
            return True
        return False

    def to_list(self):
        """返回第 0 层的所有元素（有序列表）"""
        result = []
        cur = self.head.forward[0]
        while cur:
            result.append(cur.value)
            cur = cur.forward[0]
        return result

    def __contains__(self, value):
        return self.search(value) is not None

    def __len__(self):
        return self._size


# ============================================================================
# 第四部分：可视化
# ============================================================================

def plot_heap_sort_demo():
    """可视化堆排序和并查集性能"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 左上：堆排序过程
    ax1 = axes[0, 0]
    arr = [7, 3, 9, 1, 5, 2, 8]
    sorted_arr = heap_sort(arr)
    ax1.bar(range(len(arr)), arr, color='#90CAF9', alpha=0.7, label='原始数组')
    ax1.bar(range(len(sorted_arr)), sorted_arr, color='#4CAF50', alpha=0.7, label='堆排序后')
    ax1.set_title('堆排序 (O(n log n))', fontsize=14)
    ax1.legend(fontsize=9)
    ax1.set_xticks(range(len(arr)))

    # 右上：并查集 Kruskal 演示
    ax2 = axes[0, 1]
    # 简易 Kruskal 演示
    edges = [(1, 0, 2), (2, 1, 3), (3, 0, 3), (5, 2, 4), (6, 3, 4)]
    # 格式: (weight, u, v)
    n_vertices = 5
    uf = UnionFind(n_vertices)
    mst_edges = []
    edges.sort()  # 按权重排序
    for w, u, v in edges:
        if uf.union(u, v):
            mst_edges.append((u, v, w))

    # 绘制图
    import numpy as np
    positions = {0: (1, 2), 1: (0, 1), 2: (2, 1), 3: (1, 0), 4: (3, 0)}
    for (u, v, w), color in zip(edges, ['gray'] * len(edges)):
        x1, y1 = positions[u]; x2, y2 = positions[v]
        alpha = 0.3
        ax2.plot([x1, x2], [y1, y2], 'o-', color='gray', alpha=alpha, linewidth=1)
    for u, v, w in mst_edges:
        x1, y1 = positions[u]; x2, y2 = positions[v]
        ax2.plot([x1, x2], [y1, y2], '-', color='#FF5722', linewidth=2.5)
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax2.text(mid_x, mid_y, str(w), fontsize=10, fontweight='bold')
    for v, (x, y) in positions.items():
        ax2.plot(x, y, 'o', markersize=20, color='#2196F3', markeredgecolor='#1565C0')
        ax2.text(x, y, str(v), ha='center', va='center', fontsize=12, color='white', fontweight='bold')
    ax2.set_title('Kruskal MST (并查集应用)', fontsize=14)
    ax2.axis('off')

    # 左下：build_heap vs push 效率对比
    ax3 = axes[1, 0]
    sizes = [1000, 5000, 10000, 20000, 50000]
    import time
    build_times = []
    push_times = []
    for n in sizes:
        arr = [random.randint(0, n) for _ in range(n)]
        start = time.perf_counter()
        MinHeap(arr)  # heapify
        build_times.append(time.perf_counter() - start)
        start = time.perf_counter()
        h = MinHeap()
        for x in arr:
            h.push(x)
        push_times.append(time.perf_counter() - start)
    ax3.plot(sizes, build_times, 'o-', label='heapify (O(n))', linewidth=2, color='#4CAF50')
    ax3.plot(sizes, push_times, 's-', label='逐个 push (O(n log n))', linewidth=2, color='#FF5722')
    ax3.set_xlabel('n', fontsize=12)
    ax3.set_ylabel('时间 (秒)', fontsize=12)
    ax3.set_title('建堆: heapify O(n) vs push O(n log n)', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # 右下：跳跃表层级分布
    ax4 = axes[1, 1]
    sl = SkipList()
    level_counts = [0] * (SkipList.MAX_LEVEL + 1)
    for _ in range(200):
        level = sl._random_level()
        level_counts[level] += 1

    levels = list(range(SkipList.MAX_LEVEL + 1))
    ax4.bar(levels, level_counts, color='#AB47BC', alpha=0.8)
    # 理论分布曲线
    theory = [200 * (SkipList.P ** i) * (1 - SkipList.P) for i in levels]
    theory[-1] = 200 * (SkipList.P ** SkipList.MAX_LEVEL)  # 截断修正
    ax4.plot(levels, theory, 'ro-', label='理论几何分布', markersize=4)
    ax4.set_xlabel('节点层数', fontsize=12)
    ax4.set_ylabel('节点数量', fontsize=12)
    ax4.set_title('跳跃表：随机层数分布 vs 理论值', fontsize=14)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'heap_uf_skip_demo.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[图片保存] {out}')


# ============================================================================
# 主函数
# ============================================================================

def main():
    print('=' * 60)
    print('  堆、并查集与跳跃表 — 演示程序')
    print('=' * 60)

    # 1. 最小堆
    print('\n--- 最小堆 ---')
    heap = MinHeap()
    for x in [7, 3, 9, 1, 5]:
        heap.push(x)
    print(f'push 7,3,9,1,5: {heap}')
    print(f'peek: {heap.peek()}')
    print(f'pop: {heap.pop()}, 堆变为: {heap}')
    print(f'堆排序 [7,3,9,1,5]: {heap_sort([7,3,9,1,5])}')

    # heapify 演示
    heap2 = MinHeap([7, 3, 9, 1, 5, 2, 8])
    print(f'heapify([7,3,9,1,5,2,8]): {heap2}')

    # 2. 并查集
    print('\n--- 并查集 ---')
    uf = UnionFind(5)
    print(f'初始集合数: {uf.set_count}')
    uf.union(0, 1)
    uf.union(1, 2)
    uf.union(3, 4)
    print(f'union(0,1), union(1,2), union(3,4) 后:')
    print(f'  0 和 2 连通? {uf.connected(0, 2)}')
    print(f'  0 和 3 连通? {uf.connected(0, 3)}')
    print(f'  集合数: {uf.set_count}')

    # 3. 跳跃表
    print('\n--- 跳跃表 ---')
    sl = SkipList()
    for v in [3, 1, 4, 1, 5, 9, 2, 6]:
        sl.insert(v)
    print(f'插入 [3,1,4,1,5,9,2,6] 后: {sl.to_list()}')
    print(f'搜索 5: {"找到" if sl.search(5) else "未找到"}')
    print(f'搜索 7: {"找到" if sl.search(7) else "未找到"}')
    sl.delete(4)
    print(f'删除 4 后: {sl.to_list()}')

    # 4. 可视化
    plot_heap_sort_demo()

    print('\n✅ 所有演示已完成！图片保存在 images/ 目录中。')


if __name__ == '__main__':
    main()
