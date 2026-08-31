# -*- coding: utf-8 -*-
"""
===============================================================================
algo05_heap_unionfind_skiplist/code/exercise.py — 堆、并查集与跳跃表练习
===============================================================================
练习目标：
  1. 实现最大堆（MaxHeap）的 push/pop
  2. 实现并查集的 find（路径压缩）和 union（按秩合并）
  3. 使用并查集求解连通分量数量
  4. 完成跳跃表的部分实现

运行方式：python exercise.py
===============================================================================
"""

import random


# ============================================================================
# 任务 1：实现最大堆
# ============================================================================

class MaxHeap:
    """最大堆 —— 根节点是最大值"""

    def __init__(self):
        self._data = []

    # TODO: 实现 push
    def push(self, item):
        self._data.append(item)
        self._sift_up(len(self._data) - 1)

    # TODO: 实现 pop
    def pop(self):
        if not self._data:
            raise IndexError('堆空')
        if len(self._data) == 1:
            return self._data.pop()
        root = self._data[0]
        self._data[0] = self._data.pop()
        self._sift_down(0)
        return root

    def _sift_up(self, idx):
        """上浮：大于父节点时交换"""
        parent = (idx - 1) // 2
        while idx > 0 and self._data[idx] > self._data[parent]:
            self._data[idx], self._data[parent] = self._data[parent], self._data[idx]
            idx = parent
            parent = (idx - 1) // 2

    def _sift_down(self, idx):
        """下沉：与较大的子节点交换"""
        n = len(self._data)
        while True:
            largest = idx
            left = 2 * idx + 1
            right = 2 * idx + 2
            if left < n and self._data[left] > self._data[largest]:
                largest = left
            if right < n and self._data[right] > self._data[largest]:
                largest = right
            if largest == idx:
                break
            self._data[idx], self._data[largest] = self._data[largest], self._data[idx]
            idx = largest

    def __repr__(self):
        return f'MaxHeap({self._data})'


# ============================================================================
# 任务 2：补全并查集
# ============================================================================

class UnionFind:
    """并查集（待补全）"""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.set_count = n

    # TODO: 实现 find（带路径压缩）
    def find(self, x):
        # TODO: 如果 parent[x] != x，递归查找并路径压缩
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    # TODO: 实现 union（按秩合并）
    def union(self, x, y):
        # TODO: 找到 x, y 的根，按秩合并
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1
        self.set_count -= 1
        return True


# ============================================================================
# 任务 3：使用并查集统计连通分量
# ============================================================================

def count_connected_components(n, edges):
    """给定 n 个节点和边列表，返回连通分量的数量

    参数:
      n: 节点数量 (0 到 n-1)
      edges: [(u, v), ...] 无向边列表

    返回:
      连通分量的数量
    """
    # TODO: 你的代码
    uf = UnionFind(n)
    for u, v in edges:
        uf.union(u, v)
    return uf.set_count


# ============================================================================
# 任务 4（Bonus）：补全跳跃表的 search 和 insert
# ============================================================================

class SkipNode:
    def __init__(self, value, level):
        self.value = value
        self.forward = [None] * (level + 1)


class SkipList:
    MAX_LEVEL = 8
    P = 0.5

    def __init__(self):
        self.head = SkipNode(float('-inf'), self.MAX_LEVEL)
        self.level = 0
        self._size = 0

    def _random_level(self):
        level = 0
        while random.random() < self.P and level < self.MAX_LEVEL:
            level += 1
        return level

    # TODO: 实现 search
    def search(self, target):
        cur = self.head
        for i in range(self.level, -1, -1):
            while cur.forward[i] and cur.forward[i].value < target:
                cur = cur.forward[i]
        cur = cur.forward[0]
        return cur is not None and cur.value == target

    # TODO: 实现 insert
    def insert(self, value):
        update = [None] * (self.MAX_LEVEL + 1)
        cur = self.head
        for i in range(self.level, -1, -1):
            while cur.forward[i] and cur.forward[i].value < value:
                cur = cur.forward[i]
            update[i] = cur

        new_level = self._random_level()
        if new_level > self.level:
            for i in range(self.level + 1, new_level + 1):
                update[i] = self.head
            self.level = new_level

        new_node = SkipNode(value, new_level)
        for i in range(new_level + 1):
            new_node.forward[i] = update[i].forward[i]
            update[i].forward[i] = new_node
        self._size += 1

    def to_list(self):
        result = []
        cur = self.head.forward[0]
        while cur:
            result.append(cur.value)
            cur = cur.forward[0]
        return result


# ============================================================================
# 主函数
# ============================================================================

def main():
    print('=' * 60)
    print('  堆、并查集与跳跃表 — 练习程序')
    print('=' * 60)

    # 测试任务 1
    print('\n--- 任务 1：最大堆 ---')
    mh = MaxHeap()
    for x in [3, 1, 4, 1, 5, 9, 2]:
        mh.push(x)
    print(f'push 后: {mh}')
    sorted_desc = [mh.pop() for _ in range(len(mh._data))]
    print(f'弹出顺序（降序）: {sorted_desc}')
    assert sorted_desc == sorted([3, 1, 4, 1, 5, 9, 2], reverse=True)
    print('✅ 任务 1 通过！')

    # 测试任务 2
    print('\n--- 任务 2：并查集 ---')
    uf = UnionFind(6)
    uf.union(0, 1); uf.union(1, 2)
    uf.union(3, 4)
    print(f'find(0)={uf.find(0)}, find(2)={uf.find(2)} (应相等)')
    print(f'find(3)={uf.find(3)}, find(4)={uf.find(4)} (应相等)')
    print(f'find(0)!={uf.find(3)} (应不相等)')
    assert uf.find(0) == uf.find(2)
    assert uf.find(0) != uf.find(3)
    print('✅ 任务 2 通过！')

    # 测试任务 3
    print('\n--- 任务 3：连通分量 ---')
    edges = [(0, 1), (1, 2), (3, 4)]
    count = count_connected_components(6, edges)
    print(f'n=6, edges={edges} → 连通分量数 = {count}')
    assert count == 3  # {0,1,2}, {3,4}, {5}
    print('✅ 任务 3 通过！')

    # 测试任务 4
    print('\n--- 任务 4：跳跃表 ---')
    sl = SkipList()
    for v in [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]:
        sl.insert(v)
    print(f'插入后: {sl.to_list()}')
    assert sl.search(4) and sl.search(9) and not sl.search(7)
    print('✅ 任务 4 通过！')

    print('\n🎉 所有练习已完成！')


if __name__ == '__main__':
    main()
