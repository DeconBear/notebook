# -*- coding: utf-8 -*-
"""
===============================================================================
algo02_arrays_linkedlist_hash/code/demo.py — 数组、链表与哈希表从零实现
===============================================================================
本演示从零实现四种核心数据结构：
  1. 动态数组（Dynamic Array）—— 2倍扩容策略，均摊 O(1) append
  2. 单链表 & 双链表 —— 插入、删除、查找操作
  3. 哈希表 —— 除法哈希 + 链地址法，支持扩容
  4. LRU 缓存 —— 哈希表 + 双链表的经典组合

通过本演示，你将理解：
  - 动态数组的底层机制和均摊分析
  - 链表的指针操作细节
  - 哈希表的冲突解决和 rehash
  - LRU 缓存的 O(1) 设计原理

运行方式：python demo.py
依赖：matplotlib（仅用于可视化）
===============================================================================
"""

import os
import time
import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)


# ============================================================================
# 第一部分：动态数组（Dynamic Array）
# ============================================================================

class DynamicArray:
    """从零实现的动态数组，类似 Python list 的底层机制"""

    def __init__(self):
        self._capacity = 1          # 底层数组的容量
        self._size = 0              # 当前元素个数
        self._data = [None] * 1    # 底层定长数组
        self._copy_count = 0       # 统计总复制次数（用于分析）

    def append(self, value):
        """追加元素，容量不足时扩容为原来的 2 倍"""
        if self._size == self._capacity:
            self._resize(self._capacity * 2)
        self._data[self._size] = value
        self._size += 1

    def _resize(self, new_capacity):
        """扩容：分配新数组并复制旧元素"""
        new_data = [None] * new_capacity
        for i in range(self._size):
            new_data[i] = self._data[i]
            self._copy_count += 1
        self._data = new_data
        self._capacity = new_capacity

    def insert(self, index, value):
        """在指定位置插入元素（O(n) 操作）"""
        if index < 0 or index > self._size:
            raise IndexError('索引越界')
        if self._size == self._capacity:
            self._resize(self._capacity * 2)
        # 将 index 之后的元素向后移动一位
        for i in range(self._size, index, -1):
            self._data[i] = self._data[i - 1]
        self._data[index] = value
        self._size += 1

    def delete(self, index):
        """删除指定位置的元素（O(n) 操作）"""
        if index < 0 or index >= self._size:
            raise IndexError('索引越界')
        # 将 index 之后的元素向前移动一位
        for i in range(index, self._size - 1):
            self._data[i] = self._data[i + 1]
        self._data[self._size - 1] = None
        self._size -= 1

    def __getitem__(self, index):
        if index < 0 or index >= self._size:
            raise IndexError('索引越界')
        return self._data[index]

    def __setitem__(self, index, value):
        if index < 0 or index >= self._size:
            raise IndexError('索引越界')
        self._data[index] = value

    def __len__(self):
        return self._size

    def __repr__(self):
        return f'DynamicArray({[self._data[i] for i in range(self._size)]})'


# ============================================================================
# 第二部分：链表（Linked List）
# ============================================================================

class SinglyNode:
    """单链表节点"""
    def __init__(self, data):
        self.data = data
        self.next = None


class SinglyLinkedList:
    """单链表（带头节点）"""

    def __init__(self):
        self.head = None
        self._size = 0

    def prepend(self, data):
        """头部插入 —— O(1)"""
        new_node = SinglyNode(data)
        new_node.next = self.head
        self.head = new_node
        self._size += 1

    def append(self, data):
        """尾部追加 —— O(n)（无尾指针时需遍历）"""
        new_node = SinglyNode(data)
        if self.head is None:
            self.head = new_node
        else:
            cur = self.head
            while cur.next:
                cur = cur.next
            cur.next = new_node
        self._size += 1

    def find(self, data):
        """查找 —— O(n)"""
        cur = self.head
        while cur:
            if cur.data == data:
                return cur
            cur = cur.next
        return None

    def delete(self, data):
        """删除第一个值为 data 的节点 —— O(n)"""
        if self.head is None:
            return False
        # 删除头节点
        if self.head.data == data:
            self.head = self.head.next
            self._size -= 1
            return True
        # 删除非头节点
        cur = self.head
        while cur.next:
            if cur.next.data == data:
                cur.next = cur.next.next
                self._size -= 1
                return True
            cur = cur.next
        return False

    def to_list(self):
        """转换为 Python 列表（便于查看）"""
        result = []
        cur = self.head
        while cur:
            result.append(cur.data)
            cur = cur.next
        return result

    def __len__(self):
        return self._size

    def __repr__(self):
        return f'SinglyLinkedList({self.to_list()})'


class DoublyNode:
    """双链表节点"""
    def __init__(self, data):
        self.data = data
        self.prev = None
        self.next = None


class DoublyLinkedList:
    """双链表（带头节点和尾节点）"""

    def __init__(self):
        self.head = None
        self.tail = None
        self._size = 0

    def prepend(self, data):
        """头部插入 —— O(1)"""
        new_node = DoublyNode(data)
        if self.head is None:
            self.head = self.tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
        self._size += 1

    def append(self, data):
        """尾部追加 —— O(1)（有尾指针！）"""
        new_node = DoublyNode(data)
        if self.tail is None:
            self.head = self.tail = new_node
        else:
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node
        self._size += 1

    def remove_node(self, node):
        """删除指定节点 —— O(1)（不需要遍历找前驱！）"""
        if node.prev:
            node.prev.next = node.next
        else:
            self.head = node.next
        if node.next:
            node.next.prev = node.prev
        else:
            self.tail = node.prev
        self._size -= 1

    def pop_tail(self):
        """弹出尾部节点 —— O(1)"""
        if self.tail is None:
            return None
        node = self.tail
        self.remove_node(node)
        return node

    def to_list(self):
        result = []
        cur = self.head
        while cur:
            result.append(cur.data)
            cur = cur.next
        return result

    def __len__(self):
        return self._size

    def __repr__(self):
        return f'DoublyLinkedList({self.to_list()})'


# ============================================================================
# 第三部分：哈希表（Hash Table）
# ============================================================================

class HashTable:
    """哈希表：除法哈希 + 链地址法"""

    def __init__(self, initial_capacity=8, load_factor_threshold=0.75):
        self._capacity = initial_capacity
        self._size = 0
        self._buckets = [[] for _ in range(initial_capacity)]
        self._load_factor_threshold = load_factor_threshold

    def _hash(self, key):
        """除法哈希：key mod capacity"""
        return hash(key) % self._capacity

    def _load_factor(self):
        """当前负载因子"""
        return self._size / self._capacity

    def put(self, key, value):
        """插入或更新键值对"""
        # 检查是否需要扩容
        if self._load_factor() >= self._load_factor_threshold:
            self._resize(self._capacity * 2)

        bucket_index = self._hash(key)
        bucket = self._buckets[bucket_index]

        # 检查 key 是否已存在，若是则更新
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return

        # key 不存在，追加到桶中
        bucket.append((key, value))
        self._size += 1

    def get(self, key):
        """获取键对应的值，不存在则返回 None"""
        bucket_index = self._hash(key)
        for k, v in self._buckets[bucket_index]:
            if k == key:
                return v
        return None

    def remove(self, key):
        """删除键值对"""
        bucket_index = self._hash(key)
        bucket = self._buckets[bucket_index]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket.pop(i)
                self._size -= 1
                return True
        return False

    def _resize(self, new_capacity):
        """扩容并重新哈希所有条目"""
        old_buckets = self._buckets
        self._capacity = new_capacity
        self._buckets = [[] for _ in range(new_capacity)]
        self._size = 0

        for bucket in old_buckets:
            for key, value in bucket:
                self.put(key, value)  # 重新插入（不会再次触发扩容）

    def __contains__(self, key):
        return self.get(key) is not None

    def __len__(self):
        return self._size

    def __repr__(self):
        items = []
        for bucket in self._buckets:
            for k, v in bucket:
                items.append(f'{k!r}: {v!r}')
        return 'HashTable({' + ', '.join(items) + '})'


# ============================================================================
# 第四部分：LRU 缓存（哈希表 + 双链表）
# ============================================================================

class LRUCache:
    """LRU（Least Recently Used）缓存 —— 哈希表 + 双链表的经典组合

    双链表维护使用顺序：头部 = 最近使用 (MRU)，尾部 = 最久未使用 (LRU)
    哈希表提供 O(1) 的键 → 节点映射
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}  # key → DoublyNode
        self.dll = DoublyLinkedList()  # 双链表维护访问顺序

    def get(self, key):
        """获取键对应的值，并将该条目移到头部（标记为最近使用）"""
        if key not in self.cache:
            return -1
        node = self.cache[key]
        # 将该节点移到链表头部
        self.dll.remove_node(node)
        self.dll.prepend(node.data)  # 在头部插入新节点...
        # 更新 cache 中的映射（因为节点对象变了）
        self.cache[key] = self.dll.head
        return node.data[1]  # data 是 (key, value) 元组

    def put(self, key, value):
        """插入或更新键值对"""
        if key in self.cache:
            # key 已存在：更新值，移到头部
            node = self.cache[key]
            self.dll.remove_node(node)
            self.dll.prepend((key, value))
            self.cache[key] = self.dll.head
        else:
            # key 不存在，需要插入
            if len(self.cache) >= self.capacity:
                # 缓存已满，淘汰最久未使用的尾部节点
                lru_node = self.dll.pop_tail()
                if lru_node:
                    del self.cache[lru_node.data[0]]
            # 新节点插入头部
            self.dll.prepend((key, value))
            self.cache[key] = self.dll.head

    def __repr__(self):
        order = self.dll.to_list()
        return f'LRUCache(capacity={self.capacity}, order={order})'


# ============================================================================
# 第五部分：基准测试与可视化
# ============================================================================

def benchmark_operations():
    """对比数组、单链表、双链表、哈希表在各种操作上的性能"""
    print('\n========== 基准测试：数据结构操作性能 ==========')

    n = 10000
    results = {}

    # --- 数组：随机访问 ---
    arr = DynamicArray()
    for i in range(n):
        arr.append(i)
    start = time.perf_counter()
    for _ in range(1000):
        _ = arr[random.randint(0, n - 1)]
    results['Array 随机访问'] = (time.perf_counter() - start) / 1000

    # --- 单链表：头部插入 ---
    sll = SinglyLinkedList()
    start = time.perf_counter()
    for i in range(n):
        sll.prepend(i)
    results['SinglyLinkedList prepend'] = time.perf_counter() - start

    # --- 双链表：尾部追加 ---
    dll = DoublyLinkedList()
    start = time.perf_counter()
    for i in range(n):
        dll.append(i)
    results['DoublyLinkedList append'] = time.perf_counter() - start

    # --- 哈希表：插入 ---
    ht = HashTable()
    start = time.perf_counter()
    for i in range(n):
        ht.put(f'key_{i}', i)
    results[f'HashTable put ({n} entries)'] = time.perf_counter() - start

    # --- 哈希表：查找 ---
    start = time.perf_counter()
    for _ in range(1000):
        _ = ht.get(f'key_{random.randint(0, n-1)}')
    results['HashTable get (avg)'] = (time.perf_counter() - start) / 1000

    # --- LRU Cache ---
    lru = LRUCache(500)
    start = time.perf_counter()
    for i in range(5000):
        lru.put(i, i * 10)
    results['LRUCache put (5000 ops)'] = time.perf_counter() - start

    print(f'\n{"操作":<35} {"耗时":>15}')
    print('-' * 52)
    for op, t in results.items():
        print(f'{op:<35} {t:>12.6f}s')

    return results


def plot_operation_comparison():
    """可视化数组 vs 链表的随机访问 vs 顺序访问性能"""
    sizes = [100, 500, 1000, 5000, 10000]
    arr_random = []
    arr_seq = []
    linked_seq = []

    for n in sizes:
        # 数组随机访问
        arr = list(range(n))
        start = time.perf_counter()
        for _ in range(10000):
            _ = arr[random.randint(0, n - 1)]
        arr_random.append((time.perf_counter() - start) * 1000)

        # 数组顺序访问
        start = time.perf_counter()
        for _ in range(1000):
            s = 0
            for x in arr:
                s += x
        arr_seq.append((time.perf_counter() - start) * 1000)

        # 链表顺序访问
        sll = SinglyLinkedList()
        for i in range(n):
            sll.append(i)
        start = time.perf_counter()
        for _ in range(1000):
            cur = sll.head
            s = 0
            while cur:
                s += cur.data
                cur = cur.next
        linked_seq.append((time.perf_counter() - start) * 1000)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(sizes, arr_random, 'o-', label='Array 随机访问', linewidth=2)
    ax1.plot(sizes, arr_seq, 's-', label='Array 顺序访问', linewidth=2)
    ax1.plot(sizes, linked_seq, '^-', label='链表 顺序访问', linewidth=2)
    ax1.set_xlabel('n (元素数量)', fontsize=12)
    ax1.set_ylabel('耗时 (ms)', fontsize=12)
    ax1.set_title('数组 vs 链表 — 访问性能对比', fontsize=14)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 哈希表负载因子影响
    load_factors = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    get_times = []
    for lf in load_factors:
        n_entries = int(1000 * lf)
        ht = HashTable(initial_capacity=1000, load_factor_threshold=0.99)
        for i in range(n_entries):
            ht.put(i, i)
        start = time.perf_counter()
        for _ in range(5000):
            ht.get(random.randint(0, n_entries - 1))
        get_times.append((time.perf_counter() - start) / 5000 * 1e6)

    ax2.plot(load_factors, get_times, 'o-', linewidth=2, color='#E91E63')
    ax2.set_xlabel('负载因子 α = n / m', fontsize=12)
    ax2.set_ylabel('平均查找时间 (μs)', fontsize=12)
    ax2.set_title('哈希表：负载因子对查找性能的影响', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'data_structure_benchmark.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[图片保存] {out}')


# ============================================================================
# 主函数
# ============================================================================

def main():
    print('=' * 60)
    print('  数组、链表与哈希表 — 演示程序')
    print('=' * 60)

    # 1. 动态数组演示
    print('\n--- 动态数组 ---')
    da = DynamicArray()
    for i in range(5):
        da.append(i * 10)
    print(f'append 5 个元素后: {da}')
    da.insert(2, 99)
    print(f'在索引 2 插入 99 后: {da}')
    da.delete(3)
    print(f'删除索引 3 后: {da}')
    print(f'总复制次数: {da._copy_count}')

    # 2. 单链表演示
    print('\n--- 单链表 ---')
    sll = SinglyLinkedList()
    sll.append(1); sll.append(2); sll.append(3)
    sll.prepend(0)
    print(f'追加 1,2,3 并头部插入 0 后: {sll}')
    sll.delete(2)
    print(f'删除 2 后: {sll}')

    # 3. 双链表演示
    print('\n--- 双链表 ---')
    dll = DoublyLinkedList()
    dll.append('A'); dll.append('B'); dll.append('C')
    dll.prepend('START')
    print(f'追加 A,B,C 并头部插入 START 后: {dll}')

    # 4. 哈希表演示
    print('\n--- 哈希表 ---')
    ht = HashTable()
    ht.put('apple', 10)
    ht.put('banana', 20)
    ht.put('cherry', 30)
    print(f'插入 3 个条目后: {ht}')
    print(f"get('banana') = {ht.get('banana')}")
    ht.remove('apple')
    print(f"删除 'apple' 后: {ht}")
    print(f'负载因子: {ht._load_factor():.3f}')

    # 5. LRU 缓存演示
    print('\n--- LRU 缓存 ---')
    lru = LRUCache(3)
    lru.put(1, '一'); lru.put(2, '二'); lru.put(3, '三')
    print(f'插入 1,2,3 后: {lru}')
    print(f'get(1) = {lru.get(1)}  # 访问 1，变为最近使用')
    lru.put(4, '四')  # 缓存满，淘汰 2（最久未使用）
    print(f'put(4, 四) 后: {lru}  # 2 被淘汰')

    # 6. 基准测试
    benchmark_operations()
    plot_operation_comparison()

    print('\n✅ 所有演示已完成！图片保存在 images/ 目录中。')


if __name__ == '__main__':
    main()
