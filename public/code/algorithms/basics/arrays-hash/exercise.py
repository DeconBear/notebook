# -*- coding: utf-8 -*-
"""
===============================================================================
algo02_arrays_linkedlist_hash/code/exercise.py — 数组、链表与哈希表练习
===============================================================================
本练习文件中，你需要完成以下任务：
  1. 补全动态数组的 insert 和 delete 方法
  2. 实现双链表指定节点的删除操作
  3. 实现哈希表的 get/put/remove（开放地址法版本）
  4. 实现一个简易的 LRU 缓存（Bonus）

运行方式：python exercise.py
===============================================================================
"""

import time


# ============================================================================
# 任务 1：补全动态数组
# ============================================================================

class DynamicArray:
    """动态数组（待补全）"""

    def __init__(self):
        self._capacity = 2
        self._size = 0
        self._data = [None] * 2

    def append(self, value):
        if self._size == self._capacity:
            self._resize(self._capacity * 2)
        self._data[self._size] = value
        self._size += 1

    def _resize(self, new_capacity):
        new_data = [None] * new_capacity
        for i in range(self._size):
            new_data[i] = self._data[i]
        self._data = new_data
        self._capacity = new_capacity

    # TODO: 实现 insert 方法 —— 在指定位置插入元素
    # 1. 检查索引合法性（0 <= index <= size）
    # 2. 若容量已满，先扩容
    # 3. 将 index 及之后的元素向后移动一位
    # 4. 在 index 处赋值，size += 1
    def insert(self, index, value):
        # TODO: 你的代码从这里开始
        if index < 0 or index > self._size:
            raise IndexError('索引越界')
        if self._size == self._capacity:
            self._resize(self._capacity * 2)
        # 将 index 之后的元素向后移动
        for i in range(self._size, index, -1):
            self._data[i] = self._data[i - 1]
        self._data[index] = value
        self._size += 1

    # TODO: 实现 delete 方法 —— 删除指定位置的元素
    # 1. 检查索引合法性（0 <= index < size）
    # 2. 将 index 之后的元素向前移动一位
    # 3. 将最后一个元素清空，size -= 1
    def delete(self, index):
        # TODO: 你的代码从这里开始
        if index < 0 or index >= self._size:
            raise IndexError('索引越界')
        for i in range(index, self._size - 1):
            self._data[i] = self._data[i + 1]
        self._data[self._size - 1] = None
        self._size -= 1

    def __getitem__(self, index):
        if index < 0 or index >= self._size:
            raise IndexError('索引越界')
        return self._data[index]

    def __len__(self):
        return self._size

    def __repr__(self):
        return f'[{", ".join(str(self._data[i]) for i in range(self._size))}]'


# ============================================================================
# 任务 2：实现双链表删除操作
# ============================================================================

class DoublyNode:
    def __init__(self, data):
        self.data = data
        self.prev = None
        self.next = None


class DoublyLinkedList:
    """双链表（待补全）"""

    def __init__(self):
        self.head = None
        self.tail = None
        self._size = 0

    def append(self, data):
        new_node = DoublyNode(data)
        if self.tail is None:
            self.head = self.tail = new_node
        else:
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node
        self._size += 1

    # TODO: 实现 remove_node 方法 —— 在 O(1) 时间内删除指定节点
    # 提示：利用 prev 和 next 指针，考虑四种情况：
    #   1. node 是头节点（node.prev is None）→ 更新 self.head
    #   2. node 是尾节点（node.next is None）→ 更新 self.tail
    #   3. node 在中间 → 修改前后节点的指针
    #   4. 不要忘记 size -= 1
    def remove_node(self, node):
        # TODO: 你的代码从这里开始
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
# 任务 3：实现开放地址法哈希表
# ============================================================================

class OpenAddressingHashTable:
    """开放地址法哈希表 —— 使用线性探测（待补全）

    每个槽存储 (key, value) 或 None（空槽）或 "TOMBSTONE"（已删除标记）
    """

    _TOMBSTONE = object()  # 墓碑标记

    def __init__(self, capacity=8):
        self._capacity = capacity
        self._size = 0
        self._table = [None] * capacity

    def _hash(self, key):
        return hash(key) % self._capacity

    # TODO: 实现 put 方法
    # 1. 从 h = hash(key) 开始，线性探测找到空槽（None 或 TOMBSTONE）
    # 2. 如果 key 已存在，更新 value
    # 3. 如果 key 不存在，插入到找到的空槽，size += 1
    # 4. 检查负载因子，必要时扩容
    def put(self, key, value):
        h = self._hash(key)
        first_tombstone = None

        for i in range(self._capacity):
            idx = (h + i) % self._capacity
            slot = self._table[idx]

            if slot is None:
                # 找到空槽
                insert_idx = first_tombstone if first_tombstone is not None else idx
                self._table[insert_idx] = (key, value)
                self._size += 1
                return
            elif slot == self._TOMBSTONE:
                if first_tombstone is None:
                    first_tombstone = idx
            elif slot[0] == key:
                # key 已存在，更新
                # 如果之前有墓碑标记，移到墓碑位置
                if first_tombstone is not None:
                    self._table[idx] = self._TOMBSTONE
                    self._table[first_tombstone] = (key, value)
                else:
                    self._table[idx] = (key, value)
                return

        # 表满（不应该发生，因为负载因子控制）
        self._resize(self._capacity * 2)
        self.put(key, value)

    # TODO: 实现 get 方法
    # 1. 从 h = hash(key) 开始线性探测
    # 2. 遇到 None（未探测到的位置）说明 key 不存在
    # 3. 遇到匹配的 key 返回 value
    # 4. 跳过 TOMBSTONE
    def get(self, key):
        h = self._hash(key)
        for i in range(self._capacity):
            idx = (h + i) % self._capacity
            slot = self._table[idx]
            if slot is None:
                return None
            if slot != self._TOMBSTONE and slot[0] == key:
                return slot[1]
        return None

    # TODO: 实现 remove 方法
    # 1. 找到 key 对应的槽
    # 2. 将其标记为 TOMBSTONE（不能设为 None！）
    # 3. size -= 1
    def remove(self, key):
        h = self._hash(key)
        for i in range(self._capacity):
            idx = (h + i) % self._capacity
            slot = self._table[idx]
            if slot is None:
                return False
            if slot != self._TOMBSTONE and slot[0] == key:
                self._table[idx] = self._TOMBSTONE
                self._size -= 1
                return True
        return False

    def _resize(self, new_capacity):
        old_table = self._table
        self._capacity = new_capacity
        self._table = [None] * new_capacity
        self._size = 0
        for slot in old_table:
            if slot is not None and slot != self._TOMBSTONE:
                self.put(slot[0], slot[1])

    def __len__(self):
        return self._size

    def __repr__(self):
        items = []
        for slot in self._table:
            if slot is not None and slot != self._TOMBSTONE:
                items.append(f'{slot[0]!r}: {slot[1]!r}')
        return 'OAHTable({' + ', '.join(items) + '})'


# ============================================================================
# 任务 4（Bonus）：实现 LRU 缓存
# ============================================================================

class LRUCache:
    """LRU 缓存（待补全——使用哈希表 + 双链表的组合）

    提示：
      - self.cache: dict，key → DoublyNode，提供 O(1) 查找
      - self.dll: DoublyLinkedList，维护访问顺序（头部=最近使用，尾部=最久未使用）
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.dll = DoublyLinkedList()

    # TODO: 实现 get 方法
    def get(self, key):
        if key not in self.cache:
            return -1
        node = self.cache[key]
        # 1. 将该节点从当前位置移除
        self.dll.remove_node(node)
        # 2. 将该节点插入双链表头部
        self.dll.append(node.data)  # 注意：append 到尾部，但我们需要 prepend...
        # 实际上这里我们需要 prepend 到头部。而上面 DoublyLinkedList 的 append 是加到尾部的。
        # 简化处理：使用 append 作为「最近使用」的末尾
        # 或者正确实现 prepend
        # 这里我们利用 pop_tail 淘汰尾部
        self.cache[key] = self.dll.tail  # 尾部是最新的
        return node.data[1]

    # TODO: 实现 put 方法
    def put(self, key, value):
        if key in self.cache:
            # 更新已存在的 key
            node = self.cache[key]
            self.dll.remove_node(node)
        elif len(self.cache) >= self.capacity:
            # 淘汰最久未使用的（头部）
            if self.dll.head:
                old_node = self.dll.head
                self.dll.remove_node(old_node)
                del self.cache[old_node.data[0]]

        # 插入新节点到尾部的表示（作为最新使用）
        self.dll.append((key, value))
        self.cache[key] = self.dll.tail

    def __repr__(self):
        return f'LRUCache({self.dll.to_list()})'


# ============================================================================
# 主函数
# ============================================================================

def main():
    print('=' * 60)
    print('  数组、链表与哈希表 — 练习程序')
    print('=' * 60)

    # 测试任务 1
    print('\n--- 任务 1：动态数组 ---')
    da = DynamicArray()
    for i in range(5):
        da.append(i * 10)
    print(f'初始: {da}')
    da.insert(2, 99)
    print(f'在索引 2 插入 99: {da}')
    da.delete(3)
    print(f'删除索引 3: {da}')
    assert str(da) == '[0, 10, 99, 30, 40]', f'期望 [0, 10, 99, 30, 40] 实际 {da}'
    print('✅ 任务 1 通过！')

    # 测试任务 2
    print('\n--- 任务 2：双链表 ---')
    dll = DoublyLinkedList()
    dll.append('A'); dll.append('B'); dll.append('C')
    print(f'初始: {dll}')
    # 删除中间节点
    dll.remove_node(dll.head.next)  # 删除 B
    print(f'删除 B 后: {dll}')
    assert dll.to_list() == ['A', 'C'], f'期望 [A, C] 实际 {dll.to_list()}'
    print('✅ 任务 2 通过！')

    # 测试任务 3
    print('\n--- 任务 3：开放地址法哈希表 ---')
    oaht = OpenAddressingHashTable()
    oaht.put('x', 100); oaht.put('y', 200); oaht.put('z', 300)
    print(f'插入 3 个条目: {oaht}')
    print(f'get("y") = {oaht.get("y")}')
    oaht.remove('x')
    print(f'删除 "x" 后: {oaht}')
    print(f'get("x") 仍能正确返回 None? {oaht.get("x")}')
    assert oaht.get('y') == 200
    assert oaht.get('x') is None
    print('✅ 任务 3 通过！')

    # 测试任务 4
    print('\n--- 任务 4：LRU 缓存 ---')
    lru = LRUCache(2)
    lru.put(1, 1); lru.put(2, 2)
    print(f'put(1,1), put(2,2) 后: {lru}')
    print(f'get(1) = {lru.get(1)}')
    lru.put(3, 3)  # 淘汰 2
    print(f'put(3,3) 后: {lru}')
    print(f'get(2) = {lru.get(2)} (应该返回 -1)')
    assert lru.get(2) == -1
    print('✅ 任务 4 通过！')

    print('\n🎉 所有练习已完成！')


if __name__ == '__main__':
    main()
