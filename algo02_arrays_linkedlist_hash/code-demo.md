---
title: "algo02 数组、链表与哈希表 — demo.py"
---

# algo02 数组、链表与哈希表 — demo.py 代码详解

<a href="../code/algo02_arrays_linkedlist_hash/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd algo02_arrays_linkedlist_hash/code
python demo.py
```

## 代码结构概览

本演示实现了四大数据结构，各对应一个类：

| 类名 | 功能 | 核心技巧 |
|------|------|----------|
| `DynamicArray` | 动态数组 | 2倍扩容，均摊 O(1) append |
| `SinglyLinkedList` | 单链表 | 指针操作，头部 O(1) 插入 |
| `DoublyLinkedList` | 双链表 | 双向指针，O(1) 删除任意节点 |
| `HashTable` | 哈希表 | 除法哈希 + 链地址法 |
| `LRUCache` | LRU缓存 | HashTable + DoublyLinkedList 组合 |

## 第1步：动态数组扩容机制

```python
def append(self, value):
    if self._size == self._capacity:
        self._resize(self._capacity * 2)  # 容量翻倍
    self._data[self._size] = value
    self._size += 1
```

**关键设计决策**：为什么扩容因子是 2？如果扩容因子太小（如 1.1），扩容太频繁，均摊代价高；如果太大（如 10），浪费内存。因子 2 是工程实践中的最佳平衡——Java ArrayList、C++ vector 都使用约 1.5~2 的扩容因子。

## 第2步：单链表的指针操作

```python
def prepend(self, data):
    new_node = SinglyNode(data)
    new_node.next = self.head   # 新节点指向旧的头节点
    self.head = new_node        # 更新 head
```

单链表的核心操作图解：

```
插入前: head → [A] → [B] → None
         new → [X]

步骤1: new.next = head
         new → [X] → [A] → [B] → None

步骤2: head = new
         head → [X] → [A] → [B] → None
```

## 第3步：双链表 O(1) 删除任意节点

双链表的核心优势：给定节点本身就能删除，不需要找前驱。

```python
def remove_node(self, node):
    if node.prev:
        node.prev.next = node.next   # 前驱跳过 node
    else:
        self.head = node.next        # node 是头节点
    if node.next:
        node.next.prev = node.prev   # 后继跳过 node
    else:
        self.tail = node.prev        # node 是尾节点
```

**为什么这需要双链表？** 单链表只知道 `node.next`，要删除 node 必须找到它的前驱（需要 O(n) 遍历）。

## 第4步：哈希表的链地址法

```python
class HashTable:
    def __init__(self):
        self._buckets = [[] for _ in range(capacity)]  # 每个桶是一个列表

    def _hash(self, key):
        return hash(key) % self._capacity              # 除法哈希

    def put(self, key, value):
        bucket = self._buckets[self._hash(key)]
        for i, (k, v) in enumerate(bucket):
            if k == key:             # 键已存在，更新
                bucket[i] = (key, value)
                return
        bucket.append((key, value))   # 键不存在，追加
```

## 第5步：LRU 缓存的巧妙设计

```
数据结构:
  cache = {key1 → Node1, key2 → Node2, key3 → Node3}
  dll   = Node3(MRU) ↔ Node1 ↔ Node2(LRU)

get(key2):
  1. 从 cache 找到 Node2 → O(1)
  2. dll.remove_node(Node2) → O(1)
  3. dll.prepend(Node2) → O(1)
  4. 更新 cache[key2] → O(1)
  → 总共 O(1)！

put(key4, value4) 且缓存满 (capacity=3):
  1. 淘汰 LRU → dll.pop_tail() → Node2 → O(1)
  2. 从 cache 删除 key2 → O(1)
  3. dll.prepend(Node4) → O(1)
  4. cache[key4] = Node4 → O(1)
  → 总共 O(1)！
```

## 关键概念速查表

| 概念 | 数组 | 单链表 | 双链表 | 哈希表 |
|------|------|--------|--------|--------|
| 随机访问 | O(1) | O(n) | O(n) | N/A (按键访问 O(1)) |
| 头部插入 | O(n) | O(1) | O(1) | N/A |
| 尾部插入 | O(1)* | O(n) | O(1) | N/A |
| 删除给定节点 | N/A | O(n) | O(1) | O(1) |
| 内存开销 | 无额外指针 | 1 指针/节点 | 2 指针/节点 | 桶数组 + 链表开销 |
| 缓存友好 | 极好 | 差 | 差 | 中等 |

*均摊

## 完整代码

<<< @/snippets/algo02_arrays_linkedlist_hash/demo.py
