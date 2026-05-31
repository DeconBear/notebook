---
title: "algo02 数组、链表与哈希表 — exercise.py"
---

# algo02 数组、链表与哈希表 — exercise.py 练习指南

<a href="../code/algo02_arrays_linkedlist_hash/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过四个递进任务的实现，深入掌握数组、链表和哈希表的底层操作。

## 任务清单

### 任务1：补全动态数组的 insert/delete

在 `DynamicArray` 类中实现 `insert()` 和 `delete()` 方法。

**insert 算法**：
1. 检查索引合法性 `0 <= index <= size`
2. 若 `size == capacity`，调用 `_resize(capacity * 2)`
3. `for i in range(size, index, -1): data[i] = data[i-1]`（元素后移）
4. `data[index] = value`、`size += 1`

**delete 算法**：
1. 检查索引合法性 `0 <= index < size`
2. `for i in range(index, size-1): data[i] = data[i+1]`（元素前移）
3. `data[size-1] = None`、`size -= 1`

### 任务2：实现双链表的 remove_node

**关键**：利用 `prev` 和 `next` 指针，在 O(1) 时间内删除任意节点。

需要考虑四种边界情况：
- 头节点（`node.prev is None`）→ 更新 `self.head`
- 尾节点（`node.next is None`）→ 更新 `self.tail`
- 中间节点 → 前后节点互连
- 不要忘记 `size -= 1`

### 任务3：实现开放地址法哈希表

使用线性探测实现 `put`、`get`、`remove`。

**关键概念——墓碑（Tombstone）**：
- 删除时**不能**设为 `None`（会切断探测链！）
- 必须用特殊标记 `TOMBSTONE` 替代
- `put` 时可以复用 TOMBSTONE 位置
- `get` 时遇到 TOMBSTONE 继续探测

### 任务4（Bonus）：实现 LRU 缓存

使用 `dict` + `DoublyLinkedList` 实现 O(1) 的 get/put。

**核心思路**：哈希表负责快速定位（O(1)），双链表维护访问顺序。

## 验证标准

```bash
cd algo02_arrays_linkedlist_hash/code
python exercise.py
```

期望输出：四个任务的 `✅ 通过` 信息和 `🎉 所有练习已完成！`

## 完整代码

<<< @/snippets/algo02_arrays_linkedlist_hash/exercise.py
