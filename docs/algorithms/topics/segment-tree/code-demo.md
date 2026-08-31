---
title: "algo14 线段树与树状数组 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# algo14 线段树与树状数组 — demo.py 代码详解

<a href="/notebook/code/algorithms/topics/segment-tree/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/algorithms/topics/segment-tree/code
python demo.py
```

## 代码逐段详解

### 第1步：Fenwick Tree (BIT)

BIT 的核心是 `lowbit` 操作：`i & -i` 提取 $i$ 的二进制最低位 1 所代表的值。

```python
def add(self, i, delta):
    while i <= self.n:
        self.tree[i] += delta
        i += i & -i  # 向上爬：更新所有"覆盖"位置 i 的区间

def query(self, i):
    total = 0
    while i > 0:
        total += self.tree[i]
        i -= i & -i  # 向下收集：收集覆盖前缀 [1,i] 的互不相交的区间
    return total
```

**为什么 BIT 不能求区间最大值？** 因为减法不可逆。前缀和可以用 `query(r) - query(l-1)`，但前缀最大值不能这样计算。线段树可以维护最大值，因为它直接维护区间信息。

**BIT 结构示例**（n=8）：
- tree[1] 维护 [1,1]（lowbit(1)=1）
- tree[2] 维护 [1,2]（lowbit(2)=2）
- tree[4] 维护 [1,4]（lowbit(4)=4）
- tree[6] 维护 [5,6]（lowbit(6)=2）
- tree[8] 维护 [1,8]（lowbit(8)=8）

### 第2步：线段树 + 惰性传播

惰性传播是线段树最精妙的设计。核心是 **"延迟结算"**：

```python
def _push_down(self, p, l, r):
    if self.lazy[p] != 0:
        mid = (l + r) // 2
        # 将 lazy 值传给子节点
        self.tree[p*2] += self.lazy[p] * (mid - l + 1)
        self.lazy[p*2] += self.lazy[p]
        self.tree[p*2+1] += self.lazy[p] * (r - mid)
        self.lazy[p*2+1] += self.lazy[p]
        self.lazy[p] = 0  # 清理
```

**执行流程示例**：`update_range(1, 3, 2)` 对 arr=[1,3,5,7,9,11]：

1. 根节点 [0,5] 不完全包含 → 下传 lazy
2. 左子 [0,2] 与 [1,3] 有交集 → 递归
3. [1,2] 完全包含 → 直接更新：tree+=2*2=4, lazy+=2
4. [3,3] 完全包含 → 直接更新：tree+=2*1=2, lazy+=2
5. 回溯：更新祖先节点的 tree 值

后续的 `query_range()` 会在需要时通过 `_push_down()` 将 lazy 标记向下传播。

### 第3步：持久化线段树

```python
def _update(self, node, l, r, idx, val):
    if l == r:
        return self.Node(val=val)
    mid = (l + r) // 2
    if idx <= mid:
        new_left = self._update(node.left, l, mid, idx, val)
        return self.Node(val=new_left.val + node.right.val,
                         left=new_left, right=node.right)  # ← 共享右子树！
```

**关键节约**：更新路径上的节点是新建的，**不在路径上的子树被共享**。这是持久化的核心——"Copy-on-Write"策略。更新 $O(\log n)$ 个新节点，其余节点与旧版本共享。

### 第4步：顺序统计

用 BIT 在值域上二分查找第 k 小的元素：

```python
def kth_smallest(self, k):
    lo, hi = 1, self.max_val
    while lo < hi:
        mid = (lo + hi) // 2
        if self.bit.query(mid) >= k:
            hi = mid
        else:
            lo = mid + 1
    return lo
```

这里 `query(mid)` 返回 $\leq mid$ 的元素个数，通过二分找到最小的 `mid` 使得 $\leq mid$ 的元素个数 $\geq k$。

## 关键概念速查表

| 概念 | 操作 | 复杂度 | 代码位置 |
|------|------|--------|---------|
| BIT 更新 | `i += i & -i` | $O(\log n)$ | `FenwickTree.add()` |
| BIT 查询 | `i -= i & -i` | $O(\log n)$ | `FenwickTree.query()` |
| 线段树区间更新 | lazy propagation | $O(\log n)$ | `SegmentTree._update_range()` |
| 惰性标记下传 | push_down | $O(1)$ | `SegmentTree._push_down()` |
| 持久化更新 | Copy-on-Write | $O(\log n)$ | `PersistentSegTree._update()` |
| 第 K 小 | BIT 上二分 | $O(\log^2 n)$ | `OrderStatistics.kth_smallest()` |

## 完整代码

<<< @/algorithms/topics/segment-tree/code/demo.py
