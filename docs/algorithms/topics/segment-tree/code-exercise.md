---
title: "algo14 线段树与树状数组 — exercise.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# algo14 线段树与树状数组 — exercise.py 练习指南

<a href="/notebook/code/algorithms/topics/segment-tree/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过三个练习巩固 BIT 和线段树的核心操作：差分 BIT 的区间更新、线段树的区间最大值、BIT 逆序对计数。

## 预备知识

- BIT 的 `add()` 和 `query()` 操作及 `lowbit` 原理
- 差分数组技巧：$diff[l] += x$, $diff[r+1] -= x$ 实现区间更新
- 线段树的递归构建和查询结构
- 逆序对的定义和 BIT 统计方法

## 任务清单

### 任务1：差分 BIT — 区间更新 + 点查询

- **核心原理**：维护原数组 A 的差分数组 D 的 BIT。区间 [l,r] 加 val → `bit.add(l, val); bit.add(r+1, -val)`。
- **点查询**：`bit.query(i)` 就是 A[i] 的值（因为 A[i] = sum(D[1..i])）。

### 任务2：线段树区间最大值

- **不需要惰性传播**！最大值在区间更新时无法简单地用 lazy 标签处理（除非是区间赋值操作）。
- **单点更新**：从叶子向上更新路径，每个节点取左右子树的最大值。
- **区间查询**：标准的三分支递归——完全包含/不相交/部分相交。

### 任务3：BIT 逆序对计数

- **离散化**：将原始数组的值映射到 1~N 的排名（sorted + dict）。
- **统计方法**：
  - 从左到右遍历，BIT 初始为空
  - 对每个元素 x：逆序对 += i - BIT.query(x)（已遍历的元素中 > x 的个数）
  - 然后 BIT.add(x, 1)
- **复杂度**：$O(n \log n)$。

## 提示

1. 差分 BIT 中注意 `n+2` 数组大小，为 `r+1` 越界留空间。
2. 线段树最大值查询中，不相交时返回 `-inf`（而非 0）。
3. BIT 逆序对需要离散化，否则值域太大 BIT 数组放不下。

<<< @/algorithms/topics/segment-tree/code/exercise.py
