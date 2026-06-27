# algo14 线段树与树状数组

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。


> Fenwick Tree 和 Segment Tree —— 区间查询与更新的两大高效数据结构

---

## 一、树状数组（Fenwick Tree / BIT）

### 1.1 解决的问题

**树状数组（Binary Indexed Tree, BIT）** 用于高效支持以下两种操作：
- **点更新（Point Update）**：将位置 i 的元素增加 delta
- **前缀查询（Prefix Query）**：查询前 i 个元素的和

两种操作都在 **$O(\log n)$** 时间内完成。

### 1.2 核心思想：二进制分解

BIT 的核心思想是利用**二进制表示**将前缀 $[1, i]$ 分解为 $O(\log n)$ 个不相交的区间。每个区间由 BIT 数组中的一个元素 `tree[x]` 维护，其中 `tree[x]` 存储的是以 $x$ 结尾、长度为 `lowbit(x)` 的区间和。

**lowbit 函数**：
$$
\text{lowbit}(x) = x \text{ \& } (-x)
$$

这是 $x$ 的二进制表示中最低位 1 所代表的值。例如：
- $\text{lowbit}(6) = \text{lowbit}(110_2) = 010_2 = 2$
- $\text{lowbit}(8) = \text{lowbit}(1000_2) = 1000_2 = 8$

### 1.3 实现

**更新操作**：$i \leftarrow i + \text{lowbit}(i)$ 向上爬树
```python
def add(i, delta):
    while i <= n:
        tree[i] += delta
        i += i & -i  # 跳转到下一个受影响的区间
```

**查询操作**：$i \leftarrow i - \text{lowbit}(i)$ 向下收集
```python
def query(i):
    total = 0
    while i > 0:
        total += tree[i]
        i -= i & -i  # 跳转到下一个不相交的前缀区间
    return total
```

**区间和**：$sum[l..r] = \text{query}(r) - \text{query}(l-1)$

![BIT树状数组结构图：展示一个长度为8的BIT数组的树形结构。每个节点标注tree[i]维护的区间范围——如tree[4]维护[1,4]，tree[6]维护[5,6]。用箭头标注更新操作的路径（i += lowbit(i)往上爬）和查询操作的路径（i -= lowbit(i)往下收集）。底部展示各lowbit值](./images/algo14-01-fenwick-tree.png)

### 1.4 扩展到区间更新 + 单点查询

使用**差分数组**技巧：维护原数组的差分数组 $D[i] = A[i] - A[i-1]$ 的 BIT。区间 $[l, r]$ 加 $x$ 等价于 $D[l] \,+\!=\, x$ 和 $D[r+1] \,-\!=\, x$。单点查询 $A[i] = \sum_{k=1}^{i} D[k]$。

---

## 二、线段树（Segment Tree）

### 2.1 解决的问题

**线段树** 比 BIT 更通用，支持：
- 区间查询（range query）：任意区间上的聚合操作（和、最大值、最小值、GCD 等）
- 单点更新
- **惰性传播（Lazy Propagation）** 支持区间更新

所有操作都在 **$O(\log n)$** 时间内完成。

### 2.2 线段树的结构

线段树是一棵平衡二叉树：
- 每个节点代表一个区间 $[l, r]$
- 根节点代表整个区间 $[0, n-1]$
- 叶子节点代表单个元素
- 内部节点的值 = 左右子节点的值的聚合（如和、最大值）

**存储方式**：通常用数组实现，下标 1 为根节点，节点 $p$ 的左子为 $2p$，右子为 $2p+1$。

### 2.3 基本操作

**构建（Build）**：$O(n)$，自底向上

**单点更新**：$O(\log n)$，从叶子向上更新路径上的所有祖先节点

```python
def update(p, l, r, idx, val):
    if l == r:
        tree[p] = val  # 找到叶子节点
        return
    mid = (l + r) // 2
    if idx <= mid:
        update(p*2, l, mid, idx, val)
    else:
        update(p*2+1, mid+1, r, idx, val)
    tree[p] = tree[p*2] + tree[p*2+1]  # 聚合
```

**区间查询**：$O(\log n)$，递归地将查询区间拆分为 $O(\log n)$ 个节点区间的并

```python
def query(p, l, r, ql, qr):
    if ql <= l and r <= qr:  # 当前区间完全在查询区间内
        return tree[p]
    if qr < l or r < ql:     # 当前区间与查询区间不相交
        return 0
    mid = (l + r) // 2
    left_sum = query(p*2, l, mid, ql, qr)
    right_sum = query(p*2+1, mid+1, r, ql, qr)
    return left_sum + right_sum
```

### 2.4 惰性传播（Lazy Propagation）

当需要**区间更新**（如区间 $[l, r]$ 内所有元素 $+x$）时，逐个更新每个叶子需要 $O(n)$。惰性传播将更新"推迟"到查询时才真正执行。

```python
def push_down(p, l, r):
    """将节点 p 的惰性标记传递给子节点"""
    if lazy[p] != 0:
        mid = (l + r) // 2
        tree[p*2] += lazy[p] * (mid - l + 1)  # 左子树和增加
        tree[p*2+1] += lazy[p] * (r - mid)    # 右子树和增加
        lazy[p*2] += lazy[p]                  # 传递惰性标记
        lazy[p*2+1] += lazy[p]
        lazy[p] = 0  # 清除当前节点的惰性标记
```

**核心思想**："先记账，后结算"。更新操作不做到底，而是在覆盖整个区间时就停止并留下一个 lazy tag。查询操作经过该节点时再将 lazy tag 下传给子节点。

![线段树结构示意图：上半部分展示一棵完整的线段树（区间[0,7]），每个节点标注它所代表的区间范围和当前值。下半部分用不同颜色高亮查询区间[2,5]的分解过程——被分解为[2,3]和[4,5]两个节点区间。右边展示惰性传播：区间[2,4]加3，对应的lazy标记传播路径](./images/algo14-02-segment-tree.png)

---

## 三、进阶主题

### 3.1 持久化线段树（Persistent Segment Tree）

持久化线段树（主席树）在更新时**不修改原节点**，而是新建一条从根到叶子的路径（称为"版本"）。每个版本只新增 $O(\log n)$ 个节点。

**典型应用**：查询区间第 K 小（静态 K-th smallest）。构建一棵权值线段树的历史版本链，查询时利用前缀和思想：版本 r 的树减去版本 l-1 的树即为区间 $[l, r]$ 的信息。

### 3.2 权值线段树（Weighted Segment Tree）

将线段树的值域作为区间（而非数组下标），用于**顺序统计**操作：
- 插入/删除一个数
- 查询第 k 小的数
- 查询某个数的排名

### 3.3 二维 BIT / 线段树

将 BIT 扩展到二维，支持矩阵上的点更新和子矩阵求和。每个 BIT 节点本身也是一个 BIT（树套树），时间复杂度 $O(\log n \cdot \log m)$。

---

## 四、BIT vs 线段树对比

| 特性 | BIT (Fenwick Tree) | 线段树 |
|------|-------------------|--------|
| 空间 | $O(n)$（极紧凑） | $O(4n)$（约 4 倍） |
| 代码量 | 10 行 | 50+ 行 |
| 单点更新 + 区间查询 | ✓ | ✓ |
| 区间更新 + 单点查询 | ✓（差分数组） | ✓ |
| 区间更新 + 区间查询 | ✗（需两个 BIT） | ✓（惰性传播） |
| 区间最大/最小值 | ✗（BIT 不可逆） | ✓ |
| 第 K 小/顺序统计 | ✗ | ✓（权值线段树） |
| 持久化 | ✗ | ✓ |

---

## 本章总结

| 概念 | 一句话 |
|------|--------|
| BIT 核心 | lowbit 二进制分解，$O(\log n)$ 前缀和 |
| BIT 更新 | `i += i & -i`，向上爬树 |
| BIT 查询 | `i -= i & -i`，向下收集 |
| 线段树 | 平衡二叉树，每个节点代表一个区间 |
| 惰性传播 | "先记账后结算"，区间更新 $O(\log n)$ |
| 持久化 | 不修改原节点，新建路径，历史版本可查 |
| 权值线段树 | 值域为区间，支持第 K 小和排名 |

> BIT 和线段树是解决区间问题的两大工具。数据规模 $n \leq 10^5$、操作数 $q \leq 10^5$ 时，BIT 和线段树是标准配置。BIT 简洁高效但功能有限，线段树功能全面但代码量更大。选择哪个取决于问题需求——能用 BIT 时优先用 BIT。

---

## 📥 Code

| File | View | Download |
|------|------|----------|
| demo.py | [Open](./code-demo) | <a href="../code/algo14_segment_tree/demo.py" target="_blank" download>Download</a> |
| exercise.py | [Open](./code-exercise) | <a href="../code/algo14_segment_tree/exercise.py" target="_blank" download>Download</a> |

## 参考

1. Fenwick, P. M. (1994). A new data structure for cumulative frequency tables. *Software: Practice and Experience*, 24(3), 327-336. [[doi:10.1002/spe.4380240306](https://doi.org/10.1002/spe.4380240306)]
2. Cormen, T. H. et al. (2009). *Introduction to Algorithms* (3rd ed.). MIT Press. Chapter 14: Augmenting Data Structures. (Segment Tree)
