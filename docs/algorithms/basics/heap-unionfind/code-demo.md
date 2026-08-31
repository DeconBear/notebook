---
title: "algo05 堆、并查集与跳跃表 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# algo05 堆、并查集与跳跃表 — demo.py 代码详解

<a href="/notebook/code/algorithms/basics/heap-unionfind/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/algorithms/basics/heap-unionfind/code
python demo.py
```

## 代码结构

| 类 | 功能 | 复杂度 |
|-----|------|--------|
| `MinHeap` | 最小堆 (push/pop/heapify) | push/pop O(log n), heapify O(n) |
| `UnionFind` | 并查集 (路径压缩 + 按秩合并) | find/union ~O(alpha(n)) |
| `SkipList` | 跳跃表 (概率平衡多层链表) | search/insert/delete O(log n) 期望 |

## 第1步：heapify 为什么是 O(n)？

```python
def _heapify(self):
    for i in range(len(self._data) // 2 - 1, -1, -1):
        self._sift_down(i)
```

数学证明：底层节点（叶节点）不做操作（高度 0），倒数第二层最多下沉 1 次...根节点最多下沉 log(n) 次。总操作次数：
$$\sum_{h=0}^{\log n} \frac{n}{2^{h+1}} \cdot h < n \sum_{h=0}^{\infty} \frac{h}{2^h} = 2n$$

## 第2步：路径压缩的威力

```python
def find(self, x):
    if self.parent[x] != x:
        self.parent[x] = self.find(self.parent[x])  # 压缩！
    return self.parent[x]
```

压缩后，`parent[x]` 直接指向根。后续 find(x) 只需 O(1)。

## 第3步：跳跃表的概率层数

```python
def _random_level(self):
    level = 0
    while random.random() < self.P and level < self.MAX_LEVEL:
        level += 1
    return level
```

P=0.5 时：50% 的节点在 level 0，25% 在 level 1，12.5% 在 level 2...期望总节点数 = n/(1-P) = 2n。

## 关键概念速查表

| 结构 | 本质 | push/insert | pop/delete | 查找 | 空间 |
|------|------|------------|-----------|------|------|
| MinHeap | 数组+完全二叉树 | O(log n) | O(log n) | O(n) | O(n) |
| UnionFind | 森林+路径压缩 | O(alpha(n)) | - | O(alpha(n)) | O(n) |
| SkipList | 多层有序链表 | O(log n) 期望 | O(log n) 期望 | O(log n) 期望 | O(n) 期望 |

## 完整代码

<<< @/algorithms/basics/heap-unionfind/code/demo.py
