---
title: "信源编码 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 信源编码 — demo.py 代码详解

<a href="/notebook/code/information/source-coding/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/information/source-coding/code
python demo.py
```

CPU、NumPy 即可。一张图 `huffman_len.png`：四个符号的概率柱，横轴带上码字。终端打印码本、$H$、$L$。

## 代码逐段详解

### 第1步：堆里为什么要 `tie`

```python
leaf_ids[s] = i
heapq.heappush(heap, (p, i, i))
```

`heapq` 比较元组：先比概率 $p$，相同再比整数 `i`。两个符号概率相等时，若只放 `(p, s)`，而 `s` 不可比，会 TypeError。`i` 只是打破平局，不进码字。

---

### 第2步：从叶子爬到根

合并时记下 `parent[child]=(parent_id, bit)`。对每个符号从叶子往上收集比特，再 `reversed`——因为先碰到的是靠近叶子的位，码字习惯从根往下写。

叶子和内部结点都使用互不重复的整数 ID；`leaf_ids` 把原始符号映射到叶子 ID。符号本身可以是 `'#5'` 或整数，不会与树中的内部结点相撞。不能用看起来特殊的字符串充当内部结点名：源符号也可能正好叫这个名字，导致父指针覆盖、前缀冲突，甚至形成循环。

---

### 第3步：$L$ 对 $H$

```python
L = sum(probs[s] * len(codes[s]) for s in probs)
```

加权平均码长。Huffman 对这张 4 元表是最优前缀码，但仍可能 $L>H$（概率不是 $2^{-k}$）。`assert L >= H` 只防实现写反。

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/information/source-coding/code/demo.py`
