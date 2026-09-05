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
heapq.heappush(heap, (p, i, s))
```

`heapq` 比较元组：先比概率 $p$，相同再比整数 `i`。两个符号概率相等时，若只放 `(p, s)`，而 `s` 不可比，会 TypeError。`i` 只是打破平局，不进码字。

---

### 第2步：从叶子爬到根

合并时记下 `parent[child]=(parent_id, bit)`。对每个符号从叶子往上收集比特，再 `reversed`——因为先碰到的是靠近叶子的位，码字习惯从根往下写。

内部结点名叫 `'#5'` 这种字符串，避免和符号 `'A'` 撞名。

---

### 第3步：$L$ 对 $H$

```python
L = sum(probs[s] * len(codes[s]) for s in probs)
```

加权平均码长。Huffman 对这张 4 元表是最优前缀码，但仍可能 $L>H$（概率不是 $2^{-k}$）。`assert L >= H` 只防实现写反。

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/information/source-coding/code/demo.py`
