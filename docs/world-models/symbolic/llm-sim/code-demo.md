---
title: "wm08 LLM 世界模型与路径对比 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# LLM 世界模型 — demo.py 代码详解

<a href="/notebook/code/world-models/symbolic/llm-sim/demo.py" target="_blank" download>Download demo.py</a>

```bash
cd docs/world-models/symbolic/llm-sim/code
python demo.py
```

**没有**字符级 MLP。二元语法计数 + 抽样，模拟「语言模型当转移核」。

## 代码逐段详解

`defaultdict(Counter)`：`trans[a][b] += 1` 记 bigram。`zip(seq, seq[1:])` 把相邻词配成对。

```python
items, counts = zip(*opts.items())
p = np.array(counts, dtype=float); p /= p.sum()
s = np.random.choice(items, p=p)
```

`zip(*dict.items())` 把键和值拆成两条元组。概率与计数成正比——最简 n-gram。未见过的词 `opts` 空则 `break`。

柱状图分数是教学主观分，不是评测。

规则世界对比见 [符号导论](/world-models/symbolic/overview/code-demo)。

## 源码位置

`docs/world-models/symbolic/llm-sim/code/demo.py`
