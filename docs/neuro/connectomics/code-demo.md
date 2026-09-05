---
title: "连接组学与全脑模拟入口 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 连接组学与全脑模拟入口 — demo.py 代码详解

<a href="/notebook/code/neuro/connectomics/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/neuro/connectomics/code
python demo.py
```

CPU、NumPy。一张图 `toy_adjacency.png`：20 个节点、连接概率 0.15 的有向随机图，画成突触权重矩阵。数据格式是教学用的 **SONATA-lite**（节点表 + 边表），不是 Allen/BMTK 的完整 HDF5。终端打印边数和密度。

## 代码逐段详解

### 第1步：`make_toy_connectome` — 为什么用字典列表而不是立刻 `W`

真实连接组先是「谁连谁 + 突触参数」，矩阵是派生视图。代码先生成边表，再让 `adjacency` 去填矩阵，对应「文件里存边、仿真时再装配」。

```python
def make_toy_connectome(n=20, p=0.15, seed=0):
    rng = np.random.default_rng(seed)
    edges = []
    for src in range(n):
        for dst in range(n):
            if src == dst:
                continue
            if rng.random() < p:
                edges.append({
                    'source_node_id': int(src),
                    'target_node_id': int(dst),
                    'syn_weight': float(rng.uniform(0.1, 1.0)),
                    'delay_ms': float(rng.choice([1.0, 2.0, 5.0])),
                })
```

- **`np.random.default_rng(seed)`**：独立 Generator，不碰全局 `np.random`。模块级虽有 `seed(0)`，边表仍显式传入 `seed=0`，函数可单独复现。
- **跳过 `src == dst`**：无自环。Erdős–Rényi 有向图常见约定。
- **`rng.random() < p`**：均匀 $[0,1)$ 与概率比。每对有序节点独立，期望边数 $n(n-1)p$。
- **边是字典**：键名模仿 SONATA 边表（`source_node_id` / `target_node_id`）。`int(...)` / `float(...)` 避免 NumPy 标量进 JSON 时难序列化（本 demo 没写文件，但格式是为那种心态准备的）。
- **`rng.uniform(0.1, 1.0)`**：权重连续。 **`rng.choice([1,2,5])`**：延迟只三档，提醒「边不只有权重，还有时延」；本脚本画图没用 `delay_ms`，它躺在表里。

返回的大字典：

```python
return {
    'format': 'toy-sonata-lite',
    'version': '0.1',
    'nodes': [{'node_id': i, 'node_type': 'LIF'} for i in range(n)],
    'edges': edges,
    'meta': {'n_neurons': n, 'p_connect': p, 'seed': seed},
}
```

- **列表推导** `[{...} for i in range(n)]`：每个节点一条记录，类型全标 `LIF`（玩具，没有点类型多样性）。
- **`meta`**：装配矩阵时只信这里的 `n_neurons`，不靠「节点列表长度」推断，和真实清单里 metadata 分开存的习惯一致。

---

### 第2步：`adjacency` — 行是后、列是前

$$
W_{\mathrm{post},\,\mathrm{pre}} = w_{\mathrm{syn}}
$$

仿真里电流常常是 $I = W s$， $s$ 是突触前发放，所以**行 = 突触后**。

```python
def adjacency(data):
    n = int(data['meta']['n_neurons'])
    W = np.zeros((n, n))
    for e in data['edges']:
        W[e['target_node_id'], e['source_node_id']] = e['syn_weight']
    return W
```

- **先开全零再填**：没连上的就是 0，稀疏图大部分格子空。
- **下标顺序 `[target, source]`**：若写成 `[source, target]`，热图的横纵标签会和正文「行=后、列=前」对不上，回路章 `W[post, pre]` 也不一致。
- 多条边连同一对（本生成器不会）会被后者覆盖；玩具 ER 图每对最多一条。

---

### 第3步：密度 — 分母是 $n(n-1)$ 不是 $n^2$

```python
print('边数', len(data['edges']),
      '密度', np.count_nonzero(W) / (W.shape[0] * (W.shape[0] - 1)))
```

- **`len(data['edges'])`**：边表长度。应等于 `np.count_nonzero(W)`（无重复边、无零权重；权重下限 0.1）。
- **`count_nonzero`**：非零条目数。
- **分母 $n(n-1)$**：合法有向位置，不含对角。若除以 $n^2$，密度会被自环那 $n$ 个永远为 0 的格子拉低，和生成时的 $p$ 对不齐。期望密度 ≈ `p_connect=0.15`。

---

### 第4步：热图

```python
im = ax.imshow(W, cmap='viridis', vmin=0)
ax.set_xlabel('突触前')
ax.set_ylabel('突触后')
fig.colorbar(im, ax=ax, fraction=0.046, label='权重')
```

- **`imshow`**：矩阵第 0 行画在**上面**（图像坐标）。神经元 0 在顶/左。
- **`vmin=0`**：颜色从 0 起，空连接是深色底，有边才亮。不设的话 matplotlib 会按数据最小最大值自动拉伸。
- **`colorbar(..., fraction=0.046)`**：色条宽度相对主轴的比例，避免色条比图还宽。

没有图布局（networkx），故意只给矩阵：连接组学第一眼是「矩阵有多密、块结构有没有」，不是力导向乱线。

双层 `for src / for dst` 对 $n=20$ 只需 400 次随机数，不必向量化。若写成 `rng.random((n,n)) < p` 再填边，对角还得手动清零，字典边表也不如循环里 `append` 直观。`node_type: 'LIF'` 全员相同：玩具图不区分兴奋/抑制；回路章的 E–I 是另一套 `W`，不要把两章的矩阵当成同一对象。

`main` 里 `make_toy_connectome()` 用默认 `n=20, p=0.15, seed=0`。改 `p` 应看到亮格子比例跟着变；改 `seed` 图案变、密度仍约 0.15。色条 `label='权重'` 对应 `syn_weight`，**不是** delay。延时若要可视化，需要另一张直方图，本章不做。

`figsize=(5.2, 4.6)` 接近方形，矩阵才不像被拉扁。没有 `set_aspect` 也行，因为 `imshow` 默认格子是方的。终端先打印边数和密度，再存图：若密度离谱（比如写成了 $n^2$ 分母），不用等看图就能发现。

列表 `'nodes'` 目前只服务于格式完整；`adjacency` 不读它，只读 `meta['n_neurons']` 和 `edges`。删掉节点表图仍能画——这提醒「装配矩阵的最低输入是边 + n」。

---

### 关键概念速查表

| 概念 | 直觉 | 代码 |
|------|------|------|
| SONATA-lite | 节点表 + 边表 | `make_toy_connectome` 返回字典 |
| ER 有向图 | 每对独立以 $p$ 连 | `rng.random() < p` |
| 无自环 | 跳过 `src==dst` | 双层 for |
| `W[post, pre]` | 行=后、列=前 | `adjacency` |
| 密度 | 边 / $n(n-1)$ | `count_nonzero` |
| `default_rng` | 独立可复现 | `seed=0` |
| `imshow` | 邻接热图 | `vmin=0` |
| `delay_ms` | 边属性示例 | 本图未使用 |

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/neuro/connectomics/code/demo.py`
