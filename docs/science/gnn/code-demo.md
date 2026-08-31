---
title: "as05 科学计算中的 GNN — demo.py"
---

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# as05 科学计算中的 GNN — demo.py 代码详解

<a href="/notebook/code/science/gnn/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/science/gnn/code
python demo.py
```

## 代码逐段详解

### 第1步：建图

- `build_grid_graph_2d`：四邻接网格 → 边列表  
- `make_toy_molecule`：6 节点玩具分子 + one-hot 原子类型 + 度数特征

### 第2步：两种消息传递

**不可学习平滑**（场景 A）：

```python
h_i' = (1-α) h_i + α * mean_j h_j
```

在热斑网格上迭代，可视化扩散过程 → `mesh_smoothing.png`。

**可学习层** `MessagePassingLayer`（场景 B）：

```text
messages = W_msg(h[src])
agg[dst] = mean(messages)
h' = ReLU(W_self(h) + agg)
```

用 `index_add_` 按目标节点聚合，不依赖 PyG。

### 第3步：TinyGNN 训练

两层 MP + 线性头，监督信号是「邻居原子序数均值」。训练约 300 epoch 后 MSE 应降到很小，并输出：

| 文件 | 内容 |
|------|------|
| `as05-01-message-passing.png` | 概念图 |
| `mesh_smoothing.png` | 网格扩散 |
| `molecule_mp_result.png` | 分子预测 |
| `node_features_before_after.png` | 特征空间变化 |

## 关键概念速查表

| 概念 | 含义 | 代码位置 |
|------|------|----------|
| 邻接表 / edge_index | 图的两种存储 | `edges_to_adj_list` / `edge_index` |
| 均值聚合 | AGG = mean | `message_passing_smooth` |
| 过平滑 | 多层后节点趋同 | 网格方差下降 |
| 读出 | 节点 → 任务输出 | `TinyGNN.head` |

## 完整代码

<<< @/science/gnn/code/demo.py
