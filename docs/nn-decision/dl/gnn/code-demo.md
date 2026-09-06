---
title: "图神经网络 GNN — demo.py"
---

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 图神经网络 — demo.py 代码详解

<a href="/notebook/code/nn-decision/dl/gnn/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/nn-decision/dl/gnn/code
python demo.py
```

两社团随机图（团内边密、团间稀），节点分类。`GCN` 用 \(\tilde D^{-1/2}\tilde A\tilde D^{-1/2}HW\)；`GATLayer` 按目标节点 softmax 边权。不安装 PyG / DGL。图写入 `images/gnn_gcn_gat.png`。

## 代码逐段详解

### 第1步：建图

`make_two_community_graph` 造一张随机块模型：20 个节点、两团。团内连边概率高、团间低。节点特征是 **度数 + 带噪声的社团指示**——分类器不能只靠「读标签」，得靠邻域结构把噪声抹平。

边存成 `edge_index`（\(2\times E\)，PyG 同款）和稠密 `A`（给 GCN 做矩阵乘，图很小才敢这样）。

### 第2步：GCN 的对称归一化

```python
A_hat = A + I
deg = A_hat.sum(dim=1)
d_inv = deg.pow(-0.5)
A_norm = d_inv.unsqueeze(1) * A_hat * d_inv.unsqueeze(0)
```

这就是 \(\tilde D^{-1/2}\tilde A\tilde D^{-1/2}\)。前向两行：

```text
h = ReLU(A_norm @ (x W1))
logit = A_norm @ (h W2)
```

每个邻居（含自环）贡献 \(1/\sqrt{d_i d_j}\)，度数大的节点不会把邻居冲掉。

### 第3步：GAT 按目标节点做 softmax

对每条边 \(j\to i\) 打分 \(e_{ij}=\mathrm{LeakyReLU}(a^\top[Wh_i\|Wh_j])\)，然后 **只在指向同一个 \(i\) 的边上** softmax，得到 \(\alpha_{ij}\)。`index_add_` 把 \(\alpha_{ij} Wh_j\) 加到目标节点——和 as05 的 mean 聚合同一套路，只是权重不再均匀。

玩具图上两边准确率都会很高；看图时重点看 **嵌入有没有按社团分开**，以及 GAT 是否更敢把跨团的边压小（本 demo 画的是二维 logit，不是注意力热图）。

### 第4步：训练

全图节点都有标签（转导设定，transductive）。交叉熵 + Adam 200 步。工业上的归纳设定（新节点、新图）要用 GraphSAGE 那种邻居采样，见正文第三节。

## 关键概念速查表

| 概念 | 含义 | 代码位置 |
|------|------|----------|
| `edge_index` | 边列表 \(2\times E\) | `make_two_community_graph` |
| 对称归一化 | GCN 的 \(\tilde D^{-1/2}\tilde A\tilde D^{-1/2}\) | `gcn_norm_adj` |
| 消息聚合 | 按 dst 加权求和 | `GATLayer` + `index_add_` |
| 转导 vs 归纳 | 测试节点是否在训练图里 | 本 demo 是转导 |

消息传递三步见 [消息传递](/nn-decision/dl/gnn/message-passing/)；GCN / GAT 公式见 [变体](/nn-decision/dl/gnn/variants/)；网格/分子见 [as05](/science/gnn/)。

## 源码位置

`docs/nn-decision/dl/gnn/code/demo.py`
