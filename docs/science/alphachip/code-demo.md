---
title: "as07 AlphaChip — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# as07 AlphaChip — demo.py 代码详解

<a href="/notebook/code/science/alphachip/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/science/alphachip/code
python demo.py
```

依赖 PyTorch（CPU 即可），训练 800 轮 REINFORCE 大约十几秒。

## 代码逐段详解

### 第1步：合成带簇状结构的 netlist

```python
def generate_netlist(n_macros, n_clusters, seed):
    cluster_id = rng.randint(0, n_clusters, size=n_macros)
    for i, j: prob = 0.55 if same_cluster else 0.08
```

用"簇内高概率连接、簇间低概率连接"模拟真实芯片中"功能模块内部信号密集、跨模块通信相对稀疏"的结构特征。这个先验结构也是后续对比"RL 策略能否自己发现簇状规律"的关键。

### 第2步：总线长（wirelength）—— 奖励函数的核心

```python
def wirelength(positions, edges):
    for i, j, w in edges:
        d = abs(positions[i][0]-positions[j][0]) + abs(positions[i][1]-positions[j][1])
        total += w * d
```

用 Manhattan 距离近似连线长度（半周长模型 HPWL 的简化版，真实 EDA 工具会考虑更复杂的布线拓扑）。这个函数在整个 demo 中扮演"环境模拟器"的角色——它是唯一能告诉智能体"这个布局好不好"的信号来源。

### 第3步：NetlistGNN —— 编码芯片连接结构

```python
class NetlistGNN(nn.Module):
    def forward(self, node_features, edge_index, edge_weight):
        h = self.input_proj(node_features)
        for msg_fn, upd_fn in zip(...):
            m = msg_fn(torch.cat([h[src], h[dst], edge_weight.unsqueeze(-1)], dim=-1))
            agg.index_add_(0, dst, m)
            h = upd_fn(torch.cat([h, agg], dim=-1))
```

与 as05 的 `MPNNLayer` 结构几乎一致，唯一区别是这里的消息函数额外接收 `edge_weight`（连线的"关键性"权重），让编码器能区分"粗导线"和"细导线"。节点初始特征是"归一化连接度 + 所属簇的 one-hot"——在真实系统中，这里会替换成宏单元的真实物理属性（尺寸、端口数、类型等）。

### 第4步：序贯放置策略 —— 带掩码的动作空间

```python
def forward(self, node_features, edge_index, edge_weight, order, greedy=False):
    node_repr = self.gnn(...)
    occupied_mask = torch.zeros(self.n_cells)
    for macro_id in order:
        feat = torch.cat([node_repr[macro_id], occupied_mask])
        logits = self.place_head(feat)
        logits = logits.masked_fill(occupied_mask.bool(), float('-inf'))
        dist = torch.distributions.Categorical(logits=logits)
```

**关键设计**：每一步的输入不仅有当前待放置宏单元的 GNN 表征，还有**当前网格的占用掩码**——这让策略网络"知道"哪些位置已经被占用，配合 `masked_fill(..., float('-inf'))` 把已占用位置的采样概率强制归零（softmax 后 $e^{-\infty}=0$），从工程上保证了合法性约束（不重叠）不需要额外的惩罚项，而是直接从动作空间的定义中"消除"了非法动作。

### 第5步：REINFORCE 训练循环

```python
positions, log_probs = policy(...)
reward = -wirelength(positions, EDGES)
baseline = 0.95 * baseline + 0.05 * reward
advantage = reward - baseline
loss = -advantage * torch.stack(log_probs).sum()
```

这是策略梯度的标准写法：`torch.stack(log_probs).sum()` 是整条轨迹（一次完整布局的所有放置动作）的对数概率之和，乘以优势 `advantage`（当前奖励与滑动平均基线的差值）取负数作为损失——如果这条轨迹的奖励高于近期平均水平，就提高它对应的动作序列的概率；反之降低。基线的滑动平均更新（`0.95` 的动量）让优化目标更平滑，减少梯度估计的方差。

### 第6步：基线对比与可视化

`random_placement()` 完全不利用连接信息；`greedy_heuristic_placement()` 利用"簇"这个先验知识（真实系统中可能不总是能提前知道这种结构），把 netlist、三种放置结果绘制在网格上直接对比总线长。

### 关键概念速查表

| 概念 | 一句话解释 | 代码位置 |
|------|-----------|---------|
| netlist | 宏单元(节点)+连线(边)构成的图 | `generate_netlist()` |
| 序贯放置 | 依次为每个宏单元选择位置，而非一次性输出全部坐标 | `PlacementPolicy.forward()` |
| 动作掩码 | 用 -inf 屏蔽已占用位置，保证放置合法 | `masked_fill()` |
| REINFORCE | 用优势(奖励-基线)加权轨迹对数概率之和作为损失 | `train_rl_placement()` |
| 稀疏/延迟奖励 | 只有放完所有模块才能算出总线长 | `wirelength()` 调用时机 |


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/science/alphachip/code/demo.py`
