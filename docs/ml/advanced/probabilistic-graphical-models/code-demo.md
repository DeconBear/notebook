---
title: "ml13 概率图模型基础 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# ml13 概率图模型基础 — demo.py 代码详解

<a href="/notebook/code/ml/advanced/probabilistic-graphical-models/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/ml/advanced/probabilistic-graphical-models/code
python demo.py
```

## 代码逐段详解

### 第1步：贝叶斯网络的枚举推理

```python
class BayesianNetwork:
    def joint_probability(self, assignment):
        prob = 1.0
        for node in self.nodes:
            prob *= self.get_cpt_value(node, assignment)
        return prob
```

贝叶斯网络的核心在于联合分布的因子分解：

$$
P(X_1, \dots, X_n) = \prod_{i=1}^{n} P(X_i \mid \text{Pa}(X_i))
$$

这个分解将原本需要指数级参数的联合分布表分解为每个节点的一个小型条件概率表（CPT）。`get_cpt_value` 通过多维数组索引高效查找条件概率值。

#### 枚举推理

```python
def enumerate_all(self, query, evidence=None):
    node_ranges = [range(self.cardinalities[n]) for n in all_nodes]
    for values in product(*node_ranges):
        assignment = dict(zip(all_nodes, values))
        # 过滤与证据一致的赋值
        if not consistent: continue
        prob = self.joint_probability(assignment)
        query_probs[assignment[query]] += prob
    query_probs /= query_probs.sum()
```

枚举法的思路直接但朴素：遍历所有变量取值的组合，对每个与证据一致的赋值计算联合概率，累加到查询变量的对应取值上。复杂度为 $O(\prod_i |X_i|)$，随变量数指数增长。

### 第2步：Sprinkler 网络与 Explaining Away

以经典的 Sprinkler 网络为例：

```
Cloudy → Sprinkler → WetGrass
   ↓                    ↑
   └────── Rain ────────┘
```

有趣的推理现象——**解释消除（Explaining Away）**：
- $P(\text{Rain}=1 \mid \text{WetGrass}=1) = 0.708$（草湿了 → 很高概率是下雨）
- $P(\text{Rain}=1 \mid \text{WetGrass}=1, \text{Sprinkler}=1) = 0.576$（同时知道洒水器开了 → 下雨概率下降）

解释消除的原因是：洒水器已经"解释"了草为什么湿，所以不再需要下雨来解释。这是贝叶斯网络中 collider 结构的标志性行为。

### 第3步：变量消除（Variable Elimination）

变量消除是枚举法的优化——利用分配律改变求和的顺序：

$$
\sum_C \sum_S \sum_R P(C) P(S|C) P(R|C) P(W|S,R)
$$

通过"早消除早受益"（消除一个变量时只影响包含它的因子），可以将部分计算从指数级降为局部多项式级。

在代码中，`Factor` 类封装了因子乘积和求和消除两个核心操作：
- `multiply`：两个因子相乘以创建涵盖合并变量集的更大因子
- `marginalize`：对指定变量求和以消除它

### 第4步：链状图上的信念传播

```python
def belief_propagation_chain(potentials, evidence=None):
    # 前向传递
    for i in range(n_vars - 1):
        msg_to_next = (msg_from_x[:, np.newaxis] * psi).sum(axis=0)
        fwd_messages[i+1] = msg_to_next / msg_to_next.sum()

    # 后向传递
    for i in range(n_vars - 2, -1, -1):
        msg_to_prev = (psi * msg_from_next[np.newaxis, :]).sum(axis=1)
        bwd_messages[i] = msg_to_prev / msg_to_prev.sum()

    # 边缘 = 前向消息 * 后向消息
    marginals[i] = fwd_messages[i] * bwd_messages[i]
```

在链状（树形）因子图上，BP 在两次遍历（一次前向、一次后向）后即可计算出所有节点的精确边缘分布。这本质上就是 HMM 前向-后向算法的推广。

**消息计算的核心操作**：
- 因子→变量：$\mu_{f \to x_{i+1}}(x_{i+1}) = \sum_{x_i} \psi(x_i, x_{i+1}) \cdot \mu_{x_i \to f}(x_i)$
- 边缘概率：$P(x_i) \propto \mu_{f_{i-1} \to x_i}(x_i) \cdot \mu_{f_i \to x_i}(x_i)$

### 第5步：d-分离可视化

三种基本结构：
1. **链式 (A→B→C)**：B 被观测时阻塞路径
2. **分叉 (A←B→C)**：B 被观测时阻塞路径
3. **汇合 (A→B←C)**：B **不被**观测时阻塞路径；B 被观测时反而激活

汇合结构的反直觉行为是理解贝叶斯网络推理的关键。

## 关键概念速查表

| 概念 | 数学形式 | 代码位置 | 关键说明 |
|------|---------|---------|---------|
| 联合分布分解 | $P = \prod P(X_i \mid \text{Pa})$ | `joint_probability()` | 贝叶斯网络的核心 |
| 枚举推理 | 遍历所有赋值 | `enumerate_all()` | $O(\prod |X_i|)$，指数级 |
| 变量消除 | 改变求和顺序 | `variable_elimination()` | 分配律优化 |
| 因子乘积 | $f_1 \cdot f_2$ | `Factor.multiply()` | 变量集合并 |
| 因子边缘化 | $\sum_x f(x, \dots)$ | `Factor.marginalize()` | 消除变量 |
| BP 消息 | $\sum \psi \cdot \mu_{\text{in}}$ | `belief_propagation_chain()` | 链/树上精确 |
| 解释消除 | $P(R \mid W,S) < P(R \mid W)$ | Sprinkler 网络 | Collider 标志行为 |


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/ml/advanced/probabilistic-graphical-models/code/demo.py`
