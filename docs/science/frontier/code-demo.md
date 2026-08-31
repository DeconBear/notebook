---
title: "as08 AI4S综合与前沿 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# as08 AI4S 综合与前沿 — demo.py 代码详解

<a href="/notebook/code/science/frontier/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/science/frontier/code
python demo.py
```

依赖 PyTorch（CPU 即可），约十几秒完成。

## 代码逐段详解

### 第1步：时间线与雷达图

`plot_timeline()` 与 `plot_method_radar()` 分别绘制本系列涉及的关键节点，以及五类方法在五个维度上的主观定性对比。雷达图评分是教学用的直觉打分，不是严格评测。

### 第2步：训练极简代理模型

```python
class TinySurrogate(nn.Module):
    def forward(self, a):
        return self.net(a)   # (B, 1) -> (B, N_GRID)
```

用一个小型 MLP 学习"参数 $a$ → 整条解曲线 $u(x)$"的映射（as04 变系数扩散问题的简化版）。这不是严格的算子网络，但足以支撑本节演示"对代理模型做梯度下降"这个核心思想。

### 第3步：黑盒网格搜索 vs 梯度逆向设计

```python
def blackbox_grid_search(...):
    for a in a_candidates:                       # 枚举，不使用梯度
        pred = model(torch.tensor([[a]]))
        err = mean((pred - u_target)^2)

def gradient_inverse_design(...):
    a = torch.tensor([[a_init]], requires_grad=True)
    for step:                                    # 对 a 本身做梯度下降
        loss = mean((model(a) - u_target)^2)
        loss.backward(); optimizer.step()
```

**核心对比**：网格搜索需要在整个候选区间上密集采样（评估次数 = 网格点数）；梯度方法利用代理模型的可微性，通常用远更少的迭代就能收敛到更精确的解。这个优势会随设计参数维度增加而愈发明显——回忆 as01 的维度灾难。

### 关键概念速查表

| 概念 | 一句话解释 | 代码位置 |
|------|-----------|---------|
| 代理模型 | 把"参数→输出"学成可微的神经网络 | `TinySurrogate` |
| 黑盒搜索 | 只能采样评估，不能求导 | `blackbox_grid_search()` |
| 可微逆向设计 | 直接对设计参数做梯度下降 | `gradient_inverse_design()` |

## 完整代码

<<< @/science/frontier/code/demo.py
