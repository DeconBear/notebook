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

**核心对比与边界。** 网格搜索在指定区间枚举候选；梯度方法利用可微代理的局部导数，但不保证找到全局最优或比网格搜索更准确。当前梯度例子没有区间投影，可能离开训练范围；此时低代理误差也不保证真实 PDE 误差低，应将最终参数代回数值求解器核验。

令 $J(a)=\operatorname{mean}[(\widehat u(a)-u_\mathrm{target})^2]$。每轮先算 $J(a_t)$，再由 Adam 更新到 $a_{t+1}$。因此返回最终 $a_T$ 时必须重新前向计算 $J(a_T)$，不能直接返回历史数组最后一项 $J(a_{T-1})$。损失曲线保留更新前的训练轨迹，终端的最终 MSE 对应更新后的参数。

这里记录的是梯度更新步数，每步包括前向与反向，此外还有一次最终前向核验；它与网格搜索的纯前向次数不是等成本指标。高维网格搜索的组合数量增长很快，但梯度方法的单步成本也取决于模型规模与导数计算，不能说与维度完全无关。

> [!NOTE]
> 2026-10-08：已修复返回参数与最终 MSE 的错位。本次未重跑逆向设计；现有图和耗时未作运行验证。

### 关键概念速查表

| 概念 | 一句话解释 | 代码位置 |
|------|-----------|---------|
| 代理模型 | 把"参数→输出"学成可微的神经网络 | `TinySurrogate` |
| 黑盒搜索 | 只能采样评估，不能求导 | `blackbox_grid_search()` |
| 可微逆向设计 | 直接对设计参数做梯度下降 | `gradient_inverse_design()` |


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/science/frontier/code/demo.py`
