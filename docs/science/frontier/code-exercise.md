---
title: "as08 AI4S综合与前沿 — exercise.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# as08 AI4S 综合与前沿 — exercise.py 练习指南

<a href="/notebook/code/science/frontier/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

实现黑盒网格搜索与梯度下降逆向设计两种策略，直观对比"可微代理模型"相对"黑盒采样"在搜索效率上的优势，并观察外推场景下代理模型泛化能力对逆向设计可靠性的影响。

## 任务清单

### 任务1：实现 `grid_search_1d(...)`

在区间上均匀采样候选值，对每个候选用代理模型前向评估，返回 MSE 最小的候选。

### 任务2：实现 `gradient_search_1d(...)`

把参数 $a$ 设为 `requires_grad=True` 的可训练标量，对"预测曲线与目标曲线的 MSE"直接做 Adam 梯度下降。

### 任务3（Bonus）：外推鲁棒性对比

在超出代理模型训练范围的目标参数上，对比两种方法——网格搜索被限制在训练范围内无法找到真实外推参数；梯度下降可以走出训练范围，但代理模型本身在外推区可能不准确。

## 验证标准

运行 `python exercise.py`：两个搜索方法找到的 $a$ 与真实值偏差应小于 0.15。


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/science/frontier/code/exercise.py`
