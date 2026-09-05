---
title: "wm01 世界模型导论 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# wm01 世界模型导论与分类 — demo.py 代码详解

<a href="/notebook/code/world-models/intro/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/world-models/intro/code
python demo.py
```

只依赖 NumPy 和 Matplotlib，CPU 秒级运行完毕。

## 代码逐段详解

### 第1步：五条技术路径的元数据表

```python
TAXONOMY = [
    dict(key='RSSM/Dreamer', full='隐式动力学 + 想象规划', methods=['PlaNet (RSSM)', 'DreamerV1→V3'], ...),
    dict(key='MuZero', full='隐式模型 + 树搜索', methods=['MuZero', 'EfficientZero'], ...),
    ...
]
```

用一个列表把五条技术路径的简称、全称、代表方法、颜色统一管理，后面绘制分类地图和雷达图都直接遍历这个数据结构——把"数据"和"绘图逻辑"分开，方便后续增删路径。

### 第2步：绘制分类地图 `plot_taxonomy_map()`

```python
arrow = FancyArrowPatch(root_xy, branch_xy, connectionstyle=f"arc3,rad={(y - root_xy[1]) * 0.06}", ...)
```

以"World Model"为根节点，向右延伸出五条分支（每条技术路径一个节点），再向右延伸出代表方法（叶子节点）。`connectionstyle="arc3,rad=..."` 让连接线根据纵向偏移量自动弯曲成弧线，视觉上更像一张"思维导图"，而不是死板的直线连接。

### 第3步：为什么在潜空间里做梦？—— rollout 误差累积仿真

```python
def simulate_rollout_error(horizon=30, pixel_step_error=0.045, latent_step_error=0.018, n_trials=200):
    for trial in range(n_trials):
        e_pixel = 0.02
        e_latent = 0.02
        for t in range(horizon):
            noise_p = np.random.normal(0, 0.003)
            e_pixel = e_pixel * (1 + pixel_step_error) + abs(noise_p)
            ...
```

这是一个**简化的类比仿真**，不是真实的世界模型训练：用 `error_t = error_{t-1} * (1 + step_error) + noise` 这样一个复合增长的随机游走，模拟"多步预测误差会随 rollout 步数累积"这个现象。像素空间用更大的 `step_error`（因为要建模大量与决策无关的高频视觉细节，误差更容易被放大），潜空间用更小的 `step_error`。跑 `n_trials=200` 次取平均，得到平滑的误差增长曲线用于对比，这解释了为什么 RSSM/Dreamer 选择在潜空间而不是像素空间做多步"想象"。

### 第4步：五条路径的多维度雷达图

```python
RADAR_DIMENSIONS = ['样本效率', '规划能力', '生成质量', '计算成本(越低越好)', '可解释性', '通用性']
RADAR_SCORES = {'RSSM/Dreamer': [4, 5, 3, 3, 3, 3], 'MuZero': [3, 5, 1, 2, 2, 2], ...}
```

这些分数是**教学用的主观定性打分**（1-5），不是严格的评测结果——目的是帮助建立"没有一条路径全面占优，需要按任务需求取舍"的直觉。`angles += angles[:1]` 和 `values = scores + scores[:1]` 是绘制雷达图的标准技巧：把第一个维度复制到末尾，让折线闭合成一个封闭多边形。

### 关键概念速查表

| 概念 | 一句话解释 | 代码位置 |
|------|-----------|---------|
| 分类地图 | 五条技术路径的树状可视化 | `plot_taxonomy_map()` |
| rollout 误差累积 | 用复合增长随机游走类比多步预测误差 | `simulate_rollout_error()` |
| 潜空间 vs 像素空间 | 潜空间每步误差增长率更小，能安全规划更长视野 | `simulate_rollout_error()` 参数设置 |
| 雷达图对比 | 六个维度上的主观定性打分，非严格评测 | `plot_radar_comparison()` |


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/world-models/intro/code/demo.py`
