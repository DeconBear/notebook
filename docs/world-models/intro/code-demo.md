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

CPU、NumPy、Matplotlib。三张图：`wm01-01-taxonomy.png`（五条路径地图）、`rollout_error_comparison.png`（像素 vs 潜空间误差累积）、`world_model_radar_comparison.png`（主观雷达）。没有神经网络。`TAXONOMY` 写「五条」，雷达字典也是五条；注释里偶尔写「六条」是笔误，以列表长度为准。

## 代码逐段详解

### 第1步：`TAXONOMY` — 数据与画图分开

每条路径一个 `dict`：`key`（简称）、`full`（一句话）、`methods`（叶子上的代表方法）、`color`、`note`（斜体关键词）。

```python
TAXONOMY = [
    dict(key='路径一 视频生成', full='GAN / VAE / 扩散 / Sora',
         methods=['GAN→VAE→DiT', 'Sora / Cosmos'], color='#C1666B', ...),
    ...
]
```

后面 `plot_taxonomy_map` 和雷达图都 **遍历这份表**，改一条路径只改这里。五条对应后续章节：视频生成、交互/3D、抽象状态、因果、符号。

`set_seed(42)` 只调 `np.random.seed`。分类图是确定性的；误差仿真才用到高斯噪声。

---

### 第2步：分类地图 — 弧线 `rad` 跟纵向偏移走

```python
root_xy = (1.1, 4.0)
y_positions = np.linspace(7.1, 0.9, n)
arrow = FancyArrowPatch(
    root_xy, branch_xy,
    connectionstyle=f"arc3,rad={(y - root_xy[1]) * 0.06}",
    ...
)
```

- **`linspace(7.1, 0.9, n)`**：从上到下均匀 $n$ 个 $y$。根在 $y=4$，上下分支对称。
- **`arc3,rad=...`**：贝塞尔弯曲。分支比根高则 `rad>0` 往一侧弯，比根低则反号，避免五条直线叠在一起。系数 `0.06` 是手调的视觉量。
- **`FancyBboxPatch((x-1.55, y-0.42), 3.1, 0.84)`**：盒子以 `branch_xy` 为中心。`item['key']` 写在偏上，`full` 偏下。
- 叶子：`leaf_ys = linspace(y+0.28*(n_leaves-1), y-0.28*(n_leaves-1), n_leaves)`。一条方法时 `n_leaves=1`，`linspace(y,y,1)` 就是 $y$ 本身；两条则上下各偏 0.28。
- **`bbox=dict(boxstyle='round,...')` 包在 `ax.text` 里**：叶子是文字+浅色底，不是第二套 Patch。

`axis('off')` + 自定义 `xlim/ylim`：整张图当画布。

---

### 第3步：`simulate_rollout_error` — 复合增长，不是真世界模型

直觉：多步想象时，误差会进下一步输入。像素空间要拟合高频纹理，`step_error` 取得更大。

$$
e_{t}=e_{t-1}(1+\eta)+|\xi_t|
$$

```python
e_pixel = 0.02
e_latent = 0.02
for t in range(horizon):
    noise_p = np.random.normal(0, 0.003)
    e_pixel = e_pixel * (1 + pixel_step_error) + abs(noise_p)
    e_latent = e_latent * (1 + latent_step_error) + abs(noise_l)
```

- **同一起点 `0.02`**：公平。差别只在 `pixel_step_error=0.045` vs `latent_step_error=0.018`。
- **`abs(noise)`**：噪声只往上加，误差曲线单调涨，读图简单。这**不是**无偏随机游走。
- **`n_trials=200`**：`pixel_curves[trial, t] = e_pixel`，返回 `mean(axis=0)` 和 `std(axis=0)`。`axis=0` 对试验维塌缩，留下长度 `horizon` 的均值曲线。

`main` 打印第 10、30 步和「像素/潜空间」倍数。第 30 步下标是 `[29]`（0-based）。

---

### 第4步：误差图上的置信带

```python
steps = np.arange(1, horizon + 1)   # 横轴从 1 画到 30
ax.fill_between(steps, pixel_errors - pixel_std, pixel_errors + pixel_std, alpha=0.15)
```

`fill_between` 在均值 ± 标准差之间填色。`alpha=0.15` 半透明，两条带重叠仍能分色。纵轴注释写明「玩具尺度，非真实单位」——不要把数值读成像素 RMSE。

---

### 第5步：雷达图 — 首尾相接才能闭合

```python
angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
angles += angles[:1]
values = scores + scores[:1]
ax.plot(angles, values, ...)
ax.fill(angles, values, alpha=0.06, color=color)
```

- **`endpoint=False`**：6 个角点均匀占满一圈，不重复 $0$ 与 $2\pi$。
- **`angles += angles[:1]`**：把第一个角再接到末尾。`scores + scores[:1]` 同样。否则折线缺一边。
- **`subplot_kw=dict(polar=True)`**：极坐标轴。`set_ylim(0,5)` 与打分 1–5 对齐。
- **`RADAR_SCORES` 是教学主观分**，不是评测。视频生成「生成质量=5、样本效率=1」；符号路径「可解释性=5」。颜色取 `TAXONOMY` 里同一套，图例才能对上。

`legend(..., bbox_to_anchor=(1.3, 1.1))`：图例放到极坐标外面，避免挡住轴标签。`set_xticks(angles[:-1])` 必须丢掉闭合用的最后一个角，否则最外一圈标签会重复第一个维度。`colors = [item['color'] for item in TAXONOMY]` 与 `RADAR_SCORES.items()` 靠**插入顺序**配对：两个 dict/list 都按路径一到五写，不要只改其中一份的顺序。

`main` 在误差仿真后打印 `pixel_err[9]`、`[29]`：第 10 步、第 30 步（下标从 0）。比值 `pixel_err[29] / latent_err[29]` 应明显大于 1，对应「为什么在潜空间做梦」。这是玩具随机游走，换一组 `step_error` 倍数会变，不要当论文数据。

---

### 关键概念速查表

| 概念 | 直觉 | 代码 |
|------|------|------|
| 五条路径 | 视频 / 交互 / 抽象 / 因果 / 符号 | `TAXONOMY` |
| `arc3,rad` | 按纵向偏移弯箭头 | `FancyArrowPatch` |
| rollout 误差 | $e(1+\eta)+\|\xi\|$ | `simulate_rollout_error` |
| `mean(axis=0)` | 对试验平均 | 平滑曲线 |
| `fill_between` | ±1σ 带 | 误差图 |
| 雷达闭合 | 复制第一个点到末尾 | `angles += angles[:1]` |
| `endpoint=False` | 一圈不重复 | `linspace` |
| 主观打分 | 非评测 | `RADAR_SCORES` |

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/world-models/intro/code/demo.py`
