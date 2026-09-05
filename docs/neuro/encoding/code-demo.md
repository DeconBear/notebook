---
title: "神经编码：速率、时间与群体 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 神经编码：速率、时间与群体 — demo.py 代码详解

<a href="/notebook/code/neuro/encoding/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/neuro/encoding/code
python demo.py
```

CPU、NumPy。两张图：`encoding_tuning_population.png`（余弦调谐 + 群体向量）和 `encoding_sparsity.png`（稠密速率 vs 短窗稀疏尖峰）。种子 42，终端打印群体估计与 40° 真刺激的误差。

## 代码逐段详解

### 第1步：`cosine_tuning` — 半波整流的余弦

运动皮层常用的玩具调谐：

$$
r(\theta)=r_0 + r_\max\max\bigl(0,\,\cos(\theta-\theta_\mathrm{pref})\bigr)
$$

```python
def cosine_tuning(theta, pref, r_max=40.0, r0=5.0):
    return r0 + r_max * np.maximum(0.0, np.cos(theta - pref))
```

- **`theta - pref`**：刺激方向减偏好方向。两者都是弧度。`np.cos` 对数组逐元素。
- **`np.maximum(0.0, ...)`**：逐元素和 0 取大。余弦负半周截掉，细胞在「反方向」只剩基线 $r_0=5$ Hz，不会出现负发放率。
- **`r_max=40`**：峰值为 $5+40=45$ Hz。数很小，后面泊松计数才不会爆。

这不是高斯调谐（V1 更常用高斯）；余弦 + 半波是方向选择性最简单的可微形状。

---

### 第2步：一簇偏好方向，画出调谐曲线

```python
prefs = np.linspace(0, 2 * np.pi, 12, endpoint=False)
thetas = np.linspace(0, 2 * np.pi, 180)
for p in prefs[::2]:
    axes[0].plot(np.degrees(thetas), cosine_tuning(thetas, p), lw=1.4)
```

- **`linspace(0, 2π, 12, endpoint=False)`**：12 个细胞均匀铺满一圈。**`endpoint=False`** 很关键：若 `True`，0 和 $2\pi$ 是同一方向，会重复一个细胞。
- **`prefs[::2]`**：每隔一个画，左图 6 条曲线，避免 12 条糊成一张皮。语法 `start:stop:step`，`::2` 是步长 2。
- **`np.degrees(thetas)`**：横轴用度，读图更直观。调谐函数内部仍用弧度。

---

### 第3步：群体向量 — 泊松计数再加权方向

刺激固定在 40°。每个细胞的**期望速率**由调谐给出，观测是短窗口里的泊松计数：

$$
n_i\sim\mathrm{Poisson}(r_i\Delta t),\quad
\vec v=\sum_i n_i(\cos\theta_i,\,\sin\theta_i)
$$

估计方向 $\hat\theta=\mathrm{atan2}(v_y,v_x)$。

```python
stim = np.deg2rad(40.0)
rates = cosine_tuning(stim, prefs)
spikes = np.random.poisson(rates * 0.1)  # 100 ms 窗口
vec = np.sum(spikes[:, None] * np.stack([np.cos(prefs), np.sin(prefs)], axis=1), axis=0)
est = np.arctan2(vec[1], vec[0])
```

- **`np.deg2rad(40)`**：刺激也要弧度，才能和 `prefs` 相减。
- **`rates * 0.1`**：$\Delta t=100$ ms $=0.1$ s。泊松参数是期望**个数** $r\Delta t$，不是 Hz 本身。
- **`np.random.poisson`**：对数组每个细胞独立抽样。同一套 `rates` 每次运行（种子固定则同）得到整数尖峰数。
- **`spikes[:, None]`**：`(12,)` → `(12,1)`，好和 `(12,2)` 的方向向量广播相乘。
- **`np.stack([cos, sin], axis=1)`**：两个 `(12,)` 堆成 `(12,2)`。`axis=1` 表示新轴插在列向。
- **`np.sum(..., axis=0)`**：对细胞维求和，留下 2D 向量。
- **`np.arctan2(y, x)`**：四象限反正切。只用 `arctan(y/x)` 会在第二、三象限出错。参数顺序是 **先 y 后 x**。

右图：细胞画在其偏好方向上、半径正比于速率；红箭头真刺激（单位长），绿箭头估计（0.8 倍长，避免完全重叠）。`set_aspect('equal')` 否则角度视觉会歪。

终端误差：

```python
abs(((np.degrees(est)-40+180)%360)-180)
```

把角度差折到 $[-180,180]$ 再取绝对，避免 $359^\circ$ 和 $1^\circ$ 被算成差 358。

---

### 第4步：稀疏度 — 同一套「有速率」的两种看法

```python
r = np.clip(np.random.gamma(2.0, 8.0, size=200), 0, None)
spike_win = (np.random.rand(200) < r * 0.02).astype(float)
ax.bar([0, 1], [np.mean(r < 0.05), 1.0 - np.mean(spike_win > 0)], ...)
```

- **`np.random.gamma(2, 8, size=200)`**：200 个细胞的速率，形状偏斜、多数中等、少数很高。`clip(..., 0, None)` 下限 0、上限不限（`None`）。
- **左柱 `mean(r < 0.05)`**：速率几乎为 0 的比例。连续速率很少严格是 0，这柱应较矮。
- **右柱**：20 ms 窗（$r\times 0.02$ 当发放概率）里的伯努利试验。`rand() < p` 是一次尖峰与否。`1 - mean(spike_win>0)` = 短窗内沉默细胞比例，通常高——**稀疏是时间窗的产物**，不是「速率本身到处是零」。
- **`.astype(float)`**：布尔变 0.0/1.0，后面 `> 0` 仍可用。

`set_xticks([0, 1], ['速率接近 0 的比例', '短窗内沉默细胞比例'])`：位置和标签一起给（较新 matplotlib API）。

右图群体向量里：

```python
axes[1].scatter(np.cos(prefs) * rates, np.sin(prefs) * rates, ...)
axes[1].arrow(0, 0, np.cos(stim), np.sin(stim), ...)
```

细胞画在「偏好方向 × 该刺激下的速率」上，不是画在单位圆上。速率高的点离原点更远，群体向量的几何才看得见。`arrow` 的 `width=0.03` 是箭杆宽度（数据坐标）。绿箭乘 `0.8` 只为了不和红箭完全重合，**不是**把估计缩短成 0.8 的置信度。

Gamma 的 `shape=2, scale=8` 期望是 $16$ Hz 量级。`np.random.rand(200)` 是 $[0,1)$ 均匀，与 `r * 0.02` 比大小 = 一次伯努利。两张图数据源不同：第一张 12 个余弦细胞 + 泊松；第二张 200 个独立 Gamma 速率——不要把稀疏度柱状图理解成「同一群方向细胞」。

`grid(True, axis='y')` 只画横线，柱状图竖线多余。种子 42 钉死泊松和 Gamma，终端误差可复现。

入口 `main()` 先画调谐+群体，再画稀疏度，最后一行 `print` 角度误差。没有训练循环。`cosine_tuning(stim, prefs)` 里 `stim` 是标量、`prefs` 是长度 12 的数组，广播后每个细胞一个速率。`np.maximum` 与 `np.max` 不同：前者逐元素对 0 取大，后者会把整段数组收成一个标量，调谐曲线会塌掉。

---

### 关键概念速查表

| 概念 | 数学 / 直觉 | 代码 |
|------|-------------|------|
| 余弦调谐 | $r_0+r_\max[\cos]_+$ | `cosine_tuning` |
| `endpoint=False` | 一圈不重复 0 与 $2\pi$ | `prefs` |
| `[::2]` | 隔一个取一个 | 少画几条曲线 |
| 泊松计数 | $n\sim\mathrm{Poisson}(r\Delta t)$ | `poisson(rates * 0.1)` |
| 群体向量 | $\sum n_i \vec u_i$ | `spikes[:, None] * stack(...)` |
| `arctan2` | 四象限角度 | `est` |
| `[:, None]` | 加轴广播 | 加权求和 |
| 稀疏 | 短窗大量沉默 | `rand() < r*0.02` |

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/neuro/encoding/code/demo.py`
