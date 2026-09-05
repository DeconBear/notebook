---
title: "回路：方向选择性与 E–I 平衡 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 回路：方向选择性与 E–I 平衡 — demo.py 代码详解

<a href="/notebook/code/neuro/circuits/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/neuro/circuits/code
python demo.py
```

CPU、NumPy。两张图：`direction_weights.png`（STDP 在 LR 扫过后的权重剖面）和 `ei_raster.png`（稀疏 E–I LIF 的 raster）。种子在模块级是 0，STDP 训练另用 `default_rng(0)`。没有 CNN、没有开环世界模型。

## 代码逐段详解

### 第1步：`STDPSynapse` — 痕迹而不是存整段脉冲表

指数窗 pairwise STDP 的在线版：各保留一条 pre / post 痕迹，按 $\tau_\pm$ 衰减。

$$
\dot A_\mathrm{pre}=-A_\mathrm{pre}/\tau_+,\quad
\text{post 发放时 }\Delta w \propto +A_\mathrm{pre}
$$

（pre 发放时用 post 痕迹做 LTD。）

```python
A_PLUS, A_MINUS = 0.02, 0.022
TAU_PLUS, TAU_MINUS = 20.0, 20.0

def decay(self, dt):
    self.pre_tr *= np.exp(-dt / TAU_PLUS)
    self.post_tr *= np.exp(-dt / TAU_MINUS)

def on_pre(self):
    self.w = float(np.clip(self.w - A_MINUS * self.post_tr, 0, 1))
    self.pre_tr += 1.0
```

- **`A_MINUS` 略大于 `A_PLUS`**：整体略偏 LTD，权重不会无界涨到全 1。
- **语法 `*=`**：原地乘。$e^{-\Delta t/\tau}$ 是指数衰减的精确解（这一小段 $\Delta t$ 内无事件）。
- **先改权重再 `+= 1` 痕迹**：同一时刻的 pre 不该立刻用自己刚加上的痕迹。顺序反了会变成「自己强化自己」。
- **`np.clip(..., 0, 1)`**：硬饱和。`float(...)` 把 0 维数组变回 Python 标量。

`on_post` 对称：`w += A_PLUS * pre_tr`，再 `post_tr += 1`。

---

### 第2步：`train_direction` — 扫过 + 延迟的 post

```python
order = list(range(n_inputs)) if preferred == 'LR' else list(range(n_inputs - 1, -1, -1))
```

- **`range(n-1, -1, -1)`**：从 $n-1$ 倒数到 0。RL 扫过就是输入下标反过来。
- 每个 sweep：输入按 `order` 每隔 `dt_pair=10` ms 各发一次 pre；**最后一个 pre 之后再 2 ms** 发一次 post。对「先到的输入」$\Delta t$ 更大但仍为正 → LTP 弱；对「刚扫到的输入」$\Delta t\approx 2$ ms → LTP 强。于是偏好方向上权重形成斜坡。

```python
events.append((base + k * dt_pair, 'pre', idx))
events.append((base + (n_inputs - 1) * dt_pair + 2.0, 'post', -1))
events.sort()
```

事件是 `(时间, 种类, 下标)` 元组。**`events.sort()`** 按时间排（元组先比第一项）。然后：

```python
dt = t - t_prev
for syn in syns:
    syn.decay(dt)
```

所有突触先共同衰减 `dt`，再处理当前事件。post 事件对**每一条**突触 `on_post`（一个输出细胞）。

```python
corr = float(np.corrcoef(np.arange(n_inputs), w)[0, 1])
if preferred == 'RL':
    corr = -corr
```

- **`np.corrcoef(x, w)[0,1]`**：位置下标与权重的皮尔逊相关。LR 训练应得到**正**相关（右边权更大）。RL 时剖面应反过来，代码把相关取负，让「选择性分数」仍朝正表示「符合偏好」。
- `w0 + 0.05 * rng.normal()`：初始权重略抖，避免完全对称。

---

### 第3步：`probe` — 用加权和当方向读出

```python
def probe(weights, direction):
    n = len(weights)
    order = range(n) if direction == 'LR' else range(n - 1, -1, -1)
    return float(sum(weights[idx] * (k + 1) for k, idx in enumerate(order)))
```

按扫过顺序给时间权 `1,2,...,n`（越晚越大），与突触权重点乘。同一组 `weights` 上 `probe(..., 'LR')` 应大于 `'RL'`——训练后的选择性。生成器表达式 `sum(... for ...)` 不必先建列表。

---

### 第4步：`simulate_ei` — 向量化 LIF，不是细胞对象

```python
W = np.zeros((n, n))
# W[post, pre] = ...
I_base = np.where(is_e, 1.15, 0.75).astype(float)
```

- **`W[post, pre]`**：行是突触后、列是突触前。兴奋→兴奋 `0.8`，兴奋→抑制 `1.2`，抑制→兴奋 `-0.5`，抑制→抑制 `-0.6`。`p=0.12` 稀疏随机连，对角跳过（`pre != post`）。
- **`np.where(is_e, 1.15, 0.75)`**：E 细胞外加电流略大，否则抑制群容易把网掐死。
- **`is_e[:n_e] = True`**：前 `n_e` 个是兴奋，后面是抑制。raster 图里用水平虚线切开。

时间环：

```python
active = ref <= 0
V[active] += dt * (-(V[active] - Vrest) + R * I[active]) / tau
V[~active] = Vreset
fired = active & (V >= Vth)
if np.any(fired):
    syn += W[:, fired].sum(axis=1)
```

- **布尔花式索引 `V[active]`**：只更新不在不应期的细胞。
- **`W[:, fired]`**：所有后细胞 × 本拍发放的前细胞，再 `sum(axis=1)` 把多个发放者的权重加起来。下一拍 `I` 才吃到 `syn`（延迟一拍，避免同拍瞬间环）。
- **`syn[:] = 0`**：电流型瞬时突触，脉冲只活一拍。不是指数衰减突触。
- **`spikes[t_i] = fired`**：`(n_steps, n)` 布尔矩阵。

---

### 第5步：raster 与平均发放率

```python
rows, cols = np.where(spikes)
ax.scatter(t[rows], cols, s=2, c=np.where(is_e[cols], '#1a5276', '#c0392b'), ...)
```

- **`np.where(spikes)`**：True 的坐标。`rows` 是时间下标，`cols` 是神经元编号。
- **`t[rows]`**：下标换成 ms。颜色按该细胞是否 E。
- 丢掉前 100 ms 再算平均 Hz：`spikes[start:].sum(axis=0) / duration` 再对细胞 `mean`。

---

### 关键概念速查表

| 概念 | 数学 / 直觉 | 代码 |
|------|-------------|------|
| 痕迹 STDP | 指数窗在线版 | `STDPSynapse` |
| 先 decay 再事件 | 因果 | `train_direction` 循环 |
| LR 斜坡 | 晚到的输入 LTP 更强 | `dt_pair=10` + post 延迟 2 ms |
| `corrcoef` | 位置–权重相关 | 选择性分数 |
| `W[post,pre]` | 邻接约定 | `simulate_ei` |
| 向量化 LIF | 无 Python 细胞对象 | `V[active] += ...` |
| `W[:, fired]` | 把发放者的列加总 | 突触电流 |
| `np.where(spikes)` | raster 点 | 散点坐标 |

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/neuro/circuits/code/demo.py`
