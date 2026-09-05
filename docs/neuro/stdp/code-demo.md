---
title: "Hebb 与 STDP — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# Hebb 与 STDP — demo.py 代码详解

<a href="/notebook/code/neuro/stdp/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/neuro/stdp/code
python demo.py
```

CPU、NumPy。两张图：`stdp_window.png`（pairwise 指数学习窗）和 `stdp_pair_trace.png`（$\Delta t=+10$ ms 重复配对时权重往上走）。常数：$A_+=0.01$、$A_-=0.012$、$\tau_\pm=20$ ms，权重夹在 $[0,1]$。

## 代码逐段详解

### 第1步：`stdp_dw` — 窗的定义与 $\Delta t$ 符号

约定 $\Delta t = t_\mathrm{post}-t_\mathrm{pre}$：

$$
\Delta w=
\begin{cases}
A_+ e^{-\Delta t/\tau_+} & \Delta t>0 \quad\text{（因果：先 pre 后 post → LTP）}\\
-A_- e^{\Delta t/\tau_-} & \Delta t<0 \quad\text{（反因果 → LTD）}\\
0 & \Delta t=0
\end{cases}
$$

```python
def stdp_dw(dt):
    if dt > 0:
        return A_PLUS * np.exp(-dt / TAU_PLUS)
    if dt < 0:
        return -A_MINUS * np.exp(dt / TAU_MINUS)
    return 0.0
```

- **$\Delta t>0$ 时 `-dt`**：指数衰减，隔得越远 LTP 越弱。
- **$\Delta t<0$ 时 `exp(dt / TAU_MINUS)`**：`dt` 已是负数，$e^{\Delta t/\tau}$ 仍是衰减。前面的**负号**来自 `-A_MINUS`，整段在横轴负半边是负的。
- **$\Delta t=0$ 回 0**：同时发放不更新，避免窗在 0 处跳或双算。真实生理更含糊，教学窗在 0 断开。
- **`A_MINUS` 略大**：积分 LTD 面积略大于 LTP，防止全 1 饱和（和回路章同一家族，幅度更小）。

这是**成对公式**，一次给一个 $\Delta t$。后面 `STDPSynapse` 用痕迹实现同一窗，可处理多脉冲。

---

### 第2步：`pairwise_update` — 一次配对后的权重

```python
def pairwise_update(w, dt):
    return float(np.clip(w + stdp_dw(dt), W_MIN, W_MAX))
```

硬边界 $[0,1]$。`main` 末尾打印 `pairwise_update(0.5, 10.0)`：从 0.5 出发、$\Delta t=+10$ 应略增，作为窗函数的数值抽检。

---

### 第3步：`STDPSynapse` — 与回路章相同的痕迹机制

```python
def decay(self, dt):
    self.pre_tr *= np.exp(-dt / TAU_PLUS)
    self.post_tr *= np.exp(-dt / TAU_MINUS)

def on_pre(self):
    self.w = float(np.clip(self.w - A_MINUS * self.post_tr, W_MIN, W_MAX))
    self.pre_tr += 1.0

def on_post(self):
    self.w = float(np.clip(self.w + A_PLUS * self.pre_tr, W_MIN, W_MAX))
    self.post_tr += 1.0
```

- 连续时间里两脉冲间隔 $\Delta t$ 时，痕迹里残存 $e^{-|\Delta t|/\tau}$，乘 $A_\pm$ 即窗函数。多脉冲时痕迹**累加**，不是只看最近一对（all-to-all 痕迹近似）。
- **先改 `w` 再 `+=1`**：当前脉冲不进入自己的配对。
- 本章 `A_PLUS=0.01`，回路章是 `0.02`：单独演示窗时步子更小，轨迹更好看。

---

### 第4步：画出学习窗

```python
dts = np.linspace(-80, 80, 401)
dw = np.array([stdp_dw(d) for d in dts])
ax.plot(dts, dw, ...)
```

- **401 个点**：包含 $\Delta t=0$。列表推导对每个标量调一次 `stdp_dw`（内部是 `if`，不好向量化，401 次无所谓）。
- 横轴标签是 matplotlib 数学模式：`r'$\Delta t = t_{\mathrm{post}}-t_{\mathrm{pre}}$ (ms)'`。前缀 `r` 让 `\` 不当 Python 转义。
- `axhline(0)` / `axvline(0)`：十字准线。右半应在横轴上（LTP），左半在下（LTD），0 处为 0。

---

### 第5步：重复因果配对轨迹

```python
syn = STDPSynapse(0.4)
dt_ms = 0.1
pre_times = set(int(round(x / dt_ms)) for x in np.arange(20, 380, 25))
post_times = set(int(round((x + 10.0) / dt_ms)) for x in np.arange(20, 380, 25))
```

- **时间步 0.1 ms**，总长 400 ms。权重每步都记进 `W[i]`，曲线连续。
- **`np.arange(20, 380, 25)`**：20, 45, … ms 的 pre。post 一律 **+10 ms**，即窗上 LTP 侧。
- **`set(...)`**：下标集合，`if i in pre_times` 是 $O(1)$ 查询。`round` 后再 `int`，避免 `20/0.1` 这类浮点变成 `199.999`。
- 每步：`decay(0.1)` → 若本拍有 pre 则 `on_pre` → 若有 post 则 `on_post`。同一拍不会既 pre 又 post（间隔 10 ms）。

因果配对重复约十几次，权重从 0.4 爬向 1，但被 `clip` 顶住。若把 `+10` 改成 `-10`，曲线应往下走——那是 LTD，本 demo 不画第二条，避免图挤。

时间环本体：

```python
for i in range(n):
    syn.decay(dt_ms)
    if i in pre_times:
        syn.on_pre()
    if i in post_times:
        syn.on_post()
    W[i] = syn.w
t = np.arange(n) * dt_ms
```

- **`np.empty(n)`**：先开好再填，比 `append` 快、长度固定。每步都存 `syn.w`，曲线才连续；若只在发放时记录，图会变成稀疏点。
- **`t = arange(n) * dt_ms`**：下标换回 ms。`set_ylim(0, 1.05)` 给饱和留一点头。
- 窗图与轨迹图是两条独立故事：前者扫 $\Delta t$，后者固定 $+10$ ms 看累积。`pairwise_update` 只在打印里用一次，验证窗函数与单次更新一致。

`A_MINUS=0.012 > A_PLUS=0.01`：对称窗会在噪声配对下净增长；略偏 LTD 是常见玩具选择。$\tau_+=\tau_-=20$ 让窗在 $\pm 80$ ms 处已经几乎贴零，`linspace` 范围够用。

---

### 关键概念速查表

| 概念 | 数学 / 直觉 | 代码 |
|------|-------------|------|
| $\Delta t>0$ | 因果 LTP | `A_PLUS * exp(-dt/tau)` |
| $\Delta t<0$ | 反因果 LTD | `-A_MINUS * exp(dt/tau)` |
| 痕迹 | 在线指数窗 | `pre_tr` / `post_tr` |
| `clip` | 权重盒约束 | `[W_MIN, W_MAX]` |
| `linspace(-80,80)` | 扫窗 | `stdp_window.png` |
| `set` 存发放下标 | $O(1)$ 查询 | `pre_times` |
| `r'$\Delta t$'` | 原始字符串 + 数学 | 轴标签 |
| 重复 +10 ms | 权重上升 | `stdp_pair_trace.png` |

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/neuro/stdp/code/demo.py`
