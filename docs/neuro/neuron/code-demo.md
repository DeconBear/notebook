---
title: "神经元与突触 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 神经元与突触 — demo.py 代码详解

<a href="/notebook/code/neuro/neuron/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/neuro/neuron/code
python demo.py
```

CPU、NumPy 即可。一张图 `neuron_integrate_fire.png`：几条指数突触后电位在静息电位上叠加，看会不会越过阈值。这是**分级电位的教学示意**，不是 Hodgkin–Huxley，也没有真正的尖峰复位；越过阈值只打印 `True/False` 和峰值。

## 代码逐段详解

### 第1步：常数 — 单位都是毫秒 / 毫伏

```python
np.random.seed(42)
DT = 0.1
TAU = 8.0
V_REST = -70.0
V_TH = -54.0
```

- **`DT = 0.1`**：时间步 0.1 ms。后面 `np.arange(0, 80, DT)` 得到 0…80 ms 的网格。
- **`TAU = 8.0`**：突触电流的衰减时间常数（ms）。比动作电位短、比整段 80 ms 轨迹短，叠加才看得出「来得及 / 来不及」。
- **`V_REST = -70`**：静息。**`V_TH = -54`**：阈值。间隔 16 mV，单条幅值 ~6–8 mV 的 EPSP **单独到不了**，必须时间上靠近才行——这正是「空间/时间求和」要演示的。
- 种子在本脚本里没有用到随机数；留下是全书习惯，改 arrivals 加噪声时不用再设。

---

### 第2步：`psp` — 因果指数核

到达时刻 $t_0$ 之后：

$$
\mathrm{PSP}(t)=
\begin{cases}
A\,e^{-(t-t_0)/\tau} & t\ge t_0\\
0 & t< t_0
\end{cases}
$$

```python
def psp(t, t0, amp, tau=TAU):
    x = t - t0
    out = np.zeros_like(t)
    m = x >= 0
    out[m] = amp * np.exp(-x[m] / tau)
    return out
```

- **`t - t0`**：整段时间轴相对到达时刻。$x<0$ 是「还没到」，必须是 0，否则指数会在到达前就翘起来（非因果）。
- **`np.zeros_like(t)`**：和 `t` 同形状、同 dtype 的全零。先开好再填，避免 Python 循环。
- **语法 `m = x >= 0`**：布尔掩码，shape 与 `t` 相同。`out[m] = ...` 只给「已经到达」的那些点赋值。
- **`x[m] / tau`**：只对掩码为 True 的元素做除法。`np.exp` 逐元素。
- **`amp` 可正可负**：正 = EPSP（去极化），负 = IPSP（超极化）。同一套公式，符号当兴奋/抑制。

**为什么不用 `np.heaviside`？** 掩码写法更直白：先算 $t-t_0$，再「只保留 ≥0」。教学上逐步可见。

---

### 第3步：线性叠加 — 膜电位从静息往上堆

```python
t = np.arange(0, 80, DT)
arrivals = [8.0, 14.0, 19.0, 36.0, 41.0]
amps = [6.5, 7.0, 6.8, -4.0, 8.0]
v = np.full_like(t, V_REST)
for t0, a in zip(arrivals, amps):
    v = v + psp(t, t0, a)
spiked = np.any(v >= V_TH)
```

- **`np.arange(0, 80, DT)`**：半开区间 $[0,80)$，步长 0.1。长度约 800。
- **`np.full_like(t, V_REST)`**：每个时间点先填 -70。叠加的是**相对静息的偏移**，不是从 0 开始。
- **`zip(arrivals, amps)`**：第 $k$ 次到达配第 $k$ 个幅值。前三次正、第四次 `-4`（IPSP）、第五次再正。
- **线性相加**：示意「分级电位可叠加」。真实树突有非线性，本章故意线性，才能用眼看「三次靠得近 vs 中间插一次抑制」。
- **`np.any(v >= V_TH)`**：整段有没有**任意一点**越过阈值。`True` 只表示「示意上够格发放」，**没有**把 $V$ 复位、没有画竖线尖峰。真正的积分发放在 `hh-lif` 章。

时间安排：8、14、19 ms 挤在一起（间隔 6、5 ms，小于 $\tau=8$），三次 EPSP 来得及加总；36 ms 的 IPSP 把膜往下拉；41 ms 再来一次兴奋。看图时应能看到前簇冲高、中段被紫线（抑制）压一下。

---

### 第4步：画图 — 阈值线、静息线、到达时刻

```python
ax.plot(t, v, color='#1a5276', lw=2, label='膜电位（示意）')
ax.axhline(V_TH, color='#c0392b', ls='--', label='阈值')
ax.axhline(V_REST, color='#7f8c8d', ls=':', label='静息')
for t0, a in zip(arrivals, amps):
    ax.axvline(t0, color='#27ae60' if a > 0 else '#8e44ad', alpha=0.35)
```

- **`axhline`**：水平线，贯穿 $x$ 轴。阈值虚线、静息点线，读图时有标尺。
- **`axvline(t0)`**：到达时刻竖线。`a > 0` 绿色（兴奋），否则紫色（抑制）。`alpha=0.35` 半透明，免得挡住电位曲线。
- **三元表达式 `A if cond else B`**：按幅值符号选颜色，不必再开一个列表。

终端：`是否越过阈值（示意）: True/False` 以及 `v.max()`。峰值用 `float(...)` 转成 Python 标量再打印。

**`np.arange` vs `linspace`**：这里用步长 `DT` 铺时间，和后面若改成欧拉积分时的 `dt` 一致。`linspace(0, 80, N)` 会包含右端点 80，长度由点数决定，和 `DT` 对不齐。

**为什么不复位？** 若越过阈值就把 `v` 打回静息，曲线会变成锯齿，读者容易以为这已经是 LIF。本章标题是「分级电位局部运算」：只看 EPSP/IPSP 相加，发放机制留给 `hh-lif`。`spiked` 只是一句打印，图上没有星号标记发放时刻。

`ax.grid(True, alpha=0.3)`：浅网格，方便读「离阈值还有几 mV」。`legend(fontsize=8)` 缩小图例，避免挡住峰值。`tight_layout` 后再 `savefig(..., dpi=140)`，最后 `plt.close`，和无界面脚本的惯例相同。

入口：`if __name__ == '__main__': main()`。没有第二个函数：`psp` 是唯一的核，其余都在 `main` 里串起来。

### 第5步：这张图在验证什么

保存为 `neuron_integrate_fire.png`。应看到：

1. 8–19 ms 三次绿竖线靠得很近，蓝线爬升，可能贴到或越过红虚线阈值；
2. 36 ms 紫线（`amp=-4`）把电位往下拽；
3. 41 ms 再一次 EPSP，但前面被抑制挖了一坑，不一定再越阈。

改 `TAU` 变大，指数尾巴更长，远处的 PSP 更容易叠上；改小则三次「来不及」加总。这就是时间常数的几何意义。`DT=0.1` 对纯指数核足够；它不是微分方程的稳定性限制（没有欧拉）。

---

### 关键概念速查表

| 概念 | 数学 / 直觉 | 代码 |
|------|-------------|------|
| 指数 PSP | $t\ge t_0$ 才非零 | `psp`，掩码 `x >= 0` |
| $\tau$ | 衰减快慢 | `TAU = 8.0` |
| 线性叠加 | 分级电位求和 | `v = v + psp(...)` |
| EPSP / IPSP | `amp` 正 / 负 | `amps` 第四个是 `-4` |
| `zeros_like` | 同形状全零 | 核的因果部分 |
| `full_like` | 同形状填静息 | `v` 初值 |
| `np.any` | 有没有越阈 | `spiked` |
| `axhline` / `axvline` | 水平 / 竖直线 | 阈值与到达时刻 |

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/neuro/neuron/code/demo.py`
