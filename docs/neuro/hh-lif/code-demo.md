---
title: "HH 与 LIF：把细胞写成方程 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# HH 与 LIF：把细胞写成方程 — demo.py 代码详解

<a href="/notebook/code/neuro/hh-lif/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/neuro/hh-lif/code
python demo.py
```

CPU、NumPy。四张图：`hh_trace.png` / `hh_fi.png`（枪乌贼 HH 阶跃电流与 I–f），`lif_trace.png` / `lif_fi.png`（带不应期的电流型 LIF）。前向欧拉，没有 odeint。单位：HH 用 mV、ms、µA/cm²；LIF 电流按 nA 量级的玩具参数。

## 代码逐段详解

### 第1步：门控速率 $\alpha,\beta$ — 为什么要护着 $x\approx 0$

经典 HH 的 $n,m$ 通道里有 $\frac{x}{e^{x/10}-1}$。$x\to 0$ 时分子分母都 → 0，计算机会除零。

```python
def _alpha_n(V):
    x = 10.0 - (V + 65.0)
    if abs(x) < 1e-6:
        return 0.1
    return 0.01 * x / (np.exp(x / 10.0) - 1.0)
```

- **`V + 65`**：原论文把静息放在 0；本代码静息 $-65$ mV，平移后公式与教科书一致。
- **`abs(x) < 1e-6`**：洛必达极限：$\alpha_n$ 在奇点附近用常数 `0.1`，`_alpha_m` 用 `1.0`。不是随便垫的数，是 $x\to 0$ 的解析极限。
- `_beta_n`、`_alpha_h`、`_beta_h` 没有这种奇点，直接 `exp`。

```python
def steady_gates(V):
    ...
    return an / (an + bn), am / (am + bm), ah / (ah + bh)
```

稳态 $n_\infty=\alpha/(\alpha+\beta)$。`reset` 时把门设成静息稳态，避免一开始就有假瞬态。

---

### 第2步：`HodgkinHuxley.step` — 电流与欧拉

$$
C_m \dot V = I_\mathrm{ext}-g_\mathrm{Na}m^3 h(V-E_\mathrm{Na})-g_\mathrm{K}n^4(V-E_\mathrm{K})-g_L(V-E_L)
$$

$$
\dot n = \alpha_n(V)(1-n)-\beta_n(V)n
\quad(\text{$m,h$ 同形})
$$

```python
I_Na = self.g_Na * (m ** 3) * h * (V - self.E_Na)
I_K = self.g_K * (n ** 4) * (V - self.E_K)
I_L = self.g_L * (V - self.E_L)
self.V = V + dt * (I_ext - I_Na - I_K - I_L) / self.C_m
self.n = n + dt * (_alpha_n(V) * (1.0 - n) - _beta_n(V) * n)
```

- **语法 `m ** 3`**：逐元素（此处是标量）立方。钠激活是 $m^3$，钾是 $n^4$，和教科书幂次一致。
- **门控用旧电压 `V` 更新**：前向欧拉显式格式。`dt=0.01` ms 对 HH 尖峰够用；再大容易数值炸。
- **`I_ext - I_Na - I_K - I_L`**：离子电流在公式里已经含 $(V-E)$，方向是「正电流使膜去极化」的 HH 惯例（外加电流为正）。
- 参数：`g_Na=120, g_K=36, g_L=0.3`，`E_Na=50, E_K=-77, E_L=-54.387`，`C_m=1`，都是标准枪乌贼轴突。

`simulate(I)`：`reset` 后对电流数组逐步 `step`，返回 `t = arange(len(I))*dt` 和电压轨迹。

---

### 第3步：HH 的 I–f — 过 0 mV 的上升沿计数

```python
above = V[mask] >= 0.0
nspk = int(np.sum((~above[:-1]) & above[1:]))
rates.append(nspk / ((t_ms - warmup) / 1000.0))
```

- **`mask = t >= warmup`**：丢掉前 80 ms，避免初始瞬态当尖峰。
- **过 0 mV**：HH 超射远高于 0，用 0 当阈值比再用 $V_\mathrm{th}$ 简单。LIF 则用自己的布尔 `spk`。
- **语法 `above[:-1]` / `above[1:]`**：相邻两拍。`(~above[:-1]) & above[1:]` 是「上一拍还没过、这一拍过了」= 上升沿。`~` 按位/逻辑非，`&` 必须用于布尔数组（不要写 `and`）。
- **除以秒**：`(t_ms - warmup)/1000` 把 ms 换成 s，发放率才是 Hz。

---

### 第4步：`LeakyIntegrateFire` — 漏积分、阈值、不应期

$$
\tau \dot V = -(V-V_\mathrm{rest}) + R I,\quad
V\ge V_\mathrm{th}\Rightarrow V\leftarrow V_\mathrm{reset},\ \text{锁 } t_\mathrm{ref}
$$

```python
if self.ref_left > 0:
    self.ref_left -= self.dt
    self.V = self.V_reset
    return self.V, False
self.V = self.V + self.dt * (-(self.V - self.V_rest) + self.R * I_ext) / self.tau
if self.V >= self.V_th:
    self.V = self.V_reset
    self.ref_left = self.t_ref
    return self.V_reset, True
```

- **不应期内电压钉在复位值**，返回 `False`（这拍不算发放）。`ref_left -= dt` 倒数。
- **欧拉**：`dV = dt * (...)/tau`。`R * I_ext` 是电流贡献的稳态偏移。
- **越阈立刻复位**并打开不应期 `t_ref=2` ms，返回 `(V_reset, True)`。轨迹上尖峰是「刚到阈值就掉下去」，看不到 HH 那种超射波形——这是 LIF 的定义，不是 bug。

```python
@staticmethod
def rheobase():
    return (-50.0 - (-70.0)) / 20.0
```

流变阈值：稳态 $V_\infty=V_\mathrm{rest}+R I$，令其等于 $V_\mathrm{th}$ 得 $I_\mathrm{rh}=(V_\mathrm{th}-V_\mathrm{rest})/R=1$ nA。静态方法：不需要实例状态。低于 rheobase，无限时间也发不了。

LIF 的 `f_i_curve` 直接 `np.sum(spk[t>=warmup])`，因为 `step` 已经给了布尔尖峰，不必再检测上升沿。

---

### 第5步：`main` 两套刺激

```python
hh = HodgkinHuxley()
n = int(50.0 / hh.dt)
I = np.zeros(n)
I[int(10 / hh.dt):int(40 / hh.dt)] = 10.0
```

- **语法 `I[a:b] = 10`**：切片赋值。10–40 ms 注入 10 µA/cm²，其余 0。HH 应在这段打出若干动作电位。
- LIF：`I_l[int(20/lif.dt):] = 1.5`，从 20 ms 起到结束恒定 1.5 nA（高于 rheobase=1）。

`_save_trace` / `_save_fi` 只是画图封装：`plot` + 中文轴标签 + `savefig`。`r'$I_{\mathrm{ext}}$ ...'` 是 matplotlib 数学字符串。

---

### 关键概念速查表

| 概念 | 数学 / 直觉 | 代码 |
|------|-------------|------|
| 门控极限 | $x\to 0$ 用洛必达 | `abs(x)<1e-6` |
| $m^3,n^4$ | 钠 / 钾幂次 | `m ** 3`, `n ** 4` |
| 前向欧拉 | $x+\Delta t\,f(x)$ | `step` |
| 上升沿计数 | 过 0 mV | `(~above[:-1]) & above[1:]` |
| LIF | 漏积分 + 复位 | `LeakyIntegrateFire` |
| 不应期 | 锁 `t_ref` | `ref_left` |
| rheobase | $(V_\mathrm{th}-V_\mathrm{rest})/R$ | `rheobase()` |
| `np.full` | 恒定电流数组 | `f_i_curve` |

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/neuro/hh-lif/code/demo.py`
