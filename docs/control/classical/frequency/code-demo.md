---
title: "频域分析 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 频域分析 — demo.py 代码详解

<a href="/notebook/code/control/classical/frequency/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/control/classical/frequency/code
python demo.py
```

CPU、NumPy 即可。一张图 `bode.png`：上幅频（dB）、下相频（度），横轴对数 $\omega$。植物是 $K\cdot\omega_n^2/(s^2+2\zeta\omega_n s+\omega_n^2)$，$K=8$，$\omega_n=4$，$\zeta=0.3$。$K>1$ 才有 0 dB 穿越；$s=j\omega$ 直接代入，不用 `scipy.signal.bode`。

## 代码逐段详解

### 第1步：复数 $j$ 与向量化 $G$

```python
KGAIN = 8.0  # 直流增益>1，幅频才会穿过 0 dB

def G_jw(omega):
    """向量化：omega 可以是数组。开环再乘 K，否则 |G(0)|=1，穿越频率退化。"""
    s = 1j * omega
    return KGAIN * (WN ** 2) / (s ** 2 + 2.0 * ZETA * WN * s + WN ** 2)
```

- **`1j`**：Python 的虚数单位。`j` 单独不是虚数，必须写成数字后缀 `1j`。
- **`omega` 可以是数组**：NumPy 把 `1j * omega` 变成复数数组，除法逐点做。一次算出整条 Bode，不必 `for`。
- 分母 $s^2+2\zeta\omega_n s+\omega_n^2$ 在 $s=j\omega$ 时一般不为 $0$（极点不在虚轴上），所以能除。

---

### 第2步：dB 与辐角

```python
def mag_db(g):
    return 20.0 * np.log10(np.abs(g))

def phase_deg(g):
    return np.angle(g, deg=True)
```

- **`np.abs`**：复数模 $\sqrt{a^2+b^2}$，不是绝对值函数在实数上的特例写法，对复数同样适用。
- **`np.log10`**：常用对数。$20\log_{10}$ 来自功率：$|G|^2$ 再取 $10\log_{10}$ 等于 $20\log_{10}|G|$。
- **`np.angle(..., deg=True)`**：辐角，单位度。默认是弧度。二阶系统相位从 $0^\circ$ 走到 $-180^\circ$。

---

### 第3步：网格上估 $\omega_c$ 与相位裕度

```python
w = np.logspace(-1, 2, 400)
...
crossed = np.where((mag[:-1] > 0.0) & (mag[1:] <= 0.0))[0]
idx = int(crossed[0]) if crossed.size else int(np.argmin(np.abs(mag)))
wc = float(w[idx])
pm = float(ph[idx] + 180.0)
```

- **`np.logspace(-1, 2, 400)`**：$10^{-1}$ 到 $10^{2}$，对数均匀 400 点。Bode 横轴就是这样标的。
- **不要用 `argmin(|mag|)`**：直流增益接近 $0\,\mathrm{dB}$ 时，整段低频都靠近 $0$，会把 $\omega_c$ 误标在网格左端。应找「从正 dB 跨到负 dB」的第一个点。
- **`PM \approx \phi(\omega_c)+180^\circ`**：相位是负的，加上 $180$ 得到「离 $-180^\circ$ 还剩多少」。网格近似，打印时会看到一个粗值。

`semilogx`：只对 $x$ 取对数，曲线在十年频程上看起来才是教材里的折线感。`sharex=True` 让上下两图对齐同一 $\omega$。`which='both'` 让主次网格都淡淡画出来。

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/control/classical/frequency/code/demo.py`
