---
title: "量子存储 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 量子存储 — demo.py 代码详解

<a href="/notebook/code/quantum/memory/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/quantum/memory/code
python demo.py
```

CPU、NumPy。两张图：`t1_t2_decay.png`（$T_1$ 布居 vs $T_2$ 相干）和 `write_store_read.png`（等待越久读出保真度越低）。玩具参数 `T1=1.0`、`T2=0.4`（任意时间单位），不是某实验室拟合。$T_2<T_1$：相位通常比能量掉得更快。

## 代码逐段详解

### 第1步：`amp_damping` — 振幅阻尼（$T_1$）

激发态以概率 $p=1-e^{-t/T_1}$ 「掉下来」。Kraus：

$$
E_0=\begin{pmatrix}1&0\\0&\sqrt{1-p}\end{pmatrix},\quad
E_1=\begin{pmatrix}0&\sqrt p\\0&0\end{pmatrix},\quad
\rho\mapsto E_0\rho E_0^\dagger+E_1\rho E_1^\dagger
$$

```python
def amp_damping(rho, t, t1=T1):
    p = 1.0 - np.exp(-t / t1)
    e0 = np.array([[1, 0], [0, np.sqrt(1 - p)]], dtype=complex)
    e1 = np.array([[0, np.sqrt(p)], [0, 0]], dtype=complex)
    return e0 @ rho @ e0.conj().T + e1 @ rho @ e1.conj().T
```

- **$p=1-e^{-t/T_1}$**：$t=0$ 时 $p=0$，信道是恒等；$t\to\infty$ 时 $p\to 1$，所有布居落到 $|0\rangle$。
- **`e0 @ rho @ e0.conj().T`**：每一支 Kraus 都是 $E\rho E^\dagger$。两支相加保证 CPTP（迹保持）。
- 对 $\rho=|1\rangle\langle1|$，对角元 $\rho_{11}$ 应按 $e^{-t/T_1}$ 掉。这就是能量弛豫。振幅阻尼也会顺带削弱非对角（相干寿命不会长过 $2T_1$），但本图的 $T_2$ 曲线用的是下面的纯退相位，好对比两条钟。

---

### 第2步：`dephase` — 只打非对角（$T_2$）

$$
\rho_{01}(t)=\rho_{01}(0)\,e^{-t/T_2},\quad
\rho_{10}=\rho_{01}^*
$$

```python
def dephase(rho, t, t2=T2):
    out = rho.copy()
    out[0, 1] *= np.exp(-t / t2)
    out[1, 0] *= np.exp(-t / t2)
    return out
```

- **`.copy()`**：不要改调用者手里的 $\rho$。后面写-存-读会反复从 `rho0` 出发。
- **语法 `out[0, 1]`**：第 0 行第 1 列，即 $\rho_{01}$。`*=` 乘衰减因子。对角不动：纯退相位不改变布居，只丢相对相位。
- 对赤道态 $|+\rangle$，$|\rho_{01}|$ 从 $1/2$ 指数掉到 0。这比 $T_1$ 更快（`T2=0.4 < T1=1`）。

---

### 第3步：`fidelity_pure` — 纯态相对密度矩阵的保真度

对纯态 $|\psi\rangle$ 与任意 $\rho$：

$$
F=\langle\psi|\rho|\psi\rangle
$$

（纯态–纯态时这就是 $|\langle\psi|\phi\rangle|^2$ 的推广。）

```python
def fidelity_pure(rho, psi):
    return float(np.real(psi.conj() @ rho @ psi))
```

- **`psi.conj() @ rho @ psi`**：行向量 $\langle\psi|$ 乘 $\rho$ 再乘 $|\psi\rangle$，得到标量。`conj()` 在 `psi` 上，不是对 $\rho$。
- **`np.real` + `float`**：数学上 $F$ 是实数；转 Python 标量好存进 list。

---

### 第4步：`demo_t1_t2` 两条曲线

```python
rho1 = np.array([[0, 0], [0, 1]], dtype=complex)
plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
rho_plus = np.outer(plus, plus.conj())
pop1 = [np.real(amp_damping(rho1, t)[1, 1]) for t in times]
coh = [np.abs(dephase(rho_plus, t)[0, 1]) for t in times]
```

- **`[[0,0],[0,1]]`**：$\rho=|1\rangle\langle1|$。`[1,1]` 是 $\langle 1|\rho|1\rangle$。
- **列表推导**：对每个时刻单独作用信道。80 个点，矩阵 2×2，不必向量化。
- 左曲线从 1 掉向 0（$T_1$）；右曲线从 0.5 掉向 0（$|\rho_{01}|$）。同一张图两条钟，存储章节正文的「能量 vs 相位」。

---

### 第5步：写-存-读 — 两个信道叠在一起

```python
rho = dephase(amp_damping(rho0, t), t)
fids.append(fidelity_pure(rho, psi))
```

写入 $|\psi\rangle=|+\rangle$（`outer` 成 $\rho_0$），等待时间 $t$ 内**先**振幅阻尼、**再**退相位，再用同一 $|\psi\rangle$ 算保真度。读出不是另做测量模拟，而是直接 $F(\rho(t),\psi)$：理想读出、只看存储过程把态弄糊了多少。

$t=0$ 时 $F=1$；$t$ 增大两条噪声都加重，$F$ 单调下降。这就是「等得越久越糊」，量子内存的核心约束。

`times = np.linspace(0, 3.0, 80)`（T1/T2 图）和 `linspace(0, 2.5, 40)`（写-存-读）上限不同：前者要看到 $T_1=1$ 下布居掉到接近 0，后者保真度掉到「已经很糊」即可，不必同一横轴。列表推导里 `amp_damping(rho1, t)[1, 1]`：先得到 2×2，再取 $\rho_{11}$。`np.abs(...)` 取相干模长，因为退相位后 $\rho_{01}$ 仍可能带相位，模才是「还剩多少相干」。

Kraus 完备性：$E_0^\dagger E_0+E_1^\dagger E_1=I$（对任意 $p\in[0,1]$）。代码没有断言这一点，但 `sqrt(1-p)` 与 `sqrt(p)` 就是为了满足它。$p$ 因 `exp` 不会超出 $[0,1)$，不必再 clip。

两张图都 `tight_layout` + `dpi=140`。入口连续调用 `demo_t1_t2()` 和 `demo_write_store_read()`，没有 `main` 函数。

写-存-读里对每个 $t$ 都从**同一份** `rho0` 出发，而不是把上一步的 $\rho$ 再送进信道。后者会把时间积分错成「再阻尼一次」，曲线不是 $e^{-t/T}$。`rho0 = np.outer(psi, psi.conj())` 与 overview 章构造 $|+\rangle\langle+|$ 的方式相同。

`T2=0.4 < T1=1` 写在模块级常数。若把 `T2` 改成大于 $2T_1$，物理上不自洽（振幅阻尼已经限制相干），但 `dephase` 是独立玩具信道，代码不会报错——这是教学拆分，不是完整 Lindblad。

---

### 关键概念速查表

| 概念 | 数学 / 直觉 | 代码 |
|------|-------------|------|
| $T_1$ | 布居 $e^{-t/T_1}$ | `amp_damping`，Kraus $E_0,E_1$ |
| $p=1-e^{-t/T_1}$ | 掉下来的概率 | `amp_damping` 开头 |
| $T_2$ | 非对角 $e^{-t/T_2}$ | `dephase` |
| `.copy()` | 不改原 $\rho$ | `dephase` |
| 保真度 | $\langle\psi\|\rho\|\psi\rangle$ | `fidelity_pure` |
| `outer` | 纯态 $\rho$ | `rho_plus` / `rho0` |
| 写-存-读 | 两信道串联 | `dephase(amp_damping(...), t)` |
| `[0,1]` / `[1,1]` | 相干 / 布居 | 矩阵下标 |

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/quantum/memory/code/demo.py`
