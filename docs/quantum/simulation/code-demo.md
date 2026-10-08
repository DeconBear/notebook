---
title: "量子模拟 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 量子模拟 — demo.py 代码详解

<a href="/notebook/code/quantum/simulation/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/quantum/simulation/code
python demo.py
```

CPU、NumPy。两自旋横场 Ising，比精确 $e^{-iHT}$ 与一阶 Trotter。一张图 `trotter_error.png`（对数轴：步数↑，失真度↓）。希尔伯特空间只有 4 维，对角化是精确答案不是近似。

## 代码逐段详解

### 第1步：把 $H$ 拆成可对角的两块

$$
H=A+B,\quad A=J\,Z\otimes Z,\quad B=h(X\otimes I+I\otimes X)
$$

```python
ZZ = np.kron(Z, Z)
XI = np.kron(X, I2)
IX = np.kron(I2, X)
A = J * ZZ
B = h * (XI + IX)
Htot = A + B
```

`J=1`、`h=0.7`、总时间 `T=1.2`。$A$ 在计算基对角（$ZZ$ 本征值 ±1）；$B$ 在 $X$ 基对角。分开指数化比直接对 $H$ 做 Trotter 更便宜——这正是数字模拟要把哈密顿量劈开的原因。

`np.kron` 顺序与两比特计算基 `|00⟩…|11⟩` 一致。

---

### 第2步：`expm_herm` — 厄米矩阵的 $e^{-iMt}$

$$
M=V\mathrm{diag}(\lambda)V^\dagger \Rightarrow e^{-iMt}=V\mathrm{diag}(e^{-i\lambda t})V^\dagger
$$

```python
w, v = np.linalg.eigh(m)
return v @ np.diag(np.exp(-1j * w * t)) @ v.conj().T
```

- **`eigh`**：专供厄米，特征值实。不要用通用 `eig`（可能出复本征值噪声）。
- **`-1j * w * t`**：Schrödinger 演化 $e^{-iHt}$。漏掉负号会变成时间反演。
- **`v.conj().T`**：本征矢矩阵的 dagger。`eigh` 的 `v` 列是本征矢。

精确解：`u_exact = expm_herm(Htot, T)`，一次对角化 $A+B$。

---

### 第3步：一阶 Trotter

$$
e^{-i(A+B)t}\approx\bigl(e^{-iB\Delta t}e^{-iA\Delta t}\bigr)^{n},\quad \Delta t=t/n
$$

```python
def trotter(n_steps, t=T):
    dt = t / n_steps
    u = np.eye(4, dtype=complex)
    ua, ub = expm_herm(A, dt), expm_herm(B, dt)
    for _ in range(n_steps):
        u = ub @ ua @ u
    return u
```

- **先算一次 `ua, ub`**：每步同一 $\Delta t$，不必在循环里对角化。
- **`u = ub @ ua @ u`**：右乘旧 $U$，即「再作用一层 $e^{-iA\Delta t}$ 再 $e^{-iB\Delta t}$」。次序 $B$ 在左、$A$ 在右，对应公式里 $e^{-iBdt}e^{-iAdt}$。
- **`np.eye(4)`**：从恒等开始。`for _ in range(n_steps)`：循环变量不用。

一阶演化算子误差理论上 $O(t^2/n)$（$[A,B]\neq0$）。固定 $t$ 时，下面画的纯态失真度通常为 $O(1/n^2)$，因为它对小态偏差是二阶量；两种误差度量不能混用。

---

### 第4步：失真度 $1-|\langle\psi_{\mathrm{ex}}|\psi_{\mathrm{tr}}\rangle|^2$

```python
psi0 = np.array([1, 0, 0, 0], dtype=complex)  # |00⟩
psi_exact = u_exact @ psi0
infid.append(1.0 - np.abs(np.vdot(psi_exact, psi)) ** 2)
ax.loglog(steps, infid, 'o-')
```

- **`vdot`**：共轭点积 $\psi_{\mathrm{ex}}^\dagger\psi_{\mathrm{tr}}$。普通 `dot` 对复数不共轭，会错。
- **`loglog`**：横纵都对数。步数 `1,2,4,…,32` 在对数轴上均匀。应看到近似直线往下。
- 终端 `zip(steps.tolist(), np.round(infid, 6))` 打印对照表。

`steps = np.array([1, 2, 4, 8, 16, 32])` 再 `trotter(int(n))`：`loglog` 要求正数；步数取 2 的幂，横坐标等间距。`int(n)` 因为数组元素可能是 `np.int32`，`range` 要 Python `int`（多数 NumPy 版本其实也能用，写成 `int` 更明确）。

初态 `[1,0,0,0]` 是 $|00\rangle$，不是 Bell、不是叠加。横场 $B$ 会立刻把自旋打出计算基，所以即使用 1 步 Trotter，终态也已经「动过」——误差比「$B=0$ 时 $A$ 与 $H$ 对易、任意 $n$ 都精确」更可见。

没有随机电路、没有量子硬件噪声：误差纯粹来自 Trotter 劈裂。`np.random.seed(42)` 在本文件没有用到抽样。

---

### 关键概念速查表

| 概念 | 数学 / 直觉 | 代码 |
|------|-------------|------|
| $A=JZZ$ | 对角相互作用 | `J * kron(Z,Z)` |
| $B=h(XI+IX)$ | 横场 | `h * (XI+IX)` |
| `eigh` | 厄米对角化 | `expm_herm` |
| $e^{-iMt}$ | `exp(-1j*w*t)` | 对角相位 |
| Trotter | $(e^{-iBdt}e^{-iAdt})^n$ | `ub @ ua @ u` |
| $\Delta t=T/n$ | 步数越多刀越薄 | `t / n_steps` |
| 失真度 | $1-\|\langle\mathrm{ex}|\mathrm{tr}\rangle\|^2$ | `1 - abs(vdot)**2` |
| `loglog` | 看幂律 | `trotter_error.png` |
| `vdot` | 复数内积 | 不要用 `dot` |

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/quantum/simulation/code/demo.py`
