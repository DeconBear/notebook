---
title: "量子信息全景 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 量子信息全景 — demo.py 代码详解

<a href="/notebook/code/quantum/overview/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/quantum/overview/code
python demo.py
```

CPU、NumPy。一张图 `superposition_vs_mixture.png`：左栏直接测 $Z$，叠加 $|+\rangle$ 和完全混合 $\tfrac12 I$ 都像抛硬币；右栏先 $H$ 再测 $Z$，只有叠加被「收回」到 $|0\rangle$。这是五条量子专题共用的第一课：直方图 50/50 **不能**区分相干与混合。没有 Qiskit。

## 代码逐段详解

### 第1步：Hadamard 矩阵

$$
H=\frac1{\sqrt2}\begin{pmatrix}1&1\\1&-1\end{pmatrix},\qquad
H|0\rangle=|+\rangle,\quad H|+\rangle=|0\rangle
$$

```python
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
```

- **`dtype=complex`**：密度矩阵乘法一般会出复数。这里 $H$ 碰巧是实的，但后面 `conj().T` 的写法对一般幺正门也成立。
- **`/ np.sqrt(2)`**：四个元素一起缩放，保证 $H^\dagger H=I$。不要只除第一行。

---

### 第2步：`shots_from_diag` — 用对角元当 Born 概率

密度矩阵 $\rho$ 在计算基下测 $Z$，结果 $0/1$ 的概率是对角元 $\rho_{00},\rho_{11}$（实数、非负、和为 1）。

```python
def shots_from_diag(diag, n=2000):
    p = np.real(diag)
    p = p / p.sum()
    return np.random.choice(2, size=n, p=p)
```

- **`np.real`**：数值误差可能给对角留下 $10^{-16}$ 虚部。`choice` 要实数概率。
- **`p / p.sum()`**：再归一化一次，避免「和差一点点」导致 `choice` 报错。
- **`np.random.choice(2, size=n, p=p)`**：在 $\{0,1\}$ 里按 $p$ 抽 $n$ 次。返回长度 2000 的整数数组，后面 `mean(a==0)` 就是频率。
- 只看对角 = **丢掉相干信息**。所以左图两种 $\rho$ 直方图几乎一样。这不是弱测量，就是计算基投影。

---

### 第3步：两种 $\rho$ — 外积 vs 单位矩阵

$$
|+\rangle=\frac{|0\rangle+|1\rangle}{\sqrt2},\quad
\rho_\mathrm{sup}=|+\rangle\langle+|,\quad
\rho_\mathrm{mix}=\tfrac12 I
$$

```python
plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
rho_sup = np.outer(plus, plus.conj())
rho_mix = 0.5 * np.eye(2)
```

- **`np.outer(a, b)`**：外积 $a b^\top$。纯态密度矩阵 $|\psi\rangle\langle\psi|=$ `outer(psi, psi.conj())`。`.conj()` 是复共轭；实向量时看起来多余，公式要对。
- **`np.eye(2)`**：2×2 单位阵。`0.5 * I` 是最大混合态：对角 1/2，**非对角为 0**（没有相干）。
- $\rho_\mathrm{sup}$ 的非对角是 $1/2$，这是能被 $H$ 收回的关键。两种态的对角相同，所以直接测 $Z$ 分不出来。

```python
z_sup = shots_from_diag(np.diag(rho_sup))
z_mix = shots_from_diag(np.diag(rho_mix))
```

**`np.diag(M)`** 抽对角，得到长度 2 的一维数组。两份都应约 50/50。

---

### 第4步：先 $H$ 再测 — 共轭作用在 $\rho$ 上

对密度矩阵做幺正 $U$：$\rho\mapsto U\rho U^\dagger$。

```python
rho_sup_h = H @ rho_sup @ H.conj().T
rho_mix_h = H @ rho_mix @ H.conj().T
```

- **语法 `@`**：矩阵乘。`H.conj().T` 是 $H^\dagger$。$H$ 实对称时等于 $H$ 自己，写成 dagger 是为了和课本一致。
- **叠加**：$H|+\rangle=|0\rangle$，所以 $\rho_\mathrm{sup}$ 变成 $|0\rangle\langle0|$，测 $Z$ 几乎全是 0。
- **混合**：$H(\tfrac12 I)H^\dagger=\tfrac12 I$（单位阵与任何幺正对易）。直方图仍约 50/50。

右图并排柱：叠加的 0 柱接近 1，混合两柱仍平。这说明同一套 $Z$ 测量分不出来，换一组测量（这里是 $H$ 再 $Z$，即测 $X$）才能区分这两种态；纯态的测量也可以随机，不能说“只有混合才是真随机”。

---

### 第5步：并排柱状图

```python
for ax, (a, b), title in zip(
    axes,
    [(z_sup, z_mix), (h_sup, h_mix)],
    ['直接测 Z（都像抛硬币）', '先 H 再测 Z（叠加能「收回」）'],
):
    ax.bar([-0.2, 0.8], [np.mean(a == 0), np.mean(a == 1)], width=0.35, label='叠加 |+>')
    ax.bar([0.2, 1.2], [np.mean(b == 0), np.mean(b == 1)], width=0.35, label='混合 ½I')
```

- **`zip` 三列等长**：子图、数据对、标题。
- **`a == 0`**：布尔数组，`np.mean` 等于频率。
- **x 位置错开 ±0.2**：两组柱并排，宽度 0.35 才不重叠。`set_xticks([0, 1])` 标签仍对准 0/1。
- **`sharey=True`**：两个子图共用纵轴。**`set_ylim(0, 1)`** 锁死频率轴，避免某次抽样全是 0 时纵轴自动缩到 0.02。
- **`fig.suptitle`**：整张图的结论；子图 `set_title` 讲操作。2000 shots 涨落约 $1/\sqrt n\approx 2\%$，右图叠加「几乎全 0」会远远超出这个涨落。

入口是 `if __name__ == '__main__': demo()`，函数名是 `demo` 不是 `main`。种子 42 钉死抽样，图可复现。

---

### 关键概念速查表

| 概念 | 数学 / 直觉 | 代码 |
|------|-------------|------|
| $\|+\rangle$ | 均匀叠加 | `[1,1]/sqrt(2)` |
| 纯态 $\rho$ | $\|\psi\rangle\langle\psi\|$ | `np.outer(plus, plus.conj())` |
| 混合 | $\tfrac12 I$，无相干 | `0.5 * np.eye(2)` |
| Born（计算基） | 对角元 | `shots_from_diag` |
| `choice(..., p=)` | 按概率抽样 | 2000 shots |
| 幺正作用 | $U\rho U^\dagger$ | `H @ rho @ H.conj().T` |
| 测 $X$ | 先 $H$ 再测 $Z$ | 右图 |
| `@` | 矩阵乘 | 不是逐元素 `*` |

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/quantum/overview/code/demo.py`
