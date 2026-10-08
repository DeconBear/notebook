---
title: "量子计算 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 量子计算 — demo.py 代码详解

<a href="/notebook/code/quantum/computing/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/quantum/computing/code
python demo.py
```

CPU、NumPy 态矢量。两张图：`hadamard_shots.png`（$H|0\rangle$ 近 50/50）、`bell_shots.png`（$|\Phi^+\rangle$ 只出现 `00` 与 `11`）。门矩阵手写，不用 Qiskit。

## 代码逐段详解

### 第1步：门

```python
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
```

CNOT 按计算基顺序 $|00\rangle,|01\rangle,|10\rangle,|11\rangle$：前两个基矢不动，后两个对换——控制比特为 1 时翻目标。把矩阵当「列 = 输入基、行 = 输出基」读：第 3 列（0-based 下标 2，即 $|10\rangle$）是 `[0,0,0,1]`，进 $|11\rangle$；第 4 列进 $|10\rangle$。

`I2`、`X`、`Z` 在文件顶上定义了。Hadamard 演示只用 `H`；Bell 用 `I2` 做 `kron(H, I2)`。`X`/`Z` 本文件没乘到态上（那是网络章传态修正才用），留下是为了和同目录其它脚本同一套 Pauli 字母。`dtype=complex` 一律加上，避免以后改成相位门时静默丢掉虚部。

---

### 第2步：`ket` — 用 `kron` 堆比特

```python
def ket(*bits):
    v = np.array([1.0], dtype=complex)
    for b in bits:
        v = np.kron(v, np.array([1, 0] if b == 0 else [0, 1], dtype=complex))
    return v
```

- **`*bits`**：可变参数。`ket(0)` 是二维，$|0\rangle$；`ket(0,0)` 是四维 $|00\rangle$。
- **`np.kron`**：张量积。从左到右追加比特。初始 `v=[1]` 是 0 比特的标量 1，第一轮 kron 才变成真正的量子态。
- 三元表达式 `[1,0] if b==0 else [0,1]`：计算基。

---

### 第3步：`measure_shots` — Born 规则

$$
p_i=|\langle i|\psi\rangle|^2
$$

```python
p = np.abs(state) ** 2
p = np.real(p)
p = p / p.sum()
return np.random.choice(len(state), size=n_shots, p=p)
```

- **`np.abs ** 2`**：振幅模方。复数态必须先 `abs`，不能 `state**2`。
- **`choice(len(state), ...)`**：对 **基矢下标** 抽样，单比特是 `{0,1}`，两比特是 `{0,1,2,3}` 对应 `00,01,10,11`。
- 默认 2000 shots，频率的标准差为 $\sqrt{p(1-p)/n}$，按 $1/\sqrt{n}$ 缩小；方差为 $p(1-p)/n$。

---

### 第4步：Hadamard 单比特

```python
psi = H @ ket(0)
shots = measure_shots(psi)
counts = np.bincount(shots, minlength=2)
ax.bar(['|0>', '|1>'], counts / counts.sum(), ...)
```

$H|0\rangle=|+\rangle$。**`bincount(..., minlength=2)`**：即使某次全抽到 0，长度为 2 的直方图仍有 `|1>` 那根（高度 0）。除以 `sum` 变频率。终端打印振幅向量。

---

### 第5步：Bell — 先 $H$ 在控制比特，再 CNOT

$$
|\Phi^+\rangle=\mathrm{CNOT}\,(H\otimes I)\,|00\rangle=\frac{|00\rangle+|11\rangle}{\sqrt2}
$$

```python
psi = CNOT @ np.kron(H, I2) @ ket(0, 0)
```

- **`np.kron(H, I2)`**：两比特门，$H$ 作用在**高位/左比特**，$I$ 在右比特。顺序必须和 `ket(0,0)` 的 kron 约定一致。
- 再左乘 `CNOT`（4×4）。结果振幅约 `[0.707, 0, 0, 0.707]`，`01`/`10` 为 0。
- 直方图 `minlength=4`，标签 `['00','01','10','11']`。中间两根应接近 0——这就是纠缠关联，不是两个独立硬币。

`np.round(psi, 3)` 打印时去掉浮点尾巴。

`np.kron(H, I2)` 若写成 `kron(I2, H)`，Hadamard 会打在**第二个**比特上，CNOT 的控制/目标约定就会和「先叠加控制比特」对不上，直方图会出现 `00` 与 `01` 而不是 `00` 与 `11`。读代码时先确认 kron 左因子是哪一比特。

入口：`if __name__ == '__main__'` 里直接 `demo_hadamard()` 然后 `demo_bell()`，没有再包一层 `main()`。顺序无关（两套图互不覆盖文件名）。种子 42 只影响 shots 抽样，振幅打印是确定性的。种子 42 钉死抽样，直方图可复现但仍有二项涨落，不要期望柱高精确等于 0.5。

文件里的 `X`、`Z` 本演示未调用，留给和正文 Pauli 对照。`I2` 只出现在 `kron(H, I2)`：若写成 `kron(I2, H)`，H 会打到另一个比特，Bell 直方图会错。

---

### 关键概念速查表

| 概念 | 数学 / 直觉 | 代码 |
|------|-------------|------|
| `ket(*bits)` | 计算基张量积 | 循环 `kron` |
| $H\|0\rangle$ | 均匀叠加 | `H @ ket(0)` |
| Born | $p=\|c\|^2$ | `abs(state)**2` |
| `bincount` | 下标直方图 | `minlength=2` 或 `4` |
| $H\otimes I$ | 只打第一比特 | `np.kron(H, I2)` |
| CNOT | 控翻 | 4×4 置换矩阵 |
| $\|\Phi^+\rangle$ | 只 00 与 11 | `demo_bell` |
| `@` | 矩阵×态 | 不是 `*` |

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/quantum/computing/code/demo.py`
