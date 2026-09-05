---
title: "量子网络 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 量子网络 — demo.py 代码详解

<a href="/notebook/code/quantum/network/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/quantum/network/code
python demo.py
```

CPU、NumPy。两张图：`teleport_fidelity.png`（随机纯态传态保真度贴近 1）和 `bb84_qber.png`（无窃听 vs Eve 拦截-重发的筛后误码）。三比特态矢量显式演化，不用量子线路库。

## 代码逐段详解

### 第1步：随机纯态与门

```python
def random_qubit():
    z = np.random.randn(2) + 1j * np.random.randn(2)
    return z / np.linalg.norm(z)
```

- **两个独立实高斯再拼成复数**：这是 Haar 随机单比特纯态（归一化后）的常用采样。`randn` 是 $\mathcal{N}(0,1)$。
- **`np.linalg.norm(z)`**：欧氏模。除完长度为 1。

`ket0` 返回 `[1,0]`。`H`、`CNOT`、`X`、`Z` 与计算章同一套矩阵。比特约定：0 = Alice 数据，1 = Alice 纠缠半边，2 = Bob（小端：下标的最低位是比特 0）。

---

### 第2步：`apply_3` — 把 1/2 比特门嵌进 3 比特

三比特态长度 $2^3=8$。教学优先于速度：按计算基展开，不必构造 8×8 稀疏矩阵。

```python
bits = [(i >> k) & 1 for k in range(n)]
j = sum(bit << k for k, bit in enumerate(bits2))
```

- **语法 `i >> k`**：整数 $i$ 右移 $k$ 位。**`& 1`**：取出最低位，即第 $k$ 个比特。
- **`bit << k`**：再把比特表拼回下标。
- 单比特门：对每个基矢 $i$，只改 `wires[0]` 那一位，振幅乘 `op[:, b]` 的第 `nb` 个元素（$U$ 的第 $b$ 列）。`abs(amp)<1e-15` 的项跳过。
- 两比特门：`local = bits[c]*2 + bits[t]` 把控制、目标收成 0…3，去乘 4×4 的列，再写回：

```python
bits2[c] = (nl >> 1) & 1
bits2[t] = nl & 1
```

`op.shape == (2, 2)` 走单比特分支，否则当两比特。CNOT 是 `(4, 4)`。`wires=(0, 1)` 表示控制 0、目标 1。单比特 `wires` 必须是一元元组 `(0,)`，逗号不能少（否则变成整数）。

---

### 第3步：`teleport_once` — 线路逐步对应

1. 制备 Bell：$|\Phi^+\rangle_{12}=\mathrm{CNOT}(H\otimes I)|00\rangle$，再与 $|\psi\rangle$ Kronecker 成 8 维。
2. Alice：`apply_3(CNOT, state, (0, 1))` 然后 `apply_3(H, state, (0,))`。
3. 按 Born 抽 8 个基之一：`outcome = choice(8, p=...)`。
4. **`m0 = outcome & 1`**，**`m1 = (outcome >> 1) & 1`**：最低位是比特 0，再下一位是比特 1。
5. 固定 $m_0,m_1$，把比特 2 的振幅收进 `bob`：

```python
if (i & 1) == m0 and ((i >> 1) & 1) == m1:
    bob[(i >> 2) & 1] += amp
```

6. 归一化后 Pauli 修正：`if m1: bob = X @ bob`；`if m0: bob = Z @ bob`。与标准传态表一致（$X^{m_1}Z^{m_0}$）。

`nrm < 1e-12` 时退回原 `psi`：数值上不该发生，防除零。无噪声时保真度应极接近 1（抽样投影后修正是幺正的）。

```python
def fidelity(a, b):
    return float(np.abs(np.vdot(a, b)) ** 2)
```

**`np.vdot`**：对第一个参数取共轭再点乘，即 $\langle a|b\rangle$。`abs**2` 是纯态保真度。不要用 `dot`（不对第一个共轭）。

```python
fs = [fidelity(psi, teleport_once(psi)) for psi in (random_qubit() for _ in range(n))]
```

内层是生成器，每个态用完即丢。`demo_teleport(n=40)`：折线图 + 水平虚线 1.0。`set_ylim(0.7, 1.05)` 把纵轴拉近 1。平均保真度打印到终端。

---

### 第4步：`bb84_qber` — 筛后误码

简化 BB84：Alice 随机比特、随机基（0=Z，1=X）；Bob 随机基。匹配基才留下（sift）。

```python
alice_bits = np.random.randint(0, 2, n_bits)
alice_bases = np.random.randint(0, 2, n_bits)
```

- **`randint(0, 2, n)`**：半开区间，结果是 0 或 1。

无 Eve：基不一致则 50% 翻比特（Bob 测错基），一致则原样。有 Eve：Eve 也随机基测量再转发——相对 Alice 错基 50% 错，Bob 相对 Eve 再错一次。

```python
mismatch = eve_bases != alice_bases
flip = mismatch & (np.random.rand(n_bits) < 0.5)
received = np.bitwise_xor(received, flip.astype(int))
sift = alice_bases == bob_bases
return float(np.mean(received[sift] != alice_bits[sift]))
```

- **`!=` 得到布尔掩码**，再与「均匀随机 < 0.5」做 `&`：只在基不一致时才有机会翻。
- **`received = alice_bits.copy()`**：先拷一份再 xor，不改 Alice 的真值。
- **`bitwise_xor`**：0/1 翻转。`flip` 是布尔，`.astype(int)` 才能 xor。
- **布尔索引 `received[sift]`**：只留基一致的那些比特。QBER = 这些位置上与 Alice 不同的比例。
- **`if not np.any(sift): return 0.0`**：避免对空切片 `mean` 得到 `nan`。

无窃听、只计 sift：误码应接近 0。拦截-重发会把筛后误码抬到约 25% 量级。`demo_bb84` 各跑 12 次 `boxplot`。这不是完整 BB84（没有隐私放大），只对照「偷听会在公开比对时露馅」。

---

### 关键概念速查表

| 概念 | 数学 / 直觉 | 代码 |
|------|-------------|------|
| Haar 单比特 | 复高斯归一化 | `random_qubit` |
| `>>` / `& 1` | 取第 $k$ 比特 | `apply_3`、测量 |
| `(0,)` | 一元元组 | 单比特 `wires` |
| 传态修正 | $X^{m_1}Z^{m_0}$ | `teleport_once` 末尾 |
| `vdot` | $\langle a\|b\rangle$ | `fidelity` |
| sift | 基一致才留 | `alice_bases == bob_bases` |
| 拦截-重发 | 两次错基 | `eve=True` 两段 flip |
| `bitwise_xor` | 翻转 0/1 | 模拟测错 |

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/quantum/network/code/demo.py`
