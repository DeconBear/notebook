---
title: "香农信息论 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 香农信息论 — demo.py 代码详解

<a href="/notebook/code/information/shannon/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/information/shannon/code
python demo.py
```

CPU、NumPy 即可。一张图 `bsc_capacity.png`：左是二元熵 $h_2(p)$，右是 BSC 容量 $C=1-h_2(p)$，以及固定翻转概率 $0.11$ 时互信息随输入偏置 $P(X=1)$ 的变化。对数是 $\log_2$（单位 bit），不是 `np.log`（那是自然对数，会差 $\ln 2$）。

本章算的是信道，不是分类损失。[信息论精简](/math/information/) 的 demo 扫的是交叉熵与 KL 不对称。

## 代码逐段详解

### 第1步：`h2` — 二元熵

$$
h_2(p)=-p\log_2 p-(1-p)\log_2(1-p)
$$

```python
def h2(p, eps=1e-12):
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
```

- **`np.asarray`**：既吃标量 `0.5` 也吃数组 `ps`。向量化后左图一条曲线一次算完。
- **`clip(..., eps, 1-eps)`**：$p=0$ 时 $\log p\to-\infty$。物理上 $h_2(0)=0$，数值上要躲开。`eps=1e-12` 对曲线两端够接近 0。
- **`np.log2`**：底为 2。写成 `-p*np.log(p)/np.log(2)` 等价，更丑。
- **前面的负号**：熵公式自带。漏掉会得到负「容量」。
- **语法 `1.0 - p`**：写成 `1-p` 也对；`1.0` 强调浮点。

练习 `binary_entropy(0.5)` 应精确到 $1$（在 clip 不影响 $0.5$ 的前提下）。

---

### 第2步：BSC 容量

```python
def bsc_capacity(p):
    return 1.0 - h2(p)
```

一次使用信道最多送 $1$ bit 的输入。噪声吃掉 $h_2(p)$ bit 的不确定度，剩下就是 $C$。$p$ 与 $1-p$ 对称：$C(0.11)=C(0.89)$，因为翻转标签等价于把输出取反。

---

### 第3步：`mutual_info_bsc`

BSC 上 $H(Y\mid X)=h_2(p_{\mathrm{flip}})$（与输入无关），故

$$
I(X;Y)=H(Y)-h_2(p_{\mathrm{flip}}).
$$

$Y$ 的边缘：输入为 $1$ 且未翻，或输入为 $0$ 且翻了。

```python
def mutual_info_bsc(p_flip, p_x=0.5):
    py = p_x * (1 - p_flip) + (1 - p_x) * p_flip
    return h2(py) - h2(p_flip)
```

- **`p_x`**：$\Pr(X=1)$，默认公平。
- **`py`**：$\Pr(Y=1)$。公平输入时 `py=0.5`，与 $p_{\mathrm{flip}}$ 无关，于是 $I=1-h_2(p)$，即容量。
- **偏置输入**：$H(Y)<1$，$I$ 低于容量。右图绿线在 $P(X=1)=0.5$ 处应碰到虚线 $C(0.11)$。

没有对 $p(x)$ 做数值最大化：二元信道对称，最大值已知在公平输入。扫 `px` 只为把「容量是互信息的上包络上的一点」画出来。

---

### 第4步：作图与打印

```python
    ps = np.linspace(0.0, 1.0, 201)
    print(f'公平比特 H2(0.5)={h2(0.5):.4f} bit')
    print(f'BSC p=0.11 容量 C={bsc_capacity(0.11):.4f} bit/use')
```

- **201 点**：含端点。clip 后两端 $h_2$ 接近 0，看起来像触轴。
- **`:.4f`**：四位小数。$h_2(0.5)$ 应显示 `1.0000`。

```python
    axes[0].plot(ps, h2(ps), color='#2E86AB', lw=2)
    axes[0].axvline(0.5, color='k', ls='--', alpha=0.3)
```

竖线标峰值位置。左轴标签用 raw 字符串 `r'$h_2(p)$ (bit)'`，否则 `\b` 会被当成转义。

```python
    axes[1].plot(ps, bsc_capacity(ps), color='#C1666B', lw=2, label='C=1-h2(p)')
    px = np.linspace(0.01, 0.99, 40)
    i_vals = [mutual_info_bsc(0.11, p) for p in px]
    axes[1].plot(px, i_vals, color='#27AE60', lw=2, label='I(X;Y), p_flip=0.11')
    axes[1].axhline(bsc_capacity(0.11), color='#27AE60', ls=':', alpha=0.7)
```

- **横轴两用**：红线的 $x$ 是翻转概率 $p$；绿线的 $x$ 是 $P(X=1)$。标题写成「p 或 P(X=1)」就是承认这点。不要把绿线理解成「容量随输入噪声变」。
- **绿线避开 $0$ 和 $1$**：`0.01..0.99`，极端偏置下 $Y$ 也几乎确定，$I\to 0$。
- **列表推导**：`p_flip` 固定 $0.11$，只扫 `p_x`。四十个点够光滑。

`0.11` 没有特殊公式含义，是一个让 $C$ 明显小于 $1$ 又明显大于 $0$ 的数，曲线好看。

---

### 关键概念速查表

| 概念 | 数学 / 直觉 | 代码 |
|------|-------------|------|
| 二元熵 | $-p\log_2 p-(1-p)\log_2(1-p)$ | `h2` |
| `log2` | 单位 bit | `np.log2` |
| `clip` | 躲开 $\log 0$ | `eps=1e-12` |
| 向量化 | 整条 $p$ 曲线 | `h2(ps)` |
| BSC 容量 | $1-h_2(p)$ | `bsc_capacity` |
| $H(Y\mid X)$ | 即 $h_2(p_{\mathrm{flip}})$ | `h2(p_flip)` |
| $P(Y=1)$ | $p_x(1-p)+(1-p_x)p$ | `py = ...` |
| $I(X;Y)$ | $H(Y)-H(Y\mid X)$ | `mutual_info_bsc` |
| 公平输入 | 达到 $C$ | 默认 `p_x=0.5` |
| `axvline` / `axhline` | 标 $0.5$ 与 $C$ | 左竖右横 |

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/information/shannon/code/demo.py`
