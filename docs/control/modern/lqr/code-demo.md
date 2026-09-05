---
title: "LQR 与最优控制 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# LQR 与最优控制 — demo.py 代码详解

<a href="/notebook/code/control/modern/lqr/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/control/modern/lqr/code
python demo.py
```

CPU、NumPy 即可。一张图 `lqr_kalman.png`：左是位置（无控制 / 全状态 LQR / LQR+卡尔曼，外加滤波位置估计虚线），右是加速度指令。植物是离散双积分器 $\ddot x=u$，状态 $[位置,速度]$，只测带噪位置。Riccati 与卡尔曼都手写，不调用控制工具箱。

`np.random.seed(42)` 让过程噪声、测量噪声可复现。三次 `simulate()` 会依次消耗随机数，所以三条轨迹**不是**同一噪声实现上的对照，而是三种策略各自的典型样本。

## 代码逐段详解

### 第1步：离散双积分器与权重

```python
DT = 0.05
N = 80
A = np.array([[1.0, DT], [0.0, 1.0]])
B = np.array([[0.5 * DT ** 2], [DT]])
H = np.array([[1.0, 0.0]])  # 只测位置
Q = np.diag([4.0, 0.2])
R = np.array([[0.15]])
QN = np.diag([1e-4, 1e-3])  # 过程噪声
RN = np.array([[0.04]])     # 测量噪声
```

- **`A`**：无控制时位置 $+=$ 速度 $\times\Delta t$，速度不变。
- **`B`**：$\ddot x=u$ 在一步内的精确离散：速度 $+=u\Delta t$，位置还要加上 $\tfrac12 u\Delta t^2$。`B` 是列向量 shape `(2,1)`，后面与行向量乘法才对得上。
- **`H`**：观测矩阵 $1\times 2$。`z = H @ x` 抽出位置。
- **`Q` / `R`**：LQR 权重。位置罚 $4$ 大于速度 $0.2$，更在乎回到 $x=0$；`R=0.15` 不太吝啬力。这里的 `R` 与经典章参考位置同名，含义完全不同。
- **`QN` / `RN`**：卡尔曼用的噪声协方差，**不是** LQR 的 $Q,R$。过程噪声速度通道更大（`1e-3` vs `1e-4`），测量标准差 $\sqrt{0.04}=0.2$。

语法：`DT ** 2` 是平方，不是 `^`。`np.diag` 把一维列表变成对角阵。

---

### 第2步：`dare_lqr` — 迭代离散 Riccati

$$
P \leftarrow Q + A^\top P A - A^\top P B(R+B^\top P B)^{-1} B^\top P A
$$

$$
K = (R+B^\top P B)^{-1} B^\top P A
$$

```python
def dare_lqr(A, B, Q, R, iters=200):
    P = Q.copy()
    for _ in range(iters):
        BtP = B.T @ P
        P = Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.solve(R + BtP @ B, BtP @ A)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K
```

- **`P = Q.copy()`**：从代价矩阵本身当初始 $P$。迭代 $200$ 次对 $2\times 2$ 足够收敛；不求特征值。
- **`B.T @ P`**：`@` 是矩阵乘。`B.T` 是 $1\times 2$。
- **`np.linalg.solve(M, y)`**：解 $Mx=y$，等价于 $M^{-1}y$，但比显式 `inv` 稳。这里 $R+B^\top PB$ 是 $1\times 1$。
- **为什么返回 $K$ 不返回 $P$？** 控制律只用 $K$。练习 `lqr_u` 也只吃 $K,x$。

连续 LQR 写积分 $\int(x^\top Qx+u^\top Ru)$；离散是求和。图解用连续符号，实现必须与 `A,B` 同一时间尺度。

---

### 第3步：`simulate` — 三条策略共用循环

```python
def simulate(use_lqr=True, use_kf=False, x0=None):
    K = dare_lqr(A, B, Q, R)
    x = np.array([-1.2, 0.8]) if x0 is None else np.array(x0, dtype=float)
    xhat = np.array([0.0, 0.0])
    P = np.eye(2)
```

- **每次重算 $K$**：浪费但短。$K$ 与初值无关。
- **真状态初值** $(-1.2,0.8)$：既偏又朝外飞。
- **`xhat` 从原点、`P=I` 开始**：滤波器一开始不信真值，所以「LQR+KF」前几步会和全状态 LQR 分叉。这是故意的：看出估计在追。
- **`dtype=float`**：若传入整数 `x0`，后面 `A @ x` 仍会提升，但养成习惯。

控制律：

```python
        if not use_lqr:
            u = 0.0
        elif use_kf:
            u = float((-K @ xhat).item())
        else:
            u = float((-K @ x).item())
```

- **`u = -K x`**：LQR。`K` shape `(1,2)`，`K @ x` 是 $1\times 1$ 数组；`.item()` 取出 Python 标量，再 `float`。
- **滤波时用 `xhat` 不用 `x`**：分离原理的最小演示。真 $x$ 只用来生成观测和画图。

植物 + 噪声：

```python
        w = np.random.multivariate_normal([0, 0], QN)
        x = A @ x + B.ravel() * u + w
        z = float((H @ x).item()) + np.random.normal(0.0, np.sqrt(RN[0, 0]))
```

- **`B.ravel() * u`**：把 `(2,1)` 摊成 `(2,)` 再乘标量 $u$，与 `x` 同 rank，避免广播出 `(2,2)`。写成 `B @ [[u]]` 也行，更啰嗦。
- **`multivariate_normal`**：二维过程噪声，协方差 `QN`。
- **测量**：均值 $Hx$，标准差 $\sqrt{R_n}$。`H @ x` 形状是 `(1,)`，不是 0 维，直接 `float(...)` 会 TypeError，所以先 `.item()`。`RN[0,0]` 取出标量 $0.04$。

---

### 第4步：卡尔曼预测 / 更新

```python
        if use_kf:
            xhat = A @ xhat + B.ravel() * u
            P = A @ P @ A.T + QN
            S = H @ P @ H.T + RN
            Kg = (P @ H.T) / S
            xhat = xhat + Kg.ravel() * (z - float((H @ xhat).item()))
            P = (np.eye(2) - Kg @ H) @ P
```

- **预测**：控制输入用**已经算出的** $u$（它依赖于预测前的 $\hat x$）。$P\leftarrow APA^\top+Q_n$，不确定变大。
- **`S`**：新息协方差，本问题是 $1\times 1$。
- **`Kg = (P @ H.T) / S`**：因为 $S$ 是标量，除法合法。多维测量必须改成 `solve`。卡尔曼增益 shape `(2,1)`。
- **新息** $z-H\hat x$：测量减「该看到的」。正的新息 → 把位置估计往上拉。
- **Joseph 形式没写**：用的是 $(I-K_g H)P$。数值上 Joseph 更稳，二维演示够用。
- **无滤波分支**：仍采样 $z$，但丢掉。随机数序列仍前进——这是三条轨迹噪声不对齐的原因之一。

轨迹：`xs` / `xhats` 含初值，长度 `N+1`；`us` 长度 `N`。画位置用 `len(xs)*DT` 当时间轴。

---

### 第5步：作图

```python
    axes[0].plot(t, xs0[:, 0], color='#95A5A6', label='无控制')
    axes[0].plot(t, xs1[:, 0], color='#27AE60', lw=2, label='LQR（真状态）')
    axes[0].plot(t, xs2[:, 0], color='#2980B9', lw=2, label='LQR + 卡尔曼')
    axes[0].plot(t, xh2[:, 0], color='#2980B9', ls='--', alpha=0.7, label='卡尔曼位置估计')
```

- **`[:, 0]`**：位置通道。速度在 `[:, 1]`，本图不画。
- **虚线 `xh2`**：应贴着蓝实线，前期偏差更大。
- **右图只画 `us1`/`us2`**：无控制 $u=0$ 是平线，省略。滤波后的 $u$ 更抖，因为 $\hat x$ 被测量扯动。

终端打印 `K.ravel()` 和三条终态。全状态 LQR 终态应靠近 $0$；无控制不会。

---

### 关键概念速查表

| 概念 | 数学 / 直觉 | 代码 |
|------|-------------|------|
| 双积分器 | $\ddot x=u$ 离散 | `A`,`B` |
| LQR | $u=-Kx$ | `-K @ x` |
| Riccati | 迭代 DARE | `dare_lqr` |
| `solve` | 稳的左除 | `np.linalg.solve` |
| 只测位置 | $H=[1,0]$ | `H @ x` |
| 预测 | $A\hat x+Bu$，$APA^\top+Q_n$ | `use_kf` 上半 |
| 更新 | $K_g$ 与新息 | `Kg = (P @ H.T) / S` |
| `.item()` | $1\times1$ 变标量 | 取 $u$ |
| `ravel` | 列向量变一维 | `B.ravel() * u` |

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/control/modern/lqr/code/demo.py`
