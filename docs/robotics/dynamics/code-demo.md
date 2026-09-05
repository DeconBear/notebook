---
title: "机器人动力学 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 机器人动力学 — demo.py 代码详解

<a href="/notebook/code/robotics/dynamics/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/robotics/dynamics/code
python demo.py
```

CPU、NumPy 即可。一张图 `lagrange_2r.png`：左是无/有阻尼的 $\theta_1(t),\theta_2(t)$，右是有阻尼时末端轨迹。质量集中在杆末端，$m_1=m_2=1$，$\ell=0.8$，无主动力矩 $\tau=0$。步长 `DT=0.002` 较小，因为显式积分刚体摆容易抖。

角度约定：**$\theta=0$ 为水平**（重力力矩 $\propto\cos\theta$，水平最大）。与「竖直向下为 0」的教材画法不同，公式不要混抄。

## 代码逐段详解

### 第1步：质量矩阵 $M(\theta_2)$

平面 2R、质量在杆端：

```python
def mass_matrix(th2):
    """M(q) 2x2。θ1 不进惯性（平面旋转对称）。"""
    c2 = np.cos(th2)
    m11 = (M1 + M2) * L1 ** 2 + M2 * L2 ** 2 + 2 * M2 * L1 * L2 * c2
    m12 = M2 * L2 ** 2 + M2 * L1 * L2 * c2
    m22 = M2 * L2 ** 2
    return np.array([[m11, m12], [m12, m22]])
```

- **只传入 `th2`**：绕基座转一圈，两杆相对夹角不变则惯性张量在关节坐标里长得一样，$\theta_1$ 不出现。
- **`m12 = m21`**：对称。`m22` 只有第二质量绕肘转的 $\ell_2^2$。
- **交叉项 $\propto\cos\theta_2$**：伸直时 $\theta_2=0$，等效更「重」；折叠时交叉变号。

不要把图解里的质心距离 $\ell_c$、转动惯量 $I$ 塞进这个函数——本 demo 没有那些符号。

---

### 第2步：`h_vector` — 科氏 + 重力

方程 $M\ddot q + H = \tau$，其中

```python
def h_vector(th1, th2, w1, w2):
    s2 = np.sin(th2)
    cor = -M2 * L1 * L2 * s2
    c1 = np.cos(th1)
    c12 = np.cos(th1 + th2)
    g1 = (M1 + M2) * G * L1 * c1 + M2 * G * L2 * c12
    g2 = M2 * G * L2 * c12
    h1 = cor * (2 * w1 * w2 + w2 ** 2) + g1
    h2 = cor * (-w1 ** 2) + g2
    return np.array([h1, h2])
```

- **`cor = -m_2 \ell_1 \ell_2 \sin\theta_2`**：科氏/离心的公共系数。$\theta_2=0$ 或 $\pi$ 时 $\sin=0$，这些速度二次项消失（与雅可比奇异不是同一件事，但都发生在共线）。
- **`2 w1 w2 + w2²` 与 `-w1²`**：标准 2R 科氏结构。符号必须与 $M$ 的推导一致，否则能量会假增。
- **重力 $\propto\cos$**：$\theta_1=0$ 水平，力矩最大。练习 `g_term` 就是单摆版 $m g L \cos\theta$。
- **`G = 9.81`**：与 `g1` 里的大写同义。不要和阻尼系数搞混（阻尼在 `step` 里另写）。

---

### 第3步：`step` — 正动力学一步

```python
def step(q, w, tau, damp=0.0):
    M = mass_matrix(q[1])
    h = h_vector(q[0], q[1], w[0], w[1])
    acc = np.linalg.solve(M, tau - h - damp * w)
    w = w + DT * acc
    q = q + DT * w
    return q, w
```

- **`q[1]`**：$\theta_2$。$M$ 不看 $\theta_1$。
- **`solve(M, ...)`**：$\ddot q = M^{-1}(\tau - H - c\dot q)$。$M$ 近奇异时（极少见，本模型正定）会报警。
- **`damp * w`**：线性粘滞，加在广义力里。`damp=0` 保守。
- **仍是半隐式欧拉**：先更新角速度再更新角。`DT=0.002`、`steps=2500` → $5\,\mathrm{s}$。

`simulate` 从 `q=(0.3,0.9)`、`w=0` 出发，`tau=np.zeros(2)` 自由落体，返回 `qs` shape `(steps+1, 2)`。

---

### 第4步：末端轨迹用同一套 FK

```python
def fk(th1, th2):
    x = L1 * np.cos(th1) + L2 * np.cos(th1 + th2)
    y = L1 * np.sin(th1) + L2 * np.sin(th1 + th2)
    return x, y
```

与运动学章相同，只是这里 `L1=L2=0.8`。作图：

```python
    xs, ys = fk(qs1[::40, 0], qs1[::40, 1])
```

- **`qs1[::40, 0]`**：每隔 40 步取 $\theta_1$，数组传给 `np.cos`，一次出整条 $x$。不必 Python 循环。
- **右图只画有阻尼**：无阻尼末端会来回扫，线缠成一团；有阻尼才看得出往下掉。

终端：

```python
    print('无阻尼末角速度', np.diff(qs0[-20:], axis=0).mean(0) / DT)
```

- **`qs0[-20:]`**：最后 20 个姿态。
- **`np.diff(..., axis=0)`**：相邻差分 $\approx \dot q \Delta t$，再 `/ DT` 得角速度。无阻尼不应接近 0；有阻尼应明显更小。

---

### 关键概念速查表

| 概念 | 数学 / 直觉 | 代码 |
|------|-------------|------|
| $M(q)$ | 惯性，靠 $\theta_2$ | `mass_matrix` |
| $H$ | 科氏 + 重力 | `h_vector` |
| 正动力学 | $\ddot q=M^{-1}(\tau-H)$ | `solve` |
| 阻尼 | $-c\dot q$ | `damp * w` |
| 水平为零 | $g\propto\cos\theta$ | `g1`,`g2` |
| 半隐式欧拉 | 先 $w$ 后 $q$ | `step` |
| `::40` | 抽稀画轨迹 | `qs1[::40, 0]` |
| `diff` | 数值速度 | 末段角速度 |
| `fk` | 与运动学同一三角 | 右图 |

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/robotics/dynamics/code/demo.py`
