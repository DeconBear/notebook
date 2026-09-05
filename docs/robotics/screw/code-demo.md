---
title: "旋量代数 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 旋量代数 — demo.py 代码详解

<a href="/notebook/code/robotics/screw/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/robotics/screw/code
python demo.py
```

CPU、NumPy 即可。一张图 `screw_se2.png`：小矩形沿固定螺旋「拧」七帧，颜色从 viridis 一头走到另一头；黑叉是瞬心 $q$。平面 $\mathfrak{se}(2)$：$\xi=(\omega,v_x,v_y)$。$\omega\neq 0$ 用闭式指数映射；$\omega=0$ 退化成纯平移（本图没用到这条分支，但函数写了）。

## 代码逐段详解

### 第1步：`se2_exp` — $T(t)=\exp(t\hat\xi)$

```python
def se2_exp(omega, vx, vy, t=1.0):
    if abs(omega) < 1e-10:
        return np.array([
            [1.0, 0.0, vx * t],
            [0.0, 1.0, vy * t],
            [0.0, 0.0, 1.0],
        ])
    w = omega * t
    c, s = np.cos(w), np.sin(w)
    trans = (1.0 / omega) * np.array([[s, -(1 - c)], [1 - c, s]]) @ np.array([vx, vy])
    T = np.eye(3)
    T[0, 0], T[0, 1], T[0, 2] = c, -s, trans[0]
    T[1, 0], T[1, 1], T[1, 2] = s, c, trans[1]
    return T
```

- **$\omega\approx 0$**：齐次阵就是平移 $vt$，左上 $2\times 2$ 为单位。阈值 $10^{-10}$，与李群章小角度同一理由。
- **`w = omega * t`**：有限转角。$t$ 是沿螺旋走了多久，不是「另写一个角度」。
- **平移块**：$(1/\omega)\begin{pmatrix}s & -(1-c)\\ 1-c & s\end{pmatrix}v$。这是 $\int_0^t R(\omega\tau)\,v\,\mathrm{d}\tau$ 的闭式，**除的是 $\omega$ 不是 $\omega t$**。写成 `1/w` 会错。
- **`@ np.array([vx, vy])`**：2×2 乘二维速度。
- **左上旋转** $\begin{pmatrix}c & -s\\ s & c\end{pmatrix}$：与 [线性代数](/math/linear-algebra/) 平面旋转相同。第三行 `[0,0,1]` 来自 `eye(3)` 没改。
- **逐元素赋值 `T[0,0], T[0,1], T[0,2] = ...`**：元组解包写进一行，避免再拼一个 `np.array` 切片广播出错。

返回 $3\times 3$，作用在齐次点 $(x,y,1)$ 上。

---

### 第2步：`apply` — 齐次坐标

```python
def apply(T, pts):
    h = np.c_[pts, np.ones(len(pts))]
    return (T @ h.T).T[:, :2]
```

- **`np.c_[pts, ones]`**：在右边拼一列 $1$，`pts` 是 `(N,2)` → `(N,3)`。
- **`T @ h.T`**：`(3,3)@(3,N)`，再转置，丢掉齐次 $1$，剩 $xy$。
- **为什么不直接 `pts @ R.T + t`？** 可以，但指数映射给的就是 $T$，一次乘更贴 $\mathrm{SE}(2)$ 的写法。

---

### 第3步：纯转动的 $v$（约定要记死）

```python
    omega = 1.2
    q = np.array([0.8, 0.2])
    v = omega * np.array([q[1], -q[0]])
```

瞬心在 $q$，角速度 $\omega$。三维叉乘 $\omega\hat z\times q$ 在平面上是 $\omega(-q_y, q_x)$。**本文件用相反约定** $v=\omega(q_y,-q_x)$（注释写了「相反约定见练习」）。练习 `planar_v` 必须返回 $\omega(q_y,-q_x)$：例如 $\omega=2,q=(1,0)$ → $(0,-2)$。

若改成 $\omega(-q_y,q_x)$，矩形会绕 $q$ 转另一圈方向，图仍然封闭，但和练习断言冲突。

---

### 第4步：七帧矩形

```python
    square = np.array([
        [0.15, 0.1], [0.35, 0.1], [0.35, 0.28], [0.15, 0.28], [0.15, 0.1],
    ])
    for i, t in enumerate(np.linspace(0, 1.6, 7)):
        T = se2_exp(omega, v[0], v[1], t)
        p = apply(T, square)
        ax.plot(p[:, 0], p[:, 1], color=plt.cm.viridis(i / 6), lw=2)
```

- **五个点**：闭合折线（首尾重复 $(0.15,0.1)$）。四个顶点不够闭合。
- **`linspace(0, 1.6, 7)`**：含 $t=0$ 原位。$1.6$ 弧度级转角，看得到拧过去，又没叠成一团。
- **`viridis(i/6)`**：colormap 吃 $[0,1]$。$7$ 帧分母是 $6$。`enumerate` 给 $i$。
- **`scatter(*q, marker='x')`**：瞬心。矩形应绕叉转，不要绕原点转——若 $v$ 写错成 $0$，所有帧会绕原点。

`set_aspect('equal')` 否则转圈变椭圆，螺旋看起来假。

---

### 关键概念速查表

| 概念 | 数学 / 直觉 | 代码 |
|------|-------------|------|
| twist 平面 | $(\omega,v_x,v_y)$ | `se2_exp` 参数 |
| 纯平移 | $\omega=0$ | 函数前半分支 |
| $\exp$ 平移块 | $(1/\omega)V(\omega t)\,v$ | `trans = (1/omega)*...` |
| 齐次 $T$ | $3\times 3$ | `np.eye(3)` 再填 |
| `apply` | $p\leftarrow Tp$ | `np.c_` 拼 $1$ |
| 瞬心 $q$ | 纯转中心 | 黑叉 |
| $v$ 约定 | $\omega(q_y,-q_x)$ | `v = omega * [q[1], -q[0]]` |
| `viridis` | 帧着色 | `i / 6` |
| 闭合多边形 | 首尾同点 | `square` 五行 |

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/robotics/screw/code/demo.py`
