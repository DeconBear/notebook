---
title: "状态空间 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 状态空间 — demo.py 代码详解

<a href="/notebook/code/control/modern/state-space/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/control/modern/state-space/code
python demo.py
```

CPU、NumPy 即可。一张图 `pole_place.png`：双积分器 $\ddot p=u$ 从 $x=[1,\,0.8]$ 出发，无控制 vs $u=-Kx$。$K=[6,5]$ 把闭环极点放到 $-2,-3$。

## 代码逐段详解

### 第1步：矩阵植物

```python
A = np.array([[0.0, 1.0], [0.0, 0.0]])
B = np.array([[0.0], [1.0]])
K = np.array([[6.0, 5.0]])
X0 = np.array([1.0, 0.8])  # 初速不为 0，开环才会漂
```

- **`A` $2\times 2$**：$\dot p=v$，$\dot v=0$（无控制）。
- **`B` 写成列** 形状 `(2,1)`。后面 `B @ K` 才是 $2\times 2$；若 `B` 写成一维 `(2,)`，`B @ K` 对不上。
- **`K` 写成行** `(1,2)`：$-Kx$ 是标量。希望 $(s+2)(s+3)=s^2+5s+6$，故 $k_1=6,k_2=5$。
- **初速 $0.8$**：无控制时位置 $p(t)=1+0.8t$ 直线漂。若初速为 $0$，开环位置钉死在 $1$，对照看不出反馈在干活。

`A @ x`：`x` 是一维长度 2，结果也是一维长度 2。

---

### 第2步：能控性矩阵

```python
def controllability_matrix(A, B):
    return np.hstack([B, A @ B])
```

- **`A @ B`**：$AB$，形状 `(2,1)`。
- **`np.hstack`**：按列拼 `[B, AB]`。本例是 $\begin{bmatrix}0&1\\1&0\end{bmatrix}$，秩 $2$，不是单位阵。
- **`np.linalg.matrix_rank`**：数值秩。满秩 ⇒ 可以任意配置极点。缺秩时 $K$ 怎么调都搬不动藏起来的模态。

---

### 第3步：欧拉积分

```python
u = float((-K @ x).item()) if use_feedback else 0.0
xdot = A @ x + B.ravel() * u
x = x + DT * xdot
```

- **`(-K @ x)`** 形状 `(1,)`。`.item()` 取出 Python 标量。以前用 `float(K @ x)` 在一维结果上会 TypeError，必须 `.item()`。
- **`B.ravel()`**：`(2,1)` 拉成 `(2,)`，才能和一维 `xdot` 相加。
- **显式欧拉** `x ← x+Δt ẋ`。步长 `0.02`、极点在 $-3$ 附近够用。

`np.linalg.eigvals(A - B @ K)` 应打印约 `-2` 和 `-3`（顺序不一定）。开环 `A` 的两个特征值都是 $0$。

`xs.append(x.copy())`：必须 `.copy()`。`x` 是同一块内存，只 append 引用的话，列表里每一帧最后都会变成终态。

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/control/modern/state-space/code/demo.py`
