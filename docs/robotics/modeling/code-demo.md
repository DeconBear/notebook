---
title: "机器人建模方法 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 机器人建模方法 — demo.py 代码详解

<a href="/notebook/code/robotics/modeling/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/robotics/modeling/code
python demo.py
```

CPU、NumPy 即可。一张图 `dh_3r.png`：三根平面杆的折线。标准 DH：每关节 $(a,\alpha,d,\theta)$。本例 $\alpha=d=0$，$a$ 为杆长，$\theta$ 为关节角。齐次矩阵按代码里的乘法顺序，不要把另一本教材的 modified DH 矩阵直接贴进来。

## 代码逐段详解

### 第1步：杆长表

```python
A_LEN = [0.6, 0.5, 0.35]
```

三个 $a_i$。`fk_chain` 用 `zip(A_LEN, thetas)` 一对一，长度必须相等。没有第四杆。

---

### 第2步：`dh` — 一帧 $A_i$

顺序：绕 $z$ 转 $\theta$，沿 $z$ 移 $d$，沿 $x$ 移 $a$，绕 $x$ 转 $\alpha$。乘开即

```python
def dh(a, alpha, d, theta):
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0],
    ])
```

- **先算 `ca,sa,ct,st`**：矩阵里每个三角函数最多用一次，少写错。
- **第四行 `[0,0,0,1]`**：齐次坐标。点是 $(x,y,z,1)$，方向向量 $w=0$。
- **平面关节** `alpha=0, d=0`：`ca=1, sa=0`，矩阵变成
  $$
  \begin{pmatrix}c\theta & -s\theta & 0 & a c\theta\\ s\theta & c\theta & 0 & a s\theta\\ 0&0&1&0\\ 0&0&0&1\end{pmatrix}
  $$
  右上角 $(a\cos\theta,\,a\sin\theta)$ 就是「转完再沿杆走 $a$」。练习 `dh_trans_x` 只要元素 $(0,3)=a\cos\theta$。
- **第三行 `sa, ca, d`**：空间臂扭角、偏置在这里出现。本 demo 它们是 $0,1,0$。

图解 PNG 里可能印了另一种 $A_i$ 排列（DH 有多个流行约定）。**以本函数为准**。$\alpha=d=0$ 时多数约定重合，平面 3R 画出来一样。

---

### 第3步：`fk_chain` — 左乘累加

```python
def fk_chain(thetas):
    T = np.eye(4)
    origins = [T[:3, 3].copy()]
    for a, th in zip(A_LEN, thetas):
        T = T @ dh(a, 0.0, 0.0, th)
        origins.append(T[:3, 3].copy())
    return np.array(origins), T
```

- **`T = np.eye(4)`**：基座。原点 $(0,0,0)$。
- **`T[:3, 3]`**：齐次矩阵的平移列（前三行第四列）。每一帧乘完记下，折线才有中间关节。
- **`.copy()`**：`origins` 若存切片视图，下一步改 `T` 会把历史原点一起改掉。
- **`T = T @ dh(...)`**：从基座往末端乘。`@` 不要写成 `*`（那是逐元素，4×4 会得到无意义数）。
- **返回 `origins` 与最终 `T`**：位置用前者，姿态用 `T[:3,:3]`。

---

### 第4步：主程序 — 打印姿态并画折线

```python
    th = np.array([0.4, -0.7, 0.5])
    pts, T = fk_chain(th)
    print('末端位置', pts[-1, :2])
    print('末端旋转 R=\n', np.round(T[:3, :3], 3))
```

- **`pts[-1, :2]`**：最后一个原点的 $x,y$（平面 $z=0$）。
- **`np.round(..., 3)`**：旋转块应接近平面旋转（第三行第三列 $\approx 1$，第三列前两行 $\approx 0$）。
- **`plot(pts[:,0], pts[:,1], 'o-')`**：点 + 线。`scatter` 再把基座涂黑。`set_aspect('equal')` 保杆长视觉比例。

没有逆解：给定角画一次。要解 $\theta$ 需数值 IK 或几何，留给运动学章的 2R。

---

### 关键概念速查表

| 概念 | 数学 / 直觉 | 代码 |
|------|-------------|------|
| DH 四元组 | $(a,\alpha,d,\theta)$ | `dh(a,alpha,d,theta)` |
| 平面臂 | $\alpha=d=0$ | `dh(a, 0.0, 0.0, th)` |
| 平移列 | $A$ 的 $(0{:}3,3)$ | `T[:3, 3]` |
| 连乘 | $T=A_1A_2A_3$ | `T = T @ dh(...)` |
| `@` vs `*` | 矩阵乘 vs 逐元素 | 必须 `@` |
| `.copy()` | 快照原点 | `origins.append` |
| `zip` | 杆长配关节角 | `zip(A_LEN, thetas)` |
| $R$ | $T$ 左上 $3\times3$ | `T[:3,:3]` |
| $(0,3)$ 元素 | $a\cos\theta$（平面） | 练习 `dh_trans_x` |

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/robotics/modeling/code/demo.py`
