---
title: "线性方程组与秩 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 线性方程组与秩 — Python / C++ 代码详解

<a href="/notebook/code/math/systems/demo.py" target="_blank" download>Download demo.py</a>
<a href="/notebook/code/math/systems/gauss.hpp" target="_blank" download>Download gauss.hpp</a>
<a href="/notebook/code/math/systems/demo.cpp" target="_blank" download>Download demo.cpp</a>

## 运行方式

```bash
cd docs/math/systems/code
python demo.py
g++ -std=c++17 demo.cpp -o systems_demo
```

CPU、NumPy 即可。图 `systems_lines.png`：三组 $ax+by=c$。C++ 解教科书 3×3，应得到 $(2,3,-1)$，并对行相关矩阵印 `rank=2`。

## 代码逐段详解（Python）

### 第1步：把直线画出来 — 不要对 $b=0$ 做这个 demo

三条案例都写成 $y=(c-ax)/b$，所以第二系数不能为 0（否则是竖线）。数字选的是：

- $x+y=2$ 与 $x-y=0$：交于 $(1,1)$；
- $x+y=2$ 与 $x+y=3$：平行；
- $x+y=2$ 与 $2x+2y=4$：同一条线。

```python
ax.plot(xs, (c1 - a1 * xs) / b1, ...)
ax.plot(xs, (c2 - a2 * xs) / b2, ..., ls='--')
```

`set_aspect('equal')` 让 45° 看起来是 45°。左图红点是交点，另外两张没有交点（或处处是交点）。

---

### 第2步：`gauss_solve` — 部分主元

对每一列 `col`：

1. 在 `A[col:, col]` 里找绝对值最大的行，对调，避免除以接近 0；
2. 该行除以主元，主元变成 1；
3. 其他行减倍数，把这一列清零。

```python
piv = col + int(np.argmax(np.abs(A[col:, col])))
if abs(A[piv, col]) < eps:
    return None
A[[col, piv]] = A[[piv, col]]
```

- **`A[col:, col]`**：从当前行往下的一截，才是还没当过主元的候选。
- **`A[[col, piv]] = A[[piv, col]]`**：花式索引交换两行。只写 `A[col], A[piv] = A[piv], A[col]` 在 NumPy 里可能踩视图。
- **`return None`**：这一列扫不到主元 → 当成奇异。练习里的数值秩改走 SVD，对「几乎相关」更稳。

教科书矩阵的解是 $(2,3,-1)$。`np.linalg.norm(A @ x - b)` 应接近机器精度。

---

### 第3步：`numerical_rank` — 奇异值数个数

```python
s = np.linalg.svd(A, compute_uv=False)
return int(np.sum(s > tol))
```

- **`compute_uv=False`**：只要奇异值，不要 $U,V$。
- **`tol=1e-8`**：浮点里 $2\times[1,2,3]$ 不会精确等于第二行，但奇异值会有一个 $\approx 0$。

`B` 的第二行是第一行的两倍，秩应是 2（第三行还独立）。

---

## C++：`gauss.hpp`

与 Python 同一套循环，类型是 `array<array<double,3>,3>`。`gauss_solve` 成功则 `x=b`（消完以后右端就是解）。`std::swap(A[row], A[piv])` 交换两行。

可选 Eigen：

```cpp
Eigen::ColPivHouseholderQR<Eigen::Matrix3d> qr(A);
Eigen::Vector3d x = qr.solve(b);
int rank = qr.rank();
```

教学路径不走 QR，才能看见主元。

## 源码位置

- `docs/math/systems/code/demo.py`
- `docs/math/systems/code/gauss.hpp`
- `docs/math/systems/code/demo.cpp`
