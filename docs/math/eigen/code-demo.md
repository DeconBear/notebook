---
title: "特征值与二次型 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 特征值与二次型 — Python / C++ 代码详解

<a href="/notebook/code/math/eigen/demo.py" target="_blank" download>Download demo.py</a>
<a href="/notebook/code/math/eigen/power.hpp" target="_blank" download>Download power.hpp</a>
<a href="/notebook/code/math/eigen/demo.cpp" target="_blank" download>Download demo.cpp</a>

## 运行方式

```bash
cd docs/math/eigen/code
python demo.py
g++ -std=c++17 demo.cpp -o eigen_demo
```

两张图：`eigen_ellipse.png`（$x^\top A x=1$）和 `eigen_power.png`（Rayleigh 商贴向 $\lambda_{\max}$）。$A=\begin{pmatrix}3&1\\1&2\end{pmatrix}$ 对称正定，最大特征值 $(5+\sqrt5)/2\approx 3.618$。

## 代码逐段详解（Python）

### 第1步：`np.linalg.eigh` — 对称专用

```python
w, Q = np.linalg.eigh(A)  # 升序
```

- **`eigh` 不是 `eig`**：对称矩阵用前者，返回正交 $Q$、实 $\lambda$，且按升序排。`eig` 不保证正交，还可能给出复数。
- **`w[0]` 最小**，所以椭圆长轴沿 $Q[:,0]$（$\lambda$ 小 → $1/\sqrt{\lambda}$ 大）。

椭圆参数方程：特征坐标里 $\lambda_1 y_1^2+\lambda_2 y_2^2=1$，取

```python
y = np.stack([np.cos(t) / np.sqrt(w[0]), np.sin(t) / np.sqrt(w[1])], axis=0)
pts = (Q @ y).T
```

- **`stack(..., axis=0)`**：得到 shape `(2, 200)`，每一列是一个 $y$。
- **`Q @ y`**：变回原坐标。箭头画到 $Q_{:i}/\sqrt{\lambda_i}$，端点落在椭圆上。

---

### 第2步：幂迭代

```python
Av = A @ v
lam = float(v @ Av)          # Rayleigh，已假定 ||v||=1
v = Av / np.linalg.norm(Av)
```

- **先乘再归一化**：只归一化 $Av$，不要对 $A$ 做任何分解。
- **`v @ Av`**：NumPy 一维向量的 `@` 是内积。
- **起步 `[1,0]`**：不与最大特征向正交就行。若运气差取到了较小特征向，会停在较小的 $\lambda$——随机起步更稳，本 demo 钉死向量以便和 C++ 对照。

曲线应在十几步内贴上红色虚线。差距 $|\lambda_2/\lambda_1|$ 越小收敛越慢；本例约 $1.382/3.618\approx 0.38$，很快。

---

## C++：`power.hpp`

`Vec2` / `Mat2` 是 `std::array`。`power_iteration` 原地改 `v`，返回最后一步 Rayleigh。不要和线代章的 `struct Mat2` 搞混——那是另一个头文件。

可选 Eigen：

```cpp
Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(A);
double lmax = es.eigenvalues().maxCoeff();
```

幂迭代的意义是：**不求全部特征对也能摸到最大的那个**（PageRank 同款）。

## 源码位置

- `docs/math/eigen/code/demo.py`
- `docs/math/eigen/code/power.hpp`
- `docs/math/eigen/code/demo.cpp`
