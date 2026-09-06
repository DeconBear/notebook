---
title: "积分与基本定理 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 积分与基本定理 — Python / C++ 代码详解

<a href="/notebook/code/math/integral/demo.py" target="_blank" download>Download demo.py</a>
<a href="/notebook/code/math/integral/quad.hpp" target="_blank" download>Download quad.hpp</a>
<a href="/notebook/code/math/integral/demo.cpp" target="_blank" download>Download demo.cpp</a>

## 运行方式

```bash
cd docs/math/integral/code
python demo.py
g++ -std=c++17 demo.cpp -o int_demo
```

两张图：`int_riemann.png`（左黎曼柱）、`int_ftc.png`（累加得到 $F$，再求导贴回 $f$）。C++ 多项式积分应精确到 $1/3$，梯形 $n=4,16,64$ 误差递减。

## 代码逐段详解（Python）

### 第1步：左黎曼和 — `align='edge'`

```python
h = (b - a) / n
xs = a + h * np.arange(n)
return float(np.sum(f(xs) * h)), xs, h
```

每个柱子的**左端**当高度。`bar(..., align='edge')` 让柱子从 `xs` 往右长宽 $h$，否则默认居中会错位。$x^2$ 在 $[0,1]$ 递增，左黎曼**偏低**，所以 $n=4$ 的和明显小于 $1/3$，这是特性不是 bug。

---

### 第2步：梯形

```python
xs = np.linspace(a, b, n + 1)  # n 段 → n+1 个点
h * (0.5 * ys[0] + 0.5 * ys[-1] + np.sum(ys[1:-1]))
```

两端各算一半，中间算整份。与 C++ `trapezoid` 同一公式。对 $x^2$ 仍略偏高（凸函数的弦在曲线上方）。

---

### 第3步：FTC 实验

```python
F_num = np.array([trapezoid(0.0, x, 40) if x > 0 else 0.0 for x in xs])
dF = np.gradient(F_num, xs)
```

每个 $x$ 都从 $0$ 积到 $x$，得到一条近似原函数。`np.gradient` 是对数组的中心差分，应对准 $x^2$。两端差分改单侧，所以图的两端会略歪——中间才是基本定理该看的地方。

列表推导里 $x=0$ 单独返回 0，避免 `n` 段积零宽区间。`trapezoid(0, x, 40)` 在 $x$ 很小时空步 $h=x/40$ 仍合法。

---

## C++：`quad.hpp`

`poly_int`：`q.c[i+1] = p.c[i] / (i+1)`，`q.c[0]=0`。次数加一，注意 `POLY_MAX` 别撑破。

`trapezoid` 的循环是 `i=1; i<n`，两端在循环外各算半份。`n` 是段数不是点数。`demo.cpp` 用长度为 3 的数组扫 `n=4,16,64`，避免某些环境下花括号列表还要 `#include <initializer_list>`。

## 源码位置

- `docs/math/integral/code/demo.py`
- `docs/math/integral/code/quad.hpp`
- `docs/math/integral/code/demo.cpp`
