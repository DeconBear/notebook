---
title: "导数与微分 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 导数与微分 — Python / C++ 代码详解

<a href="/notebook/code/math/derivative/demo.py" target="_blank" download>Download demo.py</a>
<a href="/notebook/code/math/derivative/dual.hpp" target="_blank" download>Download dual.hpp</a>
<a href="/notebook/code/math/derivative/demo.cpp" target="_blank" download>Download demo.cpp</a>

## 运行方式

```bash
cd docs/math/derivative/code
python demo.py
g++ -std=c++17 demo.cpp -o deriv_demo
```

两张图：`deriv_secant.png`（割线 → 切线）、`deriv_fd_error.png`（中心差分误差随 $h$）。C++ 对 $x^3-2x$ 在 $x=2$ 印 $f=4,f'=10$，三条路（对偶数、多项式、差分）应一致。

## 代码逐段详解（Python）

### 第1步：`Dual` — 为什么乘法不能只乘 `.v`

对偶数 $(v,d)$ 表示 $v+d\,\varepsilon$，$\varepsilon^2=0$。加法分量相加；乘法：

$$
(u+u'\varepsilon)(v+v'\varepsilon)=uv+(uv'+u'v)\varepsilon.
$$

```python
def __mul__(self, other):
    other = other if isinstance(other, Dual) else Dual(other, 0.0)
    return Dual(self.v * other.v, self.v * other.d + self.d * other.v)
```

- **常数变成 `Dual(other, 0)`**：`2.0 * t` 走 `__rmul__`，否则 `double` 不知道怎么乘 `Dual`。
- **`Dual.var(x)`**：种子 $d=1$，表示「对这个 $x$ 求导」。对常数求导应得到 0，不要误写成 `var`。
- **`t * t * t - 2.0 * t`**：和在纸上写 $x^3-2x$ 同一棵表达式树。`.d` 就是 $3x^2-2$。

C++ `operator*` 是同一行公式。不要写成 `a.v*b.v` 配 `a.d*b.d`——那是错的逐元乘。

---

### 第2步：割线图 — 斜率怎么画成线

```python
slope = (y2 - y1) / h
line_y = y1 + slope * (line_x - a)
```

过点 $(a,f(a))$、斜率为差商的直线。$h=0$ 那一档改用解析 $f'(a)=3a^2-2$。三张图共用 `sharey=True`，才能看出割线在转动，而不是坐标系在偷偷缩放。

---

### 第3步：多项式系数与差分误差

```python
c = np.array([0.0, -2.0, 0.0, 1.0])  # c0 + c1 x + c2 x^2 + c3 x^3
poly_diff(c)  # [1*(-2), 2*0, 3*1] → [-2, 0, 3] 即 -2 + 3x^2
```

下标 $i$ 是幂次。求导后长度减一。

中心差分扫 `hs = logspace(-1, -8, 8)`：误差先降后升。对数横轴才能看见 $10^{-6}$ 附近的槽。这是在提醒：黑盒差分能用，但不是「$h$ 越小越好」。

---

## C++：`dual.hpp`

`Dual::var` / `Dual::cnst` 比布尔参数更不容易用反。`sin`/`exp` 的 $d$ 分量乘的是各自的解析导数，再乘 `x.d`（链式法则）。`central_diff` 吃函数指针 `double (*)(double)`，所以 `cube_minus_2x` 必须是普通函数，不能是带捕获的 lambda（除非写成不捕获并转成指针，教学里不绕这个弯）。

`x * x * x - 2.0 * x` 依赖 `operator*(double, Dual)`，否则 `2.0 * x` 编不过。

## 源码位置

- `docs/math/derivative/code/demo.py`
- `docs/math/derivative/code/dual.hpp`
- `docs/math/derivative/code/demo.cpp`
