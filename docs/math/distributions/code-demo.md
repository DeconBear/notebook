---
title: "常见分布 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 常见分布 — Python / C++ 代码详解

<a href="/notebook/code/math/distributions/demo.py" target="_blank" download>Download demo.py</a>
<a href="/notebook/code/math/distributions/dist.hpp" target="_blank" download>Download dist.hpp</a>
<a href="/notebook/code/math/distributions/demo.cpp" target="_blank" download>Download demo.cpp</a>

## 运行方式

```bash
cd docs/math/distributions/code
python demo.py
g++ -std=c++17 demo.cpp -o dist_demo
```

两张图：`dist_discrete.png`（二项柱 vs 泊松柱）、`dist_continuous.png`（高斯 vs 指数）。C++ 打印 $P(K=3)$ 与 $\varphi(0)$，应和 Python 终端一致。

## 代码逐段详解（Python）

### 第1步：`math.comb` 与阶乘

```python
def binomial_pmf(k, n, p):
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
```

- **`math.comb`**（3.8+）：$\binom{n}{k}$。不要自己用 `math.factorial(n)/...`，大 $n$ 会先爆再除。
- **`p ** k`**：整数次幂。$p=0.3,k=3$ 是 `0.027`。

泊松用 $e^{-\lambda}\lambda^k/k!$。$k$ 到 10 阶乘还安全；C++ 改走 $\log$ 累加，避免更大 $k$ 溢出。

`λ=np=3`：泊松是二项在「很多次、每次很难中」极限下的亲戚。$n=10$ 还不算极限，两根柱子只是「像」，不会重合。

---

### 第2步：高斯与指数

```python
z = (x - mu) / sigma
return np.exp(-0.5 * z ** 2) / (sigma * np.sqrt(2 * np.pi))
```

- **先标准化再平方**：写成 `(x-mu)**2 / (2 sigma**2)` 等价，但 $z$ 更不容易漏系数。
- **分母有 $\sigma$**：$\sigma$ 变大，峰变矮，积分才能仍是 1。漏掉会让练习断言失败。

指数：`x>=0` 才非零。`linspace(-3,6)` 左半轴应贴着横轴。`Exp(λ=1)` 在 0 处高度为 1，和 $\mathcal{N}(0,1)$ 的 $\varphi(0)\approx 0.40$ 不是一回事——不要比峰谁高，比的是支撑集和对称性。

`bar` 的 `ks-0.15` / `ks+0.15` 让两族柱子并排，宽度 `0.3` 刚好不重叠。

---

## C++：`dist.hpp`

`n_choose_k` 用「逐步乘再除」保持整数：`r = r * (n-k+i) / i`。$n=10$ 精确。`poisson_pmf` 在对数域：$-\lambda + k\log\lambda - \sum\log i$。

可选 Eigen 在这里帮不上忙：这些是标量公式，不是矩阵。

## 源码位置

- `docs/math/distributions/code/demo.py`
- `docs/math/distributions/code/dist.hpp`
- `docs/math/distributions/code/demo.cpp`
