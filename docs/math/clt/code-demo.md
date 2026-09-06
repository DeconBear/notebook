---
title: "大数定律与中心极限 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 大数定律与中心极限 — Python / C++ 代码详解

<a href="/notebook/code/math/clt/demo.py" target="_blank" download>Download demo.py</a>
<a href="/notebook/code/math/clt/clt.hpp" target="_blank" download>Download clt.hpp</a>
<a href="/notebook/code/math/clt/demo.cpp" target="_blank" download>Download demo.cpp</a>

## 运行方式

```bash
cd docs/math/clt/code
python demo.py
g++ -std=c++17 demo.cpp -o clt_demo
```

两张图：`clt_lln.png`（运行均值贴 $p$）、`clt_hist.png`（指数均值的三档直方图）。C++ 印 $n=30$ 时 $\mathrm{Var}(\bar X)$ 是否接近 $1/30$。

## 代码逐段详解（Python）

### 第1步：运行均值 — `cumsum` 一次做完

```python
x = np.random.binomial(1, p, size=n)
running = np.cumsum(x) / np.arange(1, n + 1)
```

- **`cumsum`**：第 $k$ 项是前 $k$ 次正面数。
- **除以 `1..n`**：得到 $\bar X_1,\ldots,\bar X_n$。不要写成 `x.mean()` 一条水平线——那就看不见「慢慢老实」。
- **`arange(1, n+1)`**：从 1 起，避免除以 0。

$p=0.6$，$n=400$ 末尾应在 $0.6$ 附近晃，早期可以很歪。

---

### 第2步：CLT 直方图 — 对指数下手

指数 $\mathrm{Exp}(\lambda=1)$：$\mu=1$，$\sigma^2=1$，右偏、支撑 $[0,\infty)$。

```python
means = np.random.exponential(1.0 / lam, size=(n_rep, n)).mean(axis=1)
```

- **NumPy 的 `exponential` 参数是尺度 $\beta=1/\lambda$**，不是速率。$\lambda=1$ 时尺度也是 1。写反会让理论高斯对不上。
- **`size=(4000, n)` 再 `mean(axis=1)`**：每一行一次实验的 $n$ 个样本，压成 4000 个 $\bar X$。
- 红线是 $\mathcal{N}(\mu, \sigma^2/n)$，不是原指数密度。$n=1$ 时这条铃铛对指数是错的（故意画上，对照「CLT 还没生效」）；$n=30$ 才该贴住。

横轴固定 `[0,4]`：指数均值很少到负数，左边界是 0 不是 $-3$。

---

## C++：`clt.hpp`

最小 LCG：`state = state * 1103515245 + 12345`，再移位得到 $[0,1)$。指数用 $-\log U/\lambda$。$U$ 太靠近 0 会让 $-\log$ 爆，所以夹到 `1e-12`。

`n_rep=2000` 的经验方差会在 $1/30\approx 0.033$ 附近晃，不必等位到小数点后四位——CLT 讲的是分布形状，不是这一次 Monte Carlo 的精度。

可选 Eigen 在这里仍然帮不上：要的是随机数和均值，不是矩阵分解。

## 源码位置

- `docs/math/clt/code/demo.py`
- `docs/math/clt/code/clt.hpp`
- `docs/math/clt/code/demo.cpp`
