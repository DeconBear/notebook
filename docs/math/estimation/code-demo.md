---
title: "数理统计与估计 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 数理统计与估计 — Python / C++ 代码详解

<a href="/notebook/code/math/estimation/demo.py" target="_blank" download>Download demo.py</a>
<a href="/notebook/code/math/estimation/mle.hpp" target="_blank" download>Download mle.hpp</a>
<a href="/notebook/code/math/estimation/demo.cpp" target="_blank" download>Download demo.cpp</a>

## 运行方式

```bash
cd docs/math/estimation/code
python demo.py
g++ -std=c++17 demo.cpp -o est_demo
```

两张图：`est_bernoulli.png`（$\hat p$ 随 $n$ 变瘦）、`est_variance.png`（MLE 方差偏小）。C++ 对 7 次正面 / 10 次抛币印 `0.7`。

## 代码逐段详解（Python）

### 第1步：伯努利 MLE 就是均值

```python
def bernoulli_mle(x):
    return float(np.mean(x))
```

$X_i\in\{0,1\}$，$\bar X=k/n=\hat p_{\mathrm{MLE}}$。`np.random.binomial(n, p, size=2000) / n` 一次抽出 2000 个 $\hat p$，再直方图。$n$ 从 5 到 80，柱子往 $0.7$ 挤——这是 LLN 的预告片，形状还不必是高斯。

---

### 第2步：$n$ 还是 $n-1$

```python
den = (x.size - 1) if unbiased else x.size
return float(s / den)
```

`gaussian_mle` 里 $\sigma^2$ **固定除以 $n$**。对 $n=8$、真 $\sigma^2=4$ 重复 4000 次：

- MLE 直方图中心应明显小于 4（理论期望是 $\frac{n-1}{n}\sigma^2=3.5$）；
- 无偏 $s^2$ 中心应靠近 4。

不要用 `np.var(x)` 的默认值混着比：NumPy 默认 `ddof=0`（除以 $n$），`ddof=1` 才是 $n-1$。本 demo 自己写除数，避免这个坑。

`np.random.normal(mu, sigma, size=n)` 的第二参数是**标准差**不是方差，所以传入 `np.sqrt(sig2_true)`。

---

## C++：`mle.hpp`

`bernoulli_mle` 吃 `const int*`。`gaussian_mle` 用两个输出引用 `mu, sigma2`，因为 C++ 不能漂亮地一次返回一对而不引入 `struct`——教学里引用比 `std::pair` 更直。

可选 Eigen：`Map<VectorXd>(ptr, n).mean()` 等价于样本均值，仍然要自己决定方差除以谁。

## 源码位置

- `docs/math/estimation/code/demo.py`
- `docs/math/estimation/code/mle.hpp`
- `docs/math/estimation/code/demo.cpp`
