---
title: "概率与贝叶斯 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 概率与贝叶斯 — demo.py 代码详解

<a href="/notebook/code/math/probability/demo.py" target="_blank" download>Download demo.py</a>
<a href="/notebook/code/math/probability/stats.hpp" target="_blank" download>Download stats.hpp</a>
<a href="/notebook/code/math/probability/demo.cpp" target="_blank" download>Download demo.cpp</a>

## 运行方式

```bash
cd docs/math/probability/code
python demo.py
g++ -std=c++17 demo.cpp -o prob_demo
```

CPU、NumPy 即可。两张图：`prob_bayes_coin.png`（Beta 先验被 7/10 正面更新成后验）和 `prob_gaussian.png`（一维 $\sigma$ 胖瘦 + 相关二维云）。不依赖 SciPy：Beta 密度自己归一化。C++ 只算样本均值 / 方差，见文末。

## 代码逐段详解

### 第1步：导入、路径、种子

```python
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)
np.random.seed(42)
```

路径写法与全书一致：脚本相对路径写 `images/`。种子钉死二维高斯散点，贝叶斯那张是确定性曲线。

---

### 第2步：`beta_pdf` — 为什么自己写、怎么归一化

Beta 密度（差一个常数）是

$$
p(\theta)\propto \theta^{a-1}(1-\theta)^{b-1},\quad \theta\in(0,1)
$$

完整式子还要除 $B(a,b)$。教学画图不需要 Gamma 函数，对网格做梯形积分再除即可。

```python
def beta_pdf(theta, a, b, eps=1e-9):
    theta = np.clip(theta, eps, 1 - eps)
    unnorm = theta ** (a - 1) * (1 - theta) ** (b - 1)
    trap = getattr(np, 'trapezoid', None) or np.trapz
    z = trap(unnorm, theta)
    return unnorm / z
```

- **`np.clip(theta, eps, 1-eps)`**：把 $\theta$ 限制在开区间。$\theta=0$ 时若 $a-1<0$，$0^{负数}$ 会 `inf`/`nan`。
- **语法 `**`**：逐元素幂。`theta ** (a-1)` 对数组每个点算 $\theta^{a-1}$。不要写成 `^`（那是按位异或）。
- **`*` 在两个同形状数组之间**：逐元素乘，不是矩阵乘。
- **`getattr(np, 'trapezoid', None) or np.trapz`**：NumPy 2.x 把 `trapz` 改名为 `trapezoid`。先找新名，没有就用旧名，1.x / 2.x 都能跑。
- **梯形积分 `trap(y, x)`**：曲线下面积 ≈ 归一化常数。`unnorm / z` 后积分为 1，纵轴才是密度而不是「未归一化高度」。

---

### 第3步：`demo_bayes_coin` — 共轭更新一行算术

抛硬币：似然是二项，先验取 Beta，后验还是 Beta：

$$
\text{先验 }\mathrm{Beta}(a,b),\quad
\text{看到 }h\text{ 次正面、}t-h\text{ 次反面}
\Rightarrow
\text{后验 }\mathrm{Beta}(a+h,\, b+t-h)
$$

```python
theta = np.linspace(0, 1, 400)
prior_a, prior_b = 2.0, 2.0
heads, trials = 7, 10
post_a = prior_a + heads
post_b = prior_b + (trials - heads)
```

- **`np.linspace(0, 1, 400)`**：闭区间均匀 400 个点。画曲线够密；太稀梯形积分会偏。
- **先验 Beta(2,2)**：在 0.5 附近鼓包，比均匀 Beta(1,1) 略信「不太极端」。不是无信息先验。
- **7/10 正面**：`post_a=9`，`post_b=5`。质量往右移，峰变尖——数据把信念从「大概一半」拉向「更可能偏正面」，同时方差变小。
- **`ax.axvline(heads / trials, ...)`**：竖线标样本频率 0.7。后验众数**靠近但不等于** 0.7，因为先验还拉着往 0.5。这就是贝叶斯：后验是先验与数据的折中。

`ls='--'` 是虚线。`label=f'后验 Beta({post_a:.0f},{post_b:.0f})'`：`:.0f` 把 9.0 打成 `9`，图例干净。

---

### 第4步：一维高斯 — 公式怎么落到一行

$$
\mathcal{N}(x;0,\sigma^2)=\frac{1}{\sqrt{2\pi}\,\sigma}\exp\Bigl(-\frac{x^2}{2\sigma^2}\Bigr)
$$

```python
x = np.linspace(-4, 4, 400)
for sigma, c in [(0.5, 'C0'), (1.0, 'C1'), (2.0, 'C2')]:
    y = np.exp(-0.5 * (x / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)
```

- **`(x / sigma) ** 2`**：先除 $\sigma$ 再平方，数值上等于 $x^2/\sigma^2$，少写一层括号。
- **`-0.5 * ...`**：就是公式里的 $-x^2/(2\sigma^2)$。
- **`C0, C1, C2`**：matplotlib 默认色循环第 0、1、2 色，三根曲线自动区分。
- **$\sigma$ 越大曲线越胖、峰值越矮**：分母有 $\sigma$，积分仍为 1。左图三根曲线围成的面积应相同。

---

### 第5步：二维高斯 — 协方差让云倾斜

```python
mean = np.array([0.0, 0.0])
cov = np.array([[1.0, 0.8], [0.8, 1.0]])
pts = np.random.multivariate_normal(mean, cov, size=400)
axes[1].scatter(pts[:, 0], pts[:, 1], s=10, alpha=0.4, c='#5B8FF9')
```

- **协方差矩阵** $\begin{pmatrix}1 & 0.8\\ 0.8 & 1\end{pmatrix}$：对角是各维方差，**非对角 0.8** 是正相关。等高线从圆变成斜椭圆。
- **`multivariate_normal(mean, cov, size=400)`**：抽 400 个二维点，返回 `(400, 2)`。
- **`pts[:, 0]` / `pts[:, 1]`**：所有样本的 $x_1$、$x_2$。`s=10` 点大小，`alpha=0.4` 半透明，重叠处才看得出密度。
- **`set_aspect('equal')`**：否则倾斜会被 stretch 成「看起来不相关」。

若 `cov` 改成对角（非对角 0），云应对齐坐标轴。本章用 0.8 就是为了让「相关 ≠ 独立」一眼可见。

---

### 第6步：`main`

```python
def main():
    print('=== 概率与贝叶斯 ===')
    demo_bayes_coin()
    demo_gaussian()

if __name__ == '__main__':
    main()
```

直接运行才画图；`import` 时只加载函数。两场演示互相独立：硬币不依赖高斯，高斯也不读后验。

---

### 关键概念速查表

| 概念 | 数学 / 直觉 | 代码 |
|------|-------------|------|
| Beta 密度 | $\propto\theta^{a-1}(1-\theta)^{b-1}$ | `beta_pdf`，梯形归一化 |
| 共轭更新 | $a'=a+h,\; b'=b+t-h$ | `post_a` / `post_b` |
| `np.clip` | 躲开 0 和 1 | `eps=1e-9` |
| `**` | 逐元素幂 | 不是 `^` |
| 一维高斯 | $\sigma$ 控胖瘦 | `np.exp(-0.5*(x/sigma)**2)/...` |
| 二维相关 | 非对角协方差 | `multivariate_normal` |
| `[:, 0]` | 取一列 | 散点的 $x$ |
| `linspace` | 均匀网格 | 画密度曲线 |

## C++：`stats.hpp`

```bash
cd docs/math/probability/code
g++ -std=c++17 demo.cpp -o prob_demo
```

`sample_mean` 是循环累加再除以 $n$。`sample_var(..., true)` 除以 $n-1$：

$$
s^2=\frac1{n-1}\sum_i (x_i-\bar x)^2
$$

对 $\{1,2,3,4,5\}$，$\bar x=3$，$\sum d_i^2=10$，无偏 $10/4=2.5$，MLE $10/5=2$。贝叶斯积分仍只在 Python 里画；C++ 只钉死「数字特征是样本的函数」这件事。

可选 Eigen 只在你要向量化一整列样本时才有意义（`Map<VectorXd>` + `mean()`），本章样本只有 5 个数，手写循环更清楚。

## 源码位置

clone 后打开（相对仓库根目录）：

- `docs/math/probability/code/demo.py`
- `docs/math/probability/code/stats.hpp`
- `docs/math/probability/code/demo.cpp`
