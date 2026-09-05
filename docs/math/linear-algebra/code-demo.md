---
title: "线性代数直觉 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 线性代数直觉 — demo.py 代码详解

<a href="/notebook/code/math/linear-algebra/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/math/linear-algebra/code
python demo.py
```

CPU、NumPy 即可。会画出两张图：`la_rotation.png`（同一向量被旋转矩阵转到不同角度）和 `la_pca.png`（椭圆云上的 SVD 主方向，以及投到第一主成分）。本章不训练网络，只把「矩阵 = 几何变换」和「PCA = SVD」跑通。

## 代码逐段详解

### 第1步：导入与路径 — 图写到哪

```python
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)
np.random.seed(42)
```

- **`os.path.abspath(__file__)`**：当前 `.py` 的绝对路径。再 `dirname` 得到 `code/` 目录，`join(..., '..', 'images')` 写到章节自己的 `images/`，从别的目录启动也不会丢图。
- **`exist_ok=True`**：目录已存在也不报错。
- **`np.random.seed(42)`**：PCA 那张椭圆云可复现。旋转演示是确定性的，种子只影响散点。
- **`axes.unicode_minus = False`**：否则负号会画成方块。中文字体列表缺第一个就试下一个。

---

### 第2步：`rotation_matrix` — 为什么是这两行

平面旋转矩阵把向量绕原点转 $\theta$（弧度）：

$$
R(\theta)=\begin{pmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{pmatrix}
$$

```python
def rotation_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])
```

- **先算 `c, s` 再填矩阵**：`cos`/`sin` 各算一次，四个格子共用，避免写四遍三角函数。
- **`np.array([[...],[...]])`**：嵌套列表变成 shape `(2, 2)` 的二维数组。外层是行，内层是列元素。
- **为什么右上是 `-s`？** 这是**逆时针**约定。若写成 `[[c, s], [-s, c]]`，方向会反，后面 PCA 的 35° 椭圆也会拧反。

---

### 第3步：`demo_rotation` — `@` 不是装饰器

```python
v = np.array([1.5, 0.4])
angles = [0, np.pi / 6, np.pi / 3, np.pi / 2]
for i, th in enumerate(angles):
    w = rotation_matrix(th) @ v
```

- **`np.pi / 6`**：30° 的弧度。NumPy 三角函数吃弧度，不吃度数。后面图例用 `th * 180 / np.pi` 再换回度。
- **语法 `@`**：矩阵乘法（Python 3.5+）。`R @ v` 就是 $Rv$。不要写成 `R * v`：`*` 是**逐元素**相乘，shape 对不上还会广播出乱结果。
- **`enumerate`**：同时拿到下标 `i` 和角度 `th`。`i` 只用来给 `plt.cm.viridis` 取颜色：`i / (len(angles) - 1)` 把 0…3 映射到色带 0…1。
- **`ax.arrow(0, 0, w[0], w[1], ...)`**：从原点画到 $w$。`w[0]`、`w[1]` 是旋转后的 $x,y$。
- **`set_aspect('equal')`**：横纵比例尺相同，90° 才看起来像直角，否则椭圆会被屏幕拉伸骗过眼。

四个角：0° 沿原方向；90° 应近似竖直。这就是「矩阵作用在向量上 = 把箭头拧到新方向」，没有特征值、没有解方程。

---

### 第4步：`demo_pca` 造椭圆云 — 先轴对齐再拧

真实数据很少轴对齐。代码故意：先在轴对齐高斯上采样，再旋转，让主方向斜 35°。

```python
z = np.random.randn(n, 2) * np.array([2.0, 0.5])
R = rotation_matrix(np.deg2rad(35))
X = z @ R.T
X = X - X.mean(axis=0)
```

- **`np.random.randn(n, 2)`**：标准正态，shape `(200, 2)`。默认 $\sigma=1$。
- **`* np.array([2.0, 0.5])`**：广播。每行的 $x$ 乘 2、$y$ 乘 0.5，变成扁椭圆（还轴对齐）。
- **`np.deg2rad(35)`**：度 → 弧度。`rotation_matrix` 要弧度。
- **`z @ R.T`**：对**每一行**右乘 $R^\top$。若数据是行向量 $x^\top$，旋转后是 $(Rx)^\top = x^\top R^\top$。用 `R` 而不转置，方向会错。
- **`X.mean(axis=0)`**：沿样本维（第 0 轴）求均值，得到 shape `(2,)` 的中心。**PCA 必须先去均值**，否则第一主成分会指向「离原点最远」而不是「方差最大」。
- **语法 `axis=0`**：对哪一维塌缩。`axis=0` 对行平均，留下列；`axis=1` 对列平均，留下行。

---

### 第5步：SVD 就是 PCA — `full_matrices=False`

中心化后的数据矩阵 $X\in\mathbb{R}^{n\times 2}$：

$$
X = U\,\mathrm{diag}(\sigma_1,\sigma_2)\,V^\top
$$

$V$ 的列（代码里 `Vt` 的行）是主方向；$\sigma_i/\sqrt{n}$ 近似该方向的标准差。

```python
U, S, Vt = np.linalg.svd(X, full_matrices=False)
pcs = Vt  # 主方向（行）
```

- **`np.linalg.svd`**：返回 `U, S, Vt`。注意第三项已经是 $V^\top$，不要再 `.T` 一次。
- **`full_matrices=False`**：经济型 SVD。$X$ 是 $200\times 2$，完整 U 会是 $200\times 200$，浪费。经济型 U 是 $200\times 2$，`S` 长度 2。
- **`S` 是一维数组**，不是对角矩阵。画箭头时用标量 `S[i]` 当长度。

```python
d = pcs[i] * S[i] / np.sqrt(n)
```

用 $\sigma_i/\sqrt{n}$ 缩放，箭头长度和云的拉伸同量级，看起来才像「贴着椭圆长轴」。不除 $\sqrt{n}$，箭头会冲出坐标轴。

**语法 `X[:, 0]`**：所有行、第 0 列，即全部点的 $x$ 坐标。`:` 是「这一维全要」。

---

### 第6步：投到 PC1 — `[:, None]` 在干什么

一维 PCA：分数 $s = X v_1$，再重构 $\hat X = s v_1^\top$。

```python
scores = X @ pcs[0]
proj = scores[:, None] * pcs[0][None, :]
```

- **`pcs[0]`**：第一行，shape `(2,)`。`X @ pcs[0]` 得到每个点的一维分数，shape `(n,)`。
- **语法 `[:, None]`**：插入新轴。`(n,)` → `(n, 1)`。`None` 和 `np.newaxis` 一样。
- **`pcs[0][None, :]`**：`(2,)` → `(1, 2)`。
- **相乘靠广播**：`(n,1) * (1,2)` → `(n,2)`，等于外积。等价于 `np.outer(scores, pcs[0])`。

右图灰点是原云，红点挤在一条斜线上：用一维近似二维，垂直于 PC1 的信息被扔掉。这就是降维的几何。

终端会打印两个奇异值：第一个应明显大于第二个（扁椭圆）。

---

### 第7步：`main` 只串两场演示

```python
def main():
    print('=== 线性代数直觉 ===')
    demo_rotation()
    demo_pca()

if __name__ == '__main__':
    main()
```

**语法 `if __name__ == '__main__'`**：直接 `python demo.py` 时 `__name__` 是 `'__main__'`，才跑 `main()`。被别人 `import` 时不画图，方便以后复用 `rotation_matrix`。

---

### 关键概念速查表

| 概念 | 数学 / 直觉 | 代码 |
|------|-------------|------|
| 旋转 | $R(\theta)v$ | `rotation_matrix(th) @ v` |
| `@` | 矩阵乘 | 不是 `*` 逐元素 |
| 去均值 | $X-\bar x$ | `X - X.mean(axis=0)` |
| SVD / PCA | $X=U\Sigma V^\top$ | `np.linalg.svd(..., full_matrices=False)` |
| 主方向 | $V$ 的行 | `pcs = Vt` |
| `[:, None]` | 加轴做外积 | `scores[:, None] * pcs[0][None, :]` |
| `deg2rad` | 度→弧度 | `np.deg2rad(35)` |
| `set_aspect('equal')` | 圆看起来是圆 | 两张图都设了 |

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/math/linear-algebra/code/demo.py`
