---
title: "ml04 支持向量机 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# ml04 支持向量机 (SVM) — demo.py 代码详解

<a href="/notebook/code/ml/classic/svm/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/ml/classic/svm/code
python demo.py
```

## 代码逐段详解

> **复现提示**：已修正 `C` 的样本数缩放、软间隔支持向量候选集和测试集标准化泄漏。`soft_margin_C_effect.png`、`kernel_comparison.png` 尚未重跑生成，旧图不能作为本次修改后的结果。核对比先划分训练/测试集，再仅用训练集拟合 `StandardScaler`。

### 第1步：LinearSVM 类 — Hinge Loss + SGD

```python
class LinearSVM:
    def fit(self, X, y):
        lambda_ = 1.0 / (2.0 * self.C * len(X))
        for epoch in range(self.n_epochs):
            for each sample (x_i, y_i):
                margin = y_i * (w^T x_i + b)
                dw = 2 * lambda_ * w
                db = 0.0
                if margin < 1:
                    dw -= y_i * x_i
                    db -= y_i
                w -= lr * dw
                b -= lr * db
```

损失函数的数学形式为：

$$
J(\mathbf{w}, b) = \frac{1}{n} \sum_{i=1}^{n} \max(0, 1 - y_i(\mathbf{w}^T \mathbf{x}_i + b)) + \lambda \|\mathbf{w}\|^2
$$

**先确定求和还是平均。** `SVC` 使用 $\tfrac12\|w\|^2+C\sum_i\ell_i$。除以正数 $Cn$ 不改变最优解，得到 $\frac1n\sum_i\ell_i+\frac1{2Cn}\|w\|^2$。因此本例单样本 SGD 的对应系数是 $\lambda=1/(2Cn)$，不能漏掉 $n$。`C` 必须是有限正数；特别小的 `C` 也不能被当成关闭正则化。只有数据可分等条件成立时，大 $C$ 才趋近硬间隔解；不可分数据仍需要松弛。

**为什么每次迭代要打乱数据？** 这是 SGD 的标准做法：如果不打乱，数据的顺序会影响梯度更新的路径，可能带来顺序偏差或震荡；这个目标是凸的，不存在较差的局部极小点。`np.random.permutation` 在每轮开始前打乱索引。

### 第2步：Hinge Loss 的子梯度

```python
if margin < 1:
    dw -= y_i * x_i   # Hinge Loss 对 w 的子梯度
    db -= y_i          # Hinge Loss 对 b 的梯度
```

Hinge Loss 的数学定义为：

$$
\ell_{\text{hinge}}(y, f(\mathbf{x})) = \max(0, 1 - y f(\mathbf{x}))
$$

它在 $z = y f(\mathbf{x})$ 上的导数是分段常数：
- $z \geq 1$（正确分类且在间隔外）：梯度 = 0（无贡献）
- $z < 1$（在间隔内或错误分类）：梯度 = -1

因此对 $\mathbf{w}$ 的链式求导：
$$
\frac{\partial \max(0, 1 - y(\mathbf{w}^T \mathbf{x} + b))}{\partial \mathbf{w}} = \begin{cases} -y\mathbf{x} & \text{if } y(\mathbf{w}^T \mathbf{x} + b) < 1 \\ 0 & \text{otherwise} \end{cases}
$$

需要注意的是，Hinge Loss 在 $z=1$ 处不可导（有一个尖点），这里使用的是**子梯度**（subgradient）——即任意在导数不存在处的"单侧导数"都适用。

### 第3步：支持向量的识别

```python
def get_support_vector_mask(self, X, y):
    y_svm = np.where(y <= 0, -1, 1)
    margins = y_svm * self.decision_function(X)
    sv_mask = margins <= 1.01
    return sv_mask
```

令 $m_i=y_if(x_i)$。软间隔最优解的 KKT 条件给出：$m_i<1$ 的间隔违例点满足 $\alpha_i=C$，也属于支持向量；$0<\alpha_i<C$ 的点满足 $m_i=1$。因此只圈 $m_i\approx1$ 会漏掉误分类与间隔内部的支持向量。源码使用 $m_i\le1.01$ 作为可视化候选集；因 SGD 未求出对偶变量且可能未完全收敛，它不是精确的 $\alpha_i>0$ 判定。

### 第4步：RBF 核函数

```python
def rbf_kernel(X, Y, gamma):
    sq_X = np.sum(X ** 2, axis=1, keepdims=True)
    sq_Y = np.sum(Y ** 2, axis=1)
    sq_dists = np.maximum(sq_X + sq_Y - 2 * X @ Y.T, 0.0)
    return np.exp(-gamma * sq_dists)
```

RBF 核：

$$
K(\mathbf{x}, \mathbf{y}) = \exp(-\gamma \|\mathbf{x} - \mathbf{y}\|^2)
$$

展开平方距离：
$$
\|\mathbf{x} - \mathbf{y}\|^2 = \|\mathbf{x}\|^2 + \|\mathbf{y}\|^2 - 2\mathbf{x}^T\mathbf{y}
$$

这与 k-NN 中欧氏距离的展开技巧完全一致，避免显式广播大矩阵。

### 第5步：Gamma 参数的效果展示

```python
gammas = [0.1, 1.0, 10.0, 50.0]
```

$\gamma$ 控制 RBF 核中每个训练样本的"影响半径"：

- **$\gamma = 0.1$**：高斯函数的"钟形曲线"很宽，每个点的影响范围大 → 决策边界平滑，可能欠拟合
- **$\gamma = 1.0$**：适中的影响范围 → 边界复杂度恰当
- **$\gamma = 10.0$**：每个点影响范围很小 → 决策边界复杂，开始过拟合
- **$\gamma = 50.0$**：每个点几乎只影响自己 → 严重过拟合，决策边界围绕每个训练点形成"孤岛"

$\gamma$ 的支持向量数量也反映了过拟合程度——$\gamma$ 越大，支持向量越多（几乎所有训练样本都变成支持向量），模型的 VC 维越高。

## 关键概念速查表

| 概念 | 数学形式 | 代码位置 | 关键说明 |
|------|---------|---------|---------|
| 间隔 | $y_i(\mathbf{w}^T\mathbf{x}_i+b)/\|\mathbf{w}\|$ | `margin` | 样本到超平面的距离 |
| Hinge Loss | $\max(0, 1-y f(\mathbf{x}))$ | `fit()` 中 | 仅在违反间隔时产生梯度 |
| L2 正则化 | $\lambda\|\mathbf{w}\|^2$ | `dw = 2*lambda_*w` | 权重衰减 |
| $\lambda$ 与 $C$ | $\lambda = 1/(2Cn)$ | `lambda_ = 1/(2*C*n)` | $C$ 大 = 弱正则化 |
| 子梯度 | $-y\mathbf{x}$ 若 $yf<1$ | `dw -= y_i*x_i` | 不可导点用子梯度 |
| 支持向量候选 | margin $\le 1+0.01$ | `get_support_vector_mask()` | 含间隔违例点；精确定义为 $\alpha_i>0$ |
| RBF 核 | $\exp(-\gamma\|\mathbf{x}-\mathbf{y}\|^2)$ | `rbf_kernel()` | 无限维映射 |
| $\gamma$ 参数 | 影响半径 | `gamma` | 小 $\to$ 平滑, 大 $\to$ 复杂 |


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/ml/classic/svm/code/demo.py`

## 实现参考

- [scikit-learn 的 SVC 数学定义](https://scikit-learn.org/stable/modules/svm.html#svm-mathematical-formulation)
- [scikit-learn：避免数据泄漏](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)
