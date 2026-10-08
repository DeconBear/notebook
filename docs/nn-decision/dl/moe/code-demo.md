---
title: "混合专家 MoE — demo.py"
---

> [!WARNING]
> 2026-10-08 静态审查：已修复路由器与辅助损失的 Softmax 雅可比、归一化及偏置梯度。已有图片和数值尚未按修复代码重新生成或运行验证，不能作为修复后结果。



> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 混合专家 MoE — demo.py 代码详解

<a href="/notebook/code/nn-decision/dl/moe/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/nn-decision/dl/moe/code
python demo.py
```

CPU + NumPy 即可，几十秒。四个高斯簇上的二分类：线性路由器 + 4 个线性专家，每个样本只唤醒 Top-2。同一数据训两遍（有/无负载均衡），画出专家使用率、分类损失、决策边界。没有 Transformer、没有专家并行，只把门控、稀疏加权和 $\mathcal{L}_{\mathrm{aux}}$ 跑通。

## 代码逐段详解

### 第1步：导入与超参 — 每个名字后面干什么

```python
N_EXPERTS = 4
TOP_K = 2
LR = 0.08
STEPS = 400
AUX_COEF = 0.05
```

- **`os` / `_IMAGES_DIR`**：`os.path.dirname(os.path.abspath(__file__))` 是当前 `.py` 所在目录，再 `join(..., '..', 'images')`，无论从哪启动路径都对。
- **`numpy`**：造数据、矩阵乘、手写梯度。本章不用 PyTorch，为的是把 Softmax 门控和辅助损失摊成数组。
- **`font.sans-serif`**：中文字体列表，缺第一个就试下一个。**`axes.unicode_minus = False`**：否则负号画成方块。
- **`np.random.seed(42)`**：簇采样和 `W_r`/`W_e` 初始化共用这一份随机源。
- **`N_EXPERTS=4`**：和四个簇对齐，鼓励「一块区域一个专家」。标签却只有 0/1 交替，所以专家学的是区域，不是四分类。
- **`TOP_K=2`**：每个样本激活 2/4 专家，对应 Mixtral 一类的稀疏门控。
- **`HIDDEN=8`**：声明了，但线性专家没用到——不要按名字脑补一层 MLP。
- **`AUX_COEF`**：正文里的 $\alpha$。太大则路由被「均匀」绑架；太小则塌缩。

---

### 第2步：`make_data` — 为什么用四簇而不是一团云

路由器要学「不同区域走不同专家」。四个中心分居象限，簇内噪声 `0.35`，彼此不太糊在一起。

```python
centers = np.array([[-1.5, -1.2], [1.6, -1.0], [-1.4, 1.5], [1.5, 1.4]])
for i, c in enumerate(centers):
    xs.append(c + 0.35 * np.random.randn(n_per, 2))
    ys.append(np.full(n_per, i % 2))
```

- **`i % 2`**：簇 0、2 → 类 0，簇 1、3 → 类 1。这里是左右两侧分别同类，整体大致线性可分；本例用于观察门控与负载，不证明 MoE 优于线性分类器。
- **`np.full(n_per, i % 2)`**：长度 `n_per`、值全相同的标签数组。
- **`np.random.permutation(len(X))`**：打乱下标再切片。`astype(float)` 让标签能进 BCE 的 `(1-y)*log(1-p)`。

---

### 第3步：`softmax` / `sigmoid` — 门控概率和分类概率

路由器输出未归一化 logits，门控是

$$
g_i(x)=\frac{e^{z_i}}{\sum_j e^{z_j}},\qquad z=xW_r+b_r
$$

```python
z = logits - logits.max(axis=axis, keepdims=True)
e = np.exp(z)
return e / e.sum(axis=axis, keepdims=True)
```

- **先减 `max`**：$\mathrm{softmax}(z)=\mathrm{softmax}(z-c)$。某个 $z$ 很大时不减，`exp` 会溢出成 `inf`，整行变 NaN。
- **`keepdims=True`**：`max` 后仍留着被缩掉的那一维，才能和 `logits` 广播相减。`axis=1` 时 `(B,4)` → `(B,1)`，不是 `(B,)`。
- **`axis=-1`**：默认沿最后一维；对 `(B, n_experts)` 就是对专家维做。

```python
z = np.exp(-np.abs(x))
return np.where(x >= 0, 1.0 / (1.0 + z), z / (1.0 + z))
```

专家加权和是标量 logit，分类概率 $p=\sigma(\hat y)$。指数只计算非正数，避免溢出，同时不用裁剪输入改变函数。

---

### 第4步：`TinyMoE` 两套权重

```python
self.W_r = np.random.randn(in_dim, n_experts) * scale   # (2, 4)
self.W_e = np.random.randn(n_experts, in_dim) * scale   # (4, 2)
```

| 符号 | 代码 | 形状 | 角色 |
|------|------|------|------|
| $W_r,b_r$ | `W_r`,`b_r` | `(2,4)` / `(4,)` | 路由器：每样本一个 4 维 logit |
| $W_e^{(i)}$ | `W_e[i]` | 每行 `(2,)` | 第 $i$ 个专家：$E_i(x)=w_i^\top x+b_i$ |

专家也做成线性，是因为本章要看的是**路由**，不是专家容量。四个超平面靠门控拼出一块块决策。这不是 `nn.Module`，没有 `super().__init__()`，更新靠手写 SGD。

---

### 第5步：`route` — Softmax → Top-2 → 再归一化

$$
\tilde g = \mathrm{renorm}\big(\mathrm{Top\text{-}k}(g)\big)
$$

```python
logits = X @ self.W_r + self.b_r
probs = softmax(logits, axis=1)
top_idx = np.argsort(probs, axis=1)[:, -TOP_K:]
rows = np.arange(len(X))[:, None]
top_p = softmax(logits[rows, top_idx], axis=1)
```

- **`X @ W_r`**：`(B,2)@(2,4)→(B,4)`。`@` 是矩阵乘；`*` 才是逐元素，这里不能混。
- **`np.argsort(..., axis=1)`**：每行从小到大的**下标**。`[:, -2:]` 取最后两列 = 概率最大的两个专家。`:` 是「这一维全要」，`-2:` 是「从倒数第二个到末尾」。
- **`np.arange(B)[:, None]`**：`(B,)` 加成 `(B,1)`，才能和 `top_idx` 的 `(B,2)` 一起做高级索引。
- **`probs[rows, top_idx]`**：第 $b$ 行取出那两个专家的 $g$，得到 `(B,k)`。
- **对入选 logits 做 Softmax**：等价于只取入选概率后除以它们的和。分母严格为正，不需要额外 epsilon；权重和保持为 1。

未选中的专家本步不进 $\hat y$。实现上 `expert_out` 仍一次算出全部 4 列再切片——batch 很小，不是生产级稀疏内核。

---

### 第6步：`forward` — 只把被选专家加权求和

```python
eo = self.expert_out(X)       # (B, 4)
chosen = eo[rows, top_idx]    # (B, k)
y_hat = (chosen * top_p).sum(axis=1)
```

$$
\hat y=\sum_{j=1}^{k}\tilde g_{i_j}\,E_{i_j}(x)
$$

- **`expert_out`**：`X @ W_e.T + b_e`。`W_e` 是 `(4,2)`，`.T` 转成 `(2,4)`，一次得到每个专家的标量。
- **`chosen * top_p`**：逐元素。第 $j$ 个被选输出乘它的门控，再沿 `axis=1` 加总成一个 logit。
- 没进 Top-2 的专家对 $\hat y$ 没有前向贡献；`train_step` 也只给选中的 `e` 累加梯度。

---

### 第7步：`load_balance_loss` — 为什么要 $N\sum f_i P_i$

放任不管，路由器会把票永远投给最先碰巧有用的一两个专家，其余 `W_e` 收不到梯度（路由崩溃）。Switch 一类写法：

$$
\mathcal{L}_{\mathrm{aux}}=N\sum_{i=1}^{N}f_i P_i
$$

```python
for k in range(TOP_K):
    for i in top_idx[:, k]:
        f[i] += 1.0
f = f / (B * TOP_K)
P = probs.mean(axis=0)
return float(self.n * np.sum(f * P))
```

- **$f_i$**：专家 $i$ 在 $B\times k$ 次选择里被点到的频率。每个样本贡献 $k$ 次，所以除以 `B * TOP_K`。`top_idx[:, k]` 是「所有样本的第 $k$ 个被选下标」。
- **$P_i$**：Top-k **之前**的 Softmax 在 batch 上平均。用完整 $g$ 而不是 $\tilde g$，鼓励选之前就把质量摊开。
- **都均匀**时 $f_i=P_i=1/N$，乘 $N$ 得 $1$；塌缩到一个专家则 $f$、$P$ 都尖，乘积变大。
- **`float(...)`**：后面要和 Python 标量 CE 相加、还要 `print`。

---

### 第8步：精确反传与 Top-k 的适用边界

分类损失采用数值稳定的 BCE-with-logits：
$$
\ell(s,y)=\log(1+e^s)-ys,\qquad
\frac{\partial L}{\partial s_b}=\frac{\sigma(s_b)-y_b}{B}.
$$

```python
loss = float(np.mean(np.logaddexp(0.0, y_hat) - y * y_hat))
dlogit = (sigmoid(y_hat) - y) / len(y)
```

设某个样本入选专家集合为 $S$。在集合不变化的局部区域内，
$$
q_i=\frac{e^{z_i}}{\sum_{j\in S}e^{z_j}},\qquad
s=\sum_{i\in S}q_iE_i.
$$
由 Softmax 雅可比 $\partial q_i/\partial z_j=q_i(\mathbf1_{i=j}-q_j)$ 可得
$$
\frac{\partial s}{\partial z_j}=q_j(E_j-s)\quad(j\in S).
$$
未入选专家的分类梯度为零。专家参数的梯度则是 $d_s q_j x$ 和 $d_s q_j$。这不是对排序做直通近似，而是对固定入选集合精确求导；第 k 与第 k+1 名交换的边界处不可导，不能跨越边界做普通有限差分验证。

```python
d_logits[b, e] = dlogit[b] * top_p[b, j] * (eo[b, e] - y_hat[b])
```

辅助损失使用全量概率 $p=\mathrm{softmax}(z)$。令 $u_i=\alpha Nf_i/B$，离散频率 $f_i$ 在本次反传中视为常数，链式法则给出
$$
\frac{\partial(\alpha L_{\mathrm{aux}})}{\partial z_{bi}}
=p_{bi}\left(u_i-\sum_jp_{bj}u_j\right).
$$
因此不能把 $u_i$ 直接加到 logits 的梯度上，也不能省去最外层的 $p_{bi}$。

```python
dP = AUX_COEF * self.n * f / len(X)
d_logits += probs * (dP - (probs * dP).sum(axis=1, keepdims=True))
self.W_r -= LR * (X.T @ d_logits)
self.b_r -= LR * d_logits.sum(axis=0)
```

偏置梯度对样本求和，因为 $d\_logits$ 已包含 $1/B$，再取均值会多除一次 B。

**手算检查。** 两位专家的门控为 $(0.25,0.75)$，输出为 $(2,0)$，则 $s=0.5$，对两个 logits 的导数为 $(0.375,-0.375)$，和为零。这对应 Softmax 对共同平移不变的性质。辅助损失的每行梯度之和也应为零。以上是解析检查例，不是已运行的测试结果。

---

### 第9步：`train` / `main` — 对照实验在验证什么

`train(use_aux, X, y)` 每步吃**全量** `X`（没有再切 mini-batch）。`expert_usage` 把 `top_idx` 里出现的次数除以总和，和 $f$ 同一件事。每 50 步打印 `usage`。

`main` 先 `train(True)` 再 `train(False)`。两次各自 `TinyMoE()`，不要共用权重。

1. **专家负载柱**：无均衡往往一根特别高；有均衡四根更近。这是 $\mathcal{L}_{\mathrm{aux}}$ 存在的理由。
2. **分类 CE**：均衡不该把分类训废。两条都应下降；有时有均衡略高，那是 $\alpha$ 的税。
3. **决策边界**：`meshgrid` + `np.c_[xx.ravel(), yy.ravel()]` 铺成 `(40000,2)`，用有均衡模型的 `sigmoid(forward)` 填色。`np.c_` 按列拼接两个展平坐标。

---

### 关键概念速查表

| 概念 | 数学 / 直觉 | 代码 |
|------|-------------|------|
| 门控 | $g=\mathrm{softmax}(xW_r+b_r)$ | `route` 里 `softmax(logits)` |
| Top-2 | 只点亮最大 $k$ 个，再归一化 | `argsort` + `[:, -TOP_K:]` |
| 稀疏输出 | $\hat y=\sum_{j\in\mathcal{T}_k}\tilde g_j E_j$ | `(chosen * top_p).sum(1)` |
| 辅助损失 | $N\sum f_i P_i$，防塌缩 | `load_balance_loss` |
| BCE | $-[y\log p+(1-y)\log(1-p)]$ | `train_step` 的 `loss` |
| 专家梯度 | 只回传到被选 $e$ | `dW_e[e] += dlogit * w * x` |
| Softmax 反传 | $dz=p\odot(u-\sum p u)$ | 分类项对入选集合，辅助项对全量概率 |
| `keepdims` / `[:, None]` | 广播对齐 | `softmax`、高级索引 |

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/nn-decision/dl/moe/code/demo.py`
