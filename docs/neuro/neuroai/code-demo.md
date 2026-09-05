---
title: "NeuroAI：启发、对齐与约束 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# NeuroAI：启发、对齐与约束 — demo.py 代码详解

<a href="/notebook/code/neuro/neuroai/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/neuro/neuroai/code
python demo.py
```

CPU、NumPy。一张图 `neuroai_credit.png`：左是 Hebb 局部更新轨迹，右是线性读出在 AND / XOR 上的 BCE。没有 PyTorch。对照的是「突触只看见 pre/post」vs「一条全局损失回传到所有权重」。

## 代码逐段详解

### 第1步：`sigmoid` 与数值夹紧

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
```

$\sigma(x)=1/(1+e^{-x})$。`clip` 到 $\pm20$：挡住 `exp` 溢出。`20` 已经让 $\sigma$ 饱和到 0/1，对 BCE 足够。

---

### 第2步：`hebbian_update` — 只有共激活才加

$$
w\leftarrow w+\eta\, y\, x
$$

```python
def hebbian_update(w, pre, post, lr=0.05):
    return w + lr * post * pre
```

- **`post * pre`**：标量 post 乘向量 pre，广播。`post=0` 时整步 $\Delta w=0$，第二条输入再大也不改。
- **没有减均值、没有 Oja 归一化**：权重可以一直涨。本 demo 只走 4 步，看趋势即可。
- 返回新数组，不原地改；调用方 `traj.append(w.copy())`。`.copy()` 防止列表里存到同一引用。

左图协议：

```python
for pre, post in [([1, 0], 1.0), ([0, 1], 0.0), ([1, 1], 1.0), ([1, 0], 1.0)]:
    w = hebbian_update(w, np.array(pre, dtype=float), post)
```

1. 只有第 1 维开、post=1 → $w_1$ 升。
2. 只有第 2 维开、**post=0** → 两维都不动。
3. 两维都开、post=1 → 两维都升。
4. 再加强 $w_1$。

所以 $w_1$ 应明显高于 $w_2$：Hebb 把「从未与输出共激活」的突触留下。这是局部规则，**不需要知道任务是 AND 还是 XOR**。

---

### 第3步：`train_readout` — 全局 BCE + 线性头

$$
p=\sigma(Xw+b),\quad
\ell=-\frac1N\sum_i\bigl[y\log p+(1-y)\log(1-p)\bigr]
$$

$$
\frac{\partial\ell}{\partial w}=\frac{X^\top(p-y)}{N}
$$

```python
p = sigmoid(X @ w + b)
losses.append(float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))))
w -= lr * (X.T @ (p - y) / len(y))
b -= lr * float(np.mean(p - y))
```

- **`X @ w`**：`(4,2)@(2,)→(4,)`。`@` 矩阵乘；`*` 才是逐元素。
- **`eps=1e-9`**：挡住 `log(0)`。
- **`X.T @ (p-y)`**：`(2,4)@(4,)→(2,)`，对每个权重的梯度。除以 `len(y)` 与 `mean` 损失一致。
- **没有隐层**：这就是一个感知机 + sigmoid。线性可分的 AND BCE 应降到接近 0；XOR 四顶点不是线性可分，终损会停在高处。

`w = np.zeros(X.shape[1])`：从全零起步，和 Hebb 左图同一「空白突触」叙事，但更新规则完全不同——这里每一步四个样本**都**进损失（全量 batch）。

---

### 第4步：同一 $X$，两个标签

```python
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_and = np.array([0, 0, 0, 1], dtype=float)
y_xor = np.array([0, 1, 1, 0], dtype=float)
```

布尔立方体四个顶点。AND 只有 `(1,1)` 为正；XOR 是对角为正。`dtype=float` 才能进 `y * log(p)`。右图绿线应下降、红线应卡住——**全局损失并不能魔法般拆开线性不可分问题**；要非线性回路（或核、或隐层）。这就是 NeuroAI 里「对齐」的一半：大脑明显能做 XOR 类计算，单层读出模型对不齐。

`loss_and[-1]` / `loss_xor[-1]` 打到终端。语法 `[-1]` 最后一项。

左图 `plot(..., '-o')` / `'-s'`：点+线，四次更新步数很少，没有点会看不清。`traj` 含初值，所以曲线长度是 5 不是 4。右图 `lw=2` 只为叠在格子上仍能分色。

**和 STDP 章的关系**：`hebbian_update` 没有时间窗，只看同一时刻的 `pre`/`post` 标量；STDP 章的迹才处理「谁先谁后」。NeuroAI 这一页要的对比是 **局部相关 vs 全局 BCE**，不是时间编码。XOR 卡住不是优化器没调好（250 epoch、`lr=0.2` 对四样本足够），是模型类不够。

---

### 关键概念速查表

| 概念 | 数学 / 直觉 | 代码 |
|------|-------------|------|
| Hebb | $\Delta w\propto y x$ | `hebbian_update` |
| 局部 | 看不见全局损失 | 左图 4 步 |
| `.copy()` | 轨迹快照 | `traj.append(w.copy())` |
| 读出 | $p=\sigma(Xw+b)$ | `train_readout` |
| BCE | $-y\log p-(1-y)\log(1-p)$ | `eps` 防 log0 |
| 梯度 | $X^\top(p-y)/N$ | `w -= lr * ...` |
| AND | 线性可分 | 绿线下降 |
| XOR | 线性不可分 | 红线卡住 |
| `clip` | 防 `exp` 溢出 | `sigmoid` |

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/neuro/neuroai/code/demo.py`
