---
title: "信息论精简：熵与 KL — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 信息论精简：熵与 KL — demo.py 代码详解

<a href="/notebook/code/math/information/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/math/information/code
python demo.py
```

CPU、NumPy 即可。两张图：`info_kl_curves.png`（固定 $p$ 扫描 $q$，交叉熵 / KL / 反向 KL）和 `info_entropy_bars.png`（公平硬币 vs 偏置硬币的熵）。单位是 nat（`np.log` 是自然对数），不是 bit。

## 代码逐段详解

### 第1步：`entropy` — 丢掉零再求和

$$
H(p)=-\sum_i p_i\log p_i
$$

约定 $0\log 0=0$。直接 `log(0)` 会得到 `-inf`。

```python
def entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))
```

- **`np.asarray(..., dtype=float)`**：list 也能进。强制 float，避免整数数组 `log` 出问题。
- **语法 `p[p > 0]`**：布尔索引。`p > 0` 是 True/False 数组，只留下正分量。零概率项不参与，等价于 $0\log 0=0$。
- **`p * np.log(p)`**：逐元素。再 `np.sum` 加成标量。前面的负号是定义里的 $-$。
- **`float(...)`**：`np.sum` 可能返回 `np.float64`。转成 Python `float`，打印和画条形更干净。

公平硬币 $[0.5,0.5]$ 的熵是 $\log 2\approx 0.693$ nat；偏置 $[0.9,0.1]$ 更确定，熵更小。终端会打印这两行。

---

### 第2步：`cross_entropy` — 为什么 clip $q$ 不 clip $p$

$$
H(p,q)=-\sum_i p_i\log q_i
$$

交叉熵用**模型** $q$ 的对数。$q_i=0$ 而 $p_i>0$ 时 $\log q_i\to-\infty$，交叉熵爆炸——这正是分类里「对确定错的类给出零概率」的惩罚。数值上用极小正数顶住。

```python
def cross_entropy(p, q, eps=1e-12):
    p = np.asarray(p, dtype=float)
    q = np.clip(np.asarray(q, dtype=float), eps, 1.0)
    return float(-np.sum(p * np.log(q)))
```

- **`np.clip(q, eps, 1.0)`**：每个 $q_i$ 落到 $[10^{-12},1]$。只改 $q$，**不改 $p$**：真分布里的零应当保持零，否则等于改了数据。
- 没有对 $q$ 再归一化：clip 后和可能略大于/小于 1。本 demo 的 $q$ 是 `[q1, 1-q1]` 且 $q1\in[0.05,0.95]$，clip 几乎碰不到，足够画图。

---

### 第3步：`kl` — 一条减法

$$
\mathrm{KL}(p\|q)=H(p,q)-H(p)=\sum_i p_i\log\frac{p_i}{q_i}
$$

```python
def kl(p, q, eps=1e-12):
    return cross_entropy(p, q, eps) - entropy(p)
```

交叉熵 = 熵 + KL。$q=p$ 时 KL $=0$，交叉熵退化为熵。把 KL 建成「两个已有函数的差」，避免再写一遍求和，也保证三条曲线内部一致。

**顺序是 $\mathrm{KL}(p\|q)$**：$p$ 是「真」、$q$ 是「模型」。后面 `kl_rev` 才对调。

---

### 第4步：固定 $p$，扫描 $q$ 的第一分量

```python
p = np.array([0.7, 0.3])
qs = np.linspace(0.05, 0.95, 40)
ce_list, kl_list, kl_rev = [], [], []
for q1 in qs:
    q = np.array([q1, 1 - q1])
    ce_list.append(cross_entropy(p, q))
    kl_list.append(kl(p, q))
    kl_rev.append(kl(q, p))
```

- **$p$ 钉死在 $(0.7,0.3)$**，只动 $q$。左图应在 $q_1=0.7$ 处交叉熵最低、KL 最低（理想为 0）。
- **`1 - q1`**：二元分布第二个质量。保证 $q$ 是概率单纯形上的点。
- **`qs` 不取 0 和 1**：避免 $\log q$ 数值难看；也让反向 KL 在端点不至于完全炸掉，图还能画。
- 空 list 再 `append`：40 个标量。后面 `plot(qs, ce_list)` 直接吃 Python 列表。

```python
axes[0].axvline(p[0], color='k', ls='--', alpha=0.5, label='真实 p1=0.7')
```

竖线标真值。**交叉熵曲线应始终在 KL 上方**，差恰好是常数 $H(p)$（$p$ 固定时）。两线平行，是在验证 `kl = ce - entropy` 没写反。

---

### 第5步：KL 不对称

```python
axes[1].plot(qs, kl_list, label='KL(p‖q)')
axes[1].plot(qs, kl_rev, label='KL(q‖p)')
```

$\mathrm{KL}(p\|q)\neq\mathrm{KL}(q\|p)$。前向 KL（$p$ 真、$q$ 模型）惩罚「$q$ 在 $p$ 有质量的地方给太小」；反向 KL 倾向模式寻求。本图不讲变分推断细节，只让两条曲线**形状不同**，打破「距离都该对称」的直觉。

欧氏距离 $\|p-q\|_2$ 对换 $p,q$ 不变；KL 不是距离。

---

### 第6步：条形图 — 越确定熵越小

```python
fair = np.array([0.5, 0.5])
biased = np.array([0.9, 0.1])
ax.bar(['公平 0.5/0.5', '偏置 0.9/0.1'],
       [entropy(fair), entropy(biased)], ...)
```

`bar` 的第一个参数是类别名，第二个是高度。公平硬币不确定最大（二元时熵最大就是均匀）；偏置几乎总出正面，熵矮一截。

---

### 关键概念速查表

| 概念 | 数学 / 直觉 | 代码 |
|------|-------------|------|
| 熵 | $-\sum p\log p$ | `entropy`，丢掉 $p=0$ |
| 交叉熵 | $-\sum p\log q$ | `cross_entropy`，clip $q$ |
| KL | $H(p,q)-H(p)$ | `kl(p, q)` |
| nat | $\ln$ 不是 $\log_2$ | `np.log` |
| `p[p>0]` | 布尔索引 | 避开 $0\log 0$ |
| `np.clip` | 下限 `eps` | 只用于 $q$ |
| 不对称 | $\mathrm{KL}(p\|q)\neq\mathrm{KL}(q\|p)$ | `kl_list` vs `kl_rev` |
| `linspace` | 扫描 $q_1$ | `qs` |

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/math/information/code/demo.py`
