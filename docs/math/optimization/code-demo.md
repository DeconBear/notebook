---
title: "优化与梯度 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 优化与梯度 — demo.py 代码详解

<a href="/notebook/code/math/optimization/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/math/optimization/code
python demo.py
```

CPU、NumPy 即可。一张图 `opt_gd_traj.png`：左是参数平面等高线 + 两条梯度下降轨迹，右是损失的对数轴。对比 $\eta=0.95$（过大，来回抖）和 $\eta=0.15$（合适，走进碗底）。没有自动微分，梯度手写。

## 代码逐段详解

### 第1步：碗状损失 — 最小值故意不在原点

$$
L(w)=(w_1-1)^2 + 0.25(w_2+0.5)^2
$$

最低点在 $(1,-0.5)$，不是 $(0,0)$。若最小值在原点，初值选不好会让「往原点走」和「往最优点走」分不清。

```python
def loss(w):
    return (w[0] - 1.0) ** 2 + 0.25 * (w[1] + 0.5) ** 2
```

- **`w[0]` / `w[1]`**：把二维参数当成长度为 2 的数组。`w` 可以是 `list` 或 `ndarray`。
- **语法 `** 2`**：平方。`(w[0]-1.0) ** 2` 是 $(w_1-1)^2$。不要写成 `^ 2`（按位异或）。
- **`0.25 *`**：第二维更扁。等高线是椭圆不是圆，两条轴的曲率不同，学习率过大时更容易沿陡轴振荡。

解析梯度：

$$
\nabla L = \bigl(2(w_1-1),\; 0.5(w_2+0.5)\bigr)
$$

```python
def grad(w):
    return np.array([2 * (w[0] - 1.0), 0.5 * (w[1] + 0.5)])
```

对 $w_2$：$0.25\cdot 2(w_2+0.5)=0.5(w_2+0.5)$。系数必须和 `loss` 一致，否则轨迹不是真梯度下降。本章不调用 `autograd`，就是为了让「导数是手算出来的」可见。

---

### 第2步：`gd` — 迭代、拷贝、轨迹

$$
w \leftarrow w - \eta\nabla L(w)
$$

```python
def gd(w0, eta, steps=40):
    w = np.array(w0, dtype=float)
    traj = [w.copy()]
    losses = [loss(w)]
    for _ in range(steps):
        w = w - eta * grad(w)
        traj.append(w.copy())
        losses.append(loss(w))
    return np.array(traj), np.array(losses)
```

- **`dtype=float`**：避免传入整数数组后减法变成整除截断。
- **`w.copy()`**：`traj` 若存 `w` 本身，NumPy 数组是可变对象，下一步原地改（这里其实是重新绑定，但养成拷贝习惯），列表里会全变成终值。`.copy()` 记下这一瞬间的坐标。
- **`_`**：循环变量不用。只关心「做 `steps` 次」。
- **`eta * grad(w)`**：学习率乘梯度向量，逐元素。`w - ...` 是向量减法。
- 返回 `traj` shape `(steps+1, 2)`（含起点），`losses` 长度相同。

**为什么记录损失？** 左图画路径，右图看 $L$ 是否真的下降。路径漂亮但损失不降，可能是画错等高线；损失降但路径乱，才是学习率问题。

---

### 第3步：两条轨迹同一起点

```python
w0 = np.array([-1.5, 2.0])
traj_big, loss_big = gd(w0, eta=0.95, steps=25)
traj_ok, loss_ok = gd(w0, eta=0.15, steps=40)
```

同一 $w_0$，只改 $\eta$ 和步数。大学习率少走几步（25）已经够看出振荡；小学习率多走一点（40）才到碗底。这不是公平的「同预算对比」，是为了让两种病症都完整出现在一张图上。

二次函数沿某方向曲率若为 $\lambda$，稳定要求 $\eta < 2/\lambda$。这里 $w_1$ 方向 $\lambda=2$，理论上 $\eta<1$。`0.95` 贴着边界，容易来回跳；`0.15` 远小于 1，单调靠近。

---

### 第4步：等高线 — `meshgrid` 在铺平面

```python
xs = np.linspace(-2, 2.5, 200)
ys = np.linspace(-2, 2.5, 200)
XX, YY = np.meshgrid(xs, ys)
ZZ = (XX - 1.0) ** 2 + 0.25 * (YY + 0.5) ** 2
cs = axes[0].contour(XX, YY, ZZ, levels=20, cmap='Blues')
```

- **`meshgrid`**：把两个 1D 网格铺成 2D 坐标。`XX[i,j]` 是第 $j$ 个 $x$，`YY[i,j]` 是第 $i$ 个 $y$。然后 `ZZ` 与 `loss` **同一公式**，颜色才和轨迹可比。
- **`contour(..., levels=20)`**：20 条等值线。`clabel` 把数值写在线上。
- **`traj_big[:, 0], traj_big[:, 1]`**：轨迹的 $w_1$、$w_2$。`o-` 是「点 + 连线」，才能看见步与步之间的跳跃。
- **`scatter([1], [-0.5], marker='*')`**：最优星标。列表包一层是因为 `scatter` 要数组，标量有的版本会报错。

---

### 第5步：右图 `semilogy` — 为什么用对数轴

```python
axes[1].semilogy(loss_big, label='η=0.95')
axes[1].semilogy(loss_ok, label='η=0.15')
```

- **`semilogy`**：纵轴对数、横轴线性。损失从几十降到 $10^{-3}$，线性轴会把后期压成一条平线，看不出小学习率还在降。
- 终端打印 `loss_ok[-1]` / `loss_big[-1]`：**语法 `[-1]`** 是最后一个元素。小 $\eta$ 终损应接近 0；大 $\eta$ 往往仍停在高处或来回跳。

`np.random.seed(42)` 在文件开头设了，但本脚本轨迹是确定性的，种子不影响结果。

---

### 关键概念速查表

| 概念 | 数学 / 直觉 | 代码 |
|------|-------------|------|
| 损失 | 椭圆碗，最低点 $(1,-0.5)$ | `loss(w)` |
| 梯度 | 手写 $\nabla L$ | `grad(w)` |
| GD | $w-\eta\nabla L$ | `w = w - eta * grad(w)` |
| `.copy()` | 快照，防引用串味 | `traj.append(w.copy())` |
| `** 2` | 平方 | 损失与等高线 |
| `meshgrid` | 铺 $(x,y)$ 网格 | 画等高线 |
| `[:, 0]` | 轨迹的第一维 | `traj_big[:, 0]` |
| `semilogy` | 对数纵轴看小损失 | 右图 |
| `[-1]` | 最后一个值 | 终损 |

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/math/optimization/code/demo.py`
