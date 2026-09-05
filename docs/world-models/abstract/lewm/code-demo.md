---
title: "LeWM — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# LeWM — demo.py 代码详解

<a href="/notebook/code/world-models/abstract/lewm/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/world-models/abstract/lewm/code
python demo.py
```

CPU 一两分钟。先二维质点（线性编码器 + 预测器，MSE + 高斯代理正则，潜空间 CEM），再倒立摆（`CompactLeWM` 恒等编码，火柴杆图只做可视化）。不是论文级 SIGReg（Epps–Pulley）；玩具用均值/方差或随机投影逼近 $\mathcal{N}(0,1)$。

和 [PETS](/world-models/abstract/pets/code-demo) 对比：PETS 在**观测/状态**上拟合 $s'=f(s,a)$ 再规划；LeWM 在**嵌入** $z$ 里预测和 CEM，目标是「对准目标嵌入」。

## 代码逐段详解

### 第1步：观测为什么要手工特征

```python
def observe(pos):
    feat = np.array([
        pos[0], pos[1], pos[0] ** 2, pos[1] ** 2,
        np.sin(pos[0]), np.cos(pos[1]), pos[0] * pos[1], 1.0,
    ])
    return feat + 0.02 * np.random.randn(8)
```

线性编码器 $z=oW+b$ 表达力弱，把平方、正弦、交叉项写进 $o$，相当于固定的非线性基。噪声 0.02：强迫嵌入不要死记像素级抖动。最后的 `1.0` 是常数特征，给偏置一条通路（`be` 已经有偏置，多一维无妨）。

---

### 第2步：SIGReg 代理 — 为什么需要第二项损失

只有 MSE 时，编码器可以把所有 $z$ 缩到 0 附近，预测器学恒等，「损失」很小但 $z$ 没有几何结构，CEM 对准 $z_g$ 无意义。

```python
def sigreg_proxy(z):
    mu = z.mean(axis=0)
    std = z.std(axis=0) + 1e-6
    return float(np.mean(mu ** 2) + np.mean((std - 1.0) ** 2))
```

逼各维零均值、单位方差。`+1e-6` 防 `std=0`。这是**对角高斯**正则，不是完整特征函数 SIGReg。

倒立摆版：

```python
dirs = np.random.randn(d, n_proj)
dirs /= np.linalg.norm(dirs, axis=0, keepdims=True) + 1e-8
h = z @ dirs
```

**Cramér–Wold**：多维分布由一维投影决定。随机方向上逼近 $\mathcal{N}(0,1)$ 比只看坐标轴更接近「各向同性」。`keepdims=True` 才能按列广播除范数。

---

### 第3步：`LinearLeWM.train_step` — 停梯度与手写反传

```python
z = self.encode(o)
nz_tgt = self.encode(no)
hat = self.predict(z, a)
err = hat - nz_tgt
```

预测器拟合「下一观测的嵌入」，不是原始 $o'$。`Wp` 输入是 `concat(z, a)`，`a` 二维加速度，所以 `Wp` 形状 `(Z_DIM+2, Z_DIM)`。

```python
self.Wp -= lr * (x.T @ err) / len(o)
self.bp -= lr * err.mean(axis=0)
```

MSE 对线性层的梯度：$\partial L/\partial W = X^\top E / B$。**语法 `x.T @ err`**：`(feat, B)` × `(B, Z)`。除以 `len(o)` 当 batch 平均。

编码器：

```python
gz = (err @ Wp_z.T) / len(o)
gz = gz + LAMBDA_REG * (2.0 * mu) / len(o)
self.We -= lr * (o.T @ gz)
```

链式法则：预测误差先流过 `Wp` 里属于 $z$ 的那一块 `Wp[:Z_DIM]`。再加上 $\partial(\mu^2)/\partial z$ 的均值项，把 $z$ 往零均值推。目标侧 `no` 同样推一把，避免「当前 $z$ 被正则、目标 $z$ 仍塌缩」。

注释里的「停梯度」：本实现 **没有** `nz_tgt.detach()`（NumPy 没有计算图），意思是**预测器损失不通过 `nz_tgt` 再改编码器去追 `hat`**——目标嵌入只当固定靶（梯度只从 `hat` 一侧和正则来）。若让 `err` 同时改两端，encoder 可以把 $z$ 和 $z'$ 一起挪到让 `hat` 好猜的地方。

---

### 第4步：潜空间 CEM

```python
for a in s:
    z = model.predict(z, a)
scores.append(-np.sum((z - zg) ** 2))
```

在嵌入里滚 `HORIZON` 步，终点对齐 `zg = encode(observe(goal))`。分数是负距离，CEM 仍取 `argsort` 最大。规划**不**用真实位置，只信模型。

闭环仍 `true_step` 执行 `mu[0]`，下一步重新 `encode(observe(pos))`——MPC。

---

### 第5步：倒立摆 `CompactLeWM` — 恒等编码的教学妥协

```python
def encode(self, o):
    return np.asarray(o, dtype=np.float64)
```

像素 JEPA/LeWM 的编码器在 CPU 上难训稳，这里 $z\equiv (cos,sin,ω)$，损失仍写 MSE+SIGReg，CEM 仍在 $z$ 里对准直立嵌入 `[1,0,0]`。火柴杆 `render` **不进入训练**，只给终局图。

残差预测：

```python
nxt = z + x @ self.Wp + self.bp
```

学 $\Delta z$，再把前两维拉回单位圆（与 PETS 相同理由）。

```python
if np.ndim(a) == 0:
    a = np.full((z.shape[0], 1), float(a))
```

**语法**：规划时 `a` 是 Python float，训练时是 `(B,)`。统一成 `(B,1)` 才能 `concatenate`。`np.ndim(a)==0` 判断标量。

---

### 第6步：`cem_plan_latent` 的代价

```python
cost += np.sum((z - zg) ** 2)
cost -= z[0] * np.exp(-0.05 * z[2] ** 2)
```

既追直立嵌入，又加一项「像 PETS 的 cos 奖励」（`z[0]` 是 cos）。纯距离有时会绕远；奖励项把直立方向写进每一步。`scores = -cost` 再取精英。

`collect` 里 `|θ|>0.8` 则 reset：和 PETS 一样，只在近似线性区采样。

---

### 关键概念速查表

| 概念 | 本章落地 |
|------|----------|
| 两项损失 | MSE($ẑ',z'$) + SIGReg 代理 |
| 潜空间规划 | CEM 滚 `predict`，对齐 $z_g$ |
| 停梯度目标 | 不让目标嵌入追预测 |
| 恒等编码 | 摆的 CPU 妥协 |
| `x.T @ err` | 线性层 MSE 梯度 |
| `np.ndim` | 标量动作对齐 batch |

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/world-models/abstract/lewm/code/demo.py`
