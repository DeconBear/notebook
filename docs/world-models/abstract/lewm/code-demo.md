---
title: "LeWM — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# LeWM — demo.py 代码详解

<a href="/notebook/code/world-models/abstract/lewm/demo.py" target="_blank" download>Download demo.py</a>

> [!WARNING]
> 静态审查后的图片状态：梯度与动作边界已修正，`lewm_cem_mpc.png`、`lewm_cem_iters.png`、`lewm_pendulum.png` 尚未按新版源码重新生成。 本次未执行脚本或验证新输出。

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

逼各维零均值、单位方差。`+1e-6` 防 `std=0`。这是**均值/方差矩匹配**正则，不是完整特征函数 SIGReg。

倒立摆版：

```python
dirs = np.random.randn(d, n_proj)
dirs /= np.linalg.norm(dirs, axis=0, keepdims=True) + 1e-8
h = z @ dirs
```

**Cramér–Wold**：多维分布由一维投影决定。这里只匹配有限个投影的均值和方差，不是检验整个投影分布，不能据此推出高斯性；它提供有限的方向性矩诊断。`keepdims=True` 才能按列广播除范数。

---

### 第3步：`LinearLeWM.train_step` — 停梯度与手写反传

```python
z = self.encode(o)
nz_tgt = self.encode(no)
hat = self.predict(z, a)
err = hat - nz_tgt
```

预测器拟合「下一观测的嵌入」，不是原始 $o'$。`Wp` 输入是 `concat(z, a)`，`a` 二维加速度，所以 `Wp` 形状 `(Z_DIM+2, Z_DIM)`。

对报告的 $L_{\mathrm{pred}}=\operatorname{mean}(E^2)$，先计算 `dhat = 2 * err / err.size`，再算 `x.T @ dhat` 和 `dhat.sum(axis=0)`。归一化同时包含 batch 和嵌入维度。

编码器梯度经过**更新前**的 `Wp[:Z_DIM]`。正则在拼接后的当前/目标嵌入上求导，同时包含均值项和标准差项；只推向零均值并不能防止塌缩。全部梯度计算完成后再一起更新参数，偏置梯度对样本求和，不能重复除以 batch。

注释里的「停梯度」：本实现 **没有** `nz_tgt.detach()`（NumPy 没有计算图），意思是**预测器损失不通过 `nz_tgt` 再改编码器去追 `hat`**——目标嵌入只当固定靶（梯度只从 `hat` 一侧和正则来）。若让 `err` 同时改两端，encoder 可以把 $z$ 和 $z'$ 一起挪到让 `hat` 好猜的地方。

---

### 第4步：潜空间 CEM

```python
for a in s:
    z = model.predict(z, a)
scores.append(-np.sum((z - zg) ** 2))
```

在嵌入里滚 `HORIZON` 步，终点对齐 `zg = encode(observe(goal))`。分数是负距离，CEM 仍取 `argsort` 最大。规划**不**用真实位置，只信模型。

候选动作在评分前就限制到与执行相同的 $[-3,3]$，避免优化无法执行的动作。闭环由 `true_step` 执行 `mu[0]`，下一步重新 `encode(observe(pos))`——MPC。

---

### 第5步：倒立摆 `CompactLeWM` — 恒等编码的教学妥协

```python
def encode(self, o):
    return np.asarray(o, dtype=np.float64)
```

像素 JEPA/LeWM 的编码器在 CPU 上难训稳，这里 $z\equiv (cos,sin,ω)$，CEM 在 $z$ 里对准直立嵌入 `[1,0,0]`。编码器没有可训练参数，随机投影矩统计只作诊断，对预测器的梯度为零，不能当作有效的抗塌缩训练项。火柴杆 `render` **不进入训练**，只给终局图。

残差预测：

```python
nxt = z + x @ self.Wp + self.bp
```

学 $\Delta z$，再把前两维拉回单位圆（与 PETS 相同理由）。训练 MSE 的手写反传包含该归一化的 Jacobian；不能直接把归一化后的误差当作原始线性输出的梯度。

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
| `x.T @ (2 * err / err.size)` | mean MSE 的权重梯度 |
| `np.ndim` | 标量动作对齐 batch |

## 推导补充：从损失到手写梯度

记批量大小为 $B$、嵌入维度为 $d$，线性编码与预测为
$
Z=OW_e+\mathbf1b_e^\top,\quad Z'=O'W_e+\mathbf1b_e^\top,\quad
X=[Z,A],\quad \widehat Z'=XW_p+\mathbf1b_p^\top.
$
本实现报告的预测目标是
$
L_{\rm pred}=\frac1{Bd}\|\widehat Z'-\operatorname{sg}(Z')\|_F^2.
$
目标侧的 $\operatorname{sg}$ 表示停梯度。设 $E=\widehat Z'-Z'$，逐元素求导得
$
D=\frac{2E}{Bd},\quad \nabla_{W_p}L_{\rm pred}=X^\top D,\quad
\nabla_{b_p}L_{\rm pred}=\sum_iD_i,\quad G_Z^{\rm pred}=DW_{p,z}^\top.
$
$W_{p,z}$ 是预测器对应 $Z$ 的前 $d$ 行，必须使用本次前向时、更新前的权重。若先更新预测器再算编码器梯度，就不是在同一个参数点上求导。$E/B$ 对应的是 $\|E\|_F^2/(2B)$，与这里的 mean MSE 不同；仅对预测项漏掉 $2/d$ 会改变它和正则项的相对权重。

把两侧嵌入拼成 $Q=[Z;Z']\in\mathbb R^{M\times d}$，其中 $M=2B$。代理正则为
$
\mu_j=\frac1M\sum_iQ_{ij},\quad s_j=\sqrt{\frac1M\sum_i(Q_{ij}-\mu_j)^2},
\quad \sigma_j=s_j+\varepsilon,
$
$
R(Q)=\frac1d\sum_j[\mu_j^2+(\sigma_j-1)^2].
$
在 $s_j>0$ 时，链式法则给出
$
\frac{\partial R}{\partial Q_{ij}}
=\frac{2}{Md}\left[\mu_j+(\sigma_j-1)\frac{Q_{ij}-\mu_j}{s_j}\right].
$
第一项移动均值，第二项改变离散程度。只保留均值项无法防止嵌入全变为零。按前后 $B$ 行拆开正则梯度，得到
$
G_Z=G_Z^{\rm pred}+\lambda G_Z^{\rm reg},\quad
G_{Z'}=\lambda G_{Z'}^{\rm reg},
$
$
\nabla_{W_e}L=O^\top G_Z+O'^\top G_{Z'},\quad
\nabla_{b_e}L=\sum_iG_{Z,i}+\sum_iG_{Z',i}.
$
预测误差不通过目标侧，但正则仍作用于目标侧。梯度已含批量平均，偏置必须求和，不能再次取平均；所有梯度计算完再统一更新参数。

当 $s_j=0$ 时标准差不可微，代码把对应标准差导数置零以避免除零。这不是“保证从完全塌缩恢复”的定理。矩匹配也不保证高斯性：等概率取 $\{-1,+1\}$ 的变量均值零、方差一，却不是高斯。因此这里只是均值/方差代理，不能等同完整 SIGReg。

### 单位圆归一化也必须反传

倒立摆预测的前两维原始输出记为 $u\in\mathbb R^2$，实际输出为
$
p(u)=\frac{u}{\|u\|+\varepsilon}.
$
对 $\rho=\|u\|>0$，
$
J_p(u)=\frac{I}{\rho+\varepsilon}
-\frac{uu^\top}{\rho(\rho+\varepsilon)^2}.
$
若归一化后的 MSE 梯度为 $g$，线性层应收到 $J_p(u)^\top g$，不能直接收到 $g$。当 $\varepsilon=0$ 时，径向扰动被归一化消除，可用来检查公式。实现另处理 $\rho=0$ 的数值分支。

倒立摆编码器为恒等映射，没有可训练参数。其随机投影矩统计只作诊断，对预测器梯度为零，不能用来证明学到了抗塌缩表示。

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/world-models/abstract/lewm/code/demo.py`
