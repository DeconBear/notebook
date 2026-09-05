---
title: "观测器与卡尔曼 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 观测器与卡尔曼 — demo.py 代码详解

<a href="/notebook/code/control/modern/kalman/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/control/modern/kalman/code
python demo.py
```

CPU、NumPy 即可。一张图 `kf_1d.png`：左是真值 / 测量点 / $\hat x$，右是方差 $P_k$。植物是一维随机游走 $x\leftarrow x+w$，测量 $z=x+v$。`np.random.seed(42)` 让噪声可复现。

## 代码逐段详解

### 第1步：噪声强度

```python
N = 80
Q = 0.04
R = 0.25
```

- **`Q`**：过程噪声方差。$x$ 每步加 $\mathcal{N}(0,\sqrt{Q})$。$Q$ 越大，模型越不可信，增益会偏测量。
- **`R`**：测量噪声方差。这里 $R>Q$，测量比随机游走更吵，滤波器应当平滑掉一部分点。
- **`np.random.normal(0, np.sqrt(Q))`**：`normal` 的第二参数是**标准差**不是方差，所以要开方。

---

### 第2步：预测 / 更新

$A=1$、无 $u$，预测均值不变，只把方差加上 $Q$：

```python
P_pred = P + Q
xhat_pred = xhat
kg = P_pred / (P_pred + R)
xhat = xhat_pred + kg * (z - xhat_pred)
P = (1.0 - kg) * P_pred
```

- **`kg = P/(P+R)`**：一维时矩阵公式 $PH^\top(HPH^\top+R)^{-1}$ 退化成除法，$H=1$。
- **`z - xhat_pred`**：新息（innovation）。预测对了就接近 $0$，滤波器几乎不动。
- **`P = (1-kg) P_pred`**：Joseph 形式的一维特例。`kg` 越接近 $1$（信测量），事后方差越小。

列表 `xs` 比 `zs` 长 1：先记下 $x_0$，循环里才产生测量。所以 RMSE 用 `xs[1:]` 对齐 `zs`。

---

### 第3步：RMSE

```python
rmse_z = float(np.sqrt(np.mean((zs - xs[1:]) ** 2)))
rmse_f = float(np.sqrt(np.mean((xhs[1:] - xs[1:]) ** 2)))
```

- **`np.mean((·)**2)`**：均方；再开方是均方根误差。滤波 RMSE 应明显小于「把 $z$ 当估计」。
- **`scatter` vs `plot`**：测量画成点，强调它吵；估计画成线，强调平滑。

右图 $P$ 会先从初始 $1$ 掉下来，然后每步「$+Q$ 再乘 $(1-K_g)$」，形成锯齿。那就是预测变大、更新变小。

二维矩阵写法见 [LQR 章的 demo](/control/modern/lqr/code-demo)：那里 $H=[1,0]$，除法变成 `np.linalg.solve`。

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/control/modern/kalman/code/demo.py`
