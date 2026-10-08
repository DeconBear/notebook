---
title: "因果世界模型 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 因果世界模型 — demo.py 代码详解

<a href="/notebook/code/world-models/causal/ladder/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/world-models/causal/ladder/code
python demo.py
```

秒级，只依赖 NumPy。SCM 极简：$Z$ 上游变量 → 观测策略里的 $A$；$Y:=A+$噪声（机制**不含** $Z$）。对比「用 $Z$ 预测 $Y$」和「用 $A$ 预测 $Y$」在观测分布 vs $do(A)$ 上的 MSE。

## 代码逐段详解

### 第1步：观测抽样 — 相关从哪来

```python
def sample_observational(n=2000):
    z = np.random.randn(n)
    a = np.tanh(1.5 * z) + 0.1 * np.random.randn(n)
    y = a + 0.15 * np.random.randn(n)
    return z, a, y
```

- $Z\sim\mathcal{N}(0,1)$。
- **观测策略** $A=\tanh(1.5Z)+\varepsilon$：动作几乎由上游变量 $Z$ 决定。`tanh` 把动作压到大约 $(-1,1)$，避免线性里 $A$ 无限大。
- **机制** $Y=A+\eta$：世界只听动作。$Z$ 不进 $Y$ 的公式。

观测数据上 $Z$ 与 $Y$ 仍相关：因为 $Z\to A\to Y$。回归 $Y\sim Z$ 会看起来「准」——这条相关沿真实因果链传递，$Z$ 对 $Y$ 有经由 $A$ 的间接因果效应。它不是 $A\to Y$ 的混淆变量，因为没有绕过 $A$ 的后门路径；问题是干预 $A$ 后，这条用于预测的路径被切断。

---

### 第2步：干预抽样 — `do(A)` 切断哪条边

```python
def sample_interventional(n=2000):
    z = np.random.randn(n)
    a = np.random.uniform(-1.5, 1.5, size=n)  # 切断 Z→A
    y = a + 0.15 * np.random.randn(n)
```

$A$ 改成均匀随机，**不再**是 $Z$ 的函数。这是随机干预：每个样本先抽一个 $a$，再施加 $do(A=a)$；汇总结果是不同 $a$ 的干预分布的混合，而不是所有样本固定为同一个 $a$。$Y$ 公式不变（机制不变）。此时 $Z$ 与 $Y$ 独立，$Y\sim Z$ 应崩；$Y\sim A$ 仍对。

**语法 `uniform(-1.5, 1.5, size=n)`**：每个样本独立抽一个动作，形状 `(n,)`。

---

### 第3步：一元线性最小二乘

```python
def fit_linear(x, y):
    x1 = np.stack([x, np.ones_like(x)], axis=1)
    w, *_ = np.linalg.lstsq(x1, y, rcond=None)
    return w
```

设计矩阵两列：斜率和截距。`ones_like(x)` 与 $x$ 同形状的全 1。`lstsq` 解 $\min\|Xw-y\|^2$，返回 `(w, residuals, rank, s)`；`w, *_` 只要 $w$，其余丢掉。

**语法 `*_`**：解包时忽略后面所有值。

```python
def predict(w, x):
    return w[0] * x + w[1]
```

$w[0]$ 斜率，$w[1]$ 截距。NumPy 广播：$x$ 可以是向量。

```python
return float(np.mean((y_hat - y) ** 2))
```

`float(...)` 把 0 维 numpy 标量变成 Python float，打印更干净。

---

### 第4步：四格实验设计

在**同一套**观测数据上拟合两个模型：`w_z = fit(Z,Y)`，`w_a = fit(A,Y)`。然后：

| 测试分布 | 特征 | 预期 |
|----------|------|------|
| 新的观测样本 | $Z$ | MSE 低（相关还在） |
| 新的观测样本 | $A$ | MSE 低（真机制） |
| 干预样本 | $Z$ | MSE **高**（边断了） |
| 干预样本 | $A$ | MSE 仍低 |

`main` 里 `sample_observational` 调用两次：一次训练、一次观测测试，避免「在训练集上报测试误差」。干预集单独抽。

左图 `z_tr[::5]`：**切片步长 5**，散点少一点，图更干净。右图 `barh(..., labels[::-1])`：`[::-1]` 倒序，条形图从上到下与阅读顺序一致。

---

### 和第5步：世界模型在听什么

若世界模型只用上游变量 $Z$ 预测结果而遗漏动作，规划 $do(a)$ 时会错。本 demo 用线性回归代替深度网，只为把 **关联 vs 干预** 画成一根会爆的柱子。完整因果阶梯（L1 观测 / L2 干预 / L3 反事实）见章节正文。

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/world-models/causal/ladder/code/demo.py`
