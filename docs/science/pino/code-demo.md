---
title: "as04 PINO — demo.py"
---

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# as04 PINO — demo.py 代码详解

<a href="/notebook/code/science/pino/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/science/pino/code
python demo.py
```

CPU 即可；完整对比（含 4 个 PINN 实例）大约 1–3 分钟。

## 代码逐段详解

### 第1步：问题定义 —— 变系数扩散方程族

```python
k_a(x) = 1 + a * sin(pi * x)
f(x)   = sin(pi * x)
# 求 -(k u')' = f, u(0)=u(1)=0
```

`solve_variable_coeff_poisson(a)` 用二阶有限体积格式组装三对角矩阵并 `np.linalg.solve`，得到该离散方程的数值解（仍有空间离散误差，不是连续 PDE 的精确解）。后面 PINO 的可微残差**使用同一套离散**，避免「训练残差」和「评估真值」说的不是同一种物理。

### 第2步：PINN 基线（逐实例）

`PINN` 是输入 $x$、输出 $u(x)$ 的 MLP。`train_pinn_for_a` 用 `torch.autograd.grad` 求 $u'$、$u''$，代入展开后的残差

$$
-(k'u' + k u'') - f = 0
$$

已知的 $k$ 与 $k'$ 从 `x_interior.detach()` 计算，作为固定系数复用；$u'$、$u''$ 仍每轮从可求导的 `x_interior` 重新构图。这样不会在第二轮反传时访问已释放的系数计算图。

**为什么这里可以 detach？** 先用乘积法则展开
$
-\partial_x(k_a\partial_xu_\theta)-f
=-k_a'(x)\partial_xu_\theta-k_a(x)\partial_{xx}u_\theta-f(x).
$
本例优化的是网络参数 $\theta$，$a$ 与配点 $x_i$ 固定，所以求 $\partial_\theta L$ 时 $k_a(x_i)$、$k_a'(x_i)$ 都是常数。detach 只断开系数计算的图，不断开网络的高阶导数图。如果改成直接对 $k_a(x)u_\theta'(x)$ 再求 $x$ 导数，或把 $a$ 当成逆问题的待优化参数，就不能照搬这个 detach。

边界损失单独约束 $u(0)=u(1)=0$。每个测试 $a$ 都从 `SEED=42` 重新初始化；这控制了初始化差异，不保证不同范式的算力成本或最终精度相同。

### 第3步：共享的 FNO-lite 算子

`SpectralConv1d`：FFT → 只保留低频 `modes` 个模态 → 可学习复数线性混合 → IFFT。  
`FNO1d`：升维 → 若干「谱卷积 + 1×1 卷积」残差块 → 降维。  
输入通道是 $(k(x), x)$，输出是 $\hat{u}(x)$。

### 第4步：PINO = 同一网络 + 物理损失

```python
loss = loss_data + (0.1 * loss_pde + 1.0 * loss_bc if use_physics else 0.0)
```

- `use_physics=False, n_labeled_use=3` → **FNO-few**
- `use_physics=False, n_labeled_use=11` → **FNO-full**
- `use_physics=True,  n_labeled_use=3` → **PINO**

`pde_residual_and_bc` 在半网格点上构造通量差，与数值求解器一致。

### 第5步：评估与可视化

在 $a\in\{1.5,2.5\}$（插值）和 $\{3.4,4.0\}$（外推）上比较相对 $L^2$ 误差，并画出：

| 文件 | 内容 |
|------|------|
| `as04-01-pino-idea.png` | 三种范式概念图 |
| `pino_comparison.png` | 预测曲线 |
| `pino_error_and_cost.png` | 误差 + 上线耗时 |
| `pino_training_loss.png` | 算子训练损失 |

> [!WARNING]
> 2026-10-08：已修复 PINN 基线的跨轮计算图复用。本次为静态审查，尚未重跑训练或刷新上述结果图；现有图中的误差与耗时不能视为修复后代码的验证结果。

## 关键概念速查表

| 概念 | 含义 | 代码位置 |
|------|------|----------|
| 算子学习 | 学 $k\mapsto u$，而非单个 $u(x)$ | `FNO1d` |
| 数据损失 | 只在标注 $a$ 上监督 | `train_operator` |
| PDE 残差 | 无标签 $a$ 也能提供梯度 | `pde_residual_and_bc` |
| 外推 | 测试 $a$ 超出训练范围 | `A_TEST` 中 $>3$ 的点 |


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/science/pino/code/demo.py`
