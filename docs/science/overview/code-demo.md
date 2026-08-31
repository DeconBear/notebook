---
title: "as01 AI4S 全景 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# as01 AI4S 全景 — demo.py 代码详解

<a href="/notebook/code/science/overview/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/science/overview/code
python demo.py
```

本 demo 只依赖 NumPy 和 Matplotlib，不需要 PyTorch，可以在任何 CPU 环境下秒级运行完毕。

## 代码逐段详解

### 第1步：用"制造解方法"构造可验证的 PDE 问题

我们考虑一维 Poisson 方程边值问题：

$$
-u''(x) = f(x), \quad x \in (0,1), \qquad u(0) = u(1) = 0
$$

**制造解方法（method of manufactured solutions）**：先选定一个满足边界条件的函数作为"真解"，再反推出对应的源项 $f$，这样我们就获得了一个精确已知、可以直接验证的基准问题：

```python
def u_true(x):
    """解析真解: u(x) = sin(pi x)，自动满足 u(0)=u(1)=0"""
    return np.sin(np.pi * x)

def f_source(x):
    """反推源项: f(x) = -u_true''(x) = pi^2 sin(pi x)"""
    return (np.pi ** 2) * np.sin(np.pi * x)
```

**为什么这样做**：如果我们不知道精确解，就无法判断一个数值方法/神经网络给出的答案是否正确。制造解方法是验证任何 PDE 求解器（包括下一章的 PINN）正确性的标准手段。

### 第2步：用有限差分近似二阶导数

在没有自动微分工具的情况下，我们用**二阶中心差分**近似二阶导数：

$$
u''(x_i) \approx \frac{u_{i-1} - 2u_i + u_{i+1}}{\Delta x^2}
$$

```python
def second_derivative_fd(u_vals, dx):
    n = len(u_vals)
    d2u = np.zeros(n)
    d2u[1:-1] = (u_vals[:-2] - 2 * u_vals[1:-1] + u_vals[2:]) / (dx ** 2)
    return d2u
```

**直觉**：中心差分公式来自泰勒展开——把 $u(x_i \pm \Delta x)$ 在 $x_i$ 处展开到二阶项，消去一阶项后就能反解出二阶导数的近似。误差量级是 $O(\Delta x^2)$，网格越密，近似越精确。

在下一章 as02 中，PINN 会用**自动微分（autograd）**代替这里的有限差分——自动微分给出的是**精确**的解析导数（在浮点精度内），而不是有限差分的离散近似，这是 PINN 相比传统数值方法的一个重要优势：无网格、无离散化误差。

### 第3步：定义 PDE 残差 —— 全系列最核心的概念

```python
def pde_residual(u_vals, x_grid):
    dx = x_grid[1] - x_grid[0]
    d2u = second_derivative_fd(u_vals, dx)
    r = -d2u - f_source(x_grid)
    return r
```

**PDE 残差** $r(x) = -u''(x) - f(x)$ 衡量"候选函数 $u$ 有多不满足方程"：

- 真解代入 → $r(x) \equiv 0$（仅有离散化误差）
- 任意其他函数代入 → $r(x)$ 一般不为 0，且偏离程度反映了该函数"错得有多离谱"

这个残差正是 **PINN（as02）损失函数的核心组成部分**：PINN 用神经网络输出 $u_\theta(x)$ 代替这里的候选函数，用自动微分算出精确的 $u_\theta''$，然后最小化 $\mathbb{E}[r(x)^2]$ 作为训练目标之一。

### 第4步：构造"看起来还行但违反物理"的错解

```python
def u_wrong_linear_bump(x):
    """曲率是常数的二次函数，与真解的正弦曲率完全不同"""
    return 4.0 * x * (1.0 - x)

def u_wrong_wrong_amplitude(x):
    """频率不对的正弦函数"""
    return 0.6 * np.sin(2 * np.pi * x)
```

这两个函数都满足边界条件 $u(0)=u(1)=0$，图像形状也大致是"中间鼓起来的曲线"——如果只看形状，很容易被误认为是"差不多对的解"。但它们的二阶导数（曲率）与真解完全不同，代入残差公式后会立刻暴露出来。

### 第5步：可视化对比 —— 真解 vs 错解的残差

```python
r1 = pde_residual(u_true(x), x)                    # 真解
r2 = pde_residual(u_wrong_linear_bump(x), x)        # 错解1
r3 = pde_residual(u_wrong_wrong_amplitude(x), x)    # 错解2
```

结果保存在 `images/ai4s_poisson_residual.png` 中，上排是三个候选解的函数图像，下排是对应的残差曲线。

**关键数值对比**（脚本会打印）：

| 候选解 | RMS(residual) |
|--------|---------------|
| 真解 $\sin(\pi x)$ | ≈ 0.0001（仅离散化误差） |
| 错解 $4x(1-x)$ | ≈ 3.4 |
| 错解 $0.6\sin(2\pi x)$ | ≈ 18.3 |

残差的量级相差几万倍，即使函数图像"看起来"相差不大。这就是为什么 PINN 能够仅凭 PDE 残差就学到正确的解——残差信号对"物理错误"极其敏感。

### 第6步：维度灾难示意图

```python
classical_dof = grid_points_per_dim ** dims.astype(float)   # ~ 50^d
nn_params = 5000 * dims + 2000                                # 示意性温和增长
```

这段代码用一个简化的示意模型说明：经典网格法的自由度随维度呈指数增长（$N^d$），而神经网络的参数量增长要温和得多。这是数据驱动/算子学习方法在高维科学计算问题（如金融衍生品定价、多体量子力学）中具有潜在优势的直觉来源（注意：这只是帮助理解动机的示意图，不是严格的复杂度证明）。

### 第7步：AI4S 全景地图（用 Matplotlib 手绘的概念示意图）

```python
_draw_box(ax, (0.5, 5.6), 2.6, 1.1, 'FDM / FEM\n...', facecolor='#f4b6b6')
_draw_arrow(ax, (x0, 6.7), (x0, 6.95), color='gray')
```

用 `FancyBboxPatch` 和 `FancyArrowPatch` 手工绘制方框和箭头，构造出一张"方法论地图"：横向展示 FDM/FEM → PINN → FNO/PINO → GNN 在"物理驱动↔数据驱动"光谱上的位置，下方展示 AlphaFold / GraphCast / AlphaChip 三个代表性成果，并在底部总结它们的共同主题——"把科学问题表述为学习一个映射"。

### 关键概念速查表

| 概念 | 一句话解释 | 代码位置 |
|------|-----------|---------|
| 制造解方法 | 先定真解再反推源项，得到可验证的基准问题 | `u_true()`, `f_source()` |
| 二阶中心差分 | $u'' \approx (u_{i-1}-2u_i+u_{i+1})/\Delta x^2$ | `second_derivative_fd()` |
| PDE 残差 | $r(x)=-u''(x)-f(x)$，衡量候选解违反方程的程度 | `pde_residual()` |
| 维度灾难 | 经典网格法自由度 $\sim N^d$，随维度指数爆炸 | `plot_discretization_cost()` |
| 算子学习光谱 | FDM/FEM ↔ PINN ↔ FNO/PINO ↔ GNN 的方法论地图 | `plot_ai4s_landscape_map()` |

## 完整代码

<<< @/science/overview/code/demo.py
