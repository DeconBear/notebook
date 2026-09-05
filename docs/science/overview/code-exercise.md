---
title: "as01 AI4S 全景 — exercise.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# as01 AI4S 全景 — exercise.py 练习指南

<a href="/notebook/code/science/overview/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过补全二阶中心差分、PDE 残差函数和制造解生成器三个模块，从代码层面掌握"PDE 残差"这一贯穿整个 AI4S 系列的核心概念，为下一章训练 PINN 打下基础。

## 预备知识

在开始练习前，确保你已经理解了以下概念（参见 [demo.py 代码详解](./code-demo) 中的详细解释）：

- 二阶中心差分公式：$u''(x_i) \approx \dfrac{u_{i-1} - 2u_i + u_{i+1}}{\Delta x^2}$
- PDE 残差的定义：$r(x) = -u''(x) - f(x)$，真解应使 $r(x)\approx 0$
- 制造解方法：先选定真解 $u(x)$，再反推源项 $f(x) = -u''(x)$，得到可验证的基准问题

## 任务清单

### 任务1：实现二阶中心差分 `second_derivative_fd(u_vals, dx)`

- **用到的公式**：$u''(x_i) \approx \dfrac{u_{i-1} - 2u_i + u_{i+1}}{\Delta x^2}$
- **实现步骤**：
  1. 对内部点（数组切片 `[1:-1]`）应用中心差分公式
  2. 用 `u_vals[:-2]`（左邻居）、`u_vals[1:-1]`（自身）、`u_vals[2:]`（右邻居）三个切片对齐计算
  3. 边界两个端点因缺少邻居，保持为 0（不影响残差比较的结论）
- **需要调用的函数**：NumPy 数组切片、逐元素运算
- **验证技巧**：对 $u(x)=x^2$，解析二阶导数处处为 2，可以用这个简单例子验证你的实现是否正确

### 任务2：实现 PDE 残差函数 `pde_residual(u_vals, x_grid, f_func)`

- **用到的公式**：$r(x) = -u''(x) - f(x)$
- **实现步骤**：
  1. 计算网格间距 `dx = x_grid[1] - x_grid[0]`
  2. 调用任务1中实现的 `second_derivative_fd` 得到 `d2u`
  3. 计算源项 `f_vals = f_func(x_grid)`
  4. 返回 `r = -d2u - f_vals`
- **直觉理解**：这个函数是一个通用工具——给定任意候选解（不管是猜的、拟合的，还是神经网络输出的），都能算出它violates 物理方程的程度

### 任务3（Bonus）：实现制造解生成器 `manufacture_solution(k, n)`

- **数学背景**：取真解 $u(x) = \sin(k\pi x)$（$k$ 为正整数，自动满足边界条件），则：
  $$
  u''(x) = -(k\pi)^2 \sin(k\pi x) \implies f(x) = -u''(x) = (k\pi)^2 \sin(k\pi x)
  $$
- **实现步骤**：
  1. 用 `np.linspace(0, 1, n)` 生成网格 `x`
  2. 计算 `u = np.sin(k * np.pi * x)`
  3. 计算 `f = (k * np.pi) ** 2 * np.sin(k * np.pi * x)`
- **直觉理解**：$k$ 越大，真解振荡越快，曲率也越大——你可以观察不同 $k$ 值下解的形状差异（见 `visualize_manufactured_solutions()` 生成的预览图）

## 验证标准

运行 `python exercise.py`：

1. `test_second_derivative()`：对 $u=x^2$，二阶差分结果应接近解析值 2.0
2. `test_pde_residual()`：真解的残差 RMS 应远小于错解（振幅减半）的残差 RMS
3. `test_manufacture_solution()`（Bonus）：$k=1,2,3$ 时生成的 $(u, f)$ 对应关系应满足 PDE，残差应接近 0

## 延伸思考

- 如果把网格点数 `n` 从 201 增大到 2001，真解的残差 RMS 会如何变化？（提示：二阶中心差分的误差量级是 $O(\Delta x^2)$）
- 如果候选解只是"振幅不对"（比如整体乘以 0.9），残差会如何随振幅误差变化？是线性关系还是其他关系？
- 下一章 as02 中，PINN 会用自动微分代替这里的有限差分来计算 $u''$。你能想到自动微分相比有限差分的优势吗（提示：网格无关、精度、边界处理）？


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/science/overview/code/exercise.py`
