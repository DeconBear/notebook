---
title: "as02 PINN — exercise.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# as02 PINN — exercise.py 练习指南

<a href="/notebook/code/science/pinn/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过亲手实现自动微分求二阶导数、PINN 的复合损失函数、以及一个最小训练循环，掌握 PINN 训练的三大核心步骤，并直观验证"仅用 PDE 残差 + 边界条件就能训练出正确解"这一核心思想。

## 预备知识

在开始练习前，确保你已经理解（参见 [demo.py 代码详解](./code-demo)）：

- `torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]` 的用法
- 二阶导数 = 对一阶导数再求一次导数，且必须两次都设置 `create_graph=True`
- 复合损失：$L = L_{\text{pde}} + \lambda_{\text{bc}} L_{\text{bc}}$

## 任务清单

### 任务1：实现自动微分二阶导数 `autograd_second_derivative(model, x)`

- **用到的工具**：`torch.autograd.grad`
- **实现步骤**：
  1. 前向传播：`u = model(x)`
  2. 一阶导数：`du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]`
  3. 二阶导数：对 `du_dx` 重复上一步操作
- **验证技巧**：文件里提供了 `QuadraticProbe`，它直接计算 $u(x)=x^2$，解析二阶导数处处为 2 —— 可以用它验证你的实现是否正确，而不需要训练任何网络

### 任务2：实现复合损失函数 `compute_total_loss(model, x_f, x_b, u_b_true, lambda_bc=10.0)`

- **用到的公式**：$L = L_{\text{pde}} + \lambda_{\text{bc}} L_{\text{bc}}$，其中 $L_{\text{pde}}=\text{mean}(r(x_f)^2)$，$r(x)=-u''(x)-f(x)$
- **实现步骤**：
  1. 调用任务1实现的函数得到 `d2u_dx2`
  2. 组装残差 `residual = -d2u_dx2 - f_source(x_f)`
  3. `loss_pde = torch.mean(residual ** 2)`
  4. `u_b_pred = model(x_b)`，`loss_bc = torch.mean((u_b_pred - u_b_true) ** 2)`
  5. `loss_total = loss_pde + lambda_bc * loss_bc`
- **直觉理解**：`loss_pde` 用到的每一个点都没有"标签"（不知道真实的 $u$ 值），只知道它必须让残差趋近于 0；`loss_bc` 是唯一"知道真实数值"的部分（边界值已知为 0）

### 任务3（Bonus）：实现最小训练循环 `train_mini_pinn(n_epochs=500, lr=1e-3)`

- **实现步骤**：标准的 PyTorch 训练循环——`optimizer.zero_grad()` → 计算损失 → `loss.backward()` → `optimizer.step()`，循环 `n_epochs` 次，把每轮的 `loss_total.item()` 存入列表
- **易错点**：`x_f` 必须在训练循环外构造好，并设置 `requires_grad_(True)`；不要在循环内重新构造（否则每轮都要重新建图，也容易忘记设置 `requires_grad`）

## 验证标准

运行 `python exercise.py`：

1. `test_second_derivative()`：对 $u(x)=x^2$，自动微分算出的二阶导数应接近解析值 2.0
2. `test_composite_loss()`：`loss_total` 应严格等于 `loss_pde + lambda_bc * loss_bc`（数值验证），且两个子损失均非负
3. `test_training_reduces_loss()`（Bonus）：训练 500 轮后，损失应降到初始值的 10% 以下

## 延伸思考

- 如果把 `lambda_bc` 从 10 调到 1，训练后网络在边界处的值会偏离 0 吗？试着修改 `train_mini_pinn` 观察结果。
- 如果把激活函数从 `Tanh` 换成 `ReLU`，`compute_total_loss` 还能正常训练吗？为什么？（提示：想想 ReLU 的二阶导数是什么）
- 本练习的配点是固定的均匀网格。如果改成每个 epoch **随机重新采样**配点（而不是固定同一批），你觉得对最终精度会有什么影响？这在处理高维问题时通常是必需的技巧。

## 完整代码

<<< @/science/pinn/code/exercise.py
