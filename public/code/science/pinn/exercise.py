# -*- coding: utf-8 -*-
"""
===============================================================================
as02_pinn/code/exercise.py — PINN 练习
===============================================================================
本练习文件中，你需要完成以下任务：

练习目标：
  1. 用 torch.autograd.grad 实现二阶导数计算（PINN 的核心工具）
  2. 实现 PDE 残差损失 + 边界条件损失的复合损失函数
  3. 实现一个最小训练循环，并观察损失是否随训练下降（Bonus）

提示：
  - torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]
    对向量输出 y 关于输入 x 求导，create_graph=True 让导数本身可以再被求导
  - 二阶导数 = 对一阶导数再求一次导数
  - 复合损失: L = L_pde + lambda_bc * L_bc

运行方式：
  python exercise.py
===============================================================================
"""

import numpy as np
import torch
import torch.nn as nn

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================================
# 辅助代码（已实现，可直接使用）
# ============================================================================

class TinyPINN(nn.Module):
    """一个极简的 PINN 网络: 1 -> 16 -> 16 -> 1，tanh 激活。"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16), nn.Tanh(),
            nn.Linear(16, 16), nn.Tanh(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


class QuadraticProbe(nn.Module):
    """
    一个"假装是网络"的模块，直接计算 u(x) = x^2。
    用于验证你实现的二阶导数函数是否正确 —— 因为 u(x)=x^2 的二阶导数
    处处等于解析值 2，非常适合作为单元测试的基准。
    """

    def forward(self, x):
        return x ** 2


def f_source(x):
    """as02 主问题的源项: f(x) = pi^2 sin(pi x)"""
    return (np.pi ** 2) * torch.sin(np.pi * x)


# ============================================================================
# 任务 1: 实现二阶导数计算 (约 8 行)
# ============================================================================

def autograd_second_derivative(model, x):
    """
    用自动微分计算 model 输出关于输入 x 的二阶导数 d^2u/dx^2。

    步骤:
      1. 前向传播: u = model(x)
      2. 一阶导数: du_dx = grad(u, x, grad_outputs=ones_like(u), create_graph=True)[0]
      3. 二阶导数: d2u_dx2 = grad(du_dx, x, grad_outputs=ones_like(du_dx), create_graph=True)[0]

    参数:
        model: 一个 nn.Module（或行为类似的可调用对象），输入 (N,1) 输出 (N,1)
        x: (N, 1) tensor，必须已经设置 requires_grad_(True)
    返回:
        d2u_dx2: (N, 1) tensor，二阶导数
    """
    # TODO: 完成以下步骤
    # --- BEGIN YOUR CODE ---
    u = model(x)
    du_dx = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    d2u_dx2 = torch.autograd.grad(
        du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=True
    )[0]
    # --- END YOUR CODE ---
    # ^^^ TODO: 请删除上面的实现代码，自己重写 ^^^
    return d2u_dx2


# ============================================================================
# 任务 2: 实现复合损失函数 (约 10 行)
# ============================================================================

def compute_total_loss(model, x_f, x_b, u_b_true, lambda_bc=10.0):
    """
    计算 PINN 的复合损失: L = L_pde + lambda_bc * L_bc

    L_pde = mean(r(x_f)^2)，其中 r(x) = -u''(x) - f(x)   (PDE 残差)
    L_bc  = mean((u_hat(x_b) - u_b_true)^2)               (边界条件误差)

    参数:
        model: PINN 网络
        x_f: (N_f, 1) tensor，内部 collocation points，需要 requires_grad=True
        x_b: (N_b, 1) tensor，边界点
        u_b_true: (N_b, 1) tensor，边界点处的真实函数值
        lambda_bc: float，边界条件损失的权重
    返回:
        loss_total, loss_pde, loss_bc: 三个标量 tensor
    """
    # TODO: 完成以下步骤
    # 1. 用 autograd_second_derivative(model, x_f) 得到 d2u_dx2
    # 2. 组装残差 residual = -d2u_dx2 - f_source(x_f)
    # 3. loss_pde = mean(residual^2)
    # 4. u_b_pred = model(x_b)
    # 5. loss_bc = mean((u_b_pred - u_b_true)^2)
    # 6. loss_total = loss_pde + lambda_bc * loss_bc
    # --- BEGIN YOUR CODE ---
    d2u_dx2 = autograd_second_derivative(model, x_f)
    residual = -d2u_dx2 - f_source(x_f)
    loss_pde = torch.mean(residual ** 2)

    u_b_pred = model(x_b)
    loss_bc = torch.mean((u_b_pred - u_b_true) ** 2)

    loss_total = loss_pde + lambda_bc * loss_bc
    # --- END YOUR CODE ---
    # ^^^ TODO: 请删除上面的实现代码，自己重写 ^^^
    return loss_total, loss_pde, loss_bc


# ============================================================================
# 任务 3 (Bonus): 实现一个最小训练循环 (约 10 行)
# ============================================================================

def train_mini_pinn(n_epochs=500, lr=1e-3):
    """
    用任务1、任务2实现的工具，训练一个 TinyPINN 求解一维 Poisson 方程。

    步骤:
      1. 初始化模型和 Adam 优化器
      2. 构造 collocation points (要求 requires_grad=True) 和边界点
      3. 循环 n_epochs 次:
         - optimizer.zero_grad()
         - 用 compute_total_loss 计算损失
         - loss.backward()
         - optimizer.step()
      4. 返回训练好的模型和损失历史列表
    """
    model = TinyPINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x_f = torch.linspace(0, 1, 40).reshape(-1, 1)
    x_f.requires_grad_(True)
    x_b = torch.tensor([[0.0], [1.0]])
    u_b_true = torch.tensor([[0.0], [0.0]])

    loss_history = []
    # TODO: 完成训练循环
    # --- BEGIN YOUR CODE ---
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss_total, loss_pde, loss_bc = compute_total_loss(model, x_f, x_b, u_b_true)
        loss_total.backward()
        optimizer.step()
        loss_history.append(loss_total.item())
    # --- END YOUR CODE ---
    # ^^^ TODO: 请删除上面的实现代码，自己重写 ^^^
    return model, loss_history


# ============================================================================
# 验证代码
# ============================================================================

def test_second_derivative():
    """用 u(x)=x^2 验证二阶导数计算：解析二阶导数处处为 2。"""
    print("[测试 1] 二阶导数 (u(x)=x^2, 期望 u''=2)...")
    probe = QuadraticProbe()
    x = torch.linspace(-2, 2, 20).reshape(-1, 1)
    x.requires_grad_(True)
    d2u = autograd_second_derivative(probe, x)
    assert torch.allclose(d2u, torch.full_like(d2u, 2.0), atol=1e-4), \
        f"二阶导数不正确: {d2u[:5].flatten()}"
    print(f"  [PASS] 二阶导数计算正确! d2u/dx2 ≈ {d2u[0].item():.4f} (期望 2.0)")


def test_composite_loss():
    """验证复合损失: loss_total 应等于 loss_pde + lambda_bc * loss_bc。"""
    print("[测试 2] 复合损失函数...")
    torch.manual_seed(0)
    model = TinyPINN()
    x_f = torch.linspace(0, 1, 20).reshape(-1, 1)
    x_f.requires_grad_(True)
    x_b = torch.tensor([[0.0], [1.0]])
    u_b_true = torch.tensor([[0.0], [0.0]])

    loss_total, loss_pde, loss_bc = compute_total_loss(model, x_f, x_b, u_b_true, lambda_bc=10.0)
    expected_total = loss_pde + 10.0 * loss_bc
    assert torch.allclose(loss_total, expected_total, atol=1e-6), \
        f"复合损失公式不正确: {loss_total.item()} vs {expected_total.item()}"
    assert loss_pde.item() >= 0 and loss_bc.item() >= 0, "损失应为非负数"
    print(f"  [PASS] 复合损失正确! loss_total={loss_total.item():.4f}, "
          f"loss_pde={loss_pde.item():.4f}, loss_bc={loss_bc.item():.4f}")


def test_training_reduces_loss():
    """(Bonus) 验证训练循环能让损失显著下降。"""
    print("[测试 3 (Bonus)] 训练循环...")
    model, loss_history = train_mini_pinn(n_epochs=500, lr=1e-2)
    assert len(loss_history) == 500, f"损失历史长度不对: {len(loss_history)}"
    # 训练后损失应远小于初始损失
    assert loss_history[-1] < loss_history[0] * 0.1, \
        f"训练未能有效降低损失: 初始={loss_history[0]:.4f}, 最终={loss_history[-1]:.4f}"
    print(f"  [PASS] 训练有效! 初始损失={loss_history[0]:.4f} -> 最终损失={loss_history[-1]:.6f}")


if __name__ == "__main__":
    print("=" * 60)
    print("as02_pinn exercise.py — PINN 练习")
    print("=" * 60)

    try:
        test_second_derivative()
        test_composite_loss()
        test_training_reduces_loss()
        print("\n" + "=" * 60)
        print("所有测试通过!")
        print("=" * 60)
    except Exception as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
