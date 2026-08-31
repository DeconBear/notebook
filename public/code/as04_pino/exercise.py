# -*- coding: utf-8 -*-
"""
===============================================================================
as04_pino/code/exercise.py — PINO 动手练习
===============================================================================
本练习在「一维变系数扩散方程」上补全 PINO 的三个关键组件：

    -(k_a(x) u'(x))' = f(x),  x in [0,1],  u(0)=u(1)=0
    k_a(x) = 1 + a * sin(pi*x),  f(x) = sin(pi*x)

练习目标：
  1. 实现二阶有限体积残差：给定网格上的 u 与 k，计算 PDE 残差损失
  2. 实现 PINO 总损失：数据损失（少量标注）+ 物理残差损失（全部实例）
  3. 对比「有/无物理约束」时，外推参数 a 上的相对误差

运行方式：
  cd docs/science/pino/code
  python exercise.py

如果你正确实现了代码，你应该看到：
  - 带物理约束的模型在外推 a 上误差明显更低
  - 控制台打印「✓ PINO 物理残差实现正确」
===============================================================================
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)

N_GRID = 33
X_GRID = np.linspace(0.0, 1.0, N_GRID)
DX = X_GRID[1] - X_GRID[0]
F_FIXED = np.sin(np.pi * X_GRID)


def diffusivity_field(a, x):
    """k_a(x) = 1 + a * sin(pi*x)"""
    return 1.0 + a * np.sin(np.pi * x)


def solve_variable_coeff_poisson(a):
    """有限体积数值精确解，用作 ground truth。"""
    k = diffusivity_field(a, X_GRID)
    k_half = 0.5 * (k[:-1] + k[1:])
    n_inner = N_GRID - 2
    A = np.zeros((n_inner, n_inner))
    b = np.zeros(n_inner)
    for i in range(1, N_GRID - 1):
        idx = i - 1
        b[idx] = F_FIXED[i]
        A[idx, idx] = (k_half[i - 1] + k_half[i]) / DX ** 2
        if idx - 1 >= 0:
            A[idx, idx - 1] = -k_half[i - 1] / DX ** 2
        if idx + 1 <= n_inner - 1:
            A[idx, idx + 1] = -k_half[i] / DX ** 2
    u = np.zeros(N_GRID)
    u[1:-1] = np.linalg.solve(A, b)
    return u


# ============================================================================
# TODO 1：实现 PDE 残差损失（有限体积离散）
# ============================================================================
def pde_residual_loss(u_pred: torch.Tensor, k_field: torch.Tensor,
                      f: torch.Tensor, dx: float) -> torch.Tensor:
    """
    计算 -(k u')' - f 的均方残差。

    参数:
        u_pred:  (B, N) 预测解
        k_field: (B, N) 扩散系数场
        f:       (B, N) 源项
        dx:      网格间距

    返回:
        标量 tensor：内部点残差的均方

    提示：
      1. k_half = 0.5 * (k[:, :-1] + k[:, 1:])          # 半网格点上的 k
      2. flux_right = k_half[:, 1:]  * (u[:, 2:] - u[:, 1:-1]) / dx
      3. flux_left  = k_half[:, :-1] * (u[:, 1:-1] - u[:, :-2]) / dx
      4. residual = -(flux_right - flux_left) / dx - f[:, 1:-1]
      5. 返回 (residual ** 2).mean()
    """
    # TODO: 实现有限体积残差
    pass  # <-- 替换为你的代码


# ============================================================================
# TODO 2：实现 PINO 总损失
# ============================================================================
def pino_total_loss(u_pred: torch.Tensor, u_true: torch.Tensor,
                    k_field: torch.Tensor, f: torch.Tensor,
                    labeled_idx: torch.Tensor, dx: float,
                    lambda_pde: float = 0.1) -> torch.Tensor:
    """
    PINO 总损失 = 数据损失（只在 labeled_idx 上）+ lambda_pde * PDE 残差损失（全部样本）

    参数:
        u_pred:      (B, N) 算子网络预测
        u_true:      (B, N) 数值精确解
        k_field, f:  (B, N)
        labeled_idx: 1D LongTensor，标注样本的下标
        dx, lambda_pde: 网格间距与物理损失权重

    提示：
      loss_data = ((u_pred[labeled_idx] - u_true[labeled_idx]) ** 2).mean()
      loss_pde  = pde_residual_loss(u_pred, k_field, f, dx)
      loss_bc   = (u_pred[:, 0]**2).mean() + (u_pred[:, -1]**2).mean()
      return loss_data + lambda_pde * loss_pde + loss_bc
    """
    # TODO: 实现总损失
    pass  # <-- 替换为你的代码


# ============================================================================
# 迷你算子网络（练习用，不需要你改）
# ============================================================================
class TinyOperator(nn.Module):
    """极简 1D 算子：把 (k, x) 逐点喂进共享 MLP。"""

    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, k_field, x_coord):
        # (B, N) -> (B, N, 2) -> (B, N)
        inp = torch.stack([k_field, x_coord], dim=-1)
        return self.net(inp).squeeze(-1)


def relative_l2(pred, true):
    return float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12))


def train_and_eval(use_physics: bool, n_epochs: int = 800):
    """训练 TinyOperator，返回测试外推 a 上的平均相对误差。"""
    torch.manual_seed(SEED)
    a_train = np.linspace(1.0, 3.0, 7)
    a_test = np.array([3.5, 4.0])  # 外推

    k_train = np.stack([diffusivity_field(a, X_GRID) for a in a_train]).astype(np.float32)
    u_train = np.stack([solve_variable_coeff_poisson(a) for a in a_train]).astype(np.float32)
    f_train = np.stack([F_FIXED for _ in a_train]).astype(np.float32)

    # 只有首尾两个 a 有标签
    labeled_idx = torch.tensor([0, len(a_train) - 1], dtype=torch.long)

    k_t = torch.tensor(k_train)
    u_t = torch.tensor(u_train)
    f_t = torch.tensor(f_train)
    x_t = torch.tensor(X_GRID, dtype=torch.float32).unsqueeze(0).repeat(len(a_train), 1)

    model = TinyOperator()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    for _ in range(n_epochs):
        opt.zero_grad()
        u_pred = model(k_t, x_t)
        if use_physics:
            loss = pino_total_loss(u_pred, u_t, k_t, f_t, labeled_idx, DX)
        else:
            # 纯数据驱动：只用标注点
            loss = ((u_pred[labeled_idx] - u_t[labeled_idx]) ** 2).mean()
            loss = loss + (u_pred[:, 0] ** 2).mean() + (u_pred[:, -1] ** 2).mean()
        if loss is None:
            raise RuntimeError('TODO 尚未实现：pino_total_loss / pde_residual_loss 返回了 None')
        loss.backward()
        opt.step()

    # 外推评估
    errs = []
    with torch.no_grad():
        for a in a_test:
            k = torch.tensor(diffusivity_field(a, X_GRID), dtype=torch.float32).unsqueeze(0)
            x = torch.tensor(X_GRID, dtype=torch.float32).unsqueeze(0)
            pred = model(k, x).numpy().flatten()
            true = solve_variable_coeff_poisson(a)
            errs.append(relative_l2(pred, true))
    return float(np.mean(errs)), errs


def check_residual_impl():
    """用数值精确解验证残差接近 0。"""
    a = 2.0
    u = torch.tensor(solve_variable_coeff_poisson(a), dtype=torch.float32).unsqueeze(0)
    k = torch.tensor(diffusivity_field(a, X_GRID), dtype=torch.float32).unsqueeze(0)
    f = torch.tensor(F_FIXED, dtype=torch.float32).unsqueeze(0)
    loss = pde_residual_loss(u, k, f, DX)
    if loss is None:
        return False, None
    return float(loss.item()) < 1e-8, float(loss.item())


def main():
    print('=' * 60)
    print('as04 PINO 练习 — 请完成 TODO 1 / TODO 2')
    print('=' * 60)

    print('\n[1] 检查 PDE 残差实现...')
    ok, val = check_residual_impl()
    if not ok:
        print(f'  ✗ 残差未通过（当前值={val}）。请检查 TODO 1。')
        print('  提示：数值精确解代入离散残差后应接近 0（<1e-8）。')
        return
    print(f'  ✓ 残差检查通过（残差={val:.2e}）')

    print('\n[2] 训练对比：纯数据 vs PINO...')
    err_data, errs_data = train_and_eval(use_physics=False)
    err_pino, errs_pino = train_and_eval(use_physics=True)
    print(f'  纯数据（2 标注）外推平均相对误差: {err_data:.4f}  明细={errs_data}')
    print(f'  PINO  （2 标注+物理）外推平均相对误差: {err_pino:.4f}  明细={errs_pino}')

    if err_pino < err_data * 0.85:
        print('\n✓ PINO 物理残差实现正确：外推误差明显低于纯数据基线。')
    else:
        print('\n⚠ 外推优势不明显。请检查 TODO 2 是否把 PDE 损失加进了总损失。')

    # 可视化
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(['纯数据 FNO-like', 'PINO'], [err_data, err_pino],
           color=['#8E44AD', '#27AE60'])
    ax.set_ylabel('外推平均相对 L2 误差')
    ax.set_title('练习结果：物理约束对外推的帮助')
    ax.grid(alpha=0.3, axis='y')
    out = os.path.join(_IMAGES_DIR, 'exercise_pino_extrapolation.png')
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'\n图片已保存: {out}')
    print('=' * 60)


if __name__ == '__main__':
    main()
