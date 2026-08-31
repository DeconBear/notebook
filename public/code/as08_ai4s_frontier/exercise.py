# -*- coding: utf-8 -*-
"""
===============================================================================
as08_ai4s_frontier/code/exercise.py — AI4S 综合练习：可微逆向设计
===============================================================================
本练习聚焦"可微代理模型驱动的逆向设计"这个 AI4S 前沿主题的核心计算：
  任务1：实现黑盒网格搜索 grid_search_1d
  任务2：实现梯度下降逆向设计 gradient_search_1d
  任务3（Bonus）：在一个更难的目标（训练范围外的外推目标）上对比两种方法的鲁棒性

运行方式：python exercise.py
===============================================================================
"""

import numpy as np
import torch
import torch.nn as nn

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

N_GRID = 21
X_GRID = np.linspace(0.0, 1.0, N_GRID)


def toy_forward_true(a):
    """
    一个已知解析形式的"正向物理模型"（简化版，用于快速验证，不依赖数值求解器）：
        u(x; a) = a * sin(pi*x) + 0.1 * a^2 * sin(2*pi*x)
    这个函数本身是非线性的（含 a^2 项），逆向设计不能简单地"线性反解"。
    """
    x = X_GRID
    return a * np.sin(np.pi * x) + 0.1 * (a ** 2) * np.sin(2 * np.pi * x)


class ToySurrogate(nn.Module):
    """一个小型代理模型，学习拟合 toy_forward_true（已提供训练好的等价实现，也可以自己重新训练）。"""

    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, N_GRID),
        )

    def forward(self, a):
        return self.net(a)


def train_toy_surrogate(n_epochs=1500, lr=3e-3):
    a_train = np.linspace(-3.0, 3.0, 40)
    u_train = np.stack([toy_forward_true(a) for a in a_train]).astype(np.float32)
    a_t = torch.tensor(a_train, dtype=torch.float32).view(-1, 1)
    u_t = torch.tensor(u_train, dtype=torch.float32)

    model = ToySurrogate()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        pred = model(a_t)
        loss = torch.mean((pred - u_t) ** 2)
        loss.backward()
        optimizer.step()
    return model


# ============================================================================
# 任务 1: 实现黑盒网格搜索 (约 6 行核心代码)
# ============================================================================

def grid_search_1d(model, u_target_t, a_min=-3.0, a_max=3.0, n_grid=100):
    """
    在 [a_min, a_max] 区间均匀采样 n_grid 个候选值，对每个候选值用 model
    做一次前向评估，返回使 MSE 最小的候选值。

    参数:
        model: 代理模型，model(a_tensor) -> (1, N_GRID) 预测曲线
        u_target_t: (N_GRID,) 目标曲线 tensor
        a_min, a_max, n_grid: 搜索区间与采样点数
    返回:
        best_a: float，MSE 最小的候选值
        best_err: float，对应的最小 MSE
        n_evals: int，总共评估了多少次（= n_grid）

    实现步骤:
      1. a_candidates = np.linspace(a_min, a_max, n_grid)
      2. 对每个候选 a（在 torch.no_grad() 下）：
         pred = model(torch.tensor([[a]], dtype=torch.float32))
         err = torch.mean((pred[0] - u_target_t) ** 2).item()
      3. 记录并返回误差最小的 (a, err)
    """
    a_candidates = np.linspace(a_min, a_max, n_grid)
    best_a, best_err = None, np.inf
    # TODO: 完成网格搜索循环
    # --- BEGIN YOUR CODE ---
    pass
    # --- END YOUR CODE ---
    return best_a, best_err, n_grid


# ============================================================================
# 任务 2: 实现梯度下降逆向设计 (约 8 行核心代码)
# ============================================================================

def gradient_search_1d(model, u_target_t, a_init=0.5, n_steps=80, lr=0.1):
    """
    把 a 设置为一个 requires_grad=True 的可训练标量，直接对
    "预测曲线与目标曲线的 MSE" 做梯度下降，寻找最优 a。

    实现步骤:
      1. a = torch.tensor([[a_init]], dtype=torch.float32, requires_grad=True)
      2. optimizer = torch.optim.Adam([a], lr=lr)
      3. 循环 n_steps 次:
         optimizer.zero_grad()
         pred = model(a)
         loss = torch.mean((pred[0] - u_target_t) ** 2)
         loss.backward()
         optimizer.step()
      4. 返回 a.item()、最后一步的 loss.item()、n_steps
    """
    # TODO: 完成梯度下降逆向设计
    # --- BEGIN YOUR CODE ---
    final_a, final_err = None, None
    # --- END YOUR CODE ---
    return final_a, final_err, n_steps


# ============================================================================
# 验证代码
# ============================================================================

def test_grid_search():
    print("[测试 1] 黑盒网格搜索...")
    model = train_toy_surrogate()
    a_true = 1.5
    u_target = torch.tensor(toy_forward_true(a_true), dtype=torch.float32)
    best_a, best_err, n_evals = grid_search_1d(model, u_target, a_min=-3.0, a_max=3.0, n_grid=200)
    assert best_a is not None, "grid_search_1d 未正确实现，请完成 TODO"
    assert abs(best_a - a_true) < 0.15, f"网格搜索结果偏差过大: 找到a={best_a}, 真实a={a_true}"
    print(f"  [PASS] 真实 a={a_true}, 网格搜索找到 a={best_a:.4f} (评估{n_evals}次)")


def test_gradient_search():
    print("[测试 2] 梯度下降逆向设计...")
    model = train_toy_surrogate()
    a_true = 1.5
    u_target = torch.tensor(toy_forward_true(a_true), dtype=torch.float32)
    final_a, final_err, n_steps = gradient_search_1d(model, u_target, a_init=0.2, n_steps=80)
    assert final_a is not None, "gradient_search_1d 未正确实现，请完成 TODO"
    assert abs(final_a - a_true) < 0.15, f"梯度搜索结果偏差过大: 找到a={final_a}, 真实a={a_true}"
    print(f"  [PASS] 真实 a={a_true}, 梯度下降找到 a={final_a:.4f} (迭代{n_steps}次)")


def test_extrapolation_robustness():
    print("[测试 3 (Bonus)] 训练范围外的目标（外推场景）鲁棒性对比...")
    model = train_toy_surrogate()
    a_true_extreme = 4.5  # 超出训练范围 [-3, 3]
    u_target = torch.tensor(toy_forward_true(a_true_extreme), dtype=torch.float32)

    best_a_grid, err_grid, _ = grid_search_1d(model, u_target, a_min=-3.0, a_max=3.0, n_grid=200)
    final_a_grad, err_grad, _ = gradient_search_1d(model, u_target, a_init=2.5, n_steps=100, lr=0.05)

    if best_a_grid is None or final_a_grad is None:
        print("  [SKIP] 依赖的 TODO 尚未完成。")
        return
    print(f"  外推目标 a_true={a_true_extreme}（超出训练范围[-3,3]）")
    print(f"  网格搜索(限制在训练范围内)找到 a={best_a_grid:.4f}, MSE={err_grid:.6f}")
    print(f"  梯度下降(允许超出训练范围)找到 a={final_a_grad:.4f}, MSE={err_grad:.6f}")
    print("  观察: 网格搜索被人为限制在训练范围内，无法找到真实的外推参数；")
    print("  梯度下降没有范围限制，理论上可以走到训练范围外，但代理模型本身在")
    print("  外推区域可能不准确——这提醒我们：逆向设计的可靠性上限取决于代理模型")
    print("  本身的泛化能力，梯度方法只是让搜索过程更高效，不能创造模型没学到的知识。")


if __name__ == "__main__":
    print("=" * 60)
    print("as08_ai4s_frontier exercise.py — 可微逆向设计练习")
    print("=" * 60)
    try:
        test_grid_search()
        test_gradient_search()
        test_extrapolation_robustness()
        print("\n" + "=" * 60)
        print("所有测试通过!")
        print("=" * 60)
    except Exception as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
