# -*- coding: utf-8 -*-
"""
s08 优化器：从 SGD 到 Adam — 练习代码
======================================
请完成以下 TODO 任务，加深对优化器内部机制的理解。

每个 TODO 都有详细的中文指示和预期输出描述。
建议先阅读 README.md 并运行 demo.py，再尝试独立补全代码。
"""

import numpy as np
from typing import Tuple, List


# ============================================================================
# 辅助：损失地形（与 demo.py 一致）
# ============================================================================

class LossLandscape:
    """二维二次型损失函数 L(θ₁, θ₂) = 0.5·(a·θ₁² + b·θ₂²)"""

    def __init__(self, a: float = 20.0, b: float = 1.0):
        self.a = a
        self.b = b

    def __call__(self, theta: np.ndarray) -> float:
        return 0.5 * (self.a * theta[0] ** 2 + self.b * theta[1] ** 2)

    def gradient(self, theta: np.ndarray) -> np.ndarray:
        return np.array([self.a * theta[0], self.b * theta[1]])

    @property
    def optimum(self) -> np.ndarray:
        return np.array([0.0, 0.0])


# ============================================================================
# TODO 1: 实现 Momentum 更新规则
# ============================================================================

class MomentumOptimizerExercise:
    """
    Momentum 优化器（练习版本）。

    公式:
        m_t = β · m_{t-1} + (1-β) · g_t       ← 速度更新（指数滑动平均）
        θ_{t+1} = θ_t - α · m_t                ← 参数更新

    你需要实现 step() 方法。
    """

    def __init__(self, lr: float = 0.02, beta: float = 0.9):
        """
        初始化 Momentum 优化器。

        参数:
            lr: 学习率 α
            beta: 动量衰减系数 β
        """
        self.lr = lr
        self.beta = beta
        self.m = None  # 速度向量，首次 step 时初始化为零
        self.name = "Momentum"

    def step(self, theta: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """
        执行一步 Momentum 更新。

        提示:
          1. 如果 self.m 是 None，初始化为 np.zeros_like(theta)
          2. 更新速度: self.m = β * self.m + (1 - β) * grad
          3. 更新参数: theta = theta - α * self.m
          4. 返回更新后的 theta

        参数:
            theta: 当前参数向量
            grad: 当前梯度向量

        返回:
            更新后的参数向量
        """
        # TODO: 如果 self.m 为 None，初始化为零向量
        if self.m is None:
            pass  # ← TODO: self.m = np.zeros_like(theta)

        # TODO: 更新速度 m_t
        pass  # ← TODO: self.m = self.beta * self.m + (1 - self.beta) * grad

        # TODO: 更新参数
        pass  # ← TODO: return theta - self.lr * self.m

        return theta  # 占位返回


# ---- 测试 TODO 1 ----
def test_momentum():
    """测试 Momentum 优化器的实现"""
    print("=" * 60)
    print("TODO 1 测试: Momentum 更新规则")
    print("=" * 60)

    landscape = LossLandscape(a=20.0, b=1.0)
    theta0 = np.array([3.0, 2.5])
    opt = MomentumOptimizerExercise(lr=0.02, beta=0.9)

    # 手动运行几步
    theta = theta0.copy()
    print(f"\n  初始: θ = {theta}, L = {landscape(theta):.4f}")

    for step in range(5):
        grad = landscape.gradient(theta)
        theta_new = opt.step(theta, grad)

        if theta_new is None or np.allclose(theta_new, theta):
            print("  TODO 未完成，请补全 MomentumOptimizerExercise.step()")
            return

        theta = theta_new
        if step < 5:
            print(f"  Step {step+1}: θ = [{theta[0]:.3f}, {theta[1]:.3f}], "
                  f"L = {landscape(theta):.4f}, m = [{opt.m[0]:.3f}, {opt.m[1]:.3f}]")

    # 验证动量向量的方向大致正确（应该指向原点附近）
    if opt.m is not None:
        m_direction = opt.m / (np.linalg.norm(opt.m) + 1e-10)
        target_direction = -theta0 / (np.linalg.norm(theta0) + 1e-10)
        similarity = np.dot(m_direction, target_direction)
        print(f"\n  动量方向与目标方向的相似度: {similarity:.3f} (>0 表示大致正确)")

    print()


# ============================================================================
# TODO 2: 实现 RMSProp 更新规则（v_t 计算）
# ============================================================================

class RMSPropOptimizerExercise:
    """
    RMSProp 优化器（练习版本）。

    公式:
        v_t = β · v_{t-1} + (1-β) · g_t²       ← 梯度平方的指数滑动平均
        θ_{t+1} = θ_t - α · g_t / (√v_t + ε)    ← 自适应步长更新

    你需要实现 step() 方法。
    特别注意 v_t 的计算——g_t 要先逐元素平方。
    """

    def __init__(self, lr: float = 0.05, beta: float = 0.9, eps: float = 1e-8):
        """
        初始化 RMSProp 优化器。

        参数:
            lr: 学习率 α
            beta: 衰减系数 β
            eps: 数值稳定常数 ε
        """
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.v = None
        self.name = "RMSProp"

    def step(self, theta: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """
        执行一步 RMSProp 更新。

        提示:
          1. 如果 self.v 是 None，初始化为 np.zeros_like(theta)
          2. 更新 v_t: self.v = β * self.v + (1 - β) * (grad ** 2)
             注意：grad ** 2 是逐元素平方
          3. 更新参数: theta = theta - α * grad / (√v_t + ε)
             注意：np.sqrt(self.v) 也是逐元素操作
          4. 返回更新后的 theta

        参数:
            theta: 当前参数向量
            grad: 当前梯度向量

        返回:
            更新后的参数向量
        """
        # TODO: 如果 self.v 为 None，初始化为零向量
        if self.v is None:
            pass  # ← TODO: self.v = np.zeros_like(theta)

        # TODO: 更新梯度平方的滑动平均 v_t
        pass  # ← TODO: self.v = self.beta * self.v + (1 - self.beta) * (grad ** 2)

        # TODO: 自适应步长更新
        pass  # ← TODO: return theta - self.lr * grad / (np.sqrt(self.v) + self.eps)

        return theta  # 占位返回


# ---- 测试 TODO 2 ----
def test_rmsprop():
    """测试 RMSProp 优化器的实现"""
    print("=" * 60)
    print("TODO 2 测试: RMSProp 自适应步长")
    print("=" * 60)

    landscape = LossLandscape(a=20.0, b=1.0)
    theta0 = np.array([3.0, 2.5])
    opt = RMSPropOptimizerExercise(lr=0.05, beta=0.9)

    theta = theta0.copy()
    print(f"\n  初始: θ = {theta}, L = {landscape(theta):.4f}")

    for step in range(10):
        grad = landscape.gradient(theta)
        theta_new = opt.step(theta, grad)

        if theta_new is None or np.allclose(theta_new, theta):
            print("  TODO 未完成，请补全 RMSPropOptimizerExercise.step()")
            return

        theta = theta_new
        if step < 5 or step == 9:
            effective_lr = opt.lr / (np.sqrt(opt.v) + opt.eps)
            print(f"  Step {step+1:2d}: θ = [{theta[0]:.3f}, {theta[1]:.3f}], "
                  f"L = {landscape(theta):.4f}")
            print(f"         有效学习率: [{effective_lr[0]:.4f}, {effective_lr[1]:.4f}]")

    # 验证：陡峭方向(θ₁)的有效学习率应该小于平缓方向(θ₂)
    if opt.v is not None:
        effective_lr = opt.lr / (np.sqrt(opt.v) + opt.eps)
        print(f"\n  最终有效学习率: θ₁={effective_lr[0]:.5f}, θ₂={effective_lr[1]:.5f}")
        print(f"  θ₁ < θ₂ 吗？{effective_lr[0] < effective_lr[1]} (RMSProp 应该压小陡峭方向的步长)")

    print()


# ============================================================================
# TODO 3: 实现 Nesterov 加速梯度 (NAG)
# ============================================================================

class NAGOptimizer:
    """
    Nesterov 加速梯度 (Nesterov Accelerated Gradient, NAG)。

    NAG 和普通 Momentum 的关键区别：
      - Momentum: 在当前点计算梯度，然后沿动量方向更新
      - NAG:     先沿动量方向"前瞻"一步，在"前瞻位置"计算梯度，再更新

    公式:
        θ_lookahead = θ_t - α · β · m_{t-1}            ← 沿当前动量方向看一步
        m_t = β · m_{t-1} + (1-β) · ∇L(θ_lookahead)   ← 在"前瞻"位置计算梯度
        θ_{t+1} = θ_t - α · m_t                        ← 用前瞻梯度更新

    直观上，NAG 像是"先试着往前走一步，发现不对再调整方向"。

    你需要实现 step() 方法。

    参数:
        lr: 学习率 α
        beta: 动量衰减系数 β
    """

    def __init__(self, lr: float = 0.02, beta: float = 0.9):
        self.lr = lr
        self.beta = beta
        self.m = None
        self.name = "NAG"

    def step(self, theta: np.ndarray, grad_fn) -> np.ndarray:
        """
        执行一步 NAG 更新。

        注意：与普通优化器不同，NAG 需要 grad_fn（梯度计算函数），
        因为需要在"前瞻"位置重新计算梯度。

        提示:
          1. 如果 self.m 为 None，初始化为零向量
          2. 计算前瞻位置: theta_lookahead = theta - lr * beta * m
          3. 在前瞻位置计算梯度: grad_lookahead = grad_fn(theta_lookahead)
          4. 更新动量: m = beta * m + (1 - beta) * grad_lookahead
          5. 更新参数: theta = theta - lr * m

        参数:
            theta: 当前参数向量
            grad_fn: 梯度计算函数，签名为 grad_fn(theta) -> gradient

        返回:
            更新后的参数向量
        """
        # TODO: 初始化动量
        if self.m is None:
            pass  # ← TODO: self.m = np.zeros_like(theta)

        # TODO: 计算前瞻位置
        theta_lookahead = None  # ← TODO: theta - self.lr * self.beta * self.m

        # TODO: 在前瞻位置计算梯度
        grad_lookahead = None  # ← TODO: grad_fn(theta_lookahead)

        # TODO: 更新动量
        pass  # ← TODO: self.m = self.beta * self.m + (1 - self.beta) * grad_lookahead

        # TODO: 更新参数
        pass  # ← TODO: return theta - self.lr * self.m

        return theta  # 占位返回


# ---- 测试 TODO 3 ----
def test_nag():
    """测试 NAG 优化器"""
    print("=" * 60)
    print("TODO 3 测试: Nesterov 加速梯度 (NAG)")
    print("=" * 60)

    landscape = LossLandscape(a=20.0, b=1.0)
    theta0 = np.array([3.0, 2.5])

    # NAG
    nag = NAGOptimizer(lr=0.02, beta=0.9)
    theta_nag = theta0.copy()

    # 普通 Momentum 用于对比
    from types import SimpleNamespace
    mom = SimpleNamespace()
    mom.lr = 0.02
    mom.beta = 0.9
    mom.m = np.zeros_like(theta0)
    theta_mom = theta0.copy()

    print(f"\n  对比 NAG vs Momentum (前10步)")
    print(f"  {'Step':<6} {'NAG L':<14} {'NAG θ₁':<12} {'Mom L':<14} {'Mom θ₁'}")
    print(f"  {'-'*60}")

    for step in range(10):
        # NAG 更新
        grad_fn = landscape.gradient
        theta_nag_new = nag.step(theta_nag, grad_fn)

        if theta_nag_new is None or np.allclose(theta_nag_new, theta_nag):
            print("  TODO 未完成，请补全 NAGOptimizer.step()")
            return

        theta_nag = theta_nag_new

        # 普通 Momentum 更新
        grad_mom = landscape.gradient(theta_mom)
        mom.m = mom.beta * mom.m + (1 - mom.beta) * grad_mom
        theta_mom = theta_mom - mom.lr * mom.m

        print(f"  {step+1:<6} {landscape(theta_nag):<14.6f} {theta_nag[0]:<12.4f} "
              f"{landscape(theta_mom):<14.6f} {theta_mom[0]:.4f}")

    # NAG 通常比 Momentum 收敛更快
    loss_nag = landscape(theta_nag)
    loss_mom = landscape(theta_mom)
    print(f"\n  最终: NAG loss={loss_nag:.6f}, Momentum loss={loss_mom:.6f}")
    print(f"  NAG 是否更快: {'是' if loss_nag < loss_mom else '否'} (预期: 是)")

    print()


# ============================================================================
# 主程序
# ============================================================================
if __name__ == "__main__":
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║   s08 优化器：从 SGD 到 Adam — 动手练习                    ║")
    print("║   请依次完成 TODO 1, 2, 3                                   ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    test_momentum()
    test_rmsprop()
    test_nag()

    print("=" * 60)
    print("所有测试完成！请检查输出结果。")
    print("如有未通过的测试，请回到对应的 TODO 部分补全代码。")
    print()
    print("优化器设计演进:")
    print("  SGD → 只看当前梯度，简单但有震荡、慢缩等问题")
    print("  Momentum → 加惯性，方向更平滑")
    print("  RMSProp → 自适应步长，每个参数有自己的学习率")
    print("  Adam → Momentum + RMSProp，集两者之长")
    print("  NAG → 比 Momentum 更进一步：先"看一步"再调整方向")
    print("=" * 60)
