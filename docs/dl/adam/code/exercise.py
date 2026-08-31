# -*- coding: utf-8 -*-
"""
s09 Adam 深度解析与训练实战 — 练习代码
======================================
请完成以下 TODO 任务，加深对 Adam 内部机制和训练实践的理解。

每个 TODO 都有详细的中文指示和预期输出描述。
建议先阅读 README.md 并运行 demo.py，再尝试独立补全代码。
"""

import numpy as np
from typing import Dict, Tuple, Optional


# ============================================================================
# TODO 1: 实现偏差修正
# ============================================================================

class AdamBiasCorrectionExercise:
    """
    Adam 优化器（练习版本——需要补全偏差修正部分）。

    当前实现缺少偏差修正，导致训练初期收敛缓慢。
    你的任务是补全偏差修正的逻辑。
    """

    def __init__(self, lr: float = 0.001,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m: Dict[int, np.ndarray] = {}
        self.v: Dict[int, np.ndarray] = {}
        self.t = 0

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        """
        执行一步 Adam 更新。

        你需要在此方法中补全偏差修正：
          m̂_t = m_t / (1 - β₁^t)
          v̂_t = v_t / (1 - β₂^t)

        并在参数更新中使用修正后的 m̂_t 和 v̂_t。

        提示:
          1. 计算 1 - beta1**self.t 和 1 - beta2**self.t
          2. 将 m[param_id] 和 v[param_id] 分别除以这两个值
          3. 用修正后的值进行参数更新
        """
        self.t += 1  # 迭代步数 +1

        for key in params:
            param = params[key]
            grad = grads.get(key)
            if grad is None:
                continue

            param_id = id(param)
            if param_id not in self.m:
                self.m[param_id] = np.zeros_like(param)  # 初始化一阶矩
                self.v[param_id] = np.zeros_like(param)  # 初始化二阶矩

            # ---- 更新一阶矩和二阶矩（这部分已写好） ----
            self.m[param_id] = (self.beta1 * self.m[param_id]
                                + (1 - self.beta1) * grad)
            self.v[param_id] = (self.beta2 * self.v[param_id]
                                + (1 - self.beta2) * (grad ** 2))

            # TODO: 计算偏差修正
            # 提示: m_hat = self.m[param_id] / (1 - self.beta1 ** self.t)
            m_hat = None  # ← TODO: 计算修正后的一阶矩
            v_hat = None  # ← TODO: 计算修正后的二阶矩

            # TODO: 用修正后的值更新参数
            # 提示: param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            pass  # ← TODO: 实现参数更新

    def get_correction_factors(self) -> Tuple[float, float]:
        """
        返回当前的偏差修正因子。

        返回:
            (1-β₁^t, 1-β₂^t): 修正分母，越接近 1 说明偏差越小
        """
        # TODO: 计算并返回修正因子
        m_factor = None  # ← TODO: 1 - self.beta1 ** self.t
        v_factor = None  # ← TODO: 1 - self.beta2 ** self.t
        return m_factor, v_factor


# ---- 测试 TODO 1 ----
def test_bias_correction():
    """测试偏差修正的实现"""
    print("=" * 60)
    print("TODO 1 测试: Adam 偏差修正")
    print("=" * 60)

    # 构造一个简单的优化任务：f(θ) = (θ - 3)²，最小值在 θ=3
    theta = {"w": np.array([0.0])}  # 初始值 0

    # 用偏差修正训练几步
    opt = AdamBiasCorrectionExercise(lr=0.1)

    print("\n  前10步的偏差修正因子变化:")
    print(f"  {'Step':<8} {'1-β₁^t':<14} {'1-β₂^t':<14} {'修正幅度'}")
    print(f"  {'-'*44}")

    for step in range(10):
        # 计算梯度: ∂f/∂θ = 2(θ-3)
        grad = {"w": 2 * (theta["w"] - 3.0)}

        # 获取修正因子（在 step 之前）
        m_factor, v_factor = opt.get_correction_factors()

        if m_factor is not None and v_factor is not None:
            print(f"  {step+1:<8} {m_factor:<14.6f} {v_factor:<14.6f} "
                  f"{'大幅修正' if step < 3 else '小幅修正' if step < 7 else '接近无修正'}")
        else:
            print("  TODO 未完成，请补全 get_correction_factors 方法")
            break

        opt.step(theta, grad)

    print(f"\n  训练 10 步后: θ = {theta['w'][0]:.4f} (目标: 3.0)")
    print(f"  误差: {abs(theta['w'][0] - 3.0):.4f}")

    # 对比无修正版本
    theta_no_bc = {"w": np.array([0.0])}
    opt_no_bc = AdamBiasCorrectionExercise(lr=0.1)
    opt_no_bc.t = 1000  # 欺骗：假装已经训练了很久，让修正因子接近 1
    opt_no_bc.m = {id(theta_no_bc["w"]): np.array([0.0])}
    opt_no_bc.v = {id(theta_no_bc["w"]): np.array([0.0])}

    for step in range(10):
        grad = {"w": 2 * (theta_no_bc["w"] - 3.0)}
        opt_no_bc.step(theta_no_bc, grad)

    print(f"  无修正 10 步后: θ = {theta_no_bc['w'][0]:.4f}")
    print(f"  有修正 vs 无修正: {theta['w'][0]:.4f} vs {theta_no_bc['w'][0]:.4f}")
    print()


# ============================================================================
# TODO 2: 实现 AdamW 的解耦权重衰减
# ============================================================================

class AdamWExercise:
    """
    AdamW 优化器（练习版本——需要补全解耦的权重衰减）。

    AdamW 的关键区别：
      - Adam+L2: 梯度 = g + λθ, 然后被 √v̂ 缩放 → 衰减效果不均匀
      - AdamW:   先做 Adam 自适应更新，再独立应用权重衰减 → 衰减效果均匀

    公式:
      θ = θ - α·m̂/(√v̂+ε) - α·λ·θ
          \_______________/   \_____/
            Adam 自适应更新    独立权重衰减
    """

    def __init__(self, lr: float = 0.001, weight_decay: float = 0.01,
                 betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        self.lr = lr
        self.weight_decay = weight_decay  # λ
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m: Dict[int, np.ndarray] = {}
        self.v: Dict[int, np.ndarray] = {}
        self.t = 0

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        """
        执行一步 AdamW 更新。

        你需要实现解耦的权重衰减：
          1. 先做正常的 Adam 自适应更新
          2. 再独立地应用权重衰减: param -= lr * weight_decay * param

        提示:
          - Adam 部分（m_t, v_t, 偏差修正, 自适应更新）照常实现
          - 在自适应更新之后，再加一行: param -= self.lr * self.weight_decay * param
          - 注意权重衰减与 m̂, v̂ 完全无关
        """
        self.t += 1

        for key in params:
            param = params[key]
            grad = grads.get(key)
            if grad is None:
                continue

            param_id = id(param)
            if param_id not in self.m:
                self.m[param_id] = np.zeros_like(param)
                self.v[param_id] = np.zeros_like(param)

            # 更新一阶矩
            self.m[param_id] = (self.beta1 * self.m[param_id]
                                + (1 - self.beta1) * grad)
            # 更新二阶矩
            self.v[param_id] = (self.beta2 * self.v[param_id]
                                + (1 - self.beta2) * (grad ** 2))

            # 偏差修正
            m_hat = self.m[param_id] / (1 - self.beta1 ** self.t)
            v_hat = self.v[param_id] / (1 - self.beta2 ** self.t)

            # TODO: Adam 自适应更新
            pass  # ← TODO: param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            # TODO: 解耦的权重衰减（AdamW 的关键步骤！）
            pass  # ← TODO: param -= self.lr * self.weight_decay * param


# ---- 测试 TODO 2 ----
def test_adamw():
    """测试 AdamW 权重衰减的效果"""
    print("=" * 60)
    print("TODO 2 测试: AdamW 解耦权重衰减")
    print("=" * 60)

    # 对比实验：Adam (无衰减) vs AdamW (有衰减)
    # 目标函数 f(w) = (w-5)²，希望 w 接近 5
    # 权重衰减会把 w 往 0 拉，因此最终值会略小于 5

    params_adam = {"w": np.array([0.0])}
    params_adamw = {"w": np.array([0.0])}

    # Adam (无权重衰减)
    from types import SimpleNamespace
    adam = SimpleNamespace()
    adam.lr = 0.1
    adam.beta1, adam.beta2 = 0.9, 0.999
    adam.eps = 1e-8
    adam.m = {id(params_adam["w"]): np.array([0.0])}
    adam.v = {id(params_adam["w"]): np.array([0.0])}
    adam.t = 0

    # AdamW (weight_decay=0.1)
    adamw = AdamWExercise(lr=0.1, weight_decay=0.1)

    print("\n  对比 Adam vs AdamW (目标值=5.0, 权重衰减=0.1)")
    print(f"  {'Step':<6} {'Adam w':<14} {'AdamW w':<14}")

    for step in range(30):
        # Adam 更新
        grad_adam = 2 * (params_adam["w"] - 5.0)
        adam.t += 1
        adam.m[id(params_adam["w"])] = (adam.beta1 * adam.m[id(params_adam["w"])]
                                        + (1 - adam.beta1) * grad_adam)
        adam.v[id(params_adam["w"])] = (adam.beta2 * adam.v[id(params_adam["w"])]
                                        + (1 - adam.beta2) * (grad_adam ** 2))
        m_hat_a = adam.m[id(params_adam["w"])] / (1 - adam.beta1 ** adam.t)
        v_hat_a = adam.v[id(params_adam["w"])] / (1 - adam.beta2 ** adam.t)
        params_adam["w"] -= adam.lr * m_hat_a / (np.sqrt(v_hat_a) + adam.eps)

        # AdamW 更新
        grad_adamw = {"w": 2 * (params_adamw["w"] - 5.0)}
        adamw.step(params_adamw, grad_adamw)

        if step % 10 == 0 or step == 29:
            print(f"  {step+1:<6} {params_adam['w'][0]:<14.4f} "
                  f"{params_adamw['w'][0]:<14.4f}")

    print(f"\n  最终: Adam w = {params_adam['w'][0]:.4f} (接近 5.0)")
    print(f"         AdamW w = {params_adamw['w'][0]:.4f} (被权重衰减拉向 0)")

    # 验证：AdamW 的值应该小于 Adam
    if params_adamw["w"][0] < params_adam["w"][0]:
        print("  ✓ 权重衰减生效：AdamW 的值 < Adam 的值")
    else:
        print("  ⚠ TODO 可能不完整，请检查 AdamW step() 实现")

    print()


# ============================================================================
# TODO 3: 实现学习率 warmup 调度
# ============================================================================

class WarmupSchedulerExercise:
    """
    学习率 Warmup 调度器（练习版本）。

    Warmup 公式（线性增加）:
        α_t = α_target · t / t_warmup,  对 t ≤ t_warmup
        α_t = α_target,                  对 t > t_warmup

    为什么要 warmup？
      - 模型参数刚开始是随机的，梯度估计不可靠
      - Adam 的 m_t 和 v_t 从零开始，需要时间积累
      - 没有 warmup 可能导致训练初期的 loss 爆炸

    在 Transformer 训练中，warmup 几乎是必须的。
    """

    def __init__(self, optimizer, warmup_steps: int, target_lr: float):
        """
        初始化 warmup 调度器。

        参数:
            optimizer: 优化器对象（需要有 .lr 属性）
            warmup_steps: warmup 步数
            target_lr: 目标学习率（warmup 结束时的值）
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.target_lr = target_lr
        self.initial_lr = 0.0  # 从 0 开始
        self.current_step = 0

    def step(self):
        """
        更新学习率。

        TODO: 实现线性 warmup
          1. current_step += 1
          2. 如果 current_step <= warmup_steps:
               lr = target_lr * current_step / warmup_steps  (线性增加)
          3. 如果 current_step > warmup_steps:
               lr = target_lr  (保持不变——这里只实现 warmup，不包含 decay)

        提示:
          - 线性插值: progress = current_step / warmup_steps
          - 更新: optimizer.lr = initial_lr + (target_lr - initial_lr) * progress
          - 确保 step > warmup_steps 时 lr = target_lr
        """
        # TODO: 步数 +1
        pass  # ← TODO: self.current_step += 1

        # TODO: 计算当前学习率
        # 提示: if self.current_step <= self.warmup_steps:
        #           progress = self.current_step / self.warmup_steps
        #           self.optimizer.lr = self.target_lr * progress
        #       else:
        #           self.optimizer.lr = self.target_lr
        pass  # ← TODO: 实现

    def get_lr(self) -> float:
        """返回当前学习率"""
        return self.optimizer.lr


# ---- 测试 TODO 3 ----
def test_warmup():
    """测试 warmup 调度器"""
    print("=" * 60)
    print("TODO 3 测试: 学习率 Warmup")
    print("=" * 60)

    # 创建一个模拟优化器
    class MockOptimizer:
        def __init__(self):
            self.lr = 0.0

    opt = MockOptimizer()
    target_lr = 0.01
    warmup_steps = 10
    scheduler = WarmupSchedulerExercise(opt, warmup_steps, target_lr)

    print(f"\n  Warmup 步骤: {warmup_steps}, 目标 lr: {target_lr}")
    print(f"  {'Step':<8} {'学习率':<16}")
    print(f"  {'-'*24}")

    for step in range(15):
        scheduler.step()
        lr = scheduler.get_lr()
        if lr is not None and lr > 0:
            print(f"  {step+1:<8} {lr:<16.6f}")
        else:
            if step == 0:
                print(f"  {step+1:<8} TODO 未完成 (lr={lr})")

    # 验证 warmup 后的学习率
    if opt.lr is not None and opt.lr > 0:
        print(f"\n  最终学习率: {opt.lr:.6f}")
        print(f"  是否达到目标: {abs(opt.lr - target_lr) < 1e-6}")
        if opt.lr == target_lr / warmup_steps:
            print("  注意：学习率只增加了一步？请检查 step() 中的逻辑。")

    print()


# ============================================================================
# TODO 4: 诊断——调试 NaN loss
# ============================================================================

def debug_nan_loss():
    """
    场景：你的训练 loss 突然变成了 NaN。以下是可能的原因和排查方法。

    任务：阅读以下场景，给出最可能的诊断结果和解决方案。

    场景描述：
      你正在训练一个 5 层 MLP 用于图像分类。
      使用 Adam 优化器，学习率 = 0.1（很大！）。
      前几个 batch 的 loss 正常下降：2.3 → 1.8 → 1.2 → 0.9 → NaN。
      梯度范数：0.5 → 2.1 → 8.7 → 53.4 → NaN。

    请补全以下诊断。
    """
    print("=" * 60)
    print("TODO 4 测试: 诊断 NaN Loss")
    print("=" * 60)

    # TODO: 分析场景并补全诊断

    # 最可能的原因:
    cause = None  # ← TODO: 从以下选项中选择最可能的原因
    # 选项:
    #   A. 学习率过大导致梯度爆炸
    #   B. 数据中有缺失值
    #   C. 模型参数初始化不当
    #   D. 激活函数饱和导致梯度消失

    # 解决方案（选择最合适的）:
    solution = None  # ← TODO: 从以下选项中选择解决方案
    # 选项:
    #   A. 降低学习率（如从 0.1 降到 0.001）
    #   B. 开启梯度裁剪（gradient clipping）
    #   C. 检查并清洗训练数据
    #   D. 以上全部（A + B + C）

    if cause is not None and solution is not None:
        print(f"\n  诊断结果: {cause}")
        print(f"  建议方案: {solution}")
        print(f"\n  分析: ")
        print(f"    - loss 在逐步下降但突然 NaN → 不是数据问题")
        print(f"    - 梯度范数每步都在快速增长(0.5→2.1→8.7→53.4)→ 梯度爆炸")
        print(f"    - lr=0.1 对 Adam 来说偏大(通常用 0.001)")
        print(f"    - 最直接的方案是降低学习率 + 添加梯度裁剪作为保险")
    else:
        print("\n  TODO: 请分析上述训练日志，选择最可能的原因和解决方案。")
        print("  提示：观察梯度范数的增长趋势。")

    print()


# ============================================================================
# 主程序
# ============================================================================
if __name__ == "__main__":
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║   s09 Adam 深度解析与训练实战 — 动手练习                   ║")
    print("║   请依次完成 TODO 1, 2, 3, 4                                ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    test_bias_correction()
    test_adamw()
    test_warmup()
    debug_nan_loss()

    print("=" * 60)
    print("所有测试完成！请检查输出结果。")
    print("如有未通过的测试，请回到对应的 TODO 部分补全代码。")
    print()
    print("Adam 核心公式:")
    print("  更新: θ = θ - α·m̂/(√v̂+ε)")
    print("  偏差修正: m̂ = m/(1-β₁^t), v̂ = v/(1-β₂^t)")
    print("  AdamW: θ = θ - α·m̂/(√v̂+ε) - αλθ")
    print("  Warmup: α_t = α_target · t/t_warmup")
    print("=" * 60)
