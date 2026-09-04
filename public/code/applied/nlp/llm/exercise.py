# -*- coding: utf-8 -*-
"""
s18 大语言模型 — 练习题
==============================================
请补全以下 TODO 部分，完成后运行验证。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# ============================================================
# 练习 1：实现 Chinchilla 最优配比计算
# ============================================================

def find_optimal_ND(
    compute_budget: float,  # 计算预算 C（FLOPs）
    a: float = 1.5,
    b: float = 2.0,
    alpha: float = 0.076,
    beta: float = 0.095,
) -> tuple:
    """
    TODO: 对于给定的计算预算 C，找到最优的参数 N 和数据 D
    使得在约束 C ≈ 6ND 下最小化损失 L(N, D) = a/N^α + b/D^β + c

    用拉格朗日乘数法可以推出:
    N_opt ∝ C^(β/(α+β))
    D_opt ∝ C^(α/(α+β))

    参数：
        compute_budget: 计算预算 C（FLOPs）
        a, b, alpha, beta: Kaplan scaling law 参数
    返回：
        (N_opt, D_opt, L_opt): 最优参数量、最优数据量、对应的损失
    """
    # TODO: 实现找到最优 N 和 D
    # 步骤：
    #   1. 设置约束: C = 6 * N * D
    #   2. 将 D = C / (6N) 代入损失函数
    #   3. 使用数值方法（如简单网格搜索）找到最小化损失的 N
    #   4. 或者使用拉格朗日乘数推导的比例关系
    #
    # 提示（简化数值解法）:
    #   对 N 在合理范围内做搜索：
    #     N 从 1e6 到 1e12 对数搜索
    #     D = compute_budget / (6 * N)
    #     计算 L = a/N^α + b/D^β + c
    #     找到最小 L 对应的 N, D
    # ===== 你的代码在这里 =====
    N_opt = 1e9  # 占位值
    D_opt = compute_budget / (6 * N_opt)
    L_opt = a / (N_opt ** alpha) + b / (D_opt ** beta) + 1.0  # c=1.0
    # ==========================
    return N_opt, D_opt, L_opt


# 测试
# 假设 C = 6 * 7e9 * 1.4e12 FLOPs（LLaMA 7B 规模的计算预算）
C_test = 6 * 7e9 * 1.4e12
try:
    N_opt, D_opt, L_opt = find_optimal_ND(C_test)
    print(f"[练习1] Chinchilla 最优配比:")
    print(f"  给定计算预算 C={C_test:.2e} FLOPs")
    print(f"  最优参数量 N={N_opt:.2e} (~{N_opt/1e9:.1f}B)")
    print(f"  最优数据量 D={D_opt:.2e} tokens (~{D_opt/1e12:.2f}T)")
    print(f"  最优配比 D/N={D_opt/N_opt:.1f} (期望: ≈20)")
    print(f"  预测损失 L={L_opt:.4f}")
except Exception as e:
    print(f"[练习1] 未完成实现: {e}")


# ============================================================
# 练习 2：实现 DPO 损失函数
# ============================================================

def dpo_loss(
    pi_logps_chosen: torch.Tensor,    # (batch,) 策略模型对偏好回答的 log 概率
    pi_logps_rejected: torch.Tensor,   # (batch,) 策略模型对较差回答的 log 概率
    ref_logps_chosen: torch.Tensor,    # (batch,) 参考模型对偏好回答的 log 概率
    ref_logps_rejected: torch.Tensor,  # (batch,) 参考模型对较差回答的 log 概率
    beta: float = 0.1,                 # KL 惩罚系数
) -> torch.Tensor:
    """
    TODO: 实现 DPO（Direct Preference Optimization）损失函数

    公式:
    L_DPO = -log σ( β*(log π_θ(y_w|x) - log π_ref(y_w|x))
                   - β*(log π_θ(y_l|x) - log π_ref(y_l|x)) )

    即: -log σ( β * (pi_diff - ref_diff) )
    其中 pi_diff = log π_θ(y_w) - log π_θ(y_l)
          ref_diff = log π_ref(y_w) - log π_ref(y_l)

    参数：
        pi_logps_chosen: log π_θ(y_w | x)
        pi_logps_rejected: log π_θ(y_l | x)
        ref_logps_chosen: log π_ref(y_w | x)
        ref_logps_rejected: log π_ref(y_l | x)
        beta: KL 惩罚系数
    返回：
        loss: 标量 DPO 损失值
    """
    # TODO: 实现步骤
    #   1. pi_log_ratio = pi_logps_chosen - pi_logps_rejected
    #   2. ref_log_ratio = ref_logps_chosen - ref_logps_rejected
    #   3. logits = beta * (pi_log_ratio - ref_log_ratio)
    #   4. loss = -F.logsigmoid(logits).mean()
    # ===== 你的代码在这里 =====
    loss = torch.tensor(0.0)
    # ==========================
    return loss


# 测试 DPO 损失
chosen = torch.tensor([-1.5, -2.0, -1.8])    # 好回答 log 概率
rejected = torch.tensor([-4.0, -5.0, -4.5])  # 差回答 log 概率
ref_chosen = torch.tensor([-2.5, -2.5, -2.5])  # 参考模型均匀
ref_rejected = torch.tensor([-2.5, -2.5, -2.5])
try:
    loss = dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta=0.1)
    print(f"\n[练习2] DPO 损失:")
    print(f"  好回答log P={chosen.tolist()}, 差回答log P={rejected.tolist()}")
    print(f"  DPO Loss = {loss.item():.4f} (期望: 一个较小的正数)")
    # 好的模型应该给好回答高概率，差回答低概率 → 损失应该小
except Exception as e:
    print(f"[练习2] 未完成实现: {e}")


# ============================================================
# 练习 3：配置 LoRA 适配器
# ============================================================

def create_lora_config(
    r: int = 8,               # LoRA 秩
    lora_alpha: float = 16.0, # LoRA 缩放系数
    target_modules: list = None,  # 需要应用 LoRA 的模块名列表
    lora_dropout: float = 0.1,    # LoRA dropout
) -> dict:
    """
    TODO: 创建一个 LoRA 配置字典（模拟 PEFT 库的 LoraConfig）

    参数：
        r: LoRA 秩
        lora_alpha: LoRA alpha 缩放参数
        target_modules: 需要应用 LoRA 的目标模块（如 ["q_proj", "v_proj"]）
        lora_dropout: LoRA dropout 概率
    返回：
        config: LoRA 配置字典
    """
    if target_modules is None:
        # Qwen/Llama 系列常见的目标模块
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    # TODO: 构建配置字典
    # 包含以下键:
    #   "r": r
    #   "lora_alpha": lora_alpha
    #   "target_modules": target_modules
    #   "lora_dropout": lora_dropout
    #   "bias": "none" (LoRA 不训练 bias)
    #   "task_type": "CAUSAL_LM" (因果语言模型)
    # ===== 你的代码在这里 =====
    config = {}
    # ==========================
    return config


# 测试
config = create_lora_config(r=16, lora_alpha=32.0,
                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
print(f"\n[练习3] LoRA 配置: {config}")

print("\n所有练习测试完成！请对比 demo.py 查看参考实现。")
print("""
提示:
  - Chinchilla 最优: 用拉格朗日乘数法或网格搜索找到最小化损失的配比
  - DPO: 核心是让好回答概率 > 差回答概率，公式: -log σ(β·ΔlogP)
  - LoRA: 秩 r 通常 8-64, alpha 通常 16-32, 常见的 target_modules 是 q_proj 和 v_proj
""")
