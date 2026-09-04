# -*- coding: utf-8 -*-
"""
s18 大语言模型 demo：Scaling, Emergence, LoRA, DPO
====================================================
本文件演示大语言模型的核心概念：
  1. Scaling Law 可视化（Kaplan + Chinchilla）
  2. 涌现行为模拟（算术能力 vs 模型大小）
  3. LoRA 低秩微调（使用 PEFT 库）
  4. DPO 偏好优化（使用 TRL 库）
  5. 指令遵循对比

运行方式：在 s18_large_language_models 目录下执行 `python code/demo.py`
依赖：torch, transformers, peft, trl, matplotlib
注意：LoRA/DPO 部分需要下载小模型（~500MB，支持消费级硬件）
"""

import numpy as np
import math
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# GPU 自动检测
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")
if DEVICE.type == 'cpu':
    print("（未检测到 GPU，使用 CPU 运行。如有 GPU，请安装 CUDA 版 PyTorch 以获得加速）")

# ====== 可选：使用 LLM API ======
# 如需使用真实 LLM API，请设置环境变量：
#   export OPENAI_API_KEY=your-key
#   export OPENAI_BASE_URL=https://api.openai.com/v1
# 然后将 USE_API = False 改为 True
USE_API = False

import matplotlib.pyplot as plt
import matplotlib
# 中文字体配置
matplotlib.rcParams['axes.unicode_minus'] = False

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMAGES = os.path.join(_HERE, '..', 'images')
os.makedirs(_IMAGES, exist_ok=True)

# ============================================================
# 第一部分：Scaling Law 可视化
# ============================================================

print("=" * 60)
print("[Scaling Law] 语言模型损失的幂律下降")
print("=" * 60)

def kaplan_loss(N: float, D: float, a: float = 1.5, b: float = 2.0, alpha: float = 0.076, beta: float = 0.095, c: float = 1.0) -> float:
    """
    Kaplan 等人提出的 Scaling Law:
    L(N, D) = a/N^α + b/D^β + c

    参数：
        N: 模型参数量
        D: 训练数据量 (tokens)
        a, b: 系数
        α, β: 幂律指数
        c: 不可约减损失 (irreducible loss)
    返回：
        预测的测试损失
    """
    return a / (N ** alpha) + b / (D ** beta) + c


def chinchilla_optimal_D(N: float) -> float:
    """
    Chinchilla 最优配比: D ≈ 20 × N

    参数：
        N: 模型参数量
    返回：
        最优的训练 token 数
    """
    return 20.0 * N


# 绘制 Scaling Law 曲线
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# 图 1: 损失 vs 模型大小
N_range = np.logspace(6, 12, 100)  # 1M → 1T 参数
D_fixed = 3e11  # 固定数据量 (300B tokens, 类似 GPT-3)
losses_N = [kaplan_loss(N, D_fixed) for N in N_range]
ax1 = axes[0, 0]
ax1.loglog(N_range, losses_N, 'b-', linewidth=2)
ax1.scatter([1.17e8, 1.5e9, 1.75e11], [kaplan_loss(1.17e8, D_fixed), kaplan_loss(1.5e9, D_fixed), kaplan_loss(1.75e11, D_fixed)],
           color='red', s=80, zorder=5)
ax1.annotate('GPT-1\n117M', (1.5e8, kaplan_loss(1.17e8, D_fixed) + 0.1), fontsize=8, color='red')
ax1.annotate('GPT-2\n1.5B', (2e9, kaplan_loss(1.5e9, D_fixed) + 0.08), fontsize=8, color='red')
ax1.annotate('GPT-3\n175B', (2e11, kaplan_loss(1.75e11, D_fixed) + 0.08), fontsize=8, color='red')
ax1.set_xlabel("Model Parameters N", fontsize=11)
ax1.set_ylabel("Test Loss L", fontsize=11)
ax1.set_title("L(N) ∝ N^(-α), α≈0.076", fontsize=12)
ax1.grid(True, alpha=0.3, which='both')
ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Irreducible Loss c')
ax1.legend(fontsize=9)

# 图 2: 损失 vs 数据量
D_range = np.logspace(7, 13, 100)  # 10M → 10T tokens
N_fixed = 7e9  # 固定模型大小 (7B 参数, 类似 LLaMA)
losses_D = [kaplan_loss(N_fixed, D) for D in D_range]
ax2 = axes[0, 1]
ax2.loglog(D_range, losses_D, 'g-', linewidth=2)
ax2.set_xlabel("Training Data D (tokens)", fontsize=11)
ax2.set_ylabel("Test Loss L", fontsize=11)
ax2.set_title("L(D) ∝ D^(-β), β≈0.095", fontsize=12)
ax2.grid(True, alpha=0.3, which='both')

# 图 3: 损失 vs 计算量
C_range = np.logspace(-3, 6, 100)  # PF-days
# 计算量大约 C ≈ 6ND (经验公式)
losses_C = [kaplan_loss(np.sqrt(c/6*1e15), np.sqrt(c/6*1e15)) for c in C_range]
ax3 = axes[1, 0]
ax3.loglog(C_range, losses_C, 'purple', linewidth=2)
ax3.set_xlabel("Compute C (PF-days)", fontsize=11)
ax3.set_ylabel("Test Loss L", fontsize=11)
ax3.set_title("L(C) ∝ C^(-γ), γ≈0.057", fontsize=12)
ax3.grid(True, alpha=0.3, which='both')

# 图 4: Chinchilla 最优配比等高线
N_vals = np.logspace(6, 11, 50)
D_vals = np.logspace(8, 13, 50)
NN, DD = np.meshgrid(N_vals, D_vals)
LL = kaplan_loss(NN, DD)
ax4 = axes[1, 1]
contour = ax4.contour(np.log10(NN), np.log10(DD), LL, levels=10, cmap='RdYlBu_r')
ax4.clabel(contour, inline=True, fontsize=8)
# Chinchilla 最优线: D = 20N
optimal_D = [chinchilla_optimal_D(n) for n in N_vals]
ax4.loglog(N_vals, optimal_D, 'r--', linewidth=2, label='Chinchilla Optimal D≈20N')
# GPT-3 点
ax4.scatter([1.75e11], [3e11], color='orange', s=100, zorder=5, marker='s')
ax4.annotate('GPT-3\n(Undertrained)', (2e11, 4e11), fontsize=8, color='darkorange')
# LLaMA 7B 点
ax4.scatter([7e9], [1e12], color='green', s=100, zorder=5, marker='^')
ax4.annotate('LLaMA 7B\n(Near-optimal)', (1e10, 1.5e12), fontsize=8, color='green')
ax4.set_xlabel("Model Parameters N (log)", fontsize=11)
ax4.set_ylabel("Training Tokens D (log)", fontsize=11)
ax4.set_title("Chinchilla Optimal Ratio: D ≈ 20N", fontsize=12)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

plt.suptitle("Scaling Laws of Language Models", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(_IMAGES, 'scaling_laws.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[可视化] Scaling Laws 图表已保存至 images/scaling_laws.png")
print(f"\n[计算示例]")
print(f"  7B 模型 + 300B tokens: L={kaplan_loss(7e9, 3e11):.3f}")
print(f"  7B 模型 + 1.4T tokens (Chinchilla最优): L={kaplan_loss(7e9, 1.4e12):.3f}")
print()


# ============================================================
# 第二部分：涌现行为模拟
# ============================================================

print("=" * 60)
print("[涌现行为] 算术能力 vs 模型大小")
print("=" * 60)


def simulate_emergence(
    param_sizes: np.ndarray,   # 模型大小列表
    task: str,                 # 任务名称
    emergent: bool = True,     # 是否有涌现特性
    threshold: float = 1e9,    # 涌现阈值
    noise_level: float = 0.05, # 噪声水平
) -> np.ndarray:
    """
    模拟不同模型大小下的任务准确率。
    涌现任务：在阈值前接近随机，阈值后快速跃升。
    非涌现任务：平滑增长。

    参数：
        param_sizes: 模型参数量数组
        task: 任务名称
        emergent: 是否为涌现任务
        threshold: 涌现阈值
        noise_level: 噪声水平
    返回：
        accuracies: 模拟的准确率数组
    """
    if emergent:
        # 涌现：sigmoid 函数模拟相位转变
        # logistic function: 1 / (1 + exp(-k*(x - threshold)))
        accuracies = 1.0 / (1.0 + np.exp(-1.5 * (np.log10(param_sizes) - np.log10(threshold))))
        # 加入"近随机"基线
        accuracies = 0.05 + 0.85 * accuracies
    else:
        # 非涌现：平滑的线性 + 轻微的指数增长
        accuracies = 0.1 + 0.8 * (np.log10(param_sizes) - 6.0) / 6.0
        accuracies = np.clip(accuracies, 0.1, 0.95)
    # 加噪声
    accuracies += np.random.normal(0, noise_level, len(param_sizes))
    return np.clip(accuracies, 0.0, 1.0)


# 设定模型大小范围
param_range = np.logspace(6, 12, 30)  # 1M → 1T

# 模拟 6 个任务
tasks = {
    "3-Digit Arithmetic": (True, 8e9),        # Emergent, threshold ~8B
    "Multilingual Translation": (True, 1e10),  # Emergent, threshold ~10B
    "Chain-of-Thought (CoT)": (True, 6e10),    # Emergent, threshold ~60B
    "Instruction Following": (True, 3e10),     # Emergent, threshold ~30B
    "Sentiment Analysis": (False, None),        # Non-emergent: smooth growth
    "POS Tagging": (False, None),              # Non-emergent: smooth growth
}

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
colors_emergent = ['#E53935', '#E53935', '#E53935', '#E53935']
colors_smooth = ['#1E88E5', '#1E88E5']

emergent_idx = 0
smooth_idx = 0

for ax, (task_name, (is_emergent, threshold)) in zip(axes, tasks.items()):
    accs = simulate_emergence(param_range, task_name, emergent=is_emergent, threshold=threshold)
    color = colors_emergent[emergent_idx] if is_emergent else colors_smooth[smooth_idx]
    if is_emergent:
        emergent_idx += 1
    else:
        smooth_idx += 1

    ax.semilogx(param_range, accs * 100, 'o-', color=color, linewidth=1.5, markersize=4)
    ax.set_xlabel("Model Parameters", fontsize=9)
    ax.set_ylabel("Accuracy (%)", fontsize=9)
    ax.set_title(f"{task_name} {'(Emergent)' if is_emergent else '(Smooth Growth)'}", fontsize=11)

    if is_emergent and threshold:
        ax.axvline(x=threshold, color='gray', linestyle='--', alpha=0.5)
        ax.annotate(f'Emergence threshold\n~{threshold/1e9:.0f}B',
                   xy=(threshold, 50), fontsize=8, color='red',
                   ha='left',
                   arrowprops=dict(arrowstyle='->', color='red', lw=1))

    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=10, color='gray', linestyle=':', alpha=0.3, label='Random baseline 10%' if task_name == "3-Digit Arithmetic" else '')

plt.suptitle("Emergent vs Smooth Growth: Task Behavior by Model Scale", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(_IMAGES, 'emergent_abilities.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[可视化] 涌现能力图已保存至 images/emergent_abilities.png")
print()
print("  涌现能力特征: 小模型≈随机水平 → 跨过阈值 → 大幅跃升")
print("  非涌现能力: 随模型大小平滑增长，小模型也能做")
print()


# ============================================================
# 第三部分：DPO 损失函数实现（核心算法理解）
# ============================================================

print("=" * 60)
print("[DPO] 直接偏好优化损失函数")
print("=" * 60)


def dpo_loss(
    pi_logps_chosen: torch.Tensor,    # 策略模型对偏好回答的 log 概率 (batch,)
    pi_logps_rejected: torch.Tensor,   # 策略模型对较差回答的 log 概率 (batch,)
    ref_logps_chosen: torch.Tensor,    # 参考模型对偏好回答的 log 概率 (batch,)
    ref_logps_rejected: torch.Tensor,  # 参考模型对较差回答的 log 概率 (batch,)
    beta: float = 0.1,                 # KL 惩罚系数
) -> torch.Tensor:
    """
    计算 DPO 损失。

    DPO 损失公式:
    L_DPO = -log σ( β·log[π_θ(y_w|x)/π_ref(y_w|x)] - β·log[π_θ(y_l|x)/π_ref(y_l|x)] )

    其中 π_θ 是要优化的策略模型，π_ref 是参考模型（通常是 SFT 模型），
    y_w 是人类偏好的回答 (winner/wanted)，y_l 是较差回答 (loser)。

    直觉: 如果策略模型给好回答的概率比参考模型高，且给差回答的概率比参考模型低，
    则损失小。反之，如果策略模型在好回答和差回答上的表现和参考模型一样
    （甚至更差），损失大。

    参数：
        pi_logps_chosen: log π_θ(y_w | x), 策略模型下偏好回答的对数概率
        pi_logps_rejected: log π_θ(y_l | x), 策略模型下较差回答的对数概率
        ref_logps_chosen: log π_ref(y_w | x), 参考模型下偏好回答的对数概率
        ref_logps_rejected: log π_ref(y_l | x), 参考模型下较差回答的对数概率
        beta: KL 惩罚系数，越大越鼓励模型偏离参考模型（但也越容易过拟合）
    返回：
        loss: DPO 损失值
    """
    # 计算策略模型与参考模型在 log 概率上的差异
    pi_diff = pi_logps_chosen - pi_logps_rejected       # 策略模型: 好回答 - 差回答
    ref_diff = ref_logps_chosen - ref_logps_rejected     # 参考模型: 好回答 - 差回答

    # 加权差异，乘以 beta
    logits = beta * (pi_diff - ref_diff)  # DPO 论文中的隐式奖励

    # 二分类交叉熵损失: -log σ(logits)
    loss = -F.logsigmoid(logits).mean()
    return loss


# 模拟 DPO 训练的损失变化
print("\n[模拟] DPO 训练过程中的损失变化...")
torch.manual_seed(42)

# 假设 50 个偏好对
num_pairs = 50
# 模拟：随着训练进行，模型越来越好地学会偏好
dpo_losses = []
for step_ratio in np.linspace(0.01, 1.0, 30):
    # 模拟策略模型越来越好（与参考模型的差异越来越大）
    pi_diff_base = step_ratio * 3.0  # 初期差异小，后期差异大
    pi_diff_chosen = pi_diff_base + torch.randn(num_pairs) * 0.5
    pi_diff_rejected = -pi_diff_base + torch.randn(num_pairs) * 0.5
    ref_diff_chosen = torch.zeros(num_pairs)  # 参考模型差异为 0（它是固定的）
    ref_diff_rejected = torch.zeros(num_pairs)

    loss = dpo_loss(
        pi_logps_chosen=pi_diff_chosen,
        pi_logps_rejected=pi_diff_rejected,
        ref_logps_chosen=ref_diff_chosen,
        ref_logps_rejected=ref_diff_rejected,
        beta=0.1,
    )
    dpo_losses.append(loss.item())

# 绘制 DPO 训练损失曲线
plt.figure(figsize=(8, 4))
plt.plot(np.linspace(0.01, 1.0, 30), dpo_losses, 'o-', color='#00897B', linewidth=1.5, markersize=4)
plt.xlabel("Training Progress", fontsize=12)
plt.ylabel("DPO Loss", fontsize=12)
plt.title("DPO Training Loss (Simulated): Model Learns to Prefer Good Responses", fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(_IMAGES, 'dpo_training_loss.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[可视化] DPO 训练损失曲线已保存至 images/dpo_training_loss.png")
print()

# 展示 DPO 损失的计算示例
# 场景 1: 模型正确偏好 (好回答概率高，差回答概率低)
print("[DPO 数值示例]")
pi_good = torch.tensor([-2.0, -2.5, -1.8])  # log 概率（负数）
pi_bad = torch.tensor([-5.0, -5.5, -4.8])
ref_good = torch.tensor([-3.0, -3.0, -3.0])
ref_bad = torch.tensor([-3.0, -3.0, -3.0])
loss_correct = dpo_loss(pi_good, pi_bad, ref_good, ref_bad, beta=0.1)
print(f"  正确偏好场景 (好回答log P=-2, 差回答log P=-5): Loss={loss_correct.item():.4f} (应该较小)")

# 场景 2: 模型错误偏好 (好回答和差回答的概率差不多)
pi_bad_model = torch.tensor([-3.0, -3.2, -2.9])
pi_bad_bad = torch.tensor([-3.1, -3.3, -3.0])
loss_wrong = dpo_loss(pi_bad_model, pi_bad_bad, ref_good, ref_bad, beta=0.1)
print(f"  错误偏好场景 (好/差回答概率接近): Loss={loss_wrong.item():.4f} (应该较大)")

# ============================================================
# 第四部分：LoRA 配置（概念演示）
# ============================================================

print("\n" + "=" * 60)
print("[LoRA] 低秩适配概念演示")
print("=" * 60)


class LoRALinear(nn.Module):
    """
    简化的 LoRA 线性层（概念演示）。
    LoRA 不修改原始权重 W，而是学习一个低秩更新 ΔW = BA：
      h = Wx + (α/r) * B A x

    参数：
        in_features: 输入维度
        out_features: 输出维度
        r: LoRA 秩（低秩分解的秩，通常 8-64）
        alpha: LoRA 缩放系数
    """

    def __init__(self, in_features: int, out_features: int, r: int = 8, alpha: float = 16.0):
        super().__init__()
        # 原始权重（冻结，不参与训练）
        self.register_buffer('W', torch.randn(out_features, in_features) * 0.02)
        # LoRA 低秩矩阵 A 和 B
        # A: (r, in_features), 用 Kaiming 初始化
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.02)
        # B: (out_features, r), 初始化为 0（开始时 ΔW = 0 × A = 0）
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r  # 缩放因子

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播: h = Wx + scaling * B @ A @ x

        参数：
            x: 输入 (batch, in_features)
        返回：
            h: 输出 (batch, out_features)
        """
        # 原始路径（不计算梯度）
        original = x @ self.W.T  # (batch, out_features)
        # LoRA 路径（低秩更新）
        lora_out = x @ self.lora_A.T @ self.lora_B.T  # x → (r,) → (out_features,)
        return original + self.scaling * lora_out


# 比较 LoRA 的参数量
d = 4096  # d_model
r = 16    # LoRA 秩

vanilla_params = d * d  # 一个全连接层的参数
lora_params = 2 * d * r  # A: (r, d) + B: (d, r)

print(f"\n  全参数训练:  {vanilla_params:,} 参数")
print(f"  LoRA (r={r}): {lora_params:,} 参数 ({lora_params/vanilla_params*100:.2f}%)")
print(f"  参数量减少:  {vanilla_params/lora_params:.0f}x")
print()

# 模拟 LoRA 微调过程
print("[模拟] LoRA 微调: 模型学习新任务...")
lora_layer = LoRALinear(256, 256, r=8, alpha=16.0).to(DEVICE)
optimizer = optim.Adam(lora_layer.parameters(), lr=0.01)
# 模拟一个简单的回归任务
lora_losses = []
for epoch in range(100):
    x_batch = torch.randn(16, 256).to(DEVICE)
    y_batch = torch.randn(16, 256).to(DEVICE)  # 目标
    optimizer.zero_grad()
    pred = lora_layer(x_batch)
    loss = F.mse_loss(pred, y_batch)
    loss.backward()
    optimizer.step()
    lora_losses.append(loss.item())

plt.figure(figsize=(8, 4))
plt.plot(lora_losses, color='#7B1FA2', linewidth=1.5)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("MSE Loss", fontsize=12)
plt.title("LoRA Fine-tuning Training Loss (Simulated)", fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(_IMAGES, 'lora_training_loss.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[可视化] LoRA 训练损失曲线已保存至 images/lora_training_loss.png")
print()

# ============================================================
# 第五部分：总结
# ============================================================

print("=" * 60)
print("[总结] 大语言模型的核心概念")
print("=" * 60)
print("""
  1. Scaling Law:
     L(N, D) = a/N^α + b/D^β + c
     损失随参数/数据量呈幂律下降
     Chinchilla最优: D ≈ 20N

  2. 涌现能力:
     小模型不会 → 跨过阈值 → 突然会了
     3位算术、CoT推理、指令遵循 —— 在~10B后涌现

  3. 指令微调 (SFT):
     用 (指令, 回复) 对训练模型执行任务

  4. 对齐:
     RLHF: SFT → Reward Model → PPO
     DPO: 直接从偏好数据优化，无需奖励模型

  5. LoRA:
     不修改原始权重，学习低秩增量 ΔW = BA
     参数量减少 100-1000x，消费级硬件即可微调大模型

  下一站:
     s22 多模态模型 — CLIP, 图文对齐
     s23 RAG 与 Agent — 检索增强生成 + 工具调用
     s24 部署与推理优化 — 量化, KV Cache, Flash Attention
""")
print("\n所有 demo 运行完成！图表已保存至 images/ 目录。")
