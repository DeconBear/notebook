# -*- coding: utf-8 -*-
from __future__ import annotations
"""
wm06 Genie：交互式世界模型与潜在动作 —— 演示代码
======================================================
功能：
  1. 绘制 Genie 架构示意图：视频分词器(Video Tokenizer) + 潜在动作模型
     (Latent Action Model) + 动态模型(Dynamics Model) 三段式流水线
     (wm06-01-genie.png)
  2. 构造一个极简的网格世界（GridWorld），生成大量"帧对 + 真实动作"的
     轨迹数据（真实动作标签只用于最后的评估，不用于训练！）
  3. 从零训练一个"玩具潜在动作模型"（Latent Action Model, LAM）：
     - 动作编码器：输入 (frame_t, frame_t+1)，输出连续潜向量
     - 向量量化（VQ）：把连续潜向量量化到 K 个离散码字（模拟"发现的动作词表"）
     - 动态模型（解码器）：输入 frame_t + 量化后的潜在动作码，预测 frame_t+1
     - 训练信号：只有"预测下一帧准不准"，从未使用过真实动作标签
  4. 训练后验证：这个完全无监督发现出来的离散潜在动作码，
     是否恰好和真实的 4 个动作（上/下/左/右）形成一一对应？

这正是 Genie 论文最关键的洞察：不需要任何动作标注的视频，
仅通过"预测下一帧"这个自监督目标，就能让模型自己发现
"动作"这个抽象概念，并将其编码为离散、可控的潜在变量。

每个函数都有中文 docstring，每行逻辑代码都有中文注释。
运行方式：在 wm06_genie/ 目录下执行 python code/demo.py
依赖: pip install numpy matplotlib torch
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_IMAGES = os.path.join(_HERE, '..', 'images')
os.makedirs(_IMAGES, exist_ok=True)

ACTIONS = ['上', '下', '左', '右']            # 4 个真实动作（仅用于评估，不用于训练）
GRID_SIZE = 6                                  # 网格世界边长


def set_seed(seed: int = 42):
    """设置随机种子，保证实验可复现。"""
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================================
# 第一部分：Genie 架构示意图
# ============================================================================

def _draw_box(ax, xy, w, h, text, color, fontsize=9.5):
    from matplotlib.patches import FancyBboxPatch
    x, y = xy
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                          linewidth=1.5, edgecolor='#333333', facecolor=color)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fontsize)
    return (x + w / 2, y + h / 2)


def _draw_arrow(ax, p1, p2, color='#333333', style='-|>'):
    ax.annotate('', xy=p2, xytext=p1, arrowprops=dict(arrowstyle=style, color=color, lw=1.6))


def plot_genie_architecture():
    """
    绘制 Genie 的三段式架构示意图：
    视频分词器 → 潜在动作模型（无监督发现动作） → 动态模型（逐帧生成，可交互）
    并展示推理时"用户注入自定义潜在动作"实现交互式生成的流程。
    """
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 9.5)
    ax.axis('off')
    ax.set_title('Genie：从无标注视频中学习潜在动作，实现可交互的世界生成',
                  fontsize=13.5, fontweight='bold')

    # ---- 训练阶段（上半部分） ----
    ax.text(0.2, 8.8, '【训练阶段】输入：海量无标注视频（没有动作标签！）', fontsize=11, color='#8B0000', fontweight='bold')

    vid = _draw_box(ax, (0.3, 6.9), 2.4, 1.3, '原始视频帧序列\n(frame 1, 2, 3, ...)', '#FFE8CC')
    tok = _draw_box(ax, (3.2, 6.9), 2.2, 1.3, '视频分词器\n(VQ-VAE Tokenizer)\n像素→离散token', '#CDE7FF')
    lam = _draw_box(ax, (6.0, 6.9), 2.6, 1.3, '潜在动作模型 (LAM)\n(frame_t,frame_t+1)→\n离散潜在动作码', '#D6F5D6')
    dyn = _draw_box(ax, (9.2, 6.9), 2.6, 1.3, '动态模型\n(frame_t token +\n潜在动作码)→frame_t+1', '#FFD6D6')

    _draw_arrow(ax, (2.7, 7.55), (3.2, 7.55))
    _draw_arrow(ax, (5.4, 7.55), (6.0, 7.55))
    _draw_arrow(ax, (8.6, 7.55), (9.2, 7.55))
    ax.annotate('', xy=(9.5, 6.9), xytext=(4.0, 6.0),
                arrowprops=dict(arrowstyle='-|>', color='#006400', lw=1.4,
                                 connectionstyle='arc3,rad=-0.3'))
    ax.text(6.2, 5.6, '训练信号：仅"预测下一帧token"的重建损失\n（动作码完全由模型自己发现，不使用人工标签）',
            fontsize=9, color='#006400', ha='center')

    # ---- 推理/交互阶段（下半部分） ----
    ax.text(0.2, 4.3, '【推理阶段】用户可像打游戏一样，逐帧指定"潜在动作"来控制生成', fontsize=11, color='#00008B', fontweight='bold')

    init_f = _draw_box(ax, (0.3, 2.2), 2.4, 1.3, '起始帧\n(单张图片)', '#FFE8CC')
    user_a = _draw_box(ax, (3.2, 2.2), 2.4, 1.3, '用户选择\n潜在动作 (1个码)\n如：方向键/手柄', '#FFF3B0')
    dyn2 = _draw_box(ax, (6.2, 2.2), 2.6, 1.3, '动态模型\n(自回归逐帧生成)', '#FFD6D6')
    out_f = _draw_box(ax, (9.4, 2.2), 2.6, 1.3, '生成下一帧\n(循环反馈,持续交互)', '#D6F5D6')

    _draw_arrow(ax, (2.7, 2.85), (3.2, 2.85))
    _draw_arrow(ax, (5.6, 2.85), (6.2, 2.85))
    _draw_arrow(ax, (8.8, 2.85), (9.4, 2.85))
    ax.annotate('', xy=(3.4, 2.2), xytext=(10.7, 2.2),
                arrowprops=dict(arrowstyle='-|>', color='#333333', lw=1.2,
                                 connectionstyle='arc3,rad=0.5'))
    ax.text(6.6, 0.7, '循环：生成的帧作为下一步输入，用户持续指定潜在动作 → 逐帧"玩"出一个交互世界',
            fontsize=9, color='#333333', ha='center')

    plt.savefig(os.path.join(_IMAGES, 'wm06-01-genie.png'), dpi=120, bbox_inches='tight')
    plt.close()
    print('[可视化] Genie 架构示意图已保存至 images/wm06-01-genie.png')


# ============================================================================
# 第二部分：极简网格世界 —— 生成轨迹数据
# ============================================================================

def make_grid_frame(pos: tuple, size: int = GRID_SIZE) -> np.ndarray:
    """
    根据智能体位置生成一张 one-hot 网格图像帧。

    参数:
        pos: (row, col) 智能体位置
        size: 网格边长
    返回:
        frame: (size, size) float32 数组，智能体所在格为 1，其余为 0
    """
    frame = np.zeros((size, size), dtype=np.float32)
    frame[pos[0], pos[1]] = 1.0
    return frame


def step_grid(pos: tuple, action: int, size: int = GRID_SIZE) -> tuple:
    """
    在网格世界中执行一步动作，返回新位置（越界则贴墙不动）。

    动作编码: 0=上, 1=下, 2=左, 3=右
    """
    r, c = pos
    if action == 0:
        r = max(0, r - 1)
    elif action == 1:
        r = min(size - 1, r + 1)
    elif action == 2:
        c = max(0, c - 1)
    elif action == 3:
        c = min(size - 1, c + 1)
    return (r, c)


def make_transition_dataset(n_transitions: int, size: int = GRID_SIZE):
    """
    生成大量 (frame_t, frame_t+1, true_action) 转移三元组。
    true_action 只用于训练后的评估，绝不会传给潜在动作模型。

    注意：只从"内部格子"（非边界）采样起始位置。原因是边界格子上
    存在动作歧义——例如在第 0 行时，"向上"和贴墙的"停留"效果相同，
    导致同一个 (frame_t, frame_t+1) 对应两种不同真实动作，这是环境
    本身固有的不可辨识性，而不是模型的问题。只用内部格子训练可以让
    "4个动作 ↔ 4个转移效果"保持一一对应，方便观察潜在动作发现的效果。

    返回:
        frames_t, frames_tp1: (n, size, size)
        true_actions: (n,) int
    """
    frames_t = np.zeros((n_transitions, size, size), dtype=np.float32)
    frames_tp1 = np.zeros((n_transitions, size, size), dtype=np.float32)
    true_actions = np.zeros(n_transitions, dtype=np.int64)

    for i in range(n_transitions):
        pos = (np.random.randint(1, size - 1), np.random.randint(1, size - 1))  # 内部格子
        action = np.random.randint(0, 4)
        new_pos = step_grid(pos, action, size)
        frames_t[i] = make_grid_frame(pos, size)
        frames_tp1[i] = make_grid_frame(new_pos, size)
        true_actions[i] = action

    return frames_t, frames_tp1, true_actions


# ============================================================================
# 第三部分：潜在动作模型（Latent Action Model, LAM）
# ============================================================================

class ActionEncoder(nn.Module):
    """
    动作编码器：输入 (frame_t, frame_t+1) 拼接后的图像对，
    输出一个连续的潜在动作向量（后续会被量化为离散码）。

    直觉：帧对之间的"差异"里包含了"发生了什么动作"的信息——
    这正是 Genie 无监督发现动作概念的关键：动作被定义为
    "能解释两帧之间变化的最简变量"，而不是人工标注的类别。
    """

    def __init__(self, grid_size: int, latent_dim: int = 8):
        super().__init__()
        in_dim = grid_size * grid_size * 2                       # frame_t 和 frame_t+1 拼接
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, latent_dim),
        )

    def forward(self, frame_t: torch.Tensor, frame_tp1: torch.Tensor) -> torch.Tensor:
        B = frame_t.size(0)
        x = torch.cat([frame_t.view(B, -1), frame_tp1.view(B, -1)], dim=1)
        return self.net(x)                                         # (B, latent_dim) 连续潜向量


class VectorQuantizer(nn.Module):
    """
    向量量化层（VQ）—— 把连续潜向量映射到最近的码字，实现"离散化"。

    这是 Genie 潜在动作模型的核心机制：将连续、高维、难以解释的潜向量
    压缩到一个很小的离散码表（比如 K=4），逼迫模型把动作信息压缩成
    "一个类别选择"，这恰好符合人类对"动作"的直觉理解
    （游戏手柄上只有几个离散按键）。
    """

    def __init__(self, n_codes: int, latent_dim: int):
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(n_codes, latent_dim) * 0.5)  # 码表
        self.n_codes = n_codes

    def forward(self, z: torch.Tensor):
        """
        参数:
            z: (B, latent_dim) 编码器输出的连续潜向量
        返回:
            z_q: (B, latent_dim) 量化后的向量（用于解码器的前向传播，直通梯度）
            code_idx: (B,) 每个样本选中的码字索引
            vq_loss: 承诺损失(commitment loss) + 码本损失(codebook loss)
        """
        dist = torch.cdist(z.unsqueeze(1), self.codebook.unsqueeze(0)).squeeze(1)  # (B, n_codes)
        code_idx = dist.argmin(dim=1)                              # 找到最近的码字
        z_q = self.codebook[code_idx]                                # 量化结果

        codebook_loss = F.mse_loss(z_q, z.detach())                  # 让码字靠近编码器输出
        commitment_loss = F.mse_loss(z, z_q.detach())                # 让编码器输出靠近码字
        vq_loss = codebook_loss + 0.25 * commitment_loss

        z_q_st = z + (z_q - z).detach()                               # 直通估计器(straight-through)
        return z_q_st, code_idx, vq_loss


class DynamicsModel(nn.Module):
    """
    动态模型（此处兼作解码器）：输入 frame_t + 量化后的潜在动作向量，
    预测 frame_t+1（在真实 Genie 中这是自回归 Transformer，
    这里简化为 MLP 分类头：预测智能体的新位置属于哪个格子）。
    """

    def __init__(self, grid_size: int, latent_dim: int):
        super().__init__()
        self.grid_size = grid_size
        in_dim = grid_size * grid_size + latent_dim
        n_cells = grid_size * grid_size
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, n_cells),                                    # 输出每个格子的logit
        )

    def forward(self, frame_t: torch.Tensor, action_latent: torch.Tensor) -> torch.Tensor:
        B = frame_t.size(0)
        x = torch.cat([frame_t.view(B, -1), action_latent], dim=1)
        logits = self.net(x)                                          # (B, grid_size*grid_size)
        return logits


class LatentActionWorldModel(nn.Module):
    """
    把动作编码器 + 向量量化 + 动态模型组装成完整的"玩具 Genie"。
    """

    def __init__(self, grid_size: int = GRID_SIZE, latent_dim: int = 8, n_codes: int = 4):
        super().__init__()
        self.action_encoder = ActionEncoder(grid_size, latent_dim)
        self.vq = VectorQuantizer(n_codes, latent_dim)
        self.dynamics = DynamicsModel(grid_size, latent_dim)
        self.grid_size = grid_size

    def forward(self, frame_t, frame_tp1):
        z = self.action_encoder(frame_t, frame_tp1)                    # 连续潜在动作
        z_q, code_idx, vq_loss = self.vq(z)                             # 量化为离散码
        logits = self.dynamics(frame_t, z_q)                            # 预测下一帧
        return logits, code_idx, vq_loss


# ============================================================================
# 第四部分：训练循环
# ============================================================================

def train_latent_action_model(frames_t, frames_tp1, true_actions, n_steps=800, batch_size=64, lr=1e-3):
    """
    训练潜在动作世界模型。

    损失 = 下一帧预测的交叉熵损失 + VQ 损失（承诺损失+码本损失）

    注意：整个训练过程中，true_actions 从未作为输入或损失的一部分使用，
    只在训练结束后用于"评估潜在动作码是否学到了真实动作的语义"。
    """
    model = LatentActionWorldModel(GRID_SIZE, latent_dim=8, n_codes=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_data = frames_t.shape[0]
    frames_t_t = torch.from_numpy(frames_t).float()
    frames_tp1_t = torch.from_numpy(frames_tp1).float()
    # 下一帧的目标：智能体所在格子的展平索引（用于交叉熵分类）
    target_idx_all = frames_tp1.reshape(n_data, -1).argmax(axis=1)
    target_idx_all_t = torch.from_numpy(target_idx_all).long()

    loss_history, recon_acc_history = [], []
    for step in range(n_steps):
        idx = np.random.choice(n_data, batch_size, replace=False)
        ft, ftp1 = frames_t_t[idx], frames_tp1_t[idx]
        target_idx = target_idx_all_t[idx]

        logits, code_idx, vq_loss = model(ft, ftp1)
        recon_loss = F.cross_entropy(logits, target_idx)                # 下一帧预测损失
        loss = recon_loss + vq_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (logits.argmax(dim=1) == target_idx).float().mean().item()
        loss_history.append(loss.item())
        recon_acc_history.append(acc)

        if (step + 1) % 100 == 0:
            print(f'  Step {step+1:4d}/{n_steps}: loss={loss.item():.4f}, '
                  f'下一帧预测准确率={acc:.3f}')

    return model, loss_history, recon_acc_history


# ============================================================================
# 第五部分：评估潜在动作是否对齐真实动作
# ============================================================================

@torch.no_grad()
def evaluate_latent_action_alignment(model, frames_t, frames_tp1, true_actions):
    """
    评估无监督发现的离散潜在动作码，与真实动作(上下左右)的对应关系。

    做法：统计"给定潜在码 k，真实动作最常见是什么"的混淆矩阵，
    再计算按此映射能达到的分类准确率（类似聚类评估中的最优匹配准确率）。
    """
    frames_t_t = torch.from_numpy(frames_t).float()
    frames_tp1_t = torch.from_numpy(frames_tp1).float()
    _, code_idx, _ = model(frames_t_t, frames_tp1_t)
    code_idx = code_idx.numpy()

    n_codes = model.vq.n_codes
    n_actions = len(ACTIONS)
    confusion = np.zeros((n_codes, n_actions), dtype=np.int64)
    for k, a in zip(code_idx, true_actions):
        confusion[k, a] += 1

    # 每个潜在码 → 出现最多的真实动作
    code_to_action = confusion.argmax(axis=1)
    correct = sum(confusion[k, code_to_action[k]] for k in range(n_codes))
    alignment_acc = correct / len(true_actions)

    return confusion, code_to_action, alignment_acc


def plot_alignment(confusion, code_to_action, alignment_acc):
    """可视化潜在动作码 vs 真实动作的混淆矩阵。"""
    n_codes = confusion.shape[0]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(confusion, cmap='Blues')
    plt.colorbar(im, ax=ax, label='样本数')
    ax.set_xticks(range(len(ACTIONS)))
    ax.set_xticklabels(ACTIONS)
    ax.set_yticks(range(n_codes))
    ax.set_yticklabels([f'潜在码 {k}\n(→{ACTIONS[code_to_action[k]]})' for k in range(n_codes)])
    ax.set_xlabel('真实动作')
    ax.set_ylabel('无监督发现的潜在动作码')
    ax.set_title(f'潜在动作 vs 真实动作对齐\n(最优匹配准确率={alignment_acc:.1%})',
                 fontsize=12, fontweight='bold')
    for i in range(n_codes):
        for j in range(len(ACTIONS)):
            ax.text(j, i, confusion[i, j], ha='center', va='center',
                     color='white' if confusion[i, j] > confusion.max() / 2 else 'black')
    plt.tight_layout()
    plt.savefig(os.path.join(_IMAGES, 'genie_latent_action_alignment.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('[可视化] 潜在动作对齐混淆矩阵已保存至 images/genie_latent_action_alignment.png')


def plot_training_curves(loss_history, recon_acc_history):
    """绘制训练损失和下一帧预测准确率曲线。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(loss_history, color='#2E86AB')
    axes[0].set_xlabel('训练步数')
    axes[0].set_ylabel('总损失(重建+VQ)')
    axes[0].set_title('训练损失曲线')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(recon_acc_history, color='#F18F01')
    axes[1].axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='100%准确率')
    axes[1].set_xlabel('训练步数')
    axes[1].set_ylabel('下一帧预测准确率')
    axes[1].set_title('动态模型：下一帧预测准确率')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(_IMAGES, 'genie_training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('[可视化] 训练曲线已保存至 images/genie_training_curves.png')


def plot_rollout_demo(model, size=GRID_SIZE):
    """
    演示"交互式生成"：从起点开始，依次给定潜在码 0,1,2,3，
    展示动态模型预测的智能体轨迹（即"用潜在动作驱动世界演化"）。
    """
    model.eval()
    start_pos = (3, 3)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.3))
    for k in range(4):
        frame = make_grid_frame(start_pos, size)
        frame_t = torch.from_numpy(frame).float().unsqueeze(0)
        code_onehot_vec = model.vq.codebook[k].unsqueeze(0)              # 直接指定潜在码k
        with torch.no_grad():
            logits = model.dynamics(frame_t, code_onehot_vec)
        pred_pos_idx = logits.argmax(dim=1).item()
        pred_pos = (pred_pos_idx // size, pred_pos_idx % size)

        vis = np.zeros((size, size, 3))
        vis[start_pos[0], start_pos[1]] = [0.3, 0.3, 1.0]                # 起点=蓝
        vis[pred_pos[0], pred_pos[1]] = [1.0, 0.3, 0.3]                  # 预测终点=红
        axes[k].imshow(vis, interpolation='nearest')
        axes[k].set_title(f'潜在码 {k}\n{start_pos}→{pred_pos}', fontsize=10)
        axes[k].set_xticks([])
        axes[k].set_yticks([])
    fig.suptitle('用 4 个离散潜在动作码"驱动"世界演化(蓝=起点,红=预测下一步)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(_IMAGES, 'genie_rollout_demo.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('[可视化] 交互式生成演示已保存至 images/genie_rollout_demo.png')


# ============================================================================
# 主程序
# ============================================================================

def main():
    print('\n' + '=' * 70)
    print('    wm06 Genie：交互式世界模型与潜在动作 — 完整演示')
    print('=' * 70)

    set_seed(42)

    print('\n[第1步] 绘制 Genie 架构示意图...')
    plot_genie_architecture()

    print('\n[第2步] 生成网格世界转移数据集（真实动作标签仅用于最终评估）...')
    frames_t, frames_tp1, true_actions = make_transition_dataset(n_transitions=4000)
    print(f'  生成 {frames_t.shape[0]} 条 (frame_t, frame_t+1) 转移样本，网格大小 {GRID_SIZE}x{GRID_SIZE}')

    print('\n[第3步] 训练潜在动作模型（LAM + 动态模型），从未使用真实动作标签...')
    model, loss_history, recon_acc_history = train_latent_action_model(
        frames_t, frames_tp1, true_actions, n_steps=800, batch_size=64)

    print('\n[第4步] 评估：无监督发现的潜在动作码是否对齐真实动作？')
    confusion, code_to_action, alignment_acc = evaluate_latent_action_alignment(
        model, frames_t, frames_tp1, true_actions)
    print(f'  潜在码 → 真实动作映射: {dict(enumerate([ACTIONS[a] for a in code_to_action]))}')
    print(f'  对齐准确率: {alignment_acc:.1%}')

    print('\n[第5步] 可视化...')
    plot_training_curves(loss_history, recon_acc_history)
    plot_alignment(confusion, code_to_action, alignment_acc)
    plot_rollout_demo(model)

    print('\n' + '=' * 70)
    print('【总结】')
    print('=' * 70)
    print(f'  下一帧预测最终准确率: {np.mean(recon_acc_history[-50:]):.1%}')
    print(f'  潜在动作 vs 真实动作对齐准确率: {alignment_acc:.1%}')
    print('\n  【Genie 核心机制回顾】')
    print('  - 潜在动作模型只用"预测下一帧"作为训练信号，从不使用动作标签')
    print('  - 向量量化(VQ)把连续潜向量压缩为少量离散码，逼出"动作"这个概念')
    print('  - 训练好后，用户可在推理时手动指定潜在码，实现"逐帧可交互"的生成')
    print('  - Genie 1(2D平台游戏) → Genie 2(3D世界,更长时序一致性)')
    print('    → Genie 3(实时交互,更高分辨率,多智能体世界)')
    print('\n  所有图片已保存至 images/ 目录')
    print('=' * 70)
    print('\n  运行完成！\n')


if __name__ == '__main__':
    main()
