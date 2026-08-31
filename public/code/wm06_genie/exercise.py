# -*- coding: utf-8 -*-
from __future__ import annotations
"""
wm06 Genie：交互式世界模型与潜在动作 —— 练习代码
======================================================
请完成以下 3 个 TODO 任务，巩固对 Genie 潜在动作模型的理解：
  TODO 1: 实现 step_grid() —— 网格世界中的一步动作转移
  TODO 2: 实现 VectorQuantizer.forward() —— 向量量化 + 直通估计器
  TODO 3: 实现 compute_next_frame_loss() —— 用潜在动作预测下一帧

建议先阅读 index.md 和 demo.py，再尝试独立补全代码。
运行方式：在 wm06_genie/ 目录下执行 python code/exercise.py
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

ACTIONS = ['上', '下', '左', '右']
GRID_SIZE = 6


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_grid_frame(pos, size=GRID_SIZE):
    frame = np.zeros((size, size), dtype=np.float32)
    frame[pos[0], pos[1]] = 1.0
    return frame


# ============================================================================
# TODO 1: 实现 step_grid() —— 网格世界一步动作
# ============================================================================

def step_grid(pos: tuple, action: int, size: int = GRID_SIZE) -> tuple:
    """
    TODO 1: 在网格世界中执行一步动作，返回新位置（越界则贴墙）。

    动作编码: 0=上(行-1), 1=下(行+1), 2=左(列-1), 3=右(列+1)

    参数:
        pos: (row, col) 当前位置
        action: 0/1/2/3
        size: 网格边长

    返回:
        (new_row, new_col)

    预期行为:
        step_grid((3,3), 0) → (2,3)
        step_grid((0,0), 0) → (0,0)   # 贴墙上界
        step_grid((5,5), 3) → (5,5)   # 贴墙右界
    """
    r, c = pos
    # TODO: 根据 action 更新 r/c，并用 max/min 裁剪到 [0, size-1]
    return (r, c)  # → TODO: 返回更新后的位置


# ============================================================================
# 已提供：动作编码器 / 动态模型
# ============================================================================

class ActionEncoder(nn.Module):
    def __init__(self, grid_size: int, latent_dim: int = 8):
        super().__init__()
        in_dim = grid_size * grid_size * 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, latent_dim),
        )

    def forward(self, frame_t, frame_tp1):
        B = frame_t.size(0)
        x = torch.cat([frame_t.view(B, -1), frame_tp1.view(B, -1)], dim=1)
        return self.net(x)


class DynamicsModel(nn.Module):
    def __init__(self, grid_size: int, latent_dim: int):
        super().__init__()
        self.grid_size = grid_size
        in_dim = grid_size * grid_size + latent_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, grid_size * grid_size),
        )

    def forward(self, frame_t, action_latent):
        B = frame_t.size(0)
        x = torch.cat([frame_t.view(B, -1), action_latent], dim=1)
        return self.net(x)


# ============================================================================
# TODO 2: 实现 VectorQuantizer.forward() —— 向量量化
# ============================================================================

class VectorQuantizer(nn.Module):
    """
    TODO 2: 实现向量量化层的前向传播。

    步骤:
      1. 计算 z 到每个码字的欧氏距离: torch.cdist(z.unsqueeze(1), codebook.unsqueeze(0)).squeeze(1)
      2. code_idx = dist.argmin(dim=1)
      3. z_q = codebook[code_idx]
      4. codebook_loss = MSE(z_q, z.detach())
         commitment_loss = MSE(z, z_q.detach())
         vq_loss = codebook_loss + 0.25 * commitment_loss
      5. 直通估计器: z_q_st = z + (z_q - z).detach()
         （前向用量化值，反向梯度直接传给 z，绕过不可微的 argmin）
    """

    def __init__(self, n_codes: int, latent_dim: int):
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(n_codes, latent_dim) * 0.5)
        self.n_codes = n_codes

    def forward(self, z: torch.Tensor):
        """
        参数: z (B, latent_dim)
        返回: z_q_st, code_idx, vq_loss
        """
        # TODO: 按上面 5 步实现
        z_q_st = z
        code_idx = torch.zeros(z.size(0), dtype=torch.long)
        vq_loss = torch.tensor(0.0)
        return z_q_st, code_idx, vq_loss


# ============================================================================
# TODO 3: 实现 compute_next_frame_loss()
# ============================================================================

def compute_next_frame_loss(frame_t, frame_tp1, action_encoder, vq, dynamics):
    """
    TODO 3: 组装"潜在动作编码 → 量化 → 预测下一帧"的完整损失。

    步骤:
      1. z = action_encoder(frame_t, frame_tp1)
      2. z_q, code_idx, vq_loss = vq(z)
      3. logits = dynamics(frame_t, z_q)
      4. target_idx = frame_tp1.view(B, -1).argmax(dim=1)   # 智能体所在格子
      5. recon_loss = F.cross_entropy(logits, target_idx)
      6. total_loss = recon_loss + vq_loss

    返回: total_loss, recon_loss, vq_loss, code_idx
    """
    B = frame_t.size(0)
    # TODO: 按上面 6 步实现
    total_loss = None
    recon_loss = None
    vq_loss = None
    code_idx = None
    return total_loss, recon_loss, vq_loss, code_idx


# ============================================================================
# 数据生成与训练验证（已提供）
# ============================================================================

def make_transition_dataset(n_transitions: int, size: int = GRID_SIZE):
    frames_t = np.zeros((n_transitions, size, size), dtype=np.float32)
    frames_tp1 = np.zeros((n_transitions, size, size), dtype=np.float32)
    true_actions = np.zeros(n_transitions, dtype=np.int64)
    for i in range(n_transitions):
        pos = (np.random.randint(1, size - 1), np.random.randint(1, size - 1))
        action = np.random.randint(0, 4)
        new_pos = step_grid(pos, action, size)
        # 若 step_grid 未实现，new_pos==pos，后续训练会失败——这是预期的提示
        frames_t[i] = make_grid_frame(pos, size)
        frames_tp1[i] = make_grid_frame(new_pos, size)
        true_actions[i] = action
    return frames_t, frames_tp1, true_actions


def check_step_grid():
    """快速自检 TODO 1。"""
    tests = [
        ((3, 3), 0, (2, 3)),
        ((3, 3), 1, (4, 3)),
        ((3, 3), 2, (3, 2)),
        ((3, 3), 3, (3, 4)),
        ((0, 0), 0, (0, 0)),
        ((5, 5), 3, (5, 5)),
    ]
    ok = True
    for pos, a, expected in tests:
        got = step_grid(pos, a)
        if got != expected:
            print(f'  [TODO1 失败] step_grid({pos}, {a}) = {got}, 期望 {expected}')
            ok = False
    if ok:
        print('  [TODO1 通过] step_grid 行为正确')
    return ok


def train_and_check(n_steps=400, batch_size=64):
    frames_t, frames_tp1, true_actions = make_transition_dataset(3000)
    # 若 step_grid 未实现，几乎所有帧对都相同，准确率无法提升
    same_ratio = (frames_t == frames_tp1).all(axis=(1, 2)).mean()
    if same_ratio > 0.9:
        print(f'[提示] TODO 1 (step_grid) 似乎未实现：{same_ratio:.0%} 的转移前后帧相同。')
        return None

    action_encoder = ActionEncoder(GRID_SIZE, 8)
    vq = VectorQuantizer(4, 8)
    dynamics = DynamicsModel(GRID_SIZE, 8)
    optimizer = torch.optim.Adam(
        list(action_encoder.parameters()) + list(vq.parameters()) + list(dynamics.parameters()),
        lr=1e-3)

    frames_t_t = torch.from_numpy(frames_t).float()
    frames_tp1_t = torch.from_numpy(frames_tp1).float()
    n_data = frames_t.shape[0]
    loss_history, acc_history = [], []

    for step in range(n_steps):
        idx = np.random.choice(n_data, batch_size, replace=False)
        result = compute_next_frame_loss(
            frames_t_t[idx], frames_tp1_t[idx], action_encoder, vq, dynamics)
        total_loss, recon_loss, vq_loss, code_idx = result
        if total_loss is None:
            print('[提示] TODO 3 (compute_next_frame_loss) 尚未实现。')
            return None

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        target_idx = frames_tp1_t[idx].view(batch_size, -1).argmax(dim=1)
        with torch.no_grad():
            logits = dynamics(frames_t_t[idx], vq(action_encoder(frames_t_t[idx], frames_tp1_t[idx]))[0])
            acc = (logits.argmax(1) == target_idx).float().mean().item()
        loss_history.append(total_loss.item())
        acc_history.append(acc)
        if (step + 1) % 100 == 0:
            print(f'  Step {step+1}/{n_steps}: loss={total_loss.item():.4f}, acc={acc:.3f}')

    # 粗略检查 VQ 是否真的在用不同码字
    with torch.no_grad():
        z = action_encoder(frames_t_t[:500], frames_tp1_t[:500])
        _, codes, _ = vq(z)
        n_unique = len(torch.unique(codes))
    if n_unique <= 1:
        print('[提示] TODO 2 (VectorQuantizer) 似乎未正确实现：只用到了 1 个码字。')
    else:
        print(f'  [TODO2 检查] 量化后使用了 {n_unique}/4 个不同码字')

    return loss_history, acc_history


def main():
    print('\n' + '=' * 60)
    print('  wm06 Genie 练习：补全 step_grid / VQ / next_frame_loss')
    print('=' * 60)
    set_seed(42)

    print('\n检查 TODO 1...')
    check_step_grid()

    print('\n开始训练玩具潜在动作模型...')
    result = train_and_check()
    if result is not None:
        loss_history, acc_history = result
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        axes[0].plot(loss_history)
        axes[0].set_title('总损失')
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(acc_history)
        axes[1].set_title('下一帧预测准确率')
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(_IMAGES, 'exercise_genie_curves.png'), dpi=120, bbox_inches='tight')
        plt.close()
        final_acc = np.mean(acc_history[-30:])
        print(f'\n最终准确率: {final_acc:.1%}')
        if final_acc > 0.6:
            print('恭喜！三个 TODO 实现基本正确。')
        else:
            print('准确率偏低，请检查 VQ 直通估计器与损失组装。')
    print('\n完成！\n')


if __name__ == '__main__':
    main()
