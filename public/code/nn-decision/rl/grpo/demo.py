# -*- coding: utf-8 -*-
"""
=== GRPO 最小演示：组内相对优势，没有 Critic ===
玩具「口算题」：每道题是一个离散动作（猜答案 0..N）。
同一题采 G 个答案，用组内 z-score 当优势，再套 PPO 的 clip。
对比：
  - 无基线 REINFORCE（绝对 0/1 奖励）
  - GRPO（组相对）
运行: python demo.py
"""
import os
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)

SEED = 42
N_A, N_B = 6, 6
N_ANS = N_A + N_B + 1          # 答案 0 .. 12
N_Q = N_A * N_B                # 36 道 a+b
GROUP = 8
CLIP_EPS = 0.2
KL_BETA = 0.02
LR = 0.08
N_STEPS = 120
DEVICE = torch.device('cpu')


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def _save(fig, name):
    path = os.path.join(_IMAGES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'图已保存: {path}')


def _box(ax, x, y, w, h, text, color, fs=10):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.04',
        linewidth=1.4, edgecolor='#333333', facecolor=color,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fs, fontweight='bold', color='#222222')


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=14,
        linewidth=1.5, color='#333333',
    ))


def build_bank():
    """题目 q 编码为 a*N_B+b，正确答案是 a+b。"""
    qs, gold = [], []
    for a in range(N_A):
        for b in range(N_B):
            qs.append(a * N_B + b)
            gold.append(a + b)
    return np.array(qs), np.array(gold)


class AnswerPolicy(nn.Module):
    """每道题一组 logits，相当于「按题查表的小 LLM」。"""

    def __init__(self, n_q=N_Q, n_ans=N_ANS):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_q, n_ans))

    def dist(self, q_idx):
        return Categorical(logits=self.logits[q_idx])


def group_advantages(rewards, eps=1e-8):
    """组内 z-score；方差过小则整组优势为 0（没有学习信号）。"""
    r = np.asarray(rewards, dtype=np.float64)
    std = r.std()
    if std < 1e-6:
        return np.zeros_like(r, dtype=np.float32)
    return ((r - r.mean()) / (std + eps)).astype(np.float32)


def accuracy(policy, gold):
    with torch.no_grad():
        pred = policy.logits.argmax(dim=-1).cpu().numpy()
    return float((pred == gold).mean())


def grpo_step(policy, ref_logits, opt, q_idx, gold, group=GROUP):
    """对一道题采 G 个答案，组相对优势 + clip + KL(π||π_ref)。"""
    dist_old = Categorical(logits=policy.logits[q_idx].detach())
    answers, logp_old, rewards = [], [], []
    for _ in range(group):
        a = dist_old.sample()
        answers.append(int(a.item()))
        logp_old.append(dist_old.log_prob(a))
        rewards.append(1.0 if answers[-1] == int(gold) else 0.0)
    adv = group_advantages(rewards)
    if np.allclose(adv, 0):
        return float(np.mean(rewards)), True
    answers_t = torch.tensor(answers, device=DEVICE)
    old_lp = torch.stack(logp_old).detach()
    adv_t = torch.tensor(adv, device=DEVICE)
    dist = policy.dist(q_idx)
    new_lp = dist.log_prob(answers_t)
    ratio = torch.exp(new_lp - old_lp)
    surr1 = ratio * adv_t
    surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_t
    clip_loss = -torch.min(surr1, surr2).mean()
    ref = Categorical(logits=ref_logits[q_idx])
    kl = torch.distributions.kl.kl_divergence(dist, ref)
    loss = clip_loss + KL_BETA * kl
    opt.zero_grad()
    loss.backward()
    opt.step()
    return float(np.mean(rewards)), False


def reinforce_step(policy, opt, q_idx, gold, group=GROUP):
    """同一组采样，但用绝对 0/1 奖励、无基线、无 clip。"""
    dist = policy.dist(q_idx)
    loss = 0.0
    hits = []
    for _ in range(group):
        a = dist.sample()
        r = 1.0 if int(a.item()) == int(gold) else 0.0
        hits.append(r)
        loss = loss - dist.log_prob(a) * r
    loss = loss / group
    opt.zero_grad()
    loss.backward()
    opt.step()
    return float(np.mean(hits))


def train(kind, qs, gold):
    set_seed(SEED + (0 if kind == 'grpo' else 7))
    policy = AnswerPolicy().to(DEVICE)
    ref_logits = policy.logits.detach().clone()
    opt = optim.SGD(policy.parameters(), lr=LR)
    acc_curve, skip_frac = [], []
    skipped = 0
    for step in range(N_STEPS):
        i = int(np.random.randint(0, len(qs)))
        if kind == 'grpo':
            _, skip = grpo_step(policy, ref_logits, opt, i, gold[i])
            skipped += int(skip)
        else:
            reinforce_step(policy, opt, i, gold[i])
        acc_curve.append(accuracy(policy, gold))
        skip_frac.append(skipped / (step + 1))
        if (step + 1) % 20 == 0:
            print(f'  {kind:10s} step {step+1:3d}  正确率 {acc_curve[-1]:.2f}')
    return np.array(acc_curve), np.array(skip_frac)


# ---------------------------------------------------------------------------
# 正文示意图
# ---------------------------------------------------------------------------
def draw_group():
    fig, ax = plt.subplots(figsize=(11.4, 5.0))
    ax.set_xlim(0, 11.4)
    ax.set_ylim(0, 5.2)
    ax.axis('off')
    ax.text(5.7, 4.85, '同一 prompt 采一组输出，用组均值当基线',
            ha='center', fontsize=14, fontweight='bold')
    _box(ax, 0.35, 2.9, 2.4, 1.4, '问题 $q$\n（一道数学题）', '#E8E8E8', 10)
    _arrow(ax, 2.8, 3.6, 3.35, 3.6)
    _box(ax, 3.4, 2.55, 4.4, 2.1,
         '旧策略 $\\pi_{old}$ 采样 $G$ 条\n$o_1,o_2,\\ldots,o_G$\n验证器打分 $r_1\\ldots r_G$',
         '#CDE7F0', 10)
    _arrow(ax, 7.85, 3.6, 8.4, 3.6)
    _box(ax, 8.45, 2.7, 2.6, 1.8,
         '$\\hat{A}_i=(r_i-\\mu)/\\sigma$\n没有 $V_\\phi$',
         '#D9EAD3', 10)
    ax.text(5.7, 1.55, '比组内平均好的答案 $\\hat{A}>0$，差的 $<0$。全对或全错时 $\\sigma\\approx 0$，这一组没有梯度。',
            ha='center', fontsize=10, color='#444444')
    ax.text(5.7, 0.7, '这就是 s20「优势 = 比平均好多少」，只是平均来自并列的 G 个样本，而不是 Critic。',
            ha='center', fontsize=10, color='#444444')
    _save(fig, 'grpo-01-group.png')


def draw_vs_ppo():
    fig, ax = plt.subplots(figsize=(11.4, 5.2))
    ax.set_xlim(0, 11.4)
    ax.set_ylim(0, 5.4)
    ax.axis('off')
    ax.text(5.7, 5.05, 'GRPO 与 PPO：同一套裁剪，优势来源不同',
            ha='center', fontsize=14, fontweight='bold')
    _box(ax, 0.4, 1.7, 5.0, 2.9,
         'PPO\n\n比率 $r_t$ + clip\n优势 = GAE + Critic $V_\\phi$\n一条轨迹也可以更新',
         '#CDE7F0', 11)
    _box(ax, 6.0, 1.7, 5.0, 2.9,
         'GRPO\n\n比率 $r_t$ + clip（同一套）\n优势 = 组内 z-score\n每个 prompt 必须成组',
         '#FDE8D7', 11)
    ax.text(5.7, 0.7, 'DeepSeekMath / R1 选右边：长思维链上训 $V$ 又贵又不稳，组采样反正都要做。',
            ha='center', fontsize=10, color='#444444')
    _save(fig, 'grpo-02-vs-ppo.png')


def draw_verifier():
    fig, ax = plt.subplots(figsize=(11.4, 4.8))
    ax.set_xlim(0, 11.4)
    ax.set_ylim(0, 5.0)
    ax.axis('off')
    ax.text(5.7, 4.6, '可验证奖励：对错由规则说了算，组内比相对高低',
            ha='center', fontsize=13, fontweight='bold')
    _box(ax, 0.35, 2.3, 2.5, 1.5, '采样的\n推理链 $o_i$', '#E8E8E8', 10)
    _arrow(ax, 2.9, 3.05, 3.4, 3.05)
    _box(ax, 3.45, 2.3, 2.7, 1.5, '验证器\n最终答案 / 单测', '#D9EAD3', 10)
    _arrow(ax, 6.2, 3.05, 6.7, 3.05)
    _box(ax, 6.75, 2.3, 2.0, 1.5, '标量 $r_i$\n对=1 错=0', '#CDE7F0', 10)
    _arrow(ax, 8.8, 3.05, 9.25, 3.05)
    _box(ax, 9.3, 2.3, 1.8, 1.5, '组内\n$z$-score', '#FDE8D7', 10)
    ax.text(5.7, 1.2, '和 DPO 不同：DPO 吃离线偏好对 $(y_w,y_l)$；GRPO 是 on-policy 组采样 + 可自动打分。',
            ha='center', fontsize=10, color='#444444')
    ax.text(5.7, 0.45, 'RLHF 章里：有人类偏好走 PPO+RM；有验证器的推理任务可以换 GRPO。',
            ha='center', fontsize=10, color='#444444')
    _save(fig, 'grpo-03-verifier.png')


def draw_roadmap():
    fig, ax = plt.subplots(figsize=(11.4, 4.6))
    ax.set_xlim(0, 11.4)
    ax.set_ylim(0, 4.8)
    ax.axis('off')
    ax.text(5.7, 4.4, '接到 RLHF 之前：优化器已经齐了',
            ha='center', fontsize=13, fontweight='bold')
    _box(ax, 0.3, 2.0, 2.5, 1.5, 'PPO\n裁剪 + GAE', '#CDE7F0', 11)
    _box(ax, 3.2, 2.0, 2.5, 1.5, 'GRPO\n组相对优势', '#FDE8D7', 11)
    _box(ax, 6.1, 2.0, 2.5, 1.5, 'SFT / RM / HHH\n（下一章）', '#D9EAD3', 11)
    _box(ax, 9.0, 2.0, 2.1, 1.5, 'RLHF\n拼起来', '#E8D5F2', 11)
    _arrow(ax, 2.85, 2.75, 3.15, 2.75)
    _arrow(ax, 5.75, 2.75, 6.05, 2.75)
    _arrow(ax, 8.65, 2.75, 8.95, 2.75)
    ax.text(5.7, 0.85, 'RLHF 不再推导 $L^{CLIP}$ 或 $z$-score，只说明这些符号在 token 序列上怎么对应。',
            ha='center', fontsize=10, color='#444444')
    _save(fig, 'grpo-04-roadmap.png')


def draw_training(grpo_acc, rf_acc, skip_frac):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
    axes[0].plot(rf_acc, color='#f97316', lw=1.8, label='REINFORCE（无基线）')
    axes[0].plot(grpo_acc, color='#2563eb', lw=1.8, label='GRPO（组相对）')
    axes[0].set_xlabel('更新步（每步随机抽一题）')
    axes[0].set_ylabel('全部题目的贪心正确率')
    axes[0].set_title('口算题：组内相对分比绝对 0/1 稳')
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(skip_frac, color='#7c3aed', lw=1.6)
    axes[1].set_xlabel('更新步')
    axes[1].set_ylabel('组内奖励全相同而被跳过的比例')
    axes[1].set_title('GRPO 的「没方差就没信号」')
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, 'grpo_vs_reinforce.png')


def main():
    print('=' * 64)
    print('GRPO：组相对策略优化（口算题玩具）')
    print('=' * 64)
    print('\n[1] 绘制正文示意图…')
    draw_group()
    draw_vs_ppo()
    draw_verifier()
    draw_roadmap()

    qs, gold = build_bank()
    print(f'\n[2] {N_Q} 道 a+b 题，答案空间 0..{N_ANS-1}，每题一组 G={GROUP}')
    print('\n训练 REINFORCE（无基线）…')
    rf_acc, _ = train('reinforce', qs, gold)
    print('\n训练 GRPO…')
    grpo_acc, skip_frac = train('grpo', qs, gold)
    print(f'\n最终正确率  REINFORCE {rf_acc[-1]:.2f}  GRPO {grpo_acc[-1]:.2f}')
    draw_training(grpo_acc, rf_acc, skip_frac)
    print('\n看左图：组内 z-score 把「全错的题」和「全对的题」都当成零信号，梯度更干净。')


if __name__ == '__main__':
    main()
