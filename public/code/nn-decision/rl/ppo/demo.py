# -*- coding: utf-8 -*-
"""
=== PPO 最小演示：裁剪目标 + GAE vs REINFORCE ===
玩具 MDP「走廊平衡」：在一条奇数格走廊上左右走，越靠近中心奖励越高，
掉到两端失败。对比两种 on-policy 更新：
  1. REINFORCE：整条 Monte Carlo 回报，无裁剪，每批只走一步梯度
  2. PPO-Clip + GAE：同一批轨迹重复 K 个 epoch，比率超出 [1-ε,1+ε] 就封顶
同时画出正文里的信任域、裁剪曲线、GAE 与训练环。
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
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)

SEED = 42
N_POS = 9
MAX_STEPS = 24
N_ITERS = 90
EPISODES_PER_ITER = 12
PPO_EPOCHS = 4
CLIP_EPS = 0.2
GAMMA = 0.97
LAM = 0.95
ENTROPY_COEF = 0.02
VF_COEF = 0.5
LR = 3e-3
HIDDEN = 32
DEVICE = torch.device('cpu')


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def _save(fig, name):
    path = os.path.join(_IMAGES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'图已保存: {path}')
    return path


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


# ---------------------------------------------------------------------------
# 玩具环境
# ---------------------------------------------------------------------------
class CorridorBalance:
    """一维走廊：中心最好，两端失败。动作为 0=左、1=右。"""

    def __init__(self, n_pos=N_POS, max_steps=MAX_STEPS):
        assert n_pos % 2 == 1
        self.n_pos = n_pos
        self.center = n_pos // 2
        self.max_steps = max_steps
        self.n_actions = 2
        self.pos = self.center
        self.t = 0

    def reset(self):
        self.pos = int(np.clip(self.center + np.random.randint(-1, 2), 0, self.n_pos - 1))
        self.t = 0
        return self.pos

    def step(self, action):
        self.pos += -1 if action == 0 else 1
        self.t += 1
        if self.pos < 0 or self.pos >= self.n_pos:
            self.pos = int(np.clip(self.pos, 0, self.n_pos - 1))
            return self.pos, -1.0, True
        dist = abs(self.pos - self.center)
        reward = 1.0 - dist / self.center
        done = self.t >= self.max_steps
        return self.pos, float(reward), done


def onehot(pos, n=N_POS):
    x = torch.zeros(n, device=DEVICE)
    x[int(pos)] = 1.0
    return x


class ActorCritic(nn.Module):
    def __init__(self, n_obs=N_POS, n_act=2, hidden=HIDDEN):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(n_obs, hidden), nn.Tanh())
        self.pi = nn.Linear(hidden, n_act)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)


def compute_gae(rewards, values, dones, last_value, gamma=GAMMA, lam=LAM):
    """从后往前累加 TD 残差，得到 GAE 优势与回报目标。"""
    t_len = len(rewards)
    adv = np.zeros(t_len, dtype=np.float64)
    gae = 0.0
    for t in reversed(range(t_len)):
        next_v = last_value if t == t_len - 1 else values[t + 1]
        next_nonterminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_v * next_nonterminal - values[t]
        gae = delta + gamma * lam * next_nonterminal * gae
        adv[t] = gae
    returns = adv + values
    return adv.astype(np.float32), returns.astype(np.float32)


def collect_batch(env, net, n_episodes):
    """用当前策略采若干回合，记录 logπ、价值、奖励。"""
    obs, acts, logps, rews, vals, dones, ep_rets = [], [], [], [], [], [], []
    net.eval()
    with torch.no_grad():
        for _ in range(n_episodes):
            pos = env.reset()
            ep_ret = 0.0
            while True:
                x = onehot(pos).unsqueeze(0)
                logits, v = net(x)
                dist = Categorical(logits=logits)
                a = dist.sample()
                logp = dist.log_prob(a)
                next_pos, r, done = env.step(int(a.item()))
                obs.append(onehot(pos))
                acts.append(a.squeeze(0))
                logps.append(logp.squeeze(0))
                rews.append(r)
                vals.append(float(v.item()))
                dones.append(done)
                ep_ret += r
                pos = next_pos
                if done:
                    ep_rets.append(ep_ret)
                    break
    last_v = 0.0
    return {
        'obs': torch.stack(obs),
        'acts': torch.stack(acts),
        'logp': torch.stack(logps),
        'rews': np.array(rews, dtype=np.float32),
        'vals': np.array(vals, dtype=np.float32),
        'dones': np.array(dones, dtype=np.bool_),
        'last_v': last_v,
        'ep_rets': ep_rets,
    }


def ppo_update(net, opt, batch, epochs=PPO_EPOCHS, clip=CLIP_EPS):
    adv, ret = compute_gae(batch['rews'], batch['vals'], batch['dones'], batch['last_v'])
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    adv_t = torch.tensor(adv, device=DEVICE)
    ret_t = torch.tensor(ret, device=DEVICE)
    old_logp = batch['logp'].detach()
    obs, acts = batch['obs'], batch['acts']
    net.train()
    last = {}
    for _ in range(epochs):
        logits, v = net(obs)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(acts)
        ratio = torch.exp(logp - old_logp)
        surr1 = ratio * adv_t
        surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv_t
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(v, ret_t)
        entropy = dist.entropy().mean()
        loss = policy_loss + VF_COEF * value_loss - ENTROPY_COEF * entropy
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        last = {
            'clip_frac': float(((ratio - 1.0).abs() > clip).float().mean().item()),
            'entropy': float(entropy.item()),
        }
    return last


def reinforce_update(net, opt, batch):
    """无裁剪、无 Critic：用折扣回报当权重，每批只更新一次。"""
    rews, dones = batch['rews'], batch['dones']
    returns = np.zeros_like(rews)
    g = 0.0
    for t in reversed(range(len(rews))):
        if dones[t]:
            g = 0.0
        g = rews[t] + GAMMA * g
        returns[t] = g
    ret = torch.tensor((returns - returns.mean()) / (returns.std() + 1e-8), device=DEVICE)
    net.train()
    logits, _ = net(batch['obs'])
    dist = Categorical(logits=logits)
    loss = -(dist.log_prob(batch['acts']) * ret).mean() - ENTROPY_COEF * dist.entropy().mean()
    opt.zero_grad()
    loss.backward()
    opt.step()


def train_algo(kind):
    set_seed(SEED + (0 if kind == 'ppo' else 1))
    env = CorridorBalance()
    net = ActorCritic().to(DEVICE)
    opt = optim.Adam(net.parameters(), lr=LR)
    curve = []
    extras = []
    for it in range(N_ITERS):
        batch = collect_batch(env, net, EPISODES_PER_ITER)
        mean_ret = float(np.mean(batch['ep_rets']))
        curve.append(mean_ret)
        if kind == 'ppo':
            info = ppo_update(net, opt, batch)
            extras.append(info['clip_frac'])
        else:
            reinforce_update(net, opt, batch)
            extras.append(0.0)
        if (it + 1) % 15 == 0:
            print(f'  {kind:9s} iter {it+1:3d}  平均回报 {mean_ret:.2f}')
    return np.array(curve), np.array(extras)


# ---------------------------------------------------------------------------
# 正文示意图
# ---------------------------------------------------------------------------
def draw_trust_region():
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 5.4)
    ax.axis('off')
    ax.text(5.75, 5.05, '从 REINFORCE 到信任域，再到 PPO 裁剪', ha='center',
            fontsize=14, fontweight='bold')
    _box(ax, 0.3, 2.4, 2.6, 1.8, 'REINFORCE\n$G_t\\,\\nabla\\log\\pi$\n步子野、方差大', '#F9D5D3', 10)
    _box(ax, 4.0, 2.4, 3.4, 1.8, 'TRPO 信任域\nKL($\\pi_{old}$ || $\\pi$) ≤ $\\delta$\n二阶，工程重', '#CDE7F0', 10)
    _box(ax, 8.4, 2.4, 2.8, 1.8, 'PPO-Clip\n$r$ 锁在 $[1-\\varepsilon,1+\\varepsilon]$\n一阶就能用', '#D9EAD3', 10)
    _arrow(ax, 2.95, 3.3, 3.95, 3.3)
    _arrow(ax, 7.45, 3.3, 8.35, 3.3)
    ax.text(5.75, 1.35, 'on-policy 数据只在旧策略邻域里可信；裁剪是便宜的邻域门闩。',
            ha='center', fontsize=10, color='#444444')
    ax.text(5.75, 0.55, '后面 RLHF 直接用右边这一格，不必再推一遍 $L^{CLIP}$。',
            ha='center', fontsize=10, color='#444444')
    _save(fig, 'ppo-01-trust-region.png')


def draw_clip_curves():
    r = np.linspace(0.2, 2.4, 400)
    eps = 0.2
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    for ax, A, title, color in (
        (axes[0], 1.0, r'好动作 $\hat{A}>0$：想提高 $\pi$', '#2563eb'),
        (axes[1], -1.0, r'坏动作 $\hat{A}<0$：想压低 $\pi$', '#dc2626'),
    ):
        unclip = r * A
        clipped = np.clip(r, 1 - eps, 1 + eps) * A
        obj = np.minimum(unclip, clipped)
        ax.plot(r, unclip, '--', color='#999999', label='未裁剪 $r\\hat{A}$')
        ax.plot(r, obj, color=color, lw=2.4, label='$L^{CLIP}$')
        ax.axvline(1 - eps, color='#888888', ls=':', lw=1)
        ax.axvline(1 + eps, color='#888888', ls=':', lw=1)
        ax.axvline(1.0, color='#bbbbbb', lw=0.8)
        ax.set_xlabel(r'概率比 $r_t(\theta)$')
        ax.set_ylabel('目标值')
        ax.set_title(title)
        ax.legend(loc='best', fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle(r'PPO 裁剪：$\min(r\hat{A},\;\mathrm{clip}(r,1-\varepsilon,1+\varepsilon)\hat{A})$，不能靠极端 $r$ 刷分',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    _save(fig, 'ppo-02-clip-curves.png')


def draw_gae():
    fig, ax = plt.subplots(figsize=(11.2, 4.8))
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 5.0)
    ax.axis('off')
    ax.text(5.6, 4.65, r'GAE：$\lambda$ 在单步 TD 与整条 Monte Carlo 之间滑动',
            ha='center', fontsize=13, fontweight='bold')
    _box(ax, 0.35, 2.2, 3.1, 1.7, '$\\lambda=0$\n$\\hat{A}_t=\\delta_t$\n低方差、高偏差', '#CDE7F0', 10)
    _box(ax, 4.05, 2.2, 3.1, 1.7, '$\\lambda\\approx 0.95$\n实践默认\n多步 TD 折中', '#D9EAD3', 10)
    _box(ax, 7.75, 2.2, 3.1, 1.7, '$\\lambda=1$\n接近 $G_t-V$\n高方差、低偏差', '#FDE8D7', 10)
    _arrow(ax, 3.5, 3.05, 4.0, 3.05)
    _arrow(ax, 7.2, 3.05, 7.7, 3.05)
    ax.text(5.6, 1.2,
            r'$\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$，  $\hat{A}_t=\sum_k(\gamma\lambda)^k\delta_{t+k}$',
            ha='center', fontsize=11)
    ax.text(5.6, 0.45, 'Critic 学 $V_\\phi$，给这条滑动尺提供自举；GRPO 会把尺子换成「组内相对分」。',
            ha='center', fontsize=10, color='#444444')
    _save(fig, 'ppo-03-gae.png')


def draw_loop():
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 4.8)
    ax.axis('off')
    ax.text(5.75, 4.4, 'PPO 一次迭代：采样 → GAE → K 次裁剪更新',
            ha='center', fontsize=13, fontweight='bold')
    _box(ax, 0.3, 1.8, 2.4, 1.6, '1. 用 $\\pi_{old}$\n采轨迹', '#E8E8E8', 10)
    _box(ax, 3.1, 1.8, 2.5, 1.6, '2. 算 GAE\n标准化 $\\hat{A}$', '#CDE7F0', 10)
    _box(ax, 6.0, 1.8, 2.6, 1.6, '3. K 个 epoch\nmin + clip', '#D9EAD3', 10)
    _box(ax, 9.0, 1.8, 2.2, 1.6, '4. $\\theta_{old}$ ← $\\theta$\n再采样', '#FDE8D7', 10)
    _arrow(ax, 2.75, 2.6, 3.05, 2.6)
    _arrow(ax, 5.65, 2.6, 5.95, 2.6)
    _arrow(ax, 8.65, 2.6, 8.95, 2.6)
    ax.annotate('', xy=(1.5, 1.75), xytext=(10.1, 1.75),
                arrowprops=dict(arrowstyle='-|>', color='#888888',
                                connectionstyle='arc3,rad=-0.28', lw=1.4))
    ax.text(5.75, 0.45, '同一批 on-policy 数据能反复用，靠的就是第 3 步那道裁剪门闩。',
            ha='center', fontsize=10, color='#444444')
    _save(fig, 'ppo-04-loop.png')


def draw_training(ppo_curve, rf_curve, clip_frac):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
    axes[0].plot(rf_curve, color='#f97316', lw=1.8, label='REINFORCE')
    axes[0].plot(ppo_curve, color='#2563eb', lw=1.8, label='PPO-Clip + GAE')
    axes[0].set_xlabel('迭代')
    axes[0].set_ylabel('回合平均回报')
    axes[0].set_title('走廊平衡：裁剪后的更新更稳')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(clip_frac, color='#7c3aed', lw=1.6)
    axes[1].set_xlabel('迭代')
    axes[1].set_ylabel('比率越出 $[1\\pm\\varepsilon]$ 的比例')
    axes[1].set_title('PPO 的 clip fraction')
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, 'ppo_vs_reinforce.png')


def main():
    print('=' * 64)
    print('PPO：裁剪目标 + GAE（走廊平衡）')
    print('=' * 64)
    print('\n[1] 绘制正文示意图…')
    draw_trust_region()
    draw_clip_curves()
    draw_gae()
    draw_loop()

    print('\n[2] 训练 REINFORCE…')
    rf_curve, _ = train_algo('reinforce')
    print('\n[3] 训练 PPO…')
    ppo_curve, clip_frac = train_algo('ppo')
    print(f'\n最后 15 次迭代平均回报  REINFORCE {rf_curve[-15:].mean():.2f}  PPO {ppo_curve[-15:].mean():.2f}')
    draw_training(ppo_curve, rf_curve, clip_frac)
    print('\n看 ppo-02-clip-curves.png：A>0 时曲线在 1+ε 处封顶，这就是「近端」。')


if __name__ == '__main__':
    main()
