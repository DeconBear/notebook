# -*- coding: utf-8 -*-
"""
=== Dreamer 演示：想象 Actor-Critic ===
1) 倒立摆（与 PETS/LeWM 同一物理）：潜动力学 + V_λ，在想象里学力矩策略。
2) 离散走廊附录：对照「想象多练」相对 model-free REINFORCE。
运行: python demo.py
"""

import os
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)

DEVICE = torch.device('cpu')


def set_seed(seed: int = 42):
    """设置随机种子，保证实验可复现。"""
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================================
# 第一部分：玩具离散网格 MDP（CorridorBalance）
# ============================================================================

class CorridorBalance:
    """
    一维走廊平衡任务：Agent 在位置 {0,1,...,N-1} 上左右移动，
    目标是尽量靠近中心。每步奖励 = 1 - |pos - center| / center，
    到达两端视为失败（reward=-1, done）。

    这是一个离散、可快速交互的玩具 MDP，用来演示 Dreamer 的
    "在想象中学习策略"，而不需要完整的像素级世界模型。
    """

    def __init__(self, n_pos: int = 11, max_steps: int = 30):
        assert n_pos % 2 == 1, "位置数应为奇数，保证有唯一中心"
        self.n_pos = n_pos
        self.center = n_pos // 2
        self.max_steps = max_steps
        self.n_actions = 2                                        # 0=左, 1=右
        self.pos = self.center
        self.t = 0

    def reset(self) -> int:
        """重置到中心附近的随机位置。"""
        self.pos = self.center + np.random.randint(-1, 2)
        self.pos = int(np.clip(self.pos, 0, self.n_pos - 1))
        self.t = 0
        return self.pos

    def step(self, action: int) -> Tuple[int, float, bool]:
        """
        执行一步。

        参数:
            action: 0=向左, 1=向右

        返回:
            next_pos, reward, done
        """
        delta = -1 if action == 0 else 1
        self.pos = int(np.clip(self.pos + delta, 0, self.n_pos - 1))
        self.t += 1

        # 掉出两端（贴边）→ 失败
        if self.pos == 0 or self.pos == self.n_pos - 1:
            return self.pos, -1.0, True

        # 越靠近中心奖励越高
        reward = 1.0 - abs(self.pos - self.center) / self.center
        done = self.t >= self.max_steps
        return self.pos, float(reward), done

    def one_hot(self, pos: int) -> np.ndarray:
        """把位置编码为 one-hot 向量（作为世界模型的观测）。"""
        obs = np.zeros(self.n_pos, dtype=np.float32)
        obs[pos] = 1.0
        return obs


# ============================================================================
# 第二部分：微型世界模型 + Actor-Critic（Dreamer 风格）
# ============================================================================

class TinyWorldModel(nn.Module):
    """
    微型潜空间世界模型：把 one-hot 观测编码为潜状态，再预测
    下一潜状态、奖励、done。这是 Dreamer 世界模型的极简版
    （去掉了 RSSM 的随机状态，只用确定性 MLP，便于 CPU 快速演示）。
    """

    def __init__(self, obs_dim: int, act_dim: int, latent_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 32), nn.ELU(),
            nn.Linear(32, latent_dim),
        )
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + act_dim, 32), nn.ELU(),
            nn.Linear(32, latent_dim),
        )
        self.reward_head = nn.Linear(latent_dim, 1)
        self.done_head = nn.Linear(latent_dim, 1)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """观测 → 潜状态。"""
        return self.encoder(obs)

    def imagine_step(self, z: torch.Tensor, action_onehot: torch.Tensor):
        """
        潜空间单步想象：z_t, a_t → z_{t+1}, r̂, d̂

        返回:
            z_next, reward, done_logit
        """
        z_next = self.dynamics(torch.cat([z, action_onehot], dim=-1))
        reward = self.reward_head(z_next).squeeze(-1)
        done_logit = self.done_head(z_next).squeeze(-1)
        return z_next, reward, done_logit


class Actor(nn.Module):
    """策略网络：潜状态 → 动作 logits。"""

    def __init__(self, latent_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ELU(),
            nn.Linear(32, n_actions),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    def act(self, z: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """采样动作并返回 log_prob。"""
        logits = self.forward(z)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action)


class Critic(nn.Module):
    """价值网络：潜状态 → V(z)。"""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ELU(),
            nn.Linear(32, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


# ============================================================================
# 第三部分：收集真实经验 + 训练世界模型
# ============================================================================

def collect_episodes(env: CorridorBalance, actor: Actor, world: TinyWorldModel,
                     n_episodes: int, epsilon: float = 0.2):
    """
    用当前策略（带 ε-贪婪探索）与真实环境交互，收集 (obs, act, rew, next_obs, done)。

    返回:
        列表，每个元素是一条 episode 的字典
    """
    episodes = []
    for _ in range(n_episodes):
        pos = env.reset()
        obs = env.one_hot(pos)
        traj = {'obs': [], 'act': [], 'rew': [], 'next_obs': [], 'done': []}
        done = False
        while not done:
            obs_t = torch.from_numpy(obs).unsqueeze(0)
            with torch.no_grad():
                z = world.encode(obs_t)
                if np.random.random() < epsilon:
                    action = np.random.randint(env.n_actions)
                else:
                    action, _ = actor.act(z)
            next_pos, reward, done = env.step(action)
            next_obs = env.one_hot(next_pos)
            traj['obs'].append(obs)
            traj['act'].append(action)
            traj['rew'].append(reward)
            traj['next_obs'].append(next_obs)
            traj['done'].append(float(done))
            obs = next_obs
        episodes.append(traj)
    return episodes


def train_world_model(world: TinyWorldModel, optimizer: optim.Optimizer,
                      episodes: list, n_actions: int, n_steps: int = 40) -> float:
    """用真实经验训练世界模型（预测 next latent、reward、done）。"""
    # 展平所有转移
    obs, act, rew, next_obs, done = [], [], [], [], []
    for ep in episodes:
        obs.extend(ep['obs'])
        act.extend(ep['act'])
        rew.extend(ep['rew'])
        next_obs.extend(ep['next_obs'])
        done.extend(ep['done'])

    obs_t = torch.tensor(np.array(obs), dtype=torch.float32)
    next_obs_t = torch.tensor(np.array(next_obs), dtype=torch.float32)
    act_t = torch.tensor(act, dtype=torch.long)
    rew_t = torch.tensor(rew, dtype=torch.float32)
    done_t = torch.tensor(done, dtype=torch.float32)
    act_oh = F.one_hot(act_t, num_classes=n_actions).float()

    losses = []
    n = obs_t.shape[0]
    for _ in range(n_steps):
        idx = np.random.choice(n, size=min(64, n), replace=False)
        z = world.encode(obs_t[idx])
        z_next_target = world.encode(next_obs_t[idx]).detach()     # 目标潜状态（停止梯度）
        z_next_pred, r_pred, d_logit = world.imagine_step(z, act_oh[idx])

        loss_dyn = F.mse_loss(z_next_pred, z_next_target)
        loss_rew = F.mse_loss(r_pred, rew_t[idx])
        loss_done = F.binary_cross_entropy_with_logits(d_logit, done_t[idx])
        loss = loss_dyn + loss_rew + loss_done

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return float(np.mean(losses))


# ============================================================================
# 第四部分：在想象中训练 Actor-Critic（Dreamer 核心）
# ============================================================================

def imagine_and_train_actor_critic(
    world: TinyWorldModel,
    actor: Actor,
    critic: Critic,
    actor_opt: optim.Optimizer,
    critic_opt: optim.Optimizer,
    start_obs: torch.Tensor,
    n_actions: int,
    horizon: int = 8,
    gamma: float = 0.95,
    n_updates: int = 20,
) -> Tuple[float, float]:
    """
    Dreamer 核心循环：从真实观测编码出起始潜状态，然后在世界模型里
    "想象" horizon 步，用想象出的奖励训练 Actor-Critic。

    关键点：世界模型参数在此阶段冻结（stop-grad），只更新 Actor/Critic。
    """
    actor_losses, critic_losses = [], []

    for _ in range(n_updates):
        # ---- 采样一批起始观测 ----
        idx = np.random.choice(start_obs.shape[0], size=min(32, start_obs.shape[0]), replace=False)
        with torch.no_grad():
            z = world.encode(start_obs[idx])

        rewards, values, log_probs, dones = [], [], [], []
        for _t in range(horizon):
            logits = actor(z)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            act_oh = F.one_hot(action, num_classes=n_actions).float()

            # 世界模型想象一步（冻结）
            with torch.no_grad():
                z_next, r_pred, d_logit = world.imagine_step(z.detach(), act_oh)
                d_prob = torch.sigmoid(d_logit)

            v = critic(z)
            rewards.append(r_pred)
            values.append(v)
            log_probs.append(log_prob)
            dones.append(d_prob)
            z = z_next                                            # 继续在想象中滚动

        # ---- 从后往前算折扣回报（想象轨迹上的 Monte Carlo return）----
        returns = []
        G = critic(z).detach()                                    # bootstrap 用最后一步的价值
        for t in reversed(range(horizon)):
            cont = 1.0 - dones[t].detach()                        # 终止后不再累积
            G = rewards[t] + gamma * cont * G
            returns.insert(0, G)
        returns_t = torch.stack(returns)                          # (H, B)
        values_t = torch.stack(values)
        log_probs_t = torch.stack(log_probs)

        # ---- Critic 损失：让 V(z) 逼近想象回报 ----
        critic_loss = F.mse_loss(values_t, returns_t.detach())
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()

        # ---- Actor 损失：策略梯度，用 advantage = return - V ----
        advantage = (returns_t - values_t).detach()
        actor_loss = -(log_probs_t * advantage).mean()
        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()

        actor_losses.append(actor_loss.item())
        critic_losses.append(critic_loss.item())

    return float(np.mean(actor_losses)), float(np.mean(critic_losses))


# ============================================================================
# 第五部分：Model-free 基线（REINFORCE，只用真实经验）
# ============================================================================

def train_reinforce_baseline(env: CorridorBalance, n_episodes: int = 300,
                             lr: float = 3e-3, gamma: float = 0.95) -> List[float]:
    """
    纯 model-free REINFORCE 基线：不学世界模型，只用真实轨迹更新策略。
    用于对比 Dreamer 风格"想象多练"带来的样本效率提升。
    """
    obs_dim = env.n_pos
    policy = nn.Sequential(nn.Linear(obs_dim, 32), nn.ELU(), nn.Linear(32, env.n_actions))
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    returns_history = []

    for ep in range(n_episodes):
        pos = env.reset()
        log_probs, rewards = [], []
        done = False
        while not done:
            obs_t = torch.from_numpy(env.one_hot(pos)).unsqueeze(0)
            logits = policy(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            pos, reward, done = env.step(action.item())
            rewards.append(reward)

        # 计算回报
        G, returns = 0.0, []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns_t = torch.tensor(returns, dtype=torch.float32)
        if len(returns_t) > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
        loss = sum(-lp * G_t for lp, G_t in zip(log_probs, returns_t))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        returns_history.append(sum(rewards))

    return returns_history


# ============================================================================
# 第六部分：Dreamer 风格主训练循环
# ============================================================================

def train_dreamer_style(env: CorridorBalance, n_iters: int = 40,
                        episodes_per_iter: int = 8) -> Tuple[List[float], TinyWorldModel, Actor, Critic]:
    """
    Dreamer 风格训练：
      每轮 = 真实交互收集数据 → 更新世界模型 → 在想象中更新 Actor-Critic
    """
    obs_dim = env.n_pos
    n_actions = env.n_actions
    latent_dim = 16

    world = TinyWorldModel(obs_dim, n_actions, latent_dim)
    actor = Actor(latent_dim, n_actions)
    critic = Critic(latent_dim)

    world_opt = optim.Adam(world.parameters(), lr=3e-3)
    actor_opt = optim.Adam(actor.parameters(), lr=3e-3)
    critic_opt = optim.Adam(critic.parameters(), lr=3e-3)

    eval_returns = []
    all_start_obs = []

    for it in range(n_iters):
        # 1. 真实交互
        eps = max(0.05, 0.5 * (1 - it / n_iters))
        episodes = collect_episodes(env, actor, world, episodes_per_iter, epsilon=eps)
        for ep in episodes:
            all_start_obs.extend(ep['obs'])

        # 2. 更新世界模型
        wm_loss = train_world_model(world, world_opt, episodes, n_actions, n_steps=30)

        # 3. 在想象中更新 Actor-Critic
        start_obs_t = torch.tensor(np.array(all_start_obs[-500:]), dtype=torch.float32)
        a_loss, c_loss = imagine_and_train_actor_critic(
            world, actor, critic, actor_opt, critic_opt,
            start_obs_t, n_actions, horizon=8, n_updates=15,
        )

        # 4. 评估（无探索）
        eval_ret = evaluate_policy(env, actor, world, n_episodes=10)
        eval_returns.append(eval_ret)

        if (it + 1) % 10 == 0:
            print(f"  Iter {it+1:3d}/{n_iters}: eval_return={eval_ret:6.2f}, "
                  f"wm_loss={wm_loss:.4f}, actor_loss={a_loss:.4f}, critic_loss={c_loss:.4f}")

    return eval_returns, world, actor, critic


def evaluate_policy(env: CorridorBalance, actor: Actor, world: TinyWorldModel,
                    n_episodes: int = 10) -> float:
    """无探索地评估当前策略的平均回报。"""
    total = 0.0
    for _ in range(n_episodes):
        pos = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            obs_t = torch.from_numpy(env.one_hot(pos)).unsqueeze(0)
            with torch.no_grad():
                z = world.encode(obs_t)
                logits = actor(z)
                action = int(logits.argmax(dim=-1).item())
            pos, reward, done = env.step(action)
            ep_ret += reward
        total += ep_ret
    return total / n_episodes


# ============================================================================
# 第七部分：可视化
# ============================================================================

def plot_return_comparison(dreamer_returns: List[float], baseline_returns: List[float]):
    """对比 Dreamer 风格与 model-free REINFORCE 的学习曲线。"""
    fig, ax = plt.subplots(figsize=(9, 5))

    # Dreamer：每个 iter 的评估回报
    ax.plot(dreamer_returns, color='#2E86AB', linewidth=2, label='Dreamer 风格（想象中训练 Actor-Critic）')

    # Baseline：滑动平均对齐到相近的横轴尺度
    window = 20
    if len(baseline_returns) >= window:
        smooth = np.convolve(baseline_returns, np.ones(window) / window, mode='valid')
        # 把 baseline 的横轴缩放到与 dreamer iters 相近，便于视觉对比
        x_base = np.linspace(0, len(dreamer_returns) - 1, len(smooth))
        ax.plot(x_base, smooth, color='#C1666B', linewidth=2, linestyle='--',
                label=f'Model-free REINFORCE（滑动平均 window={window}）')

    ax.set_xlabel('训练轮次（Dreamer: iter；REINFORCE 横轴已对齐）', fontsize=10)
    ax.set_ylabel('评估 / Episode 回报', fontsize=10)
    ax.set_title('在想象中学习 vs 只用真实经验 —— 样本效率对比', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(_IMAGES_DIR, 'dreamer_vs_reinforce.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[可视化] 回报对比图已保存至 images/dreamer_vs_reinforce.png")


def plot_value_heatmap(world: TinyWorldModel, critic: Critic, env: CorridorBalance):
    """可视化 Critic 在每个位置上的价值估计。"""
    values = []
    for pos in range(env.n_pos):
        obs = torch.from_numpy(env.one_hot(pos)).unsqueeze(0)
        with torch.no_grad():
            z = world.encode(obs)
            v = critic(z).item()
        values.append(v)

    fig, ax = plt.subplots(figsize=(9, 2.8))
    im = ax.imshow([values], cmap='RdYlGn', aspect='auto')
    ax.set_yticks([])
    ax.set_xticks(range(env.n_pos))
    ax.set_xticklabels([str(i) for i in range(env.n_pos)])
    ax.set_xlabel('位置（中心 = {}）'.format(env.center))
    ax.set_title('Critic 在各位置的价值估计 V(z)（越绿越高）', fontsize=12, fontweight='bold')
    for i, v in enumerate(values):
        ax.text(i, 0, f'{v:.2f}', ha='center', va='center', fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.05)
    plt.tight_layout()
    plt.savefig(os.path.join(_IMAGES_DIR, 'dreamer_value_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[可视化] 价值热力图已保存至 images/dreamer_value_heatmap.png")


def plot_policy_bars(world: TinyWorldModel, actor: Actor, env: CorridorBalance):
    """可视化 Actor 在每个位置上选择"向右"的概率。"""
    probs_right = []
    for pos in range(env.n_pos):
        obs = torch.from_numpy(env.one_hot(pos)).unsqueeze(0)
        with torch.no_grad():
            z = world.encode(obs)
            logits = actor(z)
            p = F.softmax(logits, dim=-1)[0, 1].item()
        probs_right.append(p)

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ['#2E86AB' if p < 0.5 else '#C1666B' for p in probs_right]
    ax.bar(range(env.n_pos), probs_right, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.7, label='P(右)=0.5')
    ax.axvline(env.center, color='green', linestyle=':', alpha=0.7, label=f'中心位置={env.center}')
    ax.set_xlabel('位置')
    ax.set_ylabel('P(向右 | 位置)')
    ax.set_ylim(0, 1)
    ax.set_title('Actor 策略：各位置选择"向右"的概率（应把 Agent 推回中心）',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(_IMAGES_DIR, 'dreamer_policy_bars.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[可视化] 策略柱状图已保存至 images/dreamer_policy_bars.png")


# ============================================================================
# 倒立摆：与 PETS / LeWM 同一套物理（θ=0 竖直向上）
# ============================================================================

def _wrap_pi(angle):
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class Pendulum:
    """观测 [cosθ, sinθ, ω]，动作 u∈[-1,1]。不依赖 Gymnasium。"""

    def __init__(self, max_torque=2.0, dt=0.05):
        self.g, self.m, self.l = 10.0, 1.0, 1.0
        self.dt, self.max_torque, self.max_speed = dt, max_torque, 8.0
        self.theta, self.omega = 0.0, 0.0

    def obs(self):
        return np.array(
            [np.cos(self.theta), np.sin(self.theta), self.omega],
            dtype=np.float32,
        )

    def reset(self):
        self.theta = float(np.random.uniform(-0.35, 0.35))
        self.omega = float(np.random.uniform(-0.5, 0.5))
        return self.obs()

    def step(self, action):
        u = float(np.clip(action, -1.0, 1.0)) * self.max_torque
        theta, omega = self.theta, self.omega
        theta_acc = (3.0 * self.g / (2.0 * self.l)) * np.sin(theta) + (
            3.0 / (self.m * self.l ** 2)
        ) * u
        omega = np.clip(omega + self.dt * theta_acc, -self.max_speed, self.max_speed)
        theta = _wrap_pi(theta + self.dt * omega)
        self.theta, self.omega = theta, omega
        reward = float(
            np.exp(-8.0 * theta ** 2) * np.exp(-0.05 * omega ** 2)
            - 0.01 * (u / self.max_torque) ** 2
        )
        done = False
        return self.obs(), reward, done


class PendulumWorld(nn.Module):
    """向量观测上的确定性潜动力学：编码 → 想象一步 → 奖励头。"""

    def __init__(self, obs_dim=3, z_dim=24):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 32), nn.ELU(), nn.Linear(32, z_dim),
        )
        self.dynamics = nn.Sequential(
            nn.Linear(z_dim + 1, 48), nn.ELU(), nn.Linear(48, z_dim),
        )
        self.reward_head = nn.Linear(z_dim, 1)

    def encode(self, obs):
        return self.encoder(obs)

    def imagine_step(self, z, action):
        z_next = self.dynamics(torch.cat([z, action], dim=-1))
        reward = self.reward_head(z_next).squeeze(-1)
        return z_next, reward


class GaussianActor(nn.Module):
    """tanh-Gaussian 力矩，输出 ∈[-1,1]。"""

    def __init__(self, z_dim=24):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z_dim, 32), nn.ELU(), nn.Linear(32, 2))

    def forward(self, z):
        h = self.net(z)
        mu = torch.tanh(h[..., :1])
        std = F.softplus(h[..., 1:2]) + 0.08
        return mu, std


class PendulumCritic(nn.Module):
    def __init__(self, z_dim=24):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z_dim, 32), nn.ELU(), nn.Linear(32, 1))

    def forward(self, z):
        return self.net(z).squeeze(-1)


def _pendulum_v_lambda(rewards, values, bootstrap, gamma=0.99, lam=0.95):
    """想象轨迹上的 V_λ（Dreamer V1）：混合 n-step + bootstrap。"""
    horizon = len(rewards)
    gae = torch.zeros_like(bootstrap)
    out = []
    next_v = bootstrap
    for t in reversed(range(horizon)):
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        out.insert(0, gae + values[t])
        next_v = values[t]
    return torch.stack(out)


def collect_pendulum_episodes(env, actor, world, n_episodes, max_steps=40, noise=0.25):
    episodes = []
    for _ in range(n_episodes):
        obs = env.reset()
        traj = {'obs': [], 'act': [], 'rew': [], 'next_obs': [], 'done': []}
        done = False
        t = 0
        while (not done) and t < max_steps:
            obs_t = torch.from_numpy(obs).unsqueeze(0)
            with torch.no_grad():
                z = world.encode(obs_t)
                mu, std = actor(z)
                a = torch.clamp(mu + noise * torch.randn_like(mu), -1.0, 1.0)
            action = float(a.item())
            next_obs, reward, done = env.step(action)
            traj['obs'].append(obs)
            traj['act'].append([action])
            traj['rew'].append(reward)
            traj['next_obs'].append(next_obs)
            traj['done'].append(float(done))
            obs = next_obs
            t += 1
        episodes.append(traj)
    return episodes


def train_pendulum_world(world, optimizer, episodes, n_steps=25):
    obs, act, rew, next_obs = [], [], [], []
    for ep in episodes:
        obs.extend(ep['obs'])
        act.extend(ep['act'])
        rew.extend(ep['rew'])
        next_obs.extend(ep['next_obs'])
    obs_t = torch.tensor(np.array(obs), dtype=torch.float32)
    next_t = torch.tensor(np.array(next_obs), dtype=torch.float32)
    act_t = torch.tensor(np.array(act), dtype=torch.float32)
    rew_t = torch.tensor(rew, dtype=torch.float32)
    losses = []
    n = obs_t.shape[0]
    for _ in range(n_steps):
        idx = np.random.choice(n, size=min(64, n), replace=False)
        z = world.encode(obs_t[idx])
        z_tgt = world.encode(next_t[idx]).detach()
        z_pred, r_pred = world.imagine_step(z, act_t[idx])
        loss = F.mse_loss(z_pred, z_tgt) + F.mse_loss(r_pred, rew_t[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def imagine_train_pendulum_ac(
    world, actor, critic, actor_opt, critic_opt, start_obs,
    horizon=8, n_updates=12, gamma=0.99, lam=0.95,
):
    """世界模型参数冻结；梯度沿想象动力学回到 Actor（Dreamer V1 直通）。"""
    actor_losses, critic_losses = [], []
    for _ in range(n_updates):
        idx = np.random.choice(start_obs.shape[0], size=min(32, start_obs.shape[0]), replace=False)
        with torch.no_grad():
            z = world.encode(start_obs[idx])
        rewards, values, zs = [], [], [z]
        for _t in range(horizon):
            mu, std = actor(z)
            a = torch.clamp(mu + std * torch.randn_like(mu), -1.0, 1.0)
            z, r = world.imagine_step(z, a)
            rewards.append(r)
            values.append(critic(zs[-1]))
            zs.append(z)
        bootstrap = critic(zs[-1]).detach()
        values_t = torch.stack(values)
        rewards_t = torch.stack(rewards)
        v_lam = _pendulum_v_lambda(list(rewards_t), list(values_t.detach()), bootstrap, gamma, lam)
        v_lam = torch.clamp(v_lam, -15.0, 15.0)
        critic_loss = F.mse_loss(values_t, v_lam.detach())
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()

        # 再想象一次，让 Actor 最大化 V_λ（世界模型参数不进 actor_opt）
        with torch.no_grad():
            z = world.encode(start_obs[idx])
        rewards, values = [], []
        zs = [z]
        for _t in range(horizon):
            mu, std = actor(z)
            a = torch.clamp(mu + std * torch.randn_like(mu), -1.0, 1.0)
            z, r = world.imagine_step(z, a)
            rewards.append(r)
            values.append(critic(zs[-1]))
            zs.append(z)
        bootstrap = critic(zs[-1]).detach()
        values_t = torch.stack(values)
        v_lam = _pendulum_v_lambda(
            list(torch.stack(rewards)), list(values_t.detach()), bootstrap, gamma, lam,
        )
        v_lam = torch.clamp(v_lam, -15.0, 15.0)
        actor_loss = -v_lam.mean()
        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()
        actor_losses.append(actor_loss.item())
        critic_losses.append(critic_loss.item())
    return float(np.mean(actor_losses)), float(np.mean(critic_losses))


def eval_pendulum(env, actor, world, n_episodes=8, max_steps=40):
    rets, angs = [], []
    last_thetas = None
    for _ in range(n_episodes):
        obs = env.reset()
        ep_ret, thetas = 0.0, [env.theta]
        for _t in range(max_steps):
            with torch.no_grad():
                z = world.encode(torch.from_numpy(obs).unsqueeze(0))
                mu, _std = actor(z)
            obs, r, done = env.step(float(mu.item()))
            ep_ret += r
            thetas.append(env.theta)
            if done:
                break
        rets.append(ep_ret)
        angs.append(abs(thetas[-1]))
        last_thetas = thetas
    return float(np.mean(rets)), float(np.mean(angs)), last_thetas


def train_dreamer_pendulum(n_iters=14, episodes_per_iter=4, max_steps=36):
    print('\n=== Dreamer · 倒立摆（想象 Actor-Critic）===')
    env = Pendulum()
    z_dim = 24
    world = PendulumWorld(z_dim=z_dim)
    actor = GaussianActor(z_dim)
    critic = PendulumCritic(z_dim)
    world_opt = optim.Adam(world.parameters(), lr=3e-3)
    actor_opt = optim.Adam(actor.parameters(), lr=1e-3)
    critic_opt = optim.Adam(critic.parameters(), lr=3e-3)

    eval_rets, eval_angs, all_obs = [], [], []
    for it in range(n_iters):
        noise = max(0.08, 0.4 * (1.0 - it / n_iters))
        episodes = collect_pendulum_episodes(
            env, actor, world, episodes_per_iter, max_steps=max_steps, noise=noise,
        )
        for ep in episodes:
            all_obs.extend(ep['obs'])
        wm_loss = train_pendulum_world(world, world_opt, episodes, n_steps=20)
        start = torch.tensor(np.array(all_obs[-400:]), dtype=torch.float32)
        a_loss, c_loss = imagine_train_pendulum_ac(
            world, actor, critic, actor_opt, critic_opt, start, horizon=8, n_updates=10,
        )
        ret, ang, _ = eval_pendulum(env, actor, world, n_episodes=6, max_steps=max_steps)
        eval_rets.append(ret)
        eval_angs.append(ang)
        if (it + 1) % 6 == 0 or it == 0:
            print(f'  iter {it+1:2d}/{n_iters}  eval_ret={ret:6.2f}  |θ|={ang:.3f}  '
                  f'wm={wm_loss:.3f}  actor={a_loss:.3f}  critic={c_loss:.3f}')

    _, _, last_thetas = eval_pendulum(env, actor, world, n_episodes=1, max_steps=max_steps)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(eval_rets, color='#2E86AB', lw=2)
    axes[0].set_xlabel('训练轮次')
    axes[0].set_ylabel('评估回报')
    axes[0].set_title('倒立摆：想象中学策略')
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(eval_angs, color='#C1666B', lw=2)
    axes[1].set_xlabel('训练轮次')
    axes[1].set_ylabel(r'评估 $|θ|$ (rad)')
    axes[1].set_title('偏离直立（越低越好）')
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'dreamer_pendulum.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)

    fig2, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(last_thetas, color='#1a7f37', lw=2)
    ax.axhline(0.0, color='gray', ls='--', alpha=0.6)
    ax.set_xlabel('时间步')
    ax.set_ylabel(r'$θ$ (rad)')
    ax.set_title('评估回合：角度轨迹（0 = 直立）')
    ax.grid(True, alpha=0.3)
    fig2.tight_layout()
    out2 = os.path.join(_IMAGES_DIR, 'dreamer_pendulum_rollout.png')
    fig2.savefig(out2, dpi=140)
    plt.close(fig2)
    print('保存', out2)
    return eval_rets, eval_angs


# ============================================================================
# 主程序
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("    Dreamer：倒立摆想象循环 + 离散走廊附录")
    print("=" * 70)

    set_seed(42)
    train_dreamer_pendulum()

    print("\n[附录] 离散走廊：短跑刷新策略/价值图（对应 V1 骨架的离散版）...")
    env = CorridorBalance(n_pos=11, max_steps=30)
    dreamer_returns, world, actor, critic = train_dreamer_style(
        env, n_iters=10, episodes_per_iter=4,
    )
    set_seed(42)
    env2 = CorridorBalance(n_pos=11, max_steps=30)
    baseline_returns = train_reinforce_baseline(env2, n_episodes=80)
    plot_return_comparison(dreamer_returns, baseline_returns)
    plot_value_heatmap(world, critic, env)
    plot_policy_bars(world, actor, env)

    print("\n" + "=" * 70)
    print("【总结】")
    print("=" * 70)
    print("  倒立摆演示对应 Dreamer V1：真实交互拟合世界模型，")
    print("  再在潜空间想象轨迹上用 V_λ 更新 Actor-Critic。")
    print("  走廊附录用来对照离散动作上的同一套循环。")
    print(f"\n  所有图片已保存至 {_IMAGES_DIR}")
    print("=" * 70)
    print("\n  运行完成！\n")


if __name__ == "__main__":
    main()
