# -*- coding: utf-8 -*-
"""
wm02 经典起源与 RSSM —— 演示代码
======================================================
功能：
  1. 构造一个玩具 2D 环境：一个受"目标圆周运动"控制器驱动的质点，
     只能观测到带噪声的位置 (x, y)，真实的位置/速度/圆周参数对模型不可见。
  2. 从零实现一个简化版 RSSM (Recurrent State-Space Model)：
     - 确定性状态 h_t = GRU(h_{t-1}, s_{t-1}, a_{t-1})
     - 随机先验 (Prior)     s_t ~ p(s_t | h_t)          —— 不看观测，纯预测
     - 随机后验 (Posterior) s_t ~ q(s_t | h_t, o_t)      —— 看到观测后的修正估计
     - 解码器  o_t ~ p(o_t | h_t, s_t)
  3. 用 "重建损失 + KL(后验||先验)" 训练模型（变分下界 / ELBO 的序列版本）
  4. 评估"在潜空间里做梦"的效果：给定前几步真实观测热启动后，
     只用先验做多步想象 rollout，对比预测轨迹与真实轨迹
  5. 对比"闭环滤波"（每步都看真实观测）与"开环想象"（不看观测）
     的误差增长速度，呼应 wm01 的核心直觉

每个函数都有中文 docstring，每行逻辑代码都有中文注释。
运行方式：在 wm02_planet_rssm/ 目录下执行 python code/demo.py
依赖: pip install numpy matplotlib torch
"""

import os
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

DEVICE = torch.device('cpu')                                     # 本章模型极小，CPU 已足够快

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)


def set_seed(seed: int = 42):
    """
    设置所有随机种子，保证实验可复现。

    参数:
        seed: 随机种子值
    """
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================================
# 第一部分：玩具环境 —— 受控质点的圆周运动
# ============================================================================

class ToyPointEnv:
    """
    玩具环境：一个二维平面上的质点，由"追踪旋转目标点"的 PD 控制器驱动。

    真实状态是 (位置, 速度)，但模型只能观测到带噪声的位置 (x, y)。
    每条轨迹的圆周半径、角速度、初始相位都是随机的，
    这样模型必须真正利用动作序列 a_t 来区分不同轨迹，而不能只记住一条固定路径。

    动力学:
        target(t) = r * [cos(w*t + phi), sin(w*t + phi)]        # 旋转的目标点
        a_t = k_p * (target(t) - pos_t) + k_d * (target_vel(t) - vel_t)   # PD 控制器给出加速度
        vel_{t+1} = vel_t + dt * (a_t - damping * vel_t) + process_noise
        pos_{t+1} = pos_t + dt * vel_{t+1}
        o_t = pos_t + obs_noise                                  # 带噪声的观测
    """

    def __init__(
        self,
        dt: float = 0.1,
        damping: float = 0.15,
        process_noise_std: float = 0.01,
        obs_noise_std: float = 0.04,
        k_p: float = 3.0,
        k_d: float = 1.5,
    ):
        """
        初始化环境参数。

        参数:
            dt: 仿真步长
            damping: 速度阻尼系数
            process_noise_std: 过程噪声（真实动力学的随机扰动）标准差
            obs_noise_std: 观测噪声标准差
            k_p, k_d: PD 控制器的比例、微分增益
        """
        self.dt = dt
        self.damping = damping
        self.process_noise_std = process_noise_std
        self.obs_noise_std = obs_noise_std
        self.k_p = k_p
        self.k_d = k_d

    def rollout(self, T: int, r: float, w: float, phi: float, rng: np.random.RandomState):
        """
        生成一条完整轨迹（真实位置、噪声观测、控制动作）。

        参数:
            T: 轨迹长度（步数）
            r: 目标圆周运动的半径
            w: 目标圆周运动的角速度
            phi: 目标圆周运动的初始相位
            rng: 随机数生成器（保证每条轨迹独立可控地随机）

        返回:
            true_pos: 真实位置序列，shape (T, 2)
            obs: 带噪声的观测序列，shape (T, 2)
            actions: PD 控制器给出的动作（加速度）序列，shape (T, 2)
        """
        pos = rng.normal(0, 0.15, size=2)                         # 随机初始位置
        vel = rng.normal(0, 0.05, size=2)                         # 随机初始速度

        true_pos = np.zeros((T, 2))
        obs = np.zeros((T, 2))
        actions = np.zeros((T, 2))

        for t in range(T):
            time = t * self.dt
            target = r * np.array([np.cos(w * time + phi), np.sin(w * time + phi)])
            target_vel = r * w * np.array([-np.sin(w * time + phi), np.cos(w * time + phi)])

            action = self.k_p * (target - pos) + self.k_d * (target_vel - vel)  # PD 控制器

            true_pos[t] = pos
            obs[t] = pos + rng.normal(0, self.obs_noise_std, size=2)
            actions[t] = action

            # ---- 状态转移（真实动力学，模型不可见）----
            noise = rng.normal(0, self.process_noise_std, size=2)
            vel = vel + self.dt * (action - self.damping * vel) + noise
            pos = pos + self.dt * vel

        return true_pos, obs, actions


def generate_dataset(
    n_traj: int, T: int, seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成一批轨迹数据集，每条轨迹的圆周参数 (r, w, phi) 都是随机的。

    参数:
        n_traj: 轨迹数量
        T: 每条轨迹的长度
        seed: 随机种子

    返回:
        true_pos_all: shape (n_traj, T, 2)
        obs_all: shape (n_traj, T, 2)
        act_all: shape (n_traj, T, 2)
    """
    rng = np.random.RandomState(seed)
    env = ToyPointEnv()

    true_pos_all = np.zeros((n_traj, T, 2), dtype=np.float32)
    obs_all = np.zeros((n_traj, T, 2), dtype=np.float32)
    act_all = np.zeros((n_traj, T, 2), dtype=np.float32)

    for i in range(n_traj):
        r = rng.uniform(0.5, 1.4)                                 # 随机半径
        w = rng.uniform(0.6, 1.6) * rng.choice([-1, 1])           # 随机角速度（含方向）
        phi = rng.uniform(0, 2 * np.pi)                           # 随机初始相位
        true_pos, obs, actions = env.rollout(T, r, w, phi, rng)
        true_pos_all[i] = true_pos
        obs_all[i] = obs
        act_all[i] = actions

    return true_pos_all, obs_all, act_all


# ============================================================================
# 第二部分：简化版 RSSM (Recurrent State-Space Model)
# ============================================================================

class RSSM(nn.Module):
    """
    简化版 RSSM —— PlaNet (Hafner et al., 2019) 提出的潜空间动力学模型核心结构。

    关键设计（区别于普通 VAE 或纯 RNN）：
        - 确定性状态 h_t：由 GRU 承载，负责记住"长期、确定性强"的信息（如已知的运动趋势）
        - 随机状态 s_t：由高斯分布采样，负责表达"不确定性"（如观测噪声带来的模糊性）
        - 先验 p(s_t|h_t)：只用 h_t 预测 s_t 的分布，不看当前观测 —— 这就是"做梦"时用的部分
        - 后验 q(s_t|h_t,o_t)：额外用上当前观测 o_t 来修正对 s_t 的估计 —— 训练时用来提供更准确的监督

    训练时最小化 KL(后验 || 先验)，就是在教先验"如何在不看观测的情况下，
    尽量猜得像后验一样准"——这正是"学会做梦"的数学本质。
    """

    def __init__(
        self,
        obs_dim: int = 2,
        act_dim: int = 2,
        deter_dim: int = 32,
        stoch_dim: int = 4,
        hidden_dim: int = 32,
    ):
        """
        初始化 RSSM 各组件。

        参数:
            obs_dim: 观测维度（这里是 2D 位置）
            act_dim: 动作维度（这里是 2D 加速度）
            deter_dim: 确定性状态 h_t 的维度
            stoch_dim: 随机状态 s_t 的维度
            hidden_dim: 各 MLP 的隐藏层维度
        """
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim

        # ---- 确定性状态更新：GRU ----
        # 输入: 上一步的随机状态 s_{t-1} 与动作 a_{t-1}
        self.gru = nn.GRUCell(stoch_dim + act_dim, deter_dim)

        # ---- 先验网络: h_t -> N(mu_prior, std_prior) ----
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 2 * stoch_dim),                 # 输出 [mu, log_std]
        )

        # ---- 后验网络: (h_t, o_t) -> N(mu_post, std_post) ----
        self.posterior_net = nn.Sequential(
            nn.Linear(deter_dim + obs_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 2 * stoch_dim),
        )

        # ---- 解码器: (h_t, s_t) -> 预测观测 ----
        self.decoder = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, obs_dim),
        )

    def _split_mean_std(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        把 MLP 的输出拆分成均值和标准差。

        参数:
            x: shape (..., 2*stoch_dim)

        返回:
            mean: shape (..., stoch_dim)
            std: shape (..., stoch_dim)，经过 softplus 保证为正
        """
        mean, log_std = torch.chunk(x, 2, dim=-1)                 # 沿最后一维切成两半
        std = F.softplus(log_std) + 1e-3                          # softplus 保证标准差 > 0
        return mean, std

    def prior(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """先验分布 p(s_t | h_t)：只依赖确定性状态，不看观测。"""
        return self._split_mean_std(self.prior_net(h))

    def posterior(self, h: torch.Tensor, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """后验分布 q(s_t | h_t, o_t)：额外融合当前观测。"""
        return self._split_mean_std(self.posterior_net(torch.cat([h, obs], dim=-1)))

    def decode(self, h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """解码器：从 (h_t, s_t) 预测观测均值。"""
        return self.decoder(torch.cat([h, s], dim=-1))

    @staticmethod
    def reparameterize(mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """
        重参数化技巧：从 N(mean, std) 采样，同时保持梯度可传播。

        z = mean + std * eps,  eps ~ N(0, I)
        """
        eps = torch.randn_like(mean)
        return mean + std * eps

    @staticmethod
    def kl_divergence(mean_q, std_q, mean_p, std_p) -> torch.Tensor:
        """
        两个对角高斯分布之间的 KL 散度（逐维度求和后返回每个样本一个标量）。

        KL(N(mu_q,std_q) || N(mu_p,std_p)) =
            log(std_p/std_q) + (std_q^2 + (mu_q-mu_p)^2) / (2*std_p^2) - 0.5    (逐维求和)

        参数:
            mean_q, std_q: 后验分布参数（"教师"，看到了观测）
            mean_p, std_p: 先验分布参数（"学生"，只能靠 h_t 预测）

        返回:
            kl: shape (batch,)，每个样本的 KL 散度（对 stoch_dim 求和后的结果）
        """
        var_q, var_p = std_q ** 2, std_p ** 2
        kl = torch.log(std_p / std_q) + (var_q + (mean_q - mean_p) ** 2) / (2 * var_p) - 0.5
        return kl.sum(dim=-1)                                     # 对随机状态的每一维求和

    def forward(
        self, obs_seq: torch.Tensor, act_seq: torch.Tensor,
    ):
        """
        对一批完整轨迹做"教师强制"训练前向传播（每一步都用真实观测计算后验）。

        参数:
            obs_seq: shape (batch, T, obs_dim)
            act_seq: shape (batch, T, act_dim)，act_seq[:, t] 是从 t 到 t+1 的动作

        返回:
            recon_loss: 标量，重建损失（对所有步、所有样本求平均）
            kl_loss: 标量，KL 散度损失
            obs_preds: shape (batch, T, obs_dim)，每一步的重建观测（用于可视化）
        """
        batch, T, _ = obs_seq.shape
        h = torch.zeros(batch, self.deter_dim, device=obs_seq.device)
        s = torch.zeros(batch, self.stoch_dim, device=obs_seq.device)
        prev_action = torch.zeros(batch, act_seq.shape[-1], device=obs_seq.device)

        recon_loss = 0.0
        kl_loss = 0.0
        obs_preds = []

        for t in range(T):
            # ---- 1. 用上一步的 (s_{t-1}, a_{t-1}) 更新确定性状态 h_t ----
            h = self.gru(torch.cat([s, prev_action], dim=-1), h)

            # ---- 2. 先验分布（不看观测，纯"预测"） ----
            prior_mean, prior_std = self.prior(h)

            # ---- 3. 后验分布（看到当前观测后的修正估计） ----
            post_mean, post_std = self.posterior(h, obs_seq[:, t])

            # ---- 4. 训练时用后验采样（更准确），推理/想象时才用先验采样 ----
            s = self.reparameterize(post_mean, post_std)

            # ---- 5. 解码重建当前观测 ----
            obs_pred = self.decode(h, s)
            obs_preds.append(obs_pred)

            # ---- 6. 累积损失 ----
            recon_loss = recon_loss + F.mse_loss(obs_pred, obs_seq[:, t], reduction='none').sum(-1).mean()
            kl_loss = kl_loss + self.kl_divergence(post_mean, post_std, prior_mean, prior_std).mean()

            prev_action = act_seq[:, t]

        recon_loss = recon_loss / T
        kl_loss = kl_loss / T
        obs_preds = torch.stack(obs_preds, dim=1)
        return recon_loss, kl_loss, obs_preds

    @torch.no_grad()
    def imagine(
        self,
        context_obs: torch.Tensor,
        context_act: torch.Tensor,
        future_act: torch.Tensor,
    ) -> torch.Tensor:
        """
        "做梦"：用一小段真实观测热启动 (h, s)，之后完全脱离观测，
        只用先验分布 + 给定的未来动作序列，在潜空间里连续 rollout 多步。

        参数:
            context_obs: shape (batch, C, obs_dim)，热启动用的真实观测
            context_act: shape (batch, C, act_dim)，热启动阶段的动作
            future_act: shape (batch, K, act_dim)，想象阶段要"执行"的动作序列

        返回:
            imagined_obs: shape (batch, K, obs_dim)，纯想象出的未来观测序列
        """
        batch = context_obs.shape[0]
        h = torch.zeros(batch, self.deter_dim, device=context_obs.device)
        s = torch.zeros(batch, self.stoch_dim, device=context_obs.device)
        prev_action = torch.zeros(batch, context_act.shape[-1], device=context_obs.device)

        # ---- 热启动阶段：用真实观测的后验（"睁眼看世界"）----
        C = context_obs.shape[1]
        for t in range(C):
            h = self.gru(torch.cat([s, prev_action], dim=-1), h)
            post_mean, post_std = self.posterior(h, context_obs[:, t])
            s = post_mean                                          # 评估时用均值，减少不必要的采样噪声
            prev_action = context_act[:, t]

        # ---- 想象阶段：完全"闭眼"，只用先验 rollout ----
        imagined_obs = []
        K = future_act.shape[1]
        for t in range(K):
            h = self.gru(torch.cat([s, prev_action], dim=-1), h)
            prior_mean, prior_std = self.prior(h)
            s = prior_mean                                          # 想象时同样用先验均值（去掉采样噪声，看清均值轨迹）
            obs_pred = self.decode(h, s)
            imagined_obs.append(obs_pred)
            prev_action = future_act[:, t]

        return torch.stack(imagined_obs, dim=1)

    @torch.no_grad()
    def filter_rollout(
        self, obs_seq: torch.Tensor, act_seq: torch.Tensor,
    ) -> torch.Tensor:
        """
        "闭环滤波" rollout：每一步都能看到真实观测（用后验），用于对比"开环想象"的误差增长速度。

        参数:
            obs_seq: shape (batch, T, obs_dim)
            act_seq: shape (batch, T, act_dim)

        返回:
            filtered_obs: shape (batch, T, obs_dim)
        """
        batch, T, _ = obs_seq.shape
        h = torch.zeros(batch, self.deter_dim, device=obs_seq.device)
        s = torch.zeros(batch, self.stoch_dim, device=obs_seq.device)
        prev_action = torch.zeros(batch, act_seq.shape[-1], device=obs_seq.device)

        outputs = []
        for t in range(T):
            h = self.gru(torch.cat([s, prev_action], dim=-1), h)
            post_mean, post_std = self.posterior(h, obs_seq[:, t])
            s = post_mean
            outputs.append(self.decode(h, s))
            prev_action = act_seq[:, t]

        return torch.stack(outputs, dim=1)


# ============================================================================
# 第三部分：训练循环
# ============================================================================

def train_rssm(
    model: RSSM,
    obs_train: torch.Tensor,
    act_train: torch.Tensor,
    n_epochs: int = 200,
    batch_size: int = 32,
    lr: float = 3e-3,
    free_nats: float = 0.5,
    kl_beta: float = 1.0,
) -> Tuple[List[float], List[float]]:
    """
    训练 RSSM：最小化"重建损失 + KL 惯罚"（序列版 ELBO 的负值）。

    参数:
        model: RSSM 模型
        obs_train / act_train: 训练集，shape (N, T, dim)
        n_epochs: 训练轮数
        batch_size: 每轮用于计算梯度的小批量大小
        lr: 学习率
        free_nats: "自由信息量"阈值——KL 低于这个值时不再惩罚，
                   防止后验过早"躺平"变成先验（posterior collapse）
        kl_beta: KL 项的权重系数

    返回:
        recon_losses, kl_losses: 每轮的损失记录
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    n_train = obs_train.shape[0]
    recon_losses, kl_losses = [], []

    for epoch in range(n_epochs):
        idx = np.random.choice(n_train, batch_size, replace=False)
        obs_batch = obs_train[idx]
        act_batch = act_train[idx]

        recon_loss, kl_loss, _ = model(obs_batch, act_batch)
        # ---- free nats: 只惩罚超出自由信息量阈值的部分 ----
        kl_penalty = torch.clamp(kl_loss, min=free_nats)
        loss = recon_loss + kl_beta * kl_penalty

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        recon_losses.append(recon_loss.item())
        kl_losses.append(kl_loss.item())

        if (epoch + 1) % 40 == 0:
            print(f"  Epoch {epoch+1:4d}/{n_epochs}: recon_loss={recon_loss.item():.4f}, "
                  f"kl_loss={kl_loss.item():.4f}")

    return recon_losses, kl_losses


# ============================================================================
# 第四部分：可视化
# ============================================================================

def plot_training_curves(recon_losses: List[float], kl_losses: List[float]):
    """绘制训练过程中重建损失和 KL 损失的变化曲线。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(recon_losses, color='#2E86AB')
    axes[0].set_title('重建损失 (Reconstruction Loss)', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('训练轮数 (Epoch)')
    axes[0].set_ylabel('MSE')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(kl_losses, color='#C1666B')
    axes[1].set_title('KL 散度 KL(后验‖先验)', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('训练轮数 (Epoch)')
    axes[1].set_ylabel('nats')
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('RSSM 训练曲线', fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(_IMAGES_DIR, 'rssm_training_loss.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[可视化] 训练曲线已保存至 images/rssm_training_loss.png")


def plot_imagination_rollout(
    model: RSSM, obs_test: torch.Tensor, act_test: torch.Tensor,
    true_pos_test: np.ndarray, context_len: int = 10, n_show: int = 4,
):
    """
    可视化"在潜空间做梦"的效果：给定前 context_len 步真实观测热启动后，
    完全脱离观测，只用先验做多步想象 rollout，对比预测轨迹与真实轨迹。
    """
    T = obs_test.shape[1]
    context_obs = obs_test[:n_show, :context_len]
    context_act = act_test[:n_show, :context_len]
    future_act = act_test[:n_show, context_len:]

    imagined = model.imagine(context_obs, context_act, future_act).numpy()

    fig, axes = plt.subplots(1, n_show, figsize=(4.2 * n_show, 4.2))
    for i in range(n_show):
        ax = axes[i] if n_show > 1 else axes
        true_traj = true_pos_test[i]
        ax.plot(true_traj[:, 0], true_traj[:, 1], color='#888888', linewidth=2,
                label='真实轨迹', zorder=1)
        ax.plot(true_traj[:context_len, 0], true_traj[:context_len, 1],
                color='#2E86AB', linewidth=3, label=f'热启动 (前{context_len}步真实观测)', zorder=2)
        ax.plot(imagined[i, :, 0], imagined[i, :, 1], color='#C1666B', linewidth=2.4,
                linestyle='--', marker='o', markersize=2.5, label='潜空间想象 rollout', zorder=3)
        ax.scatter([true_traj[context_len, 0]], [true_traj[context_len, 1]],
                   color='black', s=30, zorder=4, marker='x')
        ax.set_title(f'测试轨迹 #{i+1}', fontsize=10)
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7.5, loc='upper right')

    fig.suptitle(f'RSSM 潜空间想象 rollout：前 {context_len} 步真实观测热启动，之后完全"闭眼"预测 {T-context_len} 步',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(_IMAGES_DIR, 'rssm_imagination_rollout.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[可视化] 想象 rollout 对比图已保存至 images/rssm_imagination_rollout.png")

    return imagined


def plot_rollout_error_growth(
    model: RSSM, obs_test: torch.Tensor, act_test: torch.Tensor,
    true_pos_test: np.ndarray, context_len: int = 10,
):
    """
    对比"闭环滤波"（每步都用真实观测）与"开环想象"（完全不看观测）随预测步数增长的误差差异。

    这是 wm01 中"为什么在潜空间里做梦"的定量验证版本——
    用真实训练好的 RSSM 模型，而不是玩具复合误差公式。
    """
    n_test = obs_test.shape[0]
    T = obs_test.shape[1]
    K = T - context_len

    # ---- 开环想象：只用前 context_len 步真实观测，之后纯先验 rollout ----
    context_obs = obs_test[:, :context_len]
    context_act = act_test[:, :context_len]
    future_act = act_test[:, context_len:]
    imagined = model.imagine(context_obs, context_act, future_act).numpy()   # (N, K, 2)

    # ---- 闭环滤波：每一步都看真实观测 ----
    filtered = model.filter_rollout(obs_test, act_test).numpy()[:, context_len:]  # (N, K, 2)

    true_future = true_pos_test[:, context_len:]                              # (N, K, 2)

    imagine_err = np.linalg.norm(imagined - true_future, axis=-1).mean(axis=0)   # (K,)
    filter_err = np.linalg.norm(filtered - true_future, axis=-1).mean(axis=0)    # (K,)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    steps = np.arange(1, K + 1)
    ax.plot(steps, imagine_err, color='#C1666B', linewidth=2.2, marker='o', markersize=3,
            label='开环想象 rollout（只用先验，不看观测）')
    ax.plot(steps, filter_err, color='#2E86AB', linewidth=2.2, marker='s', markersize=3,
            label='闭环滤波 rollout（每步都用后验，看得到观测）')
    ax.set_xlabel('预测步数（从热启动结束算起）', fontsize=10)
    ax.set_ylabel('平均位置误差 (L2 distance)', fontsize=10)
    ax.set_title('真实 RSSM 模型上的误差增长对比：开环想象 vs 闭环滤波', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(_IMAGES_DIR, 'rssm_rollout_error_growth.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[可视化] 误差增长对比图已保存至 images/rssm_rollout_error_growth.png")

    return imagine_err, filter_err


# ============================================================================
# 主程序
# ============================================================================

def main():
    """
    主程序：生成数据、训练 RSSM、评估想象 rollout 效果并可视化。
    """
    print("\n" + "=" * 70)
    print("    wm02 经典起源与 RSSM —— 完整演示")
    print("=" * 70)

    set_seed(42)

    # ---- 1. 生成数据集 ----
    print("\n[步骤 1] 生成玩具环境数据集（受控质点圆周运动）...")
    T = 40
    true_pos_all, obs_all, act_all = generate_dataset(n_traj=300, T=T, seed=42)
    n_train = 250
    obs_train = torch.from_numpy(obs_all[:n_train])
    act_train = torch.from_numpy(act_all[:n_train])
    obs_test = torch.from_numpy(obs_all[n_train:])
    act_test = torch.from_numpy(act_all[n_train:])
    true_pos_test = true_pos_all[n_train:]
    print(f"  训练轨迹数: {n_train}, 测试轨迹数: {obs_test.shape[0]}, 每条轨迹长度: {T}")

    # ---- 2. 初始化并训练 RSSM ----
    print("\n[步骤 2] 训练简化版 RSSM...")
    model = RSSM(obs_dim=2, act_dim=2, deter_dim=32, stoch_dim=4, hidden_dim=32)
    recon_losses, kl_losses = train_rssm(
        model, obs_train, act_train,
        n_epochs=250, batch_size=32, lr=3e-3, free_nats=0.5, kl_beta=1.0,
    )
    print(f"\n  训练完成。最终重建损失={recon_losses[-1]:.4f}, KL={kl_losses[-1]:.4f}")

    # ---- 3. 可视化训练曲线 ----
    print("\n[步骤 3] 绘制训练曲线...")
    plot_training_curves(recon_losses, kl_losses)

    # ---- 4. 可视化想象 rollout ----
    print("\n[步骤 4] 评估潜空间想象 rollout 效果...")
    context_len = 10
    imagined = plot_imagination_rollout(
        model, obs_test, act_test, true_pos_test, context_len=context_len, n_show=4,
    )

    # ---- 5. 误差增长对比：开环想象 vs 闭环滤波 ----
    print("\n[步骤 5] 对比开环想象与闭环滤波的误差增长速度...")
    imagine_err, filter_err = plot_rollout_error_growth(
        model, obs_test, act_test, true_pos_test, context_len=context_len,
    )
    print(f"  想象 rollout 第 1 步误差={imagine_err[0]:.4f}, 第 {len(imagine_err)} 步误差={imagine_err[-1]:.4f}")
    print(f"  滤波 rollout   第 1 步误差={filter_err[0]:.4f}, 第 {len(filter_err)} 步误差={filter_err[-1]:.4f}")

    # ---- 总结 ----
    print("\n" + "=" * 70)
    print("【总结】")
    print("=" * 70)
    print("  RSSM 的核心结构:")
    print("    - 确定性状态 h_t (GRU)：承载长期、确定性强的信息")
    print("    - 随机状态 s_t：先验 p(s_t|h_t) 用于'做梦'，后验 q(s_t|h_t,o_t) 用于训练监督")
    print("    - 训练目标：重建损失 + KL(后验‖先验)，即序列版 ELBO")
    print("\n  本演示验证的核心现象:")
    print("    - 热启动后，模型只用先验 + 动作序列就能在潜空间里较准确地想象未来轨迹")
    print("    - 开环想象的误差会随步数增长得比闭环滤波快（因为没有观测来纠错）")
    print("    - 但由于状态被压缩到低维潜空间（而非直接在像素/高维空间预测），")
    print("      误差增长速度远比 wm01 中'像素空间 rollout' 的类比要温和")
    print(f"\n  所有图片已保存至 {_IMAGES_DIR}")
    print("=" * 70)
    print("\n  运行完成！\n")


if __name__ == "__main__":
    main()
