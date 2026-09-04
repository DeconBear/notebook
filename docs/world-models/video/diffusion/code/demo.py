# -*- coding: utf-8 -*-
"""
路径一 · 扩散模型 demo：一维双峰混合上的微型 DDPM
================================================
不训练 U-Net，只用一层 MLP 预测噪声，展示：
  1. 前向过程如何把两个峰糊成近高斯
  2. 反向过程如何把峰「揭」回来
  3. L_simple = MSE(ε, ε_θ(x_t, t))

运行：python demo.py
依赖：numpy, matplotlib, torch
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, "..", "images")
os.makedirs(_IMAGES_DIR, exist_ok=True)

DEVICE = torch.device("cpu")
torch.manual_seed(42)
np.random.seed(42)

T = 80  # 教学用短链；真实 DDPM 常取 1000
BETA_START = 1e-4
BETA_END = 0.08


def make_betas(timesteps: int) -> torch.Tensor:
    """线性噪声调度 β_t。"""
    return torch.linspace(BETA_START, BETA_END, timesteps, device=DEVICE)


class DiffusionSchedule:
    """预计算 α_t、ᾱ_t，供 q(x_t | x_0) 闭式采样。"""

    def __init__(self, timesteps: int = T):
        self.timesteps = timesteps
        self.betas = make_betas(timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε。t 形状 (N,)，x0/noise 形状 (N, 1)。"""
        sqrt_ab = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        return sqrt_ab * x0 + sqrt_om * noise


def sample_data(n: int) -> torch.Tensor:
    """双峰混合：N(-2, 0.4) 与 N(2, 0.4) 各一半。"""
    left = torch.randn(n // 2, 1, device=DEVICE) * 0.4 - 2.0
    right = torch.randn(n - n // 2, 1, device=DEVICE) * 0.4 + 2.0
    x = torch.cat([left, right], dim=0)
    return x[torch.randperm(n)]


class NoiseMLP(nn.Module):
    """ε_θ(x_t, t)：把标量 x 与归一化时间拼起来，回归噪声。"""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_feat = (t.float() / (T - 1)).unsqueeze(-1)
        return self.net(torch.cat([x, t_feat], dim=-1))


def train(model: NoiseMLP, sched: DiffusionSchedule, steps: int = 2500, batch: int = 256) -> list:
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    losses = []
    model.train()
    for i in range(steps):
        x0 = sample_data(batch)
        t = torch.randint(0, sched.timesteps, (batch,), device=DEVICE)
        noise = torch.randn_like(x0)
        xt = sched.q_sample(x0, t, noise)
        pred = model(xt, t)
        loss = ((pred - noise) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (i + 1) % 100 == 0:
            losses.append(loss.item())
            print(f"  step {i+1:4d}/{steps}  L_simple={loss.item():.4f}")
    return losses


@torch.no_grad()
def p_sample_loop(model: NoiseMLP, sched: DiffusionSchedule, n: int = 2000) -> np.ndarray:
    """从纯噪声逐步去噪，返回每 10 步的快照，形状 (K, n)。"""
    model.eval()
    x = torch.randn(n, 1, device=DEVICE)
    snaps = []
    for t in reversed(range(sched.timesteps)):
        t_batch = torch.full((n,), t, device=DEVICE, dtype=torch.long)
        eps = model(x, t_batch)
        alpha = sched.alphas[t]
        alpha_bar = sched.alphas_cumprod[t]
        beta = sched.betas[t]
        mean = (1.0 / torch.sqrt(alpha)) * (x - (1.0 - alpha) / torch.sqrt(1.0 - alpha_bar) * eps)
        if t > 0:
            x = mean + torch.sqrt(beta) * torch.randn_like(x)
        else:
            x = mean
        if t % 10 == 0 or t == sched.timesteps - 1:
            snaps.append(x.squeeze(-1).cpu().numpy().copy())
    return np.stack(snaps, axis=0)


def plot_forward(sched: DiffusionSchedule, save_name: str = "ddpm_forward_1d.png") -> None:
    x0 = sample_data(3000)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True)
    ts = [0, T // 4, T // 2, T - 1]
    bins = np.linspace(-5, 5, 50)
    for ax, t in zip(axes, ts):
        t_batch = torch.full((x0.size(0),), t, dtype=torch.long)
        noise = torch.randn_like(x0)
        xt = sched.q_sample(x0, t_batch, noise).cpu().numpy().ravel()
        ax.hist(xt, bins=bins, density=True, color="#2E86AB", alpha=0.85)
        ax.set_title(f"t = {t}")
        ax.set_xlim(-5, 5)
    axes[0].set_ylabel("密度")
    fig.suptitle("前向过程：双峰被逐步糊成近高斯", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMAGES_DIR, save_name), dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[可视化] {save_name}")


def plot_reverse(snaps: np.ndarray, save_name: str = "ddpm_reverse_1d.png") -> None:
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True)
    idxs = [0, len(snaps) // 3, 2 * len(snaps) // 3, -1]
    bins = np.linspace(-5, 5, 50)
    for ax, i in zip(axes, idxs):
        ax.hist(snaps[i], bins=bins, density=True, color="#C1666B", alpha=0.85)
        ax.set_title(f"反向快照 {i if i >= 0 else '终态'}")
        ax.set_xlim(-5, 5)
    axes[0].set_ylabel("密度")
    fig.suptitle("反向过程：从噪声里把两个峰揭回来", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMAGES_DIR, save_name), dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[可视化] {save_name}")


def plot_loss(losses: list, save_name: str = "ddpm_loss.png") -> None:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(np.arange(len(losses)) * 100, losses, color="#3B7A57")
    ax.set_xlabel("训练步")
    ax.set_ylabel(r"$\mathcal{L}_{\mathrm{simple}}$")
    ax.set_title("噪声预测 MSE")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(_IMAGES_DIR, save_name), dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[可视化] {save_name}")


def main() -> None:
    print("=" * 56)
    print("路径一 · 一维微型 DDPM")
    print("=" * 56)
    sched = DiffusionSchedule(T)
    plot_forward(sched)
    model = NoiseMLP().to(DEVICE)
    print("\n[训练] 预测 ε …")
    losses = train(model, sched)
    snaps = p_sample_loop(model, sched)
    plot_reverse(snaps)
    plot_loss(losses)
    print(f"\n完成。图片在 {_IMAGES_DIR}")


if __name__ == "__main__":
    main()
