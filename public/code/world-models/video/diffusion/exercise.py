# -*- coding: utf-8 -*-
"""
路径一 · 扩散模型练习：实现 q_sample 与 L_simple
"""
import torch
import torch.nn.functional as F


def q_sample(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor,
             sqrt_alphas_cumprod: torch.Tensor,
             sqrt_one_minus_alphas_cumprod: torch.Tensor) -> torch.Tensor:
    """
    TODO: 闭式前向加噪
        x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε
    t 的形状为 (N,)，其余为 (N, ...)；请把 √ᾱ_t 广播到 x0 的形状。
    """
    # TODO: 用 sqrt_alphas_cumprod[t] 与 sqrt_one_minus_alphas_cumprod[t]
    return None


def noise_prediction_loss(eps_pred: torch.Tensor, eps_true: torch.Tensor) -> torch.Tensor:
    """
    TODO: L_simple = mean ||ε - ε_θ||^2
    """
    return None


if __name__ == "__main__":
    print("=" * 50)
    print("扩散练习：q_sample 与噪声 MSE")
    print("=" * 50)
    torch.manual_seed(0)
    n, t_max = 8, 10
    x0 = torch.ones(n, 1)
    noise = torch.zeros(n, 1)
    t = torch.tensor([0, 3, 9, 1, 5, 2, 7, 4])
    sqrt_ab = torch.linspace(1.0, 0.2, t_max)
    sqrt_om = torch.sqrt(1.0 - sqrt_ab ** 2)
    xt = q_sample(x0, t, noise, sqrt_ab, sqrt_om)
    if xt is None:
        print("  请实现 q_sample")
    else:
        # noise=0 时 x_t 应等于 √ᾱ_t x_0
        expected = sqrt_ab[t].unsqueeze(-1) * x0
        err = (xt - expected).abs().max().item()
        print(f"  q_sample 最大误差（期望 ≈ 0）: {err:.6f}")

    pred = torch.tensor([0.2, -0.1, 0.0])
    true = torch.zeros(3)
    loss = noise_prediction_loss(pred, true)
    if loss is None:
        print("  请实现 noise_prediction_loss")
    else:
        print(f"  L_simple={loss.item():.4f}（期望 mean(pred^2)={F.mse_loss(pred, true).item():.4f}）")
