# -*- coding: utf-8 -*-
"""路径一 · VAE 练习：重参数化与解析 KL。"""

import torch


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    TODO: z = μ + σ ⊙ ε,  σ = exp(0.5 * logvar), ε ~ N(0,1)
    """
    return None


def compute_kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    TODO: D_KL(N(μ,σ²) || N(0,1)) = -0.5 * mean_or_sum(1 + logvar - μ² - exp(logvar))
    对 latent 维求和，对 batch 取平均。
    """
    return torch.tensor(0.0)


if __name__ == "__main__":
    print("=" * 50)
    print("VAE 练习")
    print("=" * 50)
    mu = torch.tensor([[0.5, -0.3], [0.0, 0.8]])
    logvar = torch.tensor([[0.1, 0.2], [-0.5, 0.0]])
    z = reparameterize(mu, logvar)
    print("z:", z)
    mu1 = torch.zeros(10, 5)
    logvar1 = torch.zeros(10, 5)
    print("KL(后验=先验):", compute_kl_divergence(mu1, logvar1).item(), "期望≈0")
    mu2 = torch.ones(10, 5) * 2.0
    logvar2 = torch.ones(10, 5)
    print("KL(偏离):", compute_kl_divergence(mu2, logvar2).item(), "期望>0")
