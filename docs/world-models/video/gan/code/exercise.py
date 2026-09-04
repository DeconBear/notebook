# -*- coding: utf-8 -*-
"""路径一 · GAN 练习：判别器/生成器损失与模式坍塌。"""

import torch
import torch.nn.functional as F


def gan_discriminator_loss(d_real_pred: torch.Tensor,
                            d_fake_pred: torch.Tensor) -> torch.Tensor:
    """
    TODO: L_D = BCE(D(x), 1) + BCE(D(G(z)), 0)  （可取平均）
    """
    return torch.tensor(0.0)


def gan_generator_loss(d_fake_pred: torch.Tensor) -> torch.Tensor:
    """
    TODO: 非饱和损失 L_G = BCE(D(G(z)), 1)
    """
    return torch.tensor(0.0)


def explain_mode_collapse() -> str:
    """TODO: 用自己的话解释模式坍塌、为何发生、两种缓解办法。"""
    return "请在此处写下你对模式坍塌的理解。"


if __name__ == "__main__":
    print("=" * 50)
    print("GAN 练习")
    print("=" * 50)
    d_real = torch.tensor([[0.9], [0.8], [0.95]])
    d_fake = torch.tensor([[0.1], [0.2], [0.05]])
    print("D loss:", gan_discriminator_loss(d_real, d_fake).item())
    print("G loss:", gan_generator_loss(d_fake).item())
    print(explain_mode_collapse())
