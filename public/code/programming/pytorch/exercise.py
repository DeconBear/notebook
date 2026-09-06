# -*- coding: utf-8 -*-
"""
=== PyTorch 练习 ===
实现 mse_and_grad(x)：L=(x^2).mean() 对 x 的梯度。
x shape (n,)，requires_grad 可 True 可 False，返回 (loss标量, grad一维张量)。
运行: python exercise.py
"""
import torch


def mse_and_grad(x: torch.Tensor):
    """
    L = mean(x^2)。返回 (L.detach(), ∂L/∂x 的副本)。
    提示：clone 出 requires_grad=True 的叶子再 backward。
    """
    # TODO
    raise NotImplementedError


def _check():
    x = torch.tensor([1.0, 2.0, 3.0])
    loss, g = mse_and_grad(x)
    assert abs(float(loss) - (1 + 4 + 9) / 3) < 1e-5
    # d/dx_i of mean(x^2) = 2 x_i / n
    expect = 2 * x / x.numel()
    assert torch.allclose(g, expect), (g, expect)
    print('练习通过。grad = 2x / n，backward 只是把这条链式法则跑完。')


if __name__ == '__main__':
    _check()
