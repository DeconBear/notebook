# -*- coding: utf-8 -*-
"""
=== PyTorch 张量与自动求导 ===
1) Tensor 形状与矩阵乘
2) backward 对标手算导数
3) 最小线性回归训练环
运行: python demo.py
"""
import os
import torch
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)
torch.manual_seed(42)


def demo_tensor():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    w = torch.tensor([[1.0], [1.0]])
    y = x @ w
    cat = torch.cat([x, y], dim=-1)
    print('x.shape', tuple(x.shape), '  x@w shape', tuple(y.shape))
    print('cat last dim', tuple(cat.shape), '  device', x.device)


def demo_autograd():
    # L = (x^2 + 3x)^2 at x=2 → 2*(4+6)*(4+3)=2*10*7=140
    x = torch.tensor(2.0, requires_grad=True)
    y = x * x + 3 * x
    L = y * y
    L.backward()
    print('autograd dL/dx =', float(x.grad), '  手算 140')


def demo_sgd():
    n = 40
    true_w = torch.tensor([[2.0], [-1.0]])
    X = torch.randn(n, 2)
    y = X @ true_w + 0.05 * torch.randn(n, 1)
    w = torch.zeros(2, 1, requires_grad=True)
    opt = torch.optim.SGD([w], lr=0.1)
    losses = []
    for _ in range(80):
        pred = X @ w
        loss = ((pred - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    print('学到的 w', w.detach().view(-1).tolist(), '  真值 [2, -1]')

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(losses, color='#1F3A5F')
    ax.set_xlabel('step')
    ax.set_ylabel('MSE')
    ax.set_title('没有 nn.Module 也能训：清梯度 → 前向 → backward → step')
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'prog-pytorch-sgd.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)


if __name__ == '__main__':
    demo_tensor()
    demo_autograd()
    demo_sgd()
