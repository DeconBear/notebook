# -*- coding: utf-8 -*-
"""
=== NeuroAI 演示：Hebb 局部更新 vs 线性读出（AND / XOR）===
运行: python demo.py
"""
import os
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)
np.random.seed(0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def hebbian_update(w, pre, post, lr=0.05):
    return w + lr * post * pre


def train_readout(X, y, lr=0.2, n_epochs=250):
    w = np.zeros(X.shape[1])
    b = 0.0
    losses = []
    for _ in range(n_epochs):
        p = sigmoid(X @ w + b)
        eps = 1e-9
        losses.append(float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))))
        w -= lr * (X.T @ (p - y) / len(y))
        b -= lr * float(np.mean(p - y))
    return np.asarray(losses), w, b


def main():
    w = np.array([0.0, 0.0])
    traj = [w.copy()]
    for pre, post in [([1, 0], 1.0), ([0, 1], 0.0), ([1, 1], 1.0), ([1, 0], 1.0)]:
        w = hebbian_update(w, np.array(pre, dtype=float), post)
        traj.append(w.copy())
    traj = np.array(traj)

    X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y_and = np.array([0, 0, 0, 1], dtype=float)
    y_xor = np.array([0, 1, 1, 0], dtype=float)
    loss_and, _, _ = train_readout(X_and, y_and)
    loss_xor, _, _ = train_readout(X_and, y_xor)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(traj[:, 0], '-o', label=r'$w_1$', color='#1a5276')
    axes[0].plot(traj[:, 1], '-s', label=r'$w_2$', color='#bc4c00')
    axes[0].set_title('Hebb：只有共激活的输入被加强')
    axes[0].set_xlabel('更新步')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(loss_and, label='AND（线性可分）', color='#1a7f37', lw=2)
    axes[1].plot(loss_xor, label='XOR（线性不可分）', color='#c0392b', lw=2)
    axes[1].set_title('全局损失读出')
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('BCE')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'neuroai_credit.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)
    print('AND 终损', loss_and[-1], 'XOR 终损', loss_xor[-1])


if __name__ == '__main__':
    main()
