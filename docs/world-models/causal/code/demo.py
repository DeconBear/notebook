# -*- coding: utf-8 -*-
"""
=== 因果世界模型直觉：观测相关 vs do(a) ===
SCM: Z -> A（观测策略），Y := A + 噪声（真正机制不含 Z）。
对比「用 Z 预测 Y」与「用 A 预测 Y」在干预分布上的误差。
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
np.random.seed(42)


def sample_observational(n=2000):
    z = np.random.randn(n)
    # 观测策略：动作几乎由混淆决定
    a = np.tanh(1.5 * z) + 0.1 * np.random.randn(n)
    y = a + 0.15 * np.random.randn(n)  # 机制：只靠 a
    return z, a, y


def sample_interventional(n=2000):
    z = np.random.randn(n)
    a = np.random.uniform(-1.5, 1.5, size=n)  # do(A=a)：切断 Z->A
    y = a + 0.15 * np.random.randn(n)
    return z, a, y


def fit_linear(x, y):
    x1 = np.stack([x, np.ones_like(x)], axis=1)
    w, *_ = np.linalg.lstsq(x1, y, rcond=None)
    return w


def predict(w, x):
    return w[0] * x + w[1]


def mse(y_hat, y):
    return float(np.mean((y_hat - y) ** 2))


def main():
    print('=== 观测相关 vs 干预 ===')
    z_tr, a_tr, y_tr = sample_observational()
    w_z = fit_linear(z_tr, y_tr)   # 关联模型：用混淆预测结果
    w_a = fit_linear(a_tr, y_tr)   # 因果模型：用动作预测结果

    z_te, a_te, y_te = sample_observational()
    z_iv, a_iv, y_iv = sample_interventional()

    rows = [
        ('观测分布 / 用 Z', mse(predict(w_z, z_te), y_te)),
        ('观测分布 / 用 A', mse(predict(w_a, a_te), y_te)),
        ('干预分布 / 用 Z', mse(predict(w_z, z_iv), y_iv)),
        ('干预分布 / 用 A', mse(predict(w_a, a_iv), y_iv)),
    ]
    for name, err in rows:
        print(f'{name}: MSE={err:.4f}')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(z_tr[::5], y_tr[::5], s=8, alpha=0.4, label='观测 (Z,Y)')
    xs = np.linspace(z_tr.min(), z_tr.max(), 100)
    axes[0].plot(xs, predict(w_z, xs), 'C1', lw=2, label='用 Z 拟合')
    axes[0].set_xlabel('Z（混淆）')
    axes[0].set_ylabel('Y')
    axes[0].set_title('观测数据上：Z 看似能预测 Y')
    axes[0].legend()

    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = ['#5B8FF9', '#5AD8A6', '#E8684A', '#5AD8A6']
    axes[1].barh(labels[::-1], vals[::-1], color=colors[::-1])
    axes[1].set_xlabel('MSE')
    axes[1].set_title('关键：干预时「用 Z」崩、「用 A」稳')
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'causal_obs_vs_do.png')
    fig.savefig(out, dpi=140)
    print('保存', out)


if __name__ == '__main__':
    main()
