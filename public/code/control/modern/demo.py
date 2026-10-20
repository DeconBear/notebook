# -*- coding: utf-8 -*-
"""
=== 现代控制：双积分器 LQR + 卡尔曼 ===
植物 ẍ = u。对比无控制、LQR 全状态、以及「只测位置 + 卡尔曼」再 LQR。
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

DT = 0.05
N = 80
# 离散双积分器：x=[位置, 速度]
A = np.array([[1.0, DT], [0.0, 1.0]])
B = np.array([[0.5 * DT ** 2], [DT]])
H = np.array([[1.0, 0.0]])  # 只测位置
Q = np.diag([4.0, 0.2])
R = np.array([[0.15]])
QN = np.diag([1e-4, 1e-3])  # 过程噪声
RN = np.array([[0.04]])     # 测量噪声


def dare_lqr(A, B, Q, R, iters=200):
    """迭代离散代数 Riccati，得到 P，再 K = (R+B'PB)^{-1} B'PA。"""
    P = Q.copy()
    for _ in range(iters):
        BtP = B.T @ P
        P = Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.solve(R + BtP @ B, BtP @ A)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K


def simulate(use_lqr=True, use_kf=False, x0=None):
    K = dare_lqr(A, B, Q, R)
    x = np.array([-1.2, 0.8]) if x0 is None else np.array(x0, dtype=float)
    xhat = np.array([0.0, 0.0])
    P = np.eye(2)
    xs, xhats, us = [x.copy()], [xhat.copy()], []
    for _ in range(N):
        if not use_lqr:
            u = 0.0
        elif use_kf:
            u = float((-K @ xhat).item())
        else:
            u = float((-K @ x).item())
        w = np.random.multivariate_normal([0, 0], QN)
        x = A @ x + B.ravel() * u + w
        z = float((H @ x).item()) + np.random.normal(0.0, np.sqrt(RN[0, 0]))
        if use_kf:
            # 预测
            xhat = A @ xhat + B.ravel() * u
            P = A @ P @ A.T + QN
            # 更新
            S = H @ P @ H.T + RN
            Kg = (P @ H.T) / S
            xhat = xhat + Kg.ravel() * (z - float((H @ xhat).item()))
            P = (np.eye(2) - Kg @ H) @ P
        xs.append(x.copy())
        xhats.append(xhat.copy())
        us.append(u)
    return np.array(xs), np.array(xhats), np.array(us), K


def main():
    print('=== 现代控制：LQR + 卡尔曼 ===')
    xs0, _, us0, K = simulate(use_lqr=False)
    xs1, _, us1, _ = simulate(use_lqr=True, use_kf=False)
    xs2, xh2, us2, _ = simulate(use_lqr=True, use_kf=True)
    print('LQR 增益 K =', K.ravel())
    print('无控制终态', xs0[-1])
    print('全状态 LQR 终态', xs1[-1])
    print('卡尔曼+LQR 终态', xs2[-1])

    t = np.arange(len(xs0)) * DT
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(t, xs0[:, 0], color='#95A5A6', label='无控制')
    axes[0].plot(t, xs1[:, 0], color='#27AE60', lw=2, label='LQR（真状态）')
    axes[0].plot(t, xs2[:, 0], color='#2980B9', lw=2, label='LQR + 卡尔曼')
    axes[0].plot(t, xh2[:, 0], color='#2980B9', ls='--', alpha=0.7, label='卡尔曼位置估计')
    axes[0].axhline(0.0, color='k', ls=':', alpha=0.4)
    axes[0].set_xlabel('时间')
    axes[0].set_ylabel('位置')
    axes[0].set_title('把双积分器拉回原点')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(us1, color='#27AE60', label='全状态 u')
    axes[1].plot(us2, color='#2980B9', label='滤波后 u')
    axes[1].set_xlabel('步')
    axes[1].set_ylabel('加速度指令 u')
    axes[1].set_title('控制量')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'lqr_kalman.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)


if __name__ == '__main__':
    main()
