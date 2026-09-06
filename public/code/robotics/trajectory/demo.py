# -*- coding: utf-8 -*-
"""
=== 轨迹规划 ===
同一组起终点：关节线性插值 vs 三次多项式（起停速度 0）。
对比关节角曲线、角速度、以及 FK 后的末端路径。
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

L1, L2 = 1.0, 0.7


def fk(th1, th2):
    return np.array([
        L1 * np.cos(th1) + L2 * np.cos(th1 + th2),
        L1 * np.sin(th1) + L2 * np.sin(th1 + th2),
    ])


def lerp(q0, q1, t, T):
    s = t / T
    return q0 + s * (q1 - q0)


def cubic_zero_vel(q0, q1, t, T):
    """θ(0)=q0, θ(T)=q1, θ̇(0)=θ̇(T)=0。"""
    s = t / T
    # s^2 (3-2s) 是标准平滑阶跃
    a = s * s * (3.0 - 2.0 * s)
    return q0 + a * (q1 - q0)


def cubic_zero_vel_dot(q0, q1, t, T):
    s = t / T
    da_dt = (6.0 * s - 6.0 * s * s) / T
    return da_dt * (q1 - q0)


def main():
    print('=== 关节空间轨迹：线性 vs 三次 ===')
    q0 = np.array([-0.5, 1.4])
    q1 = np.array([1.2, -0.4])
    T = 1.0
    ts = np.linspace(0, T, 80)
    q_lin = np.array([lerp(q0, q1, t, T) for t in ts])
    q_cub = np.array([cubic_zero_vel(q0, q1, t, T) for t in ts])
    w_lin = np.full(len(ts), np.linalg.norm(q1 - q0) / T)
    w_cub = np.array([np.linalg.norm(cubic_zero_vel_dot(q0, q1, t, T)) for t in ts])
    ee_lin = np.array([fk(*q) for q in q_lin])
    ee_cub = np.array([fk(*q) for q in q_cub])
    print(f'起 {np.rad2deg(q0)} deg → 终 {np.rad2deg(q1)} deg')
    print(f'线性角速度常数 |qdot|={w_lin[0]:.3f} rad/s')
    print(f'三次 |qdot| 峰值 {w_cub.max():.3f}（两端为 0）')

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    axes[0].plot(ts, q_lin[:, 0], '--', color='#5B8FF9', label=r'线性 $\theta_1$')
    axes[0].plot(ts, q_lin[:, 1], '--', color='#E8684A', label=r'线性 $\theta_2$')
    axes[0].plot(ts, q_cub[:, 0], color='#2E86AB', label=r'三次 $\theta_1$')
    axes[0].plot(ts, q_cub[:, 1], color='#C1666B', label=r'三次 $\theta_2$')
    axes[0].set_xlabel('t')
    axes[0].set_title('关节角')
    axes[0].legend(fontsize=7)

    axes[1].plot(ts, w_lin, '--', color='#5B8FF9', label='线性 |qdot|')
    axes[1].plot(ts, w_cub, color='#C1666B', label='三次 |qdot|')
    axes[1].set_xlabel('t')
    axes[1].set_title('角速度模')
    axes[1].legend(fontsize=8)

    axes[2].plot(ee_lin[:, 0], ee_lin[:, 1], '--', color='#5B8FF9', lw=2, label='线性关节 → 手')
    axes[2].plot(ee_cub[:, 0], ee_cub[:, 1], color='#C1666B', lw=2, label='三次关节 → 手')
    axes[2].scatter(*fk(*q0), c='k', zorder=5)
    axes[2].scatter(*fk(*q1), c='k', marker='s', zorder=5)
    axes[2].set_aspect('equal')
    axes[2].set_title('末端路径（都不是直线）')
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'traj_joint.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)


if __name__ == '__main__':
    main()
