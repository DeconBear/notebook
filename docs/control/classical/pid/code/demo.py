# -*- coding: utf-8 -*-
"""
=== 经典控制：质量-弹簧-阻尼 + PID ===
二阶植物 mẍ + cẋ + kx = u。对比开环、P、PD、PID 的阶跃响应。
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

M, C, K = 1.0, 0.4, 2.0
DT = 0.01
T_END = 8.0
R = 1.0  # 阶跃目标位置


def plant_step(x, v, u):
    """欧拉积分：ẍ = (u - c v - k x) / m。"""
    a = (u - C * v - K * x) / M
    v = v + DT * a
    x = x + DT * v
    return x, v


def simulate(kp=0.0, ki=0.0, kd=0.0, open_loop=False):
    n = int(T_END / DT)
    xs, us, ts = [], [], []
    x, v, integ, e_prev = 0.0, 0.0, 0.0, 0.0
    for i in range(n):
        t = i * DT
        if open_loop:
            u = R * K  # 稳态力刚好抵弹簧，但暂态全靠植物
        else:
            e = R - x
            integ += e * DT
            de = (e - e_prev) / DT
            u = kp * e + ki * integ + kd * de
            e_prev = e
        x, v = plant_step(x, v, u)
        xs.append(x)
        us.append(u)
        ts.append(t)
    return np.array(ts), np.array(xs), np.array(us)


def overshoot(x):
    return float(max(0.0, np.max(x) - R) / R * 100.0)


def ss_error(x):
    return float(abs(x[-1] - R))


def main():
    print('=== 经典控制：PID 阶跃 ===')
    cases = [
        ('开环', dict(open_loop=True)),
        ('P  kp=8', dict(kp=8.0)),
        ('PD kp=8 kd=4', dict(kp=8.0, kd=4.0)),
        ('PID kp=8 ki=3 kd=4', dict(kp=8.0, ki=3.0, kd=4.0)),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, kw in cases:
        t, x, u = simulate(**kw)
        axes[0].plot(t, x, lw=2, label=name)
        axes[1].plot(t, u, lw=1.4, label=name)
        print(f'{name:20s}  超调={overshoot(x):5.1f}%  静差={ss_error(x):.4f}')

    axes[0].axhline(R, color='k', ls='--', alpha=0.4, label='参考 r=1')
    axes[0].set_xlabel('时间 t')
    axes[0].set_ylabel('位置 x')
    axes[0].set_title('阶跃响应')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel('时间 t')
    axes[1].set_ylabel('控制力 u')
    axes[1].set_title('控制量')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'pid_step.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)


if __name__ == '__main__':
    main()
