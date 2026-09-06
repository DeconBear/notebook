# -*- coding: utf-8 -*-
"""
=== 积分与基本定理 ===
1) 黎曼和把面积叠出来
2) 梯形法则 vs 多项式原函数；FTC：F' = f
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


def f(x):
    return x ** 2


def F_exact(x):
    return x ** 3 / 3.0


def riemann_left(a, b, n):
    h = (b - a) / n
    xs = a + h * np.arange(n)
    return float(np.sum(f(xs) * h)), xs, h


def trapezoid(a, b, n):
    xs = np.linspace(a, b, n + 1)
    ys = f(xs)
    h = (b - a) / n
    return float(h * (0.5 * ys[0] + 0.5 * ys[-1] + np.sum(ys[1:-1])))


def demo_riemann():
    a, b = 0.0, 1.0
    xs_curve = np.linspace(a, b, 200)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, n in zip(axes, [4, 8, 20]):
        s, left, h = riemann_left(a, b, n)
        ax.plot(xs_curve, f(xs_curve), color='#C1666B', lw=2, zorder=3)
        ax.bar(left, f(left), width=h, align='edge', color='#5B8FF9',
               edgecolor='white', alpha=0.85)
        ax.set_title(f'n={n}  和={s:.4f}')
        ax.set_xlim(a, b)
        ax.set_ylim(0, 1.15)
    axes[0].set_ylabel(r'$x^2$')
    fig.suptitle(r'积分思想：薄片越薄，和越接近 $\int_0^1 x^2=1/3$', y=1.03)
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'int_riemann.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print('保存', out)


def demo_ftc():
    xs = np.linspace(0.0, 1.5, 80)
    # 用梯形累加近似 F(x)=∫_0^x t^2 dt
    F_num = np.array([trapezoid(0.0, x, 40) if x > 0 else 0.0 for x in xs])
    # 数值导数
    dF = np.gradient(F_num, xs)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.7))
    axes[0].plot(xs, F_exact(xs), color='#2E86AB', lw=2, label=r'$x^3/3$')
    axes[0].plot(xs, F_num, '--', color='#E8684A', lw=2, label='梯形累加')
    axes[0].set_title('原函数：累加到 x 的净面积')
    axes[0].legend()
    axes[0].set_xlabel('x')

    axes[1].plot(xs, f(xs), color='#2E86AB', lw=2, label=r'$f=x^2$')
    axes[1].plot(xs, dF, '--', color='#E8684A', lw=2, label="数值 $F'$")
    axes[1].set_title('基本定理：$F\'=f$')
    axes[1].legend()
    axes[1].set_xlabel('x')
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'int_ftc.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)
    print(f'梯形 n=16 ∫0^1 x^2 = {trapezoid(0,1,16):.6f}  exact={1/3:.6f}')


def main():
    print('=== 积分与基本定理 ===')
    demo_riemann()
    demo_ftc()


if __name__ == '__main__':
    main()
