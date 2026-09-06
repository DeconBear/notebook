# -*- coding: utf-8 -*-
"""
=== 导数与微分 ===
1) 割线贴成切线（不同 h）
2) 对偶数 / 多项式系数 / 中心差分 对照解析导数
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


class Dual:
    """对偶数：值 + 导数。乘法用 uv 的乘积法则。"""
    def __init__(self, v, d):
        self.v = float(v)
        self.d = float(d)

    @staticmethod
    def var(x):
        return Dual(x, 1.0)

    def __add__(self, other):
        other = other if isinstance(other, Dual) else Dual(other, 0.0)
        return Dual(self.v + other.v, self.d + other.d)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        other = other if isinstance(other, Dual) else Dual(other, 0.0)
        return Dual(self.v - other.v, self.d - other.d)

    def __mul__(self, other):
        other = other if isinstance(other, Dual) else Dual(other, 0.0)
        return Dual(self.v * other.v, self.v * other.d + self.d * other.v)

    def __rmul__(self, other):
        return self * other


def f_scalar(x):
    return x ** 3 - 2.0 * x


def f_prime_exact(x):
    return 3.0 * x ** 2 - 2.0


def dual_f(x):
    t = Dual.var(x)
    y = t * t * t - 2.0 * t
    return y.v, y.d


def poly_diff(c):
    """c[i] = x^i 的系数 → 导函数系数。"""
    return np.array([i * c[i] for i in range(1, len(c))], dtype=float)


def poly_eval(c, x):
    return sum(c[i] * x ** i for i in range(len(c)))


def demo_secant():
    a = 1.2
    xs = np.linspace(-0.2, 2.2, 300)
    ys = f_scalar(xs)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    hs = [0.8, 0.25, 0.0]
    titles = [r'$h=0.8$ 割线', r'$h=0.25$ 更贴', r'$h\to 0$ 切线']
    for ax, h, title in zip(axes, hs, titles):
        ax.plot(xs, ys, color='#2E86AB', lw=2)
        ax.scatter([a], [f_scalar(a)], c='#E8684A', s=40, zorder=5)
        if h > 0:
            x2 = a + h
            y1, y2 = f_scalar(a), f_scalar(x2)
            slope = (y2 - y1) / h
            line_x = np.array([a - 0.6, a + 1.0])
            line_y = y1 + slope * (line_x - a)
            ax.plot(line_x, line_y, color='#C1666B', ls='--', lw=2)
            ax.scatter([x2], [y2], c='#333', s=28, zorder=5)
        else:
            slope = f_prime_exact(a)
            line_x = np.array([a - 0.6, a + 1.0])
            line_y = f_scalar(a) + slope * (line_x - a)
            ax.plot(line_x, line_y, color='#C1666B', lw=2)
        ax.set_title(title)
        ax.set_xlim(-0.2, 2.2)
        ax.set_ylim(-2.5, 6)
        ax.axhline(0, color='#ccc', lw=0.6)
        ax.axvline(0, color='#ccc', lw=0.6)
    axes[0].set_ylabel(r'$x^3-2x$')
    fig.suptitle('导数思想：让第二点贴过来，割线变成切线', y=1.03)
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'deriv_secant.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print('保存', out)


def demo_methods():
    x0 = 2.0
    exact = f_prime_exact(x0)
    _, autod = dual_f(x0)
    c = np.array([0.0, -2.0, 0.0, 1.0])  # -2x + x^3
    poly_p = poly_eval(poly_diff(c), x0)
    hs = np.logspace(-1, -8, 8)
    fd = [(f_scalar(x0 + h) - f_scalar(x0 - h)) / (2 * h) for h in hs]
    print(f'x=2  exact={exact:.6f}  dual={autod:.6f}  poly={poly_p:.6f}')

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.semilogx(hs, np.abs(np.array(fd) - exact), 'o-', color='#2E86AB',
                label='中心差分 |误差|')
    ax.axhline(0, color='#ccc', lw=0.5)
    ax.set_xlabel('h')
    ax.set_ylabel('绝对误差')
    ax.set_title('数值微分：h 太小会被浮点噪声咬一口')
    ax.legend()
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'deriv_fd_error.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)
    for h, val in zip(hs, fd):
        print(f'  h={h:.0e}  fd={val:.8f}  err={val - exact:.2e}')


def main():
    print('=== 导数与微分 ===')
    demo_secant()
    demo_methods()


if __name__ == '__main__':
    main()
