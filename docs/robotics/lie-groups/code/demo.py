# -*- coding: utf-8 -*-
"""
=== 李群 SO(3)：指数/对数映射 ===
ω̂ ∈ so(3) 反对称 → R = exp(ω̂) ∈ SO(3)。对照 C++：so3.hpp / demo.cpp。
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


def hat(w):
    """R^3 → so(3)：叉乘矩阵。"""
    x, y, z = w
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def so3_exp(w):
    """Rodrigues：R = I + sinc(θ) ω̂ + (1-cosθ)/θ² ω̂²。"""
    th = np.linalg.norm(w)
    K = hat(w)
    if th < 1e-10:
        return np.eye(3) + K
    return np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / (th * th) * (K @ K)


def so3_log(R):
    """主值轴角；输入须为 SO(3)，分别处理小角和接近 π 的退化。"""
    c = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    vee = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0],
                    R[1, 0] - R[0, 1]])
    sin_th = 0.5 * np.linalg.norm(vee)
    th = np.arctan2(sin_th, c)
    if th < 1e-7:
        # θ/(2 sinθ) → 1/2；保留微小旋转，不能直接返回零。
        return 0.5 * vee
    if np.pi - th < 1e-6:
        # 对称部分沿转轴的特征值为 1，另两个为 cosθ。
        # 这避免了在 π 附近除以几乎为零的 sinθ。
        _, axes = np.linalg.eigh(0.5 * (R + R.T))
        axis = axes[:, -1]
        if sin_th > 1e-12:
            if np.dot(axis, vee) < 0.0:
                axis = -axis
        elif axis[np.argmax(np.abs(axis))] < 0.0:
            axis = -axis  # 精确 π 时 ±轴等价，固定一种符号。
        return th * axis
    return (th / (2.0 * sin_th)) * vee

def main():
    print('=== SO(3) exp/log ===')
    w = np.array([0.3, -0.1, 0.8])
    R = so3_exp(w)
    w2 = so3_log(R)
    print('ω        ', w)
    print('log(exp) ', w2)
    print('R^T R ≈ I', np.round(R.T @ R, 6))
    print('det R    ', np.linalg.det(R))

    # 把立方体顶点转一圈
    cube = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
                     [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]], dtype=float) * 0.4
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(cube[:, 0], cube[:, 1], cube[:, 2], c='gray', label='原')
    rot = (so3_exp(w) @ cube.T).T
    ax.scatter(rot[:, 0], rot[:, 1], rot[:, 2], c='#E67E22', label='R p')
    ax.quiver(0, 0, 0, w[0], w[1], w[2], color='#2980B9', lw=2)
    ax.set_title(r'$\exp(\hat{\omega})$ 把点绕轴旋转')
    ax.legend()
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'so3_exp.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)


if __name__ == '__main__':
    main()
