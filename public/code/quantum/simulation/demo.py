# -*- coding: utf-8 -*-
"""
=== 量子模拟 ===
两自旋横场 Ising：精确演化 vs 一阶 Trotter，看步数如何压低误差。
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

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
ZZ = np.kron(Z, Z)
XI = np.kron(X, I2)
IX = np.kron(I2, X)

J, h = 1.0, 0.7
A = J * ZZ          # 对角相互作用
B = h * (XI + IX)   # 横场
Htot = A + B
T = 1.2


def expm_herm(m, t):
    """e^{-i M t}，M 厄米。"""
    w, v = np.linalg.eigh(m)
    return v @ np.diag(np.exp(-1j * w * t)) @ v.conj().T


def trotter(n_steps, t=T):
    dt = t / n_steps
    u = np.eye(4, dtype=complex)
    ua, ub = expm_herm(A, dt), expm_herm(B, dt)
    for _ in range(n_steps):
        u = ub @ ua @ u
    return u


def demo():
    psi0 = np.array([1, 0, 0, 0], dtype=complex)
    u_exact = expm_herm(Htot, T)
    psi_exact = u_exact @ psi0
    steps = np.array([1, 2, 4, 8, 16, 32])
    infid = []
    for n in steps:
        psi = trotter(int(n)) @ psi0
        infid.append(1.0 - np.abs(np.vdot(psi_exact, psi)) ** 2)
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.loglog(steps, infid, 'o-', color='#2563eb')
    ax.set_xlabel('Trotter 步数')
    ax.set_ylabel('相对精确演化的失真度')
    ax.set_title('数字模拟：多切几刀更接近 $e^{-iHt}$')
    fig.tight_layout()
    path = os.path.join(_IMAGES_DIR, 'trotter_error.png')
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print('步数 → 失真度:', list(zip(steps.tolist(), np.round(infid, 6))))
    print(f'已保存 {path}')


if __name__ == '__main__':
    demo()
    print('完成：Trotter vs 精确对角化。')
