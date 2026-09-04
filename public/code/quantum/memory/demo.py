# -*- coding: utf-8 -*-
"""
=== 量子存储 ===
1) T1：激发态 |1> 的布居指数衰减
2) T2：赤道态的相干（非对角元）衰减更快
3) 写-存-读：等待越久，读出保真度越低
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

T1 = 1.0
T2 = 0.4


def amp_damping(rho, t, t1=T1):
    """振幅阻尼信道作用在 2x2 密度矩阵上。"""
    p = 1.0 - np.exp(-t / t1)
    e0 = np.array([[1, 0], [0, np.sqrt(1 - p)]], dtype=complex)
    e1 = np.array([[0, np.sqrt(p)], [0, 0]], dtype=complex)
    return e0 @ rho @ e0.conj().T + e1 @ rho @ e1.conj().T


def dephase(rho, t, t2=T2):
    """纯退相位：非对角元乘 e^{-t/T2}。"""
    out = rho.copy()
    out[0, 1] *= np.exp(-t / t2)
    out[1, 0] *= np.exp(-t / t2)
    return out


def fidelity_pure(rho, psi):
    return float(np.real(psi.conj() @ rho @ psi))


def demo_t1_t2():
    times = np.linspace(0, 3.0, 80)
    rho1 = np.array([[0, 0], [0, 1]], dtype=complex)
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho_plus = np.outer(plus, plus.conj())
    pop1 = [np.real(amp_damping(rho1, t)[1, 1]) for t in times]
    coh = [np.abs(dephase(rho_plus, t)[0, 1]) for t in times]
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.plot(times, pop1, label=r'$T_1$：$\langle 1|\rho|1\rangle$')
    ax.plot(times, coh, label=r'$T_2$：$|\rho_{01}|$（赤道态）')
    ax.set_xlabel('等待时间（任意单位）')
    ax.set_ylabel('残留')
    ax.set_title('存储：能量弛豫 vs 相位噪声')
    ax.legend()
    fig.tight_layout()
    path = os.path.join(_IMAGES_DIR, 't1_t2_decay.png')
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f'已保存 {path}')


def demo_write_store_read():
    psi = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho0 = np.outer(psi, psi.conj())
    times = np.linspace(0, 2.5, 40)
    fids = []
    for t in times:
        rho = dephase(amp_damping(rho0, t), t)
        fids.append(fidelity_pure(rho, psi))
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot(times, fids, color='#c026d3')
    ax.set_xlabel('存储时间')
    ax.set_ylabel('读出保真度')
    ax.set_title('写-存-读：等得越久越糊')
    fig.tight_layout()
    path = os.path.join(_IMAGES_DIR, 'write_store_read.png')
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f'已保存 {path}')


if __name__ == '__main__':
    demo_t1_t2()
    demo_write_store_read()
    print('完成：T1/T2 与写-存-读保真度。')
