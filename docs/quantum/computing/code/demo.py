# -*- coding: utf-8 -*-
"""
=== 量子计算 ===
1) 单比特：Hadamard 把 |0> 打成均匀叠加，测量直方图接近 50/50
2) 两比特：H + CNOT 制备 Bell 态 |Φ+>，关联测量
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
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
], dtype=complex)


def ket(*bits):
    """计算基向量 |b0 b1 ...> 的张量积。"""
    v = np.array([1.0], dtype=complex)
    for b in bits:
        v = np.kron(v, np.array([1, 0] if b == 0 else [0, 1], dtype=complex))
    return v


def measure_shots(state, n_shots=2000):
    """按 Born 规则对计算基抽样。"""
    p = np.abs(state) ** 2
    p = np.real(p)
    p = p / p.sum()
    return np.random.choice(len(state), size=n_shots, p=p)


def demo_hadamard():
    psi = H @ ket(0)
    shots = measure_shots(psi)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    counts = np.bincount(shots, minlength=2)
    ax.bar(['|0>', '|1>'], counts / counts.sum(), color=['#3b82f6', '#f97316'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('频率')
    ax.set_title(r'$H|0\rangle$ 的测量：接近 50/50')
    fig.tight_layout()
    path = os.path.join(_IMAGES_DIR, 'hadamard_shots.png')
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f'Hadamard 振幅: {psi}')
    print(f'已保存 {path}')


def demo_bell():
    psi = CNOT @ np.kron(H, I2) @ ket(0, 0)
    shots = measure_shots(psi)
    labels = ['00', '01', '10', '11']
    counts = np.bincount(shots, minlength=4) / len(shots)
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.bar(labels, counts, color='#6366f1')
    ax.set_ylabel('频率')
    ax.set_title(r'Bell 态 $|Φ^+\rangle$：只出现 00 与 11')
    fig.tight_layout()
    path = os.path.join(_IMAGES_DIR, 'bell_shots.png')
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f'Bell 振幅: {np.round(psi, 3)}')
    print(f'已保存 {path}')


if __name__ == '__main__':
    demo_hadamard()
    demo_bell()
    print('完成：叠加（单比特）与纠缠（Bell 态）。')
