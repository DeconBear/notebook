# -*- coding: utf-8 -*-
"""
=== 量子网络 ===
1) 量子隐形传态：随机 Bloch 态平均保真度应接近 1
2) BB84 直觉：无窃听误码低，Eve 拦截-重发会抬高误码
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


def ket0():
    return np.array([1, 0], dtype=complex)


def random_qubit():
    """Haar 随机单比特纯态。"""
    z = np.random.randn(2) + 1j * np.random.randn(2)
    return z / np.linalg.norm(z)


def apply_3(op, state, wires):
    """把 1 或 2 比特门嵌进 3 比特态（wires 为小端：0=Alice数据, 1=Alice纠缠, 2=Bob）。"""
    n = 3
    # 用显式基展开，教学优先于速度
    dim = 2 ** n
    out = np.zeros(dim, dtype=complex)
    if op.shape == (2, 2):
        w = wires[0]
        for i in range(dim):
            bits = [(i >> k) & 1 for k in range(n)]
            b = bits[w]
            for nb, amp in enumerate(op[:, b]):
                if abs(amp) < 1e-15:
                    continue
                bits2 = bits.copy()
                bits2[w] = nb
                j = sum(bit << k for k, bit in enumerate(bits2))
                out[j] += amp * state[i]
        return out
    # 2-qubit gate on (c, t)
    c, t = wires
    for i in range(dim):
        bits = [(i >> k) & 1 for k in range(n)]
        local = bits[c] * 2 + bits[t]
        col = op[:, local]
        for nl, amp in enumerate(col):
            if abs(amp) < 1e-15:
                continue
            bits2 = bits.copy()
            bits2[c] = (nl >> 1) & 1
            bits2[t] = nl & 1
            j = sum(bit << k for k, bit in enumerate(bits2))
            out[j] += amp * state[i]
    return out


def teleport_once(psi):
    """把 psi 从比特 0 传到比特 2，返回 Bob 的约化态（纯态矢量）。"""
    # |ψ⟩ ⊗ |Φ+⟩_{12}
    bell = CNOT @ np.kron(H, I2) @ np.kron(ket0(), ket0())
    state = np.kron(psi, bell)
    # Alice: CNOT(0->1), H(0)
    state = apply_3(CNOT, state, (0, 1))
    state = apply_3(H, state, (0,))
    # 测量比特 0、1（投影），对比特 2 做 X^m1 Z^m0
    probs = np.real(np.abs(state) ** 2)
    outcome = np.random.choice(8, p=probs / probs.sum())
    m0 = outcome & 1
    m1 = (outcome >> 1) & 1
    # 取出比特 2 的未归一化振幅：固定 m0,m1
    bob = np.zeros(2, dtype=complex)
    for i, amp in enumerate(state):
        if (i & 1) == m0 and ((i >> 1) & 1) == m1:
            bob[(i >> 2) & 1] += amp
    nrm = np.linalg.norm(bob)
    if nrm < 1e-12:
        return psi
    bob = bob / nrm
    if m1:
        bob = X @ bob
    if m0:
        bob = Z @ bob
    return bob


def fidelity(a, b):
    return float(np.abs(np.vdot(a, b)) ** 2)


def demo_teleport(n=40):
    fs = [fidelity(psi, teleport_once(psi)) for psi in (random_qubit() for _ in range(n))]
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.plot(fs, 'o-', color='#0d9488')
    ax.axhline(1.0, color='#94a3b8', ls='--', label='理想 = 1')
    ax.set_ylim(0.7, 1.05)
    ax.set_xlabel('随机未知态编号')
    ax.set_ylabel('传态保真度')
    ax.set_title('量子隐形传态（无噪声模拟）')
    ax.legend()
    fig.tight_layout()
    path = os.path.join(_IMAGES_DIR, 'teleport_fidelity.png')
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f'平均传态保真度: {np.mean(fs):.4f}')
    print(f'已保存 {path}')


def bb84_qber(n_bits=800, eve=False):
    """简化 BB84：匹配基保留；Eve 拦截-重发时在随机基测量再转发。"""
    alice_bits = np.random.randint(0, 2, n_bits)
    alice_bases = np.random.randint(0, 2, n_bits)  # 0=Z, 1=X
    bob_bases = np.random.randint(0, 2, n_bits)
    received = alice_bits.copy()
    if eve:
        eve_bases = np.random.randint(0, 2, n_bits)
        # 基不一致则 50% 翻比特
        mismatch = eve_bases != alice_bases
        flip = mismatch & (np.random.rand(n_bits) < 0.5)
        received = np.bitwise_xor(received, flip.astype(int))
        # Bob 再测：相对 Eve 转发态，基不一致再 50% 错
        mismatch_b = bob_bases != eve_bases
        flip_b = mismatch_b & (np.random.rand(n_bits) < 0.5)
        received = np.bitwise_xor(received, flip_b.astype(int))
    else:
        mismatch_b = bob_bases != alice_bases
        flip_b = mismatch_b & (np.random.rand(n_bits) < 0.5)
        received = np.bitwise_xor(received, flip_b.astype(int))
    sift = alice_bases == bob_bases
    if not np.any(sift):
        return 0.0
    return float(np.mean(received[sift] != alice_bits[sift]))


def demo_bb84():
    qber_clean = [bb84_qber(eve=False) for _ in range(12)]
    qber_eve = [bb84_qber(eve=True) for _ in range(12)]
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.boxplot([qber_clean, qber_eve])
    ax.set_xticklabels(['无窃听', 'Eve 拦截-重发'])
    ax.set_ylabel('筛后误码率')
    ax.set_title('BB84：偷听会抬高误码')
    fig.tight_layout()
    path = os.path.join(_IMAGES_DIR, 'bb84_qber.png')
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f'无窃听 QBER≈{np.mean(qber_clean):.3f}，有 Eve≈{np.mean(qber_eve):.3f}')
    print(f'已保存 {path}')


if __name__ == '__main__':
    demo_teleport()
    demo_bb84()
    print('完成：传态保真度与 BB84 误码对比。')
