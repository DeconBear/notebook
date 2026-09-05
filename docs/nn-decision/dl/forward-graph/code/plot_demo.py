# -*- coding: utf-8 -*-
"""
=== s05 前向传播 — 配图脚本（与教学主线分开） ===
画三张图：激活函数对比、网络结构、各层激活分布。
数学与前向循环在 demo.py；本文件只负责 matplotlib。

运行: python plot_demo.py
"""
import os
from typing import Dict, List

import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from demo import (
    forward_pass,
    initialize_parameters,
    relu,
    relu_derivative,
    sigmoid,
    sigmoid_derivative,
    tanh,
    tanh_derivative,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_IMAGES = os.path.join(_HERE, '..', 'images')
os.makedirs(_IMAGES, exist_ok=True)


def plot_activation_functions():
    """四种激活函数及其导数（不依赖本次前向的权重）。"""
    z = np.linspace(-5, 5, 1000)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    funcs = [
        ("ReLU", relu, relu_derivative, "max(0, z)", "#2E86AB"),
        ("Sigmoid", sigmoid, sigmoid_derivative, "1/(1+e^{-z})", "#A23B72"),
        ("Tanh", tanh, tanh_derivative, "tanh(z)", "#F18F01"),
        ("Leaky ReLU (α=0.01)", lambda z: np.maximum(0, z) + 0.01 * np.minimum(0, z),
         lambda z: np.where(z > 0, 1.0, 0.01), "max(0,z)+0.01*min(0,z)", "#C73E1D"),
    ]

    for ax, (name, fn, fn_prime, formula, color) in zip(axes, funcs):
        y = fn(z)
        dy = fn_prime(z)
        ax.plot(z, y, 'b-', linewidth=2.5, label=f'{name}: f(z)')
        ax.plot(z, dy, 'r--', linewidth=2, label=f"{name}: f'(z)")
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlim(-5, 5)
        ax.set_title(f'{name}\n{formula}', fontsize=12, fontweight='bold')
        ax.set_xlabel('z', fontsize=10)
        ax.set_ylabel("f(z) / f'(z)", fontsize=10)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Common Activation Functions and Their Derivatives', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(_IMAGES, 'activation_functions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[可视化] 激活函数对比图已保存至 " + os.path.join(_IMAGES, 'activation_functions.png'))


def plot_network_structure(parameters: Dict[str, np.ndarray], X_sample: np.ndarray):
    """按本次网络的层宽画计算图视角的结构示意。"""
    L = len(parameters) // 2
    layer_sizes = [X_sample.shape[0]]
    for l in range(1, L + 1):
        layer_sizes.append(parameters[f"W{l}"].shape[0])

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_xlim(-0.5, L + 0.5)
    max_neurons = max(layer_sizes)
    ax.set_ylim(-max_neurons - 0.5, max_neurons + 0.5)

    neuron_positions = []
    for l_idx, n_neurons in enumerate(layer_sizes):
        y_positions = np.linspace(max_neurons / 2 - n_neurons / 2,
                                   -max_neurons / 2 + n_neurons / 2,
                                   max(n_neurons, 1))
        positions = []
        for n_idx, y in enumerate(y_positions):
            if l_idx == 0:
                color = '#4A90D9'
                label = f'x{n_idx+1}'
            elif l_idx == L:
                color = '#E74C3C'
                label = f'ŷ{n_idx+1}'
            else:
                color = '#F39C12'
                label = f'h{l_idx},{n_idx+1}'
            circle = plt.Circle((l_idx, y), 0.25, color=color, ec='white', linewidth=1.5, zorder=5)
            ax.add_patch(circle)
            ax.text(l_idx, y, label, ha='center', va='center', fontsize=7,
                    color='white', fontweight='bold', zorder=6)
            positions.append(y)

        neuron_positions.append((l_idx, positions))
        if l_idx > 0:
            prev_positions = neuron_positions[l_idx - 1][1]
            for prev_y in prev_positions:
                for curr_y in positions:
                    ax.plot([l_idx - 1, l_idx], [prev_y, curr_y],
                            color='gray', alpha=0.2, linewidth=0.5, zorder=1)

        if l_idx == 0:
            layer_name = f'Input Layer\n({n_neurons} neurons)'
        elif l_idx == L:
            layer_name = f'Output Layer\n({n_neurons} neurons)'
        else:
            layer_name = f'Hidden Layer {l_idx}\n({n_neurons} neurons)'
        ax.text(l_idx, max_neurons / 2 + 0.8, layer_name,
                ha='center', fontsize=9, fontweight='bold')

    for l in range(1, L + 1):
        W = parameters[f"W{l}"]
        x_pos = l - 0.5
        ax.annotate(f'W[{l}]\n{W.shape[0]}×{W.shape[1]}',
                    xy=(x_pos, -max_neurons / 2 - 0.3),
                    fontsize=7, ha='center', color='#2C3E50',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F8F5', alpha=0.8))

    ax.set_title('Neural Network Structure - Computation Graph View', fontsize=14, fontweight='bold')
    ax.axis('equal')
    ax.axis('off')
    legend_elements = [
        mpatches.Patch(color='#4A90D9', label='Input Layer'),
        mpatches.Patch(color='#F39C12', label='Hidden Layer'),
        mpatches.Patch(color='#E74C3C', label='Output Layer'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(_IMAGES, 'network_structure.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[可视化] 网络结构图已保存至 " + os.path.join(_IMAGES, 'network_structure.png'))


def plot_forward_data_flow(caches: List[Dict]):
    """各层激活值直方图：看 He 初始化后分布有没有塌掉或炸掉。"""
    L = len(caches)
    fig, axes = plt.subplots(1, L + 1, figsize=(4 * (L + 1), 4))

    a_prev_vals = caches[0]['a_prev'].flatten()
    axes[0].hist(a_prev_vals, bins=30, color='#4A90D9', alpha=0.7, edgecolor='white')
    axes[0].set_title(f'Input Layer a[0]\nshape={caches[0]["a_prev"].shape}', fontsize=10)
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.5)

    for l in range(L):
        a_vals = caches[l]['a'].flatten()
        axes[l + 1].hist(a_vals, bins=30, color='#F39C12', alpha=0.7, edgecolor='white')
        axes[l + 1].set_title(f'Layer {l+1} a[{l+1}]\nshape={caches[l]["a"].shape}', fontsize=10)
        axes[l + 1].set_xlabel('Value')
        axes[l + 1].set_ylabel('Frequency')
        axes[l + 1].axvline(x=0, color='red', linestyle='--', alpha=0.5)

    plt.suptitle('Layer-wise Evolution of Activation Distribution During Forward Propagation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(_IMAGES, 'forward_data_flow.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[可视化] 前向传播数据流图已保存至 " + os.path.join(_IMAGES, 'forward_data_flow.png'))


def main():
    print("s05 配图：激活函数 / 网络结构 / 激活分布")
    plot_activation_functions()

    np.random.seed(0)
    X = np.random.randn(3, 32)
    parameters = initialize_parameters([3, 4, 4, 1], verbose=False)
    _, caches = forward_pass(X, parameters, [relu, relu, sigmoid], verbose=False)
    plot_network_structure(parameters, X[:, 0:1])
    plot_forward_data_flow(caches)


if __name__ == '__main__':
    main()
