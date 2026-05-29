# -*- coding: utf-8 -*-
"""
s05 计算图与前向传播 — 演示代码
==================================
功能：用纯 NumPy 构建一个 3 层 MLP，展示完整的前向传播过程，
      包括中间值存储、计算图可视化（打印张量形状）、
      以及不同激活函数的对比。

每个函数都有中文 docstring，每行逻辑代码都有中文注释。
运行方式：在 s05_forward_computation_graph/ 目录下执行 python code/demo.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.patches as mpatches
from typing import Dict, List, Tuple, Callable

_HERE = os.path.dirname(os.path.abspath(__file__))
_IMAGES = os.path.join(_HERE, '..', 'images')
os.makedirs(_IMAGES, exist_ok=True)

# ============================================================================
# 第一部分：激活函数及其导数
# ============================================================================

def relu(z: np.ndarray) -> np.ndarray:
    """
    ReLU 激活函数：f(z) = max(0, z)

    参数:
        z: 输入数组，任意形状
    返回:
        逐元素应用的 ReLU 结果
    """
    return np.maximum(0, z)  # max(0, z) 的向量化实现


def relu_derivative(z: np.ndarray) -> np.ndarray:
    """
    ReLU 的导数：f'(z) = 1 if z > 0 else 0

    参数:
        z: 输入数组（前向传播时存储的 z 值）
    返回:
        逐元素的导数，形状与 z 相同
    """
    return (z > 0).astype(np.float64)  # z>0 处导数为 1，其余为 0


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Sigmoid 激活函数：f(z) = 1 / (1 + e^{-z})

    参数:
        z: 输入数组，任意形状
    返回:
        逐元素应用的 sigmoid 结果，范围 (0, 1)
    """
    # 为防止数值溢出，对 z 进行裁剪
    z_clipped = np.clip(z, -500, 500)  # 限制 z 的范围，避免 exp 溢出
    return 1.0 / (1.0 + np.exp(-z_clipped))  # sigmoid 公式


def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    """
    Sigmoid 的导数：f'(z) = f(z) * (1 - f(z))

    参数:
        z: 输入数组（前向传播时存储的 z 值）
    返回:
        逐元素的导数
    """
    s = sigmoid(z)  # 先计算 sigmoid(z)
    return s * (1 - s)  # 利用 f(z) 直接计算导数


def tanh(z: np.ndarray) -> np.ndarray:
    """
    Tanh 激活函数：f(z) = (e^z - e^{-z}) / (e^z + e^{-z})

    参数:
        z: 输入数组，任意形状
    返回:
        逐元素应用的 tanh 结果，范围 (-1, 1)
    """
    return np.tanh(z)  # NumPy 内置的 tanh 已经数值稳定


def tanh_derivative(z: np.ndarray) -> np.ndarray:
    """
    Tanh 的导数：f'(z) = 1 - f(z)^2 = 1 - tanh^2(z)

    参数:
        z: 输入数组（前向传播时存储的 z 值）
    返回:
        逐元素的导数
    """
    t = np.tanh(z)  # 计算 tanh(z)
    return 1 - t ** 2  # tanh 导数的简洁形式


def gelu_approximate(z: np.ndarray) -> np.ndarray:
    """
    GELU 激活函数的近似实现：f(z) ≈ 0.5 * z * (1 + tanh(√(2/π) * (z + 0.044715 * z^3)))

    这是 GELU 的高精度近似，被广泛使用。

    参数:
        z: 输入数组，任意形状
    返回:
        逐元素应用的 GELU 近似结果
    """
    # GELU tanh 近似公式的系数
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)  # √(2/π) ≈ 0.7979
    return 0.5 * z * (1.0 + np.tanh(sqrt_2_over_pi * (z + 0.044715 * z ** 3)))


# ============================================================================
# 第二部分：参数初始化
# ============================================================================

def initialize_parameters(layer_dims: List[int], seed: int = 42) -> Dict[str, np.ndarray]:
    """
    使用 He 初始化方法为每一层创建权重和偏置。

    He 初始化（Kaiming He, 2015）：W 服从 N(0, sqrt(2/n_in))，
    特别适合配合 ReLU 激活函数使用，可以有效缓解梯度消失/爆炸。

    参数:
        layer_dims: 每层神经元数量的列表，如 [3, 4, 4, 1]
        seed: 随机种子，保证结果可复现

    返回:
        parameters: 字典，包含每一层的 W{layer} 和 b{layer}
            - W1, b1: 第 1 层（输入→隐藏层1）
            - W2, b2: 第 2 层（隐藏层1→隐藏层2）
            - W3, b3: 第 3 层（隐藏层2→输出层）
    """
    np.random.seed(seed)  # 固定随机种子，保证每次运行结果一致
    parameters = {}  # 初始化参数字典
    L = len(layer_dims)  # 总层数（包含输入层）

    for l in range(1, L):  # 遍历每一层（跳过输入层 l=0）
        n_in = layer_dims[l - 1]   # 当前层的输入维度
        n_out = layer_dims[l]      # 当前层的输出维度
        # He 初始化：标准差为 sqrt(2/n_in)
        parameters[f"W{l}"] = np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in)  # 权重矩阵 (n_out x n_in)
        parameters[f"b{l}"] = np.zeros((n_out, 1))  # 偏置向量初始化为 0 (n_out x 1)
        print(f"  初始化 W{l}: shape={parameters[f'W{l}'].shape}, He init (std={np.sqrt(2.0/n_in):.4f})")
        print(f"  初始化 b{l}: shape={parameters[f'b{l}'].shape}, 全零初始化")

    return parameters


# ============================================================================
# 第三部分：前向传播
# ============================================================================

def forward_pass(
    X: np.ndarray,
    parameters: Dict[str, np.ndarray],
    activations: List[Callable],
    verbose: bool = True
) -> Tuple[np.ndarray, List[Dict[str, np.ndarray]]]:
    """
    执行完整的前向传播，并存储所有中间值（cache）。

    数学过程：
      对于第 l 层:
        z^{[l]} = W^{[l]} @ a^{[l-1]} + b^{[l]}
        a^{[l]} = φ^{[l]}(z^{[l]})

    参数:
        X: 输入数据，shape (n_features, m_samples)
        parameters: 参数字典，包含 W1,b1, W2,b2, ...
        activations: 每层的激活函数列表，长度 = L-1
        verbose: 是否打印每层的张量形状

    返回:
        aL: 最后一层的输出（模型的预测值）
        caches: 列表，每个元素是一个字典，存储了该层的 z, a, a_prev
    """
    a = X  # a^{[0]} = X，当前激活值初始化为输入
    caches = []  # 缓存列表，存储每层的中间值以供反向传播使用
    L = len(parameters) // 2  # 网络的层数（W 和 b 成对出现）

    if verbose:
        print("\n" + "=" * 70)
        print("【前向传播开始】输入 shape: {}".format(X.shape))
        print("=" * 70)

    for l in range(1, L + 1):  # 逐层前向传播
        # ---- 步骤 1: 线性变换 z^{[l]} = W^{[l]} @ a^{[l-1]} + b^{[l]} ----
        W = parameters[f"W{l}"]  # 获取第 l 层的权重矩阵
        b = parameters[f"b{l}"]  # 获取第 l 层的偏置向量
        z = W @ a + b            # 线性变换：矩阵乘法 + 广播加法

        # ---- 步骤 2: 非线性激活 a^{[l]} = φ^{[l]}(z^{[l]}) ----
        activation_fn = activations[l - 1]  # 获取第 l 层的激活函数
        a_new = activation_fn(z)            # 应用激活函数

        # ---- 步骤 3: 存储中间值（cache） ----
        cache = {
            "z": z,           # 预激活值 z^{[l]} —— 反向传播中算 φ' 时需要
            "a_prev": a,      # 上一层的激活 a^{[l-1]} —— 反向传播中算 dW 时需要
            "a": a_new,       # 当前层的激活 a^{[l]} —— 作为下一层的输入
            "W_shape": W.shape,  # 权重矩阵形状（便于调试）
        }
        caches.append(cache)

        # ---- 步骤 4: 打印该层的张量形状 ----
        if verbose:
            act_name = activation_fn.__name__  # 获取激活函数名称
            print(f"  第 {l} 层:")
            print(f"    a^{{{l-1}}}.shape = {cache['a_prev'].shape}  ← 输入")
            print(f"    W^{{{l}}}.shape   = {W.shape}        ← 权重矩阵")
            print(f"    b^{{{l}}}.shape   = {b.shape}        ← 偏置向量")
            print(f"    z^{{{l}}}.shape   = {z.shape}        ← 线性输出 (W·a + b)")
            print(f"    a^{{{l}}}.shape   = {a_new.shape}        ← 激活输出 ({act_name})")
            # 打印该层激活值的统计信息
            print(f"    a^{{{l}}} 统计: min={a_new.min():.4f}, max={a_new.max():.4f}, "
                  f"mean={a_new.mean():.4f}, std={a_new.std():.4f}")

        a = a_new  # 更新当前激活值，作为下一层的输入

    if verbose:
        print("=" * 70)
        print(f"【前向传播完成】最终输出 shape: {a.shape}")
        print(f"  输出值范围: [{a.min():.4f}, {a.max():.4f}]")
        print(f"  共缓存 {len(caches)} 层的中间值（供反向传播使用）")
        print("=" * 70)

    return a, caches


# ============================================================================
# 第四部分：可视化
# ============================================================================

def plot_network_structure(parameters: Dict[str, np.ndarray], X_sample: np.ndarray):
    """
    绘制网络结构图，显示每层的神经元数量和连接关系。
    左侧显示网络架构，右侧标注对应的数据维度。

    参数:
        parameters: 参数字典
        X_sample: 单个样本输入 (n_features, 1)，用于确定输入维度
    """
    L = len(parameters) // 2  # 层数
    layer_sizes = [X_sample.shape[0]]  # 输入层神经元数
    for l in range(1, L + 1):
        layer_sizes.append(parameters[f"W{l}"].shape[0])  # 第 l 层的输出神经元数

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_xlim(-0.5, L + 0.5)  # x 轴范围：层索引
    max_neurons = max(layer_sizes)  # 最大神经元数量，用于确定 y 轴范围
    ax.set_ylim(-max_neurons - 0.5, max_neurons + 0.5)

    # 存储每层神经元的位置
    neuron_positions = []

    # ---- 绘制神经元和连接 ----
    for l_idx, n_neurons in enumerate(layer_sizes):  # 遍历每一层
        # 计算该层神经元在 y 轴上的均匀分布位置
        y_positions = np.linspace(max_neurons / 2 - n_neurons / 2,
                                   -max_neurons / 2 + n_neurons / 2,
                                   max(n_neurons, 1))
        positions = []
        for n_idx, y in enumerate(y_positions):  # 遍历该层每个神经元
            # 确定颜色：输入层=蓝，隐藏层=橙，输出层=红
            if l_idx == 0:
                color = '#4A90D9'  # 蓝色：输入层
                label = f'x{n_idx+1}'  # 标签：x1, x2, ...
            elif l_idx == L:
                color = '#E74C3C'  # 红色：输出层
                label = f'ŷ{n_idx+1}'  # 标签：ŷ1, ŷ2, ...
            else:
                color = '#F39C12'  # 橙色：隐藏层
                label = f'h{l_idx},{n_idx+1}'

            # 绘制神经元（圆点）
            circle = plt.Circle((l_idx, y), 0.25, color=color, ec='white', linewidth=1.5, zorder=5)
            ax.add_patch(circle)
            # 标注神经元名称
            ax.text(l_idx, y, label, ha='center', va='center', fontsize=7,
                    color='white', fontweight='bold', zorder=6)
            positions.append(y)

        neuron_positions.append((l_idx, positions))

        # ---- 绘制层之间的连接线 ----
        if l_idx > 0:  # 非第一层需要绘制入边
            prev_positions = neuron_positions[l_idx - 1][1]  # 上一层神经元位置
            for prev_y in prev_positions:  # 遍历前一层每个神经元
                for curr_y in positions:  # 遍历当前层每个神经元
                    ax.plot([l_idx - 1, l_idx], [prev_y, curr_y],
                            color='gray', alpha=0.2, linewidth=0.5, zorder=1)

        # ---- 标注层名 ----
        if l_idx == 0:
            layer_name = f'Input Layer\n({n_neurons} neurons)'
        elif l_idx == L:
            layer_name = f'Output Layer\n({n_neurons} neurons)'
        else:
            layer_name = f'Hidden Layer {l_idx}\n({n_neurons} neurons)'
        ax.text(l_idx, max_neurons / 2 + 0.8, layer_name,
                ha='center', fontsize=9, fontweight='bold')

    # ---- 绘制权重矩阵标注 ----
    for l in range(1, L + 1):
        W = parameters[f"W{l}"]
        x_pos = l - 0.5  # 标注在两层之间的位置
        ax.annotate(f'W[{l}]\n{W.shape[0]}×{W.shape[1]}',
                    xy=(x_pos, -max_neurons / 2 - 0.3),
                    fontsize=7, ha='center', color='#2C3E50',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F8F5', alpha=0.8))

    ax.set_title('Neural Network Structure - Computation Graph View', fontsize=14, fontweight='bold')
    ax.axis('equal')
    ax.axis('off')

    # 图例
    legend_elements = [
        mpatches.Patch(color='#4A90D9', label='Input Layer'),
        mpatches.Patch(color='#F39C12', label='Hidden Layer'),
        mpatches.Patch(color='#E74C3C', label='Output Layer'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(_IMAGES, 'network_structure.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("\n[可视化] 网络结构图已保存至 " + os.path.join(_IMAGES, 'network_structure.png'))


def plot_activation_functions():
    """
    绘制四种常见激活函数及其导数的对比图。
    包含：ReLU, Sigmoid, Tanh, Leaky ReLU
    """
    z = np.linspace(-5, 5, 1000)  # 在 [-5, 5] 区间生成 1000 个点

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()  # 展平为 1D 数组，方便索引

    # ---- 定义要绘制的激活函数 ----
    funcs = [
        ("ReLU", relu, relu_derivative, "max(0, z)", "#2E86AB"),
        ("Sigmoid", sigmoid, sigmoid_derivative, "1/(1+e^{-z})", "#A23B72"),
        ("Tanh", tanh, tanh_derivative, "tanh(z)", "#F18F01"),
        ("Leaky ReLU (α=0.01)", lambda z: np.maximum(0, z) + 0.01 * np.minimum(0, z),
         lambda z: np.where(z > 0, 1.0, 0.01), "max(0,z)+0.01*min(0,z)", "#C73E1D"),
    ]

    for ax, (name, fn, fn_prime, formula, color) in zip(axes, funcs):
        y = fn(z)        # 计算函数值
        dy = fn_prime(z) # 计算导数值

        # 绘制函数曲线（蓝色实线）
        ax.plot(z, y, 'b-', linewidth=2.5, label=f'{name}: f(z)')
        # 绘制导数曲线（红色虚线）
        ax.plot(z, dy, 'r--', linewidth=2, label=f"{name}: f'(z)")

        # 标记饱和区（导数接近 0 的区域）
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)  # y=0 参考线
        ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)  # y=1 参考线

        # 设置坐标轴和标题
        ax.set_xlim(-5, 5)
        ax.set_title(f'{name}\n{formula}', fontsize=12, fontweight='bold')
        ax.set_xlabel('z', fontsize=10)
        ax.set_ylabel('f(z) / f\'(z)', fontsize=10)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Common Activation Functions and Their Derivatives', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(_IMAGES, 'activation_functions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[可视化] 激活函数对比图已保存至 " + os.path.join(_IMAGES, 'activation_functions.png'))


def print_tensor_shape_table(caches: List[Dict], parameters: Dict[str, np.ndarray]):
    """
    打印前向传播中所有张量的形状表格。

    参数:
        caches: 前向传播的缓存列表
        parameters: 参数字典
    """
    print("\n" + "=" * 70)
    print("【张量形状总览表】")
    print("=" * 70)
    print(f"{'步骤':<10} {'名称':<12} {'形状':<22} {'说明'}")
    print("-" * 70)

    # 输入数据
    print(f"{'输入':<10} {'X (a[0])':<12} {str(caches[0]['a_prev'].shape):<22} {'输入数据（特征数 × 样本数）'}")
    L = len(parameters) // 2  # 层数

    for l in range(1, L + 1):  # 遍历每一层
        cache = caches[l - 1]  # 获取第 l 层的缓存
        # 权重矩阵
        print(f"{'权重':<10} {f'W[{l}]':<12} {str(parameters[f'W{l}'].shape):<22} "
              f"{'第 ' + str(l) + ' 层权重矩阵'}")
        # 偏置向量
        print(f"{'偏置':<10} {f'b[{l}]':<12} {str(parameters[f'b{l}'].shape):<22} "
              f"{'第 ' + str(l) + ' 层偏置向量'}")
        # 线性输出
        print(f"{'第 ' + str(l) + ' 层':<10} {f'z[{l}]':<12} {str(cache['z'].shape):<22} "
              f"{'线性变换输出（W·a_prev + b）'}")
        # 激活输出
        print(f"{'第 ' + str(l) + ' 层':<10} {f'a[{l}]':<12} {str(cache['a'].shape):<22} "
              f"{'激活函数输出（下一层输入）'}")
        print(f"{'':<10} {'':<12} {'':<22}  min={cache['a'].min():.4f}, max={cache['a'].max():.4f}")

    print("-" * 70)
    total_params = sum(p.size for p in parameters.values())  # 计算总参数数量
    print(f"  总参数量: {total_params} 个")
    print(f"  缓存张量数: {len(caches) * 3} 个 (每层: z, a_prev, a)")
    print("=" * 70)


def plot_forward_data_flow(caches: List[Dict]):
    """
    可视化前向传播中激活值的流动变化。
    绘制每层激活值的分布直方图，观察数据在网络中的演变。

    参数:
        caches: 前向传播的缓存列表
    """
    L = len(caches)  # 层数
    fig, axes = plt.subplots(1, L + 1, figsize=(4 * (L + 1), 4))

    # ---- 绘制输入分布 ----
    a_prev_vals = caches[0]['a_prev'].flatten()  # 输入数据展开为一维
    axes[0].hist(a_prev_vals, bins=30, color='#4A90D9', alpha=0.7, edgecolor='white')
    axes[0].set_title(f'Input Layer a[0]\nshape={caches[0]["a_prev"].shape}', fontsize=10)
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.5)  # 零参考线

    # ---- 绘制每层激活分布 ----
    for l in range(L):
        a_vals = caches[l]['a'].flatten()  # 第 l 层激活值展开
        axes[l + 1].hist(a_vals, bins=30, color='#F39C12', alpha=0.7, edgecolor='white')
        axes[l + 1].set_title(f'Layer {l+1} a[{l+1}]\nshape={caches[l]["a"].shape}', fontsize=10)
        axes[l + 1].set_xlabel('Value')
        axes[l + 1].set_ylabel('Frequency')
        axes[l + 1].axvline(x=0, color='red', linestyle='--', alpha=0.5)  # 零参考线

    plt.suptitle('Layer-wise Evolution of Activation Distribution During Forward Propagation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(_IMAGES, 'forward_data_flow.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[可视化] 前向传播数据流图已保存至 " + os.path.join(_IMAGES, 'forward_data_flow.png'))


# ============================================================================
# 第五部分：主程序
# ============================================================================

def main():
    """
    主程序：演示完整的前向传播流程。

    1. 生成合成数据
    2. 初始化一个 3 层 MLP
    3. 执行前向传播，存储所有中间值
    4. 打印张量形状表格
    5. 可视化网络结构和激活函数
    """
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║        s05 计算图与前向传播 — NumPy 从头实现 MLP 前向传播       ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # ---- 1. 生成合成数据集 ----
    np.random.seed(0)  # 固定随机种子，保证可复现
    n_samples = 32       # 样本数量（mini-batch size）
    n_features = 3       # 输入特征数
    X = np.random.randn(n_features, n_samples)  # 生成随机输入数据 (3, 32)
    print(f"\n[数据] 生成了 {n_samples} 个样本，每个 {n_features} 个特征")
    print(f"  输入 X shape: {X.shape}")
    print(f"  X 范围: [{X.min():.4f}, {X.max():.4f}], 均值: {X.mean():.4f}")

    # ---- 2. 定义网络结构 ----
    # 3 层 MLP: [输入3] → [隐藏层4] → [隐藏层4] → [输出1]
    layer_dims = [3, 4, 4, 1]
    print(f"\n[网络结构] 各层神经元数量: {layer_dims}")
    print(f"  输入层: {layer_dims[0]} 个神经元")
    for l in range(1, len(layer_dims) - 1):
        print(f"  隐藏层 {l}: {layer_dims[l]} 个神经元")
    print(f"  输出层: {layer_dims[-1]} 个神经元")

    # ---- 3. 初始化参数 ----
    print(f"\n[初始化] 使用 He 初始化方法...")
    parameters = initialize_parameters(layer_dims)

    # ---- 4. 选择激活函数 ----
    # 隐藏层使用 ReLU（现代神经网络的默认选择），输出层使用 sigmoid（二分类场景）
    activations = [relu, relu, sigmoid]
    print(f"\n[激活函数] 隐藏层: ReLU × 2, 输出层: Sigmoid")

    # ---- 5. 执行前向传播 ----
    y_pred, caches = forward_pass(X, parameters, activations, verbose=True)

    # ---- 6. 打印张量形状表格 ----
    print_tensor_shape_table(caches, parameters)

    # ---- 7. 可视化 ----
    print("\n[可视化] 生成图形...")

    # 绘制网络结构图
    X_single = X[:, 0:1]  # 取第一个样本 (3, 1)
    plot_network_structure(parameters, X_single)

    # 绘制激活函数对比图
    plot_activation_functions()

    # 绘制前向传播数据流
    plot_forward_data_flow(caches)

    # ---- 8. 最终总结 ----
    print("\n" + "=" * 70)
    print("【总结】")
    print("=" * 70)
    print(f"  ✓ 完成了 {len(caches)} 层 MLP 的前向传播")
    print(f"  ✓ 输入: {X.shape} → 输出: {y_pred.shape}")
    print(f"  ✓ 共存储了 {len(caches)} 个 cache（每个含 z, a_prev, a）")
    print(f"  ✓ 总参数量: {sum(p.size for p in parameters.values())}")
    print(f"\n  这些中间值将在反向传播中被使用——")
    print(f"  - z[l] 用于计算激活函数的导数 φ'(z[l])")
    print(f"  - a[l-1] 用于计算权重梯度 dW[l] = δ[l] · (a[l-1])^T")
    print(f"  - a[l] 作为下一层的输入继续前向传播")
    print(f"\n  下一节 s06 将讲解如何利用这些缓存进行反向传播。")
    print("=" * 70)


if __name__ == "__main__":
    main()
