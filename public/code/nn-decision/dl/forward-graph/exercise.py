# -*- coding: utf-8 -*-
"""
s05 计算图与前向传播 — 练习代码
================================
请完成以下 TODO 任务，巩固对前向传播和计算图的理解。

每个 TODO 都有详细的指示和预期输出描述。
建议先阅读 README.md，再尝试独立补全代码。
"""

import numpy as np


# ============================================================================
# TODO 1: 实现单层全连接层的前向传播
# ============================================================================
def dense_layer_forward(A_prev: np.ndarray, W: np.ndarray, b: np.ndarray,
                        activation: str = "relu") -> tuple:
    """
    实现单个全连接层的前向传播。

    数学公式：
        Z = W @ A_prev + b
        A = activation(Z)

    参数:
        A_prev: 上一层的激活值，shape (n_prev, m)，其中 n_prev 是输入维度，m 是样本数
        W: 权重矩阵，shape (n_curr, n_prev)
        b: 偏置向量，shape (n_curr, 1)
        activation: 激活函数名称，可选 "relu", "sigmoid", "tanh", "none"

    返回:
        Z: 线性输出，shape (n_curr, m)
        A: 激活输出，shape (n_curr, m)
        cache: 字典，包含 Z, A_prev, W, b，供反向传播使用
    """
    # TODO: 补全以下代码

    # 步骤 1: 计算线性变换 Z = W @ A_prev + b
    # 提示: 使用 np.dot 或 @ 运算符进行矩阵乘法
    Z = None  # ← TODO: 实现 Z = W @ A_prev + b

    # 步骤 2: 应用激活函数
    # 提示: 根据 activation 参数选择对应的激活函数
    if activation == "relu":
        A = None  # ← TODO: 对 Z 应用 ReLU (np.maximum)
    elif activation == "sigmoid":
        A = None  # ← TODO: 对 Z 应用 sigmoid (1/(1+exp(-Z)))
    elif activation == "tanh":
        A = None  # ← TODO: 对 Z 应用 tanh (np.tanh)
    elif activation == "none":
        A = None  # ← TODO: 线性激活，A = Z
    else:
        raise ValueError(f"不支持的激活函数: {activation}")

    # 步骤 3: 创建 cache 字典，存储反向传播所需的所有中间值
    cache = None  # ← TODO: 创建包含 Z, A_prev, W, b 的字典

    return Z, A, cache


# ---- 测试 TODO 1 ----
def test_dense_layer():
    """测试单层全连接前向传播的实现。"""
    print("=" * 60)
    print("TODO 1 测试: 单层全连接层的前向传播")
    print("=" * 60)

    # 测试数据：3 个输入特征，2 个神经元，5 个样本
    np.random.seed(42)
    A_prev = np.random.randn(3, 5)   # (3, 5)
    W = np.random.randn(2, 3) * 0.1  # (2, 3)
    b = np.zeros((2, 1))             # (2, 1)

    for act in ["relu", "sigmoid", "tanh", "none"]:
        Z, A, cache = dense_layer_forward(A_prev, W, b, activation=act)
        if Z is None:
            print(f"  [{act}] TODO 未完成，请补全 dense_layer_forward 函数")
        else:
            print(f"  [{act}] Z.shape={Z.shape}, A.shape={A.shape}, "
                  f"A range=[{A.min():.4f}, {A.max():.4f}]")

    print()


# ============================================================================
# TODO 2: 实现 GELU 激活函数
# ============================================================================
def gelu_exact(z: np.ndarray) -> np.ndarray:
    """
    实现 GELU (Gaussian Error Linear Unit) 激活函数的精确版本。

    GELU 的精确定义：
        GELU(z) = z · Φ(z)

    其中 Φ(z) 是标准正态分布的累积分布函数 (CDF)：
        Φ(z) = 0.5 * (1 + erf(z / √2))

    参数:
        z: 输入数组，任意形状

    返回:
        逐元素应用的 GELU 结果

    参考资料:
        Hendrycks & Gimpel (2016): "Gaussian Error Linear Units (GELUs)"
        https://arxiv.org/abs/1606.08415
    """
    # TODO: 实现 GELU 的精确版本
    # 提示 1: 使用 scipy.special.erf 或手动实现 erf 近似
    # 提示 2: Φ(z) = 0.5 * (1 + erf(z / sqrt(2)))
    # 提示 3: GELU(z) = z * Φ(z)
    # 如果不想引入 scipy，可以使用下面的近似公式:
    #   GELU(z) ≈ 0.5 * z * (1 + tanh(sqrt(2/π) * (z + 0.044715 * z^3)))

    result = None  # ← TODO: 实现 GELU 函数

    return result


def gelu_derivative(z: np.ndarray) -> np.ndarray:
    """
    实现 GELU 激活函数的导数。

    GELU 的导数（精确形式）：
        GELU'(z) = Φ(z) + z · φ(z)

    其中 φ(z) 是标准正态分布的概率密度函数 (PDF)：
        φ(z) = exp(-z²/2) / √(2π)

    参数:
        z: 输入数组

    返回:
        GELU 在 z 处的导数
    """
    # TODO: 实现 GELU 的导数
    # 提示: 需要同时用到 Φ(z) 和 φ(z)

    result = None  # ← TODO: 实现 GELU 导数

    return result


# ---- 测试 TODO 2 ----
def test_gelu():
    """测试 GELU 激活函数的实现。"""
    print("=" * 60)
    print("TODO 2 测试: GELU 激活函数")
    print("=" * 60)

    z = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    result = gelu_exact(z)

    if result is None:
        print("  TODO 未完成，请补全 gelu_exact 函数")
    else:
        print(f"  输入 z:        {z}")
        print(f"  GELU(z):       {np.round(result, 4)}")
        # 预期输出（近似值）:
        # GELU(-2) ≈ -0.0454, GELU(-1) ≈ -0.1588, GELU(0) = 0,
        # GELU(0.5) ≈ 0.3457, GELU(1) ≈ 0.8413, GELU(2) ≈ 1.9546

        # 验证几个关键性质
        print(f"\n  性质验证:")
        print(f"    GELU(0) = {result[3]:.6f} (预期: 0.0)")
        print(f"    GELU(z) ≈ z for z >> 0: GELU(2)={result[6]:.4f} vs z=2.0")

    print()


# ============================================================================
# TODO 3: 追踪计算图
# ============================================================================
def trace_computational_graph(X: np.ndarray) -> dict:
    """
    对于给定的表达式，追踪并打印计算图的所有中间节点。

    表达式: f(x1, x2, x3) = sigmoid( (x1 * w1 + x2 * w2 + b) * w3 + x3 )

    任务：把这个表达式分解为计算图中的基本操作节点，
          并记录每个节点的输入、输出和操作类型。

    参数:
        X: 包含 [x1, x2, x3] 的数组，shape (3,)

    返回:
        graph_nodes: 字典，key 为节点名，value 为包含 value、inputs、op 的字典
    """
    # 给定的参数
    w1, w2, w3, b = 0.5, -0.3, 2.0, 0.1  # 权重和偏置
    x1, x2, x3 = X[0], X[1], X[2]         # 输入

    # TODO: 补全以下计算图的追踪
    # 提示: 将表达式分解为以下步骤，每一步录为一个节点:
    #   node1: u1 = x1 * w1       (乘法)
    #   node2: u2 = x2 * w2       (乘法)
    #   node3: u3 = u1 + u2       (加法)
    #   node4: u4 = u3 + b        (加法)
    #   node5: u5 = u4 * w3       (乘法)
    #   node6: u6 = u5 + x3       (加法)
    #   node7: u7 = sigmoid(u6)   (sigmoid)

    graph_nodes = {}  # 初始化计算图节点字典

    # TODO: 实现每一步计算，并记录到 graph_nodes 中
    # 示例格式:
    # graph_nodes["u1"] = {"value": 计算结果, "inputs": ["x1", "w1"], "op": "multiply"}

    # 打印计算图
    print("\n计算图结构:")
    print("-" * 40)
    for name, info in graph_nodes.items():
        inputs_str = ", ".join(info.get("inputs", []))
        print(f"  {name} = {info.get('op', '?')}({inputs_str}) = {info.get('value', '?')}")
    print("-" * 40)

    return graph_nodes


# ---- 测试 TODO 3 ----
def test_computational_graph():
    """测试计算图追踪功能。"""
    print("=" * 60)
    print("TODO 3 测试: 计算图追踪")
    print("=" * 60)

    X = np.array([1.0, 2.0, 0.5])  # x1=1.0, x2=2.0, x3=0.5
    graph = trace_computational_graph(X)

    if not graph:
        print("  TODO 未完成，请补全 trace_computational_graph 函数")
    else:
        print(f"\n  共追踪了 {len(graph)} 个计算节点")
        # 检查最终输出
        if "u7" in graph:
            print(f"  最终输出 (sigmoid): {graph['u7']['value']:.4f}")

    print()


# ============================================================================
# 主程序
# ============================================================================
if __name__ == "__main__":
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║   s05 计算图与前向传播 — 动手练习                           ║")
    print("║   请依次完成 TODO 1, 2, 3                                    ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    test_dense_layer()
    test_gelu()
    test_computational_graph()

    print("=" * 60)
    print("所有测试完成！请检查输出结果。")
    print("如有未通过的测试，请回到对应的 TODO 部分补全代码。")
    print("=" * 60)
