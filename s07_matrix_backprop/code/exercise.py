# -*- coding: utf-8 -*-
"""
s07 多层网络的矩阵反传 — 练习代码
================================
请完成以下 TODO 任务，加深对矩阵形式反向传播的理解。

每个 TODO 都有详细的中文指示和预期输出描述。
建议先阅读 README.md 并运行 demo.py，再尝试独立补全代码。
"""

import numpy as np
from typing import Dict, List, Tuple


# ============================================================================
# 辅助函数：激活函数（与 demo.py 一致）
# ============================================================================

def relu(Z: np.ndarray) -> np.ndarray:
    return np.maximum(0, Z)


def relu_derivative(Z: np.ndarray) -> np.ndarray:
    return (Z > 0).astype(np.float64)


def sigmoid(Z: np.ndarray) -> np.ndarray:
    Z = np.clip(Z, -500, 500)
    return 1.0 / (1.0 + np.exp(-Z))


def sigmoid_derivative(Z: np.ndarray) -> np.ndarray:
    s = sigmoid(Z)
    return s * (1.0 - s)


# ============================================================================
# TODO 1: 实现单隐藏层的反向传播（δ 递推计算）
# ============================================================================

def single_hidden_backward(
    W1: np.ndarray, b1: np.ndarray,
    W2: np.ndarray, b2: np.ndarray,
    X: np.ndarray, Y: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    实现一个单隐藏层（2层）网络的反向传播。

    网络结构: 输入 → 隐藏层(ReLU) → 输出层(Sigmoid)
    损失函数: MSE

    你要实现的关键步骤:
      1. 前向传播并缓存中间值
      2. 计算输出层误差 δ2
      3. 用递推公式计算隐藏层误差 δ1
      4. 计算 dW1, db1, dW2, db2

    参数:
        W1: 第一层权重 (n_hidden, n_input)
        b1: 第一层偏置 (n_hidden, 1)
        W2: 第二层权重 (n_output, n_hidden)
        b2: 第二层偏置 (n_output, 1)
        X: 输入数据 (n_input, m)
        Y: 标签 (n_output, m)

    返回:
        grads: 包含 dW1, db1, dW2, db2 的字典
    """
    m = X.shape[1]  # 样本数

    # ---- 前向传播 ----
    # TODO: 实现第一层的前向传播
    Z1 = None  # ← TODO: W1 @ X + b1
    A1 = None  # ← TODO: relu(Z1)

    # TODO: 实现第二层的前向传播
    Z2 = None  # ← TODO: W2 @ A1 + b2
    A2 = None  # ← TODO: sigmoid(Z2) —— 这是最终预测值

    # ---- 反向传播 ----
    # TODO: 计算输出层误差 δ2（MSE 损失 + sigmoid 激活）
    # 提示: MSE 损失的梯度 ∂L/∂A2 = (1/m) * (A2 - Y)
    #       δ2 = ∂L/∂A2 ⊙ sigmoid'(Z2)
    dA2 = None  # ← TODO: (1.0 / m) * (A2 - Y)
    dZ2 = None  # ← TODO: dA2 * sigmoid_derivative(Z2)

    # TODO: 计算 dW2 和 db2
    dW2 = None  # ← TODO: dZ2 @ A1.T
    db2 = None  # ← TODO: np.sum(dZ2, axis=1, keepdims=True)

    # TODO: 计算隐藏层误差 δ1（递推公式的核心！）
    # 提示: δ1 = W2^T @ δ2 ⊙ relu'(Z1)
    dZ1 = None  # ← TODO: (W2.T @ dZ2) * relu_derivative(Z1)

    # TODO: 计算 dW1 和 db1
    dW1 = None  # ← TODO: dZ1 @ X.T
    db1 = None  # ← TODO: np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return grads


# ---- 测试 TODO 1 ----
def test_single_hidden_backward():
    """测试单隐藏层的反向传播"""
    print("=" * 60)
    print("TODO 1 测试: 单隐藏层反向传播（δ 递推）")
    print("=" * 60)

    np.random.seed(42)
    # 小网络: 2输入 → 3隐藏(ReLU) → 1输出(Sigmoid)
    W1 = np.random.randn(3, 2) * 0.5  # (3, 2)
    b1 = np.zeros((3, 1))              # (3, 1)
    W2 = np.random.randn(1, 3) * 0.5  # (1, 3)
    b2 = np.zeros((1, 1))              # (1, 1)

    # 构造简单的输入和标签
    X = np.array([[0.5, 1.0], [0.3, 0.8]]).T  # (2, 2) — 2个特征，2个样本
    Y = np.array([[0.0, 1.0]])                  # (1, 2) — 2个标签

    grads = single_hidden_backward(W1, b1, W2, b2, X, Y)

    if grads["dW1"] is None:
        print("  TODO 未完成，请补全 single_hidden_backward 函数")
        return

    print(f"\n  梯度计算结果:")
    print(f"    dW1 shape: {grads['dW1'].shape} (预期: (3, 2))")
    print(f"    db1 shape: {grads['db1'].shape} (预期: (3, 1))")
    print(f"    dW2 shape: {grads['dW2'].shape} (预期: (1, 3))")
    print(f"    db2 shape: {grads['db2'].shape} (预期: (1, 1))")

    # 验证形状
    all_correct = True
    expected_shapes = {"dW1": (3, 2), "db1": (3, 1), "dW2": (1, 3), "db2": (1, 1)}
    for key, expected in expected_shapes.items():
        actual = grads[key].shape
        match = actual == expected
        if not match:
            print(f"    ✗ {key} shape 不匹配: got {actual}, expected {expected}")
            all_correct = False

    if all_correct:
        print(f"\n  ✓ 所有梯度形状正确！")

    print()


# ============================================================================
# TODO 2: 实现梯度裁剪
# ============================================================================

def clip_gradients(grads: Dict[str, np.ndarray], max_norm: float) -> Dict[str, np.ndarray]:
    """
    实现梯度裁剪：当梯度的全局 L2 范数超过阈值时，按比例缩放所有梯度。

    全局 L2 范数:
        total_norm = sqrt( Σ_i ||grad_i||^2 )

    缩放因子:
        scale = min(1.0, max_norm / total_norm)

    裁剪后:
        grad_i_clipped = scale * grad_i

    这个技术在 RNN 和 Transformer 训练中广泛使用，可以有效防止梯度爆炸。

    参数:
        grads: 梯度字典 {参数名: 梯度矩阵}
        max_norm: 梯度范数的最大允许值

    返回:
        grads_clipped: 裁剪后的梯度字典

    参考资料:
        Pascanu et al. (2013): "On the difficulty of training recurrent neural networks"
    """
    # TODO: 计算全局梯度范数
    # 提示: 遍历 grads 中的所有梯度，计算每个梯度的 L2 范数的平方和，然后开根号
    total_norm_sq = 0.0
    for grad in grads.values():
        pass  # ← TODO: total_norm_sq += np.sum(grad ** 2)

    # TODO: 计算总范数
    total_norm = None  # ← TODO: np.sqrt(total_norm_sq)

    # TODO: 计算缩放因子
    # 如果 total_norm > max_norm，则缩放；否则不变（scale=1.0）
    scale = None  # ← TODO: min(1.0, max_norm / total_norm) if total_norm > 0 else 1.0

    # TODO: 按比例缩放所有梯度
    grads_clipped = {}
    for key, grad in grads.items():
        pass  # ← TODO: grads_clipped[key] = scale * grad

    return grads_clipped


# ---- 测试 TODO 2 ----
def test_gradient_clipping():
    """测试梯度裁剪"""
    print("=" * 60)
    print("TODO 2 测试: 梯度裁剪")
    print("=" * 60)

    # 构造一些"爆炸"的梯度
    grads = {
        "dW1": np.ones((3, 2)) * 10.0,   # 每一层的梯度都很大
        "db1": np.ones((3, 1)) * 5.0,
        "dW2": np.ones((1, 3)) * 8.0,
        "db2": np.ones((1, 1)) * 3.0,
    }

    # 计算原始总范数
    total_norm_sq = sum(np.sum(g ** 2) for g in grads.values())
    total_norm = np.sqrt(total_norm_sq)
    print(f"\n  原始梯度总范数: {total_norm:.2f}")

    # 应用梯度裁剪，设 max_norm = 5.0
    max_norm = 5.0
    clipped = clip_gradients(grads, max_norm)

    if not clipped:
        print("  TODO 未完成，请补全 clip_gradients 函数")
        return

    # 计算裁剪后的总范数
    clipped_norm_sq = sum(np.sum(g ** 2) for g in clipped.values())
    clipped_norm = np.sqrt(clipped_norm_sq)
    print(f"  裁剪后梯度总范数: {clipped_norm:.2f}")
    print(f"  裁剪倍数: {total_norm / clipped_norm:.2f}x")
    print(f"  裁剪后范数 ≤ max_norm: {clipped_norm <= max_norm + 1e-6}")

    # 测试小梯度的情况（不应该被裁剪）
    small_grads = {
        "dW1": np.ones((3, 2)) * 0.1,
        "db1": np.ones((3, 1)) * 0.1,
    }
    small_clipped = clip_gradients(small_grads, max_norm=5.0)
    small_norm = np.sqrt(sum(np.sum(g ** 2) for g in small_clipped.values()))
    print(f"\n  小梯度测试 (原范数 < max_norm):")
    print(f"    原始相同: {np.allclose(list(small_grads.values())[0], list(small_clipped.values())[0])}")

    print()


# ============================================================================
# TODO 3: 实现数值梯度检查
# ============================================================================

def numerical_gradient_check(
    forward_fn,
    params: Dict[str, np.ndarray],
    grads: Dict[str, np.ndarray],
    X: np.ndarray, Y: np.ndarray,
    epsilon: float = 1e-7
) -> bool:
    """
    用双边有限差分验证解析梯度。

    对每个参数 θ，数值梯度:
        ∂L/∂θ ≈ (L(θ+ε) - L(θ-ε)) / (2ε)

    实现步骤:
      1. 对 params 中的每个参数，遍历它的每个元素
      2. 计算 L(θ+ε) 和 L(θ-ε)（需要调用 forward_fn 做前向传播）
      3. 用双边差分公式估计梯度
      4. 比较解析梯度和数值梯度的相对误差
      5. 如果任何参数的相对误差 > 1e-5，返回 False

    参数:
        forward_fn: 前向传播函数，签名为 forward_fn(params, X, Y) -> loss
        params: 参数字典
        grads: 解析梯度字典（与 params 结构一致）
        X: 输入数据
        Y: 标签
        epsilon: 微小扰动值

    返回:
        passed: 是否通过梯度检查（所有参数相对误差 < 1e-5）
    """
    # TODO: 对参数中的每一个元素逐一检查
    # 提示:
    #   for param_name in params:
    #       遍历 params[param_name] 的每个元素:
    #           保存原始值
    #           θ + ε → 前向 → loss_plus
    #           θ - ε → 前向 → loss_minus
    #           grad_numeric = (loss_plus - loss_minus) / (2ε)
    #           恢复原始值
    #           计算相对误差
    #           如果误差过大，打印警告

    passed = True  # 假设通过，如果发现错误则设为 False

    # TODO: 实现数值梯度检查
    for param_name in params:
        param = params[param_name]  # 参数矩阵
        grad_analytic = grads[param_name]  # 对应的解析梯度

        # 暂时只检查少量元素（速度优化）
        flat_param = param.flatten()
        flat_grad = grad_analytic.flatten()
        n_check = min(10, len(flat_param))  # 最多只检查 10 个元素

        for i in range(n_check):
            # TODO: 对每个检查元素计算数值梯度
            # 1. 保存原始值
            # 2. 计算 loss(theta + epsilon)
            # 3. 计算 loss(theta - epsilon)
            # 4. 计算数值梯度
            # 5. 比较相对误差
            pass  # ← TODO: 实现

    return passed


# ---- 测试 TODO 3 ----
def test_numerical_gradient_check():
    """测试数值梯度检查函数"""
    print("=" * 60)
    print("TODO 3 测试: 数值梯度检查")
    print("=" * 60)

    # 构造一个简单的测试：f(W, b) = mean((Wx + b - y)^2)
    def simple_forward(params, X, Y):
        W, b = params["W"], params["b"]
        pred = W @ X + b
        return np.mean((pred - Y) ** 2) / 2.0

    # 参数和梯度
    params = {
        "W": np.array([[0.5, -0.3]]),  # (1, 2)
        "b": np.array([[0.1]]),         # (1, 1)
    }
    X = np.array([[1.0], [2.0]])        # (2, 1)
    Y = np.array([[3.0]])               # (1, 1)

    # 解析梯度: dL/dW = (Wx+b-y) · x^T, dL/db = (Wx+b-y)
    W, b = params["W"], params["b"]
    pred = W @ X + b                     # 预测值
    error = pred - Y                     # 误差
    grads = {
        "W": error @ X.T,                # (1, 2)
        "b": error,                      # (1, 1)
    }

    passed = numerical_gradient_check(simple_forward, params, grads, X, Y)

    if passed is None:
        print("  TODO 未完成，请补全 numerical_gradient_check 函数")
    else:
        print(f"  梯度检查结果: {'✓ 通过' if passed else '✗ 失败'}")

    print()


# ============================================================================
# 主程序
# ============================================================================
if __name__ == "__main__":
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║   s07 多层网络的矩阵反传 — 动手练习                        ║")
    print("║   请依次完成 TODO 1, 2, 3                                   ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    test_single_hidden_backward()
    test_gradient_clipping()
    test_numerical_gradient_check()

    print("=" * 60)
    print("所有测试完成！请检查输出结果。")
    print("如有未通过的测试，请回到对应的 TODO 部分补全代码。")
    print()
    print("核心公式速查:")
    print("  δ[L] = ∇_A L ⊙ φ'(Z[L])")
    print("  δ[l] = (W[l+1])^T @ δ[l+1] ⊙ φ'(Z[l])")
    print("  dW[l] = (1/m) · δ[l] @ (A[l-1])^T")
    print("  db[l] = (1/m) · Σ δ[l]")
    print("=" * 60)
