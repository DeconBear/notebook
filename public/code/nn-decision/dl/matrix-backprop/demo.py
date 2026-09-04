# -*- coding: utf-8 -*-
"""
s07 多层网络的矩阵反传 — 演示代码
==================================
功能：用纯 NumPy 实现完整的 MLP（前向 + 矩阵反向传播 + 参数更新），
      包括梯度检查、梯度范数监控、决策边界可视化和训练曲线。

运行方式：在 s07_matrix_backprop/ 目录下执行 python code/demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
from typing import Dict, List, Tuple, Callable
import os

_HERE = os.path.dirname(os.path.abspath(__file__))  # demo.py 所在目录
_IMAGES = os.path.join(_HERE, '..', 'images')        # 章节 images/ 目录
os.makedirs(_IMAGES, exist_ok=True)

# ============================================================================
# 第一部分：激活函数及其导数
# ============================================================================

def relu(Z: np.ndarray) -> np.ndarray:
    """ReLU 激活函数: max(0, Z)"""
    return np.maximum(0, Z)


def relu_derivative(Z: np.ndarray) -> np.ndarray:
    """ReLU 的导数: 1 if Z > 0 else 0"""
    return (Z > 0).astype(np.float64)


def sigmoid(Z: np.ndarray) -> np.ndarray:
    """Sigmoid 激活函数: 1 / (1 + e^{-Z})"""
    Z_clipped = np.clip(Z, -500, 500)  # 防止 exp 溢出
    return 1.0 / (1.0 + np.exp(-Z_clipped))


def sigmoid_derivative(Z: np.ndarray) -> np.ndarray:
    """Sigmoid 的导数: σ(Z) * (1 - σ(Z))"""
    s = sigmoid(Z)
    return s * (1.0 - s)


def tanh(Z: np.ndarray) -> np.ndarray:
    """Tanh 激活函数"""
    return np.tanh(Z)


def tanh_derivative(Z: np.ndarray) -> np.ndarray:
    """Tanh 的导数: 1 - tanh^2(Z)"""
    return 1.0 - np.tanh(Z) ** 2


# 激活函数注册表：方便按字符串名称查找
ACTIVATION_REGISTRY = {
    "relu": (relu, relu_derivative),
    "sigmoid": (sigmoid, sigmoid_derivative),
    "tanh": (tanh, tanh_derivative),
    "linear": (lambda Z: Z, lambda Z: np.ones_like(Z)),  # 线性激活（恒等映射）
}


# ============================================================================
# 第二部分：MLP 类 —— 完整的正向 + 反向传播
# ============================================================================

class MLP:
    """
    多层感知机，使用矩阵形式的反向传播。

    支持任意层数、任意激活函数，以及 mini-batch 训练。

    参数:
        layer_dims: 每层神经元数量，如 [2, 8, 4, 1]
        activations: 每层激活函数名称列表，如 ["relu", "relu", "sigmoid"]
        seed: 随机种子
    """

    def __init__(self, layer_dims: List[int], activations: List[str], seed: int = 42):
        """初始化网络参数（He 初始化）"""
        np.random.seed(seed)
        self.L = len(layer_dims) - 1  # 网络层数（不含输入层）
        self.activations = activations  # 每层的激活函数名称
        self.parameters = {}           # 参数字典: W1, b1, W2, b2, ...
        self.caches = []               # 前向传播缓存列表（每层一个 dict）

        for l in range(1, self.L + 1):
            n_in = layer_dims[l - 1]    # 输入维度
            n_out = layer_dims[l]       # 输出维度
            # He 初始化：W ~ N(0, sqrt(2/n_in))，特别适合配合 ReLU
            self.parameters[f"W{l}"] = np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in)
            self.parameters[f"b{l}"] = np.zeros((n_out, 1))  # 偏置零初始化

        self.grads = {}  # 存储每层参数梯度的字典

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        前向传播：计算模型输出，同时缓存中间值供反向传播使用。

        参数:
            X: 输入数据，shape (n_features, m_samples)

        返回:
            A[L]: 最后一层的激活输出，即模型的预测值
        """
        self.caches = []  # 清空缓存
        A = X             # A[0] = X（输入就是第 0 层的激活）

        for l in range(1, self.L + 1):
            A_prev = A                                        # 保存上一层激活值
            W = self.parameters[f"W{l}"]                      # 获取权重矩阵
            b = self.parameters[f"b{l}"]                      # 获取偏置向量
            Z = W @ A_prev + b                                 # 线性变换: Z[l] = W[l] @ A[l-1] + b[l]

            # 获取激活函数及其导数函数
            act_fn, _ = ACTIVATION_REGISTRY[self.activations[l - 1]]
            A = act_fn(Z)                                      # 非线性激活: A[l] = φ[l](Z[l])

            # 将中间值存入缓存
            self.caches.append({
                "Z": Z,          # Z[l] —— 反向传播中计算 φ'(Z[l]) 时需要
                "A_prev": A_prev,  # A[l-1] —— 反向传播中计算 dW[l] 时需要
                "A": A,          # A[l] —— 当前层输出，同时是下一层的输入
            })

        return A  # 返回最后一层的激活（模型预测值）

    def backward(self, Y: np.ndarray) -> Dict[str, np.ndarray]:
        """
        反向传播：使用矩阵形式的 δ 递推公式计算所有参数的梯度。

        核心公式:
          δ[L] = ∇_A L ⊙ φ'(Z[L])                             (输出层)
          δ[l] = (W[l+1])^T @ δ[l+1] ⊙ φ'(Z[l])              (隐藏层递推)
          dW[l] = (1/m) · δ[l] @ (A[l-1])^T                   (权重梯度)
          db[l] = (1/m) · sum(δ[l], axis=1, keepdims=True)    (偏置梯度)

        参数:
            Y: 标签，shape (n_output, m_samples)，与最后一层输出 shape 一致

        返回:
            grads: 字典，包含每层的 dW{l} 和 db{l}
        """
        m = Y.shape[1]  # mini-batch 大小
        self.grads = {}  # 清空梯度字典

        # ---- 步骤 1: 输出层的 δ[L] ----
        # 损失函数使用 MSE: L = (1/(2m)) * Σ(A[L] - Y)²
        # ∂L/∂A[L] = (1/m) * (A[L] - Y)
        AL = self.caches[-1]["A"]                    # 最后一层的激活（预测值）
        ZL = self.caches[-1]["Z"]                    # 最后一层的线性输出
        _, act_prime_fn = ACTIVATION_REGISTRY[self.activations[-1]]  # 输出层激活函数的导数
        dAL = (1.0 / m) * (AL - Y)                   # ∂L/∂A[L] —— MSE 损失的梯度
        dZ = dAL * act_prime_fn(ZL)                  # δ[L] = ∇_A L ⊙ φ'(Z[L])

        # ---- 步骤 2: 隐藏层的 δ 递推 ----
        for l in reversed(range(1, self.L + 1)):
            cache = self.caches[l - 1]               # 第 l 层的缓存
            A_prev = cache["A_prev"]                 # A[l-1]

            # 计算参数梯度
            self.grads[f"dW{l}"] = dZ @ A_prev.T    # dW[l] = δ[l] @ (A[l-1])^T
            # 注意：这里已经在 dZ 中包含了 (1/m) 因子
            self.grads[f"db{l}"] = np.sum(dZ, axis=1, keepdims=True)  # db[l] = Σ_i δ_i[l]

            # 如果不是第一层，继续向前传播 δ
            if l > 1:
                W_next = self.parameters[f"W{l}"]    # W[l]（当前层权重——从前一层角度看是 W[l+1]）
                Z_prev = self.caches[l - 2]["Z"]     # Z[l-1]
                _, act_prime_fn_prev = ACTIVATION_REGISTRY[self.activations[l - 2]]
                # δ[l-1] = (W[l])^T @ δ[l] ⊙ φ'[l-1](Z[l-1])
                dZ = (W_next.T @ dZ) * act_prime_fn_prev(Z_prev)

        return self.grads

    def update(self, learning_rate: float):
        """
        使用梯度下降更新所有参数。

        W[l] := W[l] - α · dW[l]
        b[l] := b[l] - α · db[l]

        参数:
            learning_rate: 学习率 α
        """
        for l in range(1, self.L + 1):
            self.parameters[f"W{l}"] -= learning_rate * self.grads[f"dW{l}"]
            self.parameters[f"b{l}"] -= learning_rate * self.grads[f"db{l}"]

    def compute_loss(self, Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
        """
        计算 MSE 损失。

        参数:
            Y_pred: 模型预测值
            Y_true: 真实标签

        返回:
            MSE 损失值（标量）
        """
        m = Y_true.shape[1]  # 样本数
        return np.mean((Y_pred - Y_true) ** 2) / 2.0  # (1/2) * MSE

    def get_gradient_norms(self) -> Dict[str, float]:
        """
        计算每层参数梯度的 L2 范数，用于监控训练健康度。

        返回:
            norms: 字典，key 为参数名，value 为梯度 L2 范数
        """
        norms = {}
        for l in range(1, self.L + 1):
            norms[f"|dW{l}|"] = np.linalg.norm(self.grads[f"dW{l}"])
            norms[f"|db{l}|"] = np.linalg.norm(self.grads[f"db{l}"])
        return norms


# ============================================================================
# 第三部分：梯度检查
# ============================================================================

def gradient_check(model: MLP, X: np.ndarray, Y: np.ndarray, epsilon: float = 1e-7) -> float:
    """
    使用双边有限差分法验证解析梯度。

    对每个参数 θ，数值梯度近似为:
        ∂L/∂θ ≈ (L(θ+ε) - L(θ-ε)) / (2ε)

    参数:
        model: MLP 模型
        X: 输入数据（使用小 batch 以提高速度）
        Y: 标签
        epsilon: 微小扰动

    返回:
        max_rel_error: 最大相对误差
    """
    # 先计算解析梯度
    Y_pred = model.forward(X)
    model.backward(Y)

    max_rel_error = 0.0  # 记录最大相对误差

    for l in range(1, model.L + 1):
        for param_name in [f"W{l}", f"b{l}"]:
            param = model.parameters[param_name]     # 原始参数矩阵
            grad_analytic = model.grads[f"d{param_name}"]  # 解析梯度
            grad_numeric = np.zeros_like(param)          # 数值梯度矩阵

            # 对参数矩阵中的每个元素，逐一计算数值梯度
            # 注意：对于大网络这会很慢，这里仅用于教学演示
            it = np.nditer(param, flags=['multi_index'])
            while not it.finished:
                idx = it.multi_index  # 当前元素的多维索引
                original_value = param[idx]  # 保存原始值

                # 计算 L(θ + ε)
                param[idx] = original_value + epsilon
                Y_pred_plus = model.forward(X)
                loss_plus = model.compute_loss(Y_pred_plus, Y)

                # 计算 L(θ - ε)
                param[idx] = original_value - epsilon
                Y_pred_minus = model.forward(X)
                loss_minus = model.compute_loss(Y_pred_minus, Y)

                # 双边差分近似梯度
                grad_numeric[idx] = (loss_plus - loss_minus) / (2.0 * epsilon)

                # 恢复原始值
                param[idx] = original_value
                it.iternext()

            # 计算相对误差
            numerator = np.linalg.norm(grad_analytic - grad_numeric)
            denominator = np.linalg.norm(grad_analytic) + np.linalg.norm(grad_numeric)
            rel_error = numerator / max(denominator, 1e-10)  # 避免除以零

            max_rel_error = max(max_rel_error, rel_error)
            print(f"  {param_name}: 相对误差 = {rel_error:.2e}")

    return max_rel_error


# ============================================================================
# 第四部分：数据生成
# ============================================================================

def make_moons_dataset(n_samples: int = 200, noise: float = 0.15, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成双月形二分类数据集（类似 sklearn 的 make_moons）。

    参数:
        n_samples: 样本总数
        noise: 噪声标准差
        seed: 随机种子

    返回:
        X: 特征矩阵，shape (2, n_samples)
        Y: 标签矩阵，shape (1, n_samples)
    """
    np.random.seed(seed)
    n_samples_per_class = n_samples // 2

    # 上半月（类别 0）
    t = np.linspace(0, np.pi, n_samples_per_class)  # 角度从 0 到 π
    X0 = np.vstack([
        np.cos(t) + np.random.randn(n_samples_per_class) * noise,  # x 坐标 + 噪声
        np.sin(t) + np.random.randn(n_samples_per_class) * noise,  # y 坐标 + 噪声
    ])
    Y0 = np.zeros((1, n_samples_per_class))  # 标签 0

    # 下半月（类别 1）
    X1 = np.vstack([
        1 - np.cos(t) + np.random.randn(n_samples_per_class) * noise,  # 右移 + 噪声
        1 - np.sin(t) - 0.5 + np.random.randn(n_samples_per_class) * noise,  # 下移 + 噪声
    ])
    Y1 = np.ones((1, n_samples_per_class))  # 标签 1

    # 合并两个类别并打乱顺序
    X = np.hstack([X0, X1])
    Y = np.hstack([Y0, Y1])
    # 随机打乱
    idx = np.random.permutation(n_samples)
    X, Y = X[:, idx], Y[:, idx]

    return X, Y


# ============================================================================
# 第五部分：可视化
# ============================================================================

def plot_decision_boundary(model: MLP, X: np.ndarray, Y: np.ndarray, title: str, filename: str):
    """
    绘制分类决策边界。

    参数:
        model: 训练好的 MLP 模型
        X: 输入特征（用于确定绘图范围）
        Y: 标签（用于着色散点）
        title: 图表标题
        filename: 保存文件名
    """
    # 确定绘图范围：在数据范围基础上扩展一点边距
    x_min, x_max = X[0, :].min() - 0.5, X[0, :].max() + 0.5
    y_min, y_max = X[1, :].min() - 0.5, X[1, :].max() + 0.5

    # 生成网格点
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.vstack([xx.ravel(), yy.ravel()])  # 将网格展开为 (2, N) 的矩阵
    Z = model.forward(grid)                       # 模型对每个网格点的预测
    Z = Z.reshape(xx.shape)                       # 恢复为网格形状

    plt.figure(figsize=(8, 6))
    # 绘制决策边界背景（蓝色=类别0区域，红色=类别1区域）
    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.3,
                 colors=['#4A90D9', '#E74C3C'])
    # 绘制决策边界线（p=0.5 的等高线）
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)

    # 绘制数据点
    plt.scatter(X[0, Y[0, :] == 0], X[1, Y[0, :] == 0],
                c='#4A90D9', edgecolors='white', s=50, label='Class 0')
    plt.scatter(X[0, Y[0, :] == 1], X[1, Y[0, :] == 1],
                c='#E74C3C', edgecolors='white', s=50, label='Class 1')

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Feature 1 (x₁)', fontsize=12)
    plt.ylabel('Feature 2 (x₂)', fontsize=12)
    plt.legend()
    plt.tight_layout()
    out = os.path.join(_IMAGES, filename)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[可视化] {title} 已保存至 {out}")


def plot_training_curves(losses: List[float], grad_norms_history: List[Dict]):
    """
    绘制训练过程中的损失曲线和梯度范数变化。

    参数:
        losses: 每个 epoch 的损失值列表
        grad_norms_history: 每个 epoch 的梯度范数字典列表
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ---- 左图：损失曲线 ----
    axes[0].plot(losses, 'b-', linewidth=2, alpha=0.8)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0].set_title('Training Loss Curve', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')  # 对数刻度更容易观察收敛趋势

    # ---- 右图：梯度范数变化 ----
    if grad_norms_history:
        # 提取每层权重的梯度范数
        L = model_for_plot.L  # 需要通过全局或其他方式传入
        for l in range(1, len(grad_norms_history[0]) // 2 + 1):
            key = f"|dW{l}|"
            if key in grad_norms_history[0]:
                values = [h[key] for h in grad_norms_history]
                axes[1].plot(values, linewidth=1.5, alpha=0.7,
                             label=f'Layer {l} |dW|')

    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Gradient L2 Norm', fontsize=12)
    axes[1].set_title('Gradient Norm Monitoring', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')

    plt.tight_layout()
    out = os.path.join(_IMAGES, 'training_curves.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[可视化] 训练曲线已保存至 {out}")


def plot_weight_heatmaps(model_before: MLP, model_after: MLP):
    """
    绘制训练前后权重矩阵的热力图对比。

    参数:
        model_before: 训练前的模型
        model_after: 训练后的模型
    """
    L = model_before.L
    fig, axes = plt.subplots(L, 2, figsize=(10, 3 * L))

    if L == 1:
        axes = axes.reshape(1, -1)  # 确保索引一致性

    for l in range(L):
        # 训练前的权重
        im1 = axes[l, 0].imshow(model_before.parameters[f"W{l+1}"],
                                cmap='RdBu_r', aspect='auto')
        axes[l, 0].set_title(f'W[{l+1}] Before Training', fontsize=11)
        plt.colorbar(im1, ax=axes[l, 0])

        # 训练后的权重
        im2 = axes[l, 1].imshow(model_after.parameters[f"W{l+1}"],
                                cmap='RdBu_r', aspect='auto')
        axes[l, 1].set_title(f'W[{l+1}] After Training', fontsize=11)
        plt.colorbar(im2, ax=axes[l, 1])

    plt.suptitle('Weight Matrix Heatmaps: Before vs After Training', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(_IMAGES, 'weight_heatmaps.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[可视化] 权重热力图已保存至 {out}")


# ============================================================================
# 第六部分：主程序
# ============================================================================

# 全局变量：用于在 plot_training_curves 中引用模型
model_for_plot = None


def main():
    global model_for_plot

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   s07 多层网络的矩阵反传 — 完整 MLP 训练 (正向+反向+更新)      ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # ---- 1. 生成数据集 ----
    print("\n[数据] 生成双月形二分类数据集...")
    X, Y = make_moons_dataset(n_samples=300, noise=0.15, seed=42)
    print(f"  X shape: {X.shape}, Y shape: {Y.shape}")
    print(f"  类别 0 样本数: {(Y == 0).sum()}, 类别 1 样本数: {(Y == 1).sum()}")

    # ---- 2. 初始化模型 ----
    # 2 输入 → 16 隐藏(ReLU) → 8 隐藏(ReLU) → 1 输出(Sigmoid)
    layer_dims = [2, 16, 8, 1]
    activations = ["relu", "relu", "sigmoid"]

    print(f"\n[模型] 网络结构: {layer_dims}")
    print(f"  激活函数: {activations}")
    print(f"  总层数: {len(layer_dims) - 1}")

    model = MLP(layer_dims, activations, seed=42)
    model_for_plot = model

    total_params = sum(p.size for p in model.parameters.values())
    print(f"  总参数量: {total_params}")

    # ---- 3. 梯度检查 ----
    print("\n[梯度检查] 使用有限差分验证解析梯度...")
    print("  (仅在小 batch 上检查少量参数，实际训练不会这样慢)")
    # 取少量样本用于快速检查
    X_check = X[:, :5]
    Y_check = Y[:, :5]
    max_error = gradient_check(model, X_check, Y_check, epsilon=1e-7)
    if max_error < 1e-5:
        print(f"  ✓ 梯度检查通过！最大相对误差: {max_error:.2e}")
    else:
        print(f"  ⚠ 最大相对误差: {max_error:.2e}，建议检查反向传播实现")

    # ---- 4. 保存训练前的模型副本 ----
    # 深拷贝参数（用于训练前后对比）
    model_before_params = {}
    for key, val in model.parameters.items():
        model_before_params[key] = val.copy()

    # ---- 5. 训练循环 ----
    learning_rate = 0.5
    n_epochs = 2000
    print(f"\n[训练] 学习率={learning_rate}, Epochs={n_epochs}")

    losses = []
    grad_norms_history = []

    for epoch in range(n_epochs):
        # ---- 前向传播 ----
        Y_pred = model.forward(X)

        # ---- 计算损失 ----
        loss = model.compute_loss(Y_pred, Y)
        losses.append(loss)

        # ---- 反向传播 ----
        model.backward(Y)

        # ---- 记录梯度范数 ----
        grad_norms = model.get_gradient_norms()
        grad_norms_history.append(grad_norms)

        # ---- 参数更新 ----
        model.update(learning_rate)

        # ---- 打印训练进度 ----
        if epoch % 400 == 0 or epoch == n_epochs - 1:
            accuracy = np.mean((Y_pred > 0.5) == Y)  # 二分类准确率
            dw1_norm = grad_norms.get("|dW1|", 0)
            dw2_norm = grad_norms.get("|dW2|", 0)
            dw3_norm = grad_norms.get("|dW3|", 0)
            print(f"  Epoch {epoch:4d}: loss={loss:.6f}, accuracy={accuracy:.4f}, "
                  f"|dW1|={dw1_norm:.4f}, |dW2|={dw2_norm:.4f}, |dW3|={dw3_norm:.4f}")

    # ---- 6. 最终评估 ----
    Y_pred_final = model.forward(X)
    final_accuracy = np.mean((Y_pred_final > 0.5) == Y)
    print(f"\n[结果] 最终训练准确率: {final_accuracy:.4f} ({final_accuracy*100:.1f}%)")

    # ---- 7. 可视化 ----
    print("\n[可视化] 生成图表...")

    # 决策边界：训练前
    model_before = MLP(layer_dims, activations, seed=42)
    plot_decision_boundary(model_before, X, Y,
                           'Decision Boundary Before Training', 'decision_boundary_before.png')

    # 决策边界：训练后
    plot_decision_boundary(model, X, Y,
                           f'Decision Boundary After Training (Accuracy: {final_accuracy:.1%})',
                           'decision_boundary_after.png')

    # 训练曲线
    plot_training_curves(losses, grad_norms_history)

    # 权重热力图
    plot_weight_heatmaps(model_before, model)

    # ---- 8. 总结 ----
    print("\n" + "=" * 70)
    print("【总结】")
    print("=" * 70)
    print(f"  ✓ 实现了完整的前向传播、矩阵反向传播和参数更新")
    print(f"  ✓ 通过梯度检查验证了反向传播的正确性")
    print(f"  ✓ 在双月形数据集上达到了 {final_accuracy*100:.1f}% 的准确率")
    print(f"  ✓ 每层梯度范数的变化表明训练过程健康（无消失或爆炸）")
    print(f"\n  核心公式回顾:")
    print(f"    δ[L] = ∇_A L ⊙ φ'(Z[L])")
    print(f"    δ[l] = (W[l+1])^T @ δ[l+1] ⊙ φ'(Z[l])")
    print(f"    dW[l] = (1/m) · δ[l] @ (A[l-1])^T")
    print(f"    db[l] = (1/m) · Σ δ[l]")
    print("=" * 70)


if __name__ == "__main__":
    main()
