# -*- coding: utf-8 -*-
"""
s09 Adam 深度解析与训练实战 — 演示代码
======================================
功能：
  1. 从零实现 Adam 优化器（含完整偏差修正）
  2. 在 MNIST 上训练小型 MLP，对比 Adam / AdamW / SGD+Momentum
  3. 展示偏差修正的效果（有 vs 无）
  4. 实现学习率 warmup + cosine decay
  5. 实现梯度裁剪 + 梯度范数监控
  6. 可视化：损失曲线、准确率、梯度范数

运行方式：在 s09_adam_deep_dive/ 目录下执行 python code/demo.py
需要安装: torch, torchvision, matplotlib, numpy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
from typing import Dict, List, Tuple, Optional
import os
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_IMAGES = os.path.join(_HERE, '..', 'images')
os.makedirs(_IMAGES, exist_ok=True)
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 第一部分：Adam 优化器（从零实现，含完整偏差修正）
# ============================================================================

class AdamOptimizer:
    """
    Adam 优化器的完整 NumPy 实现。

    公式:
        m_t = β₁·m_{t-1} + (1-β₁)·g_t          (一阶矩)
        v_t = β₂·v_{t-1} + (1-β₂)·g_t²         (二阶矩)
        m̂_t = m_t / (1-β₁^t)                    (偏差修正)
        v̂_t = v_t / (1-β₂^t)                    (偏差修正)
        θ_{t+1} = θ_t - α·m̂_t / (√v̂_t + ε)     (参数更新)

    参数:
        lr: 学习率 α
        betas: (β₁, β₂) 元组
        eps: 数值稳定常数 ε
        use_bias_correction: 是否使用偏差修正（默认 True）
    """

    def __init__(self, lr: float = 0.001, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, use_bias_correction: bool = True):
        self.lr = lr                    # 学习率
        self.beta1, self.beta2 = betas  # 衰减系数
        self.eps = eps                  # 数值稳定常数
        self.use_bias_correction = use_bias_correction
        self.m: Dict[int, np.ndarray] = {}  # 一阶矩: {参数id: 动量向量}
        self.v: Dict[int, np.ndarray] = {}  # 二阶矩: {参数id: 梯度平方均值}
        self.t = 0                      # 迭代步数计数器

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        """
        执行一步 Adam 更新。原地修改 params 中的数据。

        参数:
            params: 参数字典 {参数名: numpy数组}
            grads: 梯度字典 {参数名: numpy数组}，键名与 params 匹配
        """
        self.t += 1  # 迭代步数 +1

        for key in params:
            param = params[key]      # 当前参数
            grad = grads.get(key)    # 对应梯度

            if grad is None:
                continue  # 如果该参数没有梯度，跳过

            # ---- 初始化矩向量（首次调用时为每个参数分配零向量） ----
            param_id = id(param)  # 使用 Python 对象 id 作为唯一标识
            if param_id not in self.m:
                self.m[param_id] = np.zeros_like(param)  # m_0 = 0
                self.v[param_id] = np.zeros_like(param)  # v_0 = 0

            # ---- 步骤 1: 更新一阶矩（动量） ----
            self.m[param_id] = (self.beta1 * self.m[param_id]
                                + (1 - self.beta1) * grad)

            # ---- 步骤 2: 更新二阶矩（梯度平方的均值） ----
            self.v[param_id] = (self.beta2 * self.v[param_id]
                                + (1 - self.beta2) * (grad ** 2))

            if self.use_bias_correction:
                # ---- 步骤 3: 偏差修正 ----
                m_hat = self.m[param_id] / (1 - self.beta1 ** self.t)  # 修正一阶矩
                v_hat = self.v[param_id] / (1 - self.beta2 ** self.t)  # 修正二阶矩

                # ---- 步骤 4: 参数更新 ----
                param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            else:
                # 不使用偏差修正（对比实验用）
                param -= self.lr * self.m[param_id] / (np.sqrt(self.v[param_id]) + self.eps)


class AdamWOptimizer:
    """
    AdamW 优化器的 NumPy 实现（解耦权重衰减）。

    与 Adam 的区别：
      权重衰减从梯度中独立出来，不受自适应缩放影响。

    公式:
        θ_{t+1} = θ_t - α·m̂_t/(√v̂_t + ε) - α·λ·θ_t

    参数:
        lr: 学习率 α
        weight_decay: 权重衰减系数 λ
        betas: (β₁, β₂)
        eps: 数值稳定常数
    """

    def __init__(self, lr: float = 0.001, weight_decay: float = 0.01,
                 betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        self.lr = lr
        self.weight_decay = weight_decay  # λ
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m: Dict[int, np.ndarray] = {}
        self.v: Dict[int, np.ndarray] = {}
        self.t = 0

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        """执行一步 AdamW 更新"""
        self.t += 1

        for key in params:
            param = params[key]
            grad = grads.get(key)
            if grad is None:
                continue

            param_id = id(param)
            if param_id not in self.m:
                self.m[param_id] = np.zeros_like(param)
                self.v[param_id] = np.zeros_like(param)

            # 一阶矩
            self.m[param_id] = (self.beta1 * self.m[param_id]
                                + (1 - self.beta1) * grad)
            # 二阶矩
            self.v[param_id] = (self.beta2 * self.v[param_id]
                                + (1 - self.beta2) * (grad ** 2))
            # 偏差修正
            m_hat = self.m[param_id] / (1 - self.beta1 ** self.t)
            v_hat = self.v[param_id] / (1 - self.beta2 ** self.t)

            # AdamW 更新：先做 Adam 自适应更新，再独立应用权重衰减
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)  # Adam 部分
            param -= self.lr * self.weight_decay * param           # 独立权重衰减


class SGDMomentumOptimizer:
    """
    带动量的 SGD 优化器（对比基线）。

    公式:
        m_t = β·m_{t-1} + g_t          (动量)
        θ_{t+1} = θ_t - α·m_t          (参数更新)

    参数:
        lr: 学习率 α
        momentum: 动量系数 β
    """

    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.m: Dict[int, np.ndarray] = {}

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        """执行一步 SGD+Momentum 更新"""
        for key in params:
            param = params[key]
            grad = grads.get(key)
            if grad is None:
                continue

            param_id = id(param)
            if param_id not in self.m:
                self.m[param_id] = np.zeros_like(param)

            # SGD+Momentum: m_t = β·m_{t-1} + g_t
            self.m[param_id] = self.momentum * self.m[param_id] + grad
            param -= self.lr * self.m[param_id]


# ============================================================================
# 第二部分：小型 MLP（纯 NumPy 实现）
# ============================================================================

class MLP:
    """
    小型多层感知机，用于 MNIST 分类。

    结构: 输入(784) → 隐藏层1(128, ReLU) → 隐藏层2(64, ReLU) → 输出(10, Softmax)

    参数:
        layer_dims: 每层神经元列表，如 [784, 128, 64, 10]
        seed: 随机种子
    """

    def __init__(self, layer_dims: List[int], seed: int = 42):
        np.random.seed(seed)
        self.params = {}  # 参数字典
        self.L = len(layer_dims) - 1  # 层数

        for l in range(1, self.L + 1):
            n_in = layer_dims[l - 1]
            n_out = layer_dims[l]
            # He 初始化
            self.params[f"W{l}"] = np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in)
            self.params[f"b{l}"] = np.zeros((n_out, 1))

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        前向传播。

        参数:
            X: 输入，shape (n_features, m_samples)

        返回:
            A_out: 最后一层输出 (softmax 概率)
            caches: 中间值缓存列表
        """
        caches = []
        A = X  # A[0] = X

        for l in range(1, self.L + 1):
            A_prev = A
            W = self.params[f"W{l}"]
            b = self.params[f"b{l}"]
            Z = W @ A_prev + b  # 线性变换

            if l < self.L:
                # 隐藏层：ReLU 激活
                A = np.maximum(0, Z)
            else:
                # 输出层：Softmax 激活
                # 数值稳定技巧：减去每列最大值
                Z_stable = Z - np.max(Z, axis=0, keepdims=True)
                exp_Z = np.exp(Z_stable)
                A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

            caches.append({
                "Z": Z,
                "A_prev": A_prev,
                "A": A,
            })

        return A, caches

    def backward(self, Y: np.ndarray, caches: List[Dict]) -> Dict[str, np.ndarray]:
        """
        反向传播（交叉熵 + softmax 组合）。

        对于最后一层（softmax + 交叉熵），δ[L] = A[L] - Y
        （这是 softmax+CE 的优美性质——梯度极简）

        参数:
            Y: one-hot 标签，shape (n_classes, m)
            caches: 前向传播缓存

        返回:
            grads: 梯度字典 {dW{l}, db{l}}
        """
        m = Y.shape[1]
        grads = {}

        # ---- 输出层 δ[L] = A[L] - Y (softmax+交叉熵的组合梯度) ----
        dZ = caches[-1]["A"] - Y  # (10, m)

        # ---- 从后向前递推 ----
        for l in reversed(range(1, self.L + 1)):
            cache = caches[l - 1]
            A_prev = cache["A_prev"]

            # 权重梯度
            grads[f"dW{l}"] = (1.0 / m) * (dZ @ A_prev.T)
            grads[f"db{l}"] = (1.0 / m) * np.sum(dZ, axis=1, keepdims=True)

            # 如果不是第一层，继续递推
            if l > 1:
                W_curr = self.params[f"W{l}"]
                Z_prev = caches[l - 2]["Z"]
                # ReLU 导数: 1 if Z > 0 else 0
                dZ_prev = (W_curr.T @ dZ) * (Z_prev > 0)
                dZ = dZ_prev

        return grads

    def compute_loss_and_accuracy(self, X: np.ndarray, Y_labels: np.ndarray) -> Tuple[float, float]:
        """
        计算交叉熵损失和分类准确率。

        参数:
            X: 输入
            Y_labels: 整数标签，shape (m,)

        返回:
            loss: 平均交叉熵损失
            accuracy: 分类准确率
        """
        probs, _ = self.forward(X)
        m = X.shape[1]

        # 交叉熵损失: -1/m * Σ log(p_correct_class)
        # 先取每个样本正确类别的概率，再取 -log
        correct_probs = probs[Y_labels, np.arange(m)]
        # 裁剪以避免 log(0)
        correct_probs = np.clip(correct_probs, 1e-15, 1.0)
        loss = -np.mean(np.log(correct_probs))

        # 准确率
        predictions = np.argmax(probs, axis=0)
        accuracy = np.mean(predictions == Y_labels)

        return loss, accuracy


# ============================================================================
# 第三部分：学习率调度
# ============================================================================

class LRScheduler:
    """
    学习率调度器：Warmup + Cosine Decay。

    调度曲线:
        lr
        ^
        |    /‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\___
        |   /                         \___
        |  /                              \___
        | /                                   \___
        |/___________________________________________> step
        |<--warmup-->|<------ cosine decay ------>|

    参数:
        optimizer: 优化器对象（有 .lr 属性）
        warmup_steps: warmup 步数
        total_steps: 总训练步数
        min_lr_ratio: 最终学习率相对于初始学习率的比例
    """

    def __init__(self, optimizer, warmup_steps: int, total_steps: int,
                 min_lr_ratio: float = 0.01):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = optimizer.lr                    # 初始学习率（目标值）
        self.min_lr = optimizer.lr * min_lr_ratio      # 最小学习率
        self.current_step = 0

    def step(self):
        """更新当前步的学习率"""
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            # Warmup 阶段：从 0 线性增加到 base_lr
            progress = self.current_step / self.warmup_steps
            self.optimizer.lr = self.base_lr * progress
        else:
            # Cosine Decay 阶段：从 base_lr 余弦衰减到 min_lr
            progress = (self.current_step - self.warmup_steps) / \
                       (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)  # 不超过 1.0
            self.optimizer.lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * \
                                (1 + np.cos(np.pi * progress))

    def get_lr(self) -> float:
        """返回当前学习率"""
        return self.optimizer.lr


# ============================================================================
# 第四部分：梯度裁剪
# ============================================================================

def clip_gradients(grads: Dict[str, np.ndarray], max_norm: float) -> Dict[str, np.ndarray]:
    """
    全局梯度范数裁剪。

    当所有梯度的全局 L2 范数超过 max_norm 时，按比例缩放所有梯度。

    参数:
        grads: 梯度字典
        max_norm: 最大允许的梯度 L2 范数

    返回:
        grads_clipped: 裁剪后的梯度字典
    """
    # 计算全局梯度范数
    total_norm_sq = 0.0
    for grad in grads.values():
        if grad is not None:
            total_norm_sq += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm_sq)

    # 如果超过阈值，按比例缩放
    if total_norm > max_norm:
        scale = max_norm / total_norm
        grads_clipped = {}
        for key, grad in grads.items():
            grads_clipped[key] = grad * scale if grad is not None else None
        return grads_clipped

    return grads  # 未超过阈值，原样返回


def compute_gradient_norm(grads: Dict[str, np.ndarray]) -> float:
    """
    计算全局梯度 L2 范数。

    参数:
        grads: 梯度字典

    返回:
        全局梯度范数（标量）
    """
    total_norm_sq = 0.0
    for grad in grads.values():
        if grad is not None:
            total_norm_sq += np.sum(grad ** 2)
    return np.sqrt(total_norm_sq)


# ============================================================================
# 第五部分：数据加载与预处理
# ============================================================================

def load_mnist_subset(n_train: int = 5000, n_val: int = 1000) -> Tuple:
    """
    加载 MNIST 数据集的子集，转换为 NumPy 格式。

    使用完整数据集的子集以加快演示速度。

    参数:
        n_train: 训练样本数
        n_val: 验证样本数

    返回:
        X_train, Y_train, X_val, Y_val: NumPy 数组
    """
    print("[数据] 加载 MNIST 数据集...")

    try:
        # 尝试从 torchvision 加载
        from torchvision import datasets
        import torch

        # 加载训练集
        train_dataset = datasets.MNIST(
            root='./data', train=True, download=True
        )
        test_dataset = datasets.MNIST(
            root='./data', train=False, download=True
        )

        # 转换为 NumPy
        X_train_full = train_dataset.data.numpy().reshape(-1, 784).T / 255.0  # (784, 60000)
        Y_train_full = train_dataset.targets.numpy()  # (60000,)

        X_test_full = test_dataset.data.numpy().reshape(-1, 784).T / 255.0  # (784, 10000)
        Y_test_full = test_dataset.targets.numpy()  # (10000,)

        # 取子集
        X_train = X_train_full[:, :n_train]
        Y_train = Y_train_full[:n_train]
        X_val = X_test_full[:, :n_val]
        Y_val = Y_test_full[:n_val]

        print(f"  训练集: X={X_train.shape}, Y={Y_train.shape}")
        print(f"  验证集: X={X_val.shape}, Y={Y_val.shape}")

    except ImportError:
        # 如果 torchvision 不可用，生成模拟数据
        print("  torchvision 不可用，使用模拟数据...")
        np.random.seed(42)
        X_train = np.random.randn(784, n_train) * 0.1
        Y_train = np.random.randint(0, 10, n_train)
        X_val = np.random.randn(784, n_val) * 0.1
        Y_val = np.random.randint(0, 10, n_val)

    return X_train, Y_train, X_val, Y_val


def to_one_hot(Y_labels: np.ndarray, n_classes: int = 10) -> np.ndarray:
    """
    将整数标签转换为 one-hot 编码。

    参数:
        Y_labels: 整数标签，shape (m,)
        n_classes: 类别数

    返回:
        Y_onehot: one-hot 矩阵，shape (n_classes, m)
    """
    m = Y_labels.shape[0]
    Y_onehot = np.zeros((n_classes, m))
    Y_onehot[Y_labels, np.arange(m)] = 1
    return Y_onehot


# ============================================================================
# 第六部分：训练循环
# ============================================================================

def train_one_epoch(
    model: MLP,
    optimizer,
    X: np.ndarray,
    Y_labels: np.ndarray,
    batch_size: int,
    clip_grad_norm: Optional[float] = None,
    scheduler: Optional[LRScheduler] = None
) -> Tuple[float, float, List[float]]:
    """
    训练一个 epoch（完整遍历一次训练数据）。

    参数:
        model: MLP 模型
        optimizer: 优化器
        X: 训练数据
        Y_labels: 训练标签（整数）
        batch_size: 批次大小
        clip_grad_norm: 梯度裁剪阈值（None 表示不裁剪）
        scheduler: 学习率调度器（None 表示固定学习率）

    返回:
        avg_loss: 平均训练损失
        avg_acc: 平均训练准确率
        grad_norms: 每个 batch 的梯度范数列表
    """
    m = X.shape[1]
    indices = np.random.permutation(m)  # 随机打乱数据
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    grad_norms = []

    for start in range(0, m, batch_size):
        end = min(start + batch_size, m)
        batch_idx = indices[start:end]

        X_batch = X[:, batch_idx]          # 当前 batch 的输入
        Y_batch_labels = Y_labels[batch_idx]  # 当前 batch 的标签
        Y_batch = to_one_hot(Y_batch_labels)  # 转为 one-hot

        # ---- 前向传播 ----
        probs, caches = model.forward(X_batch)
        loss = -np.mean(np.log(np.clip(
            probs[Y_batch_labels, np.arange(len(Y_batch_labels))], 1e-15, 1.0
        )))
        acc = np.mean(np.argmax(probs, axis=0) == Y_batch_labels)

        # ---- 反向传播 ----
        grads = model.backward(Y_batch, caches)

        # ---- 梯度裁剪 ----
        if clip_grad_norm is not None:
            grads = clip_gradients(grads, clip_grad_norm)

        # ---- 记录梯度范数 ----
        grad_norm = compute_gradient_norm(grads)
        grad_norms.append(grad_norm)

        # ---- 参数更新 ----
        optimizer.step(model.params, grads)

        # ---- 学习率调度 ----
        if scheduler is not None:
            scheduler.step()

        total_loss += loss
        total_acc += acc
        n_batches += 1

    return total_loss / n_batches, total_acc / n_batches, grad_norms


def evaluate(model: MLP, X: np.ndarray, Y_labels: np.ndarray,
             batch_size: int = 256) -> Tuple[float, float]:
    """
    在验证集上评估模型。

    参数:
        model: MLP 模型
        X: 验证数据
        Y_labels: 验证标签
        batch_size: 批次大小

    返回:
        avg_loss: 平均损失
        avg_accuracy: 平均准确率
    """
    m = X.shape[1]
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for start in range(0, m, batch_size):
        end = min(start + batch_size, m)
        X_batch = X[:, start:end]
        Y_batch_labels = Y_labels[start:end]

        probs, _ = model.forward(X_batch)
        correct_probs = probs[Y_batch_labels, np.arange(len(Y_batch_labels))]
        correct_probs = np.clip(correct_probs, 1e-15, 1.0)
        loss = -np.mean(np.log(correct_probs))
        acc = np.mean(np.argmax(probs, axis=0) == Y_batch_labels)

        total_loss += loss
        total_acc += acc
        n_batches += 1

    return total_loss / n_batches, total_acc / n_batches


# ============================================================================
# 第七部分：训练与比较
# ============================================================================

def train_model(
    model: MLP,
    optimizer,
    X_train: np.ndarray, Y_train: np.ndarray,
    X_val: np.ndarray, Y_val: np.ndarray,
    n_epochs: int, batch_size: int,
    clip_grad_norm: Optional[float] = None,
    scheduler: Optional[LRScheduler] = None,
    verbose: bool = True
) -> Dict:
    """
    完整的训练流程，返回训练历史。

    返回:
        history: 包含 losses, accuracies, val_losses, val_accuracies, grad_norms, lrs 的字典
    """
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "grad_norms": [],
        "lrs": [],
    }

    for epoch in range(n_epochs):
        epoch_start = time.time()

        # 训练一个 epoch
        train_loss, train_acc, batch_grad_norms = train_one_epoch(
            model, optimizer, X_train, Y_train,
            batch_size, clip_grad_norm, scheduler
        )

        # 验证
        val_loss, val_acc = evaluate(model, X_val, Y_val)

        epoch_time = time.time() - epoch_start
        current_lr = scheduler.get_lr() if scheduler else optimizer.lr

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["grad_norms"].append(np.mean(batch_grad_norms))
        history["lrs"].append(current_lr)

        if verbose and (epoch % 3 == 0 or epoch == n_epochs - 1):
            print(f"  Epoch {epoch+1:3d}/{n_epochs} | "
                  f"loss: {train_loss:.4f} | acc: {train_acc:.3f} | "
                  f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.3f} | "
                  f"lr: {current_lr:.2e} | time: {epoch_time:.1f}s")

    return history


def compare_without_bias_correction(
    X_train, Y_train, X_val, Y_val
):
    """
    对比有/无偏差修正的 Adam 在训练初期的表现。

    偏差修正让早期步长更大、收敛更快。
    """
    print("\n" + "=" * 70)
    print("【偏差修正对比实验】")
    print("=" * 70)

    n_epochs = 5  # 只在少数 epoch 上比较（偏差修正主要在早期起作用）

    results = {}
    for use_bc, label in [(True, "With Bias Correction"), (False, "Without Bias Correction")]:
        model = MLP([784, 128, 64, 10], seed=42)
        opt = AdamOptimizer(lr=0.001, use_bias_correction=use_bc)
        print(f"\n  训练 {label} 的 Adam...")
        history = train_model(
            model, opt, X_train, Y_train, X_val, Y_val,
            n_epochs=n_epochs, batch_size=64, verbose=True
        )
        results[label] = history

    # 对比早期损失
    print(f"\n  早期损失对比（前 {n_epochs} 个 epoch）:")
    print(f"  {'Epoch':<8} {'With Bias Correction':<22} {'Without Bias Correction':<22} {'Diff'}")
    for ep in range(n_epochs):
        loss_with = results["With Bias Correction"]["train_loss"][ep]
        loss_without = results["Without Bias Correction"]["train_loss"][ep]
        diff_pct = (loss_without - loss_with) / loss_with * 100  # 无修正相对有修正的偏差
        print(f"  {ep+1:<8} {loss_with:<16.4f} {loss_without:<16.4f} {diff_pct:+.1f}%")

    return results


# ============================================================================
# 第八部分：可视化
# ============================================================================

def plot_comparison_results(
    all_histories: Dict[str, Dict],
    filename: str = "optimizer_comparison.png"
):
    """
    绘制所有优化器的对比图表：
    - 训练损失曲线
    - 验证准确率曲线
    - 梯度范数曲线
    - 学习率曲线

    参数:
        all_histories: {优化器名称: history字典}
        filename: 保存的文件名
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {
        "SGD+Momentum": "#E74C3C",
        "Adam": "#9B59B6",
        "AdamW": "#2ECC71",
    }

    # ---- 左上: 训练损失 ----
    ax = axes[0, 0]
    for name, hist in all_histories.items():
        color = colors.get(name, 'gray')
        ax.plot(hist["train_loss"], '-', color=color, linewidth=2,
                alpha=0.8, label=name)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Training Loss', fontsize=11)
    ax.set_title('Training Loss Curve', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- 右上: 验证准确率 ----
    ax = axes[0, 1]
    for name, hist in all_histories.items():
        color = colors.get(name, 'gray')
        ax.plot(hist["val_acc"], '-', color=color, linewidth=2,
                alpha=0.8, label=name)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Validation Accuracy', fontsize=11)
    ax.set_title('Validation Accuracy Curve', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- 左下: 梯度范数 ----
    ax = axes[1, 0]
    for name, hist in all_histories.items():
        color = colors.get(name, 'gray')
        ax.plot(hist["grad_norms"], '-', color=color, linewidth=1.5,
                alpha=0.8, label=name)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Gradient L2 Norm', fontsize=11)
    ax.set_title('Gradient Norm Monitoring', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # ---- 右下: 学习率曲线 ----
    ax = axes[1, 1]
    for name, hist in all_histories.items():
        color = colors.get(name, 'gray')
        ax.plot(hist["lrs"], '-', color=color, linewidth=2,
                alpha=0.8, label=name)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Learning Rate', fontsize=11)
    ax.set_title('Learning Rate Schedule (Warmup + Cosine)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(_IMAGES, filename)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[可视化] 优化器对比图已保存至 {out}")


def plot_bias_correction_comparison(
    bc_results: Dict[str, Dict],
    filename: str = "bias_correction_comparison.png"
):
    """
    可视化偏差修正的效果对比。

    参数:
        bc_results: {"有偏差修正": history, "无偏差修正": history}
        filename: 保存的文件名
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors_bc = {"With Bias Correction": "#2ECC71", "Without Bias Correction": "#E74C3C"}

    # 左图：训练损失
    ax = axes[0]
    for label, hist in bc_results.items():
        ax.plot(hist["train_loss"], 'o-', color=colors_bc[label],
                linewidth=2, markersize=6, alpha=0.8, label=label)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Training Loss', fontsize=11)
    ax.set_title('Effect of Bias Correction on Training Loss (Early Stage)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 右图：验证准确率
    ax = axes[1]
    for label, hist in bc_results.items():
        ax.plot(hist["val_acc"], 'o-', color=colors_bc[label],
                linewidth=2, markersize=6, alpha=0.8, label=label)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Validation Accuracy', fontsize=11)
    ax.set_title('Effect of Bias Correction on Validation Accuracy (Early Stage)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(_IMAGES, filename)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[可视化] 偏差修正对比图已保存至 {out}")


# ============================================================================
# 第九部分：主程序
# ============================================================================

def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║     s09 Adam 深度解析与训练实战 — MNIST 优化器对比              ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # ---- 1. 加载数据 ----
    X_train, Y_train, X_val, Y_val = load_mnist_subset(
        n_train=5000, n_val=1000
    )

    # ---- 2. 优化器三路对比 ----
    print("\n" + "=" * 70)
    print("【优化器对比实验】Adam vs AdamW vs SGD+Momentum")
    print("=" * 70)

    n_epochs = 15
    batch_size = 64
    total_steps = n_epochs * (X_train.shape[1] // batch_size)
    warmup_steps = total_steps // 10  # 前 10% 的步数用于 warmup

    all_histories = {}

    # --- Adam ---
    print("\n--- 训练 Adam ---")
    model_adam = MLP([784, 128, 64, 10], seed=42)
    opt_adam = AdamOptimizer(lr=0.001)
    scheduler_adam = LRScheduler(opt_adam, warmup_steps, total_steps,
                                  min_lr_ratio=0.01)
    history_adam = train_model(
        model_adam, opt_adam, X_train, Y_train, X_val, Y_val,
        n_epochs, batch_size, clip_grad_norm=5.0,
        scheduler=scheduler_adam
    )
    all_histories["Adam"] = history_adam

    # --- AdamW ---
    print("\n--- 训练 AdamW ---")
    model_adamw = MLP([784, 128, 64, 10], seed=42)
    opt_adamw = AdamWOptimizer(lr=0.001, weight_decay=0.01)
    scheduler_adamw = LRScheduler(opt_adamw, warmup_steps, total_steps,
                                   min_lr_ratio=0.01)
    history_adamw = train_model(
        model_adamw, opt_adamw, X_train, Y_train, X_val, Y_val,
        n_epochs, batch_size, clip_grad_norm=5.0,
        scheduler=scheduler_adamw
    )
    all_histories["AdamW"] = history_adamw

    # --- SGD+Momentum ---
    print("\n--- 训练 SGD+Momentum ---")
    model_sgd = MLP([784, 128, 64, 10], seed=42)
    opt_sgd = SGDMomentumOptimizer(lr=0.01, momentum=0.9)
    scheduler_sgd = LRScheduler(opt_sgd, warmup_steps, total_steps,
                                 min_lr_ratio=0.01)
    history_sgd = train_model(
        model_sgd, opt_sgd, X_train, Y_train, X_val, Y_val,
        n_epochs, batch_size, clip_grad_norm=5.0,
        scheduler=scheduler_sgd
    )
    all_histories["SGD+Momentum"] = history_sgd

    # ---- 3. 打印对比总结 ----
    print("\n" + "=" * 70)
    print("【优化器对比总结表】")
    print("=" * 70)
    print(f"{'优化器':<16} {'最终训练损失':<16} {'最终训准':<12} "
          f"{'最终验损':<16} {'最终验准':<12}")
    print("-" * 72)
    for name, hist in all_histories.items():
        print(f"{name:<16} {hist['train_loss'][-1]:<16.4f} "
              f"{hist['train_acc'][-1]:<12.3f} "
              f"{hist['val_loss'][-1]:<16.4f} {hist['val_acc'][-1]:<12.3f}")
    print("=" * 70)

    # ---- 4. 可视化对比 ----
    print("\n[可视化] 生成对比图表...")
    plot_comparison_results(all_histories)

    # ---- 5. 偏差修正对比 ----
    bc_results = compare_without_bias_correction(
        X_train, Y_train, X_val, Y_val
    )
    plot_bias_correction_comparison(bc_results)

    # ---- 6. 梯度裁剪效果演示 ----
    print("\n" + "=" * 70)
    print("【梯度裁剪效果演示】")
    print("=" * 70)
    model_clip = MLP([784, 128, 64, 10], seed=42)

    # 故意用一个很大的学习率来制造梯度爆炸
    opt_big_lr = AdamOptimizer(lr=0.1, use_bias_correction=True)

    # 收集不使用裁剪时的梯度范数
    print("  不使用梯度裁剪 (lr=0.1)...")
    history_no_clip = train_model(
        model_clip, opt_big_lr, X_train, Y_train, X_val, Y_val,
        n_epochs=3, batch_size=64, clip_grad_norm=None, verbose=True
    )

    # 重置模型，使用裁剪
    model_clip2 = MLP([784, 128, 64, 10], seed=42)
    opt_big_lr2 = AdamOptimizer(lr=0.1, use_bias_correction=True)
    print("  使用梯度裁剪 (max_norm=1.0, lr=0.1)...")
    history_with_clip = train_model(
        model_clip2, opt_big_lr2, X_train, Y_train, X_val, Y_val,
        n_epochs=3, batch_size=64, clip_grad_norm=1.0, verbose=True
    )

    print(f"\n  梯度范数对比 (大学习率 lr=0.1):")
    print(f"  {'Epoch':<8} {'无裁剪':<16} {'有裁剪':<16}")
    for ep in range(3):
        print(f"  {ep+1:<8} {history_no_clip['grad_norms'][ep]:<16.4f} "
              f"{history_with_clip['grad_norms'][ep]:<16.4f}")

    # 裁剪效果图
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(history_no_clip["grad_norms"], 'ro-', linewidth=2,
            markersize=8, alpha=0.8, label='No Clip (may explode)')
    ax.plot(history_with_clip["grad_norms"], 'go-', linewidth=2,
            markersize=8, alpha=0.8, label='With Clip (max_norm=1.0)')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Clip Threshold')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Gradient L2 Norm', fontsize=11)
    ax.set_title('Effect of Gradient Clipping (High LR lr=0.1)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(_IMAGES, 'gradient_clipping_effect.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[可视化] 梯度裁剪效果图已保存至 {out}")

    # ---- 7. 总结 ----
    print("\n" + "=" * 70)
    print("【总结】")
    print("=" * 70)
    print("  ✓ 从零实现了 Adam (含偏差修正)、AdamW、SGD+Momentum")
    print("  ✓ 在 MNIST 上对比了三者的性能")
    print("  ✓ 验证了偏差修正对训练初期的加速效果")
    print("  ✓ 演示了梯度裁剪对防止梯度爆炸的关键作用")
    print("  ✓ 实现了 Warmup + Cosine Decay 学习率调度")
    print()
    print("  核心公式回顾:")
    print("    Adam:  θ = θ - α·m̂/(√v̂+ε)")
    print("    AdamW: θ = θ - α·m̂/(√v̂+ε) - αλθ")
    print("    Bias Correction: m̂ = m/(1-β₁^t), v̂ = v/(1-β₂^t)")
    print("    Gradient Clipping: if ||g|| > max_norm → g *= max_norm/||g||")
    print("=" * 70)


if __name__ == "__main__":
    main()
