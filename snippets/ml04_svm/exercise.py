# -*- coding: utf-8 -*-
"""
===============================================================================
ml04_svm/code/exercise.py — SVM 练习
===============================================================================
本练习文件中，你需要完成以下任务：

练习目标：
  1. 实现 Hinge Loss 的子梯度计算
  2. 实现线性 SVM 的 SGD 单步更新
  3. 实现 RBF 核矩阵的计算（Bonus）

提示：
  - Hinge Loss: L(f) = max(0, 1 - y*f), f = w^T x + b
  - Hinge Loss 子梯度: dL/dw = -y*x if y*f < 1, else 0
  - L2 正则化梯度: d(λ||w||^2)/dw = 2λ*w
  - RBF 核: K(x,y) = exp(-gamma * ||x-y||^2)

运行方式：
  python exercise.py
===============================================================================
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler


# ============================================================================
# 任务 1: 实现 Hinge Loss 及其子梯度 (约 8 行)
# ============================================================================

def hinge_loss_and_gradient(w, b, x_i, y_i, lambda_):
    """
    计算单个样本的 Hinge Loss 和参数梯度。

    损失: L = max(0, 1 - y_i*(w^T x_i + b)) + lambda * ||w||^2
    梯度:
      - 若 y_i*(w^T x_i + b) < 1: dw = 2*lambda*w - y_i*x_i, db = -y_i
      - 若 y_i*(w^T x_i + b) >= 1: dw = 2*lambda*w, db = 0

    参数:
        w: (d,), 权重向量
        b: float, 偏置
        x_i: (d,), 单个样本特征
        y_i: int, 标签 (-1 或 +1)
        lambda_: float, L2 正则化系数
    返回:
        loss: float, 损失值
        dw: (d,), w 的梯度
        db: float, b 的梯度
    """
    # TODO: 完成以下步骤
    # 1. 计算 margin = y_i * (w @ x_i + b)
    # 2. 计算 Hinge Loss: max(0, 1 - margin)
    # 3. 计算 L2 正则化损失: lambda * ||w||^2
    # 4. 总损失 = Hinge Loss + L2 损失
    # 5. L2 正则化梯度: dw = 2*lambda*w
    # 6. 如果 margin < 1: dw -= y_i*x_i, db = -y_i
    #    否则: db = 0
    # 7. 返回 loss, dw, db
    # --- BEGIN YOUR CODE ---
    margin = y_i * (np.dot(w, x_i) + b)
    hinge_loss = max(0.0, 1.0 - margin)
    reg_loss = lambda_ * np.sum(w ** 2)
    loss = hinge_loss + reg_loss

    dw = 2.0 * lambda_ * w
    db = 0.0
    if margin < 1:
        dw -= y_i * x_i
        db = -y_i
    # --- END YOUR CODE ---
    return loss, dw, db


# ============================================================================
# 任务 2: 实现 SGD 单步更新 (约 10 行)
# ============================================================================

def sgd_step(X, y, w, b, C, lr):
    """
    对整个训练集执行一轮 SGD。

    参数:
        X: (n, d), 特征矩阵
        y: (n,), 标签 (0/1, 内部转为 -1/+1)
        w: (d,), 权重
        b: float, 偏置
        C: float, SVM 惩罚参数 (lambda = 1/(2*C))
        lr: float, 学习率
    返回:
        w, b: 更新后的参数
        avg_loss: 本轮的均损失
    """
    n_samples = X.shape[0]
    y_svm = np.where(y <= 0, -1, 1)
    lambda_ = 1.0 / (2.0 * C) if C > 1e-10 else 0.0
    total_loss = 0.0

    # 随机打乱
    indices = np.random.permutation(n_samples)

    # TODO: 遍历每个样本
    # 1. for each i in indices:
    #    a. 调用 hinge_loss_and_gradient 获取 loss, dw, db
    #    b. w -= lr * dw
    #    c. b -= lr * db
    #    d. 累加 loss
    # 2. 返回 w, b, total_loss / n_samples
    # --- BEGIN YOUR CODE ---
    for i in indices:
        x_i = X[i]
        y_i = y_svm[i]
        loss, dw, db = hinge_loss_and_gradient(w, b, x_i, y_i, lambda_)
        w -= lr * dw
        b -= lr * db
        total_loss += loss
    # --- END YOUR CODE ---

    return w, b, total_loss / n_samples


# ============================================================================
# 任务 3 (Bonus): 实现 RBF 核矩阵 (约 6 行)
# ============================================================================

def rbf_kernel(X, Y, gamma):
    """
    计算 RBF (高斯) 核矩阵。

    K(x, y) = exp(-gamma * ||x - y||^2)

    展开: ||x-y||^2 = ||x||^2 + ||y||^2 - 2*x@y.T

    参数:
        X: (m, d)
        Y: (n, d)
        gamma: float
    返回:
        K: (m, n), 核矩阵
    """
    # TODO:
    # 1. sq_X = sum(X**2, axis=1, keepdims=True) → (m, 1)
    # 2. sq_Y = sum(Y**2, axis=1) → (n,)
    # 3. sq_dists = sq_X + sq_Y - 2*X @ Y.T
    # 4. sq_dists = max(sq_dists, 0)  # 数值稳定性
    # 5. 返回 exp(-gamma * sq_dists)
    # --- BEGIN YOUR CODE ---
    sq_X = np.sum(X ** 2, axis=1, keepdims=True)  # (m, 1)
    sq_Y = np.sum(Y ** 2, axis=1)                  # (n,)
    sq_dists = sq_X + sq_Y - 2.0 * X @ Y.T
    sq_dists = np.maximum(sq_dists, 0.0)
    K = np.exp(-gamma * sq_dists)
    # --- END YOUR CODE ---
    return K


# ============================================================================
# 验证代码
# ============================================================================

def test_hinge_loss():
    """验证 Hinge Loss 计算。"""
    print("[测试 1] Hinge Loss 与梯度...")
    w = np.array([1.0, 0.0])
    b = 0.0
    x_i = np.array([0.5, 1.0])
    y_i = 1

    lambda_ = 0.0
    loss, dw, db = hinge_loss_and_gradient(w, b, x_i, y_i, lambda_)

    # margin = 1 * (1.0*0.5 + 0*1.0 + 0) = 0.5
    # hinge loss = max(0, 1-0.5) = 0.5
    assert np.abs(loss - 0.5) < 1e-6, f"loss error: {loss}"
    assert np.allclose(dw, -y_i * x_i), f"dw error: {dw}"
    assert db == -1.0, f"db error: {db}"
    print(f"  [PASS] Hinge Loss = {loss:.4f}, dw = {dw}, db = {db}")


def test_sgd_step():
    """验证 SGD 步骤能降低损失。"""
    print("[测试 2] SGD 单步更新...")
    np.random.seed(42)
    X, y = make_classification(n_samples=100, n_features=3,
                                n_informative=2, random_state=42)
    X = StandardScaler().fit_transform(X)

    w = np.zeros(3)
    b = 0.0

    # 初始损失
    y_svm = np.where(y <= 0, -1, 1)
    l = 1.0 / 2.0
    init_loss = 0.0
    for i in range(len(X)):
        margin = y_svm[i] * (np.dot(w, X[i]) + b)
        init_loss += max(0.0, 1.0 - margin)
    init_loss = init_loss / len(X) + l * np.sum(w**2)

    # 执行一轮 SGD
    w, b, avg_loss = sgd_step(X, y, w, b, C=1.0, lr=0.1)

    assert avg_loss <= init_loss, f"Loss did not decrease: {avg_loss:.4f} vs {init_loss:.4f}"
    print(f"  [PASS] Loss: {init_loss:.4f} -> {avg_loss:.4f}")


def test_rbf_kernel():
    """验证 RBF 核矩阵。"""
    print("[测试 3 (Bonus)] RBF 核矩阵...")
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    Y = np.array([[0.0, 0.0], [2.0, 2.0]])

    K = rbf_kernel(X, Y, gamma=1.0)

    # K[0,0] = exp(-0) = 1
    assert np.abs(K[0, 0] - 1.0) < 1e-6, f"K[0,0] error: {K[0,0]}"
    # K[0,1] = exp(-(0^2+0^2+4+4-0)) = exp(-8) ≈ 0.000335
    assert np.abs(K[0, 1] - np.exp(-8)) < 1e-6, f"K[0,1] error: {K[0,1]}"
    print(f"  [PASS] RBF 核矩阵正确! K[0,0]={K[0,0]:.4f}, K[0,1]={K[0,1]:.6f}")


if __name__ == "__main__":
    print("=" * 60)
    print("ml04_svm exercise.py — SVM 练习")
    print("=" * 60)

    try:
        test_hinge_loss()
        test_sgd_step()
        test_rbf_kernel()
        print("\n" + "=" * 60)
        print("所有测试通过!")
        print("=" * 60)
    except Exception as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
