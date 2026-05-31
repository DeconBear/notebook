---
title: "s07 多层网络的矩阵反向传播 — exercise.py"
---

# s07 多层网络的矩阵反向传播 — exercise.py 练习指南

<a href="../code/s07_matrix_backprop/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过亲手实现单隐藏层的反向传播（$\delta$ 递推公式）、梯度裁剪和数值梯度检查，将矩阵形式的反向传播从"看懂公式"提升到"能手写实现"的水平。

## 预备知识

建议先阅读 index.md 并运行 demo.py，确保理解以下核心公式：

| 公式 | 含义 |
|------|------|
| $\delta^{[L]} = \nabla_A L \odot \phi'(Z^{[L]})$ | 输出层误差信号 |
| $\delta^{[l]} = (W^{[l+1]})^T \delta^{[l+1]} \odot \phi'(Z^{[l]})$ | 隐藏层误差递推 |
| $\frac{\partial L}{\partial W^{[l]}} = \delta^{[l]} (A^{[l-1]})^T$ | 权重梯度（外积） |
| $\frac{\partial L}{\partial b^{[l]}} = \sum_i \delta^{[l]}_i$ | 偏置梯度（求和） |

---

## 任务清单

### 任务1：实现单隐藏层的反向传播（δ 递推计算）

**描述**：补全 `single_hidden_backward()` 函数。网络结构为 输入 → 隐藏层(ReLU) → 输出层(Sigmoid)，损失函数使用 MSE。

**代码骨架**：

```python
def single_hidden_backward(W1, b1, W2, b2, X, Y):
    m = X.shape[1]

    # ---- 前向传播 ----
    Z1 = W1 @ X + b1                    # 隐藏层线性变换
    A1 = relu(Z1)                        # 隐藏层 ReLU 激活
    Z2 = W2 @ A1 + b2                   # 输出层线性变换
    A2 = sigmoid(Z2)                     # 输出层 Sigmoid 激活 → 预测值

    # ---- 反向传播 ----
    dA2 = (1.0 / m) * (A2 - Y)          # ∂L/∂A2 — MSE 损失的梯度
    dZ2 = dA2 * sigmoid_derivative(Z2)  # δ[2] = ∇_A L ⊙ φ'(Z2)

    dW2 = dZ2 @ A1.T                    # ∂L/∂W2 = δ[2] @ (A1)^T
    db2 = np.sum(dZ2, axis=1, keepdims=True)  # ∂L/∂b2

    dZ1 = (W2.T @ dZ2) * relu_derivative(Z1)  # δ[1] = W2^T @ δ[2] ⊙ ReLU'(Z1)
    dW1 = dZ1 @ X.T                     # ∂L/∂W1 = δ[1] @ X^T
    db1 = np.sum(dZ1, axis=1, keepdims=True)  # ∂L/∂b1
```

**关键提示**：

1. **前向顺序**：先 $Z_1 = W_1 X + b_1$，再 $A_1 = \text{ReLU}(Z_1)$，然后 $Z_2 = W_2 A_1 + b_2$，最后 $A_2 = \sigma(Z_2)$。不要搞反！
2. **输出层 δ**：MSE 损失对 $\sigma$ 输出的梯度是 $\frac{1}{m}(A_2 - Y)$，再乘以 $\sigma'(Z_2)$。
3. **δ₁ 递推**：这是整个练习的核心——把输出层的误差通过 $W_2^T$ 传回隐藏层，再经过 ReLU 导数门控。注意：$\text{ReLU}'(Z) = \mathbb{1}[Z > 0]$。
4. **权重梯度**：$dW_2 = \delta_2 \cdot A_1^T$，$dW_1 = \delta_1 \cdot X^T$。这是误差信号与输入的外积（矩阵乘法）。
5. **偏置梯度**：对 $\delta$ 按 axis=1 求和，并用 `keepdims=True` 保持形状。

**维度检查表**（以 2 输入 → 3 隐藏 → 1 输出，2 个样本为例）：

| 变量 | 形状 | 说明 |
|------|------|------|
| X | (2, 2) | 2特征 × 2样本 |
| W1 | (3, 2) | 3神经元 × 2输入 |
| b1 | (3, 1) | 广播到 (3, 2) |
| Z1, A1 | (3, 2) | 3神经元 × 2样本 |
| W2 | (1, 3) | 1神经元 × 3输入 |
| dZ2 | (1, 2) | 输出层 δ |
| dZ1 | (3, 2) | 隐藏层 δ（W2^T @ dZ2）|
| dW1 | (3, 2) | 必须与 W1 shape 一致 |

**期望输出**：所有梯度的形状与对应参数完全一致。

---

### 任务2：实现梯度裁剪

**描述**：补全 `clip_gradients()` 函数。当梯度的全局 L2 范数超过 `max_norm` 时，按比例缩小所有梯度。

**数学公式**：

全局 L2 范数：

$$
\|g\|_2 = \sqrt{\sum_i \|g_i\|_2^2}
$$

缩放因子：

$$
\text{scale} = \min\left(1, \frac{\text{max\_norm}}{\|g\|_2}\right)
$$

裁剪：

$$
\tilde{g}_i = \text{scale} \cdot g_i
$$

**提示**：
- 计算 `total_norm_sq`：遍历所有梯度矩阵，累加 `np.sum(grad ** 2)`
- 总范数：`total_norm = np.sqrt(total_norm_sq)`
- 如果 `total_norm > max_norm`，缩放因子 `scale = max_norm / total_norm`；否则 `scale = 1.0`
- 每个梯度 `grad * scale`

**核心思想**：梯度裁剪不改变梯度的**方向**，只限制**长度**——防止某一步的梯度过大导致参数跳到不稳定的区域。这在 RNN 和 Transformer 训练中是标配手段。

---

### 任务3：实现数值梯度检查

**描述**：补全 `numerical_gradient_check()` 函数。对每个参数的前 N 个元素（如 10 个），用双边有限差分验证解析梯度的正确性。

**数学公式**：

$$
\frac{\partial L}{\partial \theta} \approx \frac{L(\theta + \epsilon) - L(\theta - \epsilon)}{2\epsilon}
$$

**算法步骤**：

1. 对 `params` 中的每个参数：
   - 将其展平（`flatten()`）
   - 取前 `n_check` 个元素检查
2. 对每个元素：
   - 保存原始值
   - 构造 `θ + ε`：修改该元素 → 调用 `forward_fn` 计算 `loss_plus`
   - 构造 `θ - ε`：修改该元素 → 计算 `loss_minus`
   - 数值梯度 $= (\text{loss\_plus} - \text{loss\_minus}) / (2\epsilon)$
   - **恢复原始值**
   - 计算相对误差：`abs(grad_analytic - grad_numeric) / max(abs(grad_analytic) + abs(grad_numeric), 1e-10)`
   - 如果相对误差 > $10^{-5}$，标记失败

**为什么只检查前 10 个元素？** 梯度检查极其缓慢——每个参数元素需要 2 次额外前向传播。只随机抽查 10 个元素足以发现 bug，又能保证检查在合理时间内完成。

**参考判断标准**：
- 相对误差 $< 10^{-7}$：实现大概率正确
- 相对误差 $\approx 10^{-5}$：可能有小错误
- 相对误差 $> 10^{-3}$：几乎肯定有 bug

---

### 关键概念速查

| 任务 | 核心公式 | 最容易错的地方 |
|------|---------|--------------|
| TODO 1: δ 递推 | $\delta^{[1]} = (W^{[2]})^T \delta^{[2]} \odot \text{ReLU}'(Z^{[1]})$ | 忘记 ReLU 导数 `(Z1 > 0)` |
| TODO 1: 权重梯度 | $dW^{[l]} = \delta^{[l]} (A^{[l-1]})^T / m$ | 忘记除以 $m$ 或转置放错位置 |
| TODO 2: 梯度裁剪 | $\tilde{g} = g \cdot \min(1, \text{max\_norm} / \|g\|)$ | 忘记 $\min(1, x)$——小梯度不应该被放大 |
| TODO 3: 梯度检查 | $\frac{L(\theta+\epsilon) - L(\theta-\epsilon)}{2\epsilon}$ | 忘记恢复参数原始值 |

## 完整代码

<<< @/snippets/s07_matrix_backprop/exercise.py
