---
title: "s10 CNN核心原理 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# s10 CNN核心原理 — demo.py 代码详解

<a href="../code/s10_cnn_fundamentals/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd s10_cnn_fundamentals/code
python demo.py
```

## 代码逐段详解

### 第1步：导入库 — 每个库是做什么的

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import os
```

- **`numpy`**：所有张量操作的底层引擎。使用 `np.lib.stride_tricks.as_strided`（内存视图技巧）实现高效的 im2col 和池化，使用 `np.pad` 实现零填充。
- **`matplotlib`**：可视化卷积核（热力图）、逐层特征图、输入图像。
- **`typing`**：`Tuple` 标注多返回值形状，`Optional` 标注可选参数。

---

### 第2步：Im2Col — 将卷积转化为矩阵乘法

这是整个 demo 中最重要的底层工具。Im2Col 将卷积操作转化为**矩阵乘法（GEMM）**，从而利用现代硬件（GPU/CPU 的 BLAS 库）的极致并行能力。

```python
class Im2Col:
    @staticmethod
    def im2col(x, kernel_h, kernel_w, stride=1, pad=0):
        N, C, H, W = x.shape

        # 1. 计算输出尺寸
        H_out = (H + 2 * pad - kernel_h) // stride + 1
        W_out = (W + 2 * pad - kernel_w) // stride + 1

        # 2. 零填充
        if pad > 0:
            x_padded = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)),
                             mode='constant', constant_values=0)
        else:
            x_padded = x

        # 3. as_strided 高效提取所有 patches
        shape = (N, C, H_out, W_out, kernel_h, kernel_w)
        strides = (
            x_padded.strides[0],            # N 维度
            x_padded.strides[1],            # C 维度
            x_padded.strides[2] * stride,   # H 维度跳 stride 行
            x_padded.strides[3] * stride,   # W 维度跳 stride 列
            x_padded.strides[2],            # 卷积核内 H（不跳）
            x_padded.strides[3],            # 卷积核内 W（不跳）
        )

        patches = np.lib.stride_tricks.as_strided(
            x_padded, shape=shape, strides=strides
        )
        # patches: (N, C, H_out, W_out, k_h, k_w)

        # 4. 转置 + reshape → im2col 格式
        cols = patches.transpose(0, 1, 4, 5, 2, 3).reshape(
            N, C * kernel_h * kernel_w, H_out * W_out
        )
        return cols
```

**Im2Col 的核心思想**：

对于输入 $(N, C, H, W)$ 和卷积核 $(C_{out}, C_{in}, k, k)$，原始卷积是"滑动窗口"操作。Im2Col 把这个过程等效为：

1. 提取所有 $H_{out} \times W_{out}$ 个"卷积窗口"（patches），每个大小 $C_{in} \times k \times k$
2. 将这些 patches 排列成矩阵 $\tilde{X}$，形状 $(N, C_{in} \cdot k \cdot k, H_{out} \cdot W_{out})$
3. 将卷积核展开成 $\tilde{K}$，形状 $(C_{out}, C_{in} \cdot k \cdot k)$
4. 做矩阵乘法：$\tilde{Y} = \tilde{K} \cdot \tilde{X}$，再 reshape 为输出 $(N, C_{out}, H_{out}, W_{out})$

**`as_strided` 为什么重要？** 如果写 for 循环提取每个 patch，Python 的开销（每次循环的 GIL 和解释器开销）会使代码慢 100-1000 倍。`as_strided` 通过修改 NumPy 数组的**步长（strides）**来创建"视图"——不复制任何数据，只是用不同的视角看同一块内存。例如，`strides[2] * stride` 意味着"沿 H 维度每次跳 `stride` 个内存单元"，从而模拟了步长大于 1 的效果。

**col2im 的作用**：反向传播时需要把梯度的形状从列矩阵还原为图像形状——因为每个像素可能被多个 patch 重叠覆盖（stride < kernel_size 时），所以采用**累加模式**。

---

### 第3步：Conv2d — 完整卷积层实现

```python
class Conv2d:
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True):
        fan_in = in_channels * kernel_size * kernel_size
        scale = np.sqrt(2.0 / fan_in)
        # 权重形状: (C_out, C_in, k, k)
        self.W = np.random.randn(out_channels, in_channels,
                                  kernel_size, kernel_size) * scale
        self.b = np.zeros(out_channels) if bias else None

    def forward(self, x):
        # 1. Im2Col: 将输入展开为列矩阵
        x_cols = Im2Col.im2col(x, self.kernel_size, self.kernel_size,
                                self.stride, self.padding)
        # x_cols: (N, C_in*k*k, H_out*W_out)

        # 2. 权重展开为 (C_out, C_in*k*k)
        W_col = self.W.reshape(self.out_channels, -1)

        # 3. 矩阵乘法 (C_out, C_in*k*k) @ (N, C_in*k*k, H_out*W_out)
        # NumPy 自动广播 N 维度
        out_cols = W_col @ x_cols  # (N, C_out, H_out*W_out)

        # 4. Reshape + 加偏置
        out = out_cols.reshape(N, self.out_channels, H_out, W_out)
        if self.use_bias:
            out += self.b.reshape(1, -1, 1, 1)  # 广播到 (N, C_out, H_out, W_out)

        return out
```

**参数量**：

$$
\text{Params}_{\text{Conv}} = C_{out} \times C_{in} \times k \times k + C_{out}
$$

例如 Conv1: `1 × 8 × 3 × 3 + 8 = 80` 个参数。对比同等输入大小的全连接层（需要 $28 \times 28 \times 8 = 6272$ 个参数），卷积的参数效率是 $78\times$。这就是**参数共享**的力量——同一个 $3 \times 3$ 卷积核在图像的每个位置重复使用。

---

### 第4步：MaxPool2d — 最大池化的高效实现

```python
class MaxPool2d:
    def forward(self, x):
        N, C, H, W = x.shape
        k, s = self.kernel_size, self.stride
        H_out = (H - k) // s + 1
        W_out = (W - k) // s + 1

        # as_strided 提取每个池化窗口
        shape = (N, C, H_out, W_out, k, k)
        strides = (x.strides[0], x.strides[1],
                   x.strides[2] * s, x.strides[3] * s,
                   x.strides[2], x.strides[3])
        patches = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

        # 在最后两维 (k, k) 上取最大值
        out = patches.max(axis=(4, 5))

        # 缓存 argmax（用于反向传播）
        patches_flat = patches.reshape(N, C, H_out, W_out, -1)
        self.cache["argmax"] = patches_flat.argmax(axis=4)
```

**池化的三个作用**：
1. **降维（下采样）**：$2 \times 2$ 池化 stride=2 将特征图从 $28 \times 28$ 降至 $14 \times 14$，减少 75% 的计算量
2. **平移不变性**：输入图像微小平移时，池化输出可能完全不变——因为只要最大值仍在池化窗口内，输出就相同
3. **增大感受野**：不需要更大的卷积核，深层神经元就能看到更大的输入区域

**为什么保存 argmax？** 反向传播时，池化层的梯度只传回最大值位置（最大值位置梯度 = 1，其他 = 0）。存储 `argmax`（展平窗口内的位置索引）让反向传播能精确地将梯度路由到正确位置。

---

### 第5步：SimpleCNN 模型架构

```python
class SimpleCNN:
    def __init__(self):
        self.conv1 = Conv2d(1, 8, kernel_size=3, stride=1, padding=1)  # → (8, 28, 28)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2d(2, 2)  # → (8, 14, 14)

        self.conv2 = Conv2d(8, 16, kernel_size=3, stride=1, padding=1)  # → (16, 14, 14)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2d(2, 2)  # → (16, 7, 7)

        self.fc = Linear(16 * 7 * 7, 10)  # → (10,)

    def forward(self, x):
        # Block 1
        x = self.conv1.forward(x)   # (N,1,28,28) → (N,8,28,28)
        x = self.relu1.forward(x)   # ReLU 激活，形状不变
        x = self.pool1.forward(x)   # (N,8,28,28) → (N,8,14,14)

        # Block 2
        x = self.conv2.forward(x)   # (N,8,14,14) → (N,16,14,14)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)   # (N,16,14,14) → (N,16,7,7)

        # 分类器
        x_flat = x.reshape(N, -1)    # Flatten: (N, 16*7*7)
        logits = self.fc.forward(x_flat)   # (N, 10)
        return softmax(logits)
```

**数据流动（以 MNIST $28 \times 28$ 灰度图为例）**：

| 层 | 输入形状 | 输出形状 | 操作 |
|----|---------|---------|------|
| Conv1 | (1, 28, 28) | (8, 28, 28) | 8个3×3卷积核 + padding=1（保持尺寸） |
| ReLU1 | (8, 28, 28) | (8, 28, 28) | 逐元素 max(0, x) |
| Pool1 | (8, 28, 28) | (8, 14, 14) | 2×2 最大池化 stride=2 |
| Conv2 | (8, 14, 14) | (16, 14, 14) | 16个3×3卷积核 + padding=1 |
| ReLU2 | (16, 14, 14) | (16, 14, 14) | — |
| Pool2 | (16, 14, 14) | (16, 7, 7) | 2×2 最大池化 stride=2 |
| Flatten | (16, 7, 7) | (784,) | 展平为一维 |
| FC | (784,) | (10,) | 全连接（仿射变换） |
| Softmax | (10,) | (10,) | 概率归一化 |

**为什么用 padding=1？** SAME padding（$P = \lfloor 3/2 \rfloor = 1$）让输出保持与输入相同的空间尺寸，避免每次卷积都缩小特征图。这允许堆叠更深的网络，因为特征图不会因为边界效应而快速缩为零。

---

### 第6步：感受野计算

```python
def compute_receptive_field(layers, verbose=True):
    rf = 1             # 初始感受野
    cum_stride = 1     # 累积步长

    for i, (k, s) in enumerate(layers):
        rf = rf + (k - 1) * cum_stride
        cum_stride *= s

    return rf
```

**感受野递推公式**：

$$
RF_l = RF_{l-1} + (k_l - 1) \times \prod_{j=1}^{l-1} s_j
$$

以 SimpleCNN 的层序列 `[(3,1), (2,2), (3,1), (2,2)]` 为例：

| 层 | k | s | 累积步长 | 感受野 $RF_l$ |
|----|---|---|---------|-------------|
| 输入 | — | — | 1 | 1 |
| Conv1 | 3 | 1 | 1 | $1 + (3-1) \times 1 = 3$ |
| Pool1 | 2 | 2 | $1 \times 2 = 2$ | $3 + (2-1) \times 1 = 4$ |
| Conv2 | 3 | 1 | 2 | $4 + (3-1) \times 2 = 8$ |
| Pool2 | 2 | 2 | $2 \times 2 = 4$ | $8 + (2-1) \times 2 = 10$ |

**结论**：Pool2 层的每个神经元"看到"原始输入图像上的 $10 \times 10$ 区域——占 $28 \times 28$ 图像的约 36%。这个感受野对 MNIST 数字识别足够大，因为 MNIST 中的数字通常不超过 $20 \times 20$ 像素。

---

### 第7步：参数量对比 — 卷积 vs 全连接

| 层 | 参数量 | 计算公式 |
|----|--------|---------|
| Conv1 | 80 | $8 \times 1 \times 3 \times 3 + 8$ |
| Conv2 | 1,168 | $16 \times 8 \times 3 \times 3 + 16$ |
| FC | 7,850 | $784 \times 10 + 10$ |
| **CNN 总计** | **8,320** | — |
| **等效 3 层 MLP** | **~112,000** | $784 \times 128 + 128 \times 64 + 64 \times 10$ |

**参数节省约 13 倍**，这得益于卷积的两个核心设计：
1. **局部连接**：每个神经元只连接输入的一个小区域，而非全部像素
2. **权值共享**：同一个卷积核在所有空间位置重复使用，而非每位置一套参数

---

### 第8步：可视化组件

#### 卷积核可视化

```python
def visualize_kernels(conv_layer, save_path):
    kernels = conv_layer.W  # (C_out, C_in, k, k)
    for ic in range(C_in):
        for oc in range(C_out):
            ax.imshow(kernel, cmap="RdBu_r", vmin=-abs(kernel).max(),
                     vmax=abs(kernel).max())
```

使用红蓝配色（`RdBu_r`）展示卷积核：红色代表正权重（倾向于激活），蓝色代表负权重（倾向于抑制）。训练后的卷积核呈现出有意义的模式——如 Gabor 滤波器般的边缘检测器、定向条状结构等。

#### 逐层特征图可视化

```python
def visualize_feature_maps(feature_maps, save_prefix, sample_idx=0):
    for layer_name, fm in feature_maps.items():
        for c in range(C):
            ax.imshow(fm[sample_idx, c], cmap="viridis")
```

每个通道显示为一个 $H \times W$ 的小图。浅层（Conv1）通常显示低层特征（边缘、纹理），深层（Conv2）显示更抽象的模式。某些通道可能"死掉"（全零）——这在 ReLU 网络中很常见，死掉的神经元永远输出 0，等效于该卷积核完全失效。

---

### 第9步：训练 — 简化的 SGD 反向传播

```python
# Softmax + 交叉熵的梯度
dlogits = probs.copy()
dlogits[np.arange(N), y_batch] -= 1  # δ = A - Y_onehot
dlogits /= N

# FC 层反向传播
dW_fc = x_flat.T @ dlogits
db_fc = dlogits.sum(axis=0)
dx_flat = dlogits @ model.fc.W.T

# Conv 层反向传播（简化版）
d_pool2 = dx_flat.reshape(N, 16, 7, 7)
d_relu2 = np.repeat(np.repeat(d_pool2, 2, axis=2), 2, axis=3)[:, :, :14, :14]
d_relu2 *= model.relu2.cache["mask"]
```

训练中使用了手工实现的 SGD 反向传播（未用 PyTorch 或 autograd）。关键步骤：
1. **Softmax + CE 梯度**：`dlogits[np.arange(N), y_batch] -= 1` 实现了 $\delta^{[L]} = A - Y$ 的组合梯度
2. **反池化**：`np.repeat(np.repeat(d_pool2, 2, axis=2), 2, axis=3)` 简单地将池化后的梯度"放大"回原尺寸——这是一种近似的反池化（真正的反池化需要 argmax 信息将梯度仅传回最大值位置）
3. **ReLU 反向**：`d_relu2 *= mask` 将梯度在负值位置截断为 0

> 注意：本 demo 的反向传播是**简化版**，主要用于教学展示。生产代码中应使用 Conv2d 和 MaxPool2d 的完整反向实现，或直接使用 PyTorch。

---

### 关键概念速查表

| 概念 | 数学/定义 | 代码实现 |
|------|----------|---------|
| 二维卷积 | $Y[i,j] = \sum_{m,n} X[i+m, j+n] \cdot K[m,n] + b$ | `W_col @ x_cols`（通过 im2col 转化为矩阵乘法） |
| Im2Col | 将滑动窗口展开为列矩阵 | `as_strided` + transpose + reshape |
| 权值共享 | 同一卷积核在所有空间位置复用 | 只有 $C_{in} \times k \times k$ 个参数，与输入尺寸无关 |
| 感受野递推 | $RF_l = RF_{l-1} + (k_l-1) \prod s_j$ | `rf + (k-1) * cum_stride` |
| 输出尺寸（无 dilation） | $H_{out} = \lfloor \frac{H+2P-k}{S} \rfloor + 1$ | `(H + 2*pad - k) // stride + 1` |
| 最大池化 | $\max(\text{window})$ | `patches.max(axis=(4,5))` |
| He 初始化 | $W \sim \mathcal{N}(0, \sqrt{2 / fan\_in})$ | `randn * sqrt(2.0 / fan_in)` |
| 参数量（卷积层） | $C_{out} \times C_{in} \times k^2 + C_{out}$ | `self.W.size + self.b.size` |

## 完整代码

<<< @/snippets/s10_cnn_fundamentals/demo.py
