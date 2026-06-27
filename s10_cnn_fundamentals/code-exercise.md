---
title: "s10 CNN核心原理 — exercise.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# s10 CNN核心原理 — exercise.py 练习指南

<a href="../code/s10_cnn_fundamentals/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过亲手实现 Im2Col 转换、最大池化前向传播、感受野递推计算和手动单通道卷积，深度理解卷积神经网络的核心底层机制——从"滑动窗口"到"矩阵乘法"，从"逐元素运算"到"感受野增长"。

## 预备知识

建议先阅读 index.md（熟悉卷积、池化、感受野的定义）并运行 demo.py（看完整效果），确保理解：

| 概念 | 核心思想 |
|------|---------|
| 卷积操作 | 卷积核在输入上滑动，每次逐元素乘加 |
| Im2Col | 将卷积窗口展开为矩阵列，转化为矩阵乘法 |
| 最大池化 | 在 $k \times k$ 窗口内取最大值 + 记录位置 |
| 感受野 | 深层神经元在原图上对应的区域大小 |
| 输出尺寸公式 | $H_{out} = \lfloor \frac{H + 2P - k}{S} \rfloor + 1$ |

---

## 任务清单

### 任务1：实现 Im2Col 转换（显式循环版）

**描述**：补全 `im2col()` 函数——用显式双重循环实现图像到列的转换。输入形状 $(N, C, H, W)$，输出形状 $(N, C \cdot k_h \cdot k_w, H_{out} \cdot W_{out})$。

**为什么要用显式循环？** demo.py 使用了 `as_strided`（内存视图技巧）实现高效 Im2Col，但对于初学者来说不够直观。显式循环版本让你看到 Im2Col 的**真正含义**：在输入的每个位置上提取一个 patch（小块），将其展平为一列。

**算法步骤**：

1. 计算输出尺寸：

   $$
   H_{out} = \left\lfloor \frac{H + 2P - k_h}{S} \right\rfloor + 1, \quad
   W_{out} = \left\lfloor \frac{W + 2P - k_w}{S} \right\rfloor + 1
   $$

2. 零填充：`x_padded = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)))`

3. 初始化 `cols` 数组，形状 $(N, C \cdot k_h \cdot k_w, H_{out} \cdot W_{out})$

4. 双重循环填充：对于每个输出位置 $(h, w)$：
   - 提取 `x_padded[n, :, h*s:h*s+k_h, w*s:w*s+k_w]` —— 一个形状为 $(C, k_h, k_w)$ 的 patch
   - 将该 patch 展平（`reshape(-1)`），填入 `cols` 的对应列

**提示**：
- `cols` 的列索引可以用 `col_idx = h * W_out + w`
- 提取的 patch 展平后长度是 `C * k_h * k_w`

---

### 任务2：实现最大池化的前向传播

**描述**：补全 `max_pool2d_forward()` 函数——用显式四重循环实现最大池化。输入形状 $(N, C, H, W)$，输出形状 $(N, C, H_{out}, W_{out})$。

**算法步骤**：

1. 计算输出尺寸：$H_{out} = (H - k) // s + 1$

2. 初始化 `out`（全零）和 `argmax`（全零，dtype=int），形状均为 $(N, C, H_{out}, W_{out})$

3. 四重循环：对每个 $(n, c, h, w)$：
   - 提取窗口 `window = x[n, c, h*s:h*s+k, w*s:w*s+k]`——形状 $(k, k)$
   - 最大值 `out[n, c, h, w] = window.max()`
   - 最大值在展平窗口内的位置 `argmax[n, c, h, w] = window.flatten().argmax()`（值为 $0 \sim k^2-1$）

**argmax 为什么重要？**

在反向传播时，池化层需要将上游梯度传回下游。由于池化是"取最大值"操作——只有最大值位置的局部梯度为 1，其他位置为 0。因此，`argmax` 记录了"梯度应该往哪个位置传"：

$$
\frac{\partial \text{out}}{\partial x[n,c,i,j]} = \begin{cases}
1 & \text{if } (i,j) \text{ 是窗口内的最大值位置} \\
0 & \text{otherwise}
\end{cases}
$$

**提示**：
- `window.max()` 取最大值，`.flatten().argmax()` 取最大值在展开后一维数组中的索引
- 索引值在 0 到 $k^2-1$ 之间（如 $k=2$ 时，`[0,0]` 索引为 0，`[0,1]` 为 1，`[1,0]` 为 2，`[1,1]` 为 3）

---

### 任务3：计算 CNN 架构的感受野

**描述**：补全 `compute_receptive_field()` 函数。给定一组层配置（每层一个 `(kernel_size, stride)` 元组），计算每层的感受野和最终感受野。

**感受野递推公式**：

$$
RF_l = RF_{l-1} + (k_l - 1) \times \prod_{j=1}^{l-1} s_j
$$

等价于代码中的：
```
rf = rf + (k - 1) * cum_stride
cum_stride *= s
```

**提示**：
- 初始感受野 $RF_0 = 1$（输入层每个像素"看到"自己）
- 初始累积步长 `cum_stride = 1`
- 对每个 `(k, s)`：先更新 `rf`，再更新 `cum_stride`（顺序不能反！）
- 记录每层后的 `rf` 到 `history` 列表中

**验证用例**：
- `[(3,1), (3,1)]`（两个 3×3 卷积）：感受野 $= 5$（因为 $1 + (3-1) \times 1 + (3-1) \times 1 = 5$）
- `[(3,1)] \times 5`（五个 3×3 卷积）：感受野 $= 11$（线性增长）
- `[(7,1), (2,2), (3,1)]`：感受野 $= 7 + 1 + 2 \times 2 = 12$（池化加速感受野扩张）

**为什么小卷积核堆叠优于大卷积核？** 两个 $3 \times 3$ 卷积（参数量 $2 \times 9 = 18$）的感受野 = $5 \times 5$，等价于一个 $5 \times 5$ 卷积（参数量 $25$）。但两个 $3 \times 3$ 卷积中间夹了一个 ReLU，比单个 $5 \times 5$ 卷积有更强的非线性表达能力。这就是 VGG 的设计哲学——"小而深"比"大而浅"更好。

---

### 任务4：手动实现单通道 2D 卷积

**描述**：补全 `conv2d_single()` 函数——最简单的卷积实现：输入和核都是 2D 矩阵，用双重循环完成卷积。这是理解卷积操作的最佳起点。

**算法步骤**：

1. 计算输出尺寸：$H_{out} = \lfloor (H + 2P - k) / S \rfloor + 1$

2. 零填充输入：`input_padded = np.pad(input_2d, pad, mode='constant')`

3. 初始化 `output` 为全零矩阵，形状 $(H_{out}, W_{out})$

4. 双重循环：对每个 $(i, j)$ 输出位置：
   - 提取 `input_padded` 中的 $k \times k$ 区域：`patch = input_padded[i*S : i*S+k, j*S : j*S+k]`
   - 逐元素乘加：`output[i, j] = np.sum(patch * kernel_2d)`

**测试用例**（垂直边缘检测）：

```
输入 5×5         核 3×3 (Sobel 垂直)       输出 3×3
[1 2 3 0 1]     [-1  0  1]                 [6  0 -3]
[4 5 6 1 2]     [-1  0  1]                 [9  0 -3]
[7 8 9 2 3]     [-1  0  1]                 [6 -3 -6]
[0 1 2 3 4]
[1 2 3 4 5]
```

垂直边缘检测核的特点是：左列全 -1，右列全 +1，中间列全 0。这意味着该核对"左暗右亮"的垂直边缘响应最强（左列乘 -1 抵消暗区，右列乘 +1 增强亮区），对均匀区域响应为 0。

---

### 关键概念速查

| 任务 | 核心操作 | 最容易错的地方 |
|------|---------|--------------|
| TODO 1: Im2Col | 提取每个 patch → 展平 → 放入列 | 忘记 padding；索引越界 |
| TODO 2: MaxPool | 取窗口最大值 + 记录索引 | argmax 是展平索引（0~$k^2-1$），不是二维坐标 |
| TODO 3: 感受野 | `rf += (k-1) * cum_stride` | 先更新 rf 再更新 cum_stride |
| TODO 4: 手动卷积 | 提取 patch，与核逐元素乘加 | padding 后的索引计算 |

## 完整代码

<<< @/snippets/s10_cnn_fundamentals/exercise.py
