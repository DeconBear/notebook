# -*- coding: utf-8 -*-
"""
s10 CNN 核心原理 练习
=====================
完成以下 TODO 练习来加深对卷积操作的理解。
"""

import numpy as np
from typing import Tuple

# ============================================================
# 练习 1：实现 Im2Col 转换
# ============================================================

def im2col(x: np.ndarray, kernel_h: int, kernel_w: int,
           stride: int = 1, pad: int = 0) -> np.ndarray:
    """
    TODO: 实现 im2col 转换（不使用 as_strided，用显式循环）

    将输入张量的每个卷积窗口展开为列向量。

    输入: x 形状 (N, C, H, W)
    输出: cols 形状 (N, C*kernel_h*kernel_w, H_out*W_out)

    提示:
    1. 先计算输出尺寸 H_out = (H + 2*pad - k_h) // stride + 1
    2. 对输入做零填充（np.pad）
    3. 对于每个 batch、每个输出位置 (h_out, w_out)：
       - 提取大小为 (C, k_h, k_w) 的 patch
       - 将其展平放入 cols 的对应位置
    """
    N, C, H, W = x.shape

    # TODO: 计算输出尺寸
    H_out = None  # 替换为正确的公式
    W_out = None  # 替换为正确的公式

    # TODO: 对 x 做零填充（使用 np.pad，在 H 和 W 维度前后各 pad 个 0）
    x_padded = x  # 替换为 np.pad(...)

    # TODO: 初始化 cols 数组，形状 (N, C * kernel_h * kernel_w, H_out * W_out)
    cols = None

    # TODO: 双重循环填充 cols
    # for h in range(H_out):
    #     for w in range(W_out):
    #         提取 patch 并填充到 cols 的对应列

    return cols


# ============================================================
# 练习 2：实现最大池化的前向传播
# ============================================================

def max_pool2d_forward(x: np.ndarray, kernel_size: int = 2,
                       stride: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    TODO: 实现最大池化前向传播（用显式循环，不用 as_strided）

    参数:
        x: 输入特征图，形状 (N, C, H, W)
        kernel_size: 池化窗口大小
        stride: 步长

    返回:
        out: 池化后的特征图，形状 (N, C, H_out, W_out)
        argmax: 每个窗口中最大值的位置索引，形状 (N, C, H_out, W_out)
                索引是 0 ~ kernel_size*kernel_size-1 的整数

    提示:
    1. 对每个 batch、每个通道、每个输出位置：
       - 提取 kernel_size × kernel_size 的窗口
       - 找到最大值和它在窗口中的位置
    2. argmax 用于反向传播时将梯度传回最大值位置
    """
    N, C, H, W = x.shape
    k, s = kernel_size, stride

    # TODO: 计算输出尺寸
    H_out = None
    W_out = None

    # TODO: 初始化输出和 argmax
    out = None   # 形状 (N, C, H_out, W_out)
    argmax = None  # 形状 (N, C, H_out, W_out), dtype=int

    # TODO: 四重循环实现池化
    # for n in range(N):
    #     for c in range(C):
    #         for h in range(H_out):
    #             for w in range(W_out):
    #                 提取窗口 x[n, c, h*s:h*s+k, w*s:w*s+k]
    #                 取最大值和位置
    #                 写入 out 和 argmax

    return out, argmax


# ============================================================
# 练习 3：计算 CNN 架构的感受野
# ============================================================

def compute_receptive_field(layer_configs: list) -> Tuple[int, list]:
    """
    TODO: 计算给定 CNN 架构的逐层感受野

    感受野递推公式:
        RF_l = RF_{l-1} + (k_l - 1) * (s_1 * s_2 * ... * s_{l-1})
    即: RF_l = RF_{l-1} + (k_l - 1) * cum_stride

    参数:
        layer_configs: 列表，每个元素是 (kernel_size, stride) 的元组
                       例如 [(3,1), (3,1), (2,2)] 表示 Conv3→Conv3→Pool2

    返回:
        final_rf: 最终层在原图上的感受野（整数）
        history: 每层后的感受野列表 [RF_1, RF_2, ..., RF_L]

    示例:
        >>> rf, hist = compute_receptive_field([(3,1), (3,1)])
        >>> rf, hist
        (5, [3, 5])  # 两个 3×3 conv → 感受野 5×5
    """
    rf = 1                # 初始感受野（输入层每个像素"看到"自己）
    cum_stride = 1        # 累积步长：前面所有层步长的乘积
    history = []

    # TODO: 遍历 layer_configs，按递推公式计算
    # for k, s in layer_configs:
    #     rf = rf + (k - 1) * cum_stride
    #     cum_stride *= s
    #     history.append(rf)

    return rf, history


# ============================================================
# 练习 4：手动计算卷积输出（不使用任何库）
# ============================================================

def conv2d_single(input_2d: np.ndarray, kernel_2d: np.ndarray,
                  stride: int = 1, pad: int = 0) -> np.ndarray:
    """
    TODO: 实现单通道 2D 卷积（手动双重循环，输入和核都是 2D 矩阵）

    参数:
        input_2d: 输入 2D 矩阵，形状 (H, W)
        kernel_2d: 卷积核 2D 矩阵，形状 (k, k)
        stride: 步长
        pad: 零填充大小

    返回:
        output: 输出 2D 矩阵，形状 (H_out, W_out)

    这是最直观的卷积实现方式——显式双重循环，帮助理解卷积的"滑动窗口"本质。
    """
    H, W = input_2d.shape
    k = kernel_2d.shape[0]

    # TODO: 计算输出尺寸
    H_out = None
    W_out = None

    # TODO: 零填充（使用 np.pad，只需填充 input_2d）
    input_padded = None

    # TODO: 初始化输出
    output = None  # 形状 (H_out, W_out)

    # TODO: 双重循环完成卷积
    # 对于每个输出位置 (i, j)，提取 input_padded 中的对应区域，
    # 与 kernel 做逐元素乘法再求和，填入 output[i, j]

    return output


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("s10 CNN 核心原理 — 练习测试")
    print("=" * 50)

    # ---- 测试练习 3：感受野计算 ----
    print("\n[练习 3] 感受野计算测试：")

    # 测试架构 1: Conv3→Conv3→Conv3 (全 3×3 卷积)
    rf1, hist1 = compute_receptive_field([(3, 1), (3, 1), (3, 1)])
    expected_rf1 = 7
    print(f"  Conv3→Conv3→Conv3: RF = {rf1} (期望 {expected_rf1})")

    # 测试架构 2: Conv7→Pool2→Conv3 (经典架构的一小段)
    rf2, hist2 = compute_receptive_field([(7, 1), (2, 2), (3, 1)])
    expected_rf2 = 7 + (2-1)*1 + (3-1)*2
    print(f"  Conv7→Pool2→Conv3: RF = {rf2} (期望 {expected_rf2})")
    print(f"  历史: {hist2}")

    # 测试架构 3: 5 个 Conv3
    rf3, hist3 = compute_receptive_field([(3, 1)] * 5)
    expected_rf3 = 11
    print(f"  5×Conv3: RF = {rf3} (期望 {expected_rf3})")

    # ---- 测试练习 4：手动卷积 ----
    print("\n[练习 4] 手动卷积测试：")

    # 创建一个简单的测试用例
    test_input = np.array([
        [1, 2, 3, 0, 1],
        [4, 5, 6, 1, 2],
        [7, 8, 9, 2, 3],
        [0, 1, 2, 3, 4],
        [1, 2, 3, 4, 5]
    ], dtype=np.float32)

    test_kernel = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ], dtype=np.float32)  # 垂直边缘检测器

    # 期望输出（手动计算）
    expected_output = np.array([
        [6,  0, -3],
        [9,  0, -3],
        [6, -3, -6]
    ], dtype=np.float32)

    output = conv2d_single(test_input, test_kernel, stride=1, pad=0)
    if output is not None:
        print(f"  输入 {test_input.shape}，核 {test_kernel.shape} → 输出 {output.shape}")
        print(f"  输出:\n{output}")
        # 检查是否与期望值一致
        if np.allclose(output, expected_output):
            print("  ✓ 结果正确！")
        else:
            print(f"  期望输出:\n{expected_output}")
            print("  ✗ 结果与期望不符，请检查实现")

    print("\n" + "=" * 50)
    print("完成所有练习后，运行 demo.py 查看完整演示。")
    print("=" * 50)
