# -*- coding: utf-8 -*-
"""
s16 Attention 与 Transformer — 练习题
==============================================
请补全以下 TODO 部分，完成后运行验证。
"""

import torch
import torch.nn.functional as F
import math
import numpy as np


# ============================================================
# 练习 1：实现缩放点积注意力
# ============================================================

def scaled_dot_product_attention(
    Q: torch.Tensor,  # (batch, seq_len, d_k)
    K: torch.Tensor,  # (batch, seq_len, d_k)
    V: torch.Tensor,  # (batch, seq_len, d_v)
    mask: torch.Tensor = None,  # (batch, seq_len, seq_len), True=需要mask的位置
) -> torch.Tensor:
    """
    TODO: 实现缩放点积注意力
    Attention(Q, K, V) = softmax(QK^T / √d_k) V

    参数：
        Q: Query 矩阵
        K: Key 矩阵
        V: Value 矩阵
        mask: 注意力掩码，True 的位置设为 -inf
    返回：
        output: 注意力输出 (batch, seq_len, d_v)
    """
    d_k = Q.size(-1)  # 获取 d_k
    # TODO: 实现以下步骤
    #   1. scores = Q @ K^T   (batch, seq_len, seq_len)
    #   2. scores = scores / sqrt(d_k)  ← 缩放，防止大 d_k 时 softmax 饱和
    #   3. 如果 mask 不为 None，scores[mask] = -inf
    #   4. attn_weights = softmax(scores, dim=-1)
    #   5. output = attn_weights @ V
    # ===== 你的代码在这里 =====
    output = torch.zeros_like(V)
    # ==========================
    return output


# 测试
batch, seq_len, d_k, d_v = 2, 5, 8, 4
Q_test = torch.randn(batch, seq_len, d_k)
K_test = torch.randn(batch, seq_len, d_k)
V_test = torch.randn(batch, seq_len, d_v)

try:
    result = scaled_dot_product_attention(Q_test, K_test, V_test)
    print(f"[练习1] 注意力输出形状: {result.shape} (期望: [{batch}, {seq_len}, {d_v}])")
    # 检查 softmax 每行和为 1
    scores = Q_test @ K_test.transpose(-2, -1) / math.sqrt(d_k)
    attn = F.softmax(scores, dim=-1)
    row_sums = attn.sum(dim=-1)
    print(f"        softmax 行求和: 最小值={row_sums.min():.6f}, 最大值={row_sums.max():.6f} (期望: 都≈1.0)")
except Exception as e:
    print(f"[练习1] 未完成实现: {e}")


# ============================================================
# 练习 2：实现因果掩码（Causal Mask）
# ============================================================

def create_causal_mask(seq_len: int) -> torch.Tensor:
    """
    TODO: 创建因果掩码矩阵
    返回一个 (seq_len, seq_len) 的布尔矩阵，
    其中 mask[i][j] = True 当 j > i（即第 j 个位置是第 i 个位置的"未来"）

    例如 seq_len=3 时：
    [[False, True,  True],
     [False, False, True],
     [False, False, False]]

    参数：
        seq_len: 序列长度
    返回：
        mask: (seq_len, seq_len) 的布尔张量，上三角为 True
    """
    # TODO: 使用 torch.triu 创建上三角矩阵
    #   提示: torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    # ===== 你的代码在这里 =====
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    # ==========================
    return mask


# 测试
seq_test = 4
expected = torch.tensor([
    [False, True,  True,  True],
    [False, False, True,  True],
    [False, False, False, True],
    [False, False, False, False],
])

try:
    mask_test = create_causal_mask(seq_test)
    match = torch.all(mask_test == expected)
    print(f"\n[练习2] 因果掩码正确性: {match} (期望: True)")
    if not match:
        print(f"        你的输出:\n{mask_test}")
        print(f"        期望输出:\n{expected}")
except Exception as e:
    print(f"[练习2] 未完成实现: {e}")


# ============================================================
# 练习 3：实现正弦位置编码
# ============================================================

def sinusoidal_position_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """
    TODO: 实现正弦位置编码

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    参数：
        max_len: 最大序列长度
        d_model: 模型维度
    返回：
        pe: (max_len, d_model) 的位置编码矩阵
    """
    # TODO: 实现
    #   1. 创建 position 张量 (max_len, 1)
    #   2. 计算 div_term = exp(arange(0, d_model, 2) * (-log(10000)/d_model))
    #   3. pe[:, 0::2] = sin(position * div_term)
    #   4. pe[:, 1::2] = cos(position * div_term)
    # ===== 你的代码在这里 =====
    pe = torch.zeros(max_len, d_model)
    # ==========================
    return pe


# 测试
max_len_test, d_model_test = 10, 16
try:
    pe = sinusoidal_position_encoding(max_len_test, d_model_test)
    print(f"\n[练习3] 位置编码形状: {pe.shape} (期望: [{max_len_test}, {d_model_test}])")
    # 检查每行的值是否在大约 [-1, 1] 范围内
    print(f"        值域: [{pe.min():.3f}, {pe.max():.3f}] (期望: [-1.0, 1.0])")
    # 检查相邻位置是否有差异（位置编码有意义）
    diff = (pe[1] - pe[0]).abs().mean()
    print(f"        相邻位置平均差异: {diff:.6f} (期望: > 0)")
except Exception as e:
    print(f"[练习3] 未完成实现: {e}")

print("\n所有练习测试完成！请对比 demo.py 查看参考实现。")
print("""
提示:
  - 缩放点积注意力的核心是 softmax(QK^T / √d_k) V
  - 因果掩码用上三角矩阵阻挡未来信息
  - 位置编码用不同频率的正弦波给序列注入位置信息
""")
