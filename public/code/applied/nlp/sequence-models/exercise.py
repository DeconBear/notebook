# -*- coding: utf-8 -*-
"""
s15 序列模型 — 练习题
==============================================
请补全以下 TODO 部分，完成后运行验证。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 练习 1：实现 RNN 前向传播（手动循环时间步）
# ============================================================

def rnn_forward(
    x: torch.Tensor,        # (batch, seq_len, input_size)
    W_ih: torch.Tensor,     # (hidden_size, input_size) — 输入→隐藏权重
    W_hh: torch.Tensor,     # (hidden_size, hidden_size) — 隐藏→隐藏权重
    b: torch.Tensor,        # (hidden_size,) — 偏置
    h0: torch.Tensor = None,  # (batch, hidden_size) — 初始隐藏状态
) -> torch.Tensor:
    """
    TODO: 实现 RNN 前向传播
    h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b)

    参数：
        x: 输入序列 (batch, seq_len, input_size)
        W_ih: 输入权重矩阵 (hidden_size, input_size)
        W_hh: 隐藏权重矩阵 (hidden_size, hidden_size)
        b: 偏置向量 (hidden_size,)
        h0: 初始隐藏状态 (batch, hidden_size)，默认全零
    返回：
        all_h: 所有时间步的隐藏状态 (batch, seq_len, hidden_size)
    """
    batch_size, seq_len, input_size = x.shape
    hidden_size = W_hh.shape[0]

    if h0 is None:
        h = torch.zeros(batch_size, hidden_size)
    else:
        h = h0

    all_h = []
    # TODO: 对每个时间步循环
    #   1. x_t = x[:, t, :] 取出第 t 步的输入
    #   2. h = tanh(x_t @ W_ih.T + h @ W_hh.T + b)
    #   3. 将 h 存入 all_h
    # ===== 你的代码在这里 =====
    pass
    # ==========================

    return torch.stack(all_h, dim=1) if all_h else torch.zeros(batch_size, seq_len, hidden_size)


# 测试 RNN 前向传播
batch, seq_len, input_size, hidden_size = 2, 5, 4, 3
x_test = torch.randn(batch, seq_len, input_size)
W_ih_test = torch.randn(hidden_size, input_size)
W_hh_test = torch.randn(hidden_size, hidden_size)
b_test = torch.randn(hidden_size)

try:
    result = rnn_forward(x_test, W_ih_test, W_hh_test, b_test)
    print(f"[练习1] RNN 前向传播输出形状: {result.shape} (期望: ({batch}, {seq_len}, {hidden_size}))")
except Exception as e:
    print(f"[练习1] 未完成实现: {e}")


# ============================================================
# 练习 2：实现 LSTM 遗忘门的计算
# ============================================================

def lstm_forget_gate(
    h_prev: torch.Tensor,  # (batch, hidden_size) — 上一时刻隐藏状态
    x_t: torch.Tensor,     # (batch, input_size) — 当前输入
    W_f: torch.Tensor,     # (hidden_size, hidden_size + input_size) — 遗忘门权重
    b_f: torch.Tensor,     # (hidden_size,) — 遗忘门偏置
) -> torch.Tensor:
    """
    TODO: 实现 LSTM 遗忘门
    f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
    其中 σ 是 sigmoid 函数，[h, x] 表示在 dim=1 上拼接

    参数：
        h_prev: 上一时刻隐藏状态 (batch, hidden_size)
        x_t: 当前输入 (batch, input_size)
        W_f: 遗忘门权重 (hidden_size, hidden_size + input_size)
        b_f: 遗忘门偏置 (hidden_size,)
    返回：
        f_t: 遗忘门输出，范围 [0, 1] (batch, hidden_size)
    """
    # TODO: 实现
    #   1. 在 dim=1 上拼接 h_prev 和 x_t
    #   2. 计算 W_f @ combined.T → 需要转置，或使用 nn.functional.linear
    #   3. 加上 b_f 并通过 sigmoid
    # ===== 你的代码在这里 =====
    return torch.tensor([])
    # ==========================


# 测试遗忘门
batch_test, hidden_test, input_test = 4, 8, 6
h_test = torch.randn(batch_test, hidden_test)
x_test = torch.randn(batch_test, input_test)
W_f_test = torch.randn(hidden_test, hidden_test + input_test)
b_f_test = torch.randn(hidden_test)

try:
    f_t = lstm_forget_gate(h_test, x_test, W_f_test, b_f_test)
    is_valid = (f_t.min() >= 0 and f_t.max() <= 1)
    print(f"练习2] 遗忘门输出形状: {f_t.shape}, 范围在[0,1]: {is_valid} (期望: True)")
except Exception as e:
    print(f"练习2] 未完成实现: {e}")


# ============================================================
# 练习 3：实现 GRU 更新门
# ============================================================

def gru_update_gate(
    h_prev: torch.Tensor,  # (batch, hidden_size)
    x_t: torch.Tensor,     # (batch, input_size)
    W_z: torch.Tensor,     # (hidden_size, hidden_size + input_size)
    b_z: torch.Tensor,     # (hidden_size,)
) -> torch.Tensor:
    """
    TODO: 实现 GRU 更新门
    z_t = σ(W_z · [h_{t-1}, x_t] + b_z)

    更新门控制保留多少旧状态和写入多少新状态:
    h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t

    参数：
        h_prev: 上一时刻隐藏状态
        x_t: 当前输入
        W_z: 更新门权重 (hidden_size, hidden_size + input_size)
        b_z: 更新门偏置 (hidden_size,)
    返回：
        z_t: 更新门输出，范围 [0, 1] (batch, hidden_size)
    """
    # TODO: 实现（与遗忘门类似）
    # ===== 你的代码在这里 =====
    return torch.tensor([])
    # ==========================


# 测试更新门
z_t_test = gru_update_gate(h_test, x_test, W_f_test, b_f_test)
try:
    is_valid = (z_t_test.min() >= 0 and z_t_test.max() <= 1)
    print(f"练习3] 更新门输出形状: {z_t_test.shape}, 范围在[0,1]: {is_valid} (期望: True)")
except Exception as e:
    print(f"练习3] 未完成实现: {e}")

print("\n所有练习测试完成！请对比 demo.py 查看参考实现。")
print("""
提示:
  - RNN 前向传播的核心是「循环时间步 + tanh」
  - LSTM 遗忘门用 sigmoid 输出 [0,1] 决定遗忘比例
  - GRU 更新门用 sigmoid 做历史与新信息的线性插值
""")
