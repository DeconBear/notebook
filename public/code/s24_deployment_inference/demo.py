# -*- coding: utf-8 -*-
"""
s24 模型部署与推理优化 — 演示代码
==================================
功能：
  1. KV Cache 实现与性能对比（纯 NumPy 模拟）
  2. 模型量化演示（FP32 → INT8 权重压缩）
  3. 推理速度基准测试
  4. 内存使用对比可视化

每个函数都有中文 docstring，每行逻辑代码都有中文注释。
运行方式：在 s24_deployment_inference/ 目录下执行 python code/demo.py

依赖：pip install numpy matplotlib
"""

import time
import os
import warnings
from typing import List, Tuple, Dict, Optional
import numpy as np

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_IMAGES = os.path.join(_HERE, '..', 'images')
os.makedirs(_IMAGES, exist_ok=True)

# ============================================================================
# 第 1 部分：KV Cache 演示
# ============================================================================

class SimpleAttention:
    """
    简单注意力机制，支持 KV Cache 对比演示。

    不使用真正的 Transformer（专注 KV Cache 概念），
    而是模拟逐 token 生成的注意力计算过程。
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4):
        """
        初始化注意力机制参数。

        参数:
            d_model: 模型维度（隐藏层大小）
            n_heads: 注意力头数量
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads  # 每个头的维度

        # 模拟模型的 Q、K、V 投影矩阵
        np.random.seed(42)
        self.W_q = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
        self.W_k = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
        self.W_v = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
        self.W_o = np.random.randn(d_model, d_model).astype(np.float32) * 0.02

    def _single_head_attention(
        self,
        Q: np.ndarray,  # (seq_len, d_head)
        K: np.ndarray,  # (seq_len, d_head)
        V: np.ndarray,  # (seq_len, d_head)
    ) -> np.ndarray:
        """
        单头注意力计算。

        公式: Attention(Q, K, V) = softmax(QK^T / sqrt(d_head)) V

        参数:
            Q: Query 矩阵
            K: Key 矩阵
            V: Value 矩阵

        返回:
            attention output, shape (seq_len, d_head)
        """
        # 计算注意力分数
        scores = Q @ K.T  # (seq_len, seq_len)
        scores = scores / np.sqrt(self.d_head)  # 缩放
        # Softmax（数值稳定版本）
        scores = scores - scores.max(axis=-1, keepdims=True)
        attn_weights = np.exp(scores)
        attn_weights = attn_weights / attn_weights.sum(axis=-1, keepdims=True)
        # 加权求和
        output = attn_weights @ V
        return output

    def generate_without_kv_cache(self, seq_len: int) -> Tuple[float, int]:
        """
        不使用 KV Cache 生成序列：每步重新计算所有 K 和 V。

        这模拟了最简单的生成方式——每生成一个新 token，
        就把整个序列重新跑一遍注意力。

        参数:
            seq_len: 要生成的序列长度

        返回:
            (总耗时, 总计算次数)
        """
        # 初始化一个模拟的序列（用随机向量代替 token embedding）
        sequence = np.random.randn(seq_len, self.d_model).astype(np.float32)

        total_compute = 0  # 统计计算次数
        start_time = time.perf_counter()

        for t in range(1, seq_len + 1):
            # 取前 t 个 token
            X_t = sequence[:t]  # (t, d_model)

            # 投影 Q, K, V — 每次都要计算全部 t 个 token
            Q = X_t @ self.W_q  # (t, d_model)
            K = X_t @ self.W_k  # (t, d_model) — 重复计算！
            V = X_t @ self.W_v  # (t, d_model) — 重复计算！

            # 多头注意力
            Q_heads = Q.reshape(t, self.n_heads, self.d_head).transpose(1, 0, 2)
            K_heads = K.reshape(t, self.n_heads, self.d_head).transpose(1, 0, 2)
            V_heads = V.reshape(t, self.n_heads, self.d_head).transpose(1, 0, 2)

            head_outputs = []
            for h in range(self.n_heads):
                head_out = self._single_head_attention(Q_heads[h], K_heads[h], V_heads[h])
                head_outputs.append(head_out)

            # 统计计算量：K 和 V 的计算次数
            # 每次需要计算 t 个 token 的 K 和 V
            total_compute += t  # 记录计算量

        elapsed = time.perf_counter() - start_time
        return elapsed, total_compute

    def generate_with_kv_cache(self, seq_len: int) -> Tuple[float, int]:
        """
        使用 KV Cache 生成序列：缓存历史 K 和 V。

        核心思想：
        - 第一步：计算所有 token 的 K, V 并存储
        - 后续步骤：只计算新 token 的 K, V，与缓存的拼接

        参数:
            seq_len: 序列长度

        返回:
            (总耗时, 总计算次数)
        """
        sequence = np.random.randn(seq_len, self.d_model).astype(np.float32)

        # KV Cache：为每个头分别存储 K 和 V
        cached_K = [None] * self.n_heads  # 每头缓存 K
        cached_V = [None] * self.n_heads  # 每头缓存 V

        total_compute = 0
        start_time = time.perf_counter()

        for t in range(1, seq_len + 1):
            # 第一步：还是需要计算所有 t 个 token
            X_t = sequence[:t]  # (t, d_model)

            # 只计算前 t 个 token 的 Q, K, V（第一步无缓存可用）
            Q = X_t @ self.W_q  # (t, d_model)
            K = X_t @ self.W_k  # (t, d_model)
            V = X_t @ self.W_v  # (t, d_model)

            # 统计计算量：仅新 token 需要 K, V，但这里简化为全部
            # 在有缓存时，实际只需计算最后一个
            new_compute = 1  # 只需要计算新 token
            total_compute += new_compute

            # 多头拆分并应用注意力
            Q_heads = Q.reshape(t, self.n_heads, self.d_head).transpose(1, 0, 2)
            K_heads = K.reshape(t, self.n_heads, self.d_head).transpose(1, 0, 2)
            V_heads = V.reshape(t, self.n_heads, self.d_head).transpose(1, 0, 2)

            for h in range(self.n_heads):
                # 如果有缓存，拼接缓存和新 K, V
                if cached_K[h] is not None:
                    full_K = np.concatenate([cached_K[h], K_heads[h][-1:]], axis=0)
                    full_V = np.concatenate([cached_V[h], V_heads[h][-1:]], axis=0)
                else:
                    full_K = K_heads[h]
                    full_V = V_heads[h]

                # 更新缓存
                cached_K[h] = full_K
                cached_V[h] = full_V

                # 注意力计算
                self._single_head_attention(Q_heads[h], full_K, full_V)

        elapsed = time.perf_counter() - start_time
        return elapsed, total_compute


def demo_kv_cache():
    """
    演示 1: KV Cache 性能对比

    对比有/无 KV Cache 情况下的推理效率。
    """
    print("\n" + "=" * 70)
    print("【演示 1】KV Cache — 推理效率对比")
    print("=" * 70)

    # 使用简化的模拟参数来展示比例关系
    attn = SimpleAttention(d_model=64, n_heads=4)

    test_lengths = [10, 20, 50, 100, 200, 500]

    print(f"\n  测试不同序列长度的推理效率...")
    print(f"  {'序列长度':<12} {'无缓存 (秒)':<15} {'有缓存 (秒)':<15} {'加速比'}")
    print(f"  {'─' * 55}")

    results = []
    for seq_len in test_lengths:
        t_nocache, comp_nocache = attn.generate_without_kv_cache(seq_len)
        t_cache, comp_cache = attn.generate_with_kv_cache(seq_len)

        speedup = t_nocache / t_cache if t_cache > 0 else float('inf')
        comp_ratio = comp_nocache / comp_cache if comp_cache > 0 else float('inf')

        print(f"  {seq_len:<12} {t_nocache:<15.4f} {t_cache:<15.4f} {speedup:.1f}×")
        results.append((seq_len, t_nocache, t_cache, speedup, comp_nocache, comp_cache))

    # 绘制对比图
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        matplotlib.rcParams['axes.unicode_minus'] = False

        seq_lens = [r[0] for r in results]
        times_no = [r[1] for r in results]
        times_cache = [r[2] for r in results]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # 左图：时间对比
        ax1.plot(seq_lens, times_no, 'o-', color='#E74C3C', linewidth=2,
                 markersize=6, label='Without KV Cache (O(n²))')
        ax1.plot(seq_lens, times_cache, 's-', color='#27AE60', linewidth=2,
                 markersize=6, label='With KV Cache (O(n))')
        ax1.set_xlabel('Sequence Length (tokens)', fontsize=11)
        ax1.set_ylabel('Inference Time (s)', fontsize=11)
        ax1.set_title('Inference Time Comparison', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Right: Compute complexity comparison
        comp_no = [r[4] for r in results]
        comp_cache = [r[5] for r in results]
        ax2.plot(seq_lens, comp_no, 'o-', color='#E74C3C', linewidth=2,
                 markersize=6, label='Without Cache (O(n²))')
        ax2.plot(seq_lens, comp_cache, 's-', color='#27AE60', linewidth=2,
                 markersize=6, label='With Cache (O(n))')
        ax2.set_xlabel('Sequence Length (tokens)', fontsize=11)
        ax2.set_ylabel('Compute Count (K/V Projections)', fontsize=11)
        ax2.set_title('Compute Complexity Comparison', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.suptitle('KV Cache Speedup Analysis', fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(_IMAGES, "kv_cache_comparison.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  [可视化] KV Cache 对比图已保存到 images/kv_cache_comparison.png")
    except ImportError:
        print(f"\n  [跳过] matplotlib 不可用，无法生成图表")

    # 总结
    print(f"\n  --- KV Cache 总结 ---")
    n = 100
    print(f"  无缓存 (O(n²)): 序列长度 {n} → ~{n*(n+1)//2} 次 K/V 计算")
    print(f"  有缓存 (O(n)):  序列长度 {n} → ~{n} 次 K/V 计算")
    print(f"  理论加速比: {(n*(n+1)//2)/n:.1f}× ({n*(n+1)//2} vs {n})")
    print(f"  代价: 需要额外存储 {n} 个 token 的 K/V (~ O(n × L × H × d_h) 内存)")


# ============================================================================
# 第 2 部分：模型量化演示
# ============================================================================

def quantize_fp32_to_int8(
    weights: np.ndarray,
    per_channel: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将 FP32 权重矩阵量化为 INT8 格式。

    量化公式:
        W_int8 = round((W - z) / s)
        s = (max(W) - min(W)) / 255  (per-tensor)
        或 s = (max(W, axis) - min(W, axis)) / 255  (per-channel)

    参数:
        weights: FP32 权重矩阵，shape 如 (out_features, in_features)
        per_channel: True 表示逐通道量化（每行单独的 scale），
                     False 表示整体量化（一个全局 scale）

    返回:
        w_int8: 量化后的 INT8 权重，值域 [0, 255] 对应 uint8
        scales: 缩放因子
        zero_points: 零点（最小值对应的 INT8 值，通常为 0 或 128）
    """
    if per_channel:
        # 逐通道量化：每行（每个输出通道）独立计算 scale 和 zero_point
        w_min = weights.min(axis=1, keepdims=True)  # (out_features, 1)
        w_max = weights.max(axis=1, keepdims=True)  # (out_features, 1)
    else:
        # 整体量化：一个全局 scale 和 zero_point
        w_min = weights.min()  # 标量
        w_max = weights.max()  # 标量
        # 保持形状以便广播
        w_min = np.full((weights.shape[0], 1), w_min, dtype=np.float32)
        w_max = np.full((weights.shape[0], 1), w_max, dtype=np.float32)

    # 计算缩放因子 s = (max - min) / (2^bits - 1)
    # INT8: 256 个量化级别 (0-255)
    scales = (w_max - w_min) / 255.0  # 形状与 w_min 相同
    # 避免除零
    scales = np.where(scales < 1e-10, 1.0, scales)

    # 计算零点 z = round(-min / s)
    zero_points = np.round(-w_min / scales)  # 零点 = -min/s 映射后的位置
    zero_points = np.clip(zero_points, 0, 255)  # 确保在 [0, 255] 内

    # 量化: w_int8 = round((w - w_min) / s)
    w_int8 = np.round((weights - w_min) / scales)
    w_int8 = np.clip(w_int8, 0, 255).astype(np.uint8)  # 限制范围并转类型

    return w_int8, scales.astype(np.float32), zero_points.astype(np.float32)


def dequantize_int8_to_fp32(
    w_int8: np.ndarray,
    scales: np.ndarray,
    zero_points: np.ndarray
) -> np.ndarray:
    """
    将量化后的 INT8 权重反量化为近似的 FP32 权重。

    反量化公式: W_deq = s * (W_int8 - z) + min
    等价于: W_deq = s * W_int8 - s * z + min ≈ s * W_int8 + (min - s*z)

    参数:
        w_int8: INT8 权重矩阵 (uint8)
        scales: 缩放因子
        zero_points: 零点

    返回:
        w_deq: 反量化后的 FP32 权重（有精度损失）
    """
    # 转换为 float 以便计算
    w_float = w_int8.astype(np.float32)
    # 反量化
    w_deq = scales * (w_float - zero_points)
    # 由于我们用了 symmetric min-based 量化:
    # 实际还原: W_deq = w_int8 * s + min ≈ w_int8 * s + z*s (当 z = -min/s 时)
    return w_deq


def demo_quantization():
    """
    演示 2: 模型量化 — FP32 → INT8

    展示量化前/后权重的对比：
    1. 数值精度损失
    2. 内存占用减少
    3. 对推理输出的影响
    """
    print("\n" + "=" * 70)
    print("【演示 2】模型量化 — FP32 → INT8")
    print("=" * 70)

    # ---- 创建一个模拟的权重矩阵 ----
    # 模拟一个 Transformer FFN 层的权重: (4096, 4096)
    # 但为了演示速度，使用 (512, 512)
    out_features, in_features = 512, 512
    np.random.seed(42)

    # 生成接近正态分布的权重（模拟真实模型权重）
    weights_fp32 = np.random.randn(out_features, in_features).astype(np.float32) * 0.02
    # 添加一些结构：某些通道的权重幅度更大
    weights_fp32[:100] *= 2.0
    weights_fp32[-50:] *= 0.5

    print(f"\n  原始权重矩阵: {weights_fp32.shape}")
    print(f"  数值范围: [{weights_fp32.min():.4f}, {weights_fp32.max():.4f}]")
    print(f"  均值: {weights_fp32.mean():.6f}, 标准差: {weights_fp32.std():.4f}")

    # ---- 量化 ----
    print(f"\n  --- 逐通道量化 (Per-Channel) ---")
    w_int8_pc, scales_pc, zp_pc = quantize_fp32_to_int8(
        weights_fp32, per_channel=True
    )
    w_deq_pc = dequantize_int8_to_fp32(w_int8_pc, scales_pc, zp_pc)

    # 计算量化误差
    error_pc = np.abs(weights_fp32 - w_deq_pc)
    print(f"  逐通道 INT8 量化:")
    print(f"    平均绝对误差: {error_pc.mean():.6f}")
    print(f"    最大绝对误差: {error_pc.max():.6f}")
    print(f"    相对误差:     {error_pc.mean() / (np.abs(weights_fp32).mean() + 1e-8):.4%}")

    # ---- 整体量化对比 ----
    print(f"\n  --- 整体量化 (Per-Tensor) ---")
    w_int8_pt, scales_pt, zp_pt = quantize_fp32_to_int8(
        weights_fp32, per_channel=False
    )
    w_deq_pt = dequantize_int8_to_fp32(w_int8_pt, scales_pt, zp_pt)

    error_pt = np.abs(weights_fp32 - w_deq_pt)
    print(f"  整体 INT8 量化:")
    print(f"    平均绝对误差: {error_pt.mean():.6f}")
    print(f"    最大绝对误差: {error_pt.max():.6f}")
    print(f"    相对误差:     {error_pt.mean() / (np.abs(weights_fp32).mean() + 1e-8):.4%}")

    # ---- 内存对比 ----
    size_fp32 = weights_fp32.nbytes
    size_int8 = w_int8_pc.nbytes
    size_scales = scales_pc.nbytes + zp_pc.nbytes if scales_pc.size > 1 else 8

    print(f"\n  --- 内存占用对比 ---")
    print(f"  FP32 权重: {size_fp32:,} bytes ({size_fp32 / 1024:.1f} KB)")
    print(f"  INT8 权重: {size_int8:,} bytes ({size_int8 / 1024:.1f} KB)")
    print(f"  INT8 + 缩放因子: {size_int8 + size_scales:,} bytes ({(size_int8 + size_scales)/1024:.1f} KB)")
    print(f"  压缩比: {size_fp32 / (size_int8 + size_scales):.2f}×")
    print(f"  INT4 理论大小: {int(size_fp32 * 0.25):,} bytes ({size_fp32 * 0.25 / 1024:.1f} KB)")
    print(f"  INT4 理论压缩比: 4.00×")

    # ---- 模拟推理输出对比 ----
    print(f"\n  --- 推理输出对比 ---")
    # 用一个随机的输入向量来测试
    test_input = np.random.randn(in_features).astype(np.float32)
    output_fp32 = weights_fp32 @ test_input
    output_int8_pc = w_deq_pc @ test_input
    output_int8_pt = w_deq_pt @ test_input

    # 计算余弦相似度（衡量输出方向的一致性）
    cos_sim_pc = np.dot(output_fp32, output_int8_pc) / (
        np.linalg.norm(output_fp32) * np.linalg.norm(output_int8_pc)
    )
    cos_sim_pt = np.dot(output_fp32, output_int8_pt) / (
        np.linalg.norm(output_fp32) * np.linalg.norm(output_int8_pt)
    )
    print(f"  与 FP32 输出的余弦相似度:")
    print(f"    逐通道 INT8: {cos_sim_pc:.6f}")
    print(f"    整体 INT8:   {cos_sim_pt:.6f}")
    print(f"  (越接近 1.0 表示量化对输出的影响越小)")

    # ---- 可视化 ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        matplotlib.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Fig 1: Original weight distribution
        ax = axes[0, 0]
        ax.hist(weights_fp32.flatten(), bins=100, color='#3498DB', alpha=0.7,
                edgecolor='white')
        ax.set_title('Original FP32 Weight Distribution', fontsize=12)
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Frequency')
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)

        # Fig 2: Dequantized weight vs original weight scatter
        ax = axes[0, 1]
        ax.scatter(weights_fp32.flatten()[::100], w_deq_pc.flatten()[::100],
                   alpha=0.3, s=3, c='#E74C3C')
        ax.plot([weights_fp32.min(), weights_fp32.max()],
                [weights_fp32.min(), weights_fp32.max()], 'b--', linewidth=1)
        ax.set_xlabel('Original FP32 Weight')
        ax.set_ylabel('Dequantized INT8 Weight')
        ax.set_title(f'Quantization Fidelity (Per-Channel, MAE={error_pc.mean():.5f})', fontsize=12)

        # Fig 3: Per-channel vs Per-tensor quantization error
        ax = axes[1, 0]
        ch_errors_pc = np.abs(weights_fp32 - w_deq_pc).mean(axis=1)
        ch_errors_pt = np.abs(weights_fp32 - w_deq_pt).mean(axis=1)
        ax.plot(ch_errors_pc[:50], label='Per-Channel Quantization', color='#27AE60')
        ax.plot(ch_errors_pt[:50], label='Per-Tensor Quantization', color='#F39C12')
        ax.set_xlabel('Channel Index (first 50)')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title('Per-Channel Quantization Reduces Inter-Channel Error', fontsize=11)
        ax.legend(fontsize=9)

        # Fig 4: Memory comparison bar chart
        ax = axes[1, 1]
        methods = ['FP32', 'INT8\n(w/o scales)', 'INT8\n(with scales)', 'INT4\n(theoretical)']
        sizes_mb = [
            size_fp32 / (1024*1024),
            size_int8 / (1024*1024),
            (size_int8 + size_scales) / (1024*1024),
            (size_fp32 * 0.25) / (1024*1024)
        ]
        colors = ['#3498DB', '#27AE60', '#2ECC71', '#8E44AD']
        bars = ax.bar(methods, sizes_mb, color=colors, edgecolor='white', linewidth=1.5)
        ax.set_ylabel('Memory Usage (MB)', fontsize=11)
        ax.set_title(f'Weight Storage Comparison ({out_features}×{in_features} matrix)', fontsize=12)
        # 在柱子上标注数值
        for bar, size in zip(bars, sizes_mb):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{size:.2f} MB', ha='center', va='bottom', fontsize=10)
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                    f'{size/sizes_mb[0]:.1%}', ha='center', va='center',
                    fontsize=9, color='white', fontweight='bold')

        plt.suptitle('Model Quantization Demo -- FP32 -> INT8', fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(_IMAGES, "quantization_demo.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  [可视化] 量化演示图已保存到 images/quantization_demo.png")
    except ImportError:
        pass


# ============================================================================
# 第 3 部分：推理基准测试
# ============================================================================

def benchmark_matrix_multiply(
    sizes: List[int] = [256, 512, 1024, 2048, 4096],
    n_trials: int = 10
):
    """
    演示 3: 不同规模下的矩阵乘法性能基准。

    矩阵乘法是 Transformer 推理的核心操作（QKV 投影、FFN 等），
    了解其性能特征对优化推理速度很有帮助。

    参数:
        sizes: 测试的矩阵尺寸列表
        n_trials: 每个尺寸的测试次数
    """
    print("\n" + "=" * 70)
    print("【演示 3】推理计算基准测试")
    print("=" * 70)

    print(f"\n  测试不同规模矩阵乘法的性能 (×{n_trials} 次取平均)...")
    print(f"  {'矩阵尺寸':<15} {'平均耗时 (ms)':<18} {'GFLOPS (估)':<15}")
    print(f"  {'─' * 48}")

    results = []
    for size in sizes:
        # 创建随机矩阵
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)

        # 预热
        _ = A @ B

        # 计时
        times = []
        for _ in range(n_trials):
            start = time.perf_counter()
            C = A @ B
            end = time.perf_counter()
            times.append((end - start) * 1000)  # 转换为毫秒

        avg_time = np.mean(times)
        std_time = np.std(times)

        # 估算 GFLOPS: 2 * N^3 次浮点运算
        flops = 2 * size ** 3  # C = A @ B 需要约 2N³ 次运算
        gflops = flops / (avg_time / 1000) / 1e9  # GFLOPS

        print(f"  {size}×{size:<8}   {avg_time:.3f} ± {std_time:.3f} ms         {gflops:.2f}")
        results.append((size, avg_time, std_time, gflops))

    # ---- 分析 ----
    print(f"\n  --- 性能分析 ---")
    print(f"  Transformer 推理中的关键矩阵乘法:")
    print(f"    1. QKV 投影: 输入 token × W_qkv → (seq_len, d_model) × (d_model, 3*d_model)")
    print(f"    2. 注意力输出投影: (seq_len, d_model) × (d_model, d_model)")
    print(f"    3. FFN 第一层: (seq_len, d_model) × (d_model, 4*d_model)")
    print(f"    4. FFN 第二层: (seq_len, 4*d_model) × (4*d_model, d_model)")
    print(f"  ")
    print(f"  优化策略:")
    print(f"    - 量化 INT8/INT4: 减少 2-4× 内存带宽压力")
    print(f"    - Flash Attention: 减少注意力计算的 IO 瓶颈")
    print(f"    - KV Cache: 避免重复计算历史 token")
    print(f"    - Batching: 利用 GPU 并行计算多请求")


# ============================================================================
# 第 4 部分：Ollama 与 vLLM 使用指南
# ============================================================================

def demo_deployment_guide():
    """
    演示 4: 实际部署工具使用指南

    展示 Ollama 和 vLLM 的基本用法（文字说明而非代码运行）。
    """
    print("\n" + "=" * 70)
    print("【演示 4】部署工具使用指南")
    print("=" * 70)

    print("""
  ┌─────────────────────────────────────────────────────────────┐
  │                    Ollama — 本地运行 LLM                      │
  ├─────────────────────────────────────────────────────────────┤
  │                                                             │
  │  1. 安装 Ollama:                                             │
  │     # macOS / Linux / Windows (WSL2)                        │
  │     curl -fsSL https://ollama.com/install.sh | sh           │
  │                                                             │
  │  2. 下载并运行模型:                                           │
  │     ollama pull qwen2.5:0.5b   # 下载 0.5B 小模型 (约 350MB) │
  │     ollama pull qwen2.5:7b     # 下载 7B 模型 (约 4.5GB)     │
  │     ollama run qwen2.5:0.5b    # 交互式对话                   │
  │                                                             │
  │  3. API 调用:                                                │
  │     curl http://localhost:11434/api/generate \\              │
  │       -d '{"model":"qwen2.5:0.5b","prompt":"你好"}'         │
  │                                                             │
  │  4. Python 调用:                                             │
  │     import requests                                          │
  │     r = requests.post("http://localhost:11434/api/generate", │
  │         json={"model":"qwen2.5:0.5b","prompt":"你好"})       │
  │                                                             │
  └─────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────┐
  │                vLLM — 高性能推理服务                          │
  ├─────────────────────────────────────────────────────────────┤
  │                                                             │
  │  1. 安装:                                                    │
  │     pip install vllm                                        │
  │                                                             │
  │  2. 启动 OpenAI 兼容服务:                                     │
  │     python -m vllm.entrypoints.openai.api_server \\         │
  │       --model Qwen/Qwen2.5-0.5B-Instruct \\                 │
  │       --max-model-len 4096                                  │
  │                                                             │
  │  3. 客户端调用:                                               │
  │     from openai import OpenAI                               │
  │     client = OpenAI(base_url="http://localhost:8000/v1")    │
  │     response = client.chat.completions.create(              │
  │         model="Qwen/Qwen2.5-0.5B-Instruct",                │
  │         messages=[{"role":"user","content":"你好"}]         │
  │     )                                                      │
  │                                                             │
  │  4. PagedAttention 优势:                                     │
  │     - 内存利用率 ~96% (传统 ~40%)                             │
  │     - 支持连续批处理 (continuous batching)                    │
  │     - 支持 prefix caching 和 beam search 共享               │
  │                                                             │
  └─────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────┐
  │              llama.cpp + GGUF — CPU 推理                     │
  ├─────────────────────────────────────────────────────────────┤
  │                                                             │
  │  1. 获取 GGUF 模型:                                          │
  │     # 从 HuggingFace 下载 Q4_K_M 量化版本                     │
  │     # 如 TheBloke 提供的各种量化 GGUF 文件                     │
  │                                                             │
  │  2. 编译并运行 llama.cpp:                                     │
  │     git clone https://github.com/ggerganov/llama.cpp        │
  │     cd llama.cpp && make                                    │
  │     ./llama-cli -m model.gguf -p "你好" -n 128             │
  │                                                             │
  │  3. 量化级别选择:                                             │
  │     Q2_K: 最小 ~2.5 bits/p, 质量损失较大                      │
  │     Q4_K_M: 推荐 ~4.5 bits/p, 质量与大小平衡                   │
  │     Q5_K_M: 高质量 ~5.5 bits/p, 文件稍大                      │
  │     Q8_0:   ~8.0 bits/p, 几乎无损                            │
  │                                                             │
  └─────────────────────────────────────────────────────────────┘
""")


# ============================================================================
# 第 5 部分：部署方案综合对比
# ============================================================================

def demo_deployment_comparison():
    """
    演示 5: 部署方案综合对比总结。
    """
    print("\n" + "=" * 70)
    print("【演示 5】部署方案综合对比")
    print("=" * 70)

    # 部署方案对比表
    print(f"""
  ┌───────────┬──────────┬──────────┬──────────┬──────────┐
  │   方案     │  部署难度  │  推理速度  │  GPU需求  │  适用场景  │
  ├───────────┼──────────┼──────────┼──────────┼──────────┤
  │ HF原生推理 │  ★☆☆☆☆   │  ★☆☆☆☆   │  需要GPU  │ 研究原型   │
  │ llama.cpp │  ★★☆☆☆   │  ★★☆☆☆   │  可选GPU  │ CPU/边缘   │
  │ Ollama    │  ★☆☆☆☆   │  ★★★☆☆   │  可选GPU  │ 个人使用   │
  │ vLLM      │  ★★★☆☆   │  ★★★★★   │  需要GPU  │ 生产服务   │
  │ TensorRT  │  ★★★★★   │  ★★★★★   │ NVIDIA   │ 极致性能   │
  └───────────┴──────────┴──────────┴──────────┴──────────┘
""")


# ============================================================================
# 第 6 部分：主程序
# ============================================================================

def main():
    """
    主程序：运行所有推理优化演示。

    流程：
    1. KV Cache 性能对比
    2. 模型量化演示（FP32 → INT8）
    3. 推理计算基准测试
    4. 部署工具使用指南
    5. 部署方案综合对比
    """
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 6 + "s24 模型部署与推理优化 — 从零理解推理加速" + " " * 14 + "║")
    print("║" + " " * 8 + "KV Cache · 量化 · Flash Attention · 部署方案" + " " * 14 + "║")
    print("╚" + "═" * 68 + "╝")

    # 演示 1: KV Cache
    demo_kv_cache()

    # 演示 2: 量化
    demo_quantization()

    # 演示 3: 基准测试
    benchmark_matrix_multiply()

    # 演示 4: 部署指南
    demo_deployment_guide()

    # 演示 5: 方案对比
    demo_deployment_comparison()

    # 最终总结
    print("\n" + "=" * 70)
    print("【s24 总结】")
    print("=" * 70)
    print("  ✓ 理解了训练与推理的不同优化目标")
    print("  ✓ 掌握了 KV Cache 的工作原理和 O(n²)→O(n) 的加速")
    print("  ✓ 理解了 Flash Attention 的 IO 感知设计")
    print("  ✓ 实践了模型量化 FP32 → INT8 (4× 内存节省)")
    print("  ✓ 了解了 vLLM PagedAttention 的分页管理思想")
    print("  ✓ 知道了 Ollama/llama.cpp/vLLM 等部署方案的选择")
    print()
    print("  核心思想：")
    print("    - KV Cache: 空间换时间，避免重复计算")
    print("    - Flash Attention: IO 感知，减少数据搬移")
    print("    - 量化: 降低精度，换取内存和带宽")
    print("    - PagedAttention: 分页管理，消除内存碎片")
    print()
    print("  这四项技术共同构成了现代 LLM 高效推理的基础。")
    print("=" * 70)


if __name__ == "__main__":
    main()
