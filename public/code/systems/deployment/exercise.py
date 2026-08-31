# -*- coding: utf-8 -*-
"""
s24 模型部署与推理优化 — 练习代码
=================================
请完成以下 TODO 任务，巩固对推理优化技术的理解。

每个 TODO 都有详细的指示和预期输出描述。
建议先阅读 README.md，再尝试独立补全代码。
"""

import time
import numpy as np
from typing import List, Tuple, Dict, Optional


# ============================================================================
# TODO 1: 实现简单的 KV Cache
# ============================================================================
class SimpleTransformerDecoder:
    """
    简化的 Transformer Decoder（单层、单头），用于演示 KV Cache。
    """

    def __init__(self, d_model: int = 32):
        """
        初始化 decoder 参数。

        参数:
            d_model: 嵌入维度
        """
        self.d_model = d_model
        np.random.seed(42)

        # Q, K, V 投影权重
        self.W_q = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
        self.W_k = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
        self.W_v = np.random.randn(d_model, d_model).astype(np.float32) * 0.02

    def attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        计算缩放点积注意力。

        公式: softmax(Q @ K.T / sqrt(d)) @ V

        参数:
            Q: Query, shape (n_q, d_model)
            K: Key, shape (n_kv, d_model)
            V: Value, shape (n_kv, d_model)

        返回:
            output: shape (n_q, d_model)
        """
        scale = np.sqrt(self.d_model)  # 1/sqrt(d_k)
        scores = Q @ K.T / scale  # (n_q, n_kv)
        # 数值稳定的 softmax
        scores = scores - scores.max(axis=-1, keepdims=True)
        attn = np.exp(scores)
        attn = attn / attn.sum(axis=-1, keepdims=True)
        return attn @ V

    def generate_without_cache(
        self, input_tokens: np.ndarray, max_new_tokens: int = 10
    ) -> Tuple[np.ndarray, float, int]:
        """
        不使用 KV Cache 的逐 token 生成。

        每生成一个新 token，将整个序列重新跑一遍注意力。

        参数:
            input_tokens: 输入 token 嵌入，shape (seq_len, d_model)
            max_new_tokens: 最大生成 token 数

        返回:
            (完整序列嵌入, 耗时, 总计算量)
        """
        sequence = input_tokens.copy()  # 初始序列
        # TODO: 完成无缓存的生成循环

        total_compute = 0
        start = time.perf_counter()

        for _ in range(max_new_tokens):
            # TODO: 步骤 1 — 对整个序列计算 Q, K, V
            # Q = sequence @ self.W_q
            Q = None  # ← TODO
            K = None  # ← TODO
            V = None  # ← TODO

            # TODO: 步骤 2 — 计算注意力（所有 token 的所有位置）
            attn_output = None  # ← TODO: self.attention(Q, K, V)

            # TODO: 步骤 3 — 取最后一个位置的输出作为下一个 token
            next_token = None  # ← TODO: attn_output[-1:] shape (1, d_model)

            # TODO: 步骤 4 — 追加到序列
            sequence = None  # ← TODO: np.concatenate([sequence, next_token], axis=0)

            # 统计计算量（序列长度 = 每次重新计算的 K,V 数量）
            total_compute += len(sequence)

        elapsed = time.perf_counter() - start
        return sequence, elapsed, total_compute

    def generate_with_cache(
        self, input_tokens: np.ndarray, max_new_tokens: int = 10
    ) -> Tuple[np.ndarray, float, int]:
        """
        使用 KV Cache 的逐 token 生成。

        KV Cache 的核心：
        - 缓存已计算过的 K 和 V
        - 新 token 只计算自己的 K、V，与缓存的 K、V 做注意力

        参数:
            input_tokens: 输入 token 嵌入，shape (seq_len, d_model)
            max_new_tokens: 最大生成 token 数

        返回:
            (完整序列嵌入, 耗时, 总计算量)
        """
        sequence = input_tokens.copy()
        # TODO: 完成有缓存的生成循环

        total_compute = 0
        start = time.perf_counter()

        # ---- 第一步：预填充（Prefill）— 处理整个输入序列 ----
        # 对输入序列计算 K, V 并缓存
        # TODO: 计算输入序列的 K 和 V 并存入缓存
        cached_K = None  # ← TODO: 对初始序列计算 K
        cached_V = None  # ← TODO: 对初始序列计算 V

        total_compute += len(input_tokens)

        # ---- 后续步骤：自回归生成 ----
        current_token = sequence[-1:]  # shape (1, d_model)，用最后一个 token 开始

        for _ in range(max_new_tokens):
            # TODO: 步骤 1 — 只计算新 token 的 Q, K, V
            # 注意：current_token 是 (1, d_model)
            Q_new = None  # ← TODO: current_token @ self.W_q
            K_new = None  # ← TODO: current_token @ self.W_k
            V_new = None  # ← TODO: current_token @ self.W_v

            # TODO: 步骤 2 — 将新 K, V 追加到缓存中
            cached_K = None  # ← TODO: np.concatenate([cached_K, K_new], axis=0)
            cached_V = None  # ← TODO: np.concatenate([cached_V, V_new], axis=0)

            # TODO: 步骤 3 — 用 Q_new 和所有缓存的 K, V 做注意力
            attn_output = None  # ← TODO: self.attention(Q_new, cached_K, cached_V)

            # TODO: 步骤 4 — 输出作为下一个 token
            current_token = attn_output

            # TODO: 步骤 5 — 追加到完整序列（用于最终返回）
            sequence = None  # ← TODO: np.concatenate([sequence, current_token], axis=0)

            # 统计：只计算了 1 个新 token（vs 无缓存时的整个序列）
            total_compute += 1

        elapsed = time.perf_counter() - start
        return sequence, elapsed, total_compute


# ---- 测试 TODO 1 ----
def test_kv_cache():
    """测试 KV Cache 实现。"""
    print("=" * 60)
    print("TODO 1 测试: KV Cache 实现")
    print("=" * 60)

    decoder = SimpleTransformerDecoder(d_model=32)
    np.random.seed(99)

    # 创建一个初始输入序列（模拟 5 个 token）
    init_len = 5
    input_tokens = np.random.randn(init_len, 32).astype(np.float32)

    # 运行有缓存和无缓存版本
    result_cache = decoder.generate_with_cache(input_tokens, max_new_tokens=8)
    result_no_cache = decoder.generate_without_cache(input_tokens, max_new_tokens=8)

    if result_cache is None or result_no_cache is None:
        print("  TODO 未完成，请补全 generate_with_cache 和 generate_without_cache 方法")
    else:
        seq_cache, t_cache, comp_cache = result_cache
        seq_nocache, t_nocache, comp_nocache = result_no_cache

        print(f"\n  输入序列长度: {init_len}, 生成 8 个新 token")
        print(f"  {'方法':<20} {'耗时 (ms)':<15} {'计算量':<10} {'序列长度'}")
        print(f"  {'─' * 55}")
        print(f"  {'无 KV Cache':<20} {t_nocache*1000:<15.3f} {comp_nocache:<10} {len(seq_nocache)}")
        print(f"  {'有 KV Cache':<20} {t_cache*1000:<15.3f} {comp_cache:<10} {len(seq_cache)}")

        if t_cache > 0:
            print(f"\n  ✓ 加速比: {t_nocache/t_cache:.2f}×")
            print(f"  ✓ 计算量减少: {comp_nocache/comp_cache:.1f}×")
        print(f"  ✓ 输出序列长度一致: {len(seq_cache) == len(seq_nocache)}")

    print()


# ============================================================================
# TODO 2: 实现权重量化（FP32 → INT8，逐通道）
# ============================================================================
def quantize_weights_per_channel(
    weights_fp32: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    实现逐通道（Per-Channel）FP32 → INT8 对称量化。

    对称量化（不使用零点）的公式:
        s = max(|W|) / 127   (INT8 有符号范围 [-127, 127])
        W_int8 = round(W / s), clamp to [-127, 127]

    参数:
        weights_fp32: FP32 权重矩阵，shape (out_features, in_features)

    返回:
        w_int8: INT8 权重矩阵 (int8), shape (out_features, in_features)
        scales: 每行的缩放因子，shape (out_features,)
              scales[i] = max(|weights_fp32[i]|) / 127

    提示：
        1. 对每一行（axis=1）计算 max(|w|)
        2. 计算 scale = max_abs / 127
        3. 避免 scale 为 0
        4. 量化: w_int8 = round(w / scale)，并 clamp 到 [-127, 127]
        5. 转 int8 类型
    """
    # TODO: 实现逐通道对称量化

    # 步骤 1: 计算每行的最大绝对值
    max_abs = None  # ← TODO: np.max(np.abs(weights_fp32), axis=1) shape (out_features,)

    # 步骤 2: 计算缩放因子
    # s = max_abs / 127, 避免除零
    scales = None  # ← TODO

    # 步骤 3: 量化
    # w_int8 = round(weights_fp32 / scales[:, np.newaxis])
    w_int8_float = None  # ← TODO

    # 步骤 4: Clamp 到 [-127, 127] 并转 int8
    w_int8 = None  # ← TODO: np.clip(...).astype(np.int8)

    return w_int8, scales.astype(np.float32)


def dequantize_weights_per_channel(
    w_int8: np.ndarray,
    scales: np.ndarray
) -> np.ndarray:
    """
    将 INT8 权重反量化为近似的 FP32。

    反量化: W_fp32_approx = W_int8 * scale

    参数:
        w_int8: INT8 权重矩阵，shape (out_features, in_features)
        scales: 每行缩放因子，shape (out_features,)

    返回:
        w_fp32_deq: 反量化后的 FP32 权重（有精度损失）
    """
    # TODO: 实现反量化
    w_deq = None  # ← TODO: w_int8.astype(np.float32) * scales[:, np.newaxis]
    return w_deq


# ---- 测试 TODO 2 ----
def test_weight_quantization():
    """测试权重量化实现。"""
    print("=" * 60)
    print("TODO 2 测试: 权重量化")
    print("=" * 60)

    np.random.seed(42)
    out_features, in_features = 64, 64

    # 创建模拟权重（正态分布，一些通道幅度更大）
    w_fp32 = np.random.randn(out_features, in_features).astype(np.float32) * 0.05
    w_fp32[:16] *= 3.0  # 前 16 行权重幅度更大
    w_fp32[-8:] *= 0.3  # 最后 8 行权重幅度更小

    w_int8, scales = quantize_weights_per_channel(w_fp32)

    if w_int8 is None:
        print("  TODO 未完成，请补全 quantize_weights_per_channel 函数")
    else:
        w_deq = dequantize_weights_per_channel(w_int8, scales)

        if w_deq is None:
            print("  TODO 未完成，请补全 dequantize_weights_per_channel 函数")
        else:
            # 计算误差
            error = np.abs(w_fp32 - w_deq)
            mae = error.mean()  # 平均绝对误差
            max_err = error.max()

            print(f"\n  权重矩阵: {w_fp32.shape}")
            print(f"  FP32 范围: [{w_fp32.min():.4f}, {w_fp32.max():.4f}]")
            print(f"  INT8 范围: [{w_int8.min()}, {w_int8.max()}]")
            print(f"  Scales 范围: [{scales.min():.6f}, {scales.max():.6f}]")

            print(f"\n  量化误差:")
            print(f"    平均绝对误差 (MAE): {mae:.6f}")
            print(f"    最大绝对误差:       {max_err:.6f}")
            print(f"    相对误差:           {mae / (np.abs(w_fp32).mean() + 1e-8):.4%}")

            # 内存对比
            size_fp32 = w_fp32.nbytes
            size_int8 = w_int8.nbytes
            size_total = size_int8 + scales.nbytes
            print(f"\n  内存占用:")
            print(f"    FP32: {size_fp32} bytes")
            print(f"    INT8 (权重): {size_int8} bytes")
            print(f"    INT8 (权重+scales): {size_total} bytes")
            print(f"    压缩比: {size_fp32/size_total:.2f}×")

            # 测试推理输出保真度
            test_input = np.random.randn(in_features).astype(np.float32)
            out_fp32 = w_fp32 @ test_input
            out_int8 = w_deq @ test_input
            # 余弦相似度
            cos_sim = np.dot(out_fp32, out_int8) / (
                np.linalg.norm(out_fp32) * np.linalg.norm(out_int8) + 1e-10
            )
            print(f"\n  推理输出保真度:")
            print(f"    余弦相似度: {cos_sim:.6f} (1.0 = 完全一致)")
            if cos_sim > 0.99:
                print(f"    ✓ 输出质量保真度非常高")
            elif cos_sim > 0.95:
                print(f"    ✓ 输出质量可接受")
            else:
                print(f"    ⚠ 输出质量有较明显下降")

    print()


# ============================================================================
# TODO 3: 推理速度基准测试（不同 batch size）
# ============================================================================
def benchmark_inference_speed(
    model_dim: int = 1024,
    seq_len: int = 128,
    batch_sizes: List[int] = [1, 2, 4, 8, 16, 32],
    n_trials: int = 20
) -> Dict[int, float]:
    """
    对不同 batch size 下的推理速度进行基准测试。

    模拟一个简单的 Transformer 前向传播：
    - QKV 投影: (B, S, D) @ (D, 3*D)
    - 注意力: (B, S, D) @ (B, D, S) @ (B, S, D) 的简化
    - FFN: (B, S, D) @ (D, 4*D) 和 (B, S, 4*D) @ (4*D, D)

    参数:
        model_dim: 模型维度 D
        seq_len: 序列长度 S
        batch_sizes: 要测试的 batch size 列表
        n_trials: 每个 batch size 的重复次数

    返回:
        results: {batch_size: avg_time_ms}

    提示：
        1. 对每个 batch size:
           a. 创建输入张量 (B, S, D)
           b. 创建模拟权重矩阵
           c. 运行 n_trials 次简化前向传播
           d. 计算平均耗时
        2. 使用 time.perf_counter() 进行精确计时
        3. 预热一次再开始计时
    """
    # TODO: 实现推理速度基准测试

    results = {}

    print(f"\n  模型维度 D={model_dim}, 序列长度 S={seq_len}")
    print(f"  {'Batch Size':<15} {'平均耗时 (ms)':<18} {'归一化 (每样本)'}")

    # 创建模拟权重（只需创建一次）
    np.random.seed(0)
    W_qkv = np.random.randn(model_dim, 3 * model_dim).astype(np.float32)
    W_o = np.random.randn(model_dim, model_dim).astype(np.float32)
    W_ffn1 = np.random.randn(model_dim, 4 * model_dim).astype(np.float32)
    W_ffn2 = np.random.randn(4 * model_dim, model_dim).astype(np.float32)

    for batch_size in batch_sizes:
        # TODO: 步骤 1 — 创建输入张量
        x = None  # ← TODO: np.random.randn(batch_size, seq_len, model_dim).astype(np.float32)

        # 预热
        _ = x @ W_qkv

        # TODO: 步骤 2 — 计时运行
        times = []
        for _ in range(n_trials):
            start = time.perf_counter()

            # 模拟 Transformer 前向传播
            # 1. QKV 投影
            qkv = None  # ← TODO: x @ W_qkv

            # 2. 简化的注意力（用均值池化模拟，避免 O(S²) 计算）
            # 在实际 benchmark 中，简化注意力以减少测试时间
            attn = qkv.mean(axis=1, keepdims=True)  # 模拟：沿序列维度聚合

            # 3. 注意力输出投影
            attn_out = None  # ← TODO: attn @ W_o

            # 4. FFN 层
            ffn_hidden = None  # ← TODO: attn_out @ W_ffn1
            # ReLU 激活
            ffn_hidden = np.maximum(0, ffn_hidden)
            ffn_out = None  # ← TODO: ffn_hidden @ W_ffn2

            # 记录耗时
            end = time.perf_counter()
            times.append((end - start) * 1000)  # 转换为 ms

        avg_time = np.mean(times)
        std_time = np.std(times)
        per_sample = avg_time / batch_size  # 平均每样本耗时

        results[batch_size] = avg_time
        print(f"  {batch_size:<15} {avg_time:.3f} ± {std_time:.3f} ms     {per_sample:.3f} ms/样本")

    # TODO: 步骤 3 — 分析结果
    # 计算最优 batch size（吞吐量最高的）
    if results:
        throughputs = {bs: bs / (t / 1000) for bs, t in results.items()}  # 每秒处理样本数
        best_bs = max(throughputs, key=throughputs.get)
        print(f"\n  最高吞吐量: Batch Size={best_bs}, {throughputs[best_bs]:.1f} 样本/秒")
        print(f"  (注意: 在真实 GPU 上，更大的 batch size 通常带来更高的吞吐量，但受限于显存)")

    return results


# ---- 测试 TODO 3 ----
def test_inference_benchmark():
    """测试推理基准测试功能。"""
    print("=" * 60)
    print("TODO 3 测试: 推理速度基准测试")
    print("=" * 60)

    # 使用较小的模型维度以加快测试
    results = benchmark_inference_speed(
        model_dim=256,
        seq_len=64,
        batch_sizes=[1, 2, 4, 8],
        n_trials=10
    )

    if results is None or len(results) == 0:
        print("\n  TODO 未完成，请补全 benchmark_inference_speed 函数")
    else:
        print(f"\n  测试完成，共测试了 {len(results)} 个 batch size")
        # 检查 batch size 增大后每样本时间是否下降（吞吐量提升）
        if 1 in results and 4 in results:
            per_bs1 = results[1] / 1
            per_bs4 = results[4] / 4
            if per_bs4 < per_bs1:
                print(f"  ✓ Batch size 增大后，每样本耗时下降 (batching 效率提升)")
            print(f"    BS=1: {per_bs1:.3f}ms/样本, BS=4: {per_bs4:.3f}ms/样本")

    print()


# ============================================================================
# 主程序
# ============================================================================
if __name__ == "__main__":
    print("\n╔" + "═" * 58 + "╗")
    print("║" + " " * 8 + "s24 模型部署与推理优化 — 动手练习" + " " * 14 + "║")
    print("║" + " " * 6 + "请依次完成 TODO 1, 2, 3" + " " * 26 + "║")
    print("╚" + "═" * 58 + "╝\n")

    test_kv_cache()
    test_weight_quantization()
    test_inference_benchmark()

    print("=" * 60)
    print("所有测试完成！请检查输出结果。")
    print("如有未通过的测试，请回到对应的 TODO 部分补全代码。")
    print()
    print("提示：")
    print("  TODO 1: KV Cache — 理解推理加速的核心机制")
    print("  TODO 2: 量化 — 理解模型压缩的数学原理")
    print("  TODO 3: 基准测试 — 理解 batch size 对推理效率的影响")
    print()
    print("扩展思考：")
    print("  1. KV Cache 的内存占用如何随序列长度增长？")
    print("  2. 逐通道量化为什么比整体量化更精确？")
    print("  3. 为什么更大的 batch size 通常能提升吞吐量？")
    print("=" * 60)
