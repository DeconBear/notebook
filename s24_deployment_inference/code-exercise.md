---
title: "s24 模型部署与推理优化 — exercise.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# s24 模型部署与推理优化 — exercise.py 练习指南

<a href="../code/s24_deployment_inference/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过补全三个 TODO 任务，深入理解推理优化的核心技术：
1. KV Cache 实现 —— 理解自回归推理加速的核心机制
2. 权重量化 —— 掌握模型压缩的数学原理
3. 推理速度基准测试 —— 理解 batch size 对效率的影响

## 预备知识

- KV Cache 原理：缓存 K/V → 只计算新 token，复杂度 $O(n^2) \to O(n)$
- 注意力公式：$\text{Attention}(Q,K,V) = \text{softmax}(QK^T/\sqrt{d})V$
- 对称量化：$s = \max(|W|)/127,\; W_{\text{int8}} = \text{round}(W/s)$
- 反量化：$\hat{W} = W_{\text{int8}} \cdot s$

## 任务清单

### TODO 1：实现简单的 KV Cache（`SimpleTransformerDecoder` 类）

**任务 1a**：实现 `generate_without_cache(input_tokens, max_new_tokens)`

每个生成步骤都需要：
```python
for _ in range(max_new_tokens):
    # 1. 对整个序列计算 Q, K, V
    Q = sequence @ self.W_q    # (full_len, d_model)
    K = sequence @ self.W_k    # (full_len, d_model) — 重复计算！
    V = sequence @ self.W_v

    # 2. 注意力
    attn_output = self.attention(Q, K, V)

    # 3. 取最后一个输出作为新 token
    next_token = attn_output[-1:]    # (1, d_model)

    # 4. 追加到序列
    sequence = np.concatenate([sequence, next_token], axis=0)

    total_compute += len(sequence)    # 统计计算量
```

**关键理解**：每步的 `len(sequence)` 逐次增大，总计算量为 $\sum_{t=1}^{n} t = O(n^2)$——这就是 KV Cache 要消除的重复计算。

**任务 1b**：实现 `generate_with_cache(input_tokens, max_new_tokens)`

```python
# 第一步：预填充（Prefill）—— 处理整个输入序列
cached_K = input_tokens @ self.W_k   # (init_len, d_model)
cached_V = input_tokens @ self.W_v
total_compute += len(input_tokens)

# 后续步骤：自回归生成
current_token = sequence[-1:]        # (1, d_model)
for _ in range(max_new_tokens):
    # 只计算新 token 的 Q, K, V
    Q_new = current_token @ self.W_q   # (1, d_model)
    K_new = current_token @ self.W_k   # (1, d_model)
    V_new = current_token @ self.W_v

    # 追加到缓存
    cached_K = np.concatenate([cached_K, K_new], axis=0)
    cached_V = np.concatenate([cached_V, V_new], axis=0)

    # 用 Q_new 和所有缓存的 K, V 做注意力
    attn_output = self.attention(Q_new, cached_K, cached_V)

    # 输出作为下一个 token
    current_token = attn_output
    sequence = np.concatenate([sequence, current_token], axis=0)
    total_compute += 1    # 只计算了 1 个新 token！
```

**预填充 vs 自回归**：
- **预填充（Prefill）**：第一步用整个输入序列并行计算所有 K,V，存入缓存
- **自回归（Decode）**：后续每步只计算当前 token 的 K,V，与缓存拼接

**预期输出**：
```
无 KV Cache: 计算量较大（随序列长度平方增长）
有 KV Cache: 计算量很小（随序列长度线性增长）
加速比随序列长度增加而增大
输出序列长度一致 ✓（两种方法生成相同的 token 数）
```

### TODO 2：实现权重量化（FP32 → INT8，逐通道）

**任务 2a**：实现 `quantize_weights_per_channel(weights_fp32)`

**对称量化**（不使用零点，简化版）：

```python
# 1. 每行最大绝对值
max_abs = np.max(np.abs(weights_fp32), axis=1)     # (out_features,)

# 2. 缩放因子（避免除零）
scales = max_abs / 127.0
scales = np.where(scales < 1e-10, 1.0, scales)     # 安全处理全零行

# 3. 量化
w_int8_float = weights_fp32 / scales[:, np.newaxis]
# scales[:, np.newaxis]: (out_features,) → (out_features, 1) 用于广播

# 4. Clamp 并转 int8
w_int8 = np.clip(np.round(w_int8_float), -127, 127).astype(np.int8)
```

**为什么用 `max(|W|)/127` 而非 `(max-min)/255`**：这是对称量化——零点固定在 0，值域 $[-127, 127]$。优点是不需要存储零点，实现更简单。

**任务 2b**：实现 `dequantize_weights_per_channel(w_int8, scales)`

```python
w_deq = w_int8.astype(np.float32) * scales[:, np.newaxis]
```

**预期输出**：
```
FP32 范围: [-0.15, 0.15] 左右
INT8 范围: [-127, 127]
Scales 范围: [0.0001, 0.0015] 左右
平均绝对误差 (MAE): 很小的值
推理输出余弦相似度: > 0.99（几乎无损！）
内存压缩比: ~3.5-4.0×
```

### TODO 3：推理速度基准测试（`benchmark_inference_speed` 函数）

**任务**：对不同 batch size 下的模拟 Transformer 前向传播进行计时。

**实现步骤**：
```python
for batch_size in batch_sizes:
    # 1. 创建输入张量
    x = np.random.randn(batch_size, seq_len, model_dim).astype(np.float32)

    # 预热
    _ = x @ W_qkv

    # 2. 计时
    times = []
    for _ in range(n_trials):
        start = time.perf_counter()

        # 模拟 Transformer 前向传播
        qkv = x @ W_qkv                    # (B, S, D) @ (D, 3D)
        attn = qkv.mean(axis=1, keepdims=True)  # 简化注意力
        attn_out = attn @ W_o              # 注意力输出投影
        ffn_hidden = attn_out @ W_ffn1     # FFN 第一层
        ffn_hidden = np.maximum(0, ffn_hidden)  # ReLU
        ffn_out = ffn_hidden @ W_ffn2      # FFN 第二层

        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

    avg_time = np.mean(times)
    per_sample = avg_time / batch_size     # 平均每样本耗时

# 3. 分析吞吐量
throughputs = {bs: bs / (t / 1000) for bs, t in results.items()}  # 样本/秒
best_bs = max(throughputs, key=throughputs.get)
```

**为什么用均值池化替代真正的注意力**：真正注意力的计算量是 $O(B \cdot S^2 \cdot D)$，在序列较长时主导 benchmark。用均值池化简化为 $O(B \cdot S \cdot D)$，让 benchmark 聚焦于矩阵乘法的性能特征。

**`np.maximum(0, x)` 的作用**：ReLU 激活函数。在真实 Transformer 中，FFN 的第一层后接激活函数（如 SiLU/GeLU）。

**预期结果**：
- 更大的 batch size → 总体耗时增加（计算更多）
- 但**每样本耗时下降**（batching 分摊了矩阵乘法的固定开销）
- 吞吐量（样本/秒）随 batch size 增大而提升，但受限于内存

## 完成后的验证

全部三个 TODO 通过测试后，运行 `python code/demo.py` 观察：
1. KV Cache 的加速比随序列长度如何增长
2. 量化前/后权重的误差分布和输出保真度
3. 不同矩阵大小下的推理性能特征

## 完整代码

<<< @/snippets/s24_deployment_inference/exercise.py
