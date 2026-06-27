---
title: "s24 模型部署与推理优化 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# s24 模型部署与推理优化 — demo.py 代码详解

<a href="../code/s24_deployment_inference/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd s24_deployment_inference/code
python demo.py
```

## 代码逐段详解

### 第1步：导入库 — 每个库做什么

```python
import numpy as np     # 数值计算：矩阵乘法模拟注意力，量化计算
import time            # 性能计时：测量推理耗时
import matplotlib.pyplot as plt  # 可视化：KV Cache 加速比、量化误差对比
```

**设计说明**：本 demo 用纯 NumPy 实现，不依赖任何 ML 框架，专注于展示推理优化的数学原理而非工程实现。

### 第2步：KV Cache 演示 — 避免重复计算的核心技术

#### 2.1 为什么需要 KV Cache

在自回归生成中，每生成一个新 token，都需要计算所有历史 token 的 Key 和 Value。无缓存时：

$$
\text{计算量} = 1 + 2 + 3 + \dots + n = \frac{n(n+1)}{2} = O(n^2)
$$

KV Cache 将历史 K,V 存储下来，每次只计算新 token：

$$
\text{计算量} = n \times 1 = O(n)
$$

#### 2.2 无缓存的代码

```python
def generate_without_kv_cache(self, seq_len):
    for t in range(1, seq_len + 1):
        X_t = sequence[:t]                  # 取前 t 个 token
        Q = X_t @ self.W_q                  # (t, d_model)
        K = X_t @ self.W_k                  # (t, d_model) — 重复计算！
        V = X_t @ self.W_v                  # (t, d_model) — 重复计算！
        total_compute += t                  # 记录计算量：t 个 token 的 K/V
```

**每次迭代的计算量**：第 t 步需要计算 t 个 token 的 K 和 V，总计算量 $\sum_{t=1}^{n} t = O(n^2)$。

#### 2.3 有缓存的代码

```python
def generate_with_kv_cache(self, seq_len):
    cached_K = [None] * self.n_heads        # 每个头独立缓存 K
    cached_V = [None] * self.n_heads        # 每个头独立缓存 V

    for t in range(1, seq_len + 1):
        # ... 计算 Q, K, V ...
        for h in range(self.n_heads):
            if cached_K[h] is not None:
                # 拼接缓存和新 token 的 K,V
                full_K = np.concatenate([cached_K[h], K_heads[h][-1:]], axis=0)
                full_V = np.concatenate([cached_V[h], V_heads[h][-1:]], axis=0)
            else:
                full_K = K_heads[h]
                full_V = V_heads[h]

            # 更新缓存
            cached_K[h] = full_K
            cached_V[h] = full_V
            total_compute += 1               # 只计算 1 个新 token
```

**KV Cache 的核心操作**：
1. **首次**（t=1）：计算全部 t 个 token 的 K,V，存入缓存
2. **后续**（t>1）：只计算新 token 的 K,V，用 `np.concatenate` 拼接到缓存中

**内存代价**：对于 Llama 2-7B（$L=32, H=32, d_h=128$），每个 token 的 KV Cache 约 0.5 MB。2048 个 token 需要约 1 GB 额外显存。这就是为什么长序列推理的瓶颈往往是**内存而非计算**。

#### 2.4 性能对比可视化

基准测试多个序列长度（10, 20, 50, 100, 200, 500），输出两张图：
- **左图**：推理时间对比（无缓存 $O(n^2)$ 曲线 vs 有缓存 $O(n)$ 直线）
- **右图**：计算量对比（同样展示复杂度差异）

**预期结果**：序列越长，加速比越大。$n=100$ 时理论加速约 50×，$n=500$ 时约 250×。

### 第3步：模型量化演示 — FP32 → INT8

#### 3.1 量化的数学

**目标**：将 FP32 权重 $W \in \mathbb{R}^{m \times n}$ 压缩为 INT8（每个值 1 字节，而非 4 字节）。

**对称量化公式**（逐通道）：

$$
s_i = \frac{\max(|W_{i,:}|)}{\text{127}} \quad \text{（INT8 有符号范围 [-127, 127]）}
$$

$$
W_{q} = \text{round}\left(\frac{W}{s}\right),\; \text{clamp to } [-127, 127]
$$

**反量化**：$\hat{W} = s \cdot W_q$

```python
def quantize_fp32_to_int8(weights, per_channel=True):
    if per_channel:
        w_min = weights.min(axis=1, keepdims=True)     # (out_features, 1)
        w_max = weights.max(axis=1, keepdims=True)     # (out_features, 1)
    else:
        w_min = weights.min()                           # 标量 -> 整体量化
        w_max = weights.max()

    scales = (w_max - w_min) / 255.0                   # 256 个量化级别 (0-255)
    scales = np.where(scales < 1e-10, 1.0, scales)      # 避免除零
    zero_points = np.round(-w_min / scales)
    zero_points = np.clip(zero_points, 0, 255)

    w_int8 = np.round((weights - w_min) / scales)
    w_int8 = np.clip(w_int8, 0, 255).astype(np.uint8)
    return w_int8, scales, zero_points
```

**为什么做逐通道量化**：不同输出通道（行）的权重分布可能差异很大。某些通道的权重幅度是其他通道的 2-3 倍。逐通道量化给每行独立的 scale，保留更多信息。

#### 3.2 量化误差分析

代码对比了两种量化方式：
- **逐通道**：每行独立 scale，MAE（平均绝对误差）较小
- **整体**：一个全局 scale，MAE 更大——对幅度异常的行量化损失严重

**推理输出保真度**：用一个测试输入向量 $\mathbf{x}$ 做矩阵乘法：

$$
\text{output}_{\text{fp32}} = W \cdot \mathbf{x}, \quad \text{output}_{\text{int8}} = \hat{W} \cdot \mathbf{x}
$$

计算余弦相似度 $\cos(\text{output}_{\text{fp32}}, \text{output}_{\text{int8}})$ —— 越接近 1.0 表示量化对输出的影响越小。

#### 3.3 内存节省

以 512×512 权重矩阵为例：

$$
\begin{aligned}
\text{FP32: } & 512 \times 512 \times 4 \text{ bytes} = 1,048,576 \text{ bytes} \approx 1 \text{ MB} \\
\text{INT8: } & 512 \times 512 \times 1 \text{ byte} = 262,144 \text{ bytes} \approx 256 \text{ KB} \\
\text{压缩比: } & 4.00\times \\
\text{INT4 理论: } & 4.00\times \text{（仅权重，不含 scale 开销）}
\end{aligned}
$$

**实际考虑**：INT4 的 scale 开销比例更大（每个 scale 是 FP32=4 bytes，128 个权重共享一个 scale 时开销为 4/128≈3%）。

#### 3.4 量化可视化

四张子图：
1. **原始 FP32 权重分布**（直方图）：接近正态分布 $\mathcal{N}(0, 0.02^2)$
2. **反量化权重 vs 原始权重散点图**：点应该沿着 $y=x$ 对角线，偏离程度表示量化误差
3. **逐通道 vs 整体量化误差对比**（前 50 个通道）：逐通道误差均匀，整体量化对幅度异常的通道误差大
4. **内存占用柱状图**：直观对比 FP32/INT8/INT4 的存储需求

### 第4步：推理基准测试 — 矩阵乘法性能

```python
def benchmark_matrix_multiply(sizes, n_trials):
    for size in sizes:
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)

        # 计时 n_trials 次
        times = []
        for _ in range(n_trials):
            start = time.perf_counter()
            C = A @ B
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)

        # GFLOPS = 2*N^3 / (time/1000) / 1e9
        flops = 2 * size ** 3
        gflops = flops / (avg_time / 1000) / 1e9
```

**Transformer 推理中的四个关键矩阵乘法**：

| 操作 | 形状 | 计算量 |
|------|------|--------|
| QKV 投影 | $(S, D) \times (D, 3D)$ | $6SD^2$ |
| 注意力输出 | $(S, D) \times (D, D)$ | $2SD^2$ |
| FFN 第一层 | $(S, D) \times (D, 4D)$ | $8SD^2$ |
| FFN 第二层 | $(S, 4D) \times (4D, D)$ | $8SD^2$ |

**优化策略总结**：
- 量化 INT8/INT4：减少 2-4× 内存带宽压力
- Flash Attention：减少注意力计算的 IO 瓶颈
- KV Cache：避免重复计算历史 token
- Batching：利用 GPU 并行处理多个请求

### 第5步：实际部署工具指南 — Ollama / vLLM / llama.cpp

代码以文字说明的方式展示了三种部署方案的基本用法：

**Ollama**（最简单）：
- `ollama pull qwen2.5:0.5b` → 约 350MB 下载
- `ollama run qwen2.5:0.5b` → 交互式对话
- API 端点：`POST http://localhost:11434/api/generate`

**vLLM**（高性能）：
- PagedAttention 使内存利用率从 ~40% 提升到 ~96%
- 支持连续批处理（continuous batching）
- 与 OpenAI API 完全兼容

**llama.cpp + GGUF**（CPU 推理）：
- Q4_K_M (~4.5 bits/p)：推荐，质量与大小平衡
- Q8_0 (~8 bits/p)：几乎无损
- 在普通笔记本上运行 7B 模型成为可能

**方案选择建议**：

| 场景 | 推荐方案 |
|------|---------|
| 个人学习/开发 | Ollama |
| CPU/边缘设备 | llama.cpp + GGUF |
| 生产服务 | vLLM (GPU) |
| 极致性能 | TensorRT-LLM |

### 关键概念速查表

| 概念 | 一句话解释 | 代码位置 |
|------|-----------|---------|
| KV Cache | 缓存历史 Key/Value，避免重复计算 $O(n^2)\to O(n)$ | `generate_with_kv_cache()` |
| 自回归生成 | 逐个 token 生成，每步依赖之前所有 token | `for t in range(1, seq_len+1)` |
| Flash Attention | IO 感知的分块计算，减少 HBM 读写 | 文字说明（无代码实现） |
| 量化公式 | $W_q = \text{round}((W - \min)/s),\; s = (\max-\min)/255$ | `quantize_fp32_to_int8()` |
| 逐通道量化 | 每行独立 scale，保留更多信息 | `per_channel=True` |
| 反量化 | $\hat{W} = s \cdot W_q$ | `dequantize_int8_to_fp32()` |
| 余弦相似度保真度 | 量化后输出与 FP32 输出的方向一致性 | `np.dot(out_fp32, out_int8) / (...)` |
| PagedAttention | KV Cache 分页管理，消除内存碎片 | 文字说明 |
| GGUF | llama.cpp 的量化格式，专为 CPU 设计 | Q4_K_M, Q5_K_M 等 |

## 完整代码

<<< @/snippets/s24_deployment_inference/demo.py
