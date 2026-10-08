---
title: "s24 模型部署与推理优化 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# s24 模型部署与推理优化 — demo.py 代码详解

<a href="/notebook/code/applied/systems/deployment/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/applied/systems/deployment/code
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

#### 2.3 有缓存的代码与形状核对

每一步只取 `sequence[t-1:t]`，形状为 $(1,d_{\mathrm{model}})$。它经过三个投影得到新 token 的 Q、K、V；每个头的形状均为 $(1,d_{\mathrm{head}})$。历史 K/V 与新行拼接后为 $(t,d_{\mathrm{head}})$。

$$
Q_{\mathrm{new}}=x_tW_Q,\quad
K_{1:t}=[K_{1:t-1};x_tW_K],\quad
V_{1:t}=[V_{1:t-1};x_tW_V],
$$

$$
o_t=\operatorname{softmax}\left(
\frac{Q_{\mathrm{new}}K_{1:t}^{\mathsf T}}{\sqrt{d_{\mathrm{head}}}}
\right)V_{1:t}.
$$

注意力分数是 $1\times t$，不是 $t\times t$。代码中的投影 token 计数在头循环外加 1，因此不会把头数误算进去。这个简化 demo 使用固定输入向量和单层注意力，不是完整语言模型生成器；无缓存分支的早期输出被丢弃，只比较末 token 输出。两分支使用同一组固定随机输入，运行时用 `np.allclose` 检查最终输出一致。

**手算计数。** 生成 4 个 token，无缓存依次投影 $1,2,3,4$ 个 token，累计 10；缓存分支为 $1+1+1+1=4$。这里统计的是“完成一组 K/V 投影的 token 数”，不是 FLOPs，也不是实测加速比。

**内存代价。** 对于 Llama 2-7B 的常规多头注意力（$L=32,H_{\rm KV}=32,d_h=128$），FP16 的每 token 缓存为 $2LH_{\rm KV}d_h\times2=524288$ bytes，即 0.5 MiB。2048 个 token 为 1 GiB；batch 大小还会成倍放大。GQA/MQA 应使用 KV 头数，不能直接套用 Query 头数。

#### 2.4 性能对比：不要混淆三个口径

- **累计 K/V 投影 token 数**：无缓存 $n(n+1)/2$，有缓存 $n$，比值 $(n+1)/2$。
- **注意力计算**：固定隐藏维度，单步从 $O(t^2)$ 变为 $O(t)$；从头生成 $n$ 个 token 的累计注意力开销分别为 $O(n^3)$ 与 $O(n^2)$。
- **墙钟耗时**：还受矩阵大小、BLAS、内存分配与缓存拼接影响，不能把投影计数比直接说成加速倍数。`np.concatenate` 的反复复制也是教学实现的额外开销，生产实现通常预分配或分页管理缓存。

左图是实际测得的累计耗时；右图是累计投影 token 计数。$n=100$ 时 50.5、$n=500$ 时 250.5 都只是投影工作量之比，不是保证的耗时加速比。

> **旧图说明**：仓库现存 `kv_cache_comparison.png` 来自修正前的实现，不能用于验证修正后的结果。本次只静态修正代码，未运行基准测试，也未重画该图。

### 第3步：模型量化演示 — FP32 → INT8

#### 3.1 量化的数学：与实际代码一致的 affine uint8

本例实际使用 `np.uint8` 的 $[0,255]$，不是有符号对称 INT8。函数名 `quantize_fp32_to_int8` 为兼容已有调用保留；“8 bit”描述位宽，不能混同两种编码。

逐通道时每行独立设 $a_i=\min(\min_jW_{ij},0)$、$b_i=\max(\max_jW_{ij},0)$，确保表示范围包括实数零；整体量化则使用全矩阵的范围：

$$
s_i=\frac{b_i-a_i}{255},\quad
z_i=\operatorname{clip}\left(\operatorname{round}(-a_i/s_i),0,255\right).
$$

若整行全为零，取 $s_i=1,z_i=0$，避免除零。量化和反量化必须使用同一个整数零点：

$$
q_{ij}=\operatorname{clip}\left(\operatorname{round}(W_{ij}/s_i)+z_i,0,255\right),
\qquad
\widehat W_{ij}=s_i(q_{ij}-z_i).
$$

**正数行手算。** 对 $W=[1,2]$，取 $a=0,b=2,s=2/255,z=0$；NumPy 的舍入给出 $q=[128,255]$，还原为 $[256/255,2]\approx[1.00392,2]$。旧代码混用了 `(W-min)/s` 和被截断的零点，会错误还原成 $[0,1]$。

**常量和负数。** $W=[5,5]$ 对应 $s=5/255,z=0,q=[255,255]$，可还原为 5；$W=[-2,-1]$ 的范围为 $[-2,0]$，零点为 255。非零常量不能直接当作“最大值等于最小值，所以全部量化为零”。

**代码对应。** `w_min/w_max` 先扩展到包含零，`scales` 对应 $s$，`zero_points` 对应 $z$，`w_int8` 是为兼容保留的变量名，实际 dtype 为 uint8。反量化只做 `scales * (w_float - zero_points)`，不再另加最小值。

逐通道量化使小范围行拥有更细的量化间隔，但也要保存每行的 scale/zero-point。它通常有助于降低误差，并不保证对任何输入、任何误差指标都严格更好。

#### 3.2 量化误差分析

代码对比了两种量化方式：
- **逐通道**：每行独立 scale，报告实测 MAE（平均绝对误差）
- **整体**：一个全局 scale；少数幅度很大的行可能降低其余行的表示精度

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
\text{INT4 理论: } & 8.00\times \text{（仅权重，不含 scale 开销）}
\end{aligned}
$$

**实际考虑**：INT4 的 scale 开销比例更大（每个 scale 是 FP32=4 bytes，128 个权重共享一个 scale 时开销为 4/(128×0.5)=6.25%（相对打包后的 INT4 权重字节数；尚未计零点等元数据））。

#### 3.4 量化可视化

> **旧图说明**：现存 `quantization_demo.png` 来自修正前公式，本次未重新运行生成，不能视为修正后误差的验证。

重新运行后应核对四张子图：
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
| 量化公式 | $q=\operatorname{clip}(\operatorname{round}(W/s)+z,0,255)$ | `quantize_fp32_to_int8()` |
| 逐通道量化 | 每行独立 scale，保留更多信息 | `per_channel=True` |
| 反量化 | $\hat{W}=s(q-z)$ | `dequantize_int8_to_fp32()` |
| 余弦相似度保真度 | 量化后输出与 FP32 输出的方向一致性 | `np.dot(out_fp32, out_int8) / (...)` |
| PagedAttention | KV Cache 分页管理，消除内存碎片 | 文字说明 |
| GGUF | llama.cpp 的量化格式，专为 CPU 设计 | Q4_K_M, Q5_K_M 等 |


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/applied/systems/deployment/code/demo.py`
