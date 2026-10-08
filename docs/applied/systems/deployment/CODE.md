# s24 模型部署与推理优化 -- 代码说明与运行报告

## 程序做了什么
用纯 NumPy 演示大模型推理的两个核心优化技术：KV Cache（模拟自回归生成中 Key/Value 矩阵的缓存复用，对比有/无缓存在不同序列长度下的投影 token 计数和耗时差异）和模型量化（FP32 到 uint8 的仿射非对称量化过程，展示压缩率、量化前后权重分布及精度损失 MSE/MAE）。

## 运行方法
```bash
cd docs/applied/systems/deployment/code
python demo.py
```

## 输出说明（本次未重新运行）

> 代码已修正缓存分支的重复投影和量化零点错误。以下是输出结构说明，不是本次执行记录；已有图片保留但不能作为修正后结果的证据。

### 输出摘要
- KV Cache 基准测试：不同序列长度（10/20/50/100/200/500）下有/无 cache 的推理耗时对比及加速比
- 累计 K/V 投影 token 计数：无缓存 n(n+1)/2，有缓存 n；这不是 FLOPs，也不能直接换算成耗时加速比
- 固定输入上，程序会比较两分支末 token 输出；未运行前不能宣称检查已通过
- 量化统计：原始 FP32 权重的 min/max/mean/std，量化后 INT8 值的统计
- 压缩率：FP32 (4 bytes) vs INT8 (1 byte)，理论压缩比 4x
- 量化误差：MSE（均方误差）和 MAE（平均绝对误差）数值展示

### 生成图表

#### 图表 1: KV Cache 性能对比
旧文件 `images/kv_cache_comparison.png` 暂不展示，因为旧缓存分支仍重复计算整个前缀。修正后重新运行才能生成有效图：左图为实测累计耗时；右图为累计 K/V 投影 token 数，不是加速比。

#### 图表 2: 量化演示
旧文件 `images/quantization_demo.png` 暂不展示，因为旧量化公式在单侧分布和非零常量行上存在偏移错误。修正后应核对原始权重分布、还原值与原值散点、逐通道误差比较及权重存储大小。

#### 图片资源: 概念图解
- `24-01-kv-cache.png` -- KV Cache 原理：Transformer 自回归生成中 K/V 矩阵的缓存复用机制
- `24-02-flash-attention.png` -- Flash Attention：通过分块（tiling）和重计算减少 HBM 访问的 IO-aware 优化
- `24-03-quantization-comparison.png` -- 量化方案对比：对称/非对称、per-tensor/per-channel、PTQ/QAT 等
- `24-04-paged-attention.png` -- Paged Attention（vLLM 核心技术）：将 KV Cache 按页管理以提高显存利用率

## 代码结构
- `class SimpleAttention` -- 简单注意力机制，支持有/无 KV Cache 两种生成模式
  - `_single_head_attention()` -- 单头注意力 QK^T * V 计算
  - `generate_without_kv_cache()` -- 无缓存：每步重投影前缀；累计投影 token 数 O(n^2)
  - `generate_with_kv_cache()` -- 有缓存：只计算新 token 的 K/V，拼接已有缓存；累计投影 token 数 O(n)
- `demo_kv_cache()` -- 基准测试：不同序列长度下有/无 cache 的耗时和加速比
- `quantize_fp32_to_int8()` -- 实际为 uint8 仿射量化，q = clip(round(W/s) + z, 0, 255)
- `demo_quantization()` -- per-tensor vs per-channel 的量化误差对比
- `dequantize_int8_to_fp32()` -- W_hat = s * (q - z)
- 图表由 `demo_kv_cache()` 和 `demo_quantization()` 内部保存
- `main()` -- 主流程

## 运行环境
- Python 依赖: numpy, matplotlib
- 硬件需求: CPU 即可
- 预计运行时间: ~10-20 秒
