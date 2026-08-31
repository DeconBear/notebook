---
title: "as03 Neural Operator 与 FNO — exercise.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# as03 Neural Operator 与 FNO — exercise.py 练习指南

<a href="/notebook/code/science/fno/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过补全频域逐模式相乘、解析 Poisson 算子数据生成和最小 Fourier Layer 三个模块，从代码层面掌握 FNO 的核心计算单元。

## 预备知识

在开始练习前，确保你已经理解了以下概念（参见 [demo.py 代码详解](./code-demo)）：

- 卷积定理：时域卷积 = 频域逐点相乘
- 解析构造：若 $a=\sum c_k\sin(k\pi x)$，则 $u=\sum\frac{c_k}{(k\pi)^2}\sin(k\pi x)$
- Fourier Layer：$h_{l+1}=\sigma(\text{SpectralConv}(h_l)+W h_l)$

## 任务清单

### 任务1：实现频域逐模式相乘 `spectral_multiply(x_ft, weight, modes)`

- **算法步骤**：
  1. 创建全零的 `out_ft`，形状 `(batch, out_channels, n_freqs)`，dtype 为 `torch.cfloat`
  2. `m = min(modes, n_freqs, weight.shape[2])`
  3. 用 `torch.einsum('bik,iok->bok', x_ft[:,:,:m], weight[:,:,:m])` 做逐模式矩阵乘
  4. 把结果写入 `out_ft[:,:,:m]`（更高频率保持为 0，即截断）
- **直觉**：对每个保留的频率 $k$，把输入的 `in_channels` 维复向量线性变换成 `out_channels` 维复向量

### 任务2：实现解析 Poisson 对生成 `generate_poisson_pair(coeffs, x_grid)`

- **用到的公式**：
  - $a(x)=\sum_k c_k\sin(k\pi x)$
  - $u(x)=\sum_k \frac{c_k}{(k\pi)^2}\sin(k\pi x)$
- **实现步骤**：
  1. `k = np.arange(1, K+1)`，`basis = np.sin(np.outer(k, np.pi * x_grid))`
  2. `a = coeffs @ basis`，`b = coeffs / (k * np.pi)**2`，`u = b @ basis`
  3. 若原始 `coeffs` 是一维，把输出压缩回一维
- **验证技巧**：用有限差分检查 $-\hat u''\approx a$，且 $u(0)=u(1)=0$

### 任务3（Bonus）：实现 `MiniFourierLayer.forward`

- **算法步骤**：
  1. `x_ft = torch.fft.rfft(x, dim=-1)`
  2. `out_ft = spectral_multiply(x_ft, self.weight, self.modes)`
  3. `x_spec = torch.fft.irfft(out_ft, n=x.shape[-1], dim=-1)`
  4. `return F.gelu(x_spec + self.pointwise(x))`
- **关键细节**：`irfft` 的 `n=` 参数必须指定为原始空间长度，否则长度可能对不齐

## 验证标准

运行 `python exercise.py`：

1. `test_spectral_multiply()`：输出形状正确，且 `modes` 之后的频率分量为 0
2. `test_poisson_pair()`：边界条件满足，$ -u''\approx a$ 的残差足够小
3. `test_mini_fourier_layer()`（Bonus）：前向形状正确，且能正常 `backward()`

## 延伸思考

- 如果把 `modes` 设得非常小（比如 2），FNO 还能学好高频源项对应的解吗？为什么？
- 分辨率不变性是否意味着"任意高分辨率都精确"？截断高频会带来什么误差？
- 对比 as02 的 PINN：如果只有很少的 $(a,u)$ 对，但知道 PDE，你会优先选 PINN、FNO，还是下一章的 PINO？

## 完整代码

<<< @/science/fno/code/exercise.py
