---
title: "as03 Neural Operator 与 FNO — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# as03 Neural Operator 与 FNO — demo.py 代码详解

<a href="/notebook/code/science/fno/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/science/fno/code
python demo.py
```

本 demo 依赖 PyTorch（CPU 即可）。教学默认配置约 120 轮训练，普通笔记本通常一两分钟内可跑完；若需更低误差，可增大 `n_epochs` / `n_train` / `width`。

## 代码逐段详解

### 第1步：解析构造算子学习数据集

```python
def generate_dataset(n_samples, grid_size, n_modes=16, decay=1.3, rng=None):
    x_grid = np.linspace(0, 1, grid_size)
    k = np.arange(1, n_modes + 1)
    basis = np.sin(np.outer(k, np.pi * x_grid))   # (K, N)

    coeff_scale = 1.0 / (k ** decay)
    C = rng.standard_normal((n_samples, n_modes)) * coeff_scale

    A = C @ basis                                 # 源项 a(x)
    B = C / (k * np.pi) ** 2
    U = B @ basis                                 # 解析解 u(x)
    return x_grid, A, U
```

**关键洞察**：

- 每个样本是一整条函数（长度 $N$ 的向量），不是单个数字——这就是算子学习的数据结构
- `1/k^decay` 让高频振幅衰减，生成平滑的随机函数（类似一维高斯随机场的简化版）
- 解析解系数 $b_k = c_k/(k\pi)^2$ 来自特征函数性质：$-\dfrac{d^2}{dx^2}\sin(k\pi x) = (k\pi)^2\sin(k\pi x)$

完全不需要调用有限差分/有限元求解器，就能得到精确的 $(a, u)$ 训练对。

### 第2步：谱卷积层 —— FNO 的核心

```python
class SpectralConv1d(nn.Module):
    def forward(self, x):
        # x: (batch, in_channels, N)
        x_ft = torch.fft.rfft(x, dim=-1)          # 实数 FFT -> 频域
        out_ft = torch.zeros(..., dtype=torch.cfloat, device=x.device)
        m = min(self.modes, x_ft.shape[-1])
        # 逐模式复数矩阵乘: 对每个频率 k 做线性变换
        out_ft[:, :, :m] = torch.einsum(
            'bik,iok->bok', x_ft[:, :, :m], self.weight[:, :, :m]
        )
        return torch.fft.irfft(out_ft, n=n, dim=-1)
```

**卷积定理的直接应用**：时域卷积 = 频域逐点相乘。FNO 把卷积核参数化成频域的复数权重 $R$，只对最低的 `modes` 个频率学习变换，更高频率直接截断为 0。

**分辨率不变性的来源**：`self.weight` 的形状是 `(in_ch, out_ch, modes)`，与网格点数 $N$ 无关。换一个 $N$，`rfft`/`irfft` 自动适配，同一套权重直接可用。

### 第3步：Fourier Layer = 谱卷积 + 逐点旁路 + 激活

```python
class FNOBlock1d(nn.Module):
    def forward(self, x):
        return F.gelu(self.spectral_conv(x) + self.pointwise(x))
```

对应公式：

$$
h_{l+1}(x) = \sigma\big(\mathcal{F}^{-1}(R\cdot\mathcal{F}(h_l))(x) + W h_l(x)\big)
$$

- **谱卷积**：全局感受野，建模低频远距离依赖
- **1×1 卷积旁路 $W$**：保留局部逐点信息，弥补高频截断损失
- **GELU**：平滑非线性，让多层堆叠能逼近非线性算子

### 第4步：完整 FNO —— Lift → Layers → Project

```python
class FNO1d(nn.Module):
    def forward(self, a, grid):
        x = torch.stack([a, grid], dim=-1)   # 拼接 [a(x), x]
        x = self.fc0(x)                       # Lift: 2 -> width
        x = x.permute(0, 2, 1)                # (batch, width, N)
        for block in self.blocks:
            x = block(x)
        x = x.permute(0, 2, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)                       # Project: width -> 1
        return x.squeeze(-1)
```

**为什么把坐标 $x$ 也拼进输入**：纯谱卷积对空间平移近似等变，缺乏绝对位置感知；把网格坐标作为额外通道，帮助网络知道"自己在定义域的什么位置"（尤其对非周期边界条件很有用）。

### 第5步：训练 —— 普通监督学习，但数据是函数对

```python
u_pred = model(a_batch, grid_batch)
loss = F.mse_loss(u_pred, u_batch)
```

与 PINN 的对比：

| | PINN | FNO |
|--|------|-----|
| 损失 | PDE 残差 + BC | 输出函数与真解的 MSE |
| 需要标注解？ | 不需要 | 需要 |
| 一个模型覆盖？ | 单个 $a$ | 一族 $a$ |

### 第6步：分辨率不变性演示

```python
# 模型只在 grid_size=64 上训练过
# 直接在 grid_size=192 上推理——同一套权重，无需重训
x_grid, A, U = generate_dataset(1, N=192, ...)
u_pred = model(a_t, grid_t)
```

这是 FNO 相对普通 CNN / MLP 的杀手级特性：参数定义在频率空间，与空间网格解耦。

### 关键概念速查表

| 概念 | 一句话解释 | 代码位置 |
|------|-----------|---------|
| 算子学习 | 学习函数→函数的映射 $\mathcal{G}:a\mapsto u$ | `generate_dataset()` / `FNO1d` |
| 谱卷积 | FFT → 截断 → 复数权重相乘 → IFFT | `SpectralConv1d` |
| `modes` | 保留的最低频率数，控制容量与平滑度 | `SpectralConv1d.__init__` |
| 逐点旁路 | 1×1 卷积保留局部信息 | `FNOBlock1d.pointwise` |
| 分辨率不变性 | 参数与 $N$ 无关，换网格可直接推理 | `plot_resolution_invariance()` |
| Lift / Project | 进出高维特征空间的线性层 | `FNO1d.fc0` / `fc1,fc2` |

## 完整代码

<<< @/science/fno/code/demo.py
