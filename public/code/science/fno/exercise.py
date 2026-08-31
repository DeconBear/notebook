# -*- coding: utf-8 -*-
"""
===============================================================================
as03_fno/code/exercise.py — Fourier Neural Operator 练习
===============================================================================
本练习文件中，你需要完成以下任务：

练习目标：
  1. 实现谱卷积中的"频域逐模式相乘"（FNO 的核心计算）
  2. 实现解析 Poisson 算子数据集生成器（制造解方法在算子学习中的版本）
  3. 实现一个最小的 Fourier Layer 前向传播（Bonus）

提示：
  - 时域卷积 = 频域逐点相乘（卷积定理）
  - 若 a(x)=sum c_k sin(k*pi*x)，则 u(x)=sum (c_k/(k*pi)^2) sin(k*pi*x)
  - Fourier Layer: h_next = activation( SpectralConv(h) + W(h) )

运行方式：
  python exercise.py
===============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================================
# 任务 1: 实现频域逐模式相乘 (约 8 行)
# ============================================================================

def spectral_multiply(x_ft, weight, modes):
    """
    对傅立叶变换后的输入，用可学习复数权重做逐模式线性变换。

    参数:
        x_ft: (batch, in_channels, n_freqs) 复数 tensor，rfft 的输出
        weight: (in_channels, out_channels, modes) 复数权重
        modes: int，保留的最低频率分量个数
    返回:
        out_ft: (batch, out_channels, n_freqs) 复数 tensor，
                前 modes 个频率被变换，其余频率保持为 0
    """
    batch, in_ch, n_freqs = x_ft.shape
    out_ch = weight.shape[1]
    out_ft = torch.zeros(batch, out_ch, n_freqs, dtype=torch.cfloat, device=x_ft.device)

    # TODO: 完成以下步骤
    # 1. m = min(modes, n_freqs, weight.shape[2])
    # 2. 用 einsum 对前 m 个频率做: (batch, in_ch, m) x (in_ch, out_ch, m) -> (batch, out_ch, m)
    #    公式: 'bik,iok->bok'
    # 3. 把结果写入 out_ft[:, :, :m]
    # --- BEGIN YOUR CODE ---
    m = min(modes, n_freqs, weight.shape[2])
    out_ft[:, :, :m] = torch.einsum(
        'bik,iok->bok', x_ft[:, :, :m], weight[:, :, :m]
    )
    # --- END YOUR CODE ---
    # ^^^ TODO: 请删除上面的实现代码，自己重写 ^^^
    return out_ft


# ============================================================================
# 任务 2: 实现解析 Poisson 算子数据生成 (约 10 行)
# ============================================================================

def generate_poisson_pair(coeffs, x_grid):
    """
    给定正弦系数 coeffs 和网格 x_grid，解析构造 (a, u) 函数对。

    a(x) = sum_{k=1}^{K} c_k * sin(k*pi*x)
    u(x) = sum_{k=1}^{K} (c_k / (k*pi)^2) * sin(k*pi*x)

    参数:
        coeffs: (K,) 或 (n_samples, K) 的 numpy 数组，正弦系数
        x_grid: (N,) 网格坐标，范围 [0, 1]
    返回:
        a: 与 coeffs 对应的源项，形状 (N,) 或 (n_samples, N)
        u: 对应的解析解，形状同 a
    """
    coeffs = np.atleast_2d(coeffs)          # (n_samples, K)
    K = coeffs.shape[1]
    # TODO: 完成以下步骤
    # 1. k = np.arange(1, K+1)
    # 2. basis = sin(outer(k, pi * x_grid))，形状 (K, N)
    # 3. a = coeffs @ basis
    # 4. b = coeffs / (k * pi)^2
    # 5. u = b @ basis
    # 6. 如果原始 coeffs 是 1D，把 a, u 压缩回 1D
    # --- BEGIN YOUR CODE ---
    k = np.arange(1, K + 1)
    basis = np.sin(np.outer(k, np.pi * x_grid))   # (K, N)
    a = coeffs @ basis
    b = coeffs / (k * np.pi) ** 2
    u = b @ basis
    if a.shape[0] == 1:
        a, u = a[0], u[0]
    # --- END YOUR CODE ---
    # ^^^ TODO: 请删除上面的实现代码，自己重写 ^^^
    return a, u


# ============================================================================
# 任务 3 (Bonus): 实现最小 Fourier Layer 前向 (约 8 行)
# ============================================================================

class MiniFourierLayer(nn.Module):
    """
    一个极简的 Fourier Layer:
      h_next = GELU( SpectralConv(h) + PointwiseLinear(h) )

    这里 SpectralConv 用任务1的 spectral_multiply + rfft/irfft 实现，
    PointwiseLinear 用 1x1 卷积 (nn.Conv1d(..., kernel_size=1)) 实现。
    """

    def __init__(self, width=8, modes=4):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (width * width)
        self.weight = nn.Parameter(scale * torch.rand(width, width, modes, dtype=torch.cfloat))
        self.pointwise = nn.Conv1d(width, width, kernel_size=1)

    def forward(self, x):
        """
        x: (batch, width, N)
        返回: (batch, width, N)
        """
        # TODO: 完成以下步骤
        # 1. x_ft = torch.fft.rfft(x, dim=-1)
        # 2. out_ft = spectral_multiply(x_ft, self.weight, self.modes)
        # 3. x_spec = torch.fft.irfft(out_ft, n=x.shape[-1], dim=-1)
        # 4. return F.gelu(x_spec + self.pointwise(x))
        # --- BEGIN YOUR CODE ---
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = spectral_multiply(x_ft, self.weight, self.modes)
        x_spec = torch.fft.irfft(out_ft, n=x.shape[-1], dim=-1)
        return F.gelu(x_spec + self.pointwise(x))
        # --- END YOUR CODE ---
        # ^^^ TODO: 请删除上面的实现代码，自己重写 ^^^


# ============================================================================
# 验证代码
# ============================================================================

def test_spectral_multiply():
    """验证频域逐模式相乘的形状与基本性质。"""
    print("[测试 1] 频域逐模式相乘...")
    batch, in_ch, out_ch, n, modes = 2, 3, 4, 16, 5
    x = torch.randn(batch, in_ch, n)
    x_ft = torch.fft.rfft(x, dim=-1)
    weight = torch.randn(in_ch, out_ch, modes, dtype=torch.cfloat)
    out_ft = spectral_multiply(x_ft, weight, modes)

    assert out_ft.shape == (batch, out_ch, x_ft.shape[-1]), \
        f"输出形状错误: {out_ft.shape}"
    # modes 之后的频率应全为 0
    assert torch.allclose(out_ft[:, :, modes:].abs(), torch.tensor(0.0)), \
        "modes 之后的频率应被截断为 0"
    print(f"  [PASS] 形状={tuple(out_ft.shape)}, 高频截断正确")


def test_poisson_pair():
    """验证解析 Poisson 对: 用有限差分检查 -u'' ≈ a。"""
    print("[测试 2] 解析 Poisson 算子数据生成...")
    x = np.linspace(0, 1, 201)
    coeffs = np.array([1.0, 0.5, -0.3])
    a, u = generate_poisson_pair(coeffs, x)

    assert a.shape == (201,) and u.shape == (201,), f"形状错误: {a.shape}, {u.shape}"
    # 边界条件
    assert abs(u[0]) < 1e-10 and abs(u[-1]) < 1e-10, "解应满足 u(0)=u(1)=0"

    # 有限差分检查 -u'' ≈ a
    dx = x[1] - x[0]
    d2u = (u[:-2] - 2 * u[1:-1] + u[2:]) / dx ** 2
    residual = -d2u - a[1:-1]
    rms = np.sqrt(np.mean(residual ** 2))
    assert rms < 0.05, f"解析对不满足 PDE: RMS residual={rms}"
    print(f"  [PASS] 边界条件满足, PDE 残差 RMS={rms:.6f}")


def test_mini_fourier_layer():
    """(Bonus) 验证 MiniFourierLayer 前向传播形状正确且可反向传播。"""
    print("[测试 3 (Bonus)] Mini Fourier Layer...")
    layer = MiniFourierLayer(width=8, modes=4)
    x = torch.randn(2, 8, 32, requires_grad=True)
    y = layer(x)
    assert y.shape == x.shape, f"输出形状错误: {y.shape}"
    loss = y.pow(2).mean()
    loss.backward()
    assert x.grad is not None, "反向传播失败"
    print(f"  [PASS] 前向形状={tuple(y.shape)}, 反向传播正常")


if __name__ == "__main__":
    print("=" * 60)
    print("as03_fno exercise.py — FNO 练习")
    print("=" * 60)

    try:
        test_spectral_multiply()
        test_poisson_pair()
        test_mini_fourier_layer()
        print("\n" + "=" * 60)
        print("所有测试通过!")
        print("=" * 60)
    except Exception as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
