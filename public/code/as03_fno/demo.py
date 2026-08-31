# -*- coding: utf-8 -*-
"""
===============================================================================
as03_fno/code/demo.py — 傅立叶神经算子 (FNO) 学习一维 Poisson 算子
===============================================================================
本演示训练一个一维 Fourier Neural Operator (FNO)，学习"源项函数 a(x) ->
解函数 u(x)"这个算子映射，对应的 PDE 仍然是前两章的一维 Poisson 方程：

    -u''(x) = a(x),  x in (0, 1),  u(0) = u(1) = 0

与 as02 的 PINN 不同：PINN 训练一次只能得到"一个特定 a(x) 对应的一个 u(x)"，
换一个新的 a(x) 就要重新训练。FNO 训练完成后，可以直接对**任意新的 a(x)**
做一次前向传播就得到对应的 u(x)——这就是"学习算子"和"学习单个解"的本质区别。

技巧: 我们用正弦基函数解析构造数据集，完全不需要调用数值 PDE 求解器：
  若 a(x) = sum_k c_k sin(k*pi*x)，则解析解为 u(x) = sum_k (c_k/(k*pi)^2) sin(k*pi*x)
  这是因为 sin(k*pi*x) 本身就是 -d^2/dx^2 算子（配合 Dirichlet 边界条件）的
  特征函数，特征值为 (k*pi)^2 —— 这也正是 FNO 用"逐模式复数权重"来逼近的东西:
  FNO 的谱卷积在做的事，本质上和"除以 (k*pi)^2"这种频域算子是同一类操作。

通过本演示，你将理解：
  - 什么是"算子学习"：数据是"函数对" (a, u)，而不是"数字对" (x, y)
  - FNO 的核心构件——谱卷积层 (Spectral Convolution): FFT -> 截断高频模式
    -> 逐模式复数权重相乘 -> IFFT -> 加上逐点线性旁路 -> 非线性激活
  - 分辨率不变性 (resolution invariance)：同一套训练好的权重，可以直接用在
    与训练时不同的网格分辨率上，无需重新训练

作者：notebook 项目
日期：2025
===============================================================================
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

matplotlib.rcParams['axes.unicode_minus'] = False

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device('cpu')


def _save_path(filename):
    return os.path.join(_IMAGES_DIR, filename)


# ============================================================================
# 第一部分：解析构造数据集 (无需数值 PDE 求解器)
# ============================================================================

def generate_dataset(n_samples, grid_size, n_modes=16, decay=1.3, rng=None):
    """
    解析生成一批 (a(x), u(x)) 函数对样本。

    随机源项: a(x) = sum_{k=1}^{n_modes} c_k * sin(k*pi*x)，
      系数 c_k ~ N(0,1) / k^decay —— 除以 k^decay 让高频分量的振幅衰减，
      生成的是平滑的随机函数（类似"高斯随机场"的一维简化版本）。

    解析解: u(x) = sum_{k=1}^{n_modes} (c_k / (k*pi)^2) * sin(k*pi*x)
      因为 -u'' = sum_k c_k*(k*pi)^2*sin(k*pi*x)，要求这等于 a(x)=sum_k c_k*sin(k*pi*x)
      需要把 u 的系数设为 b_k = c_k/(k*pi)^2，见模块开头的推导。

    参数:
        n_samples: 样本数量
        grid_size: 空间网格点数 N（网格 x 在 [0, 1] 上均匀取 N 个点）
        n_modes: 用于生成随机函数的正弦模式数
        decay: 高频衰减指数，越大生成的函数越平滑
        rng: numpy.random.Generator，若为 None 则用全局 np.random
    返回:
        x_grid: (N,) 网格坐标
        A: (n_samples, N) 输入函数 a(x) 在网格上的取值
        U: (n_samples, N) 解析解 u(x) 在网格上的取值
    """
    if rng is None:
        rng = np.random
    x_grid = np.linspace(0, 1, grid_size)
    k = np.arange(1, n_modes + 1)                    # (n_modes,)
    basis = np.sin(np.outer(k, np.pi * x_grid))        # (n_modes, N): sin(k*pi*x)

    coeff_scale = 1.0 / (k ** decay)                   # (n_modes,) 高频衰减
    C = rng.standard_normal((n_samples, n_modes)) * coeff_scale[None, :]  # (n_samples, n_modes)

    A = C @ basis                                       # (n_samples, N)
    B = C / (k * np.pi) ** 2                             # 解的傅立叶系数 b_k = c_k/(k*pi)^2
    U = B @ basis                                        # (n_samples, N)
    return x_grid, A.astype(np.float32), U.astype(np.float32)


# ============================================================================
# 第二部分：一维 FNO 架构
# ============================================================================

class SpectralConv1d(nn.Module):
    """
    一维谱卷积层：FNO 的核心构件。

    做法:
      1. 对输入沿空间维做实数快速傅立叶变换 (rfft)，得到频域表示
      2. 只保留最低的 `modes` 个频率分量（截断高频——这是一种低通滤波，
         也是一种隐式的正则化：真实物理场往往能量集中在低频）
      3. 用一组**可学习的复数权重**逐模式相乘（相当于对每个频率做一次
         独立的线性变换），这就是"卷积定理"在频域的直接应用：
         时域卷积 = 频域逐点相乘
      4. 做逆变换 (irfft) 变回空间域

    关键性质——分辨率不变性: 这一层的可学习参数量只与 `modes` 有关，
    与输入网格点数 N 完全无关！所以同一层可以直接应用到任意分辨率的输入上。
    """

    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        # 复数权重: (in_channels, out_channels, modes)
        self.weight = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x):
        """x: (batch, in_channels, N) -> (batch, out_channels, N)"""
        batch_size, _, n = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)                      # (batch, in_ch, N//2+1)

        out_ft = torch.zeros(
            batch_size, self.out_channels, n // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        m = min(self.modes, x_ft.shape[-1])
        # 逐模式复数矩阵乘法: 对每个保留的频率 k，做 (in_ch,) @ (in_ch, out_ch) -> (out_ch,)
        out_ft[:, :, :m] = torch.einsum('bik,iok->bok', x_ft[:, :, :m], self.weight[:, :, :m])

        x_out = torch.fft.irfft(out_ft, n=n, dim=-1)          # (batch, out_ch, N)
        return x_out


class FNOBlock1d(nn.Module):
    """
    一个完整的 Fourier Layer: 谱卷积 (全局、频域) + 逐点线性旁路 (局部、空域) + 激活。

    公式: h_{l+1}(x) = sigma( SpectralConv(h_l)(x) + W h_l(x) )
    其中 W 是一个 1x1 卷积（即逐点线性变换，等价于共享权重的全连接层）。

    为什么需要旁路 W: 谱卷积只能建模"全局"的频域交互，1x1 卷积负责保留
    "局部"的逐点非线性信息通道，两者结合让 FNO 兼具全局感受野和局部灵活性。
    """

    def __init__(self, width, modes):
        super().__init__()
        self.spectral_conv = SpectralConv1d(width, width, modes)
        self.pointwise = nn.Conv1d(width, width, kernel_size=1)

    def forward(self, x):
        return F.gelu(self.spectral_conv(x) + self.pointwise(x))


class FNO1d(nn.Module):
    """
    完整的一维 FNO: Lift -> 若干 Fourier Layer -> Project

    输入通道: [a(x_i), x_i] —— 把网格坐标也作为一个输入通道拼接进去，
    这是 FNO 原论文的标准做法，帮助网络感知位置信息（否则谱卷积对空间位移
    是等变的，缺乏绝对位置感知能力）。
    """

    def __init__(self, modes=16, width=32, n_layers=4):
        super().__init__()
        self.fc0 = nn.Linear(2, width)     # 提升维度: [a(x), x] -> width 通道
        self.blocks = nn.ModuleList([FNOBlock1d(width, modes) for _ in range(n_layers)])
        self.fc1 = nn.Linear(width, 64)
        self.fc2 = nn.Linear(64, 1)         # 投影回标量输出 u(x)

    def forward(self, a, grid):
        """
        a: (batch, N) 输入函数取值
        grid: (batch, N) 网格坐标 (通常和 a 共享同一个网格)
        返回: (batch, N) 预测的输出函数取值
        """
        x = torch.stack([a, grid], dim=-1)          # (batch, N, 2)
        x = self.fc0(x)                                # (batch, N, width)
        x = x.permute(0, 2, 1)                          # (batch, width, N) —— 卷积层需要通道维在前
        for block in self.blocks:
            x = block(x)
        x = x.permute(0, 2, 1)                          # (batch, N, width)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)                                 # (batch, N, 1)
        return x.squeeze(-1)                            # (batch, N)


# ============================================================================
# 第三部分：训练
# ============================================================================

def train_fno(n_train=800, n_test=200, grid_size=64, n_modes_data=16,
              modes=16, width=32, n_layers=4, n_epochs=300, lr=1e-3, batch_size=32):
    """训练 FNO 学习 a(x) -> u(x) 算子映射。"""
    x_grid, A_train, U_train = generate_dataset(n_train, grid_size, n_modes_data)
    _, A_test, U_test = generate_dataset(n_test, grid_size, n_modes_data)

    x_grid_t = torch.tensor(x_grid, dtype=torch.float32)
    A_train_t = torch.tensor(A_train)
    U_train_t = torch.tensor(U_train)
    A_test_t = torch.tensor(A_test)
    U_test_t = torch.tensor(U_test)

    model = FNO1d(modes=modes, width=width, n_layers=n_layers).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    n_batches = n_train // batch_size
    history = {'train_loss': [], 'test_loss': []}

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            a_batch = A_train_t[idx]
            u_batch = U_train_t[idx]
            grid_batch = x_grid_t.unsqueeze(0).expand(a_batch.shape[0], -1)

            optimizer.zero_grad()
            u_pred = model(a_batch, grid_batch)
            loss = F.mse_loss(u_pred, u_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            grid_test = x_grid_t.unsqueeze(0).expand(n_test, -1)
            u_test_pred = model(A_test_t, grid_test)
            test_loss = F.mse_loss(u_test_pred, U_test_t).item()

        history['train_loss'].append(epoch_loss / n_batches)
        history['test_loss'].append(test_loss)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1:4d}/{n_epochs} | "
                  f"train_mse={history['train_loss'][-1]:.3e} | test_mse={test_loss:.3e}",
                  flush=True)

    return model, history, (x_grid, A_test, U_test)


# ============================================================================
# 第四部分：可视化
# ============================================================================

def plot_predictions(model, test_data, n_show=3):
    """展示若干测试样本的输入 a(x)、预测 u_hat(x) 与真解 u(x) 对比。"""
    x_grid, A_test, U_test = test_data
    x_grid_t = torch.tensor(x_grid, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        a_show = torch.tensor(A_test[:n_show])
        grid_show = x_grid_t.unsqueeze(0).expand(n_show, -1)
        u_pred = model(a_show, grid_show).numpy()

    fig, axes = plt.subplots(2, n_show, figsize=(5 * n_show, 8))
    for i in range(n_show):
        rel_err = np.linalg.norm(u_pred[i] - U_test[i]) / (np.linalg.norm(U_test[i]) + 1e-10)

        axes[0, i].plot(x_grid, A_test[i], color='tab:blue', linewidth=2)
        axes[0, i].set_title(f'Sample {i+1}: input a(x)', fontsize=11)
        axes[0, i].set_xlabel('x')
        axes[0, i].set_ylabel('a(x)')
        axes[0, i].grid(True, alpha=0.3)

        axes[1, i].plot(x_grid, U_test[i], 'k-', linewidth=2.5, label='Exact u(x)')
        axes[1, i].plot(x_grid, u_pred[i], 'r--', linewidth=2, label='FNO prediction')
        axes[1, i].set_title(f'Output u(x), rel. L2 err = {rel_err:.3%}', fontsize=11)
        axes[1, i].set_xlabel('x')
        axes[1, i].set_ylabel('u(x)')
        axes[1, i].legend(fontsize=9)
        axes[1, i].grid(True, alpha=0.3)

    fig.suptitle('FNO learns the operator a(x) -> u(x)  (trained at grid_size=64)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fp = _save_path('fno_prediction.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Done] FNO 预测对比图已保存: {fp}")


def plot_loss_curve(history):
    """训练/测试损失曲线。"""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    epochs = np.arange(1, len(history['train_loss']) + 1)
    ax.semilogy(epochs, history['train_loss'], label='Train MSE', linewidth=2)
    ax.semilogy(epochs, history['test_loss'], label='Test MSE', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE (log scale)')
    ax.set_title('FNO Training Curve', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    fp = _save_path('fno_loss.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Done] FNO 损失曲线已保存: {fp}")


def plot_resolution_invariance(model, n_modes_data=16):
    """
    核心亮点演示: 分辨率不变性。

    用同一个（在 grid_size=64 上训练好的）模型，直接在一个更高分辨率
    (grid_size=192) 的新样本上做预测——不需要任何重新训练或插值。
    因为 SpectralConv1d 只在频域截断到固定的 `modes` 个模式，
    这个操作对输入的空间分辨率是无感的。
    """
    resolutions = [64, 192]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, N in zip(axes, resolutions):
        # 用同一个随机种子重新生成，保证两个分辨率下采样的是"同一个函数"
        rng_fixed = np.random.default_rng(123)
        x_grid, A, U = generate_dataset(1, N, n_modes_data, rng=rng_fixed)

        model.eval()
        with torch.no_grad():
            a_t = torch.tensor(A, dtype=torch.float32)
            grid_t = torch.tensor(x_grid, dtype=torch.float32).unsqueeze(0)
            u_pred = model(a_t, grid_t).numpy()[0]

        rel_err = np.linalg.norm(u_pred - U[0]) / (np.linalg.norm(U[0]) + 1e-10)
        ax.plot(x_grid, U[0], 'k-', linewidth=2.5, label='Exact u(x)')
        ax.plot(x_grid, u_pred, 'r--', linewidth=2, label='FNO prediction (same weights)')
        ax.set_title(f'Grid size = {N}\nrel. L2 error = {rel_err:.3%}', fontsize=12, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('u(x)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Resolution Invariance: same trained weights, different grid resolutions\n'
                 '(model was only ever trained at grid_size=64)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fp = _save_path('fno_resolution_invariance.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Done] 分辨率不变性演示图已保存: {fp}")


# ============================================================================
# 第五部分：Fourier Layer 概念示意图
# ============================================================================

def _draw_box(ax, xy, w, h, text, facecolor, fontsize=9.5):
    box = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                          linewidth=1.5, edgecolor='black', facecolor=facecolor, alpha=0.92)
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha='center', va='center',
             fontsize=fontsize, fontweight='bold')


def _draw_arrow(ax, start, end, color='black', style='-|>'):
    arrow = FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=16,
                             linewidth=1.6, color=color)
    ax.add_patch(arrow)


def plot_fno_layer_diagram():
    """
    Fourier Layer 概念示意图:
      input h_l(x) --FFT--> frequency domain --truncate to `modes`-->
      multiply learnable complex weight R --IFFT--> spatial domain
      + pointwise linear W(h_l) (skip branch) --> activation --> h_{l+1}(x)
    下方再画出多层 Fourier Layer 堆叠 + lift/project 构成完整 FNO 的示意。
    """
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 10), gridspec_kw={'height_ratios': [1.3, 1]})

    # ---- 上图: 单个 Fourier Layer 内部结构 ----
    ax = axes[0]
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 6.5)
    ax.axis('off')
    ax.text(6.25, 6.1, 'Inside One Fourier Layer', ha='center', fontsize=14, fontweight='bold')

    _draw_box(ax, (0.2, 3.6), 1.8, 1.0, 'Input\nh_l(x)', '#dbe9f7')
    _draw_arrow(ax, (2.0, 4.1), (2.6, 4.1))
    _draw_box(ax, (2.6, 3.6), 1.9, 1.0, 'FFT\n(rfft)', '#ffe8b3')
    _draw_arrow(ax, (4.5, 4.1), (5.1, 4.1))
    _draw_box(ax, (5.1, 3.6), 2.3, 1.0, 'Truncate to\nlowest `modes`', '#ffe8b3')
    _draw_arrow(ax, (7.4, 4.1), (8.0, 4.1))
    _draw_box(ax, (8.0, 3.6), 2.1, 1.0, 'Multiply R\n(learnable, complex)', '#d9f0d3')
    _draw_arrow(ax, (10.1, 4.1), (10.7, 4.1))
    _draw_box(ax, (10.7, 3.6), 1.6, 1.0, 'IFFT', '#ffe8b3')

    _draw_arrow(ax, (11.5, 3.6), (11.5, 2.0))
    _draw_arrow(ax, (1.1, 3.6), (1.1, 2.0))
    _draw_box(ax, (1.5, 1.0), 3.0, 1.0, 'Pointwise linear W\n(1x1 conv, "skip" branch)', '#e6d9f5')
    _draw_arrow(ax, (3.0, 1.0), (3.0, 0.5))
    _draw_arrow(ax, (1.1, 2.0), (3.0, 1.5))
    _draw_arrow(ax, (11.5, 2.0), (7.0, 1.5))
    _draw_box(ax, (4.9, 0.1), 4.0, 0.9, 'Sum + GELU activation\n= h_{l+1}(x)', '#fef1b0')
    _draw_arrow(ax, (3.4, 1.0), (4.9, 0.5))
    _draw_arrow(ax, (9.4, 1.0), (8.9, 0.5))
    _draw_box(ax, (7.9, 1.0), 3.0, 1.0, '(same h_l(x)\nvia skip connection)', '#e6d9f5')

    ax.text(6.25, 5.3,
            'Key property: parameter count of R depends only on `modes`, NOT on grid size N\n'
            '--> the same layer works at ANY spatial resolution (resolution invariance)',
            ha='center', fontsize=10, style='italic')

    # ---- 下图: 完整 FNO 堆叠结构 ----
    ax2 = axes[1]
    ax2.set_xlim(0, 12.5)
    ax2.set_ylim(0, 3.2)
    ax2.axis('off')
    ax2.text(6.25, 2.9, 'Full FNO: Lift -> Stack of Fourier Layers -> Project',
              ha='center', fontsize=13, fontweight='bold')

    _draw_box(ax2, (0.2, 0.9), 1.9, 1.1, 'Input\n[a(x), x]', '#dbe9f7')
    _draw_arrow(ax2, (2.1, 1.45), (2.6, 1.45))
    _draw_box(ax2, (2.6, 0.9), 1.7, 1.1, 'Lift\n(Linear 2->width)', '#ffe8b3')
    _draw_arrow(ax2, (4.3, 1.45), (4.8, 1.45))
    _draw_box(ax2, (4.8, 0.9), 4.4, 1.1, 'Fourier Layer x N\n(spectral conv + skip + GELU)', '#d9f0d3')
    _draw_arrow(ax2, (9.2, 1.45), (9.7, 1.45))
    _draw_box(ax2, (9.7, 0.9), 1.7, 1.1, 'Project\n(Linear width->1)', '#ffe8b3')
    _draw_arrow(ax2, (11.4, 1.45), (11.9, 1.45))
    ax2.text(12.1, 1.45, 'u(x)', ha='left', va='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    fp = _save_path('as03-01-fno-layer.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Done] FNO 层结构示意图已保存: {fp}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 70, flush=True)
    print("as03_fno/demo.py — 训练一维傅立叶神经算子 (FNO)", flush=True)
    print("=" * 70, flush=True)

    # 为了 CPU 上快速跑通教学 demo，使用较小的训练规模；
    # 增大 n_epochs / n_train / width 可进一步压低误差。
    print("\n[1/4] 训练 FNO 学习 a(x) -> u(x) 算子映射 (120 轮)...", flush=True)
    model, history, test_data = train_fno(
        n_train=300, n_test=80, grid_size=64, n_modes_data=12,
        modes=12, width=16, n_layers=3, n_epochs=80, lr=1e-3, batch_size=30
    )

    print("\n[2/4] 绘制预测结果对比图...")
    plot_predictions(model, test_data, n_show=3)

    print("\n绘制训练/测试损失曲线...")
    plot_loss_curve(history)

    print("\n[3/4] 演示分辨率不变性 (同一模型用于 64 和 192 网格)...", flush=True)
    plot_resolution_invariance(model, n_modes_data=12)

    print("\n[4/4] 绘制 Fourier Layer 结构示意图...")
    plot_fno_layer_diagram()

    print("\n" + "=" * 70)
    print("全部完成! 图像保存在:", _IMAGES_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
