# -*- coding: utf-8 -*-
"""
===============================================================================
as04_pino/code/demo.py — PINO：物理信息神经算子演示
===============================================================================
本 demo 在一个「一维变系数扩散方程族」（1D Darcy-flow 风格）上，完整
对比三种范式：

    -(k_a(x) u'(x))' = f(x),  x in [0,1],  u(0) = u(1) = 0
    k_a(x) = 1 + a * sin(pi*x)     <- 扩散系数场（"渗透率"），随参数 a 变化
    f(x)   = sin(pi*x)             <- 固定源项（所有实例共享）

    这是标准的 FNO / PINO 基准问题（对应 Darcy 流：k 是渗透率场，u 是压力场），
    但简化到一维、扩散系数只由一个标量 a 参数化，方便可视化和快速训练。
    由于系数 k_a(x) 依赖 a 的方式是非线性地进入方程（a 出现在 k 里面，
    而不是简单地线性缩放解），"参数 a -> 解 u_a" 这个映射本身是非线性算子，
    比线性叠加问题更贴近真实的算子学习场景。
    ground truth u_a(x) 没有解析式，用二阶有限体积（finite-volume）格式
    构造三对角线性方程组精确数值求解，作为「数值精确解」。

三种模型：
    1) PINN —— 对每一个新的 a，从零训练一个 MLP，仅用 PDE 残差 + 边界条件
       监督（不需要任何标注数据），但每来一个新 a 都要重新训练。
    2) FNO（数据驱动神经算子）—— 训练一个「扩散系数场 k_a(x) -> 解 u_a(x)」的
       映射网络，一旦训练好，对任意新的 a 都是一次前向推理（zero-shot），但
       需要大量标注的 (k_a, u_a) 数据对。我们训练两个版本：
         - FNO-full：11 个 a 全部标注（数据充足）
         - FNO-few ：只有 3 个 a 标注（数据稀缺，作为「无物理约束」对照）
    3) PINO —— 与 FNO 结构相同的算子网络，但训练损失 = 数据损失（仅 3 个
       标注 a）+ PDE 残差损失（对 11 个 a 都可以算，因为残差不需要标签）。
       用来证明：物理约束能大幅缓解数据稀缺问题。

最终在若干「训练时没见过」的 a（部分在训练范围内插值，部分超出范围做外推）
上评估四个模型，画出预测曲线对比 + 误差对比图，并生成一张 PINO 思想的
概念图（as04-01-pino-idea.png）。

运行方式：cd docs/science/pino/code && python demo.py
依赖：numpy, torch, matplotlib
===============================================================================
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
# 本章配图含大量中文标注，需显式指定中文字体，否则显示为方框
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ---------------------------------------------------------------------------
# 全局设置：固定随机种子，保证结果可复现
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)

DEVICE = torch.device('cpu')  # CPU-first：本章规模很小，CPU 秒级完成


# ============================================================================
# 第 1 部分：问题定义 —— 一维变系数扩散方程族（1D Darcy-flow）
# ============================================================================

N_GRID = 65                       # 网格点数（含两端边界）
X_GRID = np.linspace(0.0, 1.0, N_GRID)
DX = X_GRID[1] - X_GRID[0]
X_TORCH = torch.tensor(X_GRID, dtype=torch.float32)

F_FIXED = np.sin(np.pi * X_GRID)                  # 固定源项 f(x)，所有实例共享
F_FIXED_TORCH = torch.tensor(F_FIXED, dtype=torch.float32)


def diffusivity_field(a: float, x: np.ndarray) -> np.ndarray:
    """扩散系数（"渗透率"）场 k_a(x) = 1 + a*sin(pi*x)，在 x in [0,1] 上恒正"""
    return 1.0 + a * np.sin(np.pi * x)


def solve_variable_coeff_poisson(a: float) -> np.ndarray:
    """
    用二阶有限体积格式，精确数值求解 -(k_a(x) u'(x))' = f(x), u(0)=u(1)=0。

    做法：先在网格点上采样 k，再取相邻两点的平均得到"半网格点"上的 k_half
    （有限体积法的标准处理），组装对称三对角矩阵后直接线性求解。
    这一套离散格式与后面 FNO/PINO 的可微残差函数完全一致，保证两者在
    描述"同一个物理"。
    """
    k = diffusivity_field(a, X_GRID)
    k_half = 0.5 * (k[:-1] + k[1:])                       # 长度 N-1，第 i 个是 (x_i, x_{i+1}) 中点处的 k
    n_inner = N_GRID - 2
    A = np.zeros((n_inner, n_inner))
    b = np.zeros(n_inner)
    for i in range(1, N_GRID - 1):
        idx = i - 1
        b[idx] = F_FIXED[i]
        A[idx, idx] = (k_half[i - 1] + k_half[i]) / DX ** 2
        if idx - 1 >= 0:
            A[idx, idx - 1] = -k_half[i - 1] / DX ** 2
        if idx + 1 <= n_inner - 1:
            A[idx, idx + 1] = -k_half[i] / DX ** 2
    u_inner = np.linalg.solve(A, b)
    u = np.zeros(N_GRID)
    u[1:-1] = u_inner
    return u


def make_dataset(a_values):
    """给定一批参数 a，返回 (k 场, u 数值解) 数组，形状均为 (len(a_values), N_GRID)"""
    k = np.stack([diffusivity_field(a, X_GRID) for a in a_values]).astype(np.float32)
    u = np.stack([solve_variable_coeff_poisson(a) for a in a_values]).astype(np.float32)
    return k, u


# 训练用的参数网格：a in [1.0, 3.0]，共 11 个
A_TRAIN_ALL = np.linspace(1.0, 3.0, 11)

# 测试用的参数：1.5/2.5 在训练范围内（插值），3.4/4.0 超出范围（外推）
A_TEST = np.array([1.5, 2.5, 3.4, 4.0])
TEST_IS_EXTRAPOLATION = A_TEST > A_TRAIN_ALL.max()

K_TRAIN_ALL, U_TRAIN_ALL = make_dataset(A_TRAIN_ALL)
K_TEST, U_TEST = make_dataset(A_TEST)


def relative_l2_error(pred: np.ndarray, true: np.ndarray) -> float:
    """相对 L2 误差 ||pred-true|| / ||true||，用来在不同量级的解之间公平比较"""
    return float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12))


# ============================================================================
# 第 2 部分：PINN —— 每个实例单独训练的物理信息神经网络
# ============================================================================

class PINN(nn.Module):
    """
    最朴素的 PINN：输入标量 x，输出标量 u(x)。
    通过自动微分求 u''(x)，用 PDE 残差 + 边界条件作为损失，完全不需要标注数据。
    """

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_pinn_for_a(a: float, n_epochs: int = 1200, lr: float = 1e-2):
    """
    针对固定参数 a，从零训练一个 PINN。

    PDE 是 -(k_a(x) u'(x))' = f(x)，展开链式法则得到：
        -(k_a'(x) u'(x) + k_a(x) u''(x)) = f(x)
    其中 k_a(x) = 1 + a*sin(pi*x)，k_a'(x) = a*pi*cos(pi*x) —— 因为我们是这个
    PDE 实例的"拥有者"，系数场的解析形式和导数都是已知的，这正是 PINN 的
    典型使用场景：手握一个具体方程，用自动微分代替数值差分构造残差。

    残差点使用内部网格（不含端点），边界条件用 x=0, x=1 两个点单独约束。
    返回：训练好模型在 X_GRID 上的预测、训练耗时（秒）。
    """
    torch.manual_seed(SEED)  # 每次都从同一初始化出发，公平对比
    model = PINN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x_interior = torch.tensor(X_GRID[1:-1], dtype=torch.float32).view(-1, 1)
    x_interior.requires_grad_(True)
    x_bc = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    f_interior = torch.tensor(F_FIXED[1:-1], dtype=torch.float32).view(-1, 1)

    k_val = 1.0 + a * torch.sin(np.pi * x_interior)            # k_a(x)
    k_prime = a * np.pi * torch.cos(np.pi * x_interior)        # k_a'(x)

    t0 = time.time()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        u = model(x_interior)
        # 自动微分求一阶、二阶导数：u' 和 u''
        du_dx = torch.autograd.grad(u, x_interior, grad_outputs=torch.ones_like(u),
                                     create_graph=True)[0]
        d2u_dx2 = torch.autograd.grad(du_dx, x_interior, grad_outputs=torch.ones_like(du_dx),
                                       create_graph=True)[0]

        residual = -(k_prime * du_dx + k_val * d2u_dx2) - f_interior  # -(k u')' - f = 0
        loss_pde = (residual ** 2).mean()

        u_bc = model(x_bc)
        loss_bc = (u_bc ** 2).mean()               # u(0)=u(1)=0

        loss = loss_pde + 10.0 * loss_bc
        loss.backward()
        optimizer.step()
    elapsed = time.time() - t0

    with torch.no_grad():
        x_full = torch.tensor(X_GRID, dtype=torch.float32).view(-1, 1)
        u_pred = model(x_full).numpy().flatten()

    return u_pred, elapsed


# ============================================================================
# 第 3 部分：FNO-lite —— 一维傅里叶神经算子（用于 FNO 与 PINO 共享架构）
# ============================================================================

class SpectralConv1d(nn.Module):
    """
    一维谱卷积层（FNO 的核心构件）。

    做法：FFT 把函数变到频域 -> 只保留低频的前 modes 个模态 -> 对每个保留的
    模态做一个可学习的复数线性变换（相当于频域上的"逐模态卷积核"）-> IFFT
    变回物理空间。高频模态被直接置零，这也是 FNO 天然的低通滤波/正则化。
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        # 复数权重用两个实数张量（实部、虚部）表示，方便用普通 Adam 优化
        self.weight_real = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes))
        self.weight_imag = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes))

    def forward(self, x):
        # x: (B, C_in, N)
        B, C, N = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)             # (B, C_in, N//2+1) 复数
        modes = min(self.modes, x_ft.shape[-1])

        weight = torch.complex(self.weight_real[:, :, :modes], self.weight_imag[:, :, :modes])
        out_ft = torch.zeros(B, self.out_channels, x_ft.shape[-1],
                              dtype=torch.cfloat, device=x.device)
        # 只对低频的 modes 个模态做可学习的线性混合（"逐模态卷积"）
        out_ft[:, :, :modes] = torch.einsum('bix,iox->box', x_ft[:, :, :modes], weight)

        x_out = torch.fft.irfft(out_ft, n=N, dim=-1)  # (B, C_out, N)
        return x_out


class FNO1d(nn.Module):
    """
    小型一维 FNO：输入 (扩散系数场 k(x), 坐标 x) 两个通道，输出解 u(x)。

    结构：升维(lift) -> [谱卷积 + 逐点卷积 + 残差 + 激活] x n_layers -> 降维(project)
    """

    def __init__(self, modes: int = 8, width: int = 16, n_layers: int = 2):
        super().__init__()
        self.width = width
        self.fc0 = nn.Linear(2, width)                       # 升维：2 通道 -> width
        self.spectral_layers = nn.ModuleList(
            [SpectralConv1d(width, width, modes) for _ in range(n_layers)])
        self.pointwise_layers = nn.ModuleList(
            [nn.Conv1d(width, width, kernel_size=1) for _ in range(n_layers)])
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, 1)                       # 降维：width -> 1
        self.act = nn.GELU()

    def forward(self, k_field, x_coord):
        # k_field, x_coord: (B, N)
        inp = torch.stack([k_field, x_coord], dim=-1)         # (B, N, 2)
        h = self.fc0(inp)                                    # (B, N, width)
        h = h.permute(0, 2, 1)                                # (B, width, N)

        for spec, point in zip(self.spectral_layers, self.pointwise_layers):
            h = self.act(spec(h) + point(h))                 # 谱卷积 + 逐点卷积残差

        h = h.permute(0, 2, 1)                                # (B, N, width)
        h = self.act(self.fc1(h))
        out = self.fc2(h).squeeze(-1)                         # (B, N)
        return out


def pde_residual_and_bc(u_pred: torch.Tensor, k_field: torch.Tensor, f: torch.Tensor):
    """
    给定网格上的预测解 u_pred (B, N) 和对应扩散系数场 k_field (B, N)，用与
    ground truth 完全一致的二阶有限体积格式构造 -(k u')' - f 残差，以及
    边界条件损失。这一步全部由张量运算完成，梯度可以直接反传到产生 u_pred
    的算子网络参数上——这正是 PINO 的关键：物理约束和数据驱动的算子共享
    同一套可微计算图，不需要额外的数值求解器。
    """
    k_half = 0.5 * (k_field[:, :-1] + k_field[:, 1:])           # (B, N-1)，半网格点上的 k
    flux_right = k_half[:, 1:] * (u_pred[:, 2:] - u_pred[:, 1:-1]) / DX     # 内部点右侧通量
    flux_left = k_half[:, :-1] * (u_pred[:, 1:-1] - u_pred[:, :-2]) / DX    # 内部点左侧通量
    residual = -(flux_right - flux_left) / DX - f[:, 1:-1]
    loss_pde = (residual ** 2).mean()
    loss_bc = (u_pred[:, 0] ** 2).mean() + (u_pred[:, -1] ** 2).mean()
    return loss_pde, loss_bc


def train_operator(use_physics: bool, n_labeled_use: int, n_epochs: int = 1200, lr: float = 3e-3):
    """
    训练一个 FNO-lite 算子网络。

    参数:
        use_physics: 是否加入 PDE 残差损失（True -> PINO，False -> 纯数据驱动 FNO）
        n_labeled_use: 训练时把 A_TRAIN_ALL 中前 n_labeled_use 个"标注 a"用于数据损失
                       （其余仍参与前向传播和物理残差，但不提供标签）
                       n_labeled_use = len(A_TRAIN_ALL) 时，等价于「全监督 FNO」
        n_epochs, lr: 训练轮数、学习率
    返回: 训练好的模型, 训练损失曲线（list）
    """
    torch.manual_seed(SEED)
    model = FNO1d(modes=8, width=16, n_layers=2).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    k_all = torch.tensor(K_TRAIN_ALL, dtype=torch.float32)              # (11, N)
    x_all = X_TORCH.unsqueeze(0).repeat(k_all.shape[0], 1)              # (11, N)
    u_all = torch.tensor(U_TRAIN_ALL, dtype=torch.float32)              # (11, N)
    f_all = F_FIXED_TORCH.unsqueeze(0).repeat(k_all.shape[0], 1)        # (11, N)，固定源项广播

    if n_labeled_use >= len(A_TRAIN_ALL):
        labeled_idx = np.arange(len(A_TRAIN_ALL))
    else:
        # 使用固定的、跨越训练范围的标注点（首、中、尾），而非任取前几个
        labeled_idx = np.round(np.linspace(0, len(A_TRAIN_ALL) - 1, n_labeled_use)).astype(int)
    labeled_idx_t = torch.tensor(labeled_idx, dtype=torch.long)

    loss_history = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        u_pred = model(k_all, x_all)                                    # (11, N)，对全部 a 前向

        loss_data = ((u_pred[labeled_idx_t] - u_all[labeled_idx_t]) ** 2).mean()

        if use_physics:
            loss_pde, loss_bc = pde_residual_and_bc(u_pred, k_all, f_all)  # 对全部 11 个 a 都能算
        else:
            loss_pde = torch.tensor(0.0)
            loss_bc = torch.tensor(0.0)

        loss = loss_data + (0.1 * loss_pde + 1.0 * loss_bc if use_physics else 0.0)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

    return model, loss_history


@torch.no_grad()
def operator_predict(model: FNO1d, a_values: np.ndarray) -> np.ndarray:
    """对给定的一批参数 a（训练中从未标注过、甚至从未见过）做零样本推理"""
    k = torch.tensor(np.stack([diffusivity_field(a, X_GRID) for a in a_values]), dtype=torch.float32)
    x = X_TORCH.unsqueeze(0).repeat(len(a_values), 1)
    u_pred = model(k, x)
    return u_pred.numpy()


# ============================================================================
# 第 4 部分：概念图 —— PINN / FNO / PINO 三种范式对比
# ============================================================================

def draw_pino_idea_diagram(save_path: str):
    """
    手绘风格的概念图：三条流水线（PINN / FNO / PINO），
    展示各自的输入、模型、损失构成与"是否需要重新训练"。
    """
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7.5)
    ax.axis('off')

    def box(x, y, w, h, text, color, fontsize=10.5, weight='normal'):
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='black',
                              linewidth=1.3, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                fontsize=fontsize, weight=weight, zorder=3)

    def arrow(x0, y0, x1, y1, color='black'):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='-|>', color=color, lw=1.6), zorder=2)

    ax.text(5.5, 7.15, 'PINN vs FNO vs PINO：三种范式怎么组合"物理"与"数据"',
            ha='center', va='center', fontsize=13.5, weight='bold')

    row_y = [5.15, 3.0, 0.85]
    row_h = 1.55
    labels = ['PINN\n（纯物理，逐实例求解）', 'FNO / 神经算子\n（纯数据，学一次用一生）',
              'PINO\n（物理 + 数据，两者结合）']
    colors = ['#FDE2E2', '#DDEBF7', '#E2F0D9']

    for y, label, color in zip(row_y, labels, colors):
        box(0.3, y, 2.35, row_h, label, color, fontsize=10.5, weight='bold')

    # ---- PINN 行 ----
    y = row_y[0]
    box(3.1, y + 0.85, 1.9, 0.6, '参数 a（新实例）', '#FFF2CC')
    arrow(5.0, y + 1.15, 5.75, y + 1.15)
    box(5.75, y + 0.85, 2.1, 0.6, 'MLP u_θ(x)\n从零训练', '#FDE2E2')
    arrow(7.85, y + 1.15, 8.6, y + 1.15)
    box(8.6, y + 0.85, 2.1, 0.6, 'PDE 残差 + BC\n(不需要标签)', '#F8CBAD')
    ax.text(8.6, y + 0.35, '⚠ 每换一个 a 都要重新训练', fontsize=9, color='#7F2704')

    # ---- FNO 行 ----
    y = row_y[1]
    box(3.1, y + 0.85, 1.9, 0.6, '大量 (f_a, u_a)\n标注数据', '#FFF2CC')
    arrow(5.0, y + 1.15, 5.75, y + 1.15)
    box(5.75, y + 0.85, 2.1, 0.6, '算子网络 G_θ\n(a) -> u_a', '#DDEBF7')
    arrow(7.85, y + 1.15, 8.6, y + 1.15)
    box(8.6, y + 0.85, 2.1, 0.6, '监督损失\n||G_θ(a)-u_a||²', '#BDD7EE')
    ax.text(8.6, y + 0.35, '✓ 训好后对新 a 一次前向  ⚠ 依赖大量标注数据', fontsize=9, color='#1F4E79')

    # ---- PINO 行 ----
    y = row_y[2]
    box(3.1, y + 0.85, 1.9, 0.6, '少量标注 + 大量\n仅有 a 的"无标签"实例', '#FFF2CC')
    arrow(5.0, y + 1.15, 5.75, y + 1.15)
    box(5.75, y + 0.85, 2.1, 0.6, '算子网络 G_θ\n(a) -> u_a', '#E2F0D9')
    arrow(7.85, y + 1.15, 8.6, y + 1.30)
    arrow(7.85, y + 1.15, 8.6, y + 0.95)
    box(8.6, y + 1.05, 2.1, 0.45, '数据损失（少量标签）', '#C6E0B4')
    box(8.6, y + 0.45, 2.1, 0.45, 'PDE 残差损失（全部实例）', '#A9D18E')
    ax.text(8.6, y - 0.05, '✓ 一次前向  ✓ 标注数据需求大幅降低', fontsize=9, color='#375623')

    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close()
    print(f'  [概念图] 已保存到 {save_path}')


# ============================================================================
# 第 5 部分：主流程 —— 训练四个模型 + 对比评估 + 画图
# ============================================================================

def main():
    print('=' * 70)
    print('as04 PINO Demo：PINN vs FNO vs PINO 对比实验')
    print('=' * 70)

    print('\n[0/5] 生成 PINO 思想概念图...')
    draw_pino_idea_diagram(os.path.join(_IMAGES_DIR, 'as04-01-pino-idea.png'))

    # ---------- 1. 训练 FNO-few（数据稀缺、无物理约束） ----------
    print('\n[1/5] 训练 FNO-few（仅 3 个标注 a，无物理约束）...')
    model_fno_few, hist_fno_few = train_operator(use_physics=False, n_labeled_use=3)
    print(f'  最终训练损失: {hist_fno_few[-1]:.6f}')

    # ---------- 2. 训练 FNO-full（11 个 a 全部标注，无物理约束） ----------
    print('\n[2/5] 训练 FNO-full（11 个 a 全部标注，无物理约束）...')
    model_fno_full, hist_fno_full = train_operator(use_physics=False, n_labeled_use=11)
    print(f'  最终训练损失: {hist_fno_full[-1]:.6f}')

    # ---------- 3. 训练 PINO（3 个标注 a + 全部 11 个 a 的物理残差） ----------
    print('\n[3/5] 训练 PINO（3 个标注 a + 11 个 a 的 PDE 残差）...')
    model_pino, hist_pino = train_operator(use_physics=True, n_labeled_use=3)
    print(f'  最终训练损失: {hist_pino[-1]:.6f}')

    # ---------- 4. 对每个测试 a，单独训练 PINN ----------
    print('\n[4/5] 对每个测试参数 a 单独训练 PINN（逐实例求解）...')
    pinn_preds = []
    pinn_times = []
    for a in A_TEST:
        u_pred, elapsed = train_pinn_for_a(a)
        pinn_preds.append(u_pred)
        pinn_times.append(elapsed)
        print(f'  a={a:.2f}: 训练耗时 {elapsed:.2f}s')
    pinn_preds = np.stack(pinn_preds)

    # ---------- 5. 零样本推理：FNO-few / FNO-full / PINO 直接前向，无需重训 ----------
    print('\n[5/5] FNO / PINO 对测试参数 a 做零样本推理（无需重新训练）...')
    t0 = time.time()
    pred_fno_few = operator_predict(model_fno_few, A_TEST)
    t_fno_few = (time.time() - t0) / len(A_TEST)
    t0 = time.time()
    pred_fno_full = operator_predict(model_fno_full, A_TEST)
    t_fno_full = (time.time() - t0) / len(A_TEST)
    t0 = time.time()
    pred_pino = operator_predict(model_pino, A_TEST)
    t_pino = (time.time() - t0) / len(A_TEST)

    # ---------- 误差统计 ----------
    methods = ['PINN(逐实例)', 'FNO-few(3标注)', 'FNO-full(11标注)', 'PINO(3标注+物理)']
    all_preds = [pinn_preds, pred_fno_few, pred_fno_full, pred_pino]
    errors = np.zeros((len(methods), len(A_TEST)))
    for m, preds in enumerate(all_preds):
        for i in range(len(A_TEST)):
            errors[m, i] = relative_l2_error(preds[i], U_TEST[i])

    print('\n' + '=' * 70)
    print(f'{"方法":<20}' + ''.join([f'a={a:<8.1f}' for a in A_TEST]) + '   平均耗时/实例')
    times_per_instance = [np.mean(pinn_times), t_fno_few, t_fno_full, t_pino]
    for name, err_row, t in zip(methods, errors, times_per_instance):
        print(f'{name:<20}' + ''.join([f'{e:<10.4f}' for e in err_row]) + f'   {t*1000:.2f} ms')
    print('=' * 70)

    # ---------- 图 1：预测曲线对比（选择一个插值 + 一个外推的测试 a） ----------
    demo_idxs = [0, 3]  # a=1.5(插值), a=4.0(外推)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, idx in zip(axes, demo_idxs):
        a = A_TEST[idx]
        ax.plot(X_GRID, U_TEST[idx], 'k-', lw=2.5, label='解析解 (ground truth)')
        ax.plot(X_GRID, pinn_preds[idx], '--', lw=1.8, label='PINN (逐实例训练)', color='#C0392B')
        ax.plot(X_GRID, pred_fno_few[idx], ':', lw=1.8, label='FNO-few (3标注,无物理)', color='#8E44AD')
        ax.plot(X_GRID, pred_fno_full[idx], '-.', lw=1.8, label='FNO-full (11标注,无物理)', color='#2980B9')
        ax.plot(X_GRID, pred_pino[idx], '-', lw=1.8, label='PINO (3标注+物理)', color='#27AE60')
        tag = '外推 (训练范围外)' if TEST_IS_EXTRAPOLATION[idx] else '插值 (训练范围内)'
        ax.set_title(f'a = {a:.1f}  [{tag}]', fontsize=12)
        ax.set_xlabel('x')
        ax.set_ylabel('u(x)')
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=9, loc='best')
    plt.suptitle('测试参数 a 上的预测对比：PINN vs FNO(few/full) vs PINO', fontsize=13)
    plt.tight_layout()
    path1 = os.path.join(_IMAGES_DIR, 'pino_comparison.png')
    plt.savefig(path1, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'\n  [可视化] 预测曲线对比已保存到 {path1}')

    # ---------- 图 2：误差 + 训练/推理成本 对比 ----------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    x_pos = np.arange(len(A_TEST))
    width = 0.2
    colors = ['#C0392B', '#8E44AD', '#2980B9', '#27AE60']
    for m, (name, err_row, color) in enumerate(zip(methods, errors, colors)):
        ax.bar(x_pos + (m - 1.5) * width, err_row, width=width, label=name, color=color)
    ax.set_xticks(x_pos)
    tags = [f'a={a:.1f}\n{"外推" if e else "插值"}' for a, e in zip(A_TEST, TEST_IS_EXTRAPOLATION)]
    ax.set_xticklabels(tags)
    ax.set_ylabel('相对 L2 误差')
    ax.set_title('测试参数上的误差对比', fontsize=12)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3, axis='y')

    ax = axes[1]
    ax.bar(methods, np.array(times_per_instance) * 1000, color=colors)
    ax.set_ylabel('新实例平均耗时 (ms, 对数轴)')
    ax.set_yscale('log')
    ax.set_title('新参数 a 到来时的"上线成本"对比', fontsize=12)
    ax.tick_params(axis='x', rotation=15)
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    path2 = os.path.join(_IMAGES_DIR, 'pino_error_and_cost.png')
    plt.savefig(path2, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  [可视化] 误差与成本对比已保存到 {path2}')

    # ---------- 图 3：训练损失曲线（FNO-few / FNO-full / PINO） ----------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(hist_fno_few, label='FNO-few (3标注,无物理)', color='#8E44AD')
    ax.plot(hist_fno_full, label='FNO-full (11标注,无物理)', color='#2980B9')
    ax.plot(hist_pino, label='PINO (3标注+物理)', color='#27AE60')
    ax.set_yscale('log')
    ax.set_xlabel('训练迭代')
    ax.set_ylabel('训练损失 (log 尺度)')
    ax.set_title('算子网络训练损失曲线')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path3 = os.path.join(_IMAGES_DIR, 'pino_training_loss.png')
    plt.savefig(path3, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  [可视化] 训练损失曲线已保存到 {path3}')

    print('\n' + '=' * 70)
    print('结论：')
    print(f'  - FNO-few 平均相对误差 {errors[1].mean():.4f} —— 数据太少，泛化差（尤其外推区）')
    print(f'  - FNO-full 平均相对误差 {errors[2].mean():.4f} —— 数据充足时表现最好，但标注成本高')
    print(f'  - PINO    平均相对误差 {errors[3].mean():.4f} —— 标注量与 FNO-few 相同，'
          f'借助物理残差逼近 FNO-full 的效果')
    print(f'  - PINN 平均每个新实例需要 {np.mean(pinn_times)*1000:.1f} ms 重新训练，'
          f'而算子方法只需 {np.mean([t_fno_few, t_fno_full, t_pino])*1000:.3f} ms 前向推理')
    print('=' * 70)
    print(f'\nDemo 完成！查看 {_IMAGES_DIR} 目录下的可视化结果。')


if __name__ == '__main__':
    main()
