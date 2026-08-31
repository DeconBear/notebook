# -*- coding: utf-8 -*-
from __future__ import annotations
"""
wm05 JEPA / V-JEPA：预测表征而非像素 —— 演示代码
======================================================
功能：
  1. 绘制 JEPA（联合嵌入预测架构）的架构示意图，对比"生成式"与"预测式"
     两条世界模型路线（wm05-01-jepa.png）
  2. 用 NumPy 生成一批合成图像（随机形状），从零实现一个"玩具版 I-JEPA"：
     - 图像分块（patchify）
     - 上下文编码器（Context Encoder，可训练）
     - 目标编码器（Target Encoder，EMA 动量更新，不接收梯度）
     - 预测器（Predictor）：给定可见 patch 的表征 + 被遮挡 patch 的位置，
       预测被遮挡 patch 在目标编码器输出空间中的表征
     - 损失：预测表征与目标表征之间的 L2 距离（表征空间回归，而非像素重建）
  3. 训练玩具 JEPA，可视化：
     - 掩码示例（上下文 patch vs 被遮挡的目标 patch）
     - 训练损失曲线
     - 逐 patch 预测误差热力图（表征空间预测得准不准）

核心教学要点：JEPA 家族（I-JEPA → V-JEPA → V-JEPA 2）的核心思想是
"预测表征，而不是预测像素"——通过在语义表征空间里做掩码预测，
模型被迫学习抛弃像素级噪声、保留语义上可预测的结构。

每个函数都有中文 docstring，每行逻辑代码都有中文注释。
运行方式：在 wm05_jepa/ 目录下执行 python code/demo.py
依赖: pip install numpy matplotlib torch
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# 本章配图含大量中文标注，需显式指定中文字体，否则显示为方框
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_IMAGES = os.path.join(_HERE, '..', 'images')
os.makedirs(_IMAGES, exist_ok=True)


def set_seed(seed: int = 42):
    """设置随机种子，保证实验可复现。"""
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================================
# 第一部分：架构示意图 —— 生成式 vs 预测式世界模型
# ============================================================================

def _draw_box(ax, xy, w, h, text, color, fontsize=9):
    """在 ax 上绘制一个带文字的圆角矩形框，返回框的中心坐标。"""
    from matplotlib.patches import FancyBboxPatch
    x, y = xy
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                          linewidth=1.5, edgecolor='#333333', facecolor=color)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
             fontsize=fontsize, wrap=True)
    return (x + w / 2, y + h / 2)


def _draw_arrow(ax, p1, p2, style='-|>', color='#333333'):
    """在两点之间绘制箭头。"""
    ax.annotate('', xy=p2, xytext=p1,
                arrowprops=dict(arrowstyle=style, color=color, lw=1.6))


def plot_jepa_architecture():
    """
    绘制 JEPA 架构示意图：
    左边——生成式路线（重建像素，如 MAE / 视频扩散）
    右边——预测式路线（JEPA：在表征空间做预测）
    并标注 I-JEPA → V-JEPA → V-JEPA 2 的演进关系。
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # ---- 左图：生成式（像素重建）路线 ----
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('生成式路线（重建像素）\n如 MAE / 扩散式视频生成', fontsize=12, fontweight='bold')

    p_in = _draw_box(ax, (0.5, 7.8), 3.2, 1.4, '掩码输入图像\n(遮挡部分patch)', '#FFE8CC')
    p_enc = _draw_box(ax, (0.5, 5.6), 3.2, 1.2, '编码器', '#CDE7FF')
    p_dec = _draw_box(ax, (0.5, 3.4), 3.2, 1.2, '解码器\n(逐像素重建)', '#CDE7FF')
    p_out = _draw_box(ax, (0.5, 1.2), 3.2, 1.4, '重建像素\n(含纹理/噪声细节)', '#FFD6D6')
    _draw_arrow(ax, (2.1, 7.8), (2.1, 6.8))
    _draw_arrow(ax, (2.1, 5.6), (2.1, 4.6))
    _draw_arrow(ax, (2.1, 3.4), (2.1, 2.6))
    ax.text(2.1, 0.3, '损失：像素级 MSE / 扩散去噪损失\n缺点：被迫拟合不可预测的高频细节',
            ha='center', fontsize=8.5, color='#8B0000')

    # ---- 右图：JEPA（预测式，表征空间）路线 ----
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('预测式路线：JEPA\n(I-JEPA → V-JEPA → V-JEPA 2)', fontsize=12, fontweight='bold')

    ctx = _draw_box(ax, (0.3, 7.8), 3.0, 1.3, '上下文 patch\n(可见区域)', '#FFE8CC')
    ctx_enc = _draw_box(ax, (0.3, 5.8), 3.0, 1.1, '上下文编码器\n(可训练)', '#CDE7FF')
    pred = _draw_box(ax, (3.6, 3.9), 3.0, 1.3, '预测器\n(输入: 上下文表征 +\n目标位置掩码token)', '#D6F5D6')
    tgt = _draw_box(ax, (7.0, 7.8), 3.0, 1.3, '目标 patch\n(被遮挡区域)', '#FFE8CC')
    tgt_enc = _draw_box(ax, (7.0, 5.8), 3.0, 1.1, '目标编码器\nEMA(动量更新, 停梯度)', '#CDE7FF')
    loss_box = _draw_box(ax, (3.6, 1.6), 3.0, 1.1, 'L2 损失\n(表征空间回归)', '#FFD6D6')

    _draw_arrow(ax, (1.8, 7.8), (1.8, 6.9))
    _draw_arrow(ax, (1.8, 5.8), (3.7, 4.7))
    _draw_arrow(ax, (8.5, 7.8), (8.5, 6.9))
    _draw_arrow(ax, (7.6, 5.8), (6.3, 2.4))
    _draw_arrow(ax, (5.1, 3.9), (5.1, 2.7))
    ax.text(5.1, 0.3, '损失：预测表征 vs 目标表征的 L2 距离\n优点：只学语义上"可预测"的结构，\n天然过滤像素噪声', ha='center', fontsize=8.5, color='#006400')

    fig.suptitle('JEPA：预测表征，而不是预测像素', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(_IMAGES, 'wm05-01-jepa.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('[可视化] JEPA 架构示意图已保存至 images/wm05-01-jepa.png')


# ============================================================================
# 第二部分：合成图像数据集 —— 随机形状图像
# ============================================================================

def make_synthetic_images(n: int, size: int = 20) -> np.ndarray:
    """
    生成 n 张合成灰度图像，每张图像上有 1~2 个随机形状（圆形/方形块），
    用来模拟"具有可预测语义结构、但像素细节随机"的图像。

    参数:
        n: 图像数量
        size: 图像边长（size x size）

    返回:
        images: (n, size, size) 的 float32 数组，取值范围 [0, 1]
    """
    images = np.zeros((n, size, size), dtype=np.float32)
    for i in range(n):
        n_shapes = np.random.randint(1, 3)                     # 1或2个形状
        for _ in range(n_shapes):
            cx, cy = np.random.randint(4, size - 4, size=2)     # 形状中心
            r = np.random.randint(2, 5)                         # 形状半径
            val = np.random.uniform(0.5, 1.0)                   # 形状亮度
            if np.random.rand() < 0.5:
                # 圆形
                yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
                mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
            else:
                # 方形
                yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
                mask = (np.abs(xx - cx) <= r) & (np.abs(yy - cy) <= r)
            images[i][mask] = val
        # 加一点像素级噪声——这是"不可预测的高频细节"，JEPA 应该学会忽略它
        images[i] += np.random.normal(0, 0.03, size=(size, size)).astype(np.float32)
    images = np.clip(images, 0, 1)
    return images


def patchify(images: np.ndarray, patch_size: int = 4) -> np.ndarray:
    """
    将图像分割成不重叠的 patch。

    参数:
        images: (n, H, W)
        patch_size: 每个 patch 的边长

    返回:
        patches: (n, n_patches, patch_size*patch_size)
    """
    n, H, W = images.shape
    n_h, n_w = H // patch_size, W // patch_size
    patches = images.reshape(n, n_h, patch_size, n_w, patch_size)
    patches = patches.transpose(0, 1, 3, 2, 4).reshape(n, n_h * n_w, patch_size * patch_size)
    return patches, n_h, n_w


# ============================================================================
# 第三部分：玩具 JEPA 模型 —— 编码器 + 预测器
# ============================================================================

class PatchEncoder(nn.Module):
    """
    Patch 编码器：将每个 patch（拉平的像素向量）+ 位置编码，
    通过一个小型 Transformer Encoder 层，输出每个 patch 的表征向量。

    这里只用一层 TransformerEncoderLayer 来模拟"编码器会在 patch 之间
    做自注意力交换信息"的真实 I-JEPA 设计，同时保持玩具规模、CPU 秒级可跑。
    """

    def __init__(self, patch_dim: int, n_patches: int, embed_dim: int = 32, n_heads: int = 4):
        super().__init__()
        self.proj = nn.Linear(patch_dim, embed_dim)                 # 像素 patch → embedding
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, embed_dim) * 0.02)  # 可学习位置编码
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 2,
            batch_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)

    def forward(self, patches: torch.Tensor, patch_idx: torch.Tensor) -> torch.Tensor:
        """
        参数:
            patches: (B, K, patch_dim) —— 只包含被选中的 K 个 patch（如可见 patch）
            patch_idx: (B, K) —— 这些 patch 在原图中的位置索引，用于取对应的位置编码
        返回:
            (B, K, embed_dim) 每个 patch 的表征
        """
        x = self.proj(patches)                                      # (B, K, embed_dim)
        pos = self.pos_embed.expand(patches.size(0), -1, -1)
        pos = torch.gather(pos, 1, patch_idx.unsqueeze(-1).expand(-1, -1, pos.size(-1)))
        x = x + pos                                                  # 加位置编码
        return self.encoder(x)                                       # 自注意力交换信息


class Predictor(nn.Module):
    """
    预测器：输入上下文 patch 的表征 + 目标位置的"掩码 token"（可学习向量 + 位置编码），
    通过一层 Transformer，输出对目标 patch 表征的预测。

    这正是 I-JEPA 的核心：预测器只被告知"要预测哪个位置"，而不知道
    该位置的像素内容，必须依靠上下文推断该处的语义表征。
    """

    def __init__(self, embed_dim: int = 32, n_patches: int = 25, n_heads: int = 4):
        super().__init__()
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)   # 共享的掩码 token
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, embed_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 2,
            batch_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(embed_dim, embed_dim)

    def forward(self, context_repr: torch.Tensor, context_idx: torch.Tensor,
                target_idx: torch.Tensor) -> torch.Tensor:
        """
        参数:
            context_repr: (B, K_ctx, embed_dim) 上下文编码器输出的表征
            context_idx: (B, K_ctx) 上下文 patch 的位置索引
            target_idx: (B, K_tgt) 目标 patch 的位置索引（要预测哪些位置）
        返回:
            (B, K_tgt, embed_dim) 对每个目标位置的表征预测
        """
        B, K_tgt = target_idx.shape
        embed_dim = context_repr.size(-1)
        pos_all = self.pos_embed.expand(B, -1, -1)

        ctx_pos = torch.gather(pos_all, 1, context_idx.unsqueeze(-1).expand(-1, -1, embed_dim))
        ctx_tokens = context_repr + ctx_pos * 0.0                     # 上下文表征已含位置信息，此处仅占位对齐

        tgt_pos = torch.gather(pos_all, 1, target_idx.unsqueeze(-1).expand(-1, -1, embed_dim))
        mask_tokens = self.mask_token.expand(B, K_tgt, -1) + tgt_pos  # 掩码 token + 目标位置编码

        seq = torch.cat([ctx_tokens, mask_tokens], dim=1)             # 拼接上下文与掩码 token
        out = self.encoder(seq)
        pred = out[:, -K_tgt:, :]                                      # 只取掩码 token 对应的输出
        return self.head(pred)


@torch.no_grad()
def ema_update(target: nn.Module, context: nn.Module, momentum: float = 0.996):
    """
    用动量（EMA）方式更新目标编码器参数：θ_target ← m·θ_target + (1-m)·θ_context。
    目标编码器不接收梯度，只通过 EMA 缓慢跟随上下文编码器，
    这是防止"表征坍缩"（模型学到常数解）的关键设计。
    """
    for p_t, p_c in zip(target.parameters(), context.parameters()):
        p_t.data.mul_(momentum).add_(p_c.data, alpha=1 - momentum)


# ============================================================================
# 第四部分：训练循环
# ============================================================================

def sample_mask(n_patches: int, mask_ratio: float = 0.4):
    """
    随机选出一块连续区域作为"目标 patch"（被遮挡），其余作为"上下文 patch"。
    真实 I-JEPA 用的是若干个矩形块掩码；这里简化为随机挑选一段索引区间。
    """
    n_target = max(1, int(n_patches * mask_ratio))
    start = np.random.randint(0, n_patches - n_target + 1)
    target_idx = np.arange(start, start + n_target)
    context_idx = np.array([i for i in range(n_patches) if i not in target_idx])
    return context_idx, target_idx


def train_toy_jepa(images: np.ndarray, n_steps: int = 300, batch_size: int = 32,
                    patch_size: int = 4, embed_dim: int = 32, lr: float = 2e-3):
    """
    训练玩具 JEPA 模型。

    每一步：
      1. 随机采样一个 batch 的图像，分块
      2. 随机采样掩码：一部分 patch 作为上下文，一部分作为目标（被遮挡）
      3. 上下文编码器编码可见 patch → 上下文表征
      4. 目标编码器（EMA，停梯度）编码全部 patch → 取出目标位置的表征作为回归目标
      5. 预测器用上下文表征 + 目标位置信息，预测目标位置的表征
      6. 损失 = 预测表征与目标表征之间的 L2 距离，反向传播更新上下文编码器和预测器
      7. EMA 更新目标编码器

    返回:
        ctx_encoder, tgt_encoder, predictor, loss_history, (patch_size, n_h, n_w)
    """
    patches_all, n_h, n_w = patchify(images, patch_size)
    n_patches = n_h * n_w
    patch_dim = patch_size * patch_size
    n_images = images.shape[0]

    ctx_encoder = PatchEncoder(patch_dim, n_patches, embed_dim)
    tgt_encoder = PatchEncoder(patch_dim, n_patches, embed_dim)
    tgt_encoder.load_state_dict(ctx_encoder.state_dict())              # 初始时目标=上下文
    for p in tgt_encoder.parameters():
        p.requires_grad_(False)                                        # 目标编码器永远不接收梯度
    predictor = Predictor(embed_dim, n_patches)

    params = list(ctx_encoder.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    loss_history = []
    for step in range(n_steps):
        idx = np.random.choice(n_images, batch_size, replace=False)
        batch_patches = torch.from_numpy(patches_all[idx]).float()     # (B, n_patches, patch_dim)

        context_idx_np, target_idx_np = sample_mask(n_patches, mask_ratio=0.4)
        context_idx = torch.from_numpy(context_idx_np).long().unsqueeze(0).expand(batch_size, -1)
        target_idx = torch.from_numpy(target_idx_np).long().unsqueeze(0).expand(batch_size, -1)

        ctx_patches = torch.gather(
            batch_patches, 1, context_idx.unsqueeze(-1).expand(-1, -1, patch_dim))

        # ---- 上下文编码器：只看得到可见 patch ----
        ctx_repr = ctx_encoder(ctx_patches, context_idx)

        # ---- 目标编码器：能看到完整图像（但停梯度），取出目标位置的表征 ----
        with torch.no_grad():
            full_idx = torch.arange(n_patches).long().unsqueeze(0).expand(batch_size, -1)
            full_repr = tgt_encoder(batch_patches, full_idx)
            tgt_repr = torch.gather(
                full_repr, 1, target_idx.unsqueeze(-1).expand(-1, -1, embed_dim))

        # ---- 预测器：用上下文表征预测目标位置的表征 ----
        pred_repr = predictor(ctx_repr, context_idx, target_idx)

        loss = F.mse_loss(pred_repr, tgt_repr)                          # 表征空间 L2 回归损失

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ema_update(tgt_encoder, ctx_encoder, momentum=0.996)             # EMA 更新目标编码器

        loss_history.append(loss.item())
        if (step + 1) % 50 == 0:
            print(f'  Step {step+1:4d}/{n_steps}: 表征预测损失 = {loss.item():.5f}')

    return ctx_encoder, tgt_encoder, predictor, loss_history, (patch_size, n_h, n_w)


# ============================================================================
# 第五部分：可视化
# ============================================================================

def plot_mask_example(image: np.ndarray, patch_size: int, n_h: int, n_w: int,
                       context_idx: np.ndarray, target_idx: np.ndarray):
    """
    可视化一次掩码采样：原图 + 上下文/目标 patch 的划分示意。
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

    axes[0].imshow(image, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('原始合成图像', fontsize=11)
    axes[0].axis('off')

    overlay = np.zeros((n_h, n_w, 3))
    for idx in context_idx:
        overlay[idx // n_w, idx % n_w] = [0.6, 0.85, 1.0]              # 上下文=蓝色
    for idx in target_idx:
        overlay[idx // n_w, idx % n_w] = [1.0, 0.6, 0.6]               # 目标(被遮挡)=红色
    axes[1].imshow(overlay, interpolation='nearest')
    axes[1].set_title('蓝=上下文patch(可见)  红=目标patch(被遮挡,待预测)', fontsize=9.5)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(_IMAGES, 'jepa_mask_example.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('[可视化] 掩码示例已保存至 images/jepa_mask_example.png')


def plot_loss_curve(loss_history):
    """绘制训练损失曲线。"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(loss_history, color='#2E86AB', linewidth=1.2, alpha=0.6, label='原始损失')
    window = 20
    if len(loss_history) > window:
        smooth = np.convolve(loss_history, np.ones(window) / window, mode='valid')
        ax.plot(np.arange(window - 1, len(loss_history)), smooth, color='#C0392B',
                 linewidth=2, label=f'滑动平均(window={window})')
    ax.set_xlabel('训练步数')
    ax.set_ylabel('表征空间 L2 损失')
    ax.set_title('玩具 JEPA 训练损失曲线', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(_IMAGES, 'jepa_loss_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('[可视化] 训练损失曲线已保存至 images/jepa_loss_curve.png')


def plot_prediction_error_map(ctx_encoder, tgt_encoder, predictor, images, patch_size, n_h, n_w,
                               n_eval: int = 200):
    """
    在测试图像上评估：逐 patch 位置的平均表征预测误差（cosine 距离），
    以热力图展示"哪些位置更容易/更难预测"。
    """
    patches_all, _, _ = patchify(images, patch_size)
    n_patches = n_h * n_w
    embed_dim = ctx_encoder.pos_embed.size(-1)
    error_sum = np.zeros(n_patches)
    error_count = np.zeros(n_patches)

    ctx_encoder.eval()
    tgt_encoder.eval()
    predictor.eval()
    with torch.no_grad():
        for i in range(min(n_eval, images.shape[0])):
            img_patches = torch.from_numpy(patches_all[i:i+1]).float()
            context_idx_np, target_idx_np = sample_mask(n_patches, mask_ratio=0.4)
            context_idx = torch.from_numpy(context_idx_np).long().unsqueeze(0)
            target_idx = torch.from_numpy(target_idx_np).long().unsqueeze(0)

            ctx_patches = torch.gather(
                img_patches, 1, context_idx.unsqueeze(-1).expand(-1, -1, patch_size ** 2))
            ctx_repr = ctx_encoder(ctx_patches, context_idx)

            full_idx = torch.arange(n_patches).long().unsqueeze(0)
            full_repr = tgt_encoder(img_patches, full_idx)
            tgt_repr = torch.gather(
                full_repr, 1, target_idx.unsqueeze(-1).expand(-1, -1, embed_dim))

            pred_repr = predictor(ctx_repr, context_idx, target_idx)
            cos_sim = F.cosine_similarity(pred_repr, tgt_repr, dim=-1)[0]  # (K_tgt,)
            err = 1 - cos_sim.numpy()                                       # 余弦距离作为误差

            for k, idx in enumerate(target_idx_np):
                error_sum[idx] += err[k]
                error_count[idx] += 1

    avg_error = np.divide(error_sum, np.maximum(error_count, 1))
    avg_error_map = avg_error.reshape(n_h, n_w)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(avg_error_map, cmap='RdYlGn_r', vmin=0)
    plt.colorbar(im, ax=ax, label='平均余弦预测误差 (越低越好)')
    ax.set_title('逐 patch 位置的表征预测误差', fontsize=12, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(_IMAGES, 'jepa_prediction_error_map.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('[可视化] 预测误差热力图已保存至 images/jepa_prediction_error_map.png')
    return avg_error.mean()


# ============================================================================
# 主程序
# ============================================================================

def main():
    print('\n' + '=' * 70)
    print('    wm05 JEPA / V-JEPA：预测表征而非像素 — 完整演示')
    print('=' * 70)

    set_seed(42)

    print('\n[第1步] 绘制 JEPA 架构示意图...')
    plot_jepa_architecture()

    print('\n[第2步] 生成合成图像数据集...')
    images = make_synthetic_images(n=600, size=20)
    print(f'  生成 {images.shape[0]} 张 {images.shape[1]}x{images.shape[2]} 的合成图像')

    patch_size = 4
    patches_all, n_h, n_w = patchify(images, patch_size)
    n_patches = n_h * n_w
    print(f'  每张图像被切分为 {n_h}x{n_w}={n_patches} 个 {patch_size}x{patch_size} patch')

    context_idx, target_idx = sample_mask(n_patches, mask_ratio=0.4)
    plot_mask_example(images[0], patch_size, n_h, n_w, context_idx, target_idx)

    print('\n[第3步] 训练玩具 JEPA（上下文编码器 + 目标编码器(EMA) + 预测器）...')
    ctx_encoder, tgt_encoder, predictor, loss_history, (ps, n_h, n_w) = train_toy_jepa(
        images, n_steps=300, batch_size=32, patch_size=patch_size, embed_dim=32)

    print('\n[第4步] 可视化训练结果...')
    plot_loss_curve(loss_history)
    avg_err = plot_prediction_error_map(
        ctx_encoder, tgt_encoder, predictor, images, patch_size, n_h, n_w)

    print('\n' + '=' * 70)
    print('【总结】')
    print('=' * 70)
    print(f'  训练前表征预测损失: {loss_history[0]:.5f}')
    print(f'  训练后表征预测损失: {np.mean(loss_history[-20:]):.5f}')
    print(f'  测试集平均余弦预测误差: {avg_err:.4f} (0=完美预测, 1=完全无关)')
    print('\n  【JEPA 核心机制回顾】')
    print('  - 预测表征而非像素：损失定义在表征空间，天然过滤像素级噪声')
    print('  - 非对称编码器：上下文编码器可训练，目标编码器只做 EMA 动量更新')
    print('  - 停止梯度 + EMA：防止表征坍缩到常数解的关键设计')
    print('  - I-JEPA(图像) → V-JEPA(视频,预测时空 patch) → V-JEPA 2(更大规模+机器人)')
    print('\n  所有图片已保存至 images/ 目录')
    print('=' * 70)
    print('\n  运行完成！\n')


if __name__ == '__main__':
    main()
