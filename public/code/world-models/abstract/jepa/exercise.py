# -*- coding: utf-8 -*-
from __future__ import annotations
"""
wm05 JEPA / V-JEPA：预测表征而非像素 —— 练习代码
======================================================
请完成以下 3 个 TODO 任务，巩固对 JEPA（联合嵌入预测架构）核心机制的理解：
  TODO 1: 实现 patchify() —— 将图像切分为不重叠的 patch
  TODO 2: 实现 ema_update() —— 用动量方式更新目标编码器参数
  TODO 3: 实现 compute_jepa_loss() —— 组装"上下文/目标掩码划分 + 表征空间损失"

建议先阅读 index.md 和 demo.py，再尝试独立补全代码。
运行方式：在 wm05_jepa/ 目录下执行 python code/exercise.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_IMAGES = os.path.join(_HERE, '..', 'images')
os.makedirs(_IMAGES, exist_ok=True)


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_synthetic_images(n: int, size: int = 20) -> np.ndarray:
    """生成合成灰度图像数据集（与 demo.py 相同，无需修改）。"""
    images = np.zeros((n, size, size), dtype=np.float32)
    for i in range(n):
        n_shapes = np.random.randint(1, 3)
        for _ in range(n_shapes):
            cx, cy = np.random.randint(4, size - 4, size=2)
            r = np.random.randint(2, 5)
            val = np.random.uniform(0.5, 1.0)
            yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
            if np.random.rand() < 0.5:
                mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
            else:
                mask = (np.abs(xx - cx) <= r) & (np.abs(yy - cy) <= r)
            images[i][mask] = val
        images[i] += np.random.normal(0, 0.03, size=(size, size)).astype(np.float32)
    return np.clip(images, 0, 1)


# ============================================================================
# TODO 1: 实现 patchify() —— 图像分块
# ============================================================================

def patchify(images: np.ndarray, patch_size: int = 4):
    """
    TODO 1: 将 (n, H, W) 的图像批量切分为不重叠的 patch。

    参数:
        images: (n, H, W) 图像数组
        patch_size: 每个正方形 patch 的边长

    返回:
        patches: (n, n_patches, patch_size*patch_size) —— 每个 patch 被拉平为向量
        n_h, n_w: 高/宽方向上各有多少个 patch

    实现提示（这是 Vision Transformer / I-JEPA 都要用到的标准操作）：
        1. n, H, W = images.shape
        2. n_h, n_w = H // patch_size, W // patch_size
        3. reshape 成 (n, n_h, patch_size, n_w, patch_size)
        4. transpose 把 patch_size 维度移到一起: (n, n_h, n_w, patch_size, patch_size)
        5. reshape 成 (n, n_h*n_w, patch_size*patch_size)

    预期行为：
        当 images.shape=(10, 20, 20)，patch_size=4 时，
        patches.shape 应为 (10, 25, 16)，n_h=n_w=5
    """
    n, H, W = images.shape
    n_h, n_w = H // patch_size, W // patch_size
    # TODO: 按照上面 5 个步骤完成分块，赋值给 patches
    patches = None  # → TODO: 用 reshape + transpose 实现
    return patches, n_h, n_w


# ============================================================================
# 编码器 / 预测器网络结构（已提供，无需修改）
# ============================================================================

class PatchEncoder(nn.Module):
    """Patch 编码器：Linear 投影 + 位置编码 + 单层 Transformer 自注意力。"""

    def __init__(self, patch_dim: int, n_patches: int, embed_dim: int = 32, n_heads: int = 4):
        super().__init__()
        self.proj = nn.Linear(patch_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, embed_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 2,
            batch_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)

    def forward(self, patches: torch.Tensor, patch_idx: torch.Tensor) -> torch.Tensor:
        x = self.proj(patches)
        pos = self.pos_embed.expand(patches.size(0), -1, -1)
        pos = torch.gather(pos, 1, patch_idx.unsqueeze(-1).expand(-1, -1, pos.size(-1)))
        x = x + pos
        return self.encoder(x)


class Predictor(nn.Module):
    """预测器：用共享掩码 token + 目标位置编码，预测目标 patch 的表征。"""

    def __init__(self, embed_dim: int = 32, n_patches: int = 25, n_heads: int = 4):
        super().__init__()
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, embed_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 2,
            batch_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(embed_dim, embed_dim)

    def forward(self, context_repr, context_idx, target_idx):
        B, K_tgt = target_idx.shape
        embed_dim = context_repr.size(-1)
        pos_all = self.pos_embed.expand(B, -1, -1)
        tgt_pos = torch.gather(pos_all, 1, target_idx.unsqueeze(-1).expand(-1, -1, embed_dim))
        mask_tokens = self.mask_token.expand(B, K_tgt, -1) + tgt_pos
        seq = torch.cat([context_repr, mask_tokens], dim=1)
        out = self.encoder(seq)
        pred = out[:, -K_tgt:, :]
        return self.head(pred)


# ============================================================================
# TODO 2: 实现 ema_update() —— 目标编码器的动量更新
# ============================================================================

@torch.no_grad()
def ema_update(target: nn.Module, context: nn.Module, momentum: float = 0.996):
    """
    TODO 2: 用 EMA（指数滑动平均）方式更新目标编码器参数：

        θ_target ← momentum * θ_target + (1 - momentum) * θ_context

    这是 JEPA / BYOL / DINO 等自监督方法防止"表征坍缩"的关键设计：
    目标网络缓慢跟随上下文网络，但从不接收梯度，
    使预测任务始终有一个"稳定但持续进化"的目标。

    参数:
        target: 目标编码器（只应被本函数原地修改，不参与反向传播）
        context: 上下文编码器（当前正在训练的编码器）
        momentum: 动量系数，越接近 1 更新越缓慢

    实现提示:
        for p_t, p_c in zip(target.parameters(), context.parameters()):
            # TODO: 用 p_t.data.mul_(...) 和 .add_(..., alpha=...) 实现原地更新
            pass

    预期行为：
        调用后，target 的每个参数应变为
        momentum * (更新前的target参数) + (1-momentum) * (context参数)
    """
    for p_t, p_c in zip(target.parameters(), context.parameters()):
        # TODO: 实现 EMA 更新公式（2行代码：先 mul_ 再 add_，或写成一行）
        pass


# ============================================================================
# TODO 3: 实现 compute_jepa_loss() —— 掩码划分 + 表征空间损失
# ============================================================================

def sample_mask(n_patches: int, mask_ratio: float = 0.4):
    """随机选一段连续区间作为目标(被遮挡) patch，其余为上下文 patch（已提供）。"""
    n_target = max(1, int(n_patches * mask_ratio))
    start = np.random.randint(0, n_patches - n_target + 1)
    target_idx = np.arange(start, start + n_target)
    context_idx = np.array([i for i in range(n_patches) if i not in target_idx])
    return context_idx, target_idx


def compute_jepa_loss(batch_patches, context_idx, target_idx,
                       ctx_encoder, tgt_encoder, predictor, embed_dim, patch_dim):
    """
    TODO 3: 组装一次完整的 JEPA 前向计算，返回标量损失。

    参数:
        batch_patches: (B, n_patches, patch_dim) 一个 batch 的全部 patch
        context_idx: (B, K_ctx) 上下文 patch 的位置索引（沿 batch 维已展开为相同索引）
        target_idx: (B, K_tgt) 目标 patch 的位置索引
        ctx_encoder / tgt_encoder / predictor: 三个网络模块
        embed_dim, patch_dim: 维度信息

    实现步骤:
        1. 用 torch.gather 从 batch_patches 中取出上下文 patch: ctx_patches
           （提示：torch.gather(batch_patches, 1,
                    context_idx.unsqueeze(-1).expand(-1, -1, patch_dim))）
        2. ctx_repr = ctx_encoder(ctx_patches, context_idx)
        3. 在 torch.no_grad() 下：
           - 用 tgt_encoder 编码全部 patch（位置索引为 0..n_patches-1）得到 full_repr
           - 用 torch.gather 从 full_repr 中取出目标位置的表征 tgt_repr
        4. pred_repr = predictor(ctx_repr, context_idx, target_idx)
        5. loss = F.mse_loss(pred_repr, tgt_repr)

    易错点:
        - 步骤 3 必须包在 torch.no_grad() 内，否则目标编码器会被误更新
        - gather 的 index 张量维度必须和 src 张量除 gather 维外的维度一致
          （用 .unsqueeze(-1).expand(...) 把 (B,K) 的索引广播成 (B,K,dim)）

    预期输出:
        返回一个 0 维的标量 Tensor，可以直接调用 .backward()
    """
    B, n_patches, _ = batch_patches.shape
    # TODO: 按照上面 5 个步骤实现，最终返回 loss
    loss = None
    return loss


# ============================================================================
# 训练与可视化（已提供，用于验证你的实现是否正确）
# ============================================================================

def train_and_check(images, patch_size=4, embed_dim=32, n_steps=300, batch_size=32):
    patches_all, n_h, n_w = patchify(images, patch_size)
    if patches_all is None:
        print('[提示] TODO 1 (patchify) 尚未实现，无法继续训练。')
        return None
    n_patches = n_h * n_w
    patch_dim = patch_size * patch_size

    ctx_encoder = PatchEncoder(patch_dim, n_patches, embed_dim)
    tgt_encoder = PatchEncoder(patch_dim, n_patches, embed_dim)
    tgt_encoder.load_state_dict(ctx_encoder.state_dict())
    for p in tgt_encoder.parameters():
        p.requires_grad_(False)
    predictor = Predictor(embed_dim, n_patches)
    optimizer = torch.optim.Adam(
        list(ctx_encoder.parameters()) + list(predictor.parameters()), lr=2e-3)

    loss_history = []
    n_images = images.shape[0]
    for step in range(n_steps):
        idx = np.random.choice(n_images, batch_size, replace=False)
        batch_patches = torch.from_numpy(patches_all[idx]).float()
        context_idx_np, target_idx_np = sample_mask(n_patches, mask_ratio=0.4)
        context_idx = torch.from_numpy(context_idx_np).long().unsqueeze(0).expand(batch_size, -1)
        target_idx = torch.from_numpy(target_idx_np).long().unsqueeze(0).expand(batch_size, -1)

        loss = compute_jepa_loss(batch_patches, context_idx, target_idx,
                                  ctx_encoder, tgt_encoder, predictor, embed_dim, patch_dim)
        if loss is None:
            print('[提示] TODO 3 (compute_jepa_loss) 尚未实现，无法继续训练。')
            return None

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        before = tgt_encoder.proj.weight.data.clone()
        ema_update(tgt_encoder, ctx_encoder, momentum=0.996)
        after = tgt_encoder.proj.weight.data
        if step == 0 and torch.allclose(before, after):
            print('[提示] TODO 2 (ema_update) 似乎尚未实现（目标网络参数未发生变化）。')

        loss_history.append(loss.item())
        if (step + 1) % 50 == 0:
            print(f'  Step {step+1:4d}/{n_steps}: loss = {loss.item():.5f}')

    return loss_history


def main():
    print('\n' + '=' * 60)
    print('  wm05 JEPA 练习：补全 patchify / ema_update / compute_jepa_loss')
    print('=' * 60)
    set_seed(42)
    images = make_synthetic_images(n=600, size=20)
    print(f'\n生成 {images.shape[0]} 张合成图像，开始训练玩具 JEPA...\n')
    loss_history = train_and_check(images)

    if loss_history is not None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(loss_history)
        ax.set_xlabel('训练步数')
        ax.set_ylabel('损失')
        ax.set_title('练习：玩具 JEPA 训练损失曲线')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(_IMAGES, 'exercise_jepa_loss_curve.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f'\n训练前损失: {loss_history[0]:.5f}  训练后损失: {np.mean(loss_history[-20:]):.5f}')
        if loss_history[0] > np.mean(loss_history[-20:]) * 1.5:
            print('恭喜！损失明显下降，三个 TODO 实现正确。')
        else:
            print('损失下降不明显，请检查你的实现。')
    print('\n完成！\n')


if __name__ == '__main__':
    main()
