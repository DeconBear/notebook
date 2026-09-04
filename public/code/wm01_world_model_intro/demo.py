# -*- coding: utf-8 -*-
"""
wm01 世界模型导论与分类 —— 演示代码
======================================================
功能：
  1. 绘制「世界模型五条技术路径」分类地图
     - 路径一：GAN / VAE / 扩散 / 视频 WM
     - 路径二：Genie / 3D
     - 路径三：PETS / Dreamer / JEPA / LeWM / MuZero
     - 路径四：因果（干预 / 反事实）
     - 路径五：符号 / 神经符号 / LLM 规则
  2. 用玩具仿真展示"为什么要在潜空间里做梦"——对比像素空间
     与潜空间做多步预测（rollout）时的误差累积速度
  3. 绘制六条路径在多个维度上的雷达图对比
     （样本效率 / 规划能力 / 生成质量 / 计算成本 / 可解释性 / 通用性）

每个函数都有中文 docstring，每行逻辑代码都有中文注释。
运行方式：在 wm01_world_model_intro/ 目录下执行 python code/demo.py
依赖: pip install numpy matplotlib
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path as MplPath

# 本章配图含大量中文标注，需显式指定中文字体，否则显示为方框
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ----------------------------------------------------------------------------
# 图片输出目录：脚本相对路径，保证无论从哪个工作目录调用都能正确写入
# ----------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)


def set_seed(seed: int = 42):
    """
    设置随机种子，保证实验可复现。

    参数:
        seed: 随机种子值
    """
    np.random.seed(seed)                                        # NumPy 全局随机种子


# ============================================================================
# 第一部分：世界模型六大技术路径分类地图（taxonomy map）
# ============================================================================

# 每条路径的元数据：(简称, 全称, 代表方法, 颜色, 关键词)
TAXONOMY = [
    dict(
        key='路径一 视频生成',
        full='GAN / VAE / 扩散 / Sora',
        methods=['GAN→VAE→DiT', 'Sora / Cosmos'],
        color='#C1666B',
        note='pixels / video latents',
    ),
    dict(
        key='路径二 交互/3D',
        full='可玩、可漫游',
        methods=['Genie', 'HunyuanWorld / Marble'],
        color='#3B7A57',
        note='action as first-class',
    ),
    dict(
        key='路径三 抽象状态',
        full='便宜的 z 上规划',
        methods=['PETS / Dreamer', 'JEPA / LeWM / MuZero'],
        color='#2E86AB',
        note='latent rollout',
    ),
    dict(
        key='路径四 因果',
        full='P vs P(·|do)',
        methods=['Pearl 阶梯', '干预评测'],
        color='#F18F01',
        note='confounding / intervention',
    ),
    dict(
        key='路径五 符号',
        full='谓词 / 规则 / 程序',
        methods=['pix2pred / COSMOS', 'WALL-E / PoE-World'],
        color='#6A4C93',
        note='executable symbols',
    ),
]


def plot_taxonomy_map(save_name: str = 'wm01-01-taxonomy.png'):
    """
    绘制"世界模型六大技术路径"分类地图。

    以"世界模型"为根节点，向右延伸出六条技术路径分支，
    每条分支下再列出 1-2 个代表性方法，直观呈现整个领域的技术地图。

    参数:
        save_name: 保存的文件名（默认带 wm01- 前缀，作为教学插图）
    """
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # ---- 根节点：世界模型 ----
    root_xy = (1.1, 4.0)
    root_box = FancyBboxPatch(
        (root_xy[0] - 0.9, root_xy[1] - 0.5), 1.8, 1.0,
        boxstyle='round,pad=0.05,rounding_size=0.15',
        linewidth=2, edgecolor='#333333', facecolor='#FFD166', zorder=3,
    )
    ax.add_patch(root_box)
    ax.text(root_xy[0], root_xy[1] + 0.12, 'World Model', ha='center', va='center',
            fontsize=11, fontweight='bold', zorder=4)
    ax.text(root_xy[0], root_xy[1] - 0.22, '世界模型', ha='center', va='center',
            fontsize=9, zorder=4)

    n = len(TAXONOMY)
    y_positions = np.linspace(7.1, 0.9, n)                       # 六条分支纵向均匀分布

    for i, item in enumerate(TAXONOMY):
        y = y_positions[i]
        branch_xy = (5.0, y)

        # ---- 根节点 → 分支节点：贝塞尔曲线箭头 ----
        arrow = FancyArrowPatch(
            root_xy, branch_xy,
            connectionstyle=f"arc3,rad={(y - root_xy[1]) * 0.06}",
            arrowstyle='-|>', mutation_scale=14,
            linewidth=1.6, color=item['color'], alpha=0.75, zorder=1,
        )
        ax.add_patch(arrow)

        # ---- 分支节点（技术路径） ----
        box = FancyBboxPatch(
            (branch_xy[0] - 1.55, branch_xy[1] - 0.42), 3.1, 0.84,
            boxstyle='round,pad=0.05,rounding_size=0.12',
            linewidth=1.8, edgecolor=item['color'], facecolor='white', zorder=3,
        )
        ax.add_patch(box)
        ax.text(branch_xy[0], branch_xy[1] + 0.15, item['key'], ha='center', va='center',
                fontsize=10.5, fontweight='bold', color=item['color'], zorder=4)
        ax.text(branch_xy[0], branch_xy[1] - 0.18, item['full'], ha='center', va='center',
                fontsize=8.3, zorder=4)

        # ---- 分支节点 → 代表方法（叶子节点） ----
        leaf_x = 9.4
        n_leaves = len(item['methods'])
        leaf_ys = np.linspace(y + 0.28 * (n_leaves - 1), y - 0.28 * (n_leaves - 1), n_leaves)
        for m_idx, method in enumerate(item['methods']):
            ly = leaf_ys[m_idx]
            arrow2 = FancyArrowPatch(
                (branch_xy[0] + 1.55, branch_xy[1]), (leaf_x - 1.7, ly),
                connectionstyle="arc3,rad=0.0",
                arrowstyle='-|>', mutation_scale=10,
                linewidth=1.1, color=item['color'], alpha=0.55, zorder=1,
            )
            ax.add_patch(arrow2)
            ax.text(leaf_x - 1.55, ly, method, ha='left', va='center', fontsize=8.6,
                    color='#222222',
                    bbox=dict(boxstyle='round,pad=0.25', facecolor=item['color'],
                              alpha=0.15, edgecolor=item['color'], linewidth=0.8),
                    zorder=4)
        # ---- 关键词标注 ----
        ax.text(branch_xy[0], branch_xy[1] - 0.62, f"« {item['note']} »", ha='center', va='center',
                fontsize=7, color='#666666', style='italic', zorder=4)

    ax.set_title('世界模型五条技术路径分类地图',
                  fontsize=13, fontweight='bold', pad=14)
    plt.tight_layout()
    out_path = os.path.join(_IMAGES_DIR, save_name)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[可视化] 分类地图已保存至 images/{save_name}")


# ============================================================================
# 第二部分：为什么要在"潜空间"里做梦？—— rollout 误差累积对比
# ============================================================================

def simulate_rollout_error(
    horizon: int = 30,
    pixel_step_error: float = 0.045,
    latent_step_error: float = 0.018,
    n_trials: int = 200,
):
    """
    用玩具仿真模拟"多步预测(rollout)误差累积"现象。

    直觉：世界模型做规划/想象时，需要连续预测未来 H 步。
    如果模型直接在像素空间里预测（下一帧图像 → 下一帧图像 → ...），
    每一步的小误差会在下一步的输入中被放大，导致误差指数级累积；
    而在压缩后的低维潜空间里预测（RSSM/Dreamer 的做法），
    每一步误差本身更小，且不需要重建像素细节，误差增长明显更慢。

    这里用简化的随机游走模型来类比：
        error_t = error_{t-1} * (1 + step_error) + noise
    "像素空间"用更大的 step_error 模拟（因为像素预测需要建模大量与决策无关的
    高频细节，误差更容易放大和复合），"潜空间"用更小的 step_error 模拟。

    参数:
        horizon: 预测的步数（rollout 长度）
        pixel_step_error: 像素空间每步的误差放大率
        latent_step_error: 潜空间每步的误差放大率
        n_trials: 蒙特卡洛模拟次数（用于得到平滑的均值曲线）

    返回:
        pixel_errors: shape (horizon,)，像素空间的平均累积误差
        latent_errors: shape (horizon,)，潜空间的平均累积误差
    """
    pixel_curves = np.zeros((n_trials, horizon))
    latent_curves = np.zeros((n_trials, horizon))

    for trial in range(n_trials):
        # 初始误差（观测/编码噪声），两种模型共享同一起点保证公平对比
        e_pixel = 0.02
        e_latent = 0.02
        for t in range(horizon):
            # 每步误差以 (1+step_error) 的比例复合增长，并叠加一点随机噪声
            noise_p = np.random.normal(0, 0.003)
            noise_l = np.random.normal(0, 0.003)
            e_pixel = e_pixel * (1 + pixel_step_error) + abs(noise_p)
            e_latent = e_latent * (1 + latent_step_error) + abs(noise_l)
            pixel_curves[trial, t] = e_pixel
            latent_curves[trial, t] = e_latent

    return pixel_curves.mean(axis=0), latent_curves.mean(axis=0), pixel_curves.std(axis=0), latent_curves.std(axis=0)


def plot_rollout_error_comparison(
    pixel_errors, latent_errors, pixel_std, latent_std,
    save_name: str = 'rollout_error_comparison.png',
):
    """
    绘制像素空间 vs 潜空间的多步预测误差累积对比图。

    参数:
        pixel_errors / latent_errors: 两种空间下每一步的平均累积误差
        pixel_std / latent_std: 对应的标准差（用于画置信区间阴影带）
        save_name: 输出文件名
    """
    horizon = len(pixel_errors)
    steps = np.arange(1, horizon + 1)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(steps, pixel_errors, color='#C1666B', linewidth=2.2, label='像素空间 rollout（如直接预测下一帧图像）')
    ax.fill_between(steps, pixel_errors - pixel_std, pixel_errors + pixel_std, color='#C1666B', alpha=0.15)

    ax.plot(steps, latent_errors, color='#2E86AB', linewidth=2.2, label='潜空间 rollout（RSSM/Dreamer 的做法）')
    ax.fill_between(steps, latent_errors - latent_std, latent_errors + latent_std, color='#2E86AB', alpha=0.15)

    ax.set_xlabel('想象/规划步数 (imagination horizon)', fontsize=10)
    ax.set_ylabel('累积预测误差（玩具尺度，非真实单位）', fontsize=10)
    ax.set_title('为什么在潜空间里"做梦"？—— 多步预测误差累积对比（玩具仿真）',
                  fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(_IMAGES_DIR, save_name)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[可视化] 潜空间 vs 像素空间误差累积对比图已保存至 images/{save_name}")


# ============================================================================
# 第三部分：六条路径的多维度雷达图对比
# ============================================================================

# 六个评价维度上的主观打分（1-5，5 为最优），仅用于教学直觉，非严格科学评估
RADAR_DIMENSIONS = ['样本效率', '规划能力', '生成质量', '计算成本(越低越好)', '可解释性', '通用性']
RADAR_SCORES = {
    '路径一 视频生成': [1, 2, 5, 1, 1, 4],
    '路径二 交互/3D':   [2, 3, 4, 2, 2, 3],
    '路径三 抽象状态': [4, 5, 3, 3, 3, 3],
    '路径四 因果':     [3, 4, 2, 3, 4, 3],
    '路径五 符号':     [4, 4, 2, 4, 5, 3],
}


def plot_radar_comparison(save_name: str = 'world_model_radar_comparison.png'):
    """
    绘制五条世界模型技术路径在多个维度上的雷达图对比。

    评分是教学用的主观定性打分（1-5），帮助建立"没有一种路径全面占优，
    需要根据任务需求选择"的直觉，而非严格的定量评测结果。

    参数:
        save_name: 输出文件名
    """
    dims = RADAR_DIMENSIONS
    n_dims = len(dims)
    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]                                          # 闭合雷达图

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = [item['color'] for item in TAXONOMY]

    for (name, scores), color in zip(RADAR_SCORES.items(), colors):
        values = scores + scores[:1]                              # 闭合曲线
        ax.plot(angles, values, linewidth=2, label=name, color=color)
        ax.fill(angles, values, alpha=0.06, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims, fontsize=10)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['1', '2', '3', '4', '5'], fontsize=8)
    ax.set_ylim(0, 5)
    ax.set_title('世界模型五条技术路径 —— 多维度直觉对比（主观定性打分）',
                  fontsize=12, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    plt.tight_layout()
    out_path = os.path.join(_IMAGES_DIR, save_name)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[可视化] 六条路径雷达图对比已保存至 images/{save_name}")


# ============================================================================
# 主程序
# ============================================================================

def main():
    """
    主程序：生成 wm01 章节的三张核心配图。
    """
    print("\n" + "=" * 70)
    print("    wm01 世界模型导论与分类 —— 完整演示")
    print("=" * 70)

    set_seed(42)                                                   # 固定随机种子，保证可复现

    # ---- 1. 分类地图 ----
    print("\n[步骤 1] 绘制世界模型五条技术路径分类地图...")
    plot_taxonomy_map()

    # ---- 2. rollout 误差累积对比 ----
    print("\n[步骤 2] 模拟像素空间 vs 潜空间的多步预测误差累积...")
    pixel_err, latent_err, pixel_std, latent_std = simulate_rollout_error(
        horizon=30, pixel_step_error=0.045, latent_step_error=0.018, n_trials=200,
    )
    print(f"  第 10 步：像素空间累积误差={pixel_err[9]:.4f}, 潜空间累积误差={latent_err[9]:.4f}")
    print(f"  第 30 步：像素空间累积误差={pixel_err[29]:.4f}, 潜空间累积误差={latent_err[29]:.4f}")
    print(f"  30 步后像素空间误差是潜空间的 {pixel_err[29] / latent_err[29]:.1f} 倍")
    plot_rollout_error_comparison(pixel_err, latent_err, pixel_std, latent_std)

    # ---- 3. 雷达图对比 ----
    print("\n[步骤 3] 绘制五条技术路径的多维度雷达图对比...")
    plot_radar_comparison()

    # ---- 总结 ----
    print("\n" + "=" * 70)
    print("【总结】")
    print("=" * 70)
    print("  世界模型的核心思想：学习环境动力学的内部预测模型，")
    print("  使 Agent 能够'在脑海中'模拟未来，而不必每次都与真实环境交互。")
    print("\n  五条技术路径各有侧重：")
    for item in TAXONOMY:
        print(f"    - {item['key']:12s} ({item['full']}): {', '.join(item['methods'])}")
    print("\n  下一节进入路径一：GAN → VAE → 扩散 → 视频世界模型。")
    print(f"\n  所有图片已保存至 {_IMAGES_DIR}")
    print("=" * 70)
    print("\n  运行完成！\n")


if __name__ == "__main__":
    main()
