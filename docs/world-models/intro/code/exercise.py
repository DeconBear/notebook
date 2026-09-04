# -*- coding: utf-8 -*-
"""
wm01 世界模型导论与分类 —— 练习代码
=====================================================
请完成以下 TODO 任务，巩固对"世界模型分类体系"的理解。

每个 TODO 都有详细的指示和预期输出描述。
建议先阅读 index.md 和 demo.py，再尝试独立补全代码。
运行方式：在 wm01_world_model_intro/ 目录下执行 python code/exercise.py
"""

import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
from typing import Dict, List


# ============================================================================
# TODO 1: 实现基于规则的世界模型分类器
# ============================================================================
def classify_world_model(features: Dict[str, bool]) -> str:
    """
    TODO 1: 根据一组二值特征，判断某个方法属于世界模型的哪条技术路径。

    这是对方法标签的代码化练习（旧六分法：RSSM/MuZero/JEPA/Genie/VideoGen/LLM）。
    正文已收成五条路径：VideoGen⊂路径一，Genie⊂路径二，RSSM/MuZero/JEPA⊂路径三，LLM⊂路径五。
    每条路径的判定规则（按优先级顺序检查）：

        1. 如果 uses_tree_search=True                     → 'MuZero'
           （用学习到的隐式模型 + 树搜索做规划，如 MuZero/EfficientZero）
        2. 如果 predicts_pixels=False 且 has_explicit_dynamics=True
                                                            → 'JEPA'
           （不重建像素，只在嵌入空间做预测，如 I-JEPA/V-JEPA）
        3. 如果 predicts_pixels=True 且 learns_action_labels=False
                                                            → 'Genie'
           （从无标签视频中自监督学出隐动作空间，如 Genie）
        4. 如果 predicts_pixels=True 且 is_diffusion_or_autoregressive=True
                                                            → 'VideoGen'
           （用扩散/自回归模型直接生成未来视频帧，如 Sora）
        5. 如果 uses_language_tokens=True                  → 'LLM'
           （用语言模型的隐状态充当世界状态，如 LLM-as-simulator）
        6. 否则（有显式的潜空间动力学模型，用于想象规划）  → 'RSSM/Dreamer'
           （如 PlaNet/Dreamer 系列）

    参数:
        features: 一个字典，包含以下布尔键（缺失的键视为 False）：
            - uses_tree_search: 是否使用树搜索（如 MCTS）做规划
            - predicts_pixels: 是否直接预测/重建像素级观测
            - has_explicit_dynamics: 是否有显式的状态转移模型 z_t -> z_{t+1}
            - learns_action_labels: 是否使用真实的动作标签训练（而非无监督学到隐动作）
            - is_diffusion_or_autoregressive: 是否用扩散模型或自回归模型直接生成像素
            - uses_language_tokens: 是否用语言 token 序列作为世界状态表示

    返回:
        技术路径名称字符串，取值之一:
        'RSSM/Dreamer', 'MuZero', 'JEPA', 'Genie', 'VideoGen', 'LLM'
    """
    # TODO: 按照上面描述的优先级规则，实现分类逻辑
    # 提示: 用 features.get(key, False) 安全地读取字典的值（缺失时默认 False）
    uses_tree_search = features.get('uses_tree_search', False)
    predicts_pixels = features.get('predicts_pixels', False)
    has_explicit_dynamics = features.get('has_explicit_dynamics', False)
    learns_action_labels = features.get('learns_action_labels', False)
    is_diffusion_or_autoregressive = features.get('is_diffusion_or_autoregressive', False)
    uses_language_tokens = features.get('uses_language_tokens', False)

    result = None  # ← TODO: 按优先级依次判断，赋值为对应的路径名称字符串

    return result


# ---- 测试 TODO 1 ----
def test_classify_world_model():
    """测试世界模型分类器在六个代表性方法上的表现。"""
    print("=" * 60)
    print("TODO 1 测试: 世界模型分类器")
    print("=" * 60)

    test_cases = [
        # (方法名, 特征字典, 期望的分类结果)
        ('PlaNet/Dreamer', dict(has_explicit_dynamics=True, predicts_pixels=True,
                                 learns_action_labels=True), 'RSSM/Dreamer'),
        ('MuZero', dict(uses_tree_search=True, has_explicit_dynamics=True), 'MuZero'),
        ('V-JEPA', dict(predicts_pixels=False, has_explicit_dynamics=True), 'JEPA'),
        ('Genie', dict(predicts_pixels=True, learns_action_labels=False), 'Genie'),
        ('Sora', dict(predicts_pixels=True, learns_action_labels=True,
                       is_diffusion_or_autoregressive=True), 'VideoGen'),
        ('LLM-as-simulator', dict(uses_language_tokens=True), 'LLM'),
    ]

    n_correct = 0
    for name, features, expected in test_cases:
        result = classify_world_model(features)
        ok = (result == expected)
        n_correct += int(ok)
        mark = '✓' if ok else '✗'
        print(f"  {mark} {name:18s}: 预测={result!r:16s} 期望={expected!r}")

    if n_correct == len(test_cases):
        print(f"\n  全部 {len(test_cases)} 个测试通过！")
    else:
        print(f"\n  {n_correct}/{len(test_cases)} 通过，请检查 classify_world_model 的规则实现。")
    print()


# ============================================================================
# TODO 2: 实现雷达图打分的归一化
# ============================================================================
def normalize_scores(scores: Dict[str, List[float]]) -> Dict[str, List[float]]:
    """
    TODO 2: 对多个方法在多个维度上的打分做"按维度 min-max 归一化"。

    在绘制雷达图/做多方法对比时，原始打分可能不在同一量纲下。
    按维度（列）做 min-max 归一化，可以让每个维度上"最强的方法"
    都归一化到 1.0，"最弱的方法"归一化到 0.0，方便可视化对比。

    公式（对第 j 个维度）:
        normalized[i][j] = (raw[i][j] - min_k(raw[k][j])) / (max_k(raw[k][j]) - min_k(raw[k][j]) + eps)

    参数:
        scores: 字典，键为方法名，值为该方法在各维度上的原始打分列表
                （所有方法的打分列表长度必须相同）

    返回:
        normalized: 结构与输入相同，但每个维度已被归一化到 [0, 1]
    """
    names = list(scores.keys())
    raw = np.array([scores[name] for name in names], dtype=np.float64)  # shape (n_methods, n_dims)

    # TODO: 计算每一维度（每一列）的最小值和最大值
    # 提示: raw.min(axis=0) 和 raw.max(axis=0) 分别得到每列的最小/最大值，shape (n_dims,)
    col_min = None  # ← TODO
    col_max = None  # ← TODO

    # TODO: 按公式做归一化，注意加一个很小的 eps（如 1e-8）避免除零
    normalized_raw = None  # ← TODO: (raw - col_min) / (col_max - col_min + 1e-8)

    if normalized_raw is None:
        return {name: None for name in names}  # TODO 未完成时的占位返回值

    normalized = {name: normalized_raw[i].tolist() for i, name in enumerate(names)}
    return normalized


# ---- 测试 TODO 2 ----
def test_normalize_scores():
    """测试雷达图打分归一化功能。"""
    print("=" * 60)
    print("TODO 2 测试: 雷达图打分归一化")
    print("=" * 60)

    raw_scores = {
        'RSSM/Dreamer': [4, 5, 3],
        'MuZero': [3, 5, 1],
        'JEPA': [4, 2, 2],
    }

    result = normalize_scores(raw_scores)

    if any(v is None for v in result.values()):
        print("  TODO 未完成，请补全 normalize_scores 函数")
        return

    print("  归一化结果:")
    for name, vals in result.items():
        print(f"    {name:14s}: {[round(v, 3) for v in vals]}")

    # 验证：第 1 维 (样本效率) 中，RSSM/Dreamer(4) 和 JEPA(4) 打分相同，应并列最高
    # 第 2 维 (规划能力) 中，RSSM/Dreamer 和 MuZero 都是 5，应归一化为 1.0，JEPA(2) 最低应为 0.0
    dim1_vals = [result[n][1] for n in result]
    if abs(max(dim1_vals) - 1.0) < 1e-6 and abs(min(dim1_vals) - 0.0) < 1e-6:
        print("  ✓ 验证通过：第 2 维（规划能力）的最大值归一化为 1.0，最小值归一化为 0.0")
    else:
        print(f"  ✗ 验证失败：第 2 维归一化结果 = {dim1_vals}，期望包含 1.0 和 0.0")
    print()


# ============================================================================
# TODO 3: 模拟"多步预测误差累积" —— 理解潜空间做梦的优势
# ============================================================================
def simulate_rollout_error_toy(
    horizon: int,
    step_error_rate: float,
    init_error: float = 0.02,
    noise_scale: float = 0.003,
    seed: int = 42,
) -> np.ndarray:
    """
    TODO 3: 实现单次"多步预测误差累积"的玩具模拟（对应 demo.py 中的核心循环）。

    直觉：世界模型做多步 rollout 时，每一步的误差会按一个比例
    (1 + step_error_rate) 复合增长，并叠加一点随机噪声。
    这个函数模拟单条轨迹（不做多次试验平均，避免和 demo.py 重复）。

    递推公式:
        e_0 = init_error
        e_t = e_{t-1} * (1 + step_error_rate) + |noise_t|,  noise_t ~ N(0, noise_scale^2)

    参数:
        horizon: 预测步数
        step_error_rate: 每步误差的复合增长率（像素空间通常更大，潜空间更小）
        init_error: 初始误差
        noise_scale: 每步叠加的随机噪声标准差
        seed: 随机种子（保证函数本身可复现）

    返回:
        errors: shape (horizon,) 的 numpy 数组，errors[t] 是第 t+1 步的累积误差
    """
    rng = np.random.RandomState(seed)                              # 局部随机数生成器，不影响全局种子
    errors = np.zeros(horizon)

    # TODO: 实现上面的递推公式，把每一步的误差存入 errors 数组
    e = init_error
    for t in range(horizon):
        noise = None  # ← TODO: rng.normal(0, noise_scale)
        e = None      # ← TODO: e * (1 + step_error_rate) + abs(noise)
        errors[t] = e if e is not None else 0.0

    return errors


# ---- 测试 TODO 3 ----
def test_simulate_rollout_error():
    """测试多步预测误差累积模拟，并对比不同误差增长率下的最终误差。"""
    print("=" * 60)
    print("TODO 3 测试: 多步预测误差累积模拟")
    print("=" * 60)

    horizon = 20
    pixel_errors = simulate_rollout_error_toy(horizon, step_error_rate=0.045, seed=42)
    latent_errors = simulate_rollout_error_toy(horizon, step_error_rate=0.018, seed=42)

    if np.allclose(pixel_errors, 0) and np.allclose(latent_errors, 0):
        print("  TODO 未完成，请补全 simulate_rollout_error_toy 函数")
        return

    print(f"  像素空间 (step_error=0.045) 第 {horizon} 步累积误差: {pixel_errors[-1]:.4f}")
    print(f"  潜空间   (step_error=0.018) 第 {horizon} 步累积误差: {latent_errors[-1]:.4f}")

    if pixel_errors[-1] > latent_errors[-1]:
        ratio = pixel_errors[-1] / (latent_errors[-1] + 1e-8)
        print(f"  ✓ 验证通过：像素空间误差增长更快，是潜空间的 {ratio:.2f} 倍")
        print("  这正是 RSSM/Dreamer 选择在潜空间中做多步'想象' rollout 的核心原因之一。")
    else:
        print("  ✗ 验证失败：期望像素空间的累积误差应该显著大于潜空间")
    print()


# ============================================================================
# 主程序
# ============================================================================
if __name__ == "__main__":
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║   wm01 世界模型导论与分类 —— 动手练习                        ║")
    print("║   请依次完成 TODO 1, 2, 3                                    ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    test_classify_world_model()
    test_normalize_scores()
    test_simulate_rollout_error()

    print("=" * 60)
    print("所有测试完成！请检查输出结果。")
    print("如有未通过的测试，请回到对应的 TODO 部分补全代码。")
    print("=" * 60)
    print()
    print("提示: 完成 TODO 后，运行 demo.py 查看完整的分类地图和对比可视化。")
    print("  python code/demo.py")
    print()
