# -*- coding: utf-8 -*-
"""
s21 RLHF：当强化学习遇见大模型 — 演示代码 (Toy/Simulated 版本)
==================================================================
⚠️ 重要说明: 完整 RLHF 训练（PPO 在数十亿参数模型上）需要数百 GPU 天。
              本 demo 实现一个学术/教学用的简化版本，包含 PPO 和 DPO 的
              核心概念，在小规模合成数据上运行，可在普通 CPU 上完成训练。

内容:
  1. 创建一个玩具"语言模型"（小型 LSTM，小词汇表）
  2. 实现一个基于规则的奖励模型（reward model）
  3. 从零实现 PPO（含裁剪目标、GAE 优势估计、KL 惩罚）
  4. 实现最小化 DPO
  5. 对比 PPO vs DPO 训练稳定性
  6. 可视化: 训练曲线、KL 散度、策略熵、输出分布变化

核心公式 (PPO):
  L_CLIP = E[min(r_t(θ)·Â_t, clip(r_t(θ), 1-ε, 1+ε)·Â_t)]
  其中 r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)
  R_total = R_RM - β·KL(π_θ || π_ref)

核心公式 (DPO):
  L_DPO = -E[log σ(β·log(π_θ(y_w)/π_ref(y_w)) - β·log(π_θ(y_l)/π_ref(y_l)))]

每个函数都有中文 docstring，每行逻辑代码都有中文注释。
运行方式：在 s21_rlhf/ 目录下执行 python code/demo.py
依赖: pip install numpy matplotlib torch
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from typing import List, Tuple, Dict, Optional, Deque
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


# ============================================================================
# 第一部分：环境设置与工具函数
# ============================================================================

# ---- 配置 ----
VOCAB_SIZE = 30                                  # 词汇表大小 (a-z + 4 特殊 token)
EMBED_DIM = 64                                   # 词嵌入维度
HIDDEN_DIM = 128                                 # LSTM 隐藏层维度
MAX_SEQ_LEN = 20                                 # 最大序列长度
# 特殊 token
PAD_TOKEN = 0                                    # 填充 token <PAD>
BOS_TOKEN = 1                                    # 开始 token <BOS>
EOS_TOKEN = 2                                    # 结束 token <EOS>
UNK_TOKEN = 3                                    # 未知 token <UNK>
# 字母 token 从索引 4 开始: a=4, b=5, ..., z=29
CHAR_OFFSET = 4                                  # 字母 token 的起始索引


def set_seed(seed: int = 42):
    """设置随机种子以保证可复现性。"""
    np.random.seed(seed)
    torch.manual_seed(seed)


def char_to_token(ch: str) -> int:
    """将字符转换为 token 索引。"""
    if 'a' <= ch <= 'z':
        return CHAR_OFFSET + (ord(ch) - ord('a'))                # a→4, b→5, ..., z→29
    return UNK_TOKEN                                              # 未知字符


def token_to_char(token: int) -> str:
    """将 token 索引转换为字符。"""
    if CHAR_OFFSET <= token < CHAR_OFFSET + 26:
        return chr(ord('a') + (token - CHAR_OFFSET))             # 4→'a', 5→'b', ...
    elif token == BOS_TOKEN:
        return '<BOS>'
    elif token == EOS_TOKEN:
        return '<EOS>'
    elif token == PAD_TOKEN:
        return '<PAD>'
    return '<UNK>'


def decode_tokens(tokens: List[int]) -> str:
    """将 token 列表解码为字符串。"""
    return ''.join(token_to_char(t) for t in tokens
                  if t >= CHAR_OFFSET)                            # 跳过特殊 token


# ============================================================================
# 第二部分：玩具语言模型（小型 LSTM）
# ============================================================================

class ToyLanguageModel(nn.Module):
    """
    玩具语言模型 —— 一个小型的 LSTM-based 自回归模型。

    用于模拟真实 LLM 在 RLHF 中的角色：
    - 输入: token 序列（prompt + 已生成部分）
    - 输出: 下一个 token 的概率分布 π_θ(a_t | s_t)

    架构: Embedding → LSTM → Linear → Softmax
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = EMBED_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = 2,
    ):
        """
        初始化玩具语言模型。

        参数:
            vocab_size: 词汇表大小
            embed_dim: 词嵌入维度
            hidden_dim: LSTM 隐藏层维度
            num_layers: LSTM 层数
        """
        super(ToyLanguageModel, self).__init__()
        self.vocab_size = vocab_size                              # 词汇表大小
        self.embed_dim = embed_dim                                # 嵌入维度
        self.hidden_dim = hidden_dim                              # 隐藏层维度

        self.embedding = nn.Embedding(vocab_size, embed_dim)     # Token 嵌入层
        self.lstm = nn.LSTM(embed_dim, hidden_dim,               # LSTM 层
                           num_layers=num_layers,
                           batch_first=True)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)          # 输出层: hidden → vocab

        # 初始化权重
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)    # 小标准差正态初始化

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播: token 序列 → 每个位置的 logits。

        参数:
            input_ids: token ID 张量，shape (batch, seq_len)
            hidden: LSTM 初始隐藏状态，如果为 None 则初始化为零

        返回:
            logits: shape (batch, seq_len, vocab_size)
            hidden: LSTM 最终的隐藏状态
        """
        embeds = self.embedding(input_ids)                        # (batch, seq_len, embed_dim)
        lstm_out, hidden = self.lstm(embeds, hidden)             # LSTM 前向计算
        logits = self.lm_head(lstm_out)                           # (batch, seq_len, vocab_size)
        return logits, hidden

    def get_log_probs(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算序列中每个位置 token 的 log 概率 log π(a_t | s_t)。

        参数:
            input_ids: token 序列，(batch, seq_len)

        返回:
            log_probs: 每个位置的 log 概率，(batch, seq_len - 1)
                      （第 t 个位置是给定 s_{:t} 时 a_t 的 log 概率）
        """
        logits, _ = self.forward(input_ids)                       # (batch, seq_len, vocab_size)
        # 预测: logits[:, :-1, :] 对应前 seq_len-1 个位置的输出
        # 标签: input_ids[:, 1:] 对应后 seq_len-1 个位置的输入
        log_probs_all = F.log_softmax(logits[:, :-1, :], dim=-1)  # (batch, seq_len-1, vocab)
        # 取出实际 token 对应的 log 概率
        targets = input_ids[:, 1:]                                # (batch, seq_len-1)
        log_probs = log_probs_all.gather(                         # gather 选取对应位置
            2, targets.unsqueeze(-1)
        ).squeeze(-1)                                             # (batch, seq_len-1)
        return log_probs

    def generate(
        self,
        prompt: torch.Tensor,
        max_len: int = MAX_SEQ_LEN,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        自回归生成 token 序列。

        参数:
            prompt: 初始 prompt token，(1, prompt_len)
            max_len: 最大生成长度
            temperature: 采样温度 (1.0 = 原始分布, <1.0 = 更确定, >1.0 = 更随机)

        返回:
            generated: 完整序列（含 prompt），(1, prompt_len + generated_len)
            log_probs: 每个生成步骤的 log 概率列表
        """
        self.eval()                                                # 评估模式
        generated = prompt.clone()                                 # 从 prompt 开始
        log_probs = []                                             # 记录 log 概率
        hidden = None                                              # LSTM 初始状态

        with torch.no_grad():
            for _ in range(max_len):
                # 取最后一个 token 作为输入
                logits, hidden = self.forward(generated[:, -1:], hidden)  # (1, 1, vocab)
                logits = logits.squeeze(1) / temperature          # 温度缩放: 控制采样随机性

                probs = F.softmax(logits, dim=-1)                  # softmax → 概率分布
                dist = Categorical(probs)                          # 类别分布
                next_token = dist.sample()                         # 采样下一个 token
                log_probs.append(dist.log_prob(next_token))        # 记录 log 概率

                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

                # 如果生成 EOS token 则停止
                if next_token.item() == EOS_TOKEN:
                    break

        self.train()                                               # 恢复训练模式
        return generated, log_probs


# ============================================================================
# 第三部分：奖励模型 (基于规则)
# ============================================================================

class RuleBasedRewardModel:
    """
    基于规则的奖励模型 —— 用于模拟 RLHF 中 RM 的角色。

    在实际 RLHF 中，RM 是一个训练好的神经网络。在这里我们用
    规则来近似，目的是展示 RL 流程而非追求真实奖励质量。

    奖励规则:
    1. 长度奖励: 序列越长越好（但不是越长越好，适中长度最优）(0 ~ 2 分)
    2. 多样性奖励: 词汇多样性（使用的独特字符越多越好）(0 ~ 3 分)
    3. 连贯性奖励: 元音-辅音交替（简单的连贯性代理）(0 ~ 3 分)
    4. 短序列惩罚: 太短的序列被惩罚 (-3 分)
    5. 重复惩罚: 连续重复 token 被惩罚 (最多 -2 分)
    """

    def score(self, tokens: List[int]) -> float:
        """
        对一个 token 序列打分。

        参数:
            tokens: token ID 列表

        返回:
            reward: 标量分数，范围约 [-3, 8]
        """
        if len(tokens) == 0:
            return -3.0                                            # 空序列严重惩罚

        # 只考虑字母 token (>= CHAR_OFFSET)
        char_tokens = [t for t in tokens if t >= CHAR_OFFSET]

        if len(char_tokens) == 0:
            return -3.0                                            # 没有字母 token

        # 1. 长度奖励: 高斯形状，最优长度 ~15
        length = len(char_tokens)
        length_reward = 2.0 * np.exp(-((length - 15) ** 2) / 50)  # 15 附近最高

        # 2. 多样性奖励: 独特 token 比例
        unique_ratio = len(set(char_tokens)) / max(1, len(char_tokens))
        diversity_reward = 3.0 * unique_ratio                      # 高多样性 = 高奖励

        # 3. 连贯性: 元音-辅音模式
        vowels = set(['a', 'e', 'i', 'o', 'u'])
        coherence_reward = 0.0
        for i in range(1, len(char_tokens)):
            ch_prev = token_to_char(char_tokens[i-1])
            ch_curr = token_to_char(char_tokens[i])
            prev_is_vowel = ch_prev in vowels
            curr_is_vowel = ch_curr in vowels
            # 元音和辅音交替 = 更连贯
            if prev_is_vowel != curr_is_vowel:
                coherence_reward += 3.0 / max(1, len(char_tokens))  # 最大 3.0

        # 4. 短序列惩罚
        short_penalty = max(0, 3.0 - length) * 1.0                # < 3 个字母则受惩罚

        # 5. 重复惩罚
        repeat_penalty = 0.0
        for i in range(1, len(tokens)):
            if tokens[i] == tokens[i-1]:                           # 连续相同 token
                repeat_penalty += 0.5                               # 每次累加惩罚

        total = (length_reward + diversity_reward
                + coherence_reward - short_penalty
                - min(2.0, repeat_penalty))                       # 最大重复惩罚 2.0

        return total


# ============================================================================
# 第四部分：训练数据集 (合成玩具数据)
# ============================================================================

def create_toy_dataset(num_prompts: int = 200) -> List[str]:
    """
    创建玩具数据集 —— 简单的 prompt 列表。

    每个 prompt 是一个简短的英文单词或短语，
    模型（在 RLHF 训练后）将学习生成"连贯多样"的回复。

    参数:
        num_prompts: 生成的 prompt 数量

    返回:
        prompts: 字符串列表
    """
    # 一组简单的 prompt 主题
    prompt_templates = [
        "hello", "thank", "please", "world", "learn",
        "think", "write", "speak", "teach", "dream",
        "build", "plant", "water", "light", "stone",
        "bread", "house", "train", "ocean", "music",
    ]

    # 循环使用模板生成足够数量的 prompt
    prompts = []
    for i in range(num_prompts):
        base = prompt_templates[i % len(prompt_templates)]
        prompts.append(base)

    return prompts


def encode_prompt(prompt: str) -> torch.Tensor:
    """
    将 prompt 字符串编码为 token 张量。

    格式: [BOS, char1, char2, ..., char_n]

    参数:
        prompt: 原始字符串

    返回:
        token_ids: shape (1, len(prompt) + 1)
    """
    tokens = [BOS_TOKEN]                                          # 加入 BOS token
    for ch in prompt.lower():                                    # 转小写
        if 'a' <= ch <= 'z':
            tokens.append(char_to_token(ch))                     # 字母 → token
    return torch.tensor([tokens], dtype=torch.long)              # (1, seq_len)


# ============================================================================
# 第五部分：PPO 实现
# ============================================================================

class PPOAgent:
    """
    PPO Agent —— 实现 Proximal Policy Optimization。

    核心组件:
    - Policy (Actor): 玩具语言模型 (ToyLanguageModel)
    - Value (Critic): 价值网络 V_ψ(s)，估计状态价值
    - Reference Model: 冻结的初始模型，用于 KL 惩罚

    超参数:
    - clip_epsilon: PPO 裁剪参数 ε (默认 0.2)
    - kl_coef: KL 惩罚系数 β
    - gamma: 折扣因子 γ
    - gae_lambda: GAE λ 参数
    - value_coef: Value 损失系数
    - entropy_coef: 熵奖励系数 (鼓励探索)
    """

    def __init__(
        self,
        policy: ToyLanguageModel,
        ref_model: ToyLanguageModel,
        value_network: nn.Module,
        lr: float = 1e-4,
        clip_epsilon: float = 0.2,
        kl_coef: float = 0.1,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        value_coef: float = 1.0,
        entropy_coef: float = 0.01,
        device: str = 'cpu',
    ):
        """
        初始化 PPO Agent。

        参数:
            policy: 策略网络（Actor）
            ref_model: 参考模型（冻结的 SFT 模型）
            value_network: 价值网络（Critic）
            lr: 学习率
            clip_epsilon: PPO 裁剪范围 ε
            kl_coef: KL 散度惩罚系数 β
            gamma: 折扣因子 γ
            gae_lambda: GAE 参数 λ
            value_coef: Value 损失权重
            entropy_coef: 熵奖励权重
            device: 设备
        """
        self.policy = policy                                      # Actor: 策略 π_θ
        self.ref_model = ref_model                                # 参考模型 π_ref (冻结)
        self.value_network = value_network                        # Critic: V_ψ
        self.clip_epsilon = clip_epsilon                          # PPO 裁剪参数 ε
        self.kl_coef = kl_coef                                    # KL 惩罚系数 β
        self.gamma = gamma                                        # 折扣因子 γ
        self.gae_lambda = gae_lambda                              # GAE λ
        self.value_coef = value_coef                              # Value 损失系数
        self.entropy_coef = entropy_coef                          # 熵系数
        self.device = device

        # ---- 优化器 ----
        # Actor 和 Critic 分别使用独立的优化器（实践中常用）
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=lr)

        # 冻结参考模型
        for p in self.ref_model.parameters():
            p.requires_grad = False                                # 参考模型不更新

        # ---- 训练记录 ----
        self.policy_loss_history = []                              # PPO 策略损失
        self.value_loss_history = []                               # Value 损失
        self.kl_history = []                                       # KL 散度
        self.entropy_history = []                                  # 策略熵
        self.reward_history = []                                   # 每 episode 的总奖励

    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        next_value: float,
        dones: List[bool],
    ) -> torch.Tensor:
        """
        计算 GAE（Generalized Advantage Estimation）优势估计。

        GAE 公式:
            Â_t^{GAE(γ,λ)} = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}

        其中 δ_t = r_t + γ·V(s_{t+1}) - V(s_t)

        实现方法: 从后往前递推计算。

        参数:
            rewards: 每步的奖励 r_t
            values: 每步的状态价值 V(s_t)
            next_value: 下一步的状态价值 V(s_{T+1}) (如果是最后一步则为 0)
            dones: 每步是否终止 (True=终止)

        返回:
            advantages: GAE 优势估计，shape (T,)
        """
        T = len(rewards)                                           # 步数
        advantages = torch.zeros(T, device=self.device)            # 初始化优势
        gae = 0.0                                                  # 累积优势

        for t in reversed(range(T)):
            if t == T - 1:
                delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            else:
                delta = (rewards[t]
                        + self.gamma * values[t + 1] * (1 - dones[t])
                        - values[t])
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae                                    # Â_t^{GAE}

        return advantages

    def compute_kl_divergence(
        self,
        log_probs_policy: torch.Tensor,
        log_probs_ref: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算当前策略与参考模型之间的 KL 散度。

        KL(π_θ || π_ref) ≈ mean(log π_θ - log π_ref)

        参数:
            log_probs_policy: 策略的 log 概率
            log_probs_ref: 参考模型的 log 概率

        返回:
            kl: KL 散度估计（标量）
        """
        kl = (log_probs_policy - log_probs_ref).mean()            # KL 近似估计
        return kl

    def ppo_update(
        self,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        values: torch.Tensor,
        ref_log_probs: torch.Tensor,
    ) -> Dict[str, float]:
        """
        执行一次 PPO 更新（一个 mini-batch）。

        PPO 裁剪损失:
            L_CLIP = E[min(r_t(θ)·Â_t, clip(r_t(θ), 1-ε, 1+ε)·Â_t)]

        其中 r_t(θ) = exp(log π_θ - log π_old)

        参数:
            old_log_probs: 旧策略的 log 概率
            advantages: GAE 优势估计 Â_t
            returns: 折扣累计回报（用于 Critic 的 TD 目标）
            states: 状态嵌入 (从序列编码得到)
            actions: 实际选择的 token
            values: Critic 的旧估计值
            ref_log_probs: 参考模型的 log 概率

        返回:
            metrics: 包含各项损失和 KL 散度的字典
        """
        # ---- 1. 计算概率比率 r_t(θ) ----
        # 使用 log 空间避免数值问题: r = exp(log π_new - log π_old)
        new_log_probs = self.policy.get_log_probs(actions)        # 新策略的 log 概率
        # 确保长度一致（可能因为自回归偏移差 1）
        min_len = min(len(old_log_probs), len(new_log_probs.flatten()))
        old_lp = old_log_probs[:min_len]
        new_lp = new_log_probs.flatten()[:min_len]
        ref_lp = ref_log_probs[:min_len]
        adv = advantages[:min_len]

        log_ratio = new_lp - old_lp.detach()                      # log r_t(θ)
        ratio = torch.exp(log_ratio)                              # r_t(θ)

        # ---- 2. 计算 PPO 裁剪损失 ----
        # 未裁剪的目标: r_t(θ) * Â_t
        surr1 = ratio * adv
        # 裁剪后的目标: clip(r, 1-ε, 1+ε) * Â_t
        surr2 = torch.clamp(ratio,                               # 限制比率在 [1-ε, 1+ε]
                           1.0 - self.clip_epsilon,
                           1.0 + self.clip_epsilon) * adv
        # 取 min: 当 advantage > 0 时防止 r 过大，当 advantage < 0 时防止 r 过小
        policy_loss = -torch.min(surr1, surr2).mean()             # 负号因为梯度下降

        # ---- 3. KL 惩罚 (加到 reward 中) ----
        kl_div = self.compute_kl_divergence(new_lp, ref_lp)       # KL(π_θ || π_ref)
        policy_loss = policy_loss + self.kl_coef * kl_div          # 加入 KL 惩罚

        # ---- 4. 更新 Actor (策略网络) ----
        self.policy_optimizer.zero_grad()
        policy_loss.backward()                                     # 反向传播
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)  # 梯度裁剪
        self.policy_optimizer.step()                               # 更新参数

        # ---- 5. 计算并更新 Critic (价值网络) ----
        # Value 损失: MSE(V(s_t), R_t) 其中 R_t 是累计回报
        values_pred = self.value_network(states[:min_len])        # 当前价值估计
        value_loss = F.mse_loss(values_pred.flatten(),            # MSE 损失
                               returns[:min_len])

        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 0.5)
        self.value_optimizer.step()

        # ---- 6. 计算策略熵 (衡量探索程度) ----
        entropy = -new_lp.mean()                                   # -E[log π] = 熵

        metrics = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'kl_divergence': kl_div.item(),
            'entropy': entropy.item(),
            'mean_ratio': ratio.mean().item(),
        }
        return metrics

    def generate_and_score(
        self,
        prompt: torch.Tensor,
        reward_model: RuleBasedRewardModel,
    ) -> Dict:
        """
        生成一条轨迹并获取奖励。

        参数:
            prompt: 编码后的 prompt
            reward_model: 奖励模型

        返回:
            trajectory: 包含 states, actions, rewards, log_probs 等的字典
        """
        # ---- 用策略生成回复 ----
        generated, gen_log_probs = self.policy.generate(prompt, MAX_SEQ_LEN)
        # 生成部分的 token（不含 prompt）
        gen_tokens = generated[0, len(prompt[0]):].tolist()

        # ---- 用参考模型计算 log 概率 ----
        with torch.no_grad():
            ref_log_probs = self.ref_model.get_log_probs(generated)  # (1, full_len-1)
            ref_log_probs_flat = ref_log_probs.flatten()
            # 取生成部分对应的 ref log 概率
            ref_log_probs_gen = ref_log_probs_flat[len(prompt[0])-1:]

        # ---- 获取奖励 ----
        rm_score = reward_model.score(gen_tokens)                  # 奖励模型打分
        kl_penalty_term = 0.0                                      # KL 惩罚单独计算

        # ---- 计算 Value 估计 ----
        with torch.no_grad():
            # 对完整序列进行编码以获取 value
            embeds = self.policy.embedding(generated)              # (1, seq_len, embed)
            values = self.value_network(embeds).squeeze(0)        # (seq_len,)
            # 取生成部分对应的 values
            values_gen = values[len(prompt[0]):]                  # (gen_len,)

        # ---- 构造每步的奖励 ----
        n_gen = len(gen_log_probs)                                 # 生成了多少步
        rewards_per_step = [0.0] * n_gen                           # 中间步奖励为 0
        if n_gen > 0:
            rewards_per_step[-1] = rm_score                        # 只在最后一步给奖励

        # ---- 组装轨迹 ----
        trajectory = {
            'generated': generated,                                # 完整序列
            'gen_tokens': gen_tokens,                              # 生成部分 token
            'log_probs': torch.stack(gen_log_probs).flatten() if gen_log_probs
                         else torch.tensor([]),                   # 策略 log 概率
            'ref_log_probs_gen': ref_log_probs_gen,               # 参考模型 log 概率
            'rewards': rewards_per_step,                           # 每步奖励
            'values': values_gen,                                  # 每步价值
            'rm_score': rm_score,                                  # RM 原始分数
            'states_embedded': embeds[0, len(prompt[0]):],        # 生成部分嵌入
        }
        return trajectory


class ValueNetwork(nn.Module):
    """
    价值网络 V_ψ(s) —— PPO 中的 Critic。

    输入: 状态嵌入（来自 LM 的 embedding 输出）
    输出: 标量状态价值 V(s)

    架构: 简单的全连接网络
    """

    def __init__(self, embed_dim: int = EMBED_DIM, hidden_dim: int = 64):
        """
        初始化价值网络。

        参数:
            embed_dim: 输入嵌入维度
            hidden_dim: 隐藏层维度
        """
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)               # 输入层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)              # 隐藏层
        self.fc3 = nn.Linear(hidden_dim, 1)                        # 输出标量

    def forward(self, embeds: torch.Tensor) -> torch.Tensor:
        """
        前向传播: 嵌入 → 状态价值。

        参数:
            embeds: 状态嵌入，(batch, seq_len, embed_dim) 或 (seq_len, embed_dim)

        返回:
            values: 状态价值估计，(batch, seq_len) 或 (seq_len,)
        """
        x = F.relu(self.fc1(embeds))                               # ReLU
        x = F.relu(self.fc2(x))                                    # ReLU
        values = self.fc3(x).squeeze(-1)                           # (..., 1) → (...)
        return values


# ============================================================================
# 第六部分：DPO 实现
# ============================================================================

def compute_dpo_loss(
    policy: ToyLanguageModel,
    ref_model: ToyLanguageModel,
    prompt: torch.Tensor,
    y_w: torch.Tensor,                                            # 偏好回复 (win)
    y_l: torch.Tensor,                                            # 不偏好回复 (lose)
    beta: float = 0.1,                                            # DPO 的 β 参数
) -> torch.Tensor:
    """
    DPO (Direct Preference Optimization) 损失函数。

    公式:
        L_DPO = -log σ( β·log(π_θ(y_w|x)/π_ref(y_w|x))
                      - β·log(π_θ(y_l|x)/π_ref(y_l|x)) )

    其中:
        - π_θ: 当前策略
        - π_ref: 参考模型（冻结的初始模型）
        - y_w: 被偏好的回复
        - y_l: 不被偏好的回复

    参数:
        policy: 当前策略模型 π_θ
        ref_model: 参考模型 π_ref (冻结)
        prompt: prompt token 序列
        y_w: 偏好回复的完整序列
        y_l: 不偏好回复的完整序列
        beta: DPO 温度参数

    返回:
        loss: DPO 损失值（标量）
    """
    # ---- 1. 计算当前策略的 log 概率 ----
    # y_w 的 log 概率
    log_p_w = policy.get_log_probs(y_w)                           # (1, seq-1)
    total_log_p_w = log_p_w.sum()                                  # 序列总 log 概率

    # y_l 的 log 概率
    log_p_l = policy.get_log_probs(y_l)                           # (1, seq-1)
    total_log_p_l = log_p_l.sum()                                  # 序列总 log 概率

    # ---- 2. 计算参考模型的 log 概率 ----
    with torch.no_grad():                                          # 参考模型不计算梯度
        ref_log_p_w = ref_model.get_log_probs(y_w).sum()
        ref_log_p_l = ref_model.get_log_probs(y_l).sum()

    # ---- 3. 计算概率比的对数 ----
    # log(π_θ / π_ref) = log π_θ - log π_ref
    log_ratio_w = total_log_p_w - ref_log_p_w                     # 偏好回复的对数比
    log_ratio_l = total_log_p_l - ref_log_p_l                     # 不偏好回复的对数比

    # ---- 4. DPO 损失 ----
    # 差值: β * (log_ratio_w - log_ratio_l)
    diff = beta * (log_ratio_w - log_ratio_l)                     # 偏好差距
    loss = -F.logsigmoid(diff)                                    # -log σ(diff)

    return loss


def generate_dpo_preference_pair(
    policy: ToyLanguageModel,
    reward_model: RuleBasedRewardModel,
    prompt: torch.Tensor,
    n_candidates: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    生成一个偏好对 (y_w, y_l) 用于 DPO 训练。

    生成多个候选回复，选得分最高的作为 y_w，最低的作为 y_l。

    参数:
        policy: 当前策略（用于生成候选回复）
        reward_model: 奖励模型（用于评分）
        prompt: prompt token 序列
        n_candidates: 生成的候选回复数量

    返回:
        y_w: 偏好回复的完整序列 (含 prompt)
        y_l: 不偏好回复的完整序列 (含 prompt)
    """
    candidates = []                                                # 候选回复列表
    scores = []                                                    # 对应的分数

    for _ in range(n_candidates):
        gen, _ = policy.generate(prompt, MAX_SEQ_LEN, temperature=1.0)
        gen_tokens = gen[0, len(prompt[0]):].tolist()             # 生成部分 token
        score = reward_model.score(gen_tokens)                    # RM 打分
        candidates.append(gen)
        scores.append(score)

    # 选取最佳和最差回复
    best_idx = np.argmax(scores)                                   # 分数最高的
    worst_idx = np.argmin(scores)                                  # 分数最低的

    return candidates[best_idx], candidates[worst_idx]


# ============================================================================
# 第七部分：训练循环
# ============================================================================

def pretrain_policy(
    policy: ToyLanguageModel,
    data: List[str],
    n_epochs: int = 50,
    lr: float = 1e-3,
    device: str = 'cpu',
) -> Tuple[ToyLanguageModel, List[float]]:
    """
    预训练策略模型（模拟 SFT 阶段）。

    在玩具数据上做简单的语言模型训练（下一个 token 预测），
    让模型学会基本的字符序列建模。

    参数:
        policy: 待预训练的策略模型
        data: 训练数据（字符串列表）
        n_epochs: 训练轮数
        lr: 学习率
        device: 设备

    返回:
        policy: 预训练后的模型
        losses: 每 epoch 的损失记录
    """
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    policy.train()
    losses = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for i, text in enumerate(data):
            # 编码文本: BOS + chars
            tokens = [BOS_TOKEN]
            for ch in text.lower():
                if 'a' <= ch <= 'z':
                    tokens.append(char_to_token(ch))
            if len(tokens) < 2:
                continue

            input_ids = torch.tensor([tokens], device=device)     # (1, len)

            # 语言模型训练: 输入 input_ids[:-1], 目标 input_ids[1:]
            logits, _ = policy(input_ids)                          # (1, len, vocab)
            shift_logits = logits[:, :-1, :].contiguous()          # (1, len-1, vocab)
            shift_labels = input_ids[:, 1:].contiguous()           # (1, len-1)
            loss = F.cross_entropy(
                shift_logits.view(-1, policy.vocab_size),          # (len-1, vocab)
                shift_labels.view(-1)                              # (len-1,)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(1, len(data))
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  [SFT] Epoch {epoch+1}/{n_epochs}, loss={avg_loss:.4f}")

    return policy, losses


def train_ppo(
    policy: ToyLanguageModel,
    ref_model: ToyLanguageModel,
    value_network: ValueNetwork,
    reward_model: RuleBasedRewardModel,
    prompts: List[str],
    n_episodes: int = 200,
    kl_coef: float = 0.1,
    device: str = 'cpu',
    verbose: bool = True,
) -> Dict:
    """
    PPO 训练循环。

    每个 episode:
    1. 采样一个 prompt
    2. 用当前策略生成回复
    3. 用 RM 打分
    4. 用 PPO 更新策略和价值网络

    参数:
        policy: 策略模型
        ref_model: 参考模型
        value_network: 价值网络
        reward_model: 奖励模型
        prompts: prompt 列表
        n_episodes: 训练 episode 数
        kl_coef: KL 惩罚系数
        device: 设备
        verbose: 是否打印进度

    返回:
        history: 训练历史指标
    """
    agent = PPOAgent(
        policy=policy,
        ref_model=ref_model,
        value_network=value_network,
        lr=1e-4,
        clip_epsilon=0.2,
        kl_coef=kl_coef,
        gamma=0.99,
        gae_lambda=0.95,
        device=device,
    )

    history = {
        'rm_scores': [],                                          # RM 分数
        'kl_divergence': [],                                      # KL 散度
        'entropy': [],                                            # 策略熵
        'policy_loss': [],                                        # PPO 策略损失
    }

    if verbose:
        print("\n  [PPO] 开始训练...")

    start_time = time.time()

    for ep in range(n_episodes):
        # ---- 选择 prompt ----
        prompt_text = prompts[ep % len(prompts)]
        prompt = encode_prompt(prompt_text).to(device)

        # ---- 生成轨迹 ----
        traj = agent.generate_and_score(prompt, reward_model)

        if len(traj['log_probs']) == 0:
            continue                                               # 跳过空生成

        # ---- 计算 GAE 优势 ----
        n_steps = len(traj['rewards'])
        rewards_t = torch.tensor(traj['rewards'], device=device)
        values_t = (traj['values'].flatten()[:n_steps]
                   if len(traj['values']) > 0
                   else torch.zeros(n_steps, device=device))
        dones = [False] * (n_steps - 1) + [True]                 # 最后一步为终止

        next_value = torch.tensor(0.0, device=device)              # 终止后价值为 0
        advantages = agent.compute_gae(
            traj['rewards'],
            values_t.tolist(),
            next_value.item() if isinstance(next_value, torch.Tensor) else next_value,
            dones,
        )

        # 累计回报 = advantages + values (因为 GAE 估计的是 A, 不是 Q)
        returns = advantages + values_t[:n_steps].detach()

        # ---- 标准化优势 ----
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ---- PPO 更新 ----
        metrics = agent.ppo_update(
            old_log_probs=traj['log_probs'][:n_steps],
            advantages=advantages,
            returns=returns,
            states=traj['states_embedded'][:n_steps],
            actions=traj['generated'][:, len(prompt[0]):],
            values=values_t[:n_steps],
            ref_log_probs=traj['ref_log_probs_gen'][:n_steps],
        )

        # ---- 记录 ----
        history['rm_scores'].append(traj['rm_score'])
        history['kl_divergence'].append(metrics['kl_divergence'])
        history['entropy'].append(metrics['entropy'])
        history['policy_loss'].append(metrics['policy_loss'])

        if verbose and (ep + 1) % 40 == 0:
            avg_score = np.mean(history['rm_scores'][-40:])
            print(f"  Episode {ep+1:4d}/{n_episodes}: "
                  f"RM={traj['rm_score']:.2f}, "
                  f"avg_RM={avg_score:.2f}, "
                  f"KL={metrics['kl_divergence']:.4f}, "
                  f"entropy={metrics['entropy']:.4f}")

    training_time = time.time() - start_time

    if verbose:
        avg_score = np.mean(history['rm_scores'][-50:]) if history['rm_scores'] else 0
        print(f"\n  [PPO] 训练完成! 耗时: {training_time:.1f}秒")
        print(f"  [PPO] 最后 50 episode 平均 RM 分数: {avg_score:.2f}")

    history['training_time'] = training_time
    return history


def train_dpo(
    policy: ToyLanguageModel,
    ref_model: ToyLanguageModel,
    reward_model: RuleBasedRewardModel,
    prompts: List[str],
    n_steps: int = 200,
    beta: float = 0.1,
    lr: float = 1e-4,
    device: str = 'cpu',
    verbose: bool = True,
) -> Dict:
    """
    DPO 训练循环。

    每步:
    1. 采样一个 prompt
    2. 用当前策略生成多个候选回复
    3. 用 RM 选出偏好对 (y_w, y_l)
    4. 计算 DPO 损失并更新策略

    参数:
        policy: 策略模型
        ref_model: 参考模型
        reward_model: 奖励模型
        prompts: prompt 列表
        n_steps: 训练步数
        beta: DPO β 参数
        lr: 学习率
        device: 设备
        verbose: 是否打印进度

    返回:
        history: 训练历史指标
    """
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    policy.train()
    ref_model.eval()

    history = {
        'loss': [],                                               # DPO 损失
        'rm_scores_win': [],                                      # y_w 的 RM 分数
        'rm_scores_lose': [],                                     # y_l 的 RM 分数
        'log_ratio_w': [],                                        # y_w 的对数比率
        'log_ratio_l': [],                                        # y_l 的对数比率
    }

    if verbose:
        print("\n  [DPO] 开始训练...")

    start_time = time.time()

    for step in range(n_steps):
        # ---- 选择 prompt ----
        prompt_text = prompts[step % len(prompts)]
        prompt = encode_prompt(prompt_text).to(device)

        # ---- 生成偏好对 ----
        y_w, y_l = generate_dpo_preference_pair(
            policy, reward_model, prompt, n_candidates=4)

        # ---- 计算 DPO 损失 ----
        loss = compute_dpo_loss(policy, ref_model, prompt, y_w, y_l, beta=beta)

        # ---- 反向传播 ----
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()

        # ---- 记录 ----
        w_tokens = y_w[0, len(prompt[0]):].tolist()
        l_tokens = y_l[0, len(prompt[0]):].tolist()
        rm_w = reward_model.score(w_tokens)
        rm_l = reward_model.score(l_tokens)

        history['loss'].append(loss.item())
        history['rm_scores_win'].append(rm_w)
        history['rm_scores_lose'].append(rm_l)

        if verbose and (step + 1) % 40 == 0:
            avg_w = np.mean(history['rm_scores_win'][-40:])
            avg_l = np.mean(history['rm_scores_lose'][-40:])
            margin = avg_w - avg_l
            print(f"  Step {step+1:4d}/{n_steps}: "
                  f"loss={loss.item():.4f}, "
                  f"RM_w={rm_w:.2f}, RM_l={rm_l:.2f}, "
                  f"margin={margin:.2f}")

    training_time = time.time() - start_time

    if verbose:
        print(f"\n  [DPO] 训练完成! 耗时: {training_time:.1f}秒")

    history['training_time'] = training_time
    return history


# ============================================================================
# 第八部分：可视化
# ============================================================================

def plot_ppo_training(history: Dict, title: str = "PPO 训练曲线"):
    """绘制 PPO 训练过程中的关键指标。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ---- RM 分数 ----
    ax = axes[0, 0]
    scores = np.array(history['rm_scores'])
    ax.plot(scores, 'b-', linewidth=1, alpha=0.5, label='RM 分数')
    if len(scores) >= 20:
        smooth = np.convolve(scores, np.ones(20)/20, mode='valid')
        ax.plot(np.arange(19, len(scores)), smooth, 'r-', linewidth=2,
               label='滑动平均 (窗口=20)')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Episode', fontsize=10)
    ax.set_ylabel('RM 分数', fontsize=10)
    ax.set_title('奖励模型分数', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- KL 散度 ----
    ax = axes[0, 1]
    kl_vals = np.array(history['kl_divergence'])
    ax.plot(kl_vals, 'g-', linewidth=1.5, label='KL(π_θ || π_ref)')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Episode', fontsize=10)
    ax.set_ylabel('KL 散度', fontsize=10)
    ax.set_title('KL 散度 (策略偏离程度)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- 策略熵 ----
    ax = axes[1, 0]
    ent_vals = np.array(history['entropy'])
    ax.plot(ent_vals, 'orange', linewidth=1.5, label='策略熵 H(π)')
    ax.set_xlabel('Episode', fontsize=10)
    ax.set_ylabel('熵 (nats)', fontsize=10)
    ax.set_title('策略熵 (探索程度)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- PPO 损失 ----
    ax = axes[1, 1]
    loss_vals = np.array(history['policy_loss'])
    ax.plot(loss_vals, 'purple', linewidth=1, alpha=0.6)
    if len(loss_vals) >= 20:
        smooth = np.convolve(loss_vals, np.ones(20)/20, mode='valid')
        ax.plot(np.arange(19, len(loss_vals)), smooth, 'r-', linewidth=2,
               label='滑动平均')
    ax.set_xlabel('Episode', fontsize=10)
    ax.set_ylabel('PPO 损失', fontsize=10)
    ax.set_title('PPO 策略损失 (含 KL 惩罚)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('images/ppo_training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[可视化] PPO 训练曲线已保存至 images/ppo_training_curves.png")


def plot_dpo_training(history: Dict, title: str = "DPO 训练曲线"):
    """绘制 DPO 训练过程中的关键指标。"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ---- DPO 损失 ----
    ax = axes[0]
    losses = np.array(history['loss'])
    ax.plot(losses, 'b-', linewidth=0.8, alpha=0.5)
    if len(losses) >= 20:
        smooth = np.convolve(losses, np.ones(20)/20, mode='valid')
        ax.plot(np.arange(19, len(losses)), smooth, 'r-', linewidth=2,
               label='滑动平均')
    ax.set_xlabel('训练步数', fontsize=10)
    ax.set_ylabel('DPO 损失', fontsize=10)
    ax.set_title('DPO 训练损失 L_DPO', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- RM 分数对比 ----
    ax = axes[1]
    w_scores = np.array(history['rm_scores_win'])
    l_scores = np.array(history['rm_scores_lose'])
    ax.plot(w_scores, 'g-', linewidth=1, alpha=0.5, label='y_w (偏好回复)')
    ax.plot(l_scores, 'r-', linewidth=1, alpha=0.5, label='y_l (不偏好回复)')
    if len(w_scores) >= 20:
        w_smooth = np.convolve(w_scores, np.ones(20)/20, mode='valid')
        l_smooth = np.convolve(l_scores, np.ones(20)/20, mode='valid')
        ax.plot(np.arange(19, len(w_scores)), w_smooth, 'g-', linewidth=2)
        ax.plot(np.arange(19, len(l_scores)), l_smooth, 'r-', linewidth=2)
    ax.set_xlabel('训练步数', fontsize=10)
    ax.set_ylabel('RM 分数', fontsize=10)
    ax.set_title('偏好 vs 非偏好回复的 RM 分数', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- 偏好差距 (margin) ----
    ax = axes[2]
    margin = w_scores - l_scores
    ax.plot(margin, 'purple', linewidth=1.5, label='y_w - y_l 分数差')
    if len(margin) >= 20:
        m_smooth = np.convolve(margin, np.ones(20)/20, mode='valid')
        ax.plot(np.arange(19, len(margin)), m_smooth, 'r-', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('训练步数', fontsize=10)
    ax.set_ylabel('分数差', fontsize=10)
    ax.set_title('偏好差距 (margin)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('images/dpo_training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[可视化] DPO 训练曲线已保存至 images/dpo_training_curves.png")


def plot_comparison(
    ppo_history: Dict,
    dpo_history: Dict,
    title: str = "PPO vs DPO — 训练对比",
):
    """对比 PPO 和 DPO 的关键指标。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ---- 子图 1: 训练时间 ----
    ax = axes[0, 0]
    methods = ['PPO', 'DPO']
    times = [ppo_history['training_time'], dpo_history['training_time']]
    bars = ax.bar(methods, times, color=['#2E86AB', '#F18F01'])
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{t:.1f}s', ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel('训练时间 (秒)', fontsize=10)
    ax.set_title('训练时间对比', fontsize=12, fontweight='bold')

    # ---- 子图 2: 最终 RM 分数 ----
    ax = axes[0, 1]
    if ppo_history['rm_scores']:
        ppo_final = np.mean(ppo_history['rm_scores'][-20:])
    else:
        ppo_final = 0
    if dpo_history['rm_scores_win']:
        dpo_final = np.mean(dpo_history['rm_scores_win'][-20:])
    else:
        dpo_final = 0
    bars = ax.bar(methods, [ppo_final, dpo_final], color=['#2E86AB', '#F18F01'])
    for bar, val in zip(bars, [ppo_final, dpo_final]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
               f'{val:.2f}', ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel('最终 RM 分数 (近 20 步平均)', fontsize=10)
    ax.set_title('最终性能对比', fontsize=12, fontweight='bold')

    # ---- 子图 3: PPO RM 分数曲线 ----
    ax = axes[1, 0]
    scores = np.array(ppo_history['rm_scores'])
    ax.plot(scores, 'b-', linewidth=1, alpha=0.5)
    if len(scores) >= 20:
        smooth = np.convolve(scores, np.ones(20)/20, mode='valid')
        ax.plot(np.arange(19, len(scores)), smooth, 'b-', linewidth=2,
               label='PPO')
    ax.set_xlabel('Episode', fontsize=10)
    ax.set_ylabel('RM 分数', fontsize=10)
    ax.set_title('PPO — RM 分数变化', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- 子图 4: DPO margin 曲线 ----
    ax = axes[1, 1]
    w = np.array(dpo_history['rm_scores_win'])
    l = np.array(dpo_history['rm_scores_lose'])
    margin = w - l
    ax.plot(margin, 'orange', linewidth=1, alpha=0.5)
    if len(margin) >= 20:
        m_smooth = np.convolve(margin, np.ones(20)/20, mode='valid')
        ax.plot(np.arange(19, len(margin)), m_smooth, 'orange',
               linewidth=2, label='DPO')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('训练步数', fontsize=10)
    ax.set_ylabel('偏好差距', fontsize=10)
    ax.set_title('DPO — 偏好差距变化', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('images/ppo_vs_dpo_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[可视化] PPO vs DPO 对比图已保存至 images/ppo_vs_dpo_comparison.png")


def plot_sample_outputs(
    policy: ToyLanguageModel,
    prompts: List[str],
    title: str = "对齐前后的样本输出对比",
    n_samples: int = 3,
):
    """
    展示训练后模型的样本生成输出。

    参数:
        policy: 训练后的策略模型
        prompts: 测试 prompt 列表
        title: 图表标题
        n_samples: 展示的样本数量
    """
    policy.eval()
    fig, axes = plt.subplots(1, n_samples, figsize=(5 * n_samples, 4))

    for i in range(n_samples):
        prompt_text = prompts[i % len(prompts)]
        prompt = encode_prompt(prompt_text)
        gen, _ = policy.generate(prompt, MAX_SEQ_LEN, temperature=0.8)
        gen_tokens = gen[0, len(prompt[0]):].tolist()
        output_text = decode_tokens(gen_tokens)

        ax = axes[i] if n_samples > 1 else axes
        ax.text(0.5, 0.5,
               f"Prompt: {prompt_text}\n\n"
               f"Output: {output_text}\n\n"
               f"({len(gen_tokens)} tokens)",
               ha='center', va='center', fontsize=10,
               fontfamily='monospace',
               transform=ax.transAxes)
        ax.set_title(f'样本 {i+1}', fontsize=11, fontweight='bold')
        ax.axis('off')

    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('images/sample_outputs.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[可视化] 样本输出已保存至 images/sample_outputs.png")


# ============================================================================
# 第九部分：主程序
# ============================================================================

def main():
    """
    主程序：
    1. 创建玩具语言模型和训练数据
    2. 预训练（模拟 SFT）
    3. 用 PPO 训练（含 KL 惩罚的裁剪目标）
    4. 用 DPO 训练
    5. 对比和可视化
    """
    print("\n" + "=" * 70)
    print("    s21 RLHF：当强化学习遇见大模型 — Toy 演示")
    print("=" * 70)
    print("\n  ⚠️ 注意: 这是玩具/教学版本，完整 RLHF 训练需要百 GPU 天。")
    print("    本 demo 在小规模合成数据上展示 PPO 和 DPO 的核心机制。\n")

    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  设备: {device}")

    # ---- 准备数据 ----
    train_data = create_toy_dataset(num_prompts=200)
    print(f"\n[数据] 创建了 {len(train_data)} 个训练 prompt")

    # ---- 创建模型 ----
    print("\n[模型] 创建玩具语言模型...")
    policy = ToyLanguageModel(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
    ).to(device)

    ref_model = ToyLanguageModel(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
    ).to(device)

    # ========================================================================
    # 阶段 1: 预训练 (模拟 SFT)
    # ========================================================================
    print("\n" + "=" * 50)
    print("【阶段 1】预训练 (模拟 SFT)")
    print("=" * 50)
    policy, sft_losses = pretrain_policy(policy, train_data, n_epochs=100, device=device)

    # 复制预训练权重到参考模型（冻结参考模型）
    ref_model.load_state_dict(
        {k: v.clone() for k, v in policy.state_dict().items()})
    ref_model.eval()                                             # 参考模型评估模式
    for p in ref_model.parameters():
        p.requires_grad = False                                   # 冻结

    print("  ✓ 预训练完成，参考模型已冻结")

    # ---- 生成预训练后的样本 ----
    print("\n  [预训练后的样本生成]:")
    policy.eval()
    for i in range(3):
        prompt_text = train_data[i]
        prompt = encode_prompt(prompt_text).to(device)
        gen, _ = policy.generate(prompt, MAX_SEQ_LEN, temperature=0.8)
        gen_tokens = gen[0, len(prompt[0]):].tolist()
        print(f"    Prompt '{prompt_text}': '{decode_tokens(gen_tokens)}'")

    # ========================================================================
    # 阶段 2: PPO 训练
    # ========================================================================
    print("\n" + "=" * 50)
    print("【阶段 2】PPO 训练 (含 KL 惩罚)")
    print("=" * 50)

    # 重置策略到 SFT 状态（创建一个新的可训练副本）
    policy_ppo = ToyLanguageModel(
        vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM
    ).to(device)
    policy_ppo.load_state_dict(
        {k: v.clone() for k, v in ref_model.state_dict().items()})

    rm = RuleBasedRewardModel()
    value_net = ValueNetwork(embed_dim=EMBED_DIM, hidden_dim=64).to(device)

    ppo_history = train_ppo(
        policy=policy_ppo,
        ref_model=ref_model,
        value_network=value_net,
        reward_model=rm,
        prompts=train_data[:50],                                 # 使用部分 prompt
        n_episodes=200,
        kl_coef=0.1,
        device=device,
        verbose=True,
    )

    # ---- PPO 训练后的样本 ----
    print("\n  [PPO 训练后的样本生成]:")
    policy_ppo.eval()
    for i in range(3):
        prompt_text = train_data[i]
        prompt = encode_prompt(prompt_text).to(device)
        gen, _ = policy_ppo.generate(prompt, MAX_SEQ_LEN, temperature=0.8)
        gen_tokens = gen[0, len(prompt[0]):].tolist()
        rm_score = rm.score(gen_tokens)
        print(f"    Prompt '{prompt_text}': '{decode_tokens(gen_tokens)}' "
              f"(RM={rm_score:.2f})")

    # ========================================================================
    # 阶段 3: DPO 训练
    # ========================================================================
    print("\n" + "=" * 50)
    print("【阶段 3】DPO 训练")
    print("=" * 50)

    # 重置策略到 SFT 状态
    policy_dpo = ToyLanguageModel(
        vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM
    ).to(device)
    policy_dpo.load_state_dict(
        {k: v.clone() for k, v in ref_model.state_dict().items()})

    dpo_history = train_dpo(
        policy=policy_dpo,
        ref_model=ref_model,
        reward_model=rm,
        prompts=train_data[:50],
        n_steps=200,
        beta=0.1,
        lr=1e-4,
        device=device,
        verbose=True,
    )

    # ---- DPO 训练后的样本 ----
    print("\n  [DPO 训练后的样本生成]:")
    policy_dpo.eval()
    for i in range(3):
        prompt_text = train_data[i]
        prompt = encode_prompt(prompt_text).to(device)
        gen, _ = policy_dpo.generate(prompt, MAX_SEQ_LEN, temperature=0.8)
        gen_tokens = gen[0, len(prompt[0]):].tolist()
        rm_score = rm.score(gen_tokens)
        print(f"    Prompt '{prompt_text}': '{decode_tokens(gen_tokens)}' "
              f"(RM={rm_score:.2f})")

    # ========================================================================
    # 阶段 4: 可视化与对比
    # ========================================================================
    print("\n" + "=" * 50)
    print("【阶段 4】可视化与对比")
    print("=" * 50)

    plot_ppo_training(ppo_history)
    plot_dpo_training(dpo_history)
    plot_comparison(ppo_history, dpo_history)
    plot_sample_outputs(policy_ppo, train_data, "PPO 对齐后的样本输出")

    # ========================================================================
    # 最终总结
    # ========================================================================
    print("\n" + "=" * 70)
    print("【总结】")
    print("=" * 70)

    ppo_final_rm = np.mean(ppo_history['rm_scores'][-20:]) \
        if ppo_history['rm_scores'] else 0
    dpo_final_rm = np.mean(dpo_history['rm_scores_win'][-20:]) \
        if dpo_history['rm_scores_win'] else 0

    print(f"\n  训练时间: PPO={ppo_history['training_time']:.1f}s, "
          f"DPO={dpo_history['training_time']:.1f}s")
    print(f"  最终 RM 分数: PPO={ppo_final_rm:.2f}, DPO={dpo_final_rm:.2f}")

    if ppo_history['kl_divergence']:
        print(f"  最终 KL(π_θ||π_ref): "
              f"{np.mean(ppo_history['kl_divergence'][-20:]):.4f}")

    print(f"\n  【PPO 核心机制】")
    print(f"  - 裁剪目标: min(r_t·Â_t, clip(r, 1-ε, 1+ε)·Â_t)")
    print(f"  - KL 惩罚: 防止策略偏离参考模型太远")
    print(f"  - GAE: 平衡偏差和方差的优势估计")
    print(f"  - 在线采样: 每步都用当前策略生成数据")

    print(f"\n  【DPO 核心机制】")
    print(f"  - 直接优化偏好: L_DPO = -log σ(β·(log π_w/π_ref_w - log π_l/π_ref_l))")
    print(f"  - 无需奖励模型: 偏好信号直接编码在损失中")
    print(f"  - 更稳定: 类似分类任务，无在线交互")
    print(f"  - Bradley-Terry 偏好模型: 理论基础保证收敛到最优策略")

    print(f"\n  【本 Demo 的局限性】")
    print(f"  - 使用基于规则的 RM，非真实神经网络 RM")
    print(f"  - 模型极小 (128 维 LSTM, 30 token 词汇表)")
    print(f"  - 合成数据，非真实指令数据集")
    print(f"  - 在规模化 RLHF 中: 模型数十亿参数, RM 经数十万标注训练")
    print(f"  - 在规模化 RLHF 中: PPO 需分布式训练 (DeepSpeed, Megatron)")
    print(f"  - 但核心数学（裁剪、KL、DPO 损失）完全相同！")

    print(f"\n  所有图片已保存至 images/ 目录")
    print("=" * 70)
    print("\n  运行完成！\n")


if __name__ == "__main__":
    main()
