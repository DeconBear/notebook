---
title: "s21 RLHF"
order: 30
legacyPaths:
  - /rl/rlhf/
  - /s21_rlhf/
---
# s21 RLHF：当强化学习遇见大模型

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

> 前置已经齐了：[PPO](/nn-decision/rl/ppo/) 的裁剪目标与 GAE，[GRPO](/nn-decision/rl/grpo/) 的组相对优势，[AlphaGo](/nn-decision/rl/alphago/) 说明策略梯度可以赢过人类。这一章不再推优化器，只做一件事：**把这些零件接到 token 序列上**——SFT、奖励模型、KL 橡皮筋、以及不必走在线 RL 的 DPO。

---

## 一、为什么大模型需要强化学习？

预训练只优化「下一个 token 的交叉熵」。它学的是互联网文本的统计分布：会续写，但不保证遵循指令、承认不知道、或拒绝有害请求。

**对齐问题（Alignment）**：如何让 LLM 的行为与人类意图和价值观一致？三个常用维度是 HHH：

1. **有用（Helpful）**：跟着指令把任务做完
2. **诚实（Honest）**：不编造、不懂装懂
3. **无害（Harmless）**：不输出危险或歧视内容

交叉熵衡量不了「整段回复好不好」。强化学习擅长的恰好是**延迟的整段奖励**——把一次生成看成一条轨迹，结束时打一个标量分。

> 目标从「预测对下一个 token」变成「生成一段人类觉得好的回复」。

![RLHF 完整三阶段流程](./images/21-01-rlhf-pipeline.png)

> **图解说明**：SFT 学会听话，奖励模型学会打分，第三段才把 [PPO](/nn-decision/rl/ppo/)（或 [GRPO](/nn-decision/rl/grpo/)）套上去。

---

## 二、RLHF 三阶段流程

RLHF（Reinforcement Learning from Human Feedback）由 Christiano et al. (2017) 提出，InstructGPT / ChatGPT（Ouyang et al., 2022）把它做成对齐的默认流水线。

### 阶段 1：监督微调（SFT）

- **数据**：人类按 prompt 写高质量回复
- **目标**：$\max \log \pi(y\mid x)$，标准监督学习
- **产出**：$\pi_{\mathrm{SFT}}$，初步会听指令

没有 SFT 就直接做 RL，模型连「什么叫遵循指令」都不知道，奖励几乎没有意义。

### 阶段 2：奖励模型（RM）

同一个 prompt 让 SFT 生成 $K$ 条回复，人类排序。用 Bradley-Terry 训练 $R_\phi(x,y)$：

$$
\mathcal{L}_R(\phi) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \big( R_\phi(x, y_w) - R_\phi(x, y_l) \big) \right]
$$

$y_w$ 比 $y_l$ 更受偏好。RM 是一个**偏好分类器**，输出标量分数，不是生成模型。

### 阶段 3：用已经学过的优化器微调策略

- 初始策略：$\pi_{\mathrm{SFT}}$
- 奖励：RM 的分数，再减一项到参考策略的 KL（下一节）
- **优化器原样拿来用**：
  - InstructGPT 路线 → [PPO](/nn-decision/rl/ppo/)（Actor + Critic + GAE + clip）
  - DeepSeek-R1 一类可验证推理 → [GRPO](/nn-decision/rl/grpo/)（组内 z-score，不训 $V$）

$L^{\mathrm{CLIP}}$、$r_t(\theta)$、GAE 的公式都在 PPO 章；组相对优势在 GRPO 章。下面只写 **LLM 记号下多出来的那几行**。

---

## 三、把生成写成 MDP

| RL 符号 | 在 LLM 里是什么 |
|---------|----------------|
| 状态 $s_t$ | prompt + 已写 token $(x, y_{<t})$ |
| 动作 $a_t$ | 下一个 token $y_t$，词表约 $10^4\sim 10^5$ |
| 策略 $\pi_\theta(a_t\mid s_t)$ | 模型本身的 next-token 分布 |
| 奖励 | 中间 token 为 0；序列结束时才有标量 $R$ |
| 轨迹 $\tau$ | 一次完整生成 $(x,y_1,\ldots,y_T)$ |

> 一次自回归生成 = 一条 RL 轨迹。每个 token = 一个动作。

奖励稀疏，所以 InstructGPT 需要 Critic 估「写到这里还值多少分」——这就是 PPO 章的 $V_\phi$ 和 GAE。若同一题采一组完整答案、用验证器打分，就可以换成 GRPO，不再训 $V$。

---

## 四、PPO 接到 LLM 上：新东西只有奖励塑形

概率比和裁剪**不要重新推**，直接抄 [PPO](/nn-decision/rl/ppo/)：

$$
r_t(\theta)=\frac{\pi_\theta(y_t\mid x,y_{<t})}{\pi_{\theta_{\mathrm{old}}}(y_t\mid x,y_{<t})}
$$

$$
L^{\mathrm{CLIP}}(\theta)=\mathbb{E}_t\Big[\min\big(r_t\hat{A}_t,\;\mathrm{clip}(r_t,1-\varepsilon,1+\varepsilon)\hat{A}_t\big)\Big]
$$

$\hat{A}_t$ 用该章的 GAE，Critic 看的是 token 前缀。

LLM 对齐真正多出来的是 **KL 橡皮筋**。把参考策略 $\pi_{\mathrm{ref}}$（通常冻结的 SFT）写进奖励：

$$
R_{\mathrm{total}} = R_\phi(x,y) - \beta\, D_{\mathrm{KL}}\big(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\mathrm{ref}}(\cdot\mid x)\big)
$$

没有这根筋，策略会 **奖励黑客**：专攻 RM 的癖好（特别长、套话、假装权威），把预训练学到的语言能力抽走。$\beta$ 越大，越不敢离开 SFT。

GRPO 论文把 $\beta\,\mathrm{KL}$ 写进损失而不是改写 $R$；作用一样，见 [GRPO](/nn-decision/rl/grpo/) 第三节。

实践里这一阶段要同时盯四个网络：Actor $\pi_\theta$、Critic $V$、冻结的 $\pi_{\mathrm{ref}}$、冻结的 $R_\phi$。这就是「PPO 训 LLM 又贵又脆」的来源——不是裁剪公式变了，是模型个数变了。

![PPO 裁剪替代目标——公式本身在上一章](./images/21-02-ppo-clipped-objective.png)

> **图解说明**：这张图是 [PPO 章](/nn-decision/rl/ppo/) 裁剪目标的回顾。$A>0$ 时 $r$ 不能过 $1+\varepsilon$；$A<0$ 时不能过 $1-\varepsilon$。

---

## 五、可验证任务可以换成 GRPO

数学、代码、带单测的推理：最终对错可以由规则说了算，不一定每一步都要人类。DeepSeekMath / DeepSeek-R1 用 [GRPO](/nn-decision/rl/grpo/)：

- 同一 prompt 采 $G$ 条完整输出；
- $\hat{A}_i=(r_i-\mathrm{mean})/\mathrm{std}$，**没有 Critic**；
- 目标仍是 PPO-Clip，外加到 $\pi_{\mathrm{ref}}$ 的 KL。

和「经典 RLHF」的分工：

| | InstructGPT 式 RLHF | R1 式强化学习 |
|--|---------------------|---------------|
| 奖励从哪来 | 人类偏好 → RM | 验证器 / 规则（可加少量偏好） |
| 优化器 | PPO + GAE + $V$ | GRPO 组相对 |
| 还要不要 SFT | 要 | 要（冷启动） |

HHH、偏好数据、奖励黑客，仍然是本章的主题；**换优化器不会自动解决对齐**。

---

## 六、DPO：离线偏好，绕过显式 RM 和在线 RL

PPO+RM 的痛点：多一个可能被黑客的 RM、四模型同步、on-policy 采样贵。DPO（Rafailov et al., 2023）从 Bradley-Terry 反解出：最优策略对应的奖励可以写成 $\pi$ 相对 $\pi_{\mathrm{ref}}$ 的 log 比。代回偏好损失后，$Z(x)$ 消掉，得到只含策略的损失：

$$
\mathcal{L}_{\mathrm{DPO}}(\theta)
= -\mathbb{E}_{(x,y_w,y_l)}\left[
  \log\sigma\left(
    \beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
    -\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
  \right)
\right]
$$

直观：相对参考模型，抬高 $y_w$、压低 $y_l$。$\beta$ 仍然是「能离开 SFT 多远」。

| 维度 | RLHF（PPO / GRPO） | DPO |
|------|-------------------|-----|
| 模型 | PPO 要 Actor/Critic/Ref/RM；GRPO 省 Critic | Policy + Ref |
| 数据 | on-policy 采样（GRPO 还必须成组） | 离线偏好对 |
| 奖励 | 显式 RM 或验证器 | 隐含在 $\pi/\pi_{\mathrm{ref}}$ 里 |
| 稳定性 | 要调 clip、$\beta$、GAE 或组大小 | 更像分类损失 |
| 灵活性 | 能接在线反馈、可验证奖励 | 偏好分布一变就要重采数据 |

高质量闭源助手仍常用 PPO 一类在线 RL；开源微调大量用 DPO / ORPO，因为省。DPO **不是** GRPO 的替代：一个吃离线对，一个吃 on-policy 组采样。

![DPO vs RLHF 对比图](./images/21-03-dpo-vs-rlhf.png)

---

## 七、挑战与前沿

1. **奖励黑客**：策略欺骗 RM。KL 只能减缓，不能从根上消掉。
2. **分布偏移**：on-policy 生成渐渐离开 RM 训练时见过的回复，RM 失效。
3. **偏好不一致**：标注者对「好」的定义不同，直接进 $R_\phi$。
4. **对齐税**：过度讨好人类，基准能力掉一点。

前沿（点到为止）：Constitutional AI（AI 按「宪法」自批评）、Iterated RLHF、多目标奖励、ORPO / RRHF 等更短的偏好损失。推理向则把验证器 + GRPO 做成主路径。

![奖励黑客——当模型学会欺骗奖励模型](./images/21-04-reward-hacking.png)

---

## 八、本节小结

| 概念 | 一句话 | 公式在哪 |
|------|--------|----------|
| 对齐 / HHH | 有用、诚实、无害 | 本章 |
| SFT | 示范数据上先学会听指令 | 本章 |
| 奖励模型 | Bradley-Terry 拟合人类排序 | 本章 |
| token-MDP | 前缀是状态，token 是动作 | 本章 |
| $L^{\mathrm{CLIP}}$、GAE | 接到 LLM 上的优化器 | [PPO](/nn-decision/rl/ppo/) |
| 组相对优势 | 可验证任务上可以不训 $V$ | [GRPO](/nn-decision/rl/grpo/) |
| KL 橡皮筋 | $R_{\mathrm{total}}=R_\phi-\beta\,\mathrm{KL}$ | 本章（接到 PPO/GRPO 上） |
| DPO | 离线偏好，不显式训 RM | 本章 |

> RLHF 不是一种新算法，是「人类反馈 → 标量奖励 → 已经学过的策略优化器」。棋上的自我对弈见 [AlphaGo](/nn-decision/rl/alphago/)；更新别迈太大见 PPO；组内相对见 GRPO。

## 📥 Code

| File | View | Download |
|------|------|----------|
| demo.py | [Open](./code-demo) | <a href="/notebook/code/nn-decision/rl/rlhf/demo.py" target="_blank" download>Download</a> |
| exercise.py | [Open](./code-exercise) | <a href="/notebook/code/nn-decision/rl/rlhf/exercise.py" target="_blank" download>Download</a> |

## 参考

1. Christiano, P., et al. (2017). Deep Reinforcement Learning from Human Preferences. *NeurIPS*. [[arXiv:1706.03741](https://arxiv.org/abs/1706.03741)]
2. Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback. *NeurIPS*. (InstructGPT) [[arXiv:2203.02155](https://arxiv.org/abs/2203.02155)]
3. Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. [[arXiv:1707.06347](https://arxiv.org/abs/1707.06347)] — 推导见 [PPO](/nn-decision/rl/ppo/)
4. Shao, Z., et al. (2024). DeepSeekMath. [[arXiv:2402.03300](https://arxiv.org/abs/2402.03300)] — GRPO，见 [GRPO](/nn-decision/rl/grpo/)
5. DeepSeek-AI (2025). DeepSeek-R1. [[arXiv:2501.12948](https://arxiv.org/abs/2501.12948)]
6. Rafailov, R., et al. (2023). Direct Preference Optimization. *NeurIPS*. [[arXiv:2305.18290](https://arxiv.org/abs/2305.18290)]
