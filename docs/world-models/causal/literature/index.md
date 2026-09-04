---
title: "因果世界模型文献"
order: 20
---
# 因果世界模型：文献里正在汇合的几条线

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

> 上一章把 $P$ 与 $P(\cdot\mid do)$ 讲清楚了。路径四目前还不是「一个统一算法家族」，而是一组正在汇合的问题。本章按线索读文献，不装成已经有标准教材。

---

## 一、评测：先问模型怕不怕干预

世界模型 survey（Lyu / Dong 等 *Learning to Model the World*）显式提出因果推理指标：比较 $\pi_\theta(y\mid x)$ 与 $\pi_\theta(y\mid do(x=i))$ 的分歧——模型若对干预不敏感，就只是关联拟合器。

Agentic World Modeling survey（Chu et al., 2026）把「干预敏感性」写成从 L1 Predictor 走向 L2 Simulator 的必要条件：能多步 rollout 还不够，还必须在 $do(a)$ 下遵守领域定律。

读任何视频 WM 论文时，先找有没有 **action-conditioned 闭环评测**，还是只有 FVD / 主观观感。

---

## 二、LLM + 显式因果模块

Gkountouras et al., *Language Agents Meet Causality – Bridging LLMs and Causal World Models*（ICLR 2025）：语言智能体的符号推理与**显式因果世界模型**对接，用因果模块约束「做什么会怎样」。这和路径五 [WALL-E](/world-models/symbolic/alignment/) 的「规则对齐 LLM 世界模型」是近亲：都承认纯语言模拟会幻觉，都要把 $do$ 语义外置。

---

## 三、下一 token 里有没有因果结构？

Rohekar et al., *A Causal World Model Underlying Next Token Prediction*（2024）：在受控环境中分析 GPT 类模型是否学到可用的因果世界结构，而不只是表面共现。结论取向是：**有时有、不稳定、依赖数据生成过程**——不要把「会续写」直接等同于「会干预」。

Spies et al., *Transformers use causal world models in maze-solving tasks*（ICLR Workshop 2025）：迷宫任务成功依赖内部是否形成**可干预的状态机**。这是路径三（隐式 $z$）与路径四的交界实验：表示里长出了可以 $do$ 的开关。

---

## 四、训练范式：Next Forcing 一类

Awesome-World-Model 列表中的 *Next Forcing: Causal World Modeling with Multi-Chunk Prediction*：把「下一块」预测做成更强调因果滚动的训练范式，与普通下一帧 / 下一 token 强制区分。教学上把它理解成：**强制模型在块与块之间遵守机制，而不是只拟合局部纹理相关**。

---

## 五、和路径五怎么分工

| | 路径四 | 路径五 |
|--|--------|--------|
| 语言 | $do$、SCM、混淆、反事实 | 谓词、PDDL、规则、程序、知识图谱 |
| 典型失败 | 旁观视频当控制器 | LLM 世界模型幻觉、规则覆盖不全 |
| 互补 | 告诉你该采集什么数据 | 告诉你定律能不能写成可执行符号 |

> 读完路径四，可去 [路径五导论](/world-models/symbolic/overview/)，或回到 [导论](/world-models/intro/) 选应用。玩具 SCM 仍在上一章 [阶梯](/world-models/causal/ladder/) 的 `demo.py`。

## 参考

1. Gkountouras, J., et al. (2025). Language Agents Meet Causality – Bridging LLMs and Causal World Models. *ICLR*.
2. Rohekar, R. Y., et al. (2024). A Causal World Model Underlying Next Token Prediction. *arXiv*.
3. Spies, A. F., et al. (2025). Transformers use causal world models in maze-solving tasks. *ICLR Workshop*.
4. Lyu, Q., Dong, J., et al. Learning to Model the World: A Survey of World Models in AI.
5. Chu, M., et al. (2026). Agentic World Modeling. [[arXiv:2604.22748](https://arxiv.org/abs/2604.22748)]
