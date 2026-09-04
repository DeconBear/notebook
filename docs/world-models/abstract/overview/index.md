---
title: "路径三导论：在紧凑状态上预测与决策"
order: 5
---
# 路径三导论：把世界压进 $z$，再在里面做梦

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

> 路径一/二把观测生成得越来越像真的。路径三反过来：**预测对象要便宜**，好让智能体用同一条真经验想象出许多条，并在想象里选动作或学策略。

---

## 一、共用骨架

$$
z_t=\mathrm{enc}(o_{\le t},a_{<t}),
\qquad
\hat z_{t+1}=f_\theta(z_t,a_t),
\qquad
\text{然后 MPC 或 Actor-Critic}
$$

[导论](/world-models/intro/) 里的序列 ELBO（先验 vs 后验）主要是为这条路径服务的：训练时用后验 $q(\cdot\mid o_t)$，部署时多步只滚先验，就是「做梦」。

| 章 | 角色 |
|----|------|
| [PETS](/world-models/abstract/pets/) | 状态已干净时：概率集成 + CEM-MPC（规划原型） |
| [RSSM / PlaNet](/world-models/abstract/rssm/) | 像素 → 随机状态空间，把 CEM 搬进潜空间 |
| [Dreamer](/world-models/abstract/dreamer/) | 想象轨迹上的 Actor-Critic；V4 开始接开放视频先验 |
| [MuZero](/world-models/abstract/muzero/) | 不重建观测，隐式模型 + 搜索 |
| [JEPA](/world-models/abstract/jepa/) | 预测表征而非像素 |
| [LeWM](/world-models/abstract/lewm/) | 把 JEPA 收成可规划的两项损失 |

建议顺序与上表一致。MuZero 可在 Dreamer 后穿插：对比「重建观测」vs「只为搜索服务的隐式模型」。

---

## 二、和其余路径

- 观感与数据引擎 → [路径一](/world-models/video/overview/)
- 可玩 / 3D 接口 → [路径二](/world-models/interactive/overview/)
- 干预 vs 相关 → [路径四](/world-models/causal/ladder/)
- 符号规则与 LLM 程序世界模型 → [路径五](/world-models/symbolic/overview/)

> 下一章：[PETS](/world-models/abstract/pets/)。
