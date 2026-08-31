---
title: "JEPA / V-JEPA"
order: 50
legacyPaths:
  - /wm05_jepa/
  - /world-models/jepa/
---
# JEPA / V-JEPA：预测表征，而不是预测像素

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

> 一个反直觉但深刻的想法：想让模型理解世界，不必让它学会"画画"

---

## 一、生成式世界模型的隐痛

前几节我们看到的世界模型——PlaNet/RSSM、Dreamer、甚至 MuZero 的隐式模型——大多要在某种程度上"重建"下一时刻的观测（像素、隐状态解码后的像素，或至少要能评估价值）。而 Sora 一类的视频生成模型，则是把"重建像素"这件事做到了极致。

但重建像素有一个根本性的代价：**像素里混杂着大量与"理解世界"无关的高频细节**——树叶的抖动纹理、噪点、光照的微小闪烁。一个被要求"逐像素重建"的模型，会被迫花费大量容量去拟合这些不可预测、也不重要的细节。

Yann LeCun 在提出 JEPA（Joint Embedding Predictive Architecture，联合嵌入预测架构）时正是针对这个问题：

> 我们真正想要的不是"生成看起来逼真的像素"，而是"预测世界在语义表征层面会如何变化"。

**核心思路的转变**：把预测任务从"像素空间"搬到"表征空间"。

$$
\text{生成式：} \quad \hat{x}_{\text{target}} = \text{Decoder}(\text{Encoder}(x_{\text{context}})) \approx x_{\text{target}}
$$

$$
\text{JEPA：} \quad \hat{z}_{\text{target}} = \text{Predictor}(\text{Encoder}_{\text{ctx}}(x_{\text{context}})) \approx \text{Encoder}_{\text{tgt}}(x_{\text{target}}) = z_{\text{target}}
$$

模型不再被要求还原出目标区域的每一个像素，只需要让预测的**表征** $\hat{z}_{\text{target}}$ 逼近目标编码器输出的表征 $z_{\text{target}}$。表征空间天然是"有损"且"语义化"的，编码器在训练中会自动学会丢弃不可预测的噪声、保留可预测的结构。

![JEPA：预测表征，而不是预测像素](./images/wm05-01-jepa.png)

> **图解说明**：左路是生成式「解码回像素」；右路 JEPA 只让预测器在表征空间逼近目标编码器输出。丢弃不可预测的高频噪声，把容量留给可预测的结构——这是与 Sora 类像素生成路径的根本分野。

---

## 二、I-JEPA：图像上的联合嵌入预测

### 2.1 三个模块

I-JEPA（Image-based JEPA，2023）由三个核心模块组成：

| 模块 | 作用 | 训练方式 |
|------|------|----------|
| **上下文编码器** $E_{\theta}$ | 编码可见（未被遮挡）的图像 patch | 梯度下降，正常训练 |
| **目标编码器** $E_{\bar\theta}$ | 编码全图（包括被遮挡区域），提供回归目标 | **EMA**（指数滑动平均）跟随上下文编码器，不接收梯度 |
| **预测器** $P_{\phi}$ | 输入上下文表征 + 待预测位置的掩码 token，输出对目标表征的预测 | 梯度下降，正常训练 |

### 2.2 掩码策略：多块（multi-block）掩码

I-JEPA 不是随机遮挡单个 patch（太容易通过局部纹理插值"抄近路"），而是遮挡若干个**大的连续矩形块**（通常占图像面积的 15%~70%），强迫预测器必须依赖对全局语义结构的理解，而非局部纹理外推。

### 2.3 损失函数

设可见（上下文）patch 索引集合为 $\mathcal{C}$，被遮挡（目标）patch 索引集合为 $\mathcal{T}$：

$$
\mathcal{L} = \frac{1}{|\mathcal{T}|}\sum_{i \in \mathcal{T}} \left\| P_{\phi}\big(E_{\theta}(x_{\mathcal{C}}),\, \text{pos}(i)\big) - \text{sg}\big[E_{\bar\theta}(x)_i\big] \right\|_2^2
$$

其中 $\text{sg}[\cdot]$ 表示 stop-gradient（停止梯度）——目标表征只作为回归目标，不参与反向传播。

### 2.4 为什么不会"表征坍缩"？

自监督学习的经典陷阱是**表征坍缩**（representational collapse）：模型发现"把所有输入都编码成同一个常数向量"也能让损失降到 0（预测器只需学会输出这个常数）。

I-JEPA 用两个设计避免坍缩：

1. **非对称结构**：上下文编码器和目标编码器不共享参数（只用 EMA 单向跟随），打破了"预测器和编码器合谋输出常数"的对称性
2. **停止梯度**：目标编码器完全不接收梯度，其参数只能通过 EMA 缓慢演化，为预测任务提供一个"移动但不塌陷"的目标

这与 BYOL、DINO 等自监督方法使用的 EMA + 停止梯度技巧是同源的。

---

## 三、V-JEPA：从图像到视频

### 3.1 时空联合掩码

V-JEPA（2024）把 I-JEPA 的思想扩展到视频：输入变成一段视频的时空 patch（每个 patch 既有空间维度又有时间维度），掩码策略变成**时空块掩码**——遮挡若干帧中的一大块连续区域，甚至可以遮挡整个未来帧。

$$
\hat{z}_{t_1:t_2}^{\text{masked}} = P_{\phi}\Big(E_{\theta}\big(v_{\text{visible}}\big)\Big) \approx \text{sg}\Big[E_{\bar\theta}\big(v_{t_1:t_2}\big)\Big]
$$

模型被迫从可见的时空上下文中推断出"被遮挡的时空区域在语义上应该是什么样"——这本质上就是在做**时间维度上的世界建模**：给定过去和部分未来的上下文，预测缺失片段的语义演化。

### 3.2 V-JEPA 的关键发现

V-JEPA 论文报告了几个重要结果：

- **冻结编码器 + 轻量探针**：把训练好的编码器权重冻结，只在其输出上训练一个简单的线性/浅层分类头，就能在动作识别、物体交互识别等任务上取得很强的效果——说明表征本身已经蕴含了丰富的动作/物理语义
- **无需人工标注**：整个预训练过程完全自监督，不需要任何动作标签或文本描述
- **对像素级细节不敏感**：与逐像素重建的视频模型相比，V-JEPA 学到的表征对光照变化、纹理噪声等"无关变化"更鲁棒

### 3.3 V-JEPA 2：走向机器人world model

V-JEPA 2（2025）在更大规模的视频数据（超百万小时）上预训练后，进一步接入一个**动作条件的预测器**，使其可以在给定机器人动作序列的情况下预测未来的表征轨迹——从而可以直接用于**零样本机器人规划**：在表征空间里做模型预测控制（MPC），选择能让预测表征轨迹接近目标表征的动作序列。

$$
a^* = \arg\min_{a_{1:H}} \; \left\| P_{\phi}\Big(z_0,\, a_{1:H}\Big) - z_{\text{goal}} \right\|_2^2
$$

这标志着 JEPA 家族从"纯表征学习"走向了"可用于决策的世界模型"，与前几节的 PlaNet/Dreamer 在功能定位上开始汇合——只不过 JEPA 走的是"预测表征"而非"预测/重建观测"的路线。

---

## 四、JEPA 家族演进一览

```mermaid
graph LR
    A["I-JEPA (2023)<br/>图像·联合嵌入预测"] --> B["V-JEPA (2024)<br/>视频·时空块掩码"]
    B --> C["V-JEPA 2 (2025)<br/>+动作条件预测器<br/>→ 机器人零样本规划"]
```

| 版本 | 输入 | 掩码单位 | 新增能力 |
|------|------|----------|----------|
| I-JEPA | 静态图像 | 图像空间的矩形块 | 高效自监督图像表征学习 |
| V-JEPA | 视频片段 | 时空块（多帧×区域） | 学到隐含运动/物理语义的视频表征 |
| V-JEPA 2 | 视频 + 动作 | 时空块 | 动作条件预测 → 机器人规划（MPC） |

---

## 五、JEPA vs 生成式世界模型：路线对比

| 维度 | 生成式（重建像素/观测） | JEPA（预测表征） |
|------|--------------------------|-------------------|
| 预测目标 | 原始观测（像素、token） | 编码器输出的表征向量 |
| 典型代表 | Dreamer 的图像解码器、Sora、Cosmos | I-JEPA、V-JEPA、V-JEPA 2 |
| 是否需要解码器 | 需要（重建像素） | 不需要（只需编码器+预测器） |
| 对像素噪声的敏感度 | 高（必须拟合所有细节） | 低（表征空间自动过滤） |
| 训练稳定性风险 | 生成质量退化、模式坍缩 | 表征坍缩（需 EMA+停止梯度规避） |
| 下游可解释性 | 可直接可视化生成结果 | 需要额外的探针/解码才能可视化 |
| 典型应用 | 内容生成、可视化规划 | 表征学习、零样本迁移、机器人 MPC |

> 没有绝对的优劣：如果你需要「生成一段逼真的视频」，走路径一；如果你需要「高效、鲁棒的语义结构以支持决策」，JEPA 属于路径三的表征预测分支。总图见 [导论 · 四路径](/world-models/intro/)，收束对照见 [附录 · LLM](/world-models/llm/)。

---

## 六、本节小结

| 概念 | 一句话 |
|------|--------|
| JEPA | 联合嵌入预测架构：预测表征而非像素/观测 |
| 上下文编码器 | 编码可见 patch，正常接收梯度训练 |
| 目标编码器 | EMA 动量更新 + 停止梯度，提供稳定但持续演化的回归目标 |
| 预测器 | 输入上下文表征 + 掩码位置 token，预测目标位置的表征 |
| 表征坍缩 | 自监督学习的经典陷阱：模型学到输出常数向量也能让损失最小化 |
| I-JEPA | 图像版，矩形块掩码 |
| V-JEPA | 视频版，时空块掩码，学到运动/物理语义 |
| V-JEPA 2 | 加入动作条件预测器，可用于机器人零样本 MPC 规划 |

> 同属路径三的下一步是 [LeWM](/world-models/abstract/lewm/)：把 JEPA 做成可端到端、少超参、能在潜空间做 MPC 的世界模型。交互生成则见路径二 [Genie](/world-models/interactive/genie/)。

## 📥 Code

| File | View | Download |
|------|------|----------|
| demo.py | [Open](./code-demo) | <a href="/notebook/code/world-models/abstract/jepa/demo.py" target="_blank" download>Download</a> |
| exercise.py | [Open](./code-exercise) | <a href="/notebook/code/world-models/abstract/jepa/exercise.py" target="_blank" download>Download</a> |

## 参考

1. LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence. *Open Review*. (JEPA 概念提出)
2. Assran, M., et al. (2023). Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture. *CVPR 2023*. (I-JEPA) [[arXiv:2301.08243](https://arxiv.org/abs/2301.08243)]
3. Bardes, A., et al. (2024). V-JEPA: Video Joint-Embedding Predictive Architecture. (V-JEPA) [[arXiv:2404.08471](https://arxiv.org/abs/2404.08471)]
4. Assran, M., et al. (2025). V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning. (V-JEPA 2) [[arXiv:2506.09985](https://arxiv.org/abs/2506.09985)]
5. Grill, J.-B., et al. (2020). Bootstrap Your Own Latent (BYOL). *NeurIPS 2020*. (EMA+停止梯度的先驱工作) [[arXiv:2006.07733](https://arxiv.org/abs/2006.07733)]
