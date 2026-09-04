<h1 align="center"> notebook（🧪 Beta公测版）</h1>

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

<p align="center">
  <strong>图解笔记 · 一个文件夹就是一章</strong>
</p>

<p align="center">
  AI · 量子信息 · 算法 · ROS 2 · 图解 + 可运行代码
</p>

<p align="center">
  <a href="#-目录总览">目录</a> ·
  <a href="#-快速开始">快速开始</a> ·
  <a href="#-每章结构">章节结构</a> ·
  <a href="#-在线阅读">文档站点</a> ·
  <a href="#-参与贡献">贡献</a>
</p>

---

## 为什么做这个仓库

AI 领域每天都有新论文、新框架、新名词。但真正关键的底层原理并不多——神经网络的训练、反向传播、注意力机制、强化学习的 Bellman 方程——几十年没变过。

这个仓库的目标：**用最直观的图解，配上一跑就能看到结果的代码，把概念一个一个讲清楚**。原先的 AI 教程仍在；ROS 2 Humble 笔记也收在同一个 `docs/` 树里。

每篇文章只聚焦一个知识点，20-30 分钟读完。所有代码**默认 CPU 运行**，消费级笔记本就能跑，GPU 作为可选项。

## 项目受众

本项目面向以下人群，帮助学习者建立可检索的笔记体系：

| 人群 | 能获得什么 | 基础要求 |
|------|-----------|----------|
| 在校学生 / 转行学习者 | 从感知机到大模型的完整知识体系 | 会一门编程语言（Python 最佳）、基础高数 |
| 大模型重度使用者（日常用 ChatGPT/Claude） | 理解 LLM 底层原理：Transformer、RLHF、RAG、推理优化 | 无需 ML 工程经验 |
| ML 工程师（已会深度学习） | 补经典 ML 理论 + 算法数据结构基础 | 熟悉 PyTorch/TensorFlow |
| 软件工程师转 AI | ML 理论 + DL 实战 + CV/NLP/RL 全链路 | 有工程经验，缺 ML 数学 |
| 其他行业转行 AI | 够用、能上手、能面试的核心内容 | 不限 CS 背景，肯动手跑代码 |
| 面试冲刺者 | 高频考点速通复习 | 已学过一遍 |
| 算法竞赛选手 | 算法与数据结构图解（algo01-algo16） | 有编程基础 |
| ROS 2 初学者 | Humble 入门（Topic → Gazebo），边做边记 | Ubuntu 22.04 + ROS 2 Humble |
| 量子信息学习者 | 计算 / 网络 / 存储 / 模拟 / 混合 QML | 线代直觉；VQNet 为可选 |

> 所有代码默认 CPU 运行，消费级笔记本即可跑通。每章 20-30 分钟读完：读正文（30min）→ 推公式（30min）→ 跑代码（30min）→ 做练习（30min）。

## 在线阅读

https://deconbear.github.io/notebook/

## 📑 目录总览

完整目录以 [在线站点侧栏](https://deconbear.github.io/notebook/) 为准。下面由 `docs/` 扫描生成；增删章节后运行 `npm run gen-toc`。

<!-- TOC:start -->


### 数学基础

- [线性代数直觉](docs/math/linear-algebra/)
- [概率与贝叶斯](docs/math/probability/)
- [优化与梯度](docs/math/optimization/)
- [信息论精简：熵与 KL](docs/math/information/)

### 机器学习

#### 基石

- [s01 AI 全景图](docs/ml/foundations/ai-overview/)
- [s02 线性回归](docs/ml/foundations/linear-regression/)
- [s03 逻辑回归](docs/ml/foundations/logistic-regression/)
- [s04 过拟合与正则化](docs/ml/foundations/bias-variance/)

#### 经典

- [ml01 k-近邻与距离度量](docs/ml/classic/knn/)
- [ml02 贝叶斯决策理论](docs/ml/classic/bayesian-decision/)
- [ml03 朴素贝叶斯与贝叶斯网络](docs/ml/classic/naive-bayes/)
- [ml04 支持向量机 (SVM)](docs/ml/classic/svm/)
- [ml05 决策树](docs/ml/classic/decision-tree/)

#### 进阶

- [ml06 集成学习：Bagging与随机森林](docs/ml/advanced/random-forest/)
- [ml07 集成学习：Boosting与Stacking](docs/ml/advanced/boosting/)
- [ml08 聚类：无监督学习的核心](docs/ml/advanced/clustering/)
- [ml09 降维与特征工程](docs/ml/advanced/dimensionality-reduction/)
- [ml10 蒙特卡洛方法](docs/ml/advanced/monte-carlo/)
- [ml11 隐马尔可夫模型 (HMM)](docs/ml/advanced/hmm/)
- [ml12 EM算法与高斯混合模型](docs/ml/advanced/em-gmm/)
- [ml13 概率图模型基础](docs/ml/advanced/probabilistic-graphical-models/)
- [ml14 核方法与高斯过程](docs/ml/advanced/kernel-gp/)

### 深度学习

- [s05 计算图与前向传播](docs/dl/forward-graph/)
- [s06 反向传播与链式法则](docs/dl/backprop/)
- [s07 多层网络矩阵反传](docs/dl/matrix-backprop/)
- [s08 优化器：SGD→Adam](docs/dl/optimizers/)
- [s09 Adam 深度解析](docs/dl/adam/)
- [混合专家 MoE](docs/dl/moe/)

### 计算机视觉

- [s10 CNN 核心原理](docs/cv/cnn/)
- [s11 经典架构演进](docs/cv/architectures/)
- [s12 目标检测](docs/cv/object-detection/)
- [s12b Vision Transformer](docs/cv/vit/)
- [s13 图像生成](docs/cv/generation/)

### 自然语言处理

- [s14 文本表示](docs/nlp/text-representation/)
- [s15 序列模型](docs/nlp/sequence-models/)
- [s16 Attention & Transformer](docs/nlp/transformer/)
- [s17 预训练范式](docs/nlp/pretrained/)
- [s18 大语言模型](docs/nlp/llm/)

### 强化学习

- [s19 MDP & Q-Learning](docs/rl/qlearning/)
- [s20 深度强化学习](docs/rl/deep-rl/)
- [s21 RLHF](docs/rl/rlhf/)

### 系统与应用

- [s22 多模态模型](docs/systems/multimodal/)
- [s23 RAG 与 Agent](docs/systems/rag-agent/)
- [s24 部署与推理优化](docs/systems/deployment/)
- [s25 AI 安全与对齐](docs/systems/safety/)

### 科学计算

- [as01 AI4S 全景](docs/science/overview/)
- [as02 PINN](docs/science/pinn/)
- [as03 Neural Operator 与 FNO](docs/science/fno/)
- [as04 PINO](docs/science/pino/)
- [as05 科学计算中的 GNN](docs/science/gnn/)
- [as06 蛋白质结构预测与 AlphaFold](docs/science/alphafold/)
- [as07 AlphaChip 与电路设计中的神经网络](docs/science/alphachip/)
- [as08 AI4S 综合与前沿](docs/science/frontier/)

### 量子信息

- [量子信息全景](docs/quantum/overview/)
- [量子计算](docs/quantum/computing/)
- [量子网络](docs/quantum/network/)
- [量子存储](docs/quantum/memory/)
- [量子模拟](docs/quantum/simulation/)
- [量子机器学习](docs/quantum/qml/)

### 世界模型

- [世界模型导论：四条路径](docs/world-models/intro/)
- [路径一 · 视频生成式世界模型](docs/world-models/video/)

#### 路径二 · 交互 / 3D 生成

- [Genie：从视频中长出可玩的世界](docs/world-models/interactive/genie/)
- [交互式 3D 世界：从 Genie 到可漫游场景](docs/world-models/interactive/scene-3d/)

#### 路径三 · 抽象状态预测

- [PETS：概率集成、轨迹采样与 MPC](docs/world-models/abstract/pets/)
- [RSSM 与 PlaNet](docs/world-models/abstract/rssm/)
- [Dreamer V1–V4：在想象里学会行动](docs/world-models/abstract/dreamer/)
- [MuZero：隐式世界模型](docs/world-models/abstract/muzero/)
- [JEPA / V-JEPA](docs/world-models/abstract/jepa/)
- [LeWM：两项损失的端到端 JEPA 世界模型](docs/world-models/abstract/lewm/)
- [路径四 · 因果世界模型](docs/world-models/causal/)
- [附录 · LLM 世界模型与路径对照](docs/world-models/llm/)

### ROS 2

- [ROS 2 导读](docs/ros2/overview/)
- [00 环境与工作区](docs/ros2/env/)
- [01 Topic 发布订阅](docs/ros2/topics/)
- [02 Service 请求应答](docs/ros2/services/)
- [03 Action 长任务](docs/ros2/actions/)
- [04 Parameter 参数](docs/ros2/parameters/)
- [05 Launch 一键启动](docs/ros2/launch/)
- [06 自定义消息](docs/ros2/custom-msg/)
- [07 自定义服务](docs/ros2/custom-srv/)
- [08 TF2 坐标变换](docs/ros2/tf2/)
- [09 速度控制 cmd_vel](docs/ros2/cmd-vel/)
- [10 URDF 与机器人描述](docs/ros2/urdf/)
- [11 Gazebo 最小仿真](docs/ros2/gazebo/)
- [12 RViz 与组合 Launch](docs/ros2/rviz-launch/)

### 算法与数据结构

#### 基础结构

- [algo01 复杂度分析与渐进记号](docs/algorithms/basics/complexity/)
- [algo02 数组、链表与哈希表](docs/algorithms/basics/arrays-hash/)
- [algo03 栈与队列](docs/algorithms/basics/stack-queue/)
- [algo04 树与二叉树](docs/algorithms/basics/tree/)
- [algo05 堆、并查集与跳跃表](docs/algorithms/basics/heap-unionfind/)

#### 图

- [algo06 图论基础](docs/algorithms/graph/basics/)
- [algo07 最短路径](docs/algorithms/graph/shortest-path/)
- [algo08 最小生成树与网络流](docs/algorithms/graph/mst-flow/)

#### 策略

- [algo09 贪心算法](docs/algorithms/strategy/greedy/)
- [algo10 递归、分治与二分](docs/algorithms/strategy/divide-conquer/)
- [algo11 动态规划（上）](docs/algorithms/strategy/dp-1/)
- [algo12 动态规划（下）](docs/algorithms/strategy/dp-2/)

#### 专题

- [algo13 字符串算法](docs/algorithms/topics/string/)
- [algo14 线段树与树状数组](docs/algorithms/topics/segment-tree/)
- [algo15 数论与组合数学](docs/algorithms/topics/number-theory/)
- [algo16 计算几何与博弈论入门](docs/algorithms/topics/geometry-game/)

### 论文精读

- [paper00 占位：经典论文讲解即将更新](docs/papers/placeholder/)

<!-- TOC:end -->

## 🧭 学习路径推荐

不同背景的学习者，建议的学习顺序不同：

| 路径 | 适用人群 | 推荐顺序 |
|------|---------|----------|
| 🔵 **系统学习** | AI 零基础，建立完整知识体系 | **数学基础** → 阶段一 → 二 → 三 → 四/五/六/七，按序推进 |
| 🟡 **LLM 重度用户** | 日常用 ChatGPT/Claude，想懂原理 | s01 → s14-s18(NLP) → s21(RLHF) → s22-s23(多模态/RAG) → s25(安全) |
| 🟢 **ML 工程师** | 已会深度学习，补经典 ML 理论 | 阶段二（ml01-ml05 必修）→ 番外一（集成树/聚类/降维）→ 附录算法 |
| 🟠 **开发转 AI** | 有编程基础，缺 ML 理论和 DL 实战 | **数学基础** → 阶段一速览 → ml01/04/05 → 阶段三~七全量 |
| 🔵 **其他行业转行** | 非 CS/数学背景，目标上手+面试 | **数学基础** → s01-s04 直觉→ml01/04/05→s05-s09→s10/s16/s18→s23/s25 |
| 🟣 **面试冲刺** | 已学过，快速复习高频考点 | s02-s04 → ml04(SVM) → ml05(树) → s06-s09 → s16 → s18 → s21 → s25 |
| 🔴 **算法竞赛** | 只关注算法与数据结构 | 直接看附录 algo01 → algo16，其余章节按需查阅 |
| 🟤 **科研向 / AI4S** | 做科学计算、生物、芯片等交叉 | 阶段三~五 → 进阶一（as01–as08） |
| 🟦 **量子信息** | 计算 / 网络 / 存储 / 模拟 / QML | 侧栏 **量子信息**（`docs/quantum/`）；QML 可选本源 VQNet |
| ⬛ **世界模型** | 关注具身智能 / 生成式模拟 | 阶段六 → 进阶二（四路径：视频 / 交互·3D / 抽象状态 / 因果） |
| 🟠 **ROS 2 / 机器人** | 要在本机跑 Humble | 侧栏 ROS 2；工作区 `workspaces/ros2-humble/` |

> 💡 **番外一**（集成学习、聚类、降维、蒙特卡洛、HMM、EM、概率图、高斯过程）在阶段二之后，默认折叠。内容独立、互不依赖，可按需跳读。**进阶一 / 进阶二**在阶段七之后，默认折叠；**番外二**为论文精读占位，敬请期待。

## 🚀 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/DeconBear/notebook.git
cd notebook

# 2. 安装 Python 依赖
pip install -r requirements.txt

# 3. 运行任意章节的代码
cd docs/ml/foundations/ai-overview/code
python demo.py

# 4. 启动文档站点（可选）
npm install
npm run dev
```

## 📂 每章结构

```
docs/<领域>/.../<slug>/
├── index.md               # 图解正文（核心阅读材料）
├── _meta.yaml             # 仅分组目录需要：标题、顺序、是否折叠
├── code-demo.md           # demo.py 保姆级逐段讲解
├── code-exercise.md       # exercise.py 练习指南
├── code/
│   ├── demo.py            # 完整教学代码（中文注释）——唯一源
│   └── exercise.py        # 动手练习（含 TODO）
└── images/                # 手绘图解
```

增删章节只动文件夹，不必改侧栏配置：

```bash
npm run new-chapter -- ml/foundations kernel-methods --title "核方法入门" --order 25
npm run gen-toc   # 更新本 README 目录
```

ROS 2 课没有 `demo.py`：笔记在 `docs/ros2/`，colcon 工作区在 `workspaces/ros2-humble/`。

## 贡献者名单

| 姓名 | 职责 | 简介 |
| :----| :---- | :---- |
| [DeconBear](https://github.com/DeconBear) | 项目负责人 | 从感知机到大模型，图解 AI 核心概念 |

> 本项目目前由项目负责人独立完成全部 55 章内容。欢迎通过 Issue / PR 参与贡献，贡献者将在此处登记。

## 参与贡献

- 如果你发现了一些问题，可以提Issue进行反馈，如果提完没有人回复你可以联系[保姆团队](https://github.com/datawhalechina/DOPMC/blob/main/OP.md)的同学进行反馈跟进~
- 如果你想参与贡献本项目，可以提Pull Request，如果提完没有人回复你可以联系[保姆团队](https://github.com/datawhalechina/DOPMC/blob/main/OP.md)的同学进行反馈跟进~
- 如果你对 Datawhale 很感兴趣并想要发起一个新的项目，请按照[Datawhale开源项目指南](https://github.com/datawhalechina/DOPMC/blob/main/GUIDE.md)进行操作即可~

## 关注我们

<div align=center>
<p>扫描下方二维码关注公众号：Datawhale</p>
<img src="https://raw.githubusercontent.com/datawhalechina/pumpkin-book/master/res/qrcode.jpeg" width = "180" height = "180">
</div>

## 致谢

受以下优秀项目启发：

- [learn-claude-code](https://github.com/shareAI-lab/learn-claude-code) — 仓库结构理念
- [3Blue1Brown](https://www.3blue1brown.com/) — 「先直觉，后公式」的教学哲学
- [nanoGPT](https://github.com/karpathy/nanoGPT) — 从零实现的教学思路
- [Distill.pub](https://distill.pub/) — 图解学术文章先驱

## LICENSE

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a><br />本作品采用<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议</a>进行许可。

> 教程内容（文章、图解）采用 CC BY-NC-SA 4.0 协议；配套示例代码沿用 MIT 协议，可自由使用。`workspaces/ros2-humble/` 中的 ROS 2 包沿用 [Apache-2.0](workspaces/ros2-humble/LICENSE)。
