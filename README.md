<h1 align="center"> learn-ai（🧪 Beta公测版）</h1>

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

<p align="center">
  <strong>图解 AI · 一行代码看懂一个概念</strong>
</p>

<p align="center">
  从感知机到大模型 · 55 篇文章 · 200+ 张图解 · 100+ 个可运行代码示例
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

这个仓库的目标：**用最直观的图解，配上一跑就能看到结果的代码，把 AI 的核心概念一个一个讲清楚**。

每篇文章只聚焦一个知识点，20-30 分钟读完。所有代码**默认 CPU 运行**，消费级笔记本就能跑，GPU 作为可选项。

## 项目受众

本项目面向以下人群，帮助学习者系统建立 AI 知识体系：

| 人群 | 能获得什么 | 基础要求 |
|------|-----------|----------|
| 在校学生 / 转行学习者 | 从感知机到大模型的完整知识体系 | 会一门编程语言（Python 最佳）、基础高数 |
| 大模型重度使用者（日常用 ChatGPT/Claude） | 理解 LLM 底层原理：Transformer、RLHF、RAG、推理优化 | 无需 ML 工程经验 |
| ML 工程师（已会深度学习） | 补经典 ML 理论 + 算法数据结构基础 | 熟悉 PyTorch/TensorFlow |
| 软件工程师转 AI | ML 理论 + DL 实战 + CV/NLP/RL 全链路 | 有工程经验，缺 ML 数学 |
| 其他行业转行 AI | 够用、能上手、能面试的核心内容 | 不限 CS 背景，肯动手跑代码 |
| 面试冲刺者 | 高频考点速通复习 | 已学过一遍 |
| 算法竞赛选手 | 算法与数据结构图解（algo01-algo16） | 有编程基础 |

> 所有代码默认 CPU 运行，消费级笔记本即可跑通。每章 20-30 分钟读完：读正文（30min）→ 推公式（30min）→ 跑代码（30min）→ 做练习（30min）。

## 在线阅读

https://deconbear.github.io/learn-ai/

## 📑 目录总览

### 阶段一：机器学习基石（4 篇）

| 编号 | 标题 | 核心内容 | 代码实操 | 状态 |
|------|------|----------|----------|------|
| [s01](s01_ai_overview/) | AI 全景图 | AI/ML/DL 关系、三大范式、发展简史 | NumPy 手写感知机 | ✅ |
| [s02](s02_linear_regression/) | 线性回归 | 模型-损失-优化三要素、梯度下降 | 从零实现 + 正规方程 + sklearn 对比 | ✅ |
| [s03](s03_logistic_regression/) | 逻辑回归与分类 | Sigmoid、交叉熵、Softmax 多分类 | 手写二分类 + 多分类器 | ✅ |
| [s04](s04_bias_variance/) | 过拟合与正则化 | Bias-Variance 权衡、L1/L2、交叉验证 | 多项式拟合 + 正则化路径 + K-Fold | ✅ |

### 阶段二：经典机器学习（5 篇）

| 编号 | 标题 | 核心内容 | 代码实操 | 状态 |
|------|------|----------|----------|------|
| [ml01](ml01_knn/) | k-近邻与距离度量 | k-NN、距离度量、维数灾难 | KNN 分类器 + 决策边界可视化 | ✅ |
| [ml02](ml02_bayesian_decision/) | 贝叶斯决策理论 | 先验/后验、最小风险决策、ROC/AUC | 高斯贝叶斯分类器 | ✅ |
| [ml03](ml03_naive_bayes/) | 朴素贝叶斯 | 条件独立性、高斯/多项式 NB、拉普拉斯平滑 | 垃圾邮件检测 + 文本分类 | ✅ |
| [ml04](ml04_svm/) | 支持向量机 (SVM) | 最大间隔、对偶问题、核技巧、软间隔 | 线性 SVM + RBF 核可视化 | ✅ |
| [ml05](ml05_decision_tree/) | 决策树 | ID3/C4.5/CART、信息增益、剪枝 | CART 决策树 + 决策面可视化 | ✅ |

### 番外：经典机器学习进阶（9 篇，可选）

| 编号 | 标题 | 核心内容 | 代码实操 | 状态 |
|------|------|----------|----------|------|
| [ml06](ml06_random_forest/) | 集成学习：Bagging 与随机森林 | Bootstrap、OOB、特征子空间 | 随机森林 + 特征重要性 | ✅ |
| [ml07](ml07_boosting/) | 集成学习：Boosting 与 Stacking | AdaBoost、GBDT、XGBoost | AdaBoost + 简单 GBDT | ✅ |
| [ml08](ml08_clustering/) | 聚类 | K-Means、DBSCAN、层次聚类 | 聚类算法对比可视化 | ✅ |
| [ml09](ml09_dimensionality_reduction/) | 降维与特征工程 | PCA、t-SNE、LDA | PCA 手写 + t-SNE 可视化 | ✅ |
| [ml10](ml10_monte_carlo/) | 蒙特卡洛方法 | MC 积分、重要性采样、MCMC | π 估计 + Metropolis-Hastings | ✅ |
| [ml11](ml11_hmm/) | 隐马尔可夫模型 (HMM) | 前向算法、Viterbi、Baum-Welch | HMM + 词性标注 | ✅ |
| [ml12](ml12_em_gmm/) | EM 算法与高斯混合模型 | E-Step/M-Step、GMM、ELBO | GMM 聚类 + 软分配可视化 | ✅ |
| [ml13](ml13_probabilistic_graphical_models/) | 概率图模型 | 贝叶斯网、d-分离、信念传播 | 变量消除 + 因子图推理 | ✅ |
| [ml14](ml14_kernel_gp/) | 核方法与高斯过程 | 核技巧、KRR、GP 回归 | GP 回归 + 不确定性可视化 | ✅ |

### 阶段三：深度学习基础（5 篇）

| 编号 | 标题 | 核心内容 | 代码实操 | 状态 |
|------|------|----------|----------|------|
| [s05](s05_forward_computation_graph/) | 计算图与前向传播 | 计算图、感知机、激活函数深度解析 | 纯 NumPy 搭建 MLP | ✅ |
| [s06](s06_backprop_chain_rule/) | 反向传播与链式法则 | 局部梯度规则、链式法则、fan-out | 从零实现 mini autograd 引擎 | ✅ |
| [s07](s07_matrix_backprop/) | 多层网络的矩阵反传 | δ 递推公式、梯度检查、消失/爆炸 | 手写 MLP + 梯度检查 | ✅ |
| [s08](s08_optimizers_sgd_to_adam/) | 优化器：从 SGD 到 Adam | Momentum、RMSProp、自适应步长 | 四种优化器轨迹对比 | ✅ |
| [s09](s09_adam_deep_dive/) | Adam 深度解析 | 偏差修正、AdamW、梯度裁剪、诊断 | MNIST 训练 + 优化器对比 | ✅ |

### 阶段四：计算机视觉（5 篇）

| 编号 | 标题 | 核心内容 | 代码实操 | 状态 |
|------|------|----------|----------|------|
| [s10](s10_cnn_fundamentals/) | CNN 核心原理 | 卷积、池化、感受野、参数共享 | 从零实现 Conv2d + 特征图 | ✅ |
| [s11](s11_cnn_architectures/) | 经典架构演进 | LeNet → ResNet → EfficientNet | 从零写 ResNet 训练 CIFAR-10 | ✅ |
| [s12](s12_object_detection/) | 目标检测 | R-CNN → YOLO、IoU、NMS、mAP | 从零实现 IoU + NMS | ✅ |
| [s12b](s11b_vit/) | Vision Transformer | Patch Embedding、位置编码 | ViT vs CNN 对比实验 | ✅ |
| [s13](s13_image_generation/) | 图像生成 | GAN、VAE、扩散模型原理 | 训练 GAN + VAE 生成 MNIST | ✅ |

### 阶段五：自然语言处理（5 篇）

| 编号 | 标题 | 核心内容 | 代码实操 | 状态 |
|------|------|----------|----------|------|
| [s14](s14_text_representation/) | 文本表示 | 词袋→TF-IDF→word2vec | 训练 Skip-gram + 词向量可视化 | ✅ |
| [s15](s15_sequence_models/) | 序列模型 | RNN、LSTM、GRU 门控机制 | 字符级语言模型 + 情感分类 | ✅ |
| [s16](s16_attention_transformer/) | Attention & Transformer | Q/K/V、多头注意力、位置编码 | 从零实现 nanoGPT | ✅ |
| [s17](s17_pretrained_models/) | 预训练范式 | BERT vs GPT、MLM vs CLM | BERT 微调 + 掩码预测 | ✅ |
| [s18](s18_large_language_models/) | 大语言模型 | Scaling Law、涌现、RLHF/DPO | LoRA 微调 + DPO 对齐 | ✅ |

### 阶段六：强化学习（3 篇）

| 编号 | 标题 | 核心内容 | 代码实操 | 状态 |
|------|------|----------|----------|------|
| [s19](s19_rl_qlearning/) | 强化学习入门 | MDP、Q 表、ε-greedy、Bellman | Q-Learning 走迷宫 | ✅ |
| [s20](s20_deep_rl/) | 深度强化学习 | DQN、经验回放、REINFORCE | DQN 玩 CartPole | ✅ |
| [s21](s21_rlhf/) | RLHF | PPO、DPO、Reward Model | PPO + DPO 对比训练 | ✅ |

### 阶段七：前沿与应用（4 篇）

| 编号 | 标题 | 核心内容 | 代码实操 | 状态 |
|------|------|----------|----------|------|
| [s22](s22_multimodal/) | 多模态模型 | CLIP、对比学习、LLaVA 架构 | CLIP 零样本分类 | ✅ |
| [s23](s23_rag_agent/) | RAG 与 AI Agent | 检索增强、ReAct、工具调用 | 完整 RAG + Agent 系统 | ✅ |
| [s24](s24_deployment_inference/) | 部署与推理优化 | KV Cache、量化、Flash Attention | KV Cache + INT8 量化 | ✅ |
| [s25](s25_ai_safety/) | AI 安全与对齐 | 幻觉、越狱、偏见、深度防御 | 安全扫描 + 幻觉检测 | ✅ |

### 附录：算法与数据结构基础（16 节，默认折叠）

| 编号 | 标题 | 核心内容 | 代码实操 | 状态 |
|------|------|----------|----------|------|
| [algo01](algo01_complexity/) | 复杂度分析 | 大 O/Ω/Θ、主定理、均摊分析 | 排序算法复杂度对比 | ✅ |
| [algo02](algo02_arrays_linkedlist_hash/) | 数组链表哈希表 | 动态数组、链表、哈希表、LRU | 各数据结构手写实现 | ✅ |
| [algo03](algo03_stack_queue/) | 栈与队列 | 单调栈/队列、表达式求值 | 表达式求值 + 滑动窗口 | ✅ |
| [algo04](algo04_tree_binarytree/) | 树与二叉树 | 遍历、BST、AVL 旋转、哈夫曼 | BST + AVL 手写实现 | ✅ |
| [algo05](algo05_heap_unionfind_skiplist/) | 堆并查集跳跃表 | 堆排序、路径压缩、SkipList | 堆 + 并查集 + Kruskal | ✅ |
| [algo06](algo06_graph_basics/) | 图论基础 | 存储方式、BFS/DFS、拓扑排序 | 图的遍历 + 拓扑排序 | ✅ |
| [algo07](algo07_shortest_path/) | 最短路径 | Dijkstra、Bellman-Ford、Floyd、A* | 最短路算法全实现 | ✅ |
| [algo08](algo08_mst_networkflow/) | MST 与网络流 | Prim/Kruskal、Dinic、二分图匹配 | MST + 最大流 | ✅ |
| [algo09](algo09_greedy/) | 贪心算法 | 活动选择、哈夫曼编码、区间调度 | 贪心 + 正确性证明 | ✅ |
| [algo10](algo10_divide_conquer/) | 递归分治与二分 | 归并排序、快速排序、二分查找 | 分治算法 + 逆序对 | ✅ |
| [algo11](algo11_dp_1/) | 动态规划（上） | 背包、LCS、LIS、编辑距离 | DP 经典问题全实现 | ✅ |
| [algo12](algo12_dp_2/) | 动态规划（下） | 区间/树形/状压/数位 DP | 高级 DP + 优化 | ✅ |
| [algo13](algo13_string/) | 字符串算法 | KMP、Trie、AC 自动机、Manacher | 字符串匹配全家桶 | ✅ |
| [algo14](algo14_segment_tree/) | 线段树与树状数组 | BIT、线段树、懒标记、可持久化 | 线段树 + BIT 手写 | ✅ |
| [algo15](algo15_number_theory/) | 数论与组合数学 | 快速幂、筛法、CRT、组合数 | 数论算法 + 取模运算 | ✅ |
| [algo16](algo16_geometry_game/) | 计算几何与博弈论 | 叉积、凸包、Nim、SG 函数 | Graham Scan + Nim 求解 | ✅ |

---

## 🧭 学习路径推荐

不同背景的学习者，建议的学习顺序不同：

| 路径 | 适用人群 | 推荐顺序 |
|------|---------|----------|
| 🔵 **系统学习** | AI 零基础，建立完整知识体系 | 阶段一 → 二 → 三 → 四/五/六/七，按序推进 |
| 🟡 **LLM 重度用户** | 日常用 ChatGPT/Claude，想懂原理 | s01 → s14-s18(NLP) → s21(RLHF) → s22-s23(多模态/RAG) → s25(安全) |
| 🟢 **ML 工程师** | 已会深度学习，补经典 ML 理论 | 阶段二（ml01-ml05 必修）→ 番外（集成树/聚类/降维）→ 附录算法 |
| 🟠 **开发转 AI** | 有编程基础，缺 ML 理论和 DL 实战 | 阶段一速览 → ml01/04/05 → 阶段三~七全量 |
| 🔵 **其他行业转行** | 非 CS/数学背景，目标上手+面试 | s01-s04 直觉→ml01/04/05→s05-s09 入门→s10/s16/s18 项目→s23/s25 加分 |
| 🟣 **面试冲刺** | 已学过，快速复习高频考点 | s02-s04 → ml04(SVM) → ml05(树) → s06-s09 → s16 → s18 → s21 → s25 |
| 🔴 **算法竞赛** | 只关注算法与数据结构 | 直接看附录 algo01 → algo16，其余章节按需查阅 |

> 💡 **番外篇**（集成学习、聚类、降维、蒙特卡洛、HMM、EM、概率图、高斯过程）在阶段二之后，默认折叠。内容独立、互不依赖，可按需跳读。

## 🚀 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/DeconBear/learn-ai.git
cd learn-ai

# 2. 安装 Python 依赖
pip install -r requirements.txt

# 3. 运行任意章节的代码
cd s01_ai_overview/code
python demo.py

# 4. 启动文档站点（可选）
npm install
npm run dev
```

## 📂 每章结构

```
sXX_topic/
├── index.md               # 图解正文（核心阅读材料）
├── code-demo.md           # demo.py 保姆级逐段讲解
├── code-exercise.md       # exercise.py 练习指南
├── code/
│   ├── demo.py            # 完整教学代码（中文注释）
│   └── exercise.py        # 动手练习（含 TODO）
└── images/                # 手绘图解
```

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

> 教程内容（文章、图解）采用 CC BY-NC-SA 4.0 协议；配套示例代码沿用 MIT 协议，可自由使用。
