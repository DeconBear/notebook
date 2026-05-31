# learn-ai：图解 AI，一行代码看懂一个概念

<p align="center">
  <strong>从感知机到大模型 · 55 篇文章 · 200+ 张图解 · 100+ 个可运行代码示例</strong>
</p>

<p align="center">
  <a href="#-目录总览">目录</a> ·
  <a href="#-快速开始">快速开始</a> ·
  <a href="#-每章结构">章节结构</a> ·
  <a href="#-文档站点">文档站点</a> ·
  <a href="#-贡献">贡献</a>
</p>

---

## 为什么做这个仓库

AI 领域每天都有新论文、新框架、新名词。但真正关键的底层原理并不多——神经网络的训练、反向传播、注意力机制、强化学习的 Bellman 方程——几十年没变过。

这个仓库的目标：**用最直观的图解，配上一跑就能看到结果的代码，把 AI 的核心概念一个一个讲清楚**。

每篇文章只聚焦一个知识点，20-30 分钟读完。所有代码**默认 CPU 运行**，消费级笔记本就能跑，GPU 作为可选项。

---

## 学习路径推荐

不同背景的学习者，建议按不同的路径学习。下面是七种典型路径：

### 🔵 路径 A：AI 零基础系统学习

> 适合：在校学生、转行学习者，希望建立完整知识体系

**按顺序推进**：阶段一 → 阶段二 → 阶段三 → 阶段四 → 阶段五 → 阶段六 → 阶段七

建议每章花 1-2 天：读正文（30min）→ 推公式（30min）→ 跑代码（30min）→ 做练习（30min）。

### 🟡 路径 B：大模型重度使用者

> 适合：日常使用 ChatGPT/Claude，想理解 AI 底层原理，不追求成为 ML 工程师

| 顺序 | 章节 | 重点 |
|------|------|------|
| 1 | s01 | AI 全景：快速搞懂 AI/ML/DL 的区别，建立概念地图 |
| 2 | s14-s18 | NLP 完整链路：词向量→RNN→Transformer→BERT/GPT→LLM，理解 ChatGPT 的"引擎" |
| 3 | s21 | RLHF：ChatGPT 怎么能"听懂人话"、对齐人类偏好 |
| 4 | s22-s23 | 多模态 + RAG/Agent：GPT-4V 怎么看图、AI 怎么调用工具 |
| 5 | s25 | AI 安全：幻觉、越狱、偏见——大模型有哪些坑 |
| 6 | s24 | 部署推理：KV Cache、量化——为什么大模型又贵又慢 |

> 💡 **跳过**：阶段一~二（数学细节）、CV、RL 理论、附录。目标不是训练模型，而是理解模型。

### 🟢 路径 C：ML 工程师补理论和算法

> 适合：已会深度学习，想补经典 ML 理论 + 算法基础

| 顺序 | 章节 | 重点 |
|------|------|------|
| 1 | ml01-ml04 | k-NN、贝叶斯、朴素贝叶斯、SVM — ML 四大基础分类器 |
| 2 | ml05 决策树 | 树模型基础，为集成学习做铺垫 |
| 3 | 番外 ml06-ml07 | 随机森林、AdaBoost/GBDT/XGBoost — 工业界最爱 |
| 4 | 番外 ml08-ml09 | 聚类、降维 — 无监督学习核心 |
| 5 | 附录 algo01-algo16 | 算法与数据结构 — 面试/竞赛必备 |

### 🟠 路径 D：软件工程师转 AI

> 适合：有编程基础，缺 ML 理论 + DL 实战经验

| 顺序 | 章节 | 重点 |
|------|------|------|
| 1 | s01 | AI 全景，快速建立大局观 |
| 2 | s02-s04 | 线性回归、逻辑回归、过拟合 — 速通 ML 三要素 |
| 3 | ml01, ml04, ml05 | k-NN、SVM、决策树 — 核心分类器 |
| 4 | s05-s09 | 计算图→反向传播→优化器 — DL 基本功 |
| 5 | s10-s13 | CNN→目标检测→图像生成 — CV 方向 |
| 6 | s14-s18 | 文本表示→Transformer→LLM — NLP 方向 |
| 7 | s22-s23 | 多模态、RAG/Agent — 工程应用 |

### 🔵 路径 E：其他行业转行 AI

> 适合：非计算机/非数学背景，从其他行业转行，需要"够用、能上手、能面试"

| 阶段 | 章节 | 重点 |
|------|------|------|
| 建立直觉 | s01 | 先搞清楚 AI 是什么，不用深究公式 |
| 核心三件套 | s02-s04 | 线性回归、逻辑回归、过拟合 — ML 面试必问 |
| 必备经典 | ml01(k-NN)、ml04(SVM)、ml05(决策树) | 三道菜就能应付大部分 ML 面试问题 |
| DL 入门 | s05-s09 | 计算图→反向传播→Adam — 一个 epoch 跑通就行 |
| 项目为王 | s10(CNN)、s16(Transformer)、s18(LLM) | 简历上的硬通货：图像分类、文本生成 |
| 面试加分 | s23(RAG)、s25(安全) | 展示你对工程落地的理解 |
| 可选补课 | 附录 algo01-algo16 | 算法弱的话，优先看 algo02(数组/哈希)、algo10(分治/二分)、algo11(DP) |

> 核心原则：**先跑通再理解，先项目再理论，先面试再深入**。不必死磕每一章的数学推导。

### 🟣 路径 F：面试冲刺（15 章核心）

> 适合：已学过，快速复习高频考点，备战 ML/AI 面试

**核心必看**：s02/s03（线性/逻辑回归）→ s04（过拟合/正则化）→ ml04（SVM）→ ml05（决策树）→ s06/s07（反向传播）→ s09（Adam）→ s10（CNN）→ s16（Transformer）→ s17（BERT/GPT）→ s18（LLM/RLHF）→ s21（RLHF）→ s23（RAG）→ s25（AI 安全）

> 番外篇和附录按面试岗位选择性翻阅：ML 岗重点看 ml06/ml07（集成学习）；算法岗重点看附录；LLM 岗重点看 s18/s21/s23。

### 🔴 路径 G：算法竞赛（数据结构与算法）

> 适合：ACM/蓝桥杯/LeetCode 选手，系统刷算法

直接进入附录，按 algo01 → algo16 顺序学习。每节包含概念图解 + 手写实现 + 练习题。其余 AI 章节按需查阅。

---

### 阶段一：机器学习基石（4 篇）

| 编号 | 标题 | 核心内容 | 代码实操 |
|------|------|----------|----------|
| [s01](s01_ai_overview/) | AI 全景图 | AI/ML/DL 关系、三大范式、发展简史 | NumPy 手写感知机 |
| [s02](s02_linear_regression/) | 线性回归 | 模型-损失-优化三要素、梯度下降 | 从零实现 + 正规方程 + sklearn 对比 |
| [s03](s03_logistic_regression/) | 逻辑回归与分类 | Sigmoid、交叉熵、Softmax 多分类 | 手写二分类 + 多分类器 |
| [s04](s04_bias_variance/) | 过拟合与正则化 | Bias-Variance 权衡、L1/L2、交叉验证 | 多项式拟合 + 正则化路径 + K-Fold |

### 阶段二：经典机器学习（5 篇）

| 编号 | 标题 | 核心内容 | 代码实操 |
|------|------|----------|----------|
| [ml01](ml01_knn/) | k-近邻与距离度量 | k-NN、距离度量、维数灾难 | KNN 分类器 + 决策边界可视化 |
| [ml02](ml02_bayesian_decision/) | 贝叶斯决策理论 | 先验/后验、最小风险决策、ROC/AUC | 高斯贝叶斯分类器 |
| [ml03](ml03_naive_bayes/) | 朴素贝叶斯 | 条件独立性、高斯/多项式 NB、拉普拉斯平滑 | 垃圾邮件检测 + 文本分类 |
| [ml04](ml04_svm/) | 支持向量机 (SVM) | 最大间隔、对偶问题、核技巧、软间隔 | 线性 SVM + RBF 核可视化 |
| [ml05](ml05_decision_tree/) | 决策树 | ID3/C4.5/CART、信息增益、剪枝 | CART 决策树 + 决策面可视化 |

### 番外：经典机器学习进阶（9 篇，可选）

| 编号 | 标题 | 核心内容 | 代码实操 |
|------|------|----------|----------|
| [ml06](ml06_random_forest/) | 集成学习：Bagging 与随机森林 | Bootstrap、OOB、特征子空间 | 随机森林 + 特征重要性 |
| [ml07](ml07_boosting/) | 集成学习：Boosting 与 Stacking | AdaBoost、GBDT、XGBoost | AdaBoost + 简单 GBDT |
| [ml08](ml08_clustering/) | 聚类 | K-Means、DBSCAN、层次聚类 | 聚类算法对比可视化 |
| [ml09](ml09_dimensionality_reduction/) | 降维与特征工程 | PCA、t-SNE、LDA | PCA 手写 + t-SNE 可视化 |
| [ml10](ml10_monte_carlo/) | 蒙特卡洛方法 | MC 积分、重要性采样、MCMC | π 估计 + Metropolis-Hastings |
| [ml11](ml11_hmm/) | 隐马尔可夫模型 (HMM) | 前向算法、Viterbi、Baum-Welch | HMM + 词性标注 |
| [ml12](ml12_em_gmm/) | EM 算法与高斯混合模型 | E-Step/M-Step、GMM、ELBO | GMM 聚类 + 软分配可视化 |
| [ml13](ml13_probabilistic_graphical_models/) | 概率图模型 | 贝叶斯网、d-分离、信念传播 | 变量消除 + 因子图推理 |
| [ml14](ml14_kernel_gp/) | 核方法与高斯过程 | 核技巧、KRR、GP 回归 | GP 回归 + 不确定性可视化 |

### 阶段三：深度学习基础（5 篇）

| 编号 | 标题 | 核心内容 | 代码实操 |
|------|------|----------|----------|
| [s05](s05_forward_computation_graph/) | 计算图与前向传播 | 计算图、感知机、激活函数深度解析 | 纯 NumPy 搭建 MLP |
| [s06](s06_backprop_chain_rule/) | 反向传播与链式法则 | 局部梯度规则、链式法则、fan-out | 从零实现 mini autograd 引擎 |
| [s07](s07_matrix_backprop/) | 多层网络的矩阵反传 | δ 递推公式、梯度检查、消失/爆炸 | 手写 MLP + 梯度检查 |
| [s08](s08_optimizers_sgd_to_adam/) | 优化器：从 SGD 到 Adam | Momentum、RMSProp、自适应步长 | 四种优化器轨迹对比 |
| [s09](s09_adam_deep_dive/) | Adam 深度解析 | 偏差修正、AdamW、梯度裁剪、诊断 | MNIST 训练 + 优化器对比 |

### 阶段四：计算机视觉（5 篇）

| 编号 | 标题 | 核心内容 | 代码实操 |
|------|------|----------|----------|
| [s10](s10_cnn_fundamentals/) | CNN 核心原理 | 卷积、池化、感受野、参数共享 | 从零实现 Conv2d + 特征图 |
| [s11](s11_cnn_architectures/) | 经典架构演进 | LeNet → ResNet → EfficientNet | 从零写 ResNet 训练 CIFAR-10 |
| [s12](s12_object_detection/) | 目标检测 | R-CNN → YOLO、IoU、NMS、mAP | 从零实现 IoU + NMS |
| [s12b](s11b_vit/) | Vision Transformer | Patch Embedding、位置编码 | ViT vs CNN 对比实验 |
| [s13](s13_image_generation/) | 图像生成 | GAN、VAE、扩散模型原理 | 训练 GAN + VAE 生成 MNIST |

### 阶段五：自然语言处理（5 篇）

| 编号 | 标题 | 核心内容 | 代码实操 |
|------|------|----------|----------|
| [s14](s14_text_representation/) | 文本表示 | 词袋→TF-IDF→word2vec | 训练 Skip-gram + 词向量可视化 |
| [s15](s15_sequence_models/) | 序列模型 | RNN、LSTM、GRU 门控机制 | 字符级语言模型 + 情感分类 |
| [s16](s16_attention_transformer/) | Attention & Transformer | Q/K/V、多头注意力、位置编码 | 从零实现 nanoGPT |
| [s17](s17_pretrained_models/) | 预训练范式 | BERT vs GPT、MLM vs CLM | BERT 微调 + 掩码预测 |
| [s18](s18_large_language_models/) | 大语言模型 | Scaling Law、涌现、RLHF/DPO | LoRA 微调 + DPO 对齐 |

### 阶段六：强化学习（3 篇）

| 编号 | 标题 | 核心内容 | 代码实操 |
|------|------|----------|----------|
| [s19](s19_rl_qlearning/) | 强化学习入门 | MDP、Q 表、ε-greedy、Bellman | Q-Learning 走迷宫 |
| [s20](s20_deep_rl/) | 深度强化学习 | DQN、经验回放、REINFORCE | DQN 玩 CartPole |
| [s21](s21_rlhf/) | RLHF | PPO、DPO、Reward Model | PPO + DPO 对比训练 |

### 阶段七：前沿与应用（4 篇）

| 编号 | 标题 | 核心内容 | 代码实操 |
|------|------|----------|----------|
| [s22](s22_multimodal/) | 多模态模型 | CLIP、对比学习、LLaVA 架构 | CLIP 零样本分类 |
| [s23](s23_rag_agent/) | RAG 与 AI Agent | 检索增强、ReAct、工具调用 | 完整 RAG + Agent 系统 |
| [s24](s24_deployment_inference/) | 部署与推理优化 | KV Cache、量化、Flash Attention | KV Cache + INT8 量化 |
| [s25](s25_ai_safety/) | AI 安全与对齐 | 幻觉、越狱、偏见、深度防御 | 安全扫描 + 幻觉检测 |

### 附录：算法与数据结构基础（16 节，默认折叠）

| 编号 | 标题 | 核心内容 | 代码实操 |
|------|------|----------|----------|
| [algo01](algo01_complexity/) | 复杂度分析 | 大 O/Ω/Θ、主定理、均摊分析 | 排序算法复杂度对比 |
| [algo02](algo02_arrays_linkedlist_hash/) | 数组链表哈希表 | 动态数组、链表、哈希表、LRU | 各数据结构手写实现 |
| [algo03](algo03_stack_queue/) | 栈与队列 | 单调栈/队列、表达式求值 | 表达式求值 + 滑动窗口 |
| [algo04](algo04_tree_binarytree/) | 树与二叉树 | 遍历、BST、AVL 旋转、哈夫曼 | BST + AVL 手写实现 |
| [algo05](algo05_heap_unionfind_skiplist/) | 堆并查集跳跃表 | 堆排序、路径压缩、SkipList | 堆 + 并查集 + Kruskal |
| [algo06](algo06_graph_basics/) | 图论基础 | 存储方式、BFS/DFS、拓扑排序 | 图的遍历 + 拓扑排序 |
| [algo07](algo07_shortest_path/) | 最短路径 | Dijkstra、Bellman-Ford、Floyd、A* | 最短路算法全实现 |
| [algo08](algo08_mst_networkflow/) | MST 与网络流 | Prim/Kruskal、Dinic、二分图匹配 | MST + 最大流 |
| [algo09](algo09_greedy/) | 贪心算法 | 活动选择、哈夫曼编码、区间调度 | 贪心 + 正确性证明 |
| [algo10](algo10_divide_conquer/) | 递归分治与二分 | 归并排序、快速排序、二分查找 | 分治算法 + 逆序对 |
| [algo11](algo11_dp_1/) | 动态规划（上） | 背包、LCS、LIS、编辑距离 | DP 经典问题全实现 |
| [algo12](algo12_dp_2/) | 动态规划（下） | 区间/树形/状压/数位 DP | 高级 DP + 优化 |
| [algo13](algo13_string/) | 字符串算法 | KMP、Trie、AC 自动机、Manacher | 字符串匹配全家桶 |
| [algo14](algo14_segment_tree/) | 线段树与树状数组 | BIT、线段树、懒标记、可持久化 | 线段树 + BIT 手写 |
| [algo15](algo15_number_theory/) | 数论与组合数学 | 快速幂、筛法、CRT、组合数 | 数论算法 + 取模运算 |
| [algo16](algo16_geometry_game/) | 计算几何与博弈论 | 叉积、凸包、Nim、SG 函数 | Graham Scan + Nim 求解 |

---

## 快速开始

### 环境配置

```bash
# 1. 克隆仓库
git clone https://github.com/DeconBear/learn-ai.git
cd learn-ai

# 2. 创建虚拟环境（推荐 Python 3.10+）
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# 3. 安装依赖
pip install -r requirements.txt
```

### 运行代码

```bash
# 每章都是独立可运行的——直接跑
cd s01_ai_overview/code
python demo.py       # 完整教学代码
python exercise.py   # 动手练习（需先补全 TODO）

# 更多章节
cd s08_optimizers_sgd_to_adam/code && python demo.py
cd s16_attention_transformer/code && python demo.py
```

### 硬件要求

| 阶段 | 最低配置 | 推荐配置 |
|------|----------|----------|
| 阶段一~三（s01-s09, ml01-ml14） | CPU，任意笔记本 | — |
| 阶段四~六（s10-s21） | CPU（已优化训练量） | GPU 4GB+ |
| 阶段七（s22-s25） | CPU | GPU 8GB+ |
| 附录（algo01-algo16） | CPU，任意笔记本 | — |

> **所有代码默认 CPU 运行**，自动检测 GPU。有 GPU 的章节在 CPU 下会自动减少训练量，保证快速出结果。

### 学习建议

1. **选择路径**：参考上面「学习路径推荐」，选最适合你背景的路线
2. **先图后文再代码**：图解建直觉 → 文字理解原理 → 跑代码验证 → 做练习巩固
3. **完成 exercise.py**：每章练习留有空缺，先独立尝试再对照 demo.py
4. **动手实验**：改超参数、换数据集、加噪音——比死记硬背有用得多
5. **番外篇按需学**：集成学习、聚类、HMM、概率图等属于进阶内容，可跳过或按需查阅

---

## 文档站点

本仓库附带 **VitePress 文档站点**，支持全文搜索、深色模式、Mermaid 流程图和 LaTeX 公式渲染。

```bash
# 启动文档站点
npm install
npm run dev        # http://localhost:5173

# 构建静态站点
npm run build      # 输出到 .vitepress/dist/
npm run preview    # 预览构建结果
```

站点功能：
- 侧边栏导航（七阶段 + 番外 + 附录，共 55 篇）
- 全文搜索
- LaTeX 公式渲染
- 深色/浅色模式
- 每章代码在线查看 + 下载

---

## 每章结构

```
sXX_topic/
├── index.md                # 图解正文（在线浏览）
├── CODE.md                 # 代码说明与运行报告
├── code-demo.md            # demo.py 查看页面（含说明 + 完整代码）
├── code-exercise.md        # exercise.py 查看页面
├── code/
│   ├── demo.py             # 完整教学代码（中文注释）
│   └── exercise.py         # 动手练习
└── images/                 # 手绘图解 + 代码运行结果
    ├── XX-01-xxx.png       # 概念图解
    ├── XX-02-xxx.png       # 概念图解
    └── ...                 # 运行生成的图表
```

每章配套：
- **正文**（index.md）：图文讲解，100+ 张手绘级插图
- **代码报告**（CODE.md）：嵌入实际运行结果的图片 + 解读说明
- **在线查看**（code-demo.md / code-exercise.md）：语法高亮代码 + 下载

---

## 贡献

欢迎贡献！你可以：

- 提交 Issue 指出错误或改进建议
- 提交 PR 修正错误、改进代码注释
- 贡献新文章（认同「图解 + 代码」风格即可）

---

## 致谢

- [learn-claude-code](https://github.com/shareAI-lab/learn-claude-code) — 仓库结构理念
- [3Blue1Brown](https://www.3blue1brown.com/) — 「先直觉，后公式」的教学哲学
- [nanoGPT](https://github.com/karpathy/nanoGPT) — 从零实现的教学思路
- [Distill.pub](https://distill.pub/) — 图解学术文章先驱

---

## License

MIT License — 自由使用、修改、分发。
