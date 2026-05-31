---
layout: home

hero:
  name: "learn-ai"
  text: "图解 AI · 一行代码看懂一个概念"
  tagline: 从感知机到大模型，用图解 + 可运行代码，把 AI 的核心原理一个一个拆给你看
  actions:
    - theme: brand
      text: 开始学习
      link: /s01_ai_overview/
    - theme: alt
      text: GitHub
      link: https://github.com/DeconBear/learn-ai

features:
  - icon: 🎨
    title: 图解优先
    details: 每篇文章配有 4-5 张手绘级示意图，先建直觉，再推公式。复杂概念可视化，一看就懂
  - icon: 💻
    title: 代码实操
    details: 每章附带完整可运行 Python 代码（逐行中文注释）+ 动手练习题，在消费级硬件上就能跑
  - icon: 📐
    title: 数学通透
    details: LaTeX 公式推导，从链式法则到 Attention，每一步都有推导过程，不留黑盒
  - icon: 🗺️
    title: 体系完整
    details: 55 篇文章覆盖 ML 基石 → 经典 ML → 深度学习 → CV → NLP → 强化学习 → 前沿应用 → 算法附录，适合不同学习路径
  - icon: 🇨🇳
    title: 中文原创
    details: 全部内容用中文撰写，术语保留英文对照，适合中文读者系统学习 AI
  - icon: 🔓
    title: 完全开源
    details: MIT 协议，自由使用、修改、分发。欢迎贡献和纠错
---

## 📖 学习路线图

```mermaid
flowchart TB
    subgraph S1["阶段一：机器学习基石（4章）"]
        s01["s01 AI 全景图"] --> s02["s02 线性回归"]
        s02 --> s03["s03 逻辑回归"]
        s03 --> s04["s04 过拟合与正则化"]
    end

    subgraph S2["阶段二：经典机器学习（5章）"]
        s04 --> ml01["ml01 k-NN"]
        ml01 --> ml02["ml02 贝叶斯决策"]
        ml02 --> ml03["ml03 朴素贝叶斯"]
        ml03 --> ml04["ml04 SVM"]
        ml04 --> ml05["ml05 决策树"]
    end

    subgraph S3["阶段三：深度学习基础（5章）"]
        ml05 --> s05["s05 计算图与前向传播"]
        s05 --> s06["s06 反向传播与链式法则"]
        s06 --> s07["s07 多层网络矩阵反传"]
        s07 --> s08["s08 优化器：SGD→Adam"]
        s08 --> s09["s09 Adam 深度解析"]
    end

    s09 --> s10_cv["s10 CNN 原理"]
    s09 --> s14_nlp["s14 文本表示"]
    s09 --> s19_rl["s19 Q-Learning"]

    subgraph S4["阶段四：计算机视觉（5章）"]
        s10_cv --> s11_cv["s11 经典架构"]
        s11_cv --> s12_cv["s12 目标检测"]
        s11_cv --> s12b_vit["s12b ViT"]
        s11_cv --> s13_cv["s13 图像生成"]
    end

    subgraph S5["阶段五：自然语言处理（5章）"]
        s14_nlp --> s15_nlp["s15 序列模型"]
        s15_nlp --> s16_nlp["s16 Transformer"]
        s16_nlp --> s17_nlp["s17 预训练范式"]
        s17_nlp --> s18_nlp["s18 大语言模型"]
    end

    subgraph S6["阶段六：强化学习（3章）"]
        s19_rl --> s20_rl["s20 深度RL"]
        s20_rl --> s21_rl["s21 RLHF"]
    end

    subgraph S7["阶段七：前沿与应用（4章）"]
        s18_nlp --> s22["s22 多模态"]
        s13_cv --> s22
        s18_nlp --> s23["s23 RAG与Agent"]
        s18_nlp --> s24["s24 部署优化"]
        s18_nlp --> s25["s25 AI安全"]
        s21_rl --> s25
    end
```

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

> 💡 **番外篇**（集成学习、聚类、降维、蒙特卡洛、HMM、EM、概率图、高斯过程）在阶段二之后，默认折叠。内容独立、互不依赖，可按需跳读。详细路径说明见 [README](https://github.com/DeconBear/learn-ai#%E5%AD%A6%E4%B9%A0%E8%B7%AF%E5%BE%84%E6%8E%A8%E8%8D%90)。

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

## 🙏 致谢

受以下优秀项目启发：

- [learn-claude-code](https://github.com/shareAI-lab/learn-claude-code) — 仓库结构理念
- [3Blue1Brown](https://www.3blue1brown.com/) — 先直觉后公式的教学哲学
- [Distill.pub](https://distill.pub/) — 图解学术文章先驱
- [Andrej Karpathy](https://github.com/karpathy) — 从零实现的教学思路
