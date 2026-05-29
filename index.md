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
    details: 25 篇文章覆盖 ML 基石 → 深度学习 → CV → NLP → 强化学习 → 前沿应用，六大阶段循序渐进
  - icon: 🇨🇳
    title: 中文原创
    details: 全部内容用中文撰写，术语保留英文对照，适合中文读者系统学习 AI
  - icon: 🔓
    title: 完全开源
    details: MIT 协议，自由使用、修改、分发。欢迎贡献和纠错
---

## 📖 学习路线

```mermaid
flowchart TB
    subgraph S1["阶段一：机器学习基石"]
        s01["s01 AI 全景图"] --> s02["s02 线性回归"]
        s02 --> s03["s03 逻辑回归"]
        s03 --> s04["s04 过拟合与正则化"]
    end

    subgraph S2["阶段二：深度学习基础"]
        s04 --> s05["s05 计算图与前向传播"]
        s05 --> s06["s06 反向传播与链式法则"]
        s06 --> s07["s07 多层网络矩阵反传"]
        s07 --> s08["s08 优化器：SGD→Adam"]
        s08 --> s09["s09 Adam 深度解析"]
    end

    s09 --> s10_cv["s10 CNN 原理"]
    s09 --> s14_nlp["s14 文本表示"]
    s09 --> s19_rl["s19 Q-Learning"]

    subgraph S3["阶段三：计算机视觉"]
        s10_cv --> s11_cv["s11 经典架构"]
        s11_cv --> s12_cv["s12 目标检测"]
        s11_cv --> s13_cv["s13 图像生成"]
    end

    subgraph S4["阶段四：自然语言处理"]
        s14_nlp --> s15_nlp["s15 序列模型"]
        s15_nlp --> s16_nlp["s16 Attention"]
        s16_nlp --> s17_nlp["s17 预训练范式"]
        s17_nlp --> s18_nlp["s18 大语言模型"]
    end

    subgraph S5["阶段五：强化学习"]
        s19_rl --> s20_rl["s20 深度RL"]
        s20_rl --> s21_rl["s21 RLHF"]
    end

    subgraph S6["阶段六：前沿与应用"]
        s18_nlp --> s22["s22 多模态"]
        s13_cv --> s22
        s18_nlp --> s23["s23 RAG与Agent"]
        s18_nlp --> s24["s24 部署优化"]
        s18_nlp --> s25["s25 AI安全"]
        s21_rl --> s25
    end
```

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
├── README.md              # 图解正文（核心阅读材料）
├── image_prompts.md       # 生图提示词
├── code/
│   ├── demo.py            # 完整教学代码（中文注释）
│   └── exercise.py        # 动手练习
└── images/                # 手绘图解
```

## 🙏 致谢

受以下优秀项目启发：

- [learn-claude-code](https://github.com/shareAI-lab/learn-claude-code) — 仓库结构理念
- [3Blue1Brown](https://www.3blue1brown.com/) — 先直觉后公式的教学哲学
- [Distill.pub](https://distill.pub/) — 图解学术文章先驱
- [Andrej Karpathy](https://github.com/karpathy) — 从零实现的教学思路
