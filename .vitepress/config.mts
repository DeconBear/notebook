import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

// https://vitepress.dev/reference/site-config
export default withMermaid(
  defineConfig({
  base: '/learn-ai/',
  title: "learn-ai",
  description: "图解 AI · 一行代码看懂一个概念",
  lang: 'zh-CN',
  ignoreDeadLinks: true,
  publicDir: 'public',
  srcExclude: ['README.md', '**/image_prompts.md', '**/CODE.md'],
  head: [
    ['link', { rel: 'icon', href: '/favicon.ico' }]
  ],

  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: '首页', link: '/' },
      { text: 'GitHub', link: 'https://github.com/DeconBear/learn-ai' },
    ],

    sidebar: [
      {
        text: '阶段一：机器学习基石',
        collapsed: false,
        items: [
          { text: 's01 AI 全景图', link: '/s01_ai_overview/' },
          { text: 's02 线性回归', link: '/s02_linear_regression/' },
          { text: 's03 逻辑回归', link: '/s03_logistic_regression/' },
          { text: 's04 过拟合与正则化', link: '/s04_bias_variance/' },
        ]
      },
      {
        text: '阶段二：深度学习基础',
        collapsed: false,
        items: [
          { text: 's05 计算图与前向传播', link: '/s05_forward_computation_graph/' },
          { text: 's06 反向传播与链式法则', link: '/s06_backprop_chain_rule/' },
          { text: 's07 多层网络矩阵反传', link: '/s07_matrix_backprop/' },
          { text: 's08 优化器：SGD→Adam', link: '/s08_optimizers_sgd_to_adam/' },
          { text: 's09 Adam 深度解析', link: '/s09_adam_deep_dive/' },
        ]
      },
      {
        text: '阶段三：计算机视觉',
        collapsed: false,
        items: [
          { text: 's10 CNN 核心原理', link: '/s10_cnn_fundamentals/' },
          { text: 's11 经典架构演进', link: '/s11_cnn_architectures/' },
          { text: 's11b Vision Transformer', link: '/s11b_vit/' },
          { text: 's12 目标检测', link: '/s12_object_detection/' },
          { text: 's13 图像生成', link: '/s13_image_generation/' },
        ]
      },
      {
        text: '阶段四：自然语言处理',
        collapsed: false,
        items: [
          { text: 's14 文本表示', link: '/s14_text_representation/' },
          { text: 's15 序列模型', link: '/s15_sequence_models/' },
          { text: 's16 Attention & Transformer', link: '/s16_attention_transformer/' },
          { text: 's17 预训练范式', link: '/s17_pretrained_models/' },
          { text: 's18 大语言模型', link: '/s18_large_language_models/' },
        ]
      },
      {
        text: '阶段五：强化学习',
        collapsed: false,
        items: [
          { text: 's19 MDP & Q-Learning', link: '/s19_rl_qlearning/' },
          { text: 's20 深度强化学习', link: '/s20_deep_rl/' },
          { text: 's21 RLHF', link: '/s21_rlhf/' },
        ]
      },
      {
        text: '阶段六：前沿与应用',
        collapsed: false,
        items: [
          { text: 's22 多模态模型', link: '/s22_multimodal/' },
          { text: 's23 RAG 与 Agent', link: '/s23_rag_agent/' },
          { text: 's24 部署与推理优化', link: '/s24_deployment_inference/' },
          { text: 's25 AI 安全与对齐', link: '/s25_ai_safety/' },
        ]
      },
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/DeconBear/learn-ai' }
    ],

    search: {
      provider: 'local'
    },

    outline: {
      level: [2, 3],
      label: '本节目录'
    },

    docFooter: {
      prev: '← 上一篇',
      next: '下一篇 →'
    },

    lastUpdated: {
      text: '最后更新'
    },

    darkModeSwitchLabel: '深色模式',
    sidebarMenuLabel: '菜单',
    returnToTopLabel: '回到顶部',
  },

  markdown: {
    math: true,
    lineNumbers: true
  }
})
)
