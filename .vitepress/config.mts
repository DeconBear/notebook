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
        text: '阶段二：经典机器学习',
        collapsed: false,
        items: [
          { text: 'ml01 k-近邻与距离度量', link: '/ml01_knn/' },
          { text: 'ml02 贝叶斯决策理论', link: '/ml02_bayesian_decision/' },
          { text: 'ml03 朴素贝叶斯与贝叶斯网络', link: '/ml03_naive_bayes/' },
          { text: 'ml04 支持向量机 (SVM)', link: '/ml04_svm/' },
          { text: 'ml05 决策树', link: '/ml05_decision_tree/' },
        ]
      },
      {
        text: '番外：经典机器学习进阶',
        collapsed: true,
        items: [
          { text: 'ml06 集成学习：Bagging与随机森林', link: '/ml06_random_forest/' },
          { text: 'ml07 集成学习：Boosting与Stacking', link: '/ml07_boosting/' },
          { text: 'ml08 聚类：无监督学习的核心', link: '/ml08_clustering/' },
          { text: 'ml09 降维与特征工程', link: '/ml09_dimensionality_reduction/' },
          { text: 'ml10 蒙特卡洛方法', link: '/ml10_monte_carlo/' },
          { text: 'ml11 隐马尔可夫模型 (HMM)', link: '/ml11_hmm/' },
          { text: 'ml12 EM算法与高斯混合模型', link: '/ml12_em_gmm/' },
          { text: 'ml13 概率图模型基础', link: '/ml13_probabilistic_graphical_models/' },
          { text: 'ml14 核方法与高斯过程', link: '/ml14_kernel_gp/' },
        ]
      },
      {
        text: '阶段三：深度学习基础',
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
        text: '阶段四：计算机视觉',
        collapsed: false,
        items: [
          { text: 's10 CNN 核心原理', link: '/s10_cnn_fundamentals/' },
          { text: 's11 经典架构演进', link: '/s11_cnn_architectures/' },
          { text: 's12 目标检测', link: '/s12_object_detection/' },
          { text: 's12b Vision Transformer', link: '/s11b_vit/' },
          { text: 's13 图像生成', link: '/s13_image_generation/' },
        ]
      },
      {
        text: '阶段五：自然语言处理',
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
        text: '阶段六：强化学习',
        collapsed: false,
        items: [
          { text: 's19 MDP & Q-Learning', link: '/s19_rl_qlearning/' },
          { text: 's20 深度强化学习', link: '/s20_deep_rl/' },
          { text: 's21 RLHF', link: '/s21_rlhf/' },
        ]
      },
      {
        text: '阶段七：前沿与应用',
        collapsed: false,
        items: [
          { text: 's22 多模态模型', link: '/s22_multimodal/' },
          { text: 's23 RAG 与 Agent', link: '/s23_rag_agent/' },
          { text: 's24 部署与推理优化', link: '/s24_deployment_inference/' },
          { text: 's25 AI 安全与对齐', link: '/s25_ai_safety/' },
        ]
      },
      {
        text: '附录：算法与数据结构基础',
        collapsed: true,
        items: [
          { text: 'algo01 复杂度分析与渐进记号', link: '/algo01_complexity/' },
          { text: 'algo02 数组、链表与哈希表', link: '/algo02_arrays_linkedlist_hash/' },
          { text: 'algo03 栈与队列', link: '/algo03_stack_queue/' },
          { text: 'algo04 树与二叉树', link: '/algo04_tree_binarytree/' },
          { text: 'algo05 堆、并查集与跳跃表', link: '/algo05_heap_unionfind_skiplist/' },
          { text: 'algo06 图论基础', link: '/algo06_graph_basics/' },
          { text: 'algo07 最短路径', link: '/algo07_shortest_path/' },
          { text: 'algo08 最小生成树与网络流', link: '/algo08_mst_networkflow/' },
          { text: 'algo09 贪心算法', link: '/algo09_greedy/' },
          { text: 'algo10 递归、分治与二分', link: '/algo10_divide_conquer/' },
          { text: 'algo11 动态规划（上）', link: '/algo11_dp_1/' },
          { text: 'algo12 动态规划（下）', link: '/algo12_dp_2/' },
          { text: 'algo13 字符串算法', link: '/algo13_string/' },
          { text: 'algo14 线段树与树状数组', link: '/algo14_segment_tree/' },
          { text: 'algo15 数论与组合数学', link: '/algo15_number_theory/' },
          { text: 'algo16 计算几何与博弈论入门', link: '/algo16_geometry_game/' },
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
