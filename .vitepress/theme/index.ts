import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import { h } from 'vue'

// 自定义主题，添加 MathJax 支持
export default {
  extends: DefaultTheme,
  enhanceApp({ app, router, siteData }) {
    // MathJax 会在客户端自动渲染
  },
  Layout: () => {
    // 扩展默认布局，注入 MathJax
    return h(DefaultTheme.Layout, null, {
      // 可以在这里扩展插槽
    })
  }
} satisfies Theme
