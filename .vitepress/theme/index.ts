import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import { h, nextTick, onMounted } from 'vue'
import { inBrowser, onContentUpdated } from 'vitepress'
import mediumZoom from 'medium-zoom'
import './style.css'

export default {
  extends: DefaultTheme,
  enhanceApp() {
    // MathJax 会在客户端自动渲染
  },
  Layout: () => {
    return h(DefaultTheme.Layout)
  },
  setup() {
    let zoom: ReturnType<typeof mediumZoom> | undefined

    const initZoom = () => {
      if (!inBrowser) return
      zoom?.detach()
      const images = Array.from(
        document.querySelectorAll<HTMLImageElement>('.vp-doc img'),
      ).filter((img) => !img.closest('a'))
      if (!images.length) return
      zoom = mediumZoom(images, {
        background: 'var(--vp-c-bg)',
        margin: 24,
        scrollOffset: 0,
      })
    }

    onMounted(() => nextTick(initZoom))
    onContentUpdated(() => nextTick(initZoom))
  },
} satisfies Theme
