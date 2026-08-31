import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'
import { walkDocs, loadGone, writeRedirects } from '../scripts/lib/docs-tree.mjs'

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const docsDir = path.join(repoRoot, 'docs')
const { sidebar, chapters } = walkDocs(docsDir)
const gone = loadGone(docsDir)

export default withMermaid(
  defineConfig({
    srcDir: 'docs',
    base: '/notebook/',
    title: 'notebook',
    description: '图解笔记 · AI、算法、ROS 2',
    lang: 'zh-CN',
    ignoreDeadLinks: true,
    publicDir: path.join(repoRoot, 'public'),
    srcExclude: ['README.md', '**/image_prompts.md', '**/CODE.md', '**/_gone.yaml', '**/_meta.yaml'],
    head: [
      ['link', { rel: 'icon', href: '/favicon.ico' }]
    ],

    themeConfig: {
      nav: [
        { text: '首页', link: '/' },
        { text: 'GitHub', link: 'https://github.com/DeconBear/notebook' },
      ],
      sidebar,
      socialLinks: [
        { icon: 'github', link: 'https://github.com/DeconBear/notebook' }
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
    },

    buildEnd(site) {
      const n = writeRedirects({
        outDir: site.outDir,
        base: site.site?.base || '/notebook/',
        chapters,
        gone,
      })
      console.log(`redirects: wrote ${n} legacy mappings`)
    },
  })
)
