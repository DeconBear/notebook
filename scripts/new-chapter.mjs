#!/usr/bin/env node
import fs from 'node:fs'
import path from 'node:path'
import { DOCS_DIR } from './lib/docs-tree.mjs'

function arg(name, fallback) {
  const idx = process.argv.indexOf(`--${name}`)
  if (idx !== -1 && process.argv[idx + 1]) return process.argv[idx + 1]
  return fallback
}

const positional = process.argv.slice(2).filter((a) => !a.startsWith('--'))
const parent = positional[0]
const slug = positional[1]
const title = arg('title', slug || '新章节')
const order = Number(arg('order', '50'))

if (!parent || !slug) {
  console.error('用法: npm run new-chapter -- <领域路径> <slug> --title "标题" --order 25')
  console.error('示例: npm run new-chapter -- ml/foundations kernel-methods --title "核方法入门" --order 25')
  process.exit(1)
}

if (!/^[a-z0-9][a-z0-9-]*$/.test(slug)) {
  console.error('slug 只能包含小写字母、数字和连字符')
  process.exit(1)
}

const dest = path.join(DOCS_DIR, parent, slug)
if (fs.existsSync(dest)) {
  console.error(`已存在: ${dest}`)
  process.exit(1)
}

fs.mkdirSync(path.join(dest, 'code'), { recursive: true })
fs.mkdirSync(path.join(dest, 'images'), { recursive: true })

const fm = [
  '---',
  `title: ${JSON.stringify(title)}`,
  `order: ${order}`,
  '---',
  '',
  `# ${title}`,
  '',
  '在这里写图解正文。',
  '',
].join('\n')

fs.writeFileSync(path.join(dest, 'index.md'), fm, 'utf8')

fs.writeFileSync(path.join(dest, 'code-demo.md'), `---
title: "${title} — demo.py"
---

# ${title} — demo.py 代码详解

<a href="/notebook/code/${parent}/${slug}/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

\`\`\`bash
cd docs/${parent}/${slug}/code
python demo.py
\`\`\`

## 完整代码

<<< @/${parent}/${slug}/code/demo.py
`, 'utf8')

fs.writeFileSync(path.join(dest, 'code-exercise.md'), `---
title: "${title} — exercise.py"
---

# ${title} — 练习

<a href="/notebook/code/${parent}/${slug}/exercise.py" target="_blank" download>Download exercise.py</a>

## 运行方式

\`\`\`bash
cd docs/${parent}/${slug}/code
python exercise.py
\`\`\`

## 完整代码

<<< @/${parent}/${slug}/code/exercise.py
`, 'utf8')

fs.writeFileSync(path.join(dest, 'code', 'demo.py'), `# -*- coding: utf-8 -*-
"""
=== ${title} ===
运行: python demo.py
"""
print("TODO: 实现 ${title} demo")
`, 'utf8')

fs.writeFileSync(path.join(dest, 'code', 'exercise.py'), `# -*- coding: utf-8 -*-
"""
=== ${title} 练习 ===
运行: python exercise.py
"""
# TODO: 完成练习
print("TODO")
`, 'utf8')

const metaPath = path.join(DOCS_DIR, parent, '_meta.yaml')
if (!fs.existsSync(metaPath)) {
  console.warn(`提示: ${parent} 没有 _meta.yaml，侧栏可能不会把该目录当成分组。`)
}

console.log(`已创建章节: docs/${parent}/${slug}/`)
console.log('保存后运行 npm run dev，侧栏会自动出现。')
