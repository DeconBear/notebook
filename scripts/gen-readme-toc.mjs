import fs from 'node:fs'
import path from 'node:path'
import { REPO_ROOT, walkDocs } from './lib/docs-tree.mjs'

const START = '<!-- TOC:start -->'
const END = '<!-- TOC:end -->'

function renderTree(nodes, depth = 0) {
  const lines = []
  for (const node of nodes) {
    if (node.kind === 'group') {
      const heading = '#'.repeat(Math.min(depth + 3, 6))
      lines.push('', `${heading} ${node.title}`, '')
      lines.push(...renderTree(node.children || [], depth + 1))
    } else {
      lines.push(`- [${node.title}](docs/${node.rel}/)`)
    }
  }
  return lines
}

const { tree } = walkDocs()
const toc = [
  START,
  '',
  ...renderTree(tree).filter((line, i, arr) => !(line === '' && arr[i - 1] === '')),
  '',
  END,
].join('\n')

const readmePath = path.join(REPO_ROOT, 'README.md')
let readme = fs.readFileSync(readmePath, 'utf8')
if (!readme.includes(START) || !readme.includes(END)) {
  console.error('README.md 缺少 <!-- TOC:start --> / <!-- TOC:end --> 标记')
  process.exit(1)
}
readme = readme.replace(
  new RegExp(`${START}[\\s\\S]*?${END}`),
  toc,
)
fs.writeFileSync(readmePath, readme, 'utf8')
console.log('README TOC updated')
