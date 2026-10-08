import fs from 'node:fs'
import path from 'node:path'
import { DOCS_DIR, REPO_ROOT, walkDocs, loadGone } from './lib/docs-tree.mjs'

function copyFile(src, dest) {
  fs.mkdirSync(path.dirname(dest), { recursive: true })
  fs.copyFileSync(src, dest)
}

function collectPy(dir, relBase, out) {
  if (!fs.existsSync(dir)) return
  for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
    const abs = path.join(dir, ent.name)
    const rel = relBase ? `${relBase}/${ent.name}` : ent.name
    if (ent.isDirectory()) {
      if (ent.name === 'code') {
        for (const f of fs.readdirSync(abs, { withFileTypes: true })) {
          if (f.isFile() && (f.name.endsWith('.py') || f.name.endsWith('.cpp') || f.name.endsWith('.hpp') || f.name.endsWith('.h') || f.name.endsWith('.cc'))) {
            out.push({
              abs: path.join(abs, f.name),
              relPath: `${relBase}/${f.name}`,
              chapterRel: relBase,
            })
          }
        }
      } else if (!ent.name.startsWith('.') && ent.name !== 'node_modules') {
        collectPy(abs, rel, out)
      }
    }
  }
}

const files = []
collectPy(DOCS_DIR, '', files)

const publicCode = path.join(REPO_ROOT, 'public', 'code')
fs.mkdirSync(publicCode, { recursive: true })

const { chapters } = walkDocs(DOCS_DIR)
const legacyDirsOf = new Map()
for (const ch of chapters) {
  const dirs = []
  for (const legacy of ch.legacyPaths || []) {
    const slug = String(legacy).replace(/^\//, '').replace(/\/$/, '')
    if (slug) dirs.push(slug)
  }
  if (dirs.length) legacyDirsOf.set(ch.rel, dirs)
}

let copied = 0
for (const f of files) {
  const dest = path.join(publicCode, f.relPath)
  copyFile(f.abs, dest)
  copied++
  const name = path.basename(f.abs)
  for (const slug of legacyDirsOf.get(f.chapterRel) || []) {
    copyFile(f.abs, path.join(publicCode, slug, name))
    copied++
  }
}

// GitHub Pages 无法通过 .py.html 重定向原始 .py 下载请求。
// 为已迁移且不再属于 legacyPaths 的下载 URL 复制当前源文件。
const sourceByDownload = new Map(files.map((f) => [`/code/${f.relPath}`, f.abs]))
for (const { from, to } of loadGone(DOCS_DIR)) {
  if (typeof from !== 'string' || typeof to !== 'string') continue
  if (!from.startsWith('/code/') || !to.startsWith('/code/')) continue
  const src = sourceByDownload.get(to)
  if (!src) throw new Error(`下载别名目标没有对应源文件: ${from} -> ${to}`)
  const dest = path.resolve(publicCode, from.slice('/code/'.length))
  const relative = path.relative(publicCode, dest)
  if (relative === '..' || relative.startsWith('..' + path.sep) || path.isAbsolute(relative)) {
    throw new Error(`下载别名超出 public/code: ${from}`)
  }
  copyFile(src, dest)
  copied++
}

console.log(`sync-public-code: wrote ${copied} files under public/code/`)
