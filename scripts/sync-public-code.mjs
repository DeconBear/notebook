import fs from 'node:fs'
import path from 'node:path'
import { DOCS_DIR, REPO_ROOT, walkDocs } from './lib/docs-tree.mjs'

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
          if (f.isFile() && f.name.endsWith('.py')) {
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
const legacySlugOf = new Map()
for (const ch of chapters) {
  for (const legacy of ch.legacyPaths || []) {
    const slug = String(legacy).replace(/^\//, '').replace(/\/$/, '')
    if (slug && !slug.includes('/')) legacySlugOf.set(ch.rel, slug)
  }
}

let copied = 0
for (const f of files) {
  const dest = path.join(publicCode, f.relPath)
  copyFile(f.abs, dest)
  copied++
  const slug = legacySlugOf.get(f.chapterRel)
  if (slug) {
    const name = path.basename(f.abs)
    copyFile(f.abs, path.join(publicCode, slug, name))
    copied++
  }
}

console.log(`sync-public-code: wrote ${copied} files under public/code/`)
