#!/usr/bin/env node
/**
 * One-shot: move chapter folders into docs/<domain>/..., add frontmatter/_meta.yaml,
 * rewrite snippet includes, download links, and cross-chapter markdown links.
 */
import fs from 'node:fs'
import path from 'node:path'
import { execSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { CHAPTERS, GROUPS } from './lib/chapter-map.mjs'
import { parseFrontmatter, REPO_ROOT, DOCS_DIR } from './lib/docs-tree.mjs'

const TEXT_EXT = new Set(['.md', '.py', '.mts', '.ts', '.mjs', '.txt', '.yml', '.yaml'])

const byFrom = new Map(CHAPTERS.map((c) => [c.from, c]))
const slugsDesc = [...byFrom.keys()].sort((a, b) => b.length - a.length)

function posixRel(fromDir, toPath) {
  const rel = path.relative(fromDir, toPath).split(path.sep).join('/')
  return rel.startsWith('.') ? rel : `./${rel}`
}

function yamlDump(obj) {
  const lines = []
  for (const [k, v] of Object.entries(obj)) {
    if (v === undefined || v === null) continue
    if (typeof v === 'boolean' || typeof v === 'number') lines.push(`${k}: ${v}`)
    else lines.push(`${k}: ${JSON.stringify(v)}`)
  }
  return lines.join('\n') + '\n'
}

function ensureFrontmatter(indexPath, chapter) {
  const raw = fs.readFileSync(indexPath, 'utf8')
  const { data, body } = parseFrontmatter(raw)
  const merged = {
    ...data,
    title: chapter.title,
    order: chapter.order,
  }
  const lines = ['---']
  lines.push(`title: ${JSON.stringify(merged.title)}`)
  lines.push(`order: ${merged.order}`)
  lines.push('legacyPaths:')
  lines.push(`  - /${chapter.from}/`)
  if (merged.layout) lines.push(`layout: ${merged.layout}`)
  lines.push('---', '')
  const nextBody = raw.startsWith('---') ? body : raw
  fs.writeFileSync(indexPath, `${lines.join('\n')}${nextBody.startsWith('\n') ? nextBody.slice(1) : nextBody}`, 'utf8')
}

function rewriteContent(content, fileAbs) {
  let out = content
  const fileDir = path.dirname(fileAbs)

  for (const from of slugsDesc) {
    const ch = byFrom.get(from)
    const to = ch.to

    out = out.replaceAll(`<<< @/snippets/${from}/`, `<<< @/${to}/code/`)

    out = out.replaceAll(`href="../code/${from}/`, `href="/notebook/code/${to}/`)
    out = out.replaceAll(`href="/code/${from}/`, `href="/notebook/code/${to}/`)

    out = out.replaceAll(`cd ${from}/code`, `cd docs/${to}/code`)
    out = out.replaceAll(`cd ${from}\\code`, `cd docs/${to}/code`)

    const imgRe = new RegExp(`\\]\\(\\.{1,2}/${from}/images/([^)]+)\\)`, 'g')
    out = out.replace(imgRe, (_, imgPath) => {
      const absImg = path.join(DOCS_DIR, to, 'images', imgPath)
      return `](${posixRel(fileDir, absImg)})`
    })

    out = out.replaceAll(`](../${from}/)`, `](/${to}/)`)
    out = out.replaceAll(`](../${from})`, `](/${to}/)`)
    out = out.replaceAll(`](/${from}/)`, `](/${to}/)`)
    out = out.replaceAll(`](/${from})`, `](/${to}/)`)
    out = out.replaceAll(`(${from}/)`, `(docs/${to}/)`)
  }
  return out
}

function walkFiles(dir, acc = []) {
  if (!fs.existsSync(dir)) return acc
  for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
    if (ent.name === 'node_modules' || ent.name === '.git' || ent.name === '.vitepress') continue
    const abs = path.join(dir, ent.name)
    if (ent.isDirectory()) walkFiles(abs, acc)
    else acc.push(abs)
  }
  return acc
}

function moveDir(fromAbs, toAbs) {
  fs.mkdirSync(path.dirname(toAbs), { recursive: true })
  if (fs.existsSync(toAbs)) {
    throw new Error(`target exists: ${toAbs}`)
  }
  const fromRel = path.relative(REPO_ROOT, fromAbs).split(path.sep).join('/')
  const toRel = path.relative(REPO_ROOT, toAbs).split(path.sep).join('/')
  try {
    execSync(`git mv "${fromRel}" "${toRel}"`, { stdio: 'pipe', cwd: REPO_ROOT })
  } catch {
    fs.renameSync(fromAbs, toAbs)
  }
}

function main() {
  fs.mkdirSync(DOCS_DIR, { recursive: true })

  const homeSrc = path.join(REPO_ROOT, 'index.md')
  const homeDest = path.join(DOCS_DIR, 'index.md')
  if (fs.existsSync(homeSrc) && !fs.existsSync(homeDest)) {
    try {
      execSync('git mv "index.md" "docs/index.md"', { stdio: 'pipe', cwd: REPO_ROOT })
    } catch {
      fs.renameSync(homeSrc, homeDest)
    }
  }

  for (const ch of CHAPTERS) {
    const fromAbs = path.join(REPO_ROOT, ch.from)
    const toAbs = path.join(DOCS_DIR, ch.to)
    if (!fs.existsSync(fromAbs)) {
      if (fs.existsSync(toAbs)) {
        console.log(`skip (already moved): ${ch.from}`)
        continue
      }
      console.warn(`missing source: ${ch.from}`)
      continue
    }
    moveDir(fromAbs, toAbs)
    console.log(`moved ${ch.from} -> docs/${ch.to}`)
  }

  for (const g of GROUPS) {
    const dir = path.join(DOCS_DIR, g.rel)
    fs.mkdirSync(dir, { recursive: true })
    fs.writeFileSync(
      path.join(dir, '_meta.yaml'),
      yamlDump({ title: g.title, order: g.order, collapsed: g.collapsed }),
      'utf8',
    )
  }

  fs.writeFileSync(
    path.join(DOCS_DIR, '_gone.yaml'),
    '# 已删除章节的旧 URL。每项: { from: /old-slug/, to: /parent/ }\nredirects: []\n',
    'utf8',
  )

  for (const ch of CHAPTERS) {
    const indexPath = path.join(DOCS_DIR, ch.to, 'index.md')
    if (fs.existsSync(indexPath)) ensureFrontmatter(indexPath, ch)
  }

  const rewriteRoots = [
    DOCS_DIR,
    path.join(REPO_ROOT, 'README.md'),
    path.join(REPO_ROOT, 'AGENTS.md'),
  ]
  const files = []
  for (const root of rewriteRoots) {
    if (!fs.existsSync(root)) continue
    if (fs.statSync(root).isDirectory()) walkFiles(root, files)
    else files.push(root)
  }

  for (const abs of files) {
    if (!TEXT_EXT.has(path.extname(abs))) continue
    const before = fs.readFileSync(abs, 'utf8')
    const after = rewriteContent(before, abs)
    if (after !== before) fs.writeFileSync(abs, after, 'utf8')
  }

  console.log('migration rewrite done')
}

main()
