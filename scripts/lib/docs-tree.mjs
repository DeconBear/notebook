import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const here = path.dirname(fileURLToPath(import.meta.url))
export const REPO_ROOT = path.resolve(here, '../..')
export const DOCS_DIR = path.join(REPO_ROOT, 'docs')

const SKIP_DIRS = new Set(['code', 'images', 'samples', 'node_modules', '.git', 'public'])
const EXTRA_PAGES = ['code-demo', 'code-exercise', 'nanogpt']

export function unquote(value) {
  const v = value.trim()
  if ((v.startsWith('"') && v.endsWith('"')) || (v.startsWith("'") && v.endsWith("'"))) {
    return v.slice(1, -1)
  }
  return v
}

export function parseSimpleYaml(text) {
  const data = {}
  let currentKey = null
  for (const rawLine of String(text).split(/\r?\n/)) {
    const line = rawLine.replace(/\t/g, '  ')
    const trimmed = line.replace(/#.*$/, '')
    if (!trimmed.trim()) continue
    const listItem = trimmed.match(/^\s*-\s+(.*)$/)
    if (listItem && currentKey) {
      if (!Array.isArray(data[currentKey])) data[currentKey] = []
      const inline = unquote(listItem[1])
      const pair = inline.match(/^from:\s*(\S+)\s+to:\s*(\S+)\s*$/)
      if (pair) {
        data[currentKey].push({ from: unquote(pair[1]), to: unquote(pair[2]) })
      } else {
        data[currentKey].push(inline)
      }
      continue
    }
    const kv = trimmed.match(/^([A-Za-z_][\w]*)\s*:\s*(.*)$/)
    if (!kv) continue
    const key = kv[1]
    const val = kv[2]
    currentKey = key
    if (val === '' || val === '|' || val === '>') {
      data[key] = []
      continue
    }
    if (val === 'true' || val === 'false') {
      data[key] = val === 'true'
      continue
    }
    if (/^-?\d+(\.\d+)?$/.test(val)) {
      data[key] = Number(val)
      continue
    }
    data[key] = unquote(val)
  }
  return data
}

export function parseFrontmatter(content) {
  const text = String(content)
  if (!text.startsWith('---')) return { data: {}, body: text }
  const rest = text.slice(3)
  const end = rest.search(/\r?\n---[ \t]*\r?\n/)
  if (end === -1) return { data: {}, body: text }
  const matter = rest.slice(0, end).replace(/^\r?\n/, '')
  const after = rest.slice(end).replace(/^\r?\n---[ \t]*\r?\n/, '')
  return { data: parseSimpleYaml(matter), body: after }
}

function byOrderThenName(a, b) {
  const ao = a.order ?? 999
  const bo = b.order ?? 999
  if (ao !== bo) return ao - bo
  return String(a.rel).localeCompare(String(b.rel))
}

function listSubdirs(dir) {
  if (!fs.existsSync(dir)) return []
  return fs.readdirSync(dir, { withFileTypes: true })
    .filter((d) => d.isDirectory() && !SKIP_DIRS.has(d.name) && !d.name.startsWith('.'))
    .map((d) => d.name)
}

function extraPagesOf(absDir) {
  return EXTRA_PAGES.filter((name) => fs.existsSync(path.join(absDir, `${name}.md`)))
}

function walkNode(docsDir, rel) {
  const abs = path.join(docsDir, rel)
  const metaPath = path.join(abs, '_meta.yaml')
  const indexPath = path.join(abs, 'index.md')
  const hasMeta = fs.existsSync(metaPath)
  const hasIndex = fs.existsSync(indexPath)
  const childNames = listSubdirs(abs)
  const children = []
  for (const name of childNames) {
    const childRel = rel ? `${rel}/${name}` : name
    const node = walkNode(docsDir, childRel)
    if (node) children.push(node)
  }
  children.sort(byOrderThenName)

  if (hasMeta) {
    const meta = parseSimpleYaml(fs.readFileSync(metaPath, 'utf8'))
    return {
      kind: 'group',
      rel,
      title: meta.title || path.basename(rel),
      order: meta.order ?? 999,
      collapsed: Boolean(meta.collapsed),
      children,
    }
  }

  if (hasIndex) {
    const { data } = parseFrontmatter(fs.readFileSync(indexPath, 'utf8'))
    const legacyPaths = Array.isArray(data.legacyPaths)
      ? data.legacyPaths
      : data.legacyPaths
        ? [data.legacyPaths]
        : []
    return {
      kind: 'chapter',
      rel,
      title: data.title || path.basename(rel),
      order: data.order ?? 999,
      legacyPaths,
      extraPages: extraPagesOf(abs),
      hasCode: fs.existsSync(path.join(abs, 'code')),
      absDir: abs,
    }
  }

  if (children.length) {
    return {
      kind: 'group',
      rel,
      title: path.basename(rel),
      order: 999,
      collapsed: false,
      children,
    }
  }
  return null
}

export function toSidebarItems(nodes) {
  return nodes.map((node) => {
    if (node.kind === 'group') {
      return {
        text: node.title,
        collapsed: node.collapsed,
        items: toSidebarItems(node.children || []),
      }
    }
    return {
      text: node.title,
      link: `/${node.rel}/`,
    }
  })
}

export function flattenChapters(nodes, acc = []) {
  for (const node of nodes) {
    if (node.kind === 'chapter') acc.push(node)
    if (node.children) flattenChapters(node.children, acc)
  }
  return acc
}

export function loadGone(docsDir = DOCS_DIR) {
  const gonePath = path.join(docsDir, '_gone.yaml')
  if (!fs.existsSync(gonePath)) return []
  const raw = parseSimpleYaml(fs.readFileSync(gonePath, 'utf8'))
  // Support either a top-level list under `redirects:` or repeated from/to pairs.
  if (Array.isArray(raw.redirects)) return raw.redirects
  return []
}

export function walkDocs(docsDir = DOCS_DIR) {
  if (!fs.existsSync(docsDir)) {
    return { sidebar: [], chapters: [], tree: [] }
  }
  const tree = []
  for (const name of listSubdirs(docsDir)) {
    const node = walkNode(docsDir, name)
    if (node) tree.push(node)
  }
  tree.sort(byOrderThenName)
  return {
    tree,
    sidebar: toSidebarItems(tree),
    chapters: flattenChapters(tree),
  }
}

export function findCodeFiles(docsDir = DOCS_DIR) {
  const files = []
  function walk(dir, rel) {
    if (!fs.existsSync(dir)) return
    for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
      if (ent.name.startsWith('.')) continue
      const abs = path.join(dir, ent.name)
      const childRel = rel ? `${rel}/${ent.name}` : ent.name
      if (ent.isDirectory()) {
        if (ent.name === 'code') {
          for (const f of fs.readdirSync(abs, { withFileTypes: true })) {
            if (f.isFile() && f.name.endsWith('.py')) {
              files.push({
                abs: path.join(abs, f.name),
                rel: `${childRel}/${f.name}`,
                chapterRel: rel,
              })
            }
          }
        } else if (!SKIP_DIRS.has(ent.name) || ent.name === 'code') {
          walk(abs, childRel)
        } else {
          walk(abs, childRel)
        }
      }
    }
  }
  walk(docsDir, '')
  return files
}

function redirectTargetFile(outDir, fromPath) {
  const trimmed = fromPath.replace(/^\//, '').replace(/\/$/, '')
  const parts = trimmed.split('/').filter(Boolean)
  const last = parts[parts.length - 1]
  if (EXTRA_PAGES.includes(last)) {
    return path.join(outDir, ...parts.slice(0, -1), `${last}.html`)
  }
  return path.join(outDir, ...parts, 'index.html')
}

function redirectHtml(destUrl) {
  return `<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="0; url=${destUrl}">
  <link rel="canonical" href="${destUrl}">
  <title>Moved</title>
  <script>location.replace(${JSON.stringify(destUrl)})</script>
</head>
<body>
  <p>页面已移动到 <a href="${destUrl}">${destUrl}</a></p>
</body>
</html>
`
}

export function writeRedirects({ outDir, base, chapters, gone = [] }) {
  const prefix = base.endsWith('/') ? base.slice(0, -1) : base
  const jobs = []

  for (const ch of chapters) {
    const dest = `${prefix}/${ch.rel}/`.replace(/([^:]\/)\/+/g, '$1')
    for (const legacy of ch.legacyPaths || []) {
      jobs.push({ from: legacy, to: dest })
      for (const extra of ch.extraPages || []) {
        const from = `${String(legacy).replace(/\/$/, '')}/${extra}`
        jobs.push({ from, to: `${prefix}/${ch.rel}/${extra}` })
      }
    }
    for (const legacy of ch.legacyPaths || []) {
      const slug = String(legacy).replace(/^\//, '').replace(/\/$/, '')
      if (!slug || slug.includes('/')) continue
      if (!ch.hasCode) continue
      const codeDir = path.join(ch.absDir, 'code')
      if (!fs.existsSync(codeDir)) continue
      for (const f of fs.readdirSync(codeDir)) {
        if (!f.endsWith('.py')) continue
        jobs.push({
          from: `/code/${slug}/${f}`,
          to: `${prefix}/code/${ch.rel}/${f}`,
        })
      }
    }
  }

  for (const g of gone) {
    if (g.from && g.to) jobs.push({ from: g.from, to: g.to.startsWith('http') ? g.to : `${prefix}${g.to.startsWith('/') ? '' : '/'}${g.to}` })
  }

  for (const job of jobs) {
    const destUrl = job.to
    const isPy = job.from.endsWith('.py')
    if (isPy) {
      // HTML fallback if a raw file is requested at the old download URL.
      const trimmed = job.from.replace(/^\//, '')
      const target = path.join(outDir, trimmed + '.html')
      fs.mkdirSync(path.dirname(target), { recursive: true })
      fs.writeFileSync(target, redirectHtml(destUrl), 'utf8')
      continue
    }
    const target = redirectTargetFile(outDir, job.from)
    fs.mkdirSync(path.dirname(target), { recursive: true })
    fs.writeFileSync(target, redirectHtml(destUrl), 'utf8')
  }
  return jobs.length
}
