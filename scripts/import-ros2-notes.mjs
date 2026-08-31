#!/usr/bin/env node
/** Import DeconBear/ros2-humble-notes into docs/ros2 + workspaces/ros2-humble. */
import fs from 'node:fs'
import path from 'node:path'
import { REPO_ROOT, DOCS_DIR } from './lib/docs-tree.mjs'

const SRC = process.argv[2]
if (!SRC || !fs.existsSync(path.join(SRC, 'docs'))) {
  console.error('用法: node scripts/import-ros2-notes.mjs <cloned-ros2-humble-notes>')
  process.exit(1)
}

const LESSONS = [
  { file: '00-环境与工作区.md', slug: 'env', title: '00 环境与工作区', order: 10 },
  { file: '01-Topic发布订阅.md', slug: 'topics', title: '01 Topic 发布订阅', order: 20 },
  { file: '02-Service请求应答.md', slug: 'services', title: '02 Service 请求应答', order: 30 },
  { file: '03-Action长任务.md', slug: 'actions', title: '03 Action 长任务', order: 40 },
  { file: '04-Parameter参数.md', slug: 'parameters', title: '04 Parameter 参数', order: 50 },
  { file: '05-Launch一键启动.md', slug: 'launch', title: '05 Launch 一键启动', order: 60 },
  { file: '06-自定义消息.md', slug: 'custom-msg', title: '06 自定义消息', order: 70 },
  { file: '07-自定义服务.md', slug: 'custom-srv', title: '07 自定义服务', order: 80 },
  { file: '08-TF2坐标变换.md', slug: 'tf2', title: '08 TF2 坐标变换', order: 90 },
  { file: '09-速度控制cmd_vel.md', slug: 'cmd-vel', title: '09 速度控制 cmd_vel', order: 100 },
  { file: '10-URDF与机器人描述.md', slug: 'urdf', title: '10 URDF 与机器人描述', order: 110 },
  { file: '11-Gazebo最小仿真.md', slug: 'gazebo', title: '11 Gazebo 最小仿真', order: 120 },
  { file: '12-RViz与组合Launch.md', slug: 'rviz-launch', title: '12 RViz 与组合 Launch', order: 130 },
]

const FILE_TO_URL = Object.fromEntries(LESSONS.map((l) => [l.file, `/ros2/${l.slug}/`]))
FILE_TO_URL['README.md'] = '/ros2/overview/'

function rewriteBody(text) {
  let out = text
  const files = Object.keys(FILE_TO_URL).sort((a, b) => b.length - a.length)
  for (const file of files) {
    const url = FILE_TO_URL[file]
    out = out.replaceAll(`](${file})`, `](${url})`)
  }
  out = out.replaceAll('](../src/', '](../../../workspaces/ros2-humble/src/')
  out = out.replaceAll('$REPO', '$ROS_WS')
  out = out.replaceAll('cd /path/to/ros2-humble-notes', 'cd workspaces/ros2-humble')
  return out
}

function writeChapter(destRel, title, order, body) {
  const dir = path.join(DOCS_DIR, destRel)
  fs.mkdirSync(dir, { recursive: true })
  const fm = [
    '---',
    `title: ${JSON.stringify(title)}`,
    `order: ${order}`,
    '---',
    '',
  ].join('\n')
  fs.writeFileSync(path.join(dir, 'index.md'), fm + rewriteBody(body).replace(/^\uFEFF/, ''), 'utf8')
}

function copyDir(from, to) {
  fs.mkdirSync(to, { recursive: true })
  fs.cpSync(from, to, { recursive: true, filter: (src) => !src.includes(`${path.sep}.git`) })
}

const ws = path.join(REPO_ROOT, 'workspaces', 'ros2-humble')
fs.mkdirSync(ws, { recursive: true })
copyDir(path.join(SRC, 'src'), path.join(ws, 'src'))
fs.mkdirSync(path.join(ws, 'scripts'), { recursive: true })
fs.copyFileSync(path.join(SRC, 'scripts', 'check_env.sh'), path.join(ws, 'scripts', 'check_env.sh'))
fs.copyFileSync(path.join(SRC, 'TUTOR_PROMPT.md'), path.join(ws, 'TUTOR_PROMPT.md'))
fs.copyFileSync(path.join(SRC, 'progress.template.md'), path.join(ws, 'progress.template.md'))
if (fs.existsSync(path.join(SRC, 'LICENSE'))) {
  fs.copyFileSync(path.join(SRC, 'LICENSE'), path.join(ws, 'LICENSE'))
}

let check = fs.readFileSync(path.join(ws, 'scripts', 'check_env.sh'), 'utf8')
check = check.replace(
  'if [[ -d "$REPO_ROOT/src/py_pubsub" && -d "$REPO_ROOT/docs" ]]; then',
  'if [[ -d "$REPO_ROOT/src/py_pubsub" ]]; then',
)
fs.writeFileSync(path.join(ws, 'scripts', 'check_env.sh'), check, 'utf8')

fs.writeFileSync(
  path.join(ws, 'README.md'),
  `# ROS 2 Humble 工作区

本目录是 notebook 里 ROS 2 课程的 **colcon 工作区**（源自 [ros2-humble-notes](https://github.com/DeconBear/ros2-humble-notes)）。笔记在站点 \`/ros2/\`，源码在本目录 \`src/\`。

\`$ROS_WS\` 就是本文件夹（相对仓库根：\`workspaces/ros2-humble\`）。

\`\`\`bash
cd workspaces/ros2-humble
export ROS_WS="$PWD"
export PATH="/usr/bin:$PATH"
bash scripts/check_env.sh
source /opt/ros/humble/setup.bash
colcon build --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source "$ROS_WS/install/setup.bash"
\`\`\`

许可证：Apache-2.0，见 [LICENSE](LICENSE)。
`,
  'utf8',
)

const ros2Docs = path.join(DOCS_DIR, 'ros2')
fs.mkdirSync(ros2Docs, { recursive: true })
fs.writeFileSync(
  path.join(ros2Docs, '_meta.yaml'),
  'title: "ROS 2"\norder: 85\ncollapsed: false\n',
  'utf8',
)

const overview = fs.readFileSync(path.join(SRC, 'docs', 'README.md'), 'utf8')
writeChapter('ros2/overview', 'ROS 2 导读', 1, overview)

for (const lesson of LESSONS) {
  const body = fs.readFileSync(path.join(SRC, 'docs', lesson.file), 'utf8')
  writeChapter(`ros2/${lesson.slug}`, lesson.title, lesson.order, body)
}

const tutor = fs.readFileSync(path.join(SRC, 'TUTOR_PROMPT.md'), 'utf8')
  .replaceAll('`docs/README.md`', '`docs/ros2/overview/`')
  .replaceAll('`docs/00-环境与工作区.md`', '`docs/ros2/env/`')
  .replaceAll('`scripts/check_env.sh`', '`workspaces/ros2-humble/scripts/check_env.sh`')
  .replaceAll('代码以 `src/` 为准', '代码以 `workspaces/ros2-humble/src/` 为准')
  .replaceAll('仓库根目录的 `AGENTS.md`', '仓库根目录的 `AGENTS.md`（其中 ROS 2 一节）')
fs.writeFileSync(path.join(ws, 'TUTOR_PROMPT.md'), tutor, 'utf8')

console.log('imported ROS 2 notes -> docs/ros2 + workspaces/ros2-humble')
