---
title: "algo06 图论基础 — exercise.py"
---

# algo06 图论基础 — exercise.py 练习指南

<a href="../code/algo06_graph_basics/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过四个任务掌握图的遍历、环检测、拓扑排序和二分性判断。

## 任务清单

### 任务1：BFS 最短路径

在 BFS 队列中携带 `(node, distance)` 元组。遇到 target 时直接返回当前距离。

### 任务2：DFS 环检测（无向图）

DFS 时携带 `parent` 参数。若遇到已访问的邻居且不是 `parent`，则发现环。

### 任务3：Kahn 拓扑排序

1. 构建邻接表 + 计算入度
2. 入度为 0 的顶点入队
3. BFS 出队，将其后继入度减 1，入度为 0 者入队
4. 结果长度 < n 则存在环

### 任务4：二分图检测

BFS 染色：每层交替染 0 和 1。若发现相邻同色则不是二分图。

## 验证

```bash
cd algo06_graph_basics/code
python exercise.py
```

## 完整代码

<<< @/snippets/algo06_graph_basics/exercise.py
