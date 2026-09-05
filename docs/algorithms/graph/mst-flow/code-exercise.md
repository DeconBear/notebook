---
title: "algo08 最小生成树与网络流 — exercise.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# algo08 最小生成树与网络流 — exercise.py 练习指南

<a href="/notebook/code/algorithms/graph/mst-flow/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过四个任务掌握 MST 和最大流算法的核心实现。

## 任务清单

### 任务1：朴素 Prim (O(V²))

不使用优先队列，使用 `key` 数组存储连接已选集合的最小边权重，每轮扫描找最小值节点。

### 任务2：Kruskal + 并查集

实现带路径压缩的 `find` 和按秩合并的 `union`。边排序后，对每条边——若两端不在同一集合则加入 MST。

### 任务3：Ford-Fulkerson (DFS)

在残量网络中 DFS 寻找增广路径。找到后沿路径推送 bottleneck 流量，并更新残量（正向减少、反向增加）。

### 任务4（Bonus）：Dinic BFS 层图

BFS 从源点出发，对每条残量 > 0 的边标记层号。返回 level 数组和是否还能到达汇点。

## 验证

```bash
cd docs/algorithms/graph/mst-flow/code
python exercise.py
```


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/algorithms/graph/mst-flow/code/exercise.py`
