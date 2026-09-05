---
title: "algo08 最小生成树与网络流 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# algo08 最小生成树与网络流 — demo.py 代码详解

<a href="/notebook/code/algorithms/graph/mst-flow/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/algorithms/graph/mst-flow/code
python demo.py
```

## 代码结构

| 函数 | 算法 | 复杂度 | 关键数据结构 |
|------|------|--------|-------------|
| `prim()` | Prim MST | O((V+E)log V) | 优先队列 |
| `kruskal()` | Kruskal MST | O(E log E) | 并查集 |
| `Dinic.max_flow()` | Dinic 最大流 | O(V²E) | BFS层图+DFS阻塞流 |
| `bipartite_max_matching()` | 二分图最大匹配 | O(E sqrt(V)) | 转化为最大流 |

## 第1步：Prim 与 Dijkstra 的微妙区别

```python
# Dijkstra: 松弛时累加路径权重
if dist[u] + w < dist[v]:
    dist[v] = dist[u] + w

# Prim: 松弛时只看单条边的权重
heapq.heappush(pq, (weight, u, v))  # 只存边权，不累加
```

Prim 不关心从起点来的累积距离，只关心连接"已选集合"的最小权重边。

## 第2步：Dinic 算法两大核心

**BFS 构建层图**：每个顶点分配层号（从 s 的最短距离），只保留从第 i 层到第 i+1 层的边。

**DFS 找阻塞流**：在层图中找一条饱和流——使每条从 s 到 t 的路径上至少有一条边被耗尽。

```python
# 当前弧优化：避免重复扫描已经饱和的边
self.it[u] = i  # 记录每个顶点当前正在处理的边
```

## 第3步：反向边——网络流最巧妙的机制

```python
def add_edge(self, u, v, cap):
    self.adj[u].append([v, cap, len(self.adj[v])])     # 正向
    self.adj[v].append([u, 0, len(self.adj[u]) - 1])   # 反向（容量 0）
```

当向正向边推送流量时，反向边的容量增加同等数量——这意味着可以"撤销"之前的流量。

## 关键概念速查表

| 概念 | 含义 | 在 MST 中 | 在网络流中 |
|------|------|-----------|-----------|
| 贪心 | 每次选局部最优 | Prim 选最小权重边 | 沿增广路径推送 |
| 图论基础 | - | 割性质、环性质 | 残量网络、增广路径 |
| 关键结构 | - | 优先队列/并查集 | BFS层图/反向边 |
| 复杂度 | - | O(E log E) | O(V²E) (Dinic) |


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/algorithms/graph/mst-flow/code/demo.py`
