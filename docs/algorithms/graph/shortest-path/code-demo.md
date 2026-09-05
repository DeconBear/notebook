---
title: "algo07 最短路径 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# algo07 最短路径 — demo.py 代码详解

<a href="/notebook/code/algorithms/graph/shortest-path/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/algorithms/graph/shortest-path/code
python demo.py
```

## 代码结构

| 函数 | 算法 | 复杂度 | 特点 |
|------|------|--------|------|
| `dijkstra()` | Dijkstra堆优化 | O((V+E)log V) | 非负权图 |
| `bellman_ford()` | Bellman-Ford | O(VE) | 负权+负环检测 |
| `spfa()` | SPFA | O(kE) 平均 | 队列优化版BF |
| `floyd_warshall()` | Floyd-Warshall | O(V³) | 全源最短路径 |
| `a_star()` | A* | 启发式 | 带方向搜索 |

## 第1步：Dijkstra 的正确性关键

```python
if d > dist[u]:
    continue  # 过时条目，跳过！
```

这行代码至关重要。因为堆中可能存储了同一个顶点在**不同时刻**的距离。当弹出一个 $(d, u)$ 时，$d$ 可能已经不是 $u$ 当前的最短距离了。

## 第2步：松弛操作的本质

```
if dist[u] + w < dist[v]:
    dist[v] = dist[u] + w
```

松弛是**所有最短路径算法的核心操作**。它检查"绕道"（经过新的顶点）是否能缩短已有路径。Dijkstra 贪心选择下一个要用的源是什么，Bellman-Ford 重复松弛所有边。

## 第3步：Floyd-Warshall 为何 k 在最外层？

```python
for k in range(n):      # 第 k 阶段：允许经过前 k 个顶点
    for i in range(n):
        for j in range(n):
            dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
```

DP 定义：$dp[k][i][j]$ = 只允许经过顶点 $0..k$ 时 $i$ 到 $j$ 的最短距离。$k$ 代表 DP 阶段的推进，必须放在最外层。

## 关键概念速查表

| 算法 | 贪心/DP | 数据结构 | 允许负权 | 检测负环 | 全源 |
|------|---------|----------|----------|----------|------|
| Dijkstra | 贪心 | 优先队列 | 否 | 否 | 否 |
| Bellman-Ford | DP | 边列表 | 是 | 是 | 否 |
| SPFA | 贪婪+队列 | 队列 | 是 | 是 | 否 |
| Floyd-Warshall | DP | 矩阵 | 是 | 否(检测对角) | 是 |
| A* | 启发式贪心 | 优先队列 | 否 | 否 | 否 |


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/algorithms/graph/shortest-path/code/demo.py`
