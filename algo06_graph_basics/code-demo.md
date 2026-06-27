---
title: "algo06 图论基础 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# algo06 图论基础 — demo.py 代码详解

<a href="../code/algo06_graph_basics/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd algo06_graph_basics/code
python demo.py
```

## 代码结构

| 类/函数 | 功能 | 复杂度 |
|---------|------|--------|
| `Graph` | 通用图类（邻接表+邻接矩阵+边列表） | O(V+E) 空间 |
| `bfs()` | 广度优先搜索 | O(V+E) |
| `dfs()` | 深度优先搜索 | O(V+E) |
| `topological_sort_kahn()` | 拓扑排序（入度+BFS） | O(V+E) |
| `topological_sort_dfs()` | 拓扑排序（DFS后序逆序） | O(V+E) |
| `has_cycle_undirected/directed()` | 环检测 | O(V+E) |
| `is_bipartite()` | 二分图检测（染色） | O(V+E) |

## 第1步：图类的设计

```python
class Graph:
    def __init__(self, n_vertices, directed=False):
        self.adj_list = {i: [] for i in range(n_vertices)}  # 邻接表
        self.adj_matrix = [[0]*n for _ in range(n)]          # 邻接矩阵
        self.edge_list = []                                   # 边列表
```

三种表示各有用处：邻接表用于 BFS/DFS 遍历（O(deg) 获取邻居），邻接矩阵用于快速查询边存在性（O(1)），边列表用于 Kruskal 等以边为中心的算法。

## 第2步：BFS 的层序特性

BFS 自然按层扩展——距离起点为 d 的节点在第 d 轮被访问。这使得 BFS 天然适合求解**无权图的最短路径**。

## 第3步：有向图环检测的三色标记

| 颜色 | 含义 | 遇到时的操作 |
|------|------|-------------|
| 白 (0) | 未访问 | 递归探索 |
| 灰 (1) | 正在探索中 | **发现回边→有环!** |
| 黑 (2) | 探索完成 | 跳过 |

## 关键概念速查表

| 问题 | 算法 | 关键数据结构 | 复杂度 |
|------|------|-------------|--------|
| 无权最短路径 | BFS | 队列 | O(V+E) |
| 环检测(无向) | DFS+父节点检查 | 栈(递归) | O(V+E) |
| 环检测(有向) | DFS三色标记 | 颜色数组 | O(V+E) |
| 拓扑排序 | Kahn / DFS后序 | 队列 / 栈 | O(V+E) |
| 二分图检测 | BFS/DFS染色 | 颜色数组 | O(V+E) |
| 连通分量 | BFS/DFS遍历 | visited集合 | O(V+E) |

## 完整代码

<<< @/snippets/algo06_graph_basics/demo.py
