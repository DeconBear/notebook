---
title: "algo07 最短路径 — exercise.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# algo07 最短路径 — exercise.py 练习指南

<a href="/notebook/code/algorithms/graph/shortest-path/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过五个任务实现最短路算法的核心组件。

## 任务清单

### 任务1：朴素 Dijkstra (O(V²))

不用优先队列，用数组扫描找最小 `dist` 节点。适合稠密图（$E \approx V^2$）。

**核心**：每轮在未确定的顶点中找到 `dist` 最小的，标记为已确定，松弛其所有邻居。

### 任务2：负环检测

Bellman-Ford 做 V 轮松弛。第 V 轮仍能松弛 → 存在负环。

技巧：将 `dist` 全部初始化为 0，等价于添加虚拟源点连接到所有顶点（边权为 0）。

### 任务3：Floyd-Warshall

**关键**：`k` 在最外层循环！

### 任务4：路径重建

从 target 出发，沿着 `prev` 数组回溯到起点（prev[start] = None），反转路径。

### 任务5：Dijkstra + 路径（Bonus）

在堆优化 Dijkstra 中维护 `prev` 数组。到达 target 时停止并重建路径。

## 验证

```bash
cd docs/algorithms/graph/shortest-path/code
python exercise.py
```

## 完整代码

<<< @/algorithms/graph/shortest-path/code/exercise.py
