---
title: "algo05 堆、并查集与跳跃表 — exercise.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# algo05 堆、并查集与跳跃表 — exercise.py 练习指南

<a href="/notebook/code/algorithms/basics/heap-unionfind/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过四个任务掌握堆、并查集和跳跃表的核心操作。

## 任务清单

### 任务1：实现最大堆

最大堆与最小堆的唯一区别是 `_sift_up` 和 `_sift_down` 中的比较方向反转。`push` 时大于父节点就交换，`pop` 时与较大的子节点交换。

### 任务2：补全并查集

**find 路径压缩**：`if self.parent[x] != x: self.parent[x] = self.find(self.parent[x])`

**union 按秩合并**：总是将秩（树高上界）较小的根挂到秩较大的根下。秩相等时，选一个作为新的根，秩+1。

### 任务3：连通分量计数

将每条边的两端点 union，所有操作结束后，`set_count` 就是连通分量数量。

### 任务4（Bonus）：跳跃表 search 和 insert

`search`：从最高层向下搜索，每层尽可能前进直到遇到大于等于目标的值。
`insert`：记录每层前驱，随机生成层数后插入。

## 验证

```bash
cd docs/algorithms/basics/heap-unionfind/code
python exercise.py
```


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/algorithms/basics/heap-unionfind/code/exercise.py`
