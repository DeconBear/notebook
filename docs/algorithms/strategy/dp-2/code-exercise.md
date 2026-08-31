---
title: "algo12 动态规划（下）— exercise.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# algo12 动态规划（下）— exercise.py 练习指南

<a href="/notebook/code/algorithms/strategy/dp-2/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过四个进阶 DP 练习题，掌握区间 DP、树形 DP、状态压缩 DP 和数位 DP 的设计模式。

## 预备知识

- 区间 DP 的按长度递增枚举模式
- 树形 DP 的后序遍历 + 状态聚合模式
- 状态压缩 DP 的二进制位运算
- 数位 DP 的 DFS + 记忆化搜索模板

## 任务清单

### 任务1：最长回文子序列 `longest_palindromic_subsequence(s)`

- **状态**：`dp[i][j]` = s[i..j] 的最长回文子序列长度
- **转移**：
  - `s[i] == s[j]`：两端字符相同，可以同时纳入 → `dp[i+1][j-1] + 2`
  - `s[i] != s[j]`：至少一端不能用 → `max(dp[i+1][j], dp[i][j-1])`
- **初始化**：`dp[i][i] = 1`（单个字符是回文）
- **回溯**：从 `dp[0][n-1]` 开始，根据转移来源决定包含哪些字符。

### 任务2：树的最小顶点覆盖 `tree_min_vertex_cover(n, edges)`

- **问题**：选最少的节点使得每条边至少有一个端点被选中。
- **与最大独立集的关系**：在任意图中，最小顶点覆盖 + 最大独立集 = 总节点数。但注意：**等式只在一般图中成立需要二分图条件**。在树中（树是二分图），可直接用 $VC = n - MIS$，也可以独立 DP。
- **独立 DP 状态**：
  - `dp[u][1] = 1 + sum(min(dp[v][0], dp[v][1]))`（选了 u，子节点自由）
  - `dp[u][0] = sum(dp[v][1])`（不选 u，所有子节点必须选——否则边(u,v) 没有端点被覆盖）

### 任务3：状态压缩 DP — 集合划分

- **思路**：枚举所有子集 mask，预计算每个子集的和 `subset_sum[mask]`。
- **DP**：`dp[mask]` = 已安排 mask 中元素后，最小化的最大子集和。
- **转移**：枚举 mask 的子集 sub → `dp[mask] = min(dp[mask], max(dp[mask ^ sub], subset_sum[sub]))`

### 任务4：数位 DP — 各位乘积

- **模板**：定义 `dfs(pos, tight, leading_zero, product)` 返回满足乘积条件且不越界的数的个数。
- **关键问题**：如果 target_product 很大（如 > 81），许多 branch 会直接剪枝（因为 9 位以内数字的各位乘积最大是 $9^9$）。
- **记忆化**：用 `@lru_cache` 或自定义 dict 存储 `(pos, tight, product)`。

## 提示

1. 最长回文子序列注意初始化 `dp[i][i]=1` 和 `dp[i][i-1]=0`（空区间）。
2. 树的最小顶点覆盖中，DFS 遍历时注意避免访问父节点。
3. 状态压缩中子集枚举的高效写法：`sub = (sub - 1) & mask`。

<<< @/algorithms/strategy/dp-2/code/exercise.py
