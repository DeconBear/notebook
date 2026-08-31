---
title: "algo11 动态规划（上）— exercise.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# algo11 动态规划（上）— exercise.py 练习指南

<a href="/notebook/code/algorithms/strategy/dp-1/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过四个 DP 变体的实现，巩固 DP 五步设计法：状态定义 → 转移方程 → 初始条件 → 计算顺序 → 提取答案。

## 预备知识

- 0-1 背包的两种实现（Top-Down/Bottom-Up）
- 子集和、LCS、编辑距离的 DP 模板
- 0-1 背包空间优化中 j 逆序的关键原因

## 任务清单

### 任务1：完全背包 `unbounded_knapsack(weights, values, capacity)`

- **与 0-1 背包的唯一区别**：j 的遍历方向从**逆序变为正序**！
- **原因**：正序遍历时，`dp[j - w]` 可能已经被当前物品更新过，相当于允许同一物品被多次使用——这正是完全背包的语义。
- **回溯**：用二维 DP 表 `dp2d[i][j]` 记录完整状态，从 `dp2d[n][capacity]` 回溯每种物品的使用数量。

### 任务2：不同子序列计数 `num_distinct_subsequences(s, t)`

- **状态**：`dp[i][j]` = s[:i] 中 t[:j] 作为子序列出现的不同方式数
- **转移**：
  - `s[i-1] == t[j-1]` → `dp[i][j] = dp[i-1][j-1] + dp[i-1][j]`
    - `dp[i-1][j-1]`：使用 s[i-1] 匹配 t[j-1]
    - `dp[i-1][j]`：不使用 s[i-1] 但仍然匹配 t[:j]
  - 否则 → `dp[i][j] = dp[i-1][j]`（只能跳过 s[i-1]）
- **初始化**：`dp[i][0] = 1`（空串在任何字符串中出现恰好 1 次——什么都不选）

### 任务3：最短公共超序列 `shortest_common_supersequence(s1, s2)`

- **关键公式**：$\text{SCS长度} = \text{len}(s1) + \text{len}(s2) - \text{LCS长度}$
- **构建 SCS 字符串**：在 LCS 回溯的基础上，当字符不匹配时，需要**同时保留两个字符串的字符**（因为 SCS 必须包含两者）。
- **示例**："abac" + "cab" → LCS="ab" → SCS="cabac"

### 任务4：分割等和子集 `can_partition(nums)`

- **归约为子集和问题**：$target = sum(nums) / 2$，能否找到和为 target 的子集？
- **空间优化**：一维布尔 DP 数组，逆向遍历。

## 提示

1. **完全背包的正序遍历**容易理解：想象你有一个无限的物品仓库，每次考虑一个物品时，你可以多次取用，所以 j 从小到大更新。
2. **子序列计数**注意边界：空串的情况。dp[i][0] 总是 1。
3. **SCS 字符串构建**的关键是：每一步不仅要看 LCS 匹配，还要确保所有字符都被包含。

<<< @/algorithms/strategy/dp-1/code/exercise.py
