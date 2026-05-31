---
title: "algo09 贪心算法 — exercise.py"
---

# algo09 贪心算法 — exercise.py 练习指南

<a href="../code/algo09_greedy/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过实现三个贪心相关的练习——石子合并、区间覆盖、贪心正确性验证，深入理解贪心算法的设计、实现和正确性分析方法。

## 预备知识

在开始练习前，确保你已经理解了以下概念（参见 [demo.py 代码详解](./code-demo) 中的详细解释）：

- 贪心选择性质：每一步的局部最优选择不会破坏全局最优性
- 最优子结构：问题的最优解包含了子问题的最优解
- `heapq` 模块的基本用法：`heapify()`, `heappush()`, `heappop()`
- 活动选择问题的贪心策略（按结束时间排序）

## 任务清单

### 任务1：石子合并最小代价 `min_merge_cost(arr)`

- **问题描述**：将 $n$ 堆石子合并为一堆，每次可合并任意两堆，代价为两堆之和。求最小总合并代价。
- **贪心策略**：每次选最小的两堆合并（与 Huffman 编码的做法完全一致）。
- **实现步骤**：
  1. 将输入数组转为最小堆：`heapq.heapify(heap)`
  2. 循环直到只剩 1 堆：
     - `a = heapq.heappop(heap)` — 取最小堆
     - `b = heapq.heappop(heap)` — 取次小堆
     - `cost = a + b` — 合并代价
     - `total_cost += cost` — 累加总代价
     - `heapq.heappush(heap, cost)` — 将合并后的新堆放回
- **期望输出**：`arr = [4, 3, 2, 6]` → 总代价 = 29
  - 步骤: 2+3=5 (累计5) → 4+5=9 (累计14) → 6+9=15 (累计29)

### 任务2：区间覆盖 `min_intervals_to_cover(intervals, target_range)`

- **问题描述**：用最少的区间完全覆盖目标区间 $[L, R]$。
- **贪心策略**：在已覆盖的右边界以内，选择能延伸到最远的区间。
- **算法框架**：
  1. 将所有区间按左端点升序排列
  2. 维护 `current_end` = 当前已覆盖到的右边界（初始为 L）
  3. 在 `current_end < R` 时循环：
     - 找到所有左端点 ≤ `current_end` 的区间中，右端点最大的那一个
     - 如果找不到能推进 `current_end` 的区间 → 无法覆盖，返回 -1
     - 否则选择该区间，更新 `current_end`
- **注意**：原 exercise.py 中有一行 `i += 1` 位置有 bug（在 while 循环的每次迭代中重复递增了），请修正为在遍历区间时正确递增。

### 任务3：验证贪心找零正确性 `verify_greedy_coin_system(coins, max_amount)`

- **目标**：判断给定的面额系统下，贪心找零对 $0$ 到 $\text{max\_amount}$ 的所有金额是否都是最优解。
- **实现方法**：
  1. 用 DP 基准算法计算出所有金额的真实最优解（最少硬币数）
  2. 用贪心算法计算每个金额的结果
  3. 比较两者，记录所有不一致的金额
- **DP 基准**：
  ```
  dp[0] = 0
  INF = max_amount + 1 (或 float('inf'))
  for i in range(1, max_amount+1):
      dp[i] = min(dp[i-coin] + 1 for coin in coins if i >= coin)
  ```
- **贪心算法**：每次取最大面额，用 `//` 取整和 `%` 取余。

## 提示

1. **堆操作**：Python 的 `heapq` 默认是最小堆（小顶堆）。`heappop()` 总是返回最小的元素。
2. **区间覆盖**：先排序是关键，排序后才能用贪心高效扫描。
3. **贪心正确性**：失败案例通常出现在小面额上——测试 max_amount=100 基本能覆盖所有常见非规范系统的反例。

<<< @/snippets/algo09_greedy/exercise.py
