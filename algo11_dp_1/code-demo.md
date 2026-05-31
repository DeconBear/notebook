---
title: "algo11 动态规划（上）— demo.py"
---

# algo11 动态规划（上）— demo.py 代码详解

<a href="../code/algo11_dp_1/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd algo11_dp_1/code
python demo.py
```

## 代码逐段详解

### 第1步：0-1 背包 — Top-Down 与 Bottom-Up 对比

0-1 背包是最经典的 DP 问题，它完美展示了记忆化递归和迭代制表两种实现方式。

#### Top-Down（记忆化递归）

```python
def knapsack_01_topdown(weights, values, capacity):
    memo = [[-1] * (capacity + 1) for _ in range(n + 1)]
    def solve(i, cap):
        if i == 0 or cap == 0: return 0       # 基线条件
        if memo[i][cap] != -1: return memo[i][cap]  # 查备忘录
        result = solve(i - 1, cap)            # 不选当前物品
        if cap >= weights[i-1]:
            result = max(result, solve(i-1, cap-weights[i-1]) + values[i-1])
        memo[i][cap] = result                 # 存入备忘录
        return result
    return solve(n, capacity)
```

**Top-Down 的执行流程**：从 `solve(n, capacity)` 开始，递归地"往下"分解子问题。每个格子 `(i, cap)` 被计算**恰好一次**（因为 memo 记录），但递归结构反映了问题自然的依赖关系。

#### Bottom-Up（迭代制表，空间优化版）

```python
def knapsack_01_bottomup(weights, values, capacity):
    dp = [0] * (capacity + 1)  # 一维滚动数组
    for i in range(n):
        for j in range(capacity, weights[i] - 1, -1):  # 逆序！
            dp[j] = max(dp[j], dp[j - weights[i]] + values[i])
    return dp[capacity]
```

**为什么 j 必须逆序遍历？** 这是理解 0-1 背包空间优化的关键点：

- `dp[j - w_i]` 必须来自**上一个物品考虑完后的状态**（即第 i-1 行的值）
- 如果正序遍历 j：当计算 `dp[j]` 时，`dp[j - w_i]` 可能已经被当前物品 i 更新过了——这意味着当前物品被使用了多次（退化为完全背包）
- 逆序遍历保证：计算 `dp[j]` 时，`dp[j - w_i]` 还是上一轮的值

**实例**：物品 (w=2, v=3), (w=3, v=4), (w=4, v=5), 容量 8。

| 容量 | 初始 | 物品1后 | 物品2后 | 物品3后 | 物品4后 |
|------|------|--------|--------|--------|--------|
| 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 0 | 0 |
| 2 | 0 | 3 | 3 | 3 | 3 |
| 3 | 0 | 3 | 4 | 4 | 4 |
| 4 | 0 | 3 | 4 | 5 | 5 |
| 5 | 0 | 3 | 7 | 7 | 7 |
| 6 | 0 | 3 | 7 | 8 | 8 |
| 7 | 0 | 3 | 7 | 9 | 9 |
| 8 | 0 | 3 | 7 | 9 | 10 |

最大价值 = 10（选物品 2 和 4）。

### 第2步：LCS — 两串 DP 的经典模板

```python
def lcs(s1, s2):
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1   # 匹配：长度 +1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])  # 跳过一方
    # 回溯...
```

**LCS DP 表的物理含义**：`dp[i][j]` 记录的是 s1 前 i 个字符和 s2 前 j 个字符的 LCS 长度。两条规则对应了两种自然策略：
- 当前字符相同 → 同时纳入（左上角 +1）
- 当前字符不同 → 至少得跳过其中一个（上或左取 max）

### 第3步：LIS — 从 O(n^2) 到 O(n log n)

**O(n^2) DP**：
- `dp[i]` = 以 `arr[i]` 结尾的最长递增子序列长度
- 转移：`dp[i] = max{dp[j] + 1 | j < i, arr[j] < arr[i]}`

**O(n log n) Patience Sorting**：
- 维护 `tails` 数组，`tails[k]` = 长度为 k+1 的递增子序列的最小末尾值
- 对每个新元素 `x`，二分找 `tails` 中第一个 `>= x` 的位置并替换
- `tails` 的长度即为 LIS 长度

**为什么 Patience Sorting 是正确的？**
- `tails` 数组始终严格递增（证明：替换一个元素只会让它变小，而插入在末尾意味着更大的值）
- 二分搜索保证了每次更新都维护了这个性质
- 最终 `tails` 的长度就是能找到的最长递增链

### 第4步：编辑距离

```python
dp[i][j] = 1 + min(
    dp[i-1][j],      # 删除 s1[i-1]
    dp[i][j-1],      # 插入 s2[j-1]
    dp[i-1][j-1]     # 替换 s1[i-1] → s2[j-1]
)
```

**三种操作在 DP 表中的几何意义**：
- 删除 → 向上移动（减少 s1）
- 插入 → 向左移动（减少 s2）
- 替换 → 向左上移动（同时减少）

**回溯**：从 `dp[m][n]` 出发，根据转移来源逆推出每步操作。

### 第5步：DP 表可视化

代码中的 `visualize_knapsack()` 使用 `matplotlib` 的 `imshow` 绘制 DP 表的热力图，每个格子的数值直接标注。这帮助学生直观理解 DP 表的填充过程——从 (0,0) 到 (n,capacity)，价值逐渐累积。

## 关键概念速查表

| 概念 | 状态/转移 | 代码位置 |
|------|----------|---------|
| 0-1 背包 | dp[j] = max(dp[j], dp[j-w]+v), j 逆序 | `knapsack_01_bottomup()` |
| 完全背包 | dp[j] = max(dp[j], dp[j-w]+v), j 正序 | exercise TODO 1 |
| 子集和 | dp[i][j] = dp[i-1][j] \| dp[i-1][j-nums[i]] | `subset_sum()` |
| LCS | 匹配+1 / max(上,左) | `lcs()` |
| LIS O(n^2) | dp[i] = max(dp[j]+1) | `lis_dp()` |
| LIS O(n log n) | Patience Sorting + 二分 | `lis_patience_sort()` |
| 编辑距离 | min(删,插,替) + 1 | `edit_distance()` |
| 找零 DP | dp[i] = min(dp[i-c]+1) | `coin_change_min()` |

## 完整代码

<<< @/snippets/algo11_dp_1/demo.py
