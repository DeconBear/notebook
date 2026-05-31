---
title: "algo12 动态规划（下）— demo.py"
---

# algo12 动态规划（下）— demo.py 代码详解

<a href="../code/algo12_dp_2/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd algo12_dp_2/code
python demo.py
```

## 代码逐段详解

### 第1步：矩阵链乘法 — 区间 DP 的经典

矩阵链乘法的核心是**括号化**：不同的乘法顺序对应不同的计算代价。如 $A_1(30 \times 35) \times A_2(35 \times 15) \times A_3(15 \times 5)$：
- 方案 (A1A2)A3：$30 \times 35 \times 15 + 30 \times 15 \times 5 = 15750 + 2250 = 18000$
- 方案 A1(A2A3)：$35 \times 15 \times 5 + 30 \times 35 \times 5 = 2625 + 5250 = 7875$

**DP 实现**：

```python
dp[i][j] = min_{i≤k<j} (dp[i][k] + dp[k+1][j] + p[i] * p[k+1] * p[j+1])
```

三个组成部分：
1. `dp[i][k]`：计算左半括号 `(A_i ... A_k)` 的最少乘法次数
2. `dp[k+1][j]`：计算右半括号 `(A_{k+1} ... A_j)` 的最少乘法次数
3. `p[i]*p[k+1]*p[j+1]`：将两个子结果（维度 $p_i \times p_{k+1}$ 和 $p_{k+1} \times p_{j+1}$）相乘的代价

**填充顺序至关重要**：按区间长度 `length` 递增遍历。因为大区间依赖于小区间，必须确保小区间先被计算。

**括号化回溯**：`split[i][j]` 记录最优分割点，递归构建括号表达式。

### 第2步：石子合并 — 带前缀和的区间 DP

与矩阵链乘法类似，但多了**前缀和优化**：

```python
prefix = [0] * (n + 1)
for i in range(1, n + 1):
    prefix[i] = prefix[i - 1] + stones[i - 1]
```

有了前缀和，区间和可以在 $O(1)$ 内计算：`range_sum(i, j) = prefix[j+1] - prefix[i]`。

转移方程：$dp[i][j] = \min_k(dp[i][k] + dp[k+1][j]) + sum(stones[i..j])$

### 第3步：树形 DP — 后序遍历 + 状态聚合

#### 最大独立集

```python
dp[u][0] = 0  # 不选 u
dp[u][1] = 1  # 选 u（自身算 1 个）
for v in adj[u]:
    dfs(v)
    dp[u][1] += dp[v][0]    # 选了 u → v 不能选
    dp[u][0] += max(dp[v][0], dp[v][1])  # 不选 u → v 自由选择
```

**关键点**：DFS 后序遍历保证了子节点的 dp 值在父节点被处理前已经计算完毕。

#### 树的直径（双 DFS）

方法比树形 DP 更简洁：从任意点出发找到最远点 A，再从 A 找最远点 B，A-B 即为直径。正确性基于树的无环性——任何一条最长路径的两个端点，从其中一个出发 DFS 必然能找到另一个。

### 第4步：状态压缩 DP — TSP

```python
dp[mask][i] = 当前已访问城市集合 mask，最后在城市 i 的最短距离
for mask in range(1 << n):
    for i in range(n):
        if mask & (1 << i):  # i 在 mask 中
            for j in range(n):
                if not (mask & (1 << j)):  # j 不在 mask 中
                    new_mask = mask | (1 << j)
                    dp[new_mask][j] = min(dp[new_mask][j],
                                          dp[mask][i] + dist[i][j])
```

**复杂度**：$O(n^2 \cdot 2^n)$。对 $n \leq 16$ 可在秒级完成，$n=20$ 约需数分钟。

**空间优化**：如果不需要回溯路径，可以只保留 `dp[mask][i]`，去掉 `parent` 数组。

### 第5步：数位 DP — 数字和统计

`digit_dp_sum_of_digits(N)` 使用组合数学方法而非 DP 记忆化搜索——对每一位，统计该位上每个数字 0-9 的出现次数。

**核心公式**：
- 考虑第 pos 位（从高位开始 0-indexed）
- 设当前位数字为 `d_cur`，高位部分为 `prefix`，低位部分为 `suffix`
- 对于数字 `d`：
  - 若 `d < d_cur`：计数 = `(prefix + 1) * 10^(剩余位数)`
  - 若 `d == d_cur`：计数 = `prefix * 10^(剩余位数) + suffix + 1`
  - 若 `d > d_cur`：计数 = `prefix * 10^(剩余位数)`

### 第6步：数位 DP 的记忆化搜索版本

`digit_dp_count_no_digit_4(N)` 展示数位 DP 的另一种实现——DFS + 记忆化搜索（使用 `functools.lru_cache`）：

```python
@lru_cache(maxsize=None)
def dfs(pos, tight, has_leading_zero):
    if pos == n: return 1 if not has_leading_zero else 0
    limit = digits[pos] if tight else 9
    count = 0
    for d in range(limit + 1):
        if d == 4: continue
        count += dfs(pos+1, tight and d==limit, ...)
    return count
```

## 关键概念速查表

| 概念 | 状态/转移 | 复杂度 |
|------|----------|--------|
| 矩阵链乘法 | $dp[i][j] = \min_k(dp[i][k]+dp[k+1][j]+p_i p_k p_j)$ | $O(n^3)$ |
| 石子合并 | 同上 + 前缀和 | $O(n^3)$ |
| 树最大独立集 | dp[u][0/1], 后序遍历 | $O(n)$ |
| 树直径 | 双 DFS/BFS | $O(n)$ |
| TSP (Held-Karp) | $dp[mask][i]$, $O(n^2 2^n)$ | $O(n^2 2^n)$ |
| 数位 DP | DFS + 记忆化, tight 约束 | O(位数 * 状态数) |

## 完整代码

<<< @/snippets/algo12_dp_2/demo.py
