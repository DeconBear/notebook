---
title: "algo09 贪心算法 — demo.py"
---

# algo09 贪心算法 — demo.py 代码详解

<a href="../code/algo09_greedy/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd algo09_greedy/code
python demo.py
```

## 代码逐段详解

### 第1步：活动选择问题 — 贪心选择结束最早的活动

这是最经典的贪心算法案例。问题核心：从 $n$ 个有时间冲突的活动中，选出最多互不重叠的活动。

**贪心策略**：每次选择**结束时间最早**且与已选活动不冲突的活动。

```python
def activity_selection(activities):
    n = len(activities)
    indexed = [(activities[i][0], activities[i][1], i) for i in range(n)]
    indexed.sort(key=lambda x: x[1])  # 按结束时间升序
    selected = []
    last_end = -1
    for start, end, idx in indexed:
        if start >= last_end:  # 不冲突 → 可选
            selected.append(idx)
            last_end = end
    return selected
```

**为什么选择结束最早的活动是正确的？**

使用**交换论证法**证明：假设最优解 OPT 没有选第一个结束的活动 $a$，则 OPT 中必定有另一个活动 $b$ 占据了第一个时间段。由于 $a$ 的结束时间 ≤ $b$ 的结束时间，用 $a$ 替换 $b$ 不会引入任何冲突，且保持了相同的活动数量。因此一定存在包含 $a$ 的最优解。

**时间复杂度**：排序 $O(n \log n)$ + 遍历 $O(n)$。

### 第2步：分数背包问题 — 按性价比贪心

分数背包是贪心能正确求解的经典例子。关键在于物品**可分割**。

```python
def fractional_knapsack(items, capacity):
    n = len(items)
    indexed = [(items[i][0], items[i][1],
                items[i][1] / items[i][0], i) for i in range(n)]
    indexed.sort(key=lambda x: x[2], reverse=True)  # 按单位价值降序
    taken = [0.0] * n
    total_value = 0.0
    remaining = capacity
    for w, v, ratio, idx in indexed:
        if remaining <= 0: break
        take = min(w, remaining)
        taken[idx] = take / w
        total_value += take * ratio
        remaining -= take
    return total_value, taken
```

**运行示例**：物品 A (10kg, ¥60, 性价比6)、B (20kg, ¥100, 性价比5)、C (30kg, ¥120, 性价比4)，容量 50kg。

贪心顺序：先装 A (10kg, +¥60)，再装 B (20kg, +¥100)，剩余 20kg 装 C 的 2/3 (20kg, +¥80)。总计 ¥240。

**为什么贪心正确？** 因为物品可分割，如果当前最"划算"的物品没有被全取，那一定是背包已满——此时没有任何其他选择能获得更高的总价值。用数学语言说，分数背包的解空间构成一个拟阵。

### 第3步：Huffman 编码 — 贪心构建最优前缀树

Huffman 编码的核心是用最小堆实现"每次合并最小频率"的贪心策略。

```python
def build_huffman_tree(freq_dict):
    heap = []
    for char, freq in freq_dict.items():
        heapq.heappush(heap, HuffmanNode(char, freq))
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(None, left.freq + right.freq, left, right)
        heapq.heappush(heap, merged)
    return heapq.heappop(heap)
```

**逐步演示**（字符 A:5, B:9, C:12, D:13, E:16, F:45）：

| 步骤 | 取出 | 合并后频率 | 放回后的堆 |
|------|------|-----------|-----------|
| 1 | A(5), B(9) | 14 | {C:12, D:13, E:16, AB:14, F:45} |
| 2 | C(12), D(13) | 25 | {E:16, AB:14, CD:25, F:45} |
| 3 | AB(14), E(16) | 30 | {CD:25, ABE:30, F:45} |
| 4 | CD(25), ABE(30) | 55 | {F:45, CDABE:55} |
| 5 | F(45), CDABE(55) | 100 | {根:100} |

**递归生成编码表**：从根出发，走左分支加 0，走右分支加 1。到达叶子节点时，路径上的 01 串就是该字符的 Huffman 编码。

```python
def generate_huffman_codes(root):
    codes = {}
    def dfs(node, code):
        if node.char is not None:  # 叶子节点
            codes[node.char] = code
            return
        dfs(node.left, code + '0')   # 左分支 → 0
        dfs(node.right, code + '1')  # 右分支 → 1
    dfs(root, '')
    return codes
```

**Huffman 编码的唯一性**：虽然同一频率分布可能有多棵不同的 Huffman 树（交换 0/1 标签或同频率节点的顺序），但**最优的带权路径长度是唯一确定的**。

### 第4步：找零问题 — 贪心 vs 动态规划

这是展示贪心局限性最直观的例子。

```python
def coin_change_dp(coins, amount):
    INF = float('inf')
    dp = [INF] * (amount + 1)
    dp[0] = 0
    coin_used = [-1] * (amount + 1)

    for i in range(1, amount + 1):
        for coin in coins:
            if i >= coin and dp[i - coin] + 1 < dp[i]:
                dp[i] = dp[i - coin] + 1
                coin_used[i] = coin
    # ...回溯重建结果...
```

**DP 状态定义**：`dp[i]` = 凑出金额 $i$ 所需的最少硬币数。

**转移方程**：$dp[i] = \min_{c \in coins}\{dp[i - c] + 1\}$（当 $i \geq c$ 时）。

**贪心 vs DP 对比**：

| 面额系统 | 金额 | 贪心结果 | DP 最优 | 一致? |
|----------|------|---------|---------|-------|
| {1,5,10,20,50,100} | 167 | 100+50+10+5+1+1 = 6枚 | 6枚 | ✓ |
| {1,3,4} | 6 | 4+1+1 = 3枚 | 3+3 = 2枚 | ✗ |
| {1,3,4} | 10 | 4+4+1+1 = 4枚 | 4+3+3 = 3枚 | ✗ |

**关键认识**：贪心算法能否正确，取决于**面额系统本身的性质**，而非算法技巧。多数国家的货币系统是规范性的（canonical），贪心就是最优的。

### 第5步：可视化 — 活动选择甘特图

`plot_activity_selection()` 用水平条形图直观展示活动的时间安排：

- 绿色横条 = 被选中的活动
- 灰色横条 = 被跳过的活动（与已选活动冲突）
- y 轴标签显示活动和起止时间
- 可以直观验证：所有绿色条之间没有重叠

## 关键概念速查表

| 概念 | 要点 | 代码位置 |
|------|------|---------|
| 贪心选择性质 | 局部最优选择包含在全局最优解中 | `activity_selection()` |
| 最优子结构 | 贪心选择后余下的子问题仍可最优求解 | 所有函数 |
| 交换论证 | 将最优解转换为贪心解的证明方法 | 活动选择注释 |
| Huffman 编码 | 每次合并最小频率 → 最优前缀码 | `build_huffman_tree()` |
| 分数背包 | 按性价比降序贪心 → 全局最优 | `fractional_knapsack()` |
| 贪心找零 | 规范面额 → 正确；非规范 → 可能失败 | `coin_change_greedy()` |
| DP 找零 | 任意面额系统都能正确的最少硬币数 | `coin_change_dp()` |

## 完整代码

<<< @/snippets/algo09_greedy/demo.py
