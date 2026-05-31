---
title: "algo13 字符串算法 — demo.py"
---

# algo13 字符串算法 — demo.py 代码详解

<a href="../code/algo13_string/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd algo13_string/code
python demo.py
```

## 代码逐段详解

### 第1步：KMP next 数组的计算

next 数组是 KMP 的灵魂。它记录了模式串每个前缀的"最长相等前后缀"信息。

```python
def compute_next(pattern):
    m = len(pattern)
    next_arr = [-1] * m
    k = -1  # k = 已匹配前缀的末尾索引
    for i in range(1, m):
        while k >= 0 and pattern[k + 1] != pattern[i]:
            k = next_arr[k]  # 关键：利用已计算的 next 信息回退
        if pattern[k + 1] == pattern[i]:
            k += 1
        next_arr[i] = k
    return next_arr
```

**next 数组的物理含义**：`next[i]` = 模式串 `P[0..i]` 的最长相等前后缀长度 - 1。例如 `P="ABABCABAB"` 中 `next[8]=3`，意味着前缀 "ABABCABAB" 的最长相等前后缀是 "ABAB"（长度为 4，但 next 存的是 3 = 长度-1，这是代码的 convention）。

**为什么 `k = next_arr[k]`？** 当 `P[k+1] != P[i]` 时，我们不能简单地重置 k=-1，因为可能有一个更短的相等前后缀可用。`next_arr[k]` 正好指向了那个"次长的相等前后缀"的末尾。

### 第2步：KMP 匹配过程

```python
for i in range(n):
    while k >= 0 and pattern[k + 1] != text[i]:
        k = next_arr[k]
    if pattern[k + 1] == text[i]:
        k += 1
    if k == m - 1:
        matches.append(i - m + 1)
        k = next_arr[k]  # 继续找
```

**文本指针 i 永不回溯**——这是 KMP 性能的关键。当发生失配时，通过 next 数组"跳跃"模式串指针 k，文本串指针 i 继续前进。这保证了整个匹配过程的比较总次数为 $O(n)$（每个字符最多被比较常数次）。

### 第3步：Trie 前缀树

```python
class TrieNode:
    __slots__ = ('children', 'is_end', 'word')
    def __init__(self):
        self.children = {}    # 字符 → 子节点
        self.is_end = False   # 单词结束标记
        self.word = None      # 完整单词（叶子节点）
```

**自动补全的核心**：先沿着前缀走到对应节点，然后对该节点的所有后代进行 DFS，收集所有标记为 `is_end` 的单词。

### 第4步：AC 自动机

AC 自动机 = Trie + 失败指针，是 KMP 在多模式匹配上的自然推广。

**失败指针的构建（BFS）**：
- 根的直接子节点 → fail 指向根
- 对节点 `u` 的字符 `c` 子节点 `v`，沿 `u.fail` 链找第一个有 `c` 出边的节点

**匹配过程**：
1. 从根出发，按文本字符在 Trie 上转移
2. 如果当前节点没有对应字符的出边，沿 fail 指针回退
3. 每一步到达一个节点后，沿 fail 链收集所有 output

### 第5步：Manacher 算法

```python
T = '#' + '#'.join(s) + '#'  # 插入 '#' 分隔符
R = [0] * n  # R[i] = 以 i 为中心的回文半径
```

**为什么插入 `#`？** 原始字符串中，"aba"（奇长度，以 b 为中心）和 "abba"（偶长度，以两 b 间隙为中心）的回文中心不同。插入 `#` 后，所有回文都统一为奇长度，中心都在字符位置上。

**`R[i]` 的含义**：以 T[i] 为中心（不含 T[i] 自身），向左右扩展的最大回文半径。原始串中对应的回文子串起点为 `(center - R[center]) // 2`。

### 第6步：滚动哈希

```python
class RollingHash:
    def get_hash(self, l, r):
        h = (self.prefix[r+1] - self.prefix[l] * self.power[r-l+1]) % self.mod
        return h if h >= 0 else h + self.mod
```

哈希公式：$\text{hash}(S[l..r]) = (\text{prefix}[r+1] - \text{prefix}[l] \cdot B^{r-l+1}) \bmod M$

原理类似于十进制数的提取：要从 12345 中提取子串 34，计算 $12345 - 12 \times 10^2 = 345$，再除以 $10^1$...等等。滚动哈希通过前缀哈希表实现了 $O(1)$ 子串比较。

## 关键概念速查表

| 算法 | 核心数据结构 | 复杂度 | 代码位置 |
|------|------------|--------|---------|
| KMP | next 数组 | $O(n+m)$ | `kmp_search()` |
| Trie | 多叉前缀树 | 插入 $O(\|S\|)$ | `Trie.insert()` |
| AC 自动机 | Trie + fail 指针 | 构建 $O(\sum\|P_i\|)$ | `AhoCorasick.build_failure_links()` |
| Manacher | R 数组 + 镜像对称 | $O(n)$ | `manacher()` |
| 滚动哈希 | 前缀哈希 + 次幂表 | $O(1)$ 子串哈希 | `RollingHash.get_hash()` |

## 完整代码

<<< @/snippets/algo13_string/demo.py
