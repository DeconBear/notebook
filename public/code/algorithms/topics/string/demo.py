# -*- coding: utf-8 -*-
"""
algo13 字符串算法 — 演示代码
=============================
功能：KMP 字符串匹配（暴力 vs KMP 对比）、Trie 前缀树（含自动补全）、
      AC 自动机（多模式匹配）、Manacher 最长回文子串、
      滚动哈希（Rabin-Karp 匹配 + O(1) 子串比较）。

每个函数都有中文 docstring，每行逻辑代码都有中文注释。
运行方式：在 algo13_string/ 目录下执行 python code/demo.py
"""

import os
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False

_HERE = os.path.dirname(os.path.abspath(__file__))
_IMAGES = os.path.join(_HERE, '..', 'images')
os.makedirs(_IMAGES, exist_ok=True)


# ============================================================================
# 第一部分：KMP 算法
# ============================================================================

def compute_next(pattern):
    """
    计算 KMP 的 next 数组（failure function）。
    next[i] = pattern[0..i] 的最长相等前后缀的长度 - 1

    例如 pattern="ABABCABAB":
      i=0: A → next=-1 (单字符，无真前后缀)
      i=1: AB → next=-1 (A≠B)
      i=2: ABA → next=0 (A==A)
      i=3: ABAB → next=1 (AB==AB)
      ...
    时间复杂度: O(m)，文本串指针从不回溯
    """
    m = len(pattern)
    next_arr = [-1] * m
    k = -1  # k = 已匹配前缀的末尾索引
    for i in range(1, m):
        # 不匹配时，利用 next 回溯 k
        while k >= 0 and pattern[k + 1] != pattern[i]:
            k = next_arr[k]  # 核心：用已计算的 next 信息加速
        # 匹配时，k 前进
        if pattern[k + 1] == pattern[i]:
            k += 1
        next_arr[i] = k
    return next_arr


def kmp_search(text, pattern):
    """
    KMP 字符串匹配算法。
    利用 next 数组避免文本指针回溯，时间复杂度 O(n+m)。

    返回所有匹配位置的起始索引。
    """
    n, m = len(text), len(pattern)
    if m == 0:
        return [0]  # 空模式串匹配任意位置

    next_arr = compute_next(pattern)
    matches = []
    k = -1  # 已匹配的模式串长度 - 1

    for i in range(n):
        # 不匹配：回溯 k（利用 next 跳跃）
        while k >= 0 and pattern[k + 1] != text[i]:
            k = next_arr[k]
        # 匹配：k 前进
        if pattern[k + 1] == text[i]:
            k += 1
        # 完整匹配：记录位置，并回溯 k 找下一个匹配
        if k == m - 1:
            matches.append(i - m + 1)
            k = next_arr[k]

    return matches


def brute_force_search(text, pattern):
    """暴力字符串匹配，用于与 KMP 对比。O(n*m)。"""
    n, m = len(text), len(pattern)
    matches = []
    for i in range(n - m + 1):
        if text[i:i + m] == pattern:
            matches.append(i)
    return matches


# ============================================================================
# 第二部分：Trie 前缀树
# ============================================================================

class TrieNode:
    """Trie 树的节点。"""
    __slots__ = ('children', 'is_end', 'word')
    def __init__(self):
        self.children = {}      # 字符 → 子节点
        self.is_end = False     # 是否是一个完整单词的结尾
        self.word = None        # 如果 is_end=True，存储完整单词


class Trie:
    """Trie 前缀树，支持插入、搜索、前缀搜索、自动补全。"""
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        """插入一个单词到 Trie 中。O(|word|)。"""
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end = True
        node.word = word

    def search(self, word):
        """精确搜索：word 是否在 Trie 中。"""
        node = self.root
        for ch in word:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return node.is_end

    def starts_with(self, prefix):
        """检查是否有单词以 prefix 开头。"""
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return True

    def autocomplete(self, prefix, limit=10):
        """
        自动补全：返回所有以 prefix 为前缀的单词。
        先走到 prefix 的末端节点，然后 DFS 收集所有单词。
        """
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return []
            node = node.children[ch]

        results = []
        # DFS 收集所有子节点中的完整单词
        def dfs(n):
            if n.is_end:
                results.append(n.word)
            for ch, child in n.children.items():
                dfs(child)

        dfs(node)
        return results[:limit]


# ============================================================================
# 第三部分：AC 自动机（简化版，用于演示概念）
# ============================================================================

class ACNode:
    """AC 自动机的节点。"""
    __slots__ = ('children', 'fail', 'output')
    def __init__(self):
        self.children = {}   # 字符 → 子节点
        self.fail = None     # 失败指针
        self.output = []     # 以该节点结尾的模式串列表


class AhoCorasick:
    """
    AC 自动机（Aho-Corasick Automaton）。
    支持多模式串的批量匹配。
    """
    def __init__(self):
        self.root = ACNode()

    def add_pattern(self, pattern):
        """向 Trie 中添加一个模式串。"""
        node = self.root
        for ch in pattern:
            if ch not in node.children:
                node.children[ch] = ACNode()
            node = node.children[ch]
        node.output.append(pattern)

    def build_failure_links(self):
        """BFS 构建所有节点的失败指针。"""
        from collections import deque
        queue = deque()

        # 根节点的直系子节点的 fail 指向根
        for ch, child in self.root.children.items():
            child.fail = self.root
            queue.append(child)

        # BFS 构建 fail 指针
        while queue:
            current = queue.popleft()
            for ch, child in current.children.items():
                # 沿 fail 链寻找第一个也有字符 ch 出边的节点
                fail_node = current.fail
                while fail_node is not None and ch not in fail_node.children:
                    fail_node = fail_node.fail
                # 设置 child 的 fail 指针
                child.fail = (fail_node.children[ch]
                              if fail_node and ch in fail_node.children
                              else self.root)
                # 合并 output：子节点也包含父节点 fail 链上的输出
                child.output.extend(child.fail.output)
                queue.append(child)

    def search(self, text):
        """在文本中搜索所有模式串的出现位置。返回 {pattern: [positions]}。"""
        from collections import defaultdict
        results = defaultdict(list)
        node = self.root

        for i, ch in enumerate(text):
            # 沿 fail 链回退直到匹配
            while node is not self.root and ch not in node.children:
                node = node.fail
            # 前进
            if ch in node.children:
                node = node.children[ch]
            # 收集当前节点的所有输出
            for pattern in node.output:
                results[pattern].append(i - len(pattern) + 1)

        return dict(results)


# ============================================================================
# 第四部分：Manacher 算法
# ============================================================================

def manacher(s):
    """
    Manacher 算法：O(n) 求最长回文子串。
    核心技巧：
    1. 插入 '#' 分隔符统一处理奇偶长度
    2. 利用回文的镜像对称性减少计算
    """
    if not s:
        return '', 0

    # 预处理：在字符间和两端插入 '#'
    T = '#' + '#'.join(s) + '#'
    n = len(T)
    R = [0] * n  # R[i] = 以 i 为中心的回文半径（不含中心）
    center = 0   # 当前最右回文边界的中心
    right = 0    # 当前最右回文边界

    for i in range(1, n):
        # 利用镜像对称性：i 关于 center 的对称点 mirror
        mirror = 2 * center - i
        if i < right:
            R[i] = min(right - i, R[mirror])

        # 尝试扩展回文
        while (i - R[i] - 1 >= 0 and i + R[i] + 1 < n and
               T[i - R[i] - 1] == T[i + R[i] + 1]):
            R[i] += 1

        # 更新最右边界
        if i + R[i] > right:
            center = i
            right = i + R[i]

    # 找出最大回文半径
    max_center = max(range(n), key=lambda i: R[i])
    max_len = R[max_center]

    # 还原原始字符串中的位置和长度
    start = (max_center - max_len) // 2
    return s[start:start + max_len], max_len


# ============================================================================
# 第五部分：滚动哈希（Rabin-Karp）
# ============================================================================

class RollingHash:
    """
    滚动哈希：支持 O(1) 子串哈希值查询。
    用于字符串快速比较和 Rabin-Karp 匹配。
    """
    def __init__(self, s, base=131, mod=10**9 + 7):
        self.s = s
        self.n = len(s)
        self.base = base
        self.mod = mod

        # 前缀哈希：prefix[i] = hash(s[0..i-1])
        self.prefix = [0] * (self.n + 1)
        # power[i] = base^i mod mod
        self.power = [1] * (self.n + 1)

        for i in range(self.n):
            self.prefix[i + 1] = (self.prefix[i] * base + ord(s[i])) % mod
            self.power[i + 1] = (self.power[i] * base) % mod

    def get_hash(self, l, r):
        """
        返回子串 s[l..r] (inclusive) 的哈希值。
        hash = prefix[r+1] - prefix[l] * base^(r-l+1) mod mod
        """
        h = (self.prefix[r + 1] - self.prefix[l] * self.power[r - l + 1]) % self.mod
        return h if h >= 0 else h + self.mod


def rabin_karp(text, pattern):
    """
    Rabin-Karp 字符串匹配算法（使用滚动哈希）。
    O(n + m) 期望时间。
    """
    n, m = len(text), len(pattern)
    if m == 0:
        return [0]
    if m > n:
        return []

    rh = RollingHash(text)
    # 计算模式串的哈希值
    pattern_hash = 0
    base, mod = 131, 10**9 + 7
    for ch in pattern:
        pattern_hash = (pattern_hash * base + ord(ch)) % mod

    matches = []
    for i in range(n - m + 1):
        if rh.get_hash(i, i + m - 1) == pattern_hash:
            # 哈希值相同，需二次确认（避免哈希冲突）
            if text[i:i + m] == pattern:
                matches.append(i)

    return matches


# ============================================================================
# 第六部分：可视化
# ============================================================================

def visualize_kmp_next(pattern, next_arr):
    """可视化 next 数组。"""
    m = len(pattern)
    fig, ax = plt.subplots(figsize=(max(m * 1.2, 8), 4))

    # 显示模式串的每个字符
    for i in range(m):
        ax.text(i, 1.0, pattern[i], ha='center', va='center',
                fontsize=16, fontweight='bold',
                bbox=dict(boxstyle='square,pad=0.3', facecolor='#E3F2FD'))

    # 显示 next 数组的值
    for i in range(m):
        ax.text(i, 0.5, f'next={next_arr[i]}', ha='center', va='center',
                fontsize=10, color='#D32F2F')

    ax.set_xlim(-0.5, m - 0.5)
    ax.set_ylim(0, 1.5)
    ax.axis('off')
    ax.set_title(f'KMP Next Array for "{pattern}"', fontsize=14, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(_IMAGES, 'algo13_kmp_next.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[可视化] KMP next 数组图已保存: {path}')


# ============================================================================
# 主程序
# ============================================================================

def main():
    print('=' * 60)
    print('algo13 字符串算法 — 演示')
    print('=' * 60)

    # --- 1. KMP ---
    print('\n### 1. KMP 字符串匹配 ###')
    text = "ABABDABACDABABCABAB"
    pattern = "ABABCABAB"
    print(f'文本串: "{text}"')
    print(f'模式串: "{pattern}"')

    next_arr = compute_next(pattern)
    print(f'next 数组: {next_arr}')
    visualize_kmp_next(pattern, next_arr)

    matches_kmp = kmp_search(text, pattern)
    matches_bf = brute_force_search(text, pattern)
    print(f'KMP 匹配位置: {matches_kmp}')
    print(f'暴力匹配位置: {matches_bf}')
    print(f'结果一致: {matches_kmp == matches_bf}')

    # --- 2. Trie ---
    print('\n### 2. Trie 前缀树 ###')
    trie = Trie()
    words = ["cat", "car", "cart", "dog", "door", "dorm", "dove"]
    for w in words:
        trie.insert(w)
    print(f'插入单词: {words}')
    print(f'  search("cat"): {trie.search("cat")}')
    print(f'  search("card"): {trie.search("card")}')
    print(f'  starts_with("do"): {trie.starts_with("do")}')
    print(f'  autocomplete("ca"): {trie.autocomplete("ca")}')
    print(f'  autocomplete("do"): {trie.autocomplete("do")}')

    # --- 3. AC 自动机 ---
    print('\n### 3. AC 自动机 — 多模式匹配 ###')
    ac = AhoCorasick()
    patterns = ["he", "she", "his", "hers"]
    for p in patterns:
        ac.add_pattern(p)
    ac.build_failure_links()
    test_text = "ushersheishis"
    results = ac.search(test_text)
    print(f'文本: "{test_text}"')
    print(f'模式串: {patterns}')
    for pat, positions in sorted(results.items()):
        print(f'  "{pat}" 出现在位置: {positions}')

    # --- 4. Manacher ---
    print('\n### 4. Manacher 最长回文子串 ###')
    test_strings = ["babad", "cbbd", "racecar", "a"]
    for s in test_strings:
        palindrome, length = manacher(s)
        print(f'  "{s}" → "{palindrome}", 长度={length}')

    # --- 5. 滚动哈希 ---
    print('\n### 5. 滚动哈希 — Rabin-Karp ###')
    rh_test_text = "ABABDABACDABABCABAB"
    rh_test_pat = "ABAB"
    rh_matches = rabin_karp(rh_test_text, rh_test_pat)
    print(f'文本: "{rh_test_text}"')
    print(f'模式: "{rh_test_pat}" → 匹配位置: {rh_matches}')

    # 演示 O(1) 子串哈希
    rh = RollingHash("abcdefg")
    print(f'\n字符串: "abcdefg"')
    print(f'  hash("abc") = {rh.get_hash(0, 2)}')
    print(f'  hash("cde") = {rh.get_hash(2, 4)}')

    print('\n' + '=' * 60)
    print('总结：KMP/Trie/AC自动机/Manacher/滚动哈希覆盖字符串算法核心')
    print('=' * 60)


if __name__ == '__main__':
    main()
