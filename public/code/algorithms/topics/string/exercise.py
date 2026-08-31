# -*- coding: utf-8 -*-
"""
algo13 字符串算法 — 练习代码
=============================
请完成以下 TODO 任务，巩固字符串算法的核心技能。
"""


# ============================================================================
# TODO 1: 实现 KMP 的 next 数组，并扩展为统计模式串出现次数
# ============================================================================
def kmp_count_occurrences(text, pattern):
    """
    使用 KMP 统计模式串在文本串中出现的次数（包括重叠匹配）。
    例如 "AAAA" 中 "AA" 出现 3 次（位置 0,1,2）。

    参数:
        text: 文本串
        pattern: 模式串
    返回:
        出现次数
    """
    # TODO: 补全实现
    # 提示：可以使用 demo.py 中的 compute_next() 和 kmp_search() 作为参考
    pass


# ============================================================================
# TODO 2: 实现 Trie 的删除操作和最长公共前缀查询
# ============================================================================
class TrieExtended:
    """扩展 Trie，支持删除和最长公共前缀查询。"""
    def __init__(self):
        self.root = {}

    def insert(self, word):
        """插入单词。"""
        # TODO: 实现
        pass

    def delete(self, word):
        """
        删除单词。只删除单词结尾标记，不删除节点（简化实现）。
        返回 True 表示删除成功，False 表示单词不存在。
        """
        # TODO: 实现
        pass

    def longest_common_prefix(self):
        """
        返回所有插入单词的最长公共前缀。
        提示：从根出发，只要某个节点只有一个子节点且非单词结尾，就继续。
        """
        # TODO: 实现
        return ''


# ============================================================================
# TODO 3: Manacher 算法扩展 — 统计所有回文子串数量
# ============================================================================
def count_all_palindromic_substrings(s):
    """
    使用 Manacher 算法的 R 数组，统计 s 中所有回文子串的数量。
    Manacher 算法的 R[i] 表示以 i 为中心的回文半径，
    每个中心 i 贡献了 R[i] 个回文子串。

    参数:
        s: 输入字符串
    返回:
        所有回文子串的总数
    """
    # TODO: 使用 Manacher 预处理（插入 #），然后统计
    # 提示：在原串预处理后的 T 中，R[i] 是以 i 为中心的回文半径
    # 原串中实际的回文子串数 = sum((R[i] + 1) // 2)
    return 0


# ============================================================================
# TODO 4: 滚动哈希实现重复子串检测
# ============================================================================
def find_longest_repeated_substring(s):
    """
    使用滚动哈希 + 二分搜索，找到字符串 s 中最长的重复子串。
    思路：二分答案长度 L，检查是否存在长度为 L 的重复子串。

    参数:
        s: 输入字符串
    返回:
        最长的重复子串（若有多个，返回任意一个）及其长度
    """
    # TODO: 实现
    # 1. 二分搜索长度 L
    # 2. 对给定的 L，用滚动哈希检查是否所有长度为 L 的子串都唯一
    # 3. 对于每个 L，用 set 存储哈希值，若有重复则 L 可行
    return '', 0


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == '__main__':
    print('=' * 50)
    print('algo13 字符串算法 — 练习')
    print('=' * 50)

    # 测试 TODO 1
    print('\n--- TODO 1: KMP 统计出现次数 ---')
    print(f'  "AAAA" 中 "AA": {kmp_count_occurrences("AAAA", "AA")} (期望: 3)')
    print(f'  "ABABAB" 中 "ABA": {kmp_count_occurrences("ABABAB", "ABA")} (期望: 2)')

    # 测试 TODO 2
    print('\n--- TODO 2: Trie 扩展 ---')
    trie2 = TrieExtended()
    for w in ['flower', 'flow', 'flight']:
        trie2.insert(w)
    print(f'  最长公共前缀: {trie2.longest_common_prefix()} (期望: "fl")')

    # 测试 TODO 3
    print('\n--- TODO 3: 统计所有回文子串 ---')
    print(f'  "abc" → {count_all_palindromic_substrings("abc")} (期望: 3: a,b,c)')
    print(f'  "aaa" → {count_all_palindromic_substrings("aaa")} (期望: 6)')

    # 测试 TODO 4
    print('\n--- TODO 4: 最长重复子串 ---')
    sub, length = find_longest_repeated_substring('banana')
    print(f'  "banana" → "{sub}", 长度={length}')

    print('\n提示: 请补全上方所有 TODO 函数后再运行测试。')
