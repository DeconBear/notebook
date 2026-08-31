# -*- coding: utf-8 -*-
"""
===============================================================================
algo04_tree_binarytree/code/exercise.py — 树与二叉树练习
===============================================================================
练习目标：
  1. 实现二叉树的中序和后序遍历（迭代版）
  2. 实现 BST 的插入和查找操作
  3. 实现 AVL 树的右旋和左旋操作
  4. 实现哈夫曼编码的完整编码/解码

运行方式：python exercise.py
===============================================================================
"""


class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


# ============================================================================
# 任务 1：迭代版后序遍历
# ============================================================================

def postorder_iterative(root):
    """后序遍历（迭代）—— 使用栈

    思路：使用两个栈，或者用一个栈 + 记录 last_visited
    技巧：后序遍历的逆序 = 根→右→左 的逆序
    """
    # TODO: 实现迭代版后序遍历
    if not root:
        return []
    result = []
    stack = [root]
    # 先做 根→右→左 的遍历，然后反转
    while stack:
        node = stack.pop()
        result.append(node.val)
        if node.left:
            stack.append(node.left)
        if node.right:
            stack.append(node.right)
    # 反转得到 左→右→根
    return result[::-1]


# ============================================================================
# 任务 2：实现 BST 的插入和查找
# ============================================================================

class BST:
    def __init__(self):
        self.root = None

    # TODO: 实现 insert 方法（迭代版）
    def insert(self, val):
        if not self.root:
            self.root = TreeNode(val)
            return
        cur = self.root
        while True:
            if val < cur.val:
                if not cur.left:
                    cur.left = TreeNode(val)
                    return
                cur = cur.left
            else:
                if not cur.right:
                    cur.right = TreeNode(val)
                    return
                cur = cur.right

    # TODO: 实现 search 方法
    def search(self, val):
        cur = self.root
        while cur:
            if val == cur.val:
                return True
            elif val < cur.val:
                cur = cur.left
            else:
                cur = cur.right
        return False

    def inorder(self):
        result = []
        def dfs(node):
            if not node: return
            dfs(node.left)
            result.append(node.val)
            dfs(node.right)
        dfs(self.root)
        return result


# ============================================================================
# 任务 3（Bonus）：AVL 树的旋转
# ============================================================================

class AVLNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.height = 1


def get_height(node):
    return node.height if node else 0


def update_height(node):
    node.height = 1 + max(get_height(node.left), get_height(node.right))


# TODO: 实现右旋（LL 旋转）
def rotate_right(y):
    r"""
       y              x
      / \            / \
     x   T4   ->    z    y
    / \                / \
   z   T3            T3  T4
   图示：y 的平衡因子 = 2，x 的平衡因子 = 1（LL 情况）
   """
    # TODO: 你的代码
    x = y.left
    T3 = x.right
    x.right = y
    y.left = T3
    update_height(y)
    update_height(x)
    return x


# TODO: 实现左旋（RR 旋转）
def rotate_left(x):
    r"""
   x                y
  / \              / \
 T1  y     ->     x    z
    / \         / \
   T2  z       T1 T2
   图示：x 的平衡因子 = -2，y 的平衡因子 = -1（RR 情况）
   """
    # TODO: 你的代码
    y = x.right
    T2 = y.left
    y.left = x
    x.right = T2
    update_height(x)
    update_height(y)
    return y


# ============================================================================
# 任务 4：哈夫曼编解码
# ============================================================================

import heapq


class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


def huffman_encode(text):
    """哈夫曼编码：返回 (编码后的二进制串, 编码表)"""
    # 统计频率
    freq = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1

    # 构建哈夫曼树
    # TODO: 你的代码 —— 使用优先队列合并节点
    heap = [HuffmanNode(ch, f) for ch, f in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    root = heap[0]

    # 生成编码表
    # TODO: 你的代码 —— DFS 遍历哈夫曼树生成 0/1 编码
    codes = {}
    def dfs(node, code):
        if not node: return
        if node.char is not None:
            codes[node.char] = code
            return
        dfs(node.left, code + '0')
        dfs(node.right, code + '1')
    dfs(root, '')

    # 编码
    encoded = ''.join(codes[ch] for ch in text)
    return encoded, codes


def huffman_decode(encoded, codes):
    """哈夫曼解码 —— 根据编码表还原原文"""
    # 反转编码表：code → char
    # TODO: 你的代码
    reverse_codes = {code: char for char, code in codes.items()}
    result = []
    current = ''
    for bit in encoded:
        current += bit
        if current in reverse_codes:
            result.append(reverse_codes[current])
            current = ''
    return ''.join(result)


# ============================================================================
# 主函数
# ============================================================================

def main():
    print('=' * 60)
    print('  树与二叉树 — 练习程序')
    print('=' * 60)

    # 测试任务 1
    print('\n--- 任务 1：迭代版后序遍历 ---')
    root = TreeNode(1)
    root.left = TreeNode(2); root.right = TreeNode(3)
    root.left.left = TreeNode(4); root.left.right = TreeNode(5)
    post = postorder_iterative(root)
    print(f'后序: {post}')
    assert post == [4, 5, 2, 3, 1]
    print('✅ 任务 1 通过！')

    # 测试任务 2
    print('\n--- 任务 2：BST ---')
    bst = BST()
    for v in [5, 3, 7, 2, 4, 6, 8]:
        bst.insert(v)
    print(f'BST 中序: {bst.inorder()}')
    assert bst.search(4) and bst.search(7) and not bst.search(99)
    print('✅ 任务 2 通过！')

    # 测试任务 3
    print('\n--- 任务 3：AVL 旋转 ---')
    # 构建 LL 情况
    y = AVLNode(30); y.left = AVLNode(20); y.left.left = AVLNode(10)
    y.height = 3; y.left.height = 2
    new_root = rotate_right(y)
    print(f'右旋后: root={new_root.val}, left={new_root.left.val}, right={new_root.right.val}')
    assert new_root.val == 20
    print('✅ 任务 3 通过！')

    # 测试任务 4
    print('\n--- 任务 4：哈夫曼编解码 ---')
    text = "ABRACADABRA"
    encoded, codes = huffman_encode(text)
    decoded = huffman_decode(encoded, codes)
    print(f'原文: {text}')
    print(f'编码表: {codes}')
    print(f'编码: {encoded} (长度: {len(encoded)} bits)')
    print(f'解码: {decoded}')
    assert decoded == text
    print(f'原始大小: {len(text) * 8} bits, 压缩后: {len(encoded)} bits, 压缩率: {len(encoded)/(len(text)*8)*100:.1f}%')
    print('✅ 任务 4 通过！')

    print('\n🎉 所有练习已完成！')


if __name__ == '__main__':
    main()
