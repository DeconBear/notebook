# -*- coding: utf-8 -*-
"""
===============================================================================
algo04_tree_binarytree/code/demo.py — 树与二叉树从零实现
===============================================================================
本演示从零实现：
  1. 二叉树构建与四种遍历（先序、中序、后序、层序）
  2. Morris 遍历（O(1) 空间中序遍历）
  3. 二叉搜索树（BST）—— 插入、查找、删除
  4. AVL 树 —— 带旋转的自平衡 BST
  5. 哈夫曼编码树

运行方式：python demo.py
依赖：matplotlib（仅用于可视化树结构）
===============================================================================
"""

import os
import heapq
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)


# ============================================================================
# 第一部分：二叉树节点与遍历
# ============================================================================

class TreeNode:
    """二叉树节点"""
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


def preorder_recursive(root):
    """先序遍历（递归）：根 → 左 → 右"""
    result = []
    def dfs(node):
        if not node: return
        result.append(node.val)
        dfs(node.left)
        dfs(node.right)
    dfs(root)
    return result


def inorder_recursive(root):
    """中序遍历（递归）：左 → 根 → 右"""
    result = []
    def dfs(node):
        if not node: return
        dfs(node.left)
        result.append(node.val)
        dfs(node.right)
    dfs(root)
    return result


def postorder_recursive(root):
    """后序遍历（递归）：左 → 右 → 根"""
    result = []
    def dfs(node):
        if not node: return
        dfs(node.left)
        dfs(node.right)
        result.append(node.val)
    dfs(root)
    return result


def inorder_iterative(root):
    """中序遍历（迭代）：使用栈模拟递归"""
    result = []
    stack = []
    cur = root
    while stack or cur:
        while cur:               # 一路向左
            stack.append(cur)
            cur = cur.left
        cur = stack.pop()        # 弹出访问
        result.append(cur.val)
        cur = cur.right           # 转向右子树
    return result


def level_order(root):
    """层序遍历（BFS）"""
    from collections import deque
    if not root:
        return []
    result = []
    q = deque([root])
    while q:
        node = q.popleft()
        result.append(node.val)
        if node.left:
            q.append(node.left)
        if node.right:
            q.append(node.right)
    return result


def morris_inorder(root):
    """Morris 中序遍历 —— O(1) 空间复杂度（不使用栈/递归）

    利用叶子节点的空右指针建立临时"回链"。
    """
    result = []
    cur = root
    while cur:
        if not cur.left:
            # 没有左子树，访问当前节点，向右走
            result.append(cur.val)
            cur = cur.right
        else:
            # 找左子树的最右节点（中序遍历的前驱）
            pre = cur.left
            while pre.right and pre.right != cur:
                pre = pre.right

            if not pre.right:
                # 建立回链（第一次到达 cur）
                pre.right = cur
                cur = cur.left
            else:
                # 断开回链（第二次到达 cur，左子树已访问完）
                pre.right = None
                result.append(cur.val)
                cur = cur.right
    return result


# ============================================================================
# 第二部分：二叉搜索树（BST）
# ============================================================================

class BST:
    """二叉搜索树"""

    def __init__(self):
        self.root = None

    def insert(self, val):
        """插入 —— O(h)"""
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

    def search(self, val):
        """查找 —— O(h)，返回节点或 None"""
        cur = self.root
        while cur:
            if val == cur.val:
                return cur
            elif val < cur.val:
                cur = cur.left
            else:
                cur = cur.right
        return None

    def delete(self, val):
        """删除 —— O(h)"""
        self.root = self._delete(self.root, val)

    def _delete(self, node, val):
        if not node:
            return None
        if val < node.val:
            node.left = self._delete(node.left, val)
        elif val > node.val:
            node.right = self._delete(node.right, val)
        else:
            # 找到待删除节点
            # 情况 1 & 2：0 或 1 个子节点
            if not node.left:
                return node.right
            if not node.right:
                return node.left
            # 情况 3：两个子节点 → 用后继替换
            succ = node.right
            while succ.left:
                succ = succ.left
            node.val = succ.val
            node.right = self._delete(node.right, succ.val)
        return node

    def inorder(self):
        return inorder_recursive(self.root)


# ============================================================================
# 第三部分：AVL 树
# ============================================================================

class AVLNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.height = 1


class AVLTree:
    """AVL 树（自平衡二叉搜索树）"""

    def __init__(self):
        self.root = None

    def _height(self, node):
        return node.height if node else 0

    def _balance_factor(self, node):
        if not node:
            return 0
        return self._height(node.left) - self._height(node.right)

    def _update_height(self, node):
        node.height = 1 + max(self._height(node.left), self._height(node.right))

    def _rotate_right(self, y):
        """右旋（LL 情况）"""
        x = y.left
        T2 = x.right
        x.right = y
        y.left = T2
        self._update_height(y)
        self._update_height(x)
        return x

    def _rotate_left(self, x):
        """左旋（RR 情况）"""
        y = x.right
        T2 = y.left
        y.left = x
        x.right = T2
        self._update_height(x)
        self._update_height(y)
        return y

    def insert(self, val):
        self.root = self._insert(self.root, val)

    def _insert(self, node, val):
        # 1. 标准 BST 插入
        if not node:
            return AVLNode(val)
        if val < node.val:
            node.left = self._insert(node.left, val)
        else:
            node.right = self._insert(node.right, val)

        # 2. 更新高度
        self._update_height(node)

        # 3. 检查平衡并旋转
        balance = self._balance_factor(node)

        # LL 情况
        if balance > 1 and val < node.left.val:
            return self._rotate_right(node)
        # RR 情况
        if balance < -1 and val > node.right.val:
            return self._rotate_left(node)
        # LR 情况
        if balance > 1 and val > node.left.val:
            node.left = self._rotate_left(node.left)
            return self._rotate_right(node)
        # RL 情况
        if balance < -1 and val < node.right.val:
            node.right = self._rotate_right(node.right)
            return self._rotate_left(node)

        return node

    def inorder(self):
        return inorder_recursive(self.root)


# ============================================================================
# 第四部分：哈夫曼编码树
# ============================================================================

class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_tree(freq_dict):
    """构建哈夫曼树

    参数: freq_dict = {'A': 5, 'B': 2, 'C': 1, 'D': 1}
    返回: HuffmanNode (根节点)
    """
    heap = [HuffmanNode(char, freq) for char, freq in freq_dict.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        # 合并为一个新节点
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    return heap[0]


def generate_huffman_codes(root):
    """从哈夫曼树生成编码表"""
    codes = {}

    def dfs(node, code):
        if not node:
            return
        if node.char is not None:
            codes[node.char] = code
            return
        dfs(node.left, code + '0')
        dfs(node.right, code + '1')

    dfs(root, '')
    return codes


# ============================================================================
# 第五部分：树可视化
# ============================================================================

def plot_tree(node, title, filename, node_radius=0.3):
    """使用 matplotlib 绘制二叉树"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # 计算每个节点的位置（使用中序遍历分配 x 坐标）
    positions = {}
    inorder_idx = [0]

    def assign_positions(node, depth):
        if not node:
            return
        assign_positions(node.left, depth + 1)
        positions[node] = (inorder_idx[0], -depth)
        inorder_idx[0] += 1
        assign_positions(node.right, depth + 1)

    assign_positions(node, 0)

    # 画边
    def draw_edges(node):
        if not node:
            return
        if node.left:
            x1, y1 = positions[node]
            x2, y2 = positions[node.left]
            ax.plot([x1, x2], [y1, y2], 'k-', lw=1.5, alpha=0.6)
            draw_edges(node.left)
        if node.right:
            x1, y1 = positions[node]
            x2, y2 = positions[node.right]
            ax.plot([x1, x2], [y1, y2], 'k-', lw=1.5, alpha=0.6)
            draw_edges(node.right)

    draw_edges(node)

    # 画节点
    for n, (x, y) in positions.items():
        circle = plt.Circle((x, y), node_radius, facecolor='#90CAF9',
                           edgecolor='#1565C0', linewidth=2, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, str(n.val), ha='center', va='center', fontsize=10,
               fontweight='bold', zorder=4)

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14, pad=10)

    out = os.path.join(_IMAGES_DIR, filename)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[图片保存] {out}')
    return out


# ============================================================================
# 主函数
# ============================================================================

def main():
    print('=' * 60)
    print('  树与二叉树 — 演示程序')
    print('=' * 60)

    # 1. 构建示例树
    print('\n--- 二叉树的遍历 ---')
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.left.left = TreeNode(4)
    root.left.right = TreeNode(5)

    print(f'先序: {preorder_recursive(root)}')
    print(f'中序: {inorder_recursive(root)}')
    print(f'后序: {postorder_recursive(root)}')
    print(f'层序: {level_order(root)}')
    print(f'Morris中序: {morris_inorder(root)}')

    # 2. BST 演示
    print('\n--- 二叉搜索树 ---')
    bst = BST()
    for v in [5, 3, 7, 2, 4, 6, 8]:
        bst.insert(v)
    print(f'BST 中序: {bst.inorder()}')
    print(f'查找 4: {"找到" if bst.search(4) else "未找到"}')
    bst.delete(3)
    print(f'删除 3 后: {bst.inorder()}')

    # 3. AVL 树演示
    print('\n--- AVL 树（自平衡） ---')
    avl = AVLTree()
    # 故意按升序插入（普通 BST 会退化成链表）
    for v in [10, 20, 30, 40, 50, 25]:
        avl.insert(v)
    print(f'AVL 中序: {avl.inorder()}')

    # 与普通 BST 对比
    bst2 = BST()
    for v in [10, 20, 30, 40, 50, 25]:
        bst2.insert(v)
    # BST 退化成单链：10→20→30→40→50
    avl_h = avl.root.height if avl.root else 0
    print(f'AVL 树高度: {avl_h} (log₂(6) ≈ 2.58)')
    print(f'普通 BST (升序插入) 退化高度会接近 6')

    # 4. 哈夫曼编码
    print('\n--- 哈夫曼编码 ---')
    freq = {'A': 5, 'B': 2, 'C': 1, 'D': 1}
    huff_root = build_huffman_tree(freq)
    codes = generate_huffman_codes(huff_root)
    print(f'频率: {freq}')
    print(f'哈夫曼编码: {codes}')

    # 5. 可视化
    plot_tree(root, '示例二叉树', 'binary_tree.png')
    plot_tree(bst.root, '二叉搜索树 (BST)', 'bst_tree.png')
    plot_tree(avl.root, 'AVL 树 (自平衡)', 'avl_tree.png')

    print('\n✅ 所有演示已完成！图片保存在 images/ 目录中。')


if __name__ == '__main__':
    main()
