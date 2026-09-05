---
title: "algo04 树与二叉树 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# algo04 树与二叉树 — demo.py 代码详解

<a href="/notebook/code/algorithms/basics/tree/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/algorithms/basics/tree/code
python demo.py
```

## 代码结构

| 类/函数 | 功能 | 关键算法 |
|---------|------|----------|
| `preorder/inorder/postorder_recursive` | 递归遍历 | DFS |
| `inorder_iterative` | 迭代中序遍历 | 栈模拟递归 |
| `morris_inorder` | O(1) 空间中序遍历 | 利用空指针建回链 |
| `BST` | 二叉搜索树 | 插入/查找/删除 |
| `AVLTree` | 自平衡 BST | LL/RR/LR/RL 旋转 |
| `build_huffman_tree` + `generate_huffman_codes` | 哈夫曼编码 | 贪心 + 优先队列 |

## 第1步：Morris 遍历的核心技巧

```python
if not pre.right:
    pre.right = cur      # 建立"回链"
    cur = cur.left
else:
    pre.right = None     # 断开"回链"，访问 cur
    result.append(cur.val)
    cur = cur.right
```

**直觉**：利用叶子节点空闲的 `right` 指针，指向中序后继，从而在不使用栈的情况下实现回溯。

## 第2步：BST 删除的三种情况

```
情况 1（叶节点）:  直接删除，父节点的引用设为 None
情况 2（一个子节点）: 用子节点替代当前节点
情况 3（两个子节点）: 用后继（右子树的最小值）替换值，然后删除后继
```

## 第3步：AVL 树的四种旋转

| 旋转 | 平衡因子模式 | 操作 |
|------|-------------|------|
| LL | node.bf=2, left.bf=1 | 右旋一次 |
| RR | node.bf=-2, right.bf=-1 | 左旋一次 |
| LR | node.bf=2, left.bf=-1 | 先左旋左子，再右旋 |
| RL | node.bf=-2, right.bf=1 | 先右旋右子，再左旋 |

## 第4步：哈夫曼树的贪心证明（直觉）

哈夫曼算法的高明之处在于它每次取最小的两个频率合并。为什么这是最优的？

**交换论证**：假设最优树中频率最小的两个字符不是兄弟，我们可以将它们与最深层的两个兄弟交换（因为最深层的频率一定不大于最优树中任何节点的频率），交换后的树总代价不增加。因此，最小频率的两个字符一定在最优树的最深层且互为兄弟——这正是哈夫曼算法做的。

## 关键概念速查表

| 遍历 | 顺序 | 递归 | 迭代 | 应用 |
|------|------|------|------|------|
| 先序 | 根-左-右 | O(n), O(h)空间 | 栈 | 序列化、前缀表达式 |
| 中序 | 左-根-右 | O(n), O(h)空间 | 栈/Morris | BST 有序输出 |
| 后序 | 左-右-根 | O(n), O(h)空间 | 双栈 | 计算子树大小、后缀表达式 |
| 层序 | 逐层从左到右 | - | 队列 O(n), O(w)空间 | BFS、最短路径 |


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/algorithms/basics/tree/code/demo.py`
