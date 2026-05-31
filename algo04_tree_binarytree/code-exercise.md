---
title: "algo04 树与二叉树 — exercise.py"
---

# algo04 树与二叉树 — exercise.py 练习指南

<a href="../code/algo04_tree_binarytree/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过四个任务掌握树的遍历、BST 操作、AVL 旋转和哈夫曼编码。

## 任务清单

### 任务1：迭代版后序遍历

**技巧**：后序（左-右-根）= 反序（根-右-左）。用一个栈先做根-右-左，然后反转。

```
标准后序: 4, 5, 2, 3, 1
根右左:   1, 3, 2, 5, 4
→ 反转即可!
```

### 任务2：BST 插入和查找

`insert`: 从根出发，根据比较结果左转或右转，直到找到空位。
`search`: 同样逻辑，找到匹配值或抵达 None。

### 任务3：AVL 旋转

**右旋**（LL）：`y` 是失衡节点，`x = y.left`，`T3 = x.right`。旋转后 `x` 成为新的根。

**左旋**（RR）：`x` 是失衡节点，`y = x.right`，`T2 = y.left`。旋转后 `y` 成为新的根。

### 任务4（Bonus）：哈夫曼编解码

编码：频率统计 → 优先队列合并 → DFS 生成编码表 → 逐字符编码。
解码：反转编码表，从编码串头部开始匹配。

## 验证

```bash
cd algo04_tree_binarytree/code
python exercise.py
```

预期全部通过，并展示哈夫曼编码的压缩效果。

## 完整代码

<<< @/snippets/algo04_tree_binarytree/exercise.py
