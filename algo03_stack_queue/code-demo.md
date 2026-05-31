---
title: "algo03 栈与队列 — demo.py"
---

# algo03 栈与队列 — demo.py 代码详解

<a href="../code/algo03_stack_queue/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd algo03_stack_queue/code
python demo.py
```

## 代码结构

| 类/函数 | 功能 | 重要性 |
|---------|------|--------|
| `ArrayStack` / `LinkedStack` | 栈的两种实现 | 基础 |
| `is_valid_parentheses()` | 括号匹配 | 面试经典题 |
| `infix_to_postfix()` + `evaluate_postfix()` | 表达式求值 | Shunting-yard 算法 |
| `CircularQueue` | 循环队列 | 固定大小队列的标准实现 |
| `next_greater_element()` | 单调栈 | 将 O(n²) 优化到 O(n) |
| `sliding_window_maximum()` | 单调队列 | LeetCode 239 经典 |
| `MinHeap` | 最小堆 | 优先队列的底层实现 |

## 第1步：栈的两种实现

**ArrayStack** 直接使用 Python list，栈顶 = list 末尾。`push = append`, `pop = pop()`。

**LinkedStack** 用链表实现，栈顶 = 链表头。`push` 在头部插入，`pop` 从头部删除，都是 O(1)。

## 第2步：Shunting-yard 算法核心

Dijkstra 的 Shunting-yard 算法由三个规则驱动：

```
操作数 → 直接输出
'('     → 入栈
')'     → 弹出栈直到遇到 '('
运算符  → 弹出栈中所有优先级 >= 当前运算符的运算符，然后当前入栈
```

**为什么 '(' 优先级最低？** 因为 '(' 只有在遇到 ')' 时才弹出，其他运算符不应"越"过它。

## 第3步：循环队列的三个关键变量

```
class CircularQueue:
    _capacity = k + 1    # 多一个位置区分空/满
    _front = 0           # 队头索引
    _rear = 0            # 队尾索引（下一个空位）

判空: front == rear
判满: (rear + 1) % capacity == front
```

**为什么要牺牲一个位置？** 如果所有 k 个位置都存元素，front == rear 既可能是空也可能是满。牺牲一个位置后，front == rear 只能是空。

## 第4步：单调栈的 O(n) 分析

```
for i in range(n):
    while stack and nums[i] > nums[stack[-1]]:
        stack.pop()    # 每个元素最多 pop 一次
    stack.append(i)    # 每个元素最多 push 一次
```

总操作次数 = push n 次 + pop 最多 n 次 = O(n)

## 第5步：最小堆的数组表示

```
对于索引 i 的节点（根节点索引 0）：
  父节点: (i - 1) // 2
  左子节点: 2*i + 1
  右子节点: 2*i + 2

示例: [1, 3, 5, 7, 9]
树形:       1
          /   \
         3     5
        / \
       7   9
```

## 关键概念速查表

| 结构 | 规则 | push/enqueue | pop/dequeue | peek | 应用 |
|------|------|-------------|-------------|------|------|
| Stack | LIFO | O(1) | O(1) | O(1) | 括号匹配、DFS |
| Queue | FIFO | O(1) | O(1) | O(1) | BFS、消息队列 |
| Mono Stack | 单调 | O(1)均摊 | O(1)均摊 | - | 下一个更大元素 |
| Mono Queue | 单调 | O(1)均摊 | O(1)均摊 | O(1) | 滑动窗口最大值 |
| MinHeap | 堆序 | O(log n) | O(log n) | O(1) | 优先队列、堆排序 |

## 完整代码

<<< @/snippets/algo03_stack_queue/demo.py
