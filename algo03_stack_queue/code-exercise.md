---
title: "algo03 栈与队列 — exercise.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# algo03 栈与队列 — exercise.py 练习指南

<a href="../code/algo03_stack_queue/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过四个递进任务掌握栈和队列的核心实现与经典应用。

## 任务清单

### 任务1：补全循环队列

实现 `CircularQueue.enqueue()` 和 `dequeue()`。

**关键**：使用取模运算实现循环：`rear = (rear + 1) % capacity`。

**判空/判满机制**：牺牲一个位置。`capacity` 参数是用户期望的容量，实际分配 `capacity + 1` 个位置。

### 任务2：实现最小栈（Min Stack）

**核心技巧**：维护两个栈：
- `main_stack`：正常存元素
- `min_stack`：栈顶始终是当前状态下的最小值

`push(x)` 时：只有当 `x <= min_stack[-1]` 才将 x 压入 `min_stack`。
`pop()` 时：弹出元素若等于 `min_stack[-1]`，也弹出 min_stack。

### 任务3：每日温度（单调栈）

使用单调递减栈解决 LeetCode 739：

```
对于 temperatures[i]：
  只要栈顶温度 < 当前温度 → 弹出栈顶，answer[栈顶] = i - 栈顶
  将 i 入栈
```

### 任务4（Bonus）：支持多位数的 Shunting-yard

针对空格分隔的 token 实现中缀转后缀。

**示例**：`"12 + 34 * 5"` → 分割为 `["12", "+", "34", "*", "5"]` → `"12 34 5 * +"`

## 验证

```bash
cd algo03_stack_queue/code
python exercise.py
```

预期：四任务全部 `✅ 通过`，`🎉 所有练习已完成！`

## 完整代码

<<< @/snippets/algo03_stack_queue/exercise.py
