# -*- coding: utf-8 -*-
"""
===============================================================================
algo03_stack_queue/code/exercise.py — 栈与队列练习
===============================================================================
本练习文件中，你需要完成以下任务：
  1. 实现循环队列的 enqueue/dequeue（基于数组）
  2. 实现最小栈（支持 O(1) 时间获取最小值）
  3. 实现单调栈解决「每日温度」问题
  4. 实现 Shunting-yard 算法（支持多位数和括号）

运行方式：python exercise.py
===============================================================================
"""


# ============================================================================
# 任务 1：补全循环队列
# ============================================================================

class CircularQueue:
    """循环队列 —— 固定容量，牺牲一个位置区分空/满"""

    def __init__(self, capacity):
        self._capacity = capacity + 1  # 多一个位置
        self._data = [None] * self._capacity
        self._front = 0
        self._rear = 0

    def is_empty(self):
        return self._front == self._rear

    def is_full(self):
        return (self._rear + 1) % self._capacity == self._front

    # TODO: 实现 enqueue
    # 1. 检查队列是否已满
    # 2. 在 rear 位置放入元素
    # 3. rear = (rear + 1) % capacity
    def enqueue(self, item):
        # TODO: 你的代码
        if self.is_full():
            raise OverflowError('队列已满')
        self._data[self._rear] = item
        self._rear = (self._rear + 1) % self._capacity

    # TODO: 实现 dequeue
    # 1. 检查队列是否为空
    # 2. 取出 front 位置的元素
    # 3. front = (front + 1) % capacity
    # 4. 返回取出的元素
    def dequeue(self):
        # TODO: 你的代码
        if self.is_empty():
            raise IndexError('队列空')
        item = self._data[self._front]
        self._data[self._front] = None
        self._front = (self._front + 1) % self._capacity
        return item

    def peek(self):
        if self.is_empty():
            raise IndexError('队列空')
        return self._data[self._front]


# ============================================================================
# 任务 2：实现最小栈（Min Stack）
# ============================================================================

class MinStack:
    """最小栈 —— 支持 push、pop、top 和 getMin（均为 O(1) 时间）

    提示：使用两个栈——
      - main_stack：存所有元素
      - min_stack：存每个状态下的最小值
    """

    def __init__(self):
        self.main_stack = []
        self.min_stack = []

    # TODO: 实现 push
    # 1. 将 x 压入 main_stack
    # 2. 若 min_stack 为空或 x <= min_stack[-1]，将 x 压入 min_stack
    def push(self, x):
        # TODO: 你的代码
        self.main_stack.append(x)
        if not self.min_stack or x <= self.min_stack[-1]:
            self.min_stack.append(x)

    # TODO: 实现 pop
    # 1. 弹出 main_stack 的栈顶元素
    # 2. 若该元素等于 min_stack 的栈顶，也弹出 min_stack 的栈顶
    def pop(self):
        # TODO: 你的代码
        if not self.main_stack:
            raise IndexError('栈空')
        x = self.main_stack.pop()
        if x == self.min_stack[-1]:
            self.min_stack.pop()
        return x

    def top(self):
        if not self.main_stack:
            raise IndexError('栈空')
        return self.main_stack[-1]

    def getMin(self):
        if not self.min_stack:
            raise IndexError('栈空')
        return self.min_stack[-1]


# ============================================================================
# 任务 3：单调栈 —— 每日温度问题
# ============================================================================

def daily_temperatures(temperatures):
    """LeetCode 739: 对于每一天，找出需要等待多少天才能遇到更高的温度。

    输入: temperatures = [73, 74, 75, 71, 69, 72, 76, 73]
    输出: [1, 1, 4, 2, 1, 1, 0, 0]

    提示：使用单调递减栈（存储索引）。
    当遇到比栈顶高的温度时，当前索引与栈顶索引的差就是答案。
    """
    n = len(temperatures)
    answer = [0] * n
    stack = []  # 存储索引，对应温度单调递减

    # TODO: 你的代码
    for i in range(n):
        # 当前温度比栈顶温度高时，更新答案
        while stack and temperatures[i] > temperatures[stack[-1]]:
            idx = stack.pop()
            answer[idx] = i - idx
        stack.append(i)

    return answer


# ============================================================================
# 任务 4（Bonus）：实现支持多位数的 Shunting-yard
# ============================================================================

def infix_to_postfix_advanced(expression):
    """中缀转后缀 —— 支持多位数数字

    输入: "12 + 34 * 5"
    输出: "12 34 5 * +"

    提示：
    - 用空格分隔 token 便于解析
    - 多位数数字需要累积读取
    """
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
    output = []
    stack = []

    # TODO: 你的代码
    tokens = expression.split()
    for token in tokens:
        if token.isdigit():
            output.append(token)
        elif token == '(':
            stack.append(token)
        elif token == ')':
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            if stack and stack[-1] == '(':
                stack.pop()
        elif token in precedence:
            while (stack and stack[-1] != '(' and
                   precedence.get(stack[-1], 0) >= precedence[token]):
                output.append(stack.pop())
            stack.append(token)

    while stack:
        output.append(stack.pop())

    return ' '.join(output)


# ============================================================================
# 主函数
# ============================================================================

def main():
    print('=' * 60)
    print('  栈与队列 — 练习程序')
    print('=' * 60)

    # 测试任务 1
    print('\n--- 任务 1：循环队列 ---')
    cq = CircularQueue(4)
    for x in [1, 2, 3, 4]:
        cq.enqueue(x)
    print(f'enqueue 1,2,3,4: 满? {cq.is_full()}')
    print(f'dequeue: {cq.dequeue()}')  # 1
    print(f'dequeue: {cq.dequeue()}')  # 2
    cq.enqueue(5)
    cq.enqueue(6)
    print(f'enqueue 5,6: 满? {cq.is_full()}')
    assert cq.is_full()
    print('✅ 任务 1 通过！')

    # 测试任务 2
    print('\n--- 任务 2：最小栈 ---')
    ms = MinStack()
    ms.push(-2); ms.push(0); ms.push(-3)
    print(f'push -2, 0, -3 → min = {ms.getMin()} (期望 -3)')
    ms.pop()
    print(f'pop → top = {ms.top()}, min = {ms.getMin()} (期望-2)')
    assert ms.getMin() == -2
    print('✅ 任务 2 通过！')

    # 测试任务 3
    print('\n--- 任务 3：每日温度 ---')
    temps = [73, 74, 75, 71, 69, 72, 76, 73]
    result = daily_temperatures(temps)
    print(f'输入: {temps}')
    print(f'输出: {result}')
    print(f'期望: [1, 1, 4, 2, 1, 1, 0, 0]')
    assert result == [1, 1, 4, 2, 1, 1, 0, 0]
    print('✅ 任务 3 通过！')

    # 测试任务 4
    print('\n--- 任务 4：高级中缀转后缀 ---')
    expr = "12 + 34 * 5"
    postfix = infix_to_postfix_advanced(expr)
    print(f'中缀: {expr}')
    print(f'后缀: {postfix}')
    assert postfix == "12 34 5 * +", f'期望 "12 34 5 * +"，实际 "{postfix}"'
    print('✅ 任务 4 通过！')

    print('\n🎉 所有练习已完成！')


if __name__ == '__main__':
    main()
