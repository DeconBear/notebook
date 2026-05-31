# -*- coding: utf-8 -*-
"""
===============================================================================
algo03_stack_queue/code/demo.py — 栈与队列从零实现
===============================================================================
本演示从零实现以下数据结构与算法：
  1. 栈（Stack）—— 基于 Python list 和自实现的链式栈
  2. 队列（Queue）—— 循环队列
  3. 单调栈 —— 下一个更大元素问题
  4. 单调队列 —— 滑动窗口最大值
  5. 表达式求值 —— 中缀转后缀（Shunting-yard）+ 后缀求值
  6. 优先队列 —— 最小堆（二叉堆）

运行方式：python demo.py
依赖：matplotlib（仅用于可视化）
===============================================================================
"""

import os
import math
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)


# ============================================================================
# 第一部分：栈（Stack）—— 基于列表和链表两种实现
# ============================================================================

class ArrayStack:
    """基于 Python list 的栈实现（list 末尾 = 栈顶）"""
    def __init__(self):
        self._data = []

    def push(self, item):
        self._data.append(item)

    def pop(self):
        if self.is_empty():
            raise IndexError('栈空，无法弹出')
        return self._data.pop()

    def peek(self):
        if self.is_empty():
            raise IndexError('栈空')
        return self._data[-1]

    def is_empty(self):
        return len(self._data) == 0

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f'Stack(bottom→{self._data}←top)'


class Node:
    """链表节点"""
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedStack:
    """基于链表的栈实现（链表头 = 栈顶，push/pop 都是 O(1)）"""
    def __init__(self):
        self._top = None
        self._size = 0

    def push(self, item):
        new_node = Node(item)
        new_node.next = self._top
        self._top = new_node
        self._size += 1

    def pop(self):
        if self.is_empty():
            raise IndexError('栈空')
        item = self._top.data
        self._top = self._top.next
        self._size -= 1
        return item

    def peek(self):
        if self.is_empty():
            raise IndexError('栈空')
        return self._top.data

    def is_empty(self):
        return self._top is None

    def __len__(self):
        return self._size


# ============================================================================
# 第二部分：括号匹配
# ============================================================================

def is_valid_parentheses(s: str) -> bool:
    """判断括号字符串是否有效匹配。

    支持: (), [], {}
    时间复杂度 O(n)，空间复杂度 O(n)
    """
    stack = []
    pairs = {')': '(', ']': '[', '}': '{'}

    for ch in s:
        if ch in '([{':
            stack.append(ch)
        elif ch in ')]}':
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()
    return len(stack) == 0


# ============================================================================
# 第三部分：表达式求值（Shunting-yard 算法）
# ============================================================================

def infix_to_postfix(expression: str) -> str:
    """将中缀表达式转换为后缀表达式（Shunting-yard 算法）

    支持的运算符: +, -, *, /, (, )
    假定操作数为单个字母或数字字符
    """
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
    output = []
    stack = []

    for token in expression:
        if token == ' ':
            continue
        if token.isalnum():  # 操作数（字母或数字）
            output.append(token)
        elif token == '(':
            stack.append(token)
        elif token == ')':
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            if stack and stack[-1] == '(':
                stack.pop()  # 弹出 '('
        elif token in precedence:
            while (stack and stack[-1] != '(' and
                   precedence.get(stack[-1], 0) >= precedence[token]):
                output.append(stack.pop())
            stack.append(token)

    while stack:
        output.append(stack.pop())

    return ''.join(output)


def evaluate_postfix(postfix: str) -> float:
    """计算后缀表达式的值

    假定操作数为 0-9 的单个数字，运算符为 +, -, *, /
    """
    stack = []
    for token in postfix:
        if token.isdigit():
            stack.append(float(token))
        elif token in '+-*/':
            b = stack.pop()
            a = stack.pop()
            if token == '+':
                stack.append(a + b)
            elif token == '-':
                stack.append(a - b)
            elif token == '*':
                stack.append(a * b)
            elif token == '/':
                stack.append(a / b)
    return stack[0]


# ============================================================================
# 第四部分：循环队列
# ============================================================================

class CircularQueue:
    """循环队列（固定容量，牺牲一个位置区分空/满）"""

    def __init__(self, capacity):
        self._capacity = capacity + 1  # 多分配一个位置
        self._data = [None] * self._capacity
        self._front = 0  # 队头索引
        self._rear = 0   # 队尾索引（指向下一个空位）

    def is_empty(self):
        return self._front == self._rear

    def is_full(self):
        return (self._rear + 1) % self._capacity == self._front

    def enqueue(self, item):
        if self.is_full():
            raise OverflowError('队列已满')
        self._data[self._rear] = item
        self._rear = (self._rear + 1) % self._capacity

    def dequeue(self):
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

    def __repr__(self):
        return f'CircularQueue(front={self._front}, rear={self._rear}, data={self._data})'


# ============================================================================
# 第五部分：单调栈 —— 下一个更大元素
# ============================================================================

def next_greater_element(nums):
    """使用单调递减栈找出每个元素的下一个更大元素。

    返回一个列表，result[i] 是 nums[i] 右侧第一个比它大的元素，没有则为 -1。
    时间复杂度 O(n)
    """
    n = len(nums)
    result = [-1] * n
    stack = []  # 存储索引，栈中元素对应的值单调递减

    for i in range(n):
        # nums[i] 比栈顶元素大 → 更新栈顶元素的答案
        while stack and nums[i] > nums[stack[-1]]:
            idx = stack.pop()
            result[idx] = nums[i]
        stack.append(i)

    return result


# ============================================================================
# 第六部分：单调队列 —— 滑动窗口最大值
# ============================================================================

def sliding_window_maximum(nums, k):
    """使用单调递减队列找出每个滑动窗口的最大值。

    时间复杂度 O(n)
    """
    from collections import deque
    result = []
    dq = deque()  # 存储索引，对应值单调递减

    for i in range(len(nums)):
        # 删除队尾所有比当前元素小的索引
        while dq and nums[i] >= nums[dq[-1]]:
            dq.pop()
        dq.append(i)

        # 删除超出窗口范围的队头元素
        if dq[0] <= i - k:
            dq.popleft()

        # 窗口形成后记录结果
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result


# ============================================================================
# 第七部分：最小堆（二叉堆实现优先队列）
# ============================================================================

class MinHeap:
    """最小堆 —— 根节点是最小值"""

    def __init__(self):
        self._data = []

    def push(self, item):
        """插入元素，上浮调整 —— O(log n)"""
        self._data.append(item)
        self._sift_up(len(self._data) - 1)

    def pop(self):
        """弹出最小值，下沉调整 —— O(log n)"""
        if self.is_empty():
            raise IndexError('堆空')
        if len(self._data) == 1:
            return self._data.pop()

        root = self._data[0]
        self._data[0] = self._data.pop()  # 末尾元素放到堆顶
        self._sift_down(0)
        return root

    def peek(self):
        """查看最小值 —— O(1)"""
        if self.is_empty():
            raise IndexError('堆空')
        return self._data[0]

    def _sift_up(self, idx):
        """上浮：当前节点比父节点小时，与父节点交换"""
        parent = (idx - 1) // 2
        while idx > 0 and self._data[idx] < self._data[parent]:
            self._data[idx], self._data[parent] = self._data[parent], self._data[idx]
            idx = parent
            parent = (idx - 1) // 2

    def _sift_down(self, idx):
        """下沉：当前节点比子节点大时，与较小的子节点交换"""
        n = len(self._data)
        while True:
            smallest = idx
            left = 2 * idx + 1
            right = 2 * idx + 2

            if left < n and self._data[left] < self._data[smallest]:
                smallest = left
            if right < n and self._data[right] < self._data[smallest]:
                smallest = right

            if smallest == idx:
                break

            self._data[idx], self._data[smallest] = self._data[smallest], self._data[idx]
            idx = smallest

    def is_empty(self):
        return len(self._data) == 0

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f'MinHeap({self._data})'


def heap_sort(arr):
    """堆排序 —— O(n log n)，原地排序"""
    heap = MinHeap()
    for x in arr:
        heap.push(x)
    result = []
    while not heap.is_empty():
        result.append(heap.pop())
    return result


# ============================================================================
# 第八部分：可视化
# ============================================================================

def plot_monotonic_structures():
    """可视化单调栈和单调队列的算法过程"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 左上：单调栈示例
    nums = [2, 1, 2, 4, 3]
    result = next_greater_element(nums)
    ax1 = axes[0, 0]
    x = range(len(nums))
    bars = ax1.bar(x, nums, color='#90CAF9', edgecolor='#1565C0', linewidth=1.5)
    for i, (num, res) in enumerate(zip(nums, result)):
        ax1.text(i, num + 0.1, f'{num}\n→{res}', ha='center', va='bottom', fontsize=10)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([str(i) for i in range(len(nums))])
    ax1.set_title('单调栈：下一个更大元素', fontsize=14)
    ax1.set_ylabel('值', fontsize=12)

    # 右上：滑动窗口最大值
    nums2 = [1, 3, -1, -3, 5, 3, 6, 7]
    k = 3
    swm = sliding_window_maximum(nums2, k)
    ax2 = axes[0, 1]
    x2 = range(len(nums2))
    ax2.bar(x2, nums2, color='#A5D6A7', edgecolor='#2E7D32', linewidth=1)
    # 标注窗口最大值
    for i in range(k - 1, len(nums2)):
        ax2.text(i, nums2[i] + 0.5, f'max={swm[i-k+1]}',
                ha='center', fontsize=9, color='red', fontweight='bold')
    ax2.set_xticks(list(x2))
    ax2.set_title(f'单调队列：滑动窗口最大值 (k={k})', fontsize=14)
    ax2.set_ylabel('值', fontsize=12)

    # 左下：堆排序过程
    ax3 = axes[1, 0]
    arr = [7, 3, 9, 1, 5, 2, 8]
    heap = MinHeap()
    snapshots = []
    for x in arr:
        heap.push(x)
        snapshots.append(list(heap._data))

    ax3.plot(range(1, len(arr) + 1), arr, 'o-', label='原始数组', linewidth=2)
    ax3.set_title('最小堆的构建过程', fontsize=14)
    ax3.set_xlabel('元素索引', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # 右下：表达式求值过程表
    ax4 = axes[1, 1]
    ax4.axis('off')
    table_data = [
        ['中缀', '3+4*2/(1-5)'],
        ['后缀', infix_to_postfix('3+4*2/(1-5)')],
        ['结果', str(evaluate_postfix(infix_to_postfix('3+4*2/(1-5)')))],
        ['', ''],
        ['括号匹配: "{[()]}"', str(is_valid_parentheses('{[()]}'))],
        ['括号匹配: "[(])"', str(is_valid_parentheses('[(])'))],
    ]
    table = ax4.table(cellText=table_data, cellLoc='left',
                       colWidths=[0.3, 0.7], loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    for key, cell in table.get_celld().items():
        if key[0] == 0 or key[0] == 3:
            cell.set_facecolor('#E3F2FD')
    ax4.set_title('表达式求值与括号匹配', fontsize=14)

    plt.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'stack_queue_demo.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[图片保存] {out}')


# ============================================================================
# 主函数
# ============================================================================

def main():
    print('=' * 60)
    print('  栈与队列 — 演示程序')
    print('=' * 60)

    # 1. 栈
    print('\n--- 栈 ---')
    stack = ArrayStack()
    for x in [1, 2, 3]:
        stack.push(x)
    print(f'push 1,2,3 后: {stack}')
    print(f'pop: {stack.pop()}, 栈变为: {stack}')
    print(f'peek: {stack.peek()}')

    # 2. 括号匹配
    print('\n--- 括号匹配 ---')
    test_cases = ['()', '()[]{}', '(]', '([)]', '{[]}', '']
    for s in test_cases:
        print(f'  "{s}" → {"有效 ✅" if is_valid_parentheses(s) else "无效 ❌"}')

    # 3. 表达式求值
    print('\n--- 表达式求值 ---')
    expr = '3+4*2'
    postfix = infix_to_postfix(expr)
    result = evaluate_postfix(postfix)
    print(f'中缀: {expr}')
    print(f'后缀: {postfix}')
    print(f'结果: {result} (预期: 11)')

    expr2 = '1+2*3-4/2'
    postfix2 = infix_to_postfix(expr2)
    print(f'\n中缀: {expr2}')
    print(f'后缀: {postfix2}')
    print(f'结果: {evaluate_postfix(postfix2)} (预期: 5.0)')

    # 4. 循环队列
    print('\n--- 循环队列 ---')
    cq = CircularQueue(4)
    for x in [10, 20, 30]:
        cq.enqueue(x)
    print(f'enqueue 10,20,30: {cq}')
    print(f'dequeue: {cq.dequeue()} → {cq}')
    cq.enqueue(40); cq.enqueue(50)
    print(f'enqueue 40,50: {cq}')
    print(f'队列满? {cq.is_full()}')

    # 5. 单调栈
    print('\n--- 单调栈：下一个更大元素 ---')
    nums = [2, 1, 2, 4, 3]
    print(f'输入: {nums}')
    print(f'输出: {next_greater_element(nums)}')

    # 6. 单调队列
    print('\n--- 单调队列：滑动窗口最大值 ---')
    nums2 = [1, 3, -1, -3, 5, 3, 6, 7]
    k = 3
    print(f'输入: {nums2}, k={k}')
    print(f'输出: {sliding_window_maximum(nums2, k)}')

    # 7. 最小堆
    print('\n--- 最小堆 ---')
    heap = MinHeap()
    for x in [7, 3, 9, 1, 5]:
        heap.push(x)
    print(f'push 7,3,9,1,5: {heap}')
    print(f'堆顶 (最小值): {heap.peek()}')
    sorted_result = heap_sort([7, 3, 9, 1, 5, 2])
    print(f'堆排序: {sorted_result}')

    # 8. 可视化
    plot_monotonic_structures()

    print('\n✅ 所有演示已完成！图片保存在 images/ 目录中。')


if __name__ == '__main__':
    main()
