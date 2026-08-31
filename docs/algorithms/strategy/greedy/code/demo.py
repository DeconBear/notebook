# -*- coding: utf-8 -*-
"""
algo09 贪心算法 — 演示代码
===========================
功能：展示四大经典贪心算法的完整实现——活动选择、分数背包、
      Huffman 编码、找零问题（含贪心 vs DP 对比）。
      + 可视化：活动选择甘特图、Huffman 树结构图。

每个函数都有中文 docstring，每行逻辑代码都有中文注释。
运行方式：在 algo09_greedy/ 目录下执行 python code/demo.py
"""

import os
import heapq
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False

_HERE = os.path.dirname(os.path.abspath(__file__))
_IMAGES = os.path.join(_HERE, '..', 'images')
os.makedirs(_IMAGES, exist_ok=True)


# ============================================================================
# 第一部分：活动选择问题（Activity Selection）
# ============================================================================

def activity_selection(activities):
    """
    活动选择问题：选择最多数量互不重叠的活动。
    贪心策略：按结束时间升序排列，每次选结束最早且与已选不冲突的。

    参数:
        activities: list of (start_time, end_time)，每个活动的起止时间
    返回:
        selected: 被选中的活动的索引列表
    """
    # 步骤1: 将活动按结束时间升序排列，同时记住原始索引
    # 使用 sorted 的 key 参数指定排序依据（结束时间）
    n = len(activities)
    indexed = [(activities[i][0], activities[i][1], i) for i in range(n)]
    indexed.sort(key=lambda x: x[1])  # 按结束时间升序

    selected = []       # 存储被选中的活动（原始索引）
    last_end = -1       # 上一个选中活动的结束时间（初始为负无穷）

    # 步骤2: 遍历排序后的活动，贪心选择
    for start, end, idx in indexed:
        if start >= last_end:  # 不冲突 → 可选
            selected.append(idx)
            last_end = end      # 更新最近结束时间

    return selected


# ============================================================================
# 第二部分：分数背包问题（Fractional Knapsack）
# ============================================================================

def fractional_knapsack(items, capacity):
    """
    分数背包问题：物品可分割，求最大总价值。
    贪心策略：按单位重量价值（v/w）降序排列，依次装入直到背包装满。

    参数:
        items: list of (weight, value)，每件物品的重量和价值
        capacity: 背包容量
    返回:
        total_value: 最大总价值
        taken: 每件物品取走的比例（0.0 ~ 1.0）
    """
    n = len(items)
    # 步骤1: 计算每件物品的单位价值，附加原始索引
    indexed = [(items[i][0], items[i][1], items[i][1] / items[i][0], i)
               for i in range(n)]
    # 按单位价值降序排列
    indexed.sort(key=lambda x: x[2], reverse=True)

    taken = [0.0] * n   # 记录每件物品取走的比例
    total_value = 0.0   # 累计总价值
    remaining = capacity  # 剩余容量

    # 步骤2: 按性价比依次装入
    for w, v, ratio, idx in indexed:
        if remaining <= 0:
            break
        take = min(w, remaining)  # 取走量（不超过剩余容量）
        taken[idx] = take / w     # 记录比例
        total_value += take * ratio  # 累加价值（重量 × 单位价值）
        remaining -= take         # 更新剩余容量

    return total_value, taken


# ============================================================================
# 第三部分：Huffman 编码
# ============================================================================

class HuffmanNode:
    """Huffman 树节点"""
    def __init__(self, char, freq, left=None, right=None):
        self.char = char      # 字符（叶子节点才有）
        self.freq = freq      # 频率
        self.left = left      # 左子节点（编码 0）
        self.right = right    # 右子节点（编码 1）

    def __lt__(self, other):
        """定义小于运算符，用于最小堆排序（频率小的优先）"""
        return self.freq < other.freq


def build_huffman_tree(freq_dict):
    """
    构建 Huffman 编码树。
    贪心策略：每次从最小堆中取出频率最小的两个节点，合并为新节点。

    参数:
        freq_dict: dict，{'字符': 频率}，例如 {'a': 5, 'b': 9, 'c': 12}
    返回:
        root: HuffmanNode，树的根节点
    """
    # 步骤1: 为每个字符创建叶子节点，放入最小堆
    heap = []
    for char, freq in freq_dict.items():
        heapq.heappush(heap, HuffmanNode(char, freq))

    # 步骤2: 重复合并最小的两个节点
    # 当堆中只剩一个节点时，它就是 Huffman 树的根
    while len(heap) > 1:
        left = heapq.heappop(heap)   # 取出最小频率节点
        right = heapq.heappop(heap)  # 取出次小频率节点
        # 合并：新节点频率 = 两子树频率之和
        merged = HuffmanNode(None, left.freq + right.freq, left, right)
        heapq.heappush(heap, merged)  # 将合并后的节点放回堆中

    return heapq.heappop(heap)  # 返回根节点


def generate_huffman_codes(root):
    """
    从 Huffman 树生成编码表。
    递归遍历树：左分支追加 '0'，右分支追加 '1'。

    参数:
        root: HuffmanNode，树的根节点
    返回:
        codes: dict，{'字符': '二进制编码'}，例如 {'a': '110', ...}
    """
    codes = {}

    def dfs(node, code):
        """深度优先遍历 Huffman 树，累积路径上的编码"""
        if node is None:
            return
        # 叶子节点：记录该字符的编码
        if node.char is not None:
            codes[node.char] = code
            return
        # 非叶子节点：继续向下递归
        dfs(node.left, code + '0')   # 左分支 → 编码追加 0
        dfs(node.right, code + '1')  # 右分支 → 编码追加 1

    dfs(root, '')
    return codes


def huffman_encode(text, codes):
    """使用 Huffman 编码表压缩文本"""
    return ''.join(codes[ch] for ch in text)


def huffman_decode(encoded, root):
    """使用 Huffman 树解码二进制串"""
    decoded = []
    node = root
    for bit in encoded:
        # 根据比特位走到对应的子节点
        node = node.left if bit == '0' else node.right
        if node.char is not None:  # 到达叶子节点 → 输出字符
            decoded.append(node.char)
            node = root             # 回到根节点，准备解码下一个字符
    return ''.join(decoded)


# ============================================================================
# 第四部分：找零问题（贪心 vs 动态规划对比）
# ============================================================================

def coin_change_greedy(coins, amount):
    """
    找零问题的贪心解法。
    策略：每次选面额最大且不超过剩余金额的硬币。

    参数:
        coins: list，面额列表（需从大到小排列，例如 [100, 50, 20, 10, 5, 1]）
        amount: 目标金额
    返回:
        result: dict，{面额: 使用数量}，若不能恰好凑出则返回 None
    """
    result = {}
    remaining = amount

    for coin in sorted(coins, reverse=True):  # 确保从大到小
        if remaining <= 0:
            break
        count = remaining // coin  # 当前面额最多能用几枚
        if count > 0:
            result[coin] = count
            remaining -= count * coin

    return result if remaining == 0 else None


def coin_change_dp(coins, amount):
    """
    找零问题的动态规划解法（找最优解——最少硬币数）。
    这是唯一保证正确的方法，适用于任意面额系统。

    dp[i] = 凑出金额 i 所需的最少硬币数

    参数:
        coins: list，面额列表
        amount: 目标金额
    返回:
        dp[amount]: 最少硬币数，-1 表示无法凑出
        coin_used: 用于回溯每种面额用了多少枚
    """
    INF = float('inf')
    dp = [INF] * (amount + 1)
    dp[0] = 0  # 凑出 0 元需要 0 枚硬币

    # 记录每个金额最后使用的那枚硬币的面额，用于回溯
    coin_used = [-1] * (amount + 1)

    for i in range(1, amount + 1):
        for coin in coins:
            if i >= coin and dp[i - coin] + 1 < dp[i]:
                dp[i] = dp[i - coin] + 1
                coin_used[i] = coin

    if dp[amount] == INF:
        return -1, None  # 无法凑出

    # 回溯：从 coin_used 重建用了哪些硬币
    result = {}
    remaining = amount
    while remaining > 0:
        c = coin_used[remaining]
        result[c] = result.get(c, 0) + 1
        remaining -= c

    return dp[amount], result


# ============================================================================
# 第五部分：可视化
# ============================================================================

def plot_activity_selection(activities, selected):
    """绘制活动选择的甘特图，绿色=选中，灰色=未选中"""
    fig, ax = plt.subplots(figsize=(12, 5))

    colors = ['#4CAF50' if i in selected else '#BDBDBD'
              for i in range(len(activities))]
    labels = [f'Activity {i}\n({s},{e})' for i, (s, e) in enumerate(activities)]

    for i, (start, end) in enumerate(activities):
        ax.barh(i, end - start, left=start, height=0.6,
                color=colors[i], edgecolor='white', linewidth=1.5)
        ax.text((start + end) / 2, i, labels[i], ha='center', va='center',
                fontsize=9, fontweight='bold', color='white'
                if i in selected else '#555')

    ax.set_yticks(range(len(activities)))
    ax.set_yticklabels([f'Act {i}' for i in range(len(activities))])
    ax.set_xlabel('Time', fontsize=12)
    ax.set_title('Activity Selection (Greedy: Earliest Finish Time First)',
                 fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlim(0, max(e for _, e in activities) + 2)
    ax.grid(axis='x', alpha=0.3)

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#4CAF50', label='Selected'),
                       Patch(facecolor='#BDBDBD', label='Not Selected')]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    path = os.path.join(_IMAGES, 'algo09_activity_gantt.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[可视化] 活动选择甘特图已保存: {path}')


def print_huffman_tree(root, indent=0):
    """以缩进格式打印 Huffman 树的结构"""
    if root is None:
        return
    prefix = '  ' * indent
    if root.char is not None:
        print(f'{prefix}├── [{root.char}] freq={root.freq}')
    else:
        print(f'{prefix}├── (*) freq={root.freq}')
    print_huffman_tree(root.left, indent + 1)
    print_huffman_tree(root.right, indent + 1)


# ============================================================================
# 主程序
# ============================================================================

def main():
    print('=' * 60)
    print('algo09 贪心算法 — 演示')
    print('=' * 60)

    # --- 1. 活动选择 ---
    print('\n### 1. 活动选择问题 ###')
    activities = [(1, 4), (3, 5), (0, 6), (5, 7), (3, 9),
                  (5, 9), (6, 10), (8, 11), (8, 12), (2, 14),
                  (12, 16)]
    print(f'输入活动（起止时间）: {activities}')
    selected = activity_selection(activities)
    print(f'贪心选择的活动索引: {selected}')
    print(f'共选出 {len(selected)} 个活动')
    plot_activity_selection(activities, selected)

    # --- 2. 分数背包 ---
    print('\n### 2. 分数背包问题 ###')
    items = [(10, 60), (20, 100), (30, 120)]  # (重量, 价值)
    capacity = 50
    print(f'物品（重量, 价值）: {items}')
    print(f'背包容量: {capacity}')
    total_value, taken = fractional_knapsack(items, capacity)
    print(f'最大总价值: {total_value:.2f}')
    for i, (w, v) in enumerate(items):
        print(f'  物品 {i} (w={w}, v={v}, 性价比={v/w}): 取 {taken[i]*100:.0f}%')

    # --- 3. Huffman 编码 ---
    print('\n### 3. Huffman 编码 ###')
    freq_dict = {'A': 5, 'B': 9, 'C': 12, 'D': 13, 'E': 16, 'F': 45}
    print(f'字符频率: {freq_dict}')
    root = build_huffman_tree(freq_dict)
    codes = generate_huffman_codes(root)
    print(f'Huffman 编码表:')
    for char in sorted(freq_dict.keys()):
        print(f'  {char}: {codes[char]} (频率: {freq_dict[char]})')

    # 测试编解码
    test_text = 'BADFEED'
    encoded = huffman_encode(test_text, codes)
    decoded = huffman_decode(encoded, root)
    print(f'\n原文本: {test_text}')
    print(f'编码后: {encoded}')
    print(f'解码后: {decoded}')
    print(f'压缩率: {len(encoded)} bits vs 原始 {len(test_text)*8} bits'
          f' (ASCII) = {len(encoded)/(len(test_text)*8)*100:.1f}%')

    print('\nHuffman 树结构:')
    print_huffman_tree(root)

    # --- 4. 找零问题：贪心 vs DP ---
    print('\n### 4. 找零问题：贪心 vs 动态规划对比 ###')

    # 测试1: 规范面额系统（人民币）→ 贪心正确
    coins_canonical = [1, 5, 10, 20, 50, 100]
    test_amounts = [167, 99, 38]
    print(f'\n-- 规范面额系统: {coins_canonical} --')
    for amount in test_amounts:
        greedy_result = coin_change_greedy(coins_canonical, amount)
        dp_count, dp_result = coin_change_dp(coins_canonical, amount)
        g_total = sum(greedy_result.values())
        print(f'  金额 {amount}: 贪心={g_total}枚 {greedy_result}, DP={dp_count}枚 {dp_result}'
              f' → {"✓一致" if g_total == dp_count else "✗不一致"}')

    # 测试2: 非规范面额系统 → 贪心可能失败
    coins_nonc = [1, 3, 4]
    print(f'\n-- 非规范面额系统: {coins_nonc} --')
    for amount in [6, 8, 10, 15]:
        greedy_result = coin_change_greedy(coins_nonc, amount)
        dp_count, dp_result = coin_change_dp(coins_nonc, amount)
        g_total = sum(greedy_result.values()) if greedy_result else float('inf')
        print(f'  金额 {amount}: 贪心={g_total}枚 {greedy_result}, DP={dp_count}枚 {dp_result}'
              f' → {"✓一致" if g_total == dp_count else "✗贪心不是最优！"}')

    # --- 5. 贪心正确性总结 ---
    print('\n' + '=' * 60)
    print('总结：贪心算法的高效与局限')
    print('=' * 60)
    print('✓ 活动选择:  贪心 → 全局最优（可证）')
    print('✓ 分数背包:  贪心 → 全局最优（可证）')
    print('✓ Huffman:   贪心 → 最优前缀码（可证）')
    print('✓ 找零(规范面额): 贪心 → 全局最优（可证）')
    print('✗ 找零(非规范):   贪心 → 可能非最优，需用 DP')
    print('✗ 0-1背包:  贪心 → 通常非最优，需用 DP')


if __name__ == '__main__':
    main()
