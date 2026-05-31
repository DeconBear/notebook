# -*- coding: utf-8 -*-
"""
algo16 计算几何与博弈论入门 — 演示代码
======================================
功能：叉积/点积、线段相交检测、点在多边形内（Ray Casting）、
      Graham Scan 凸包、最近点对（分治）、
      Nim 游戏求解器、SG 函数、Minimax 井字棋。

每个函数都有中文 docstring，每行逻辑代码都有中文注释。
运行方式：在 algo16_geometry_game/ 目录下执行 python code/demo.py
"""

import os
import math
import random
import itertools
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False

_HERE = os.path.dirname(os.path.abspath(__file__))
_IMAGES = os.path.join(_HERE, '..', 'images')
os.makedirs(_IMAGES, exist_ok=True)


# ============================================================================
# 第一部分：计算几何 — 基础操作
# ============================================================================

def cross_product(o, a, b):
    """
    叉积：计算向量 OA × OB。
    返回 >0: 逆时针（左转）, <0: 顺时针（右转）, =0: 共线。
    """
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def dot_product(o, a, b):
    """点积：向量 OA · OB。"""
    return (a[0] - o[0]) * (b[0] - o[0]) + (a[1] - o[1]) * (b[1] - o[1])


def distance2(a, b):
    """两点间欧几里得距离的平方。"""
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def on_segment(p, a, b):
    """判断点 p 是否在线段 ab 上（假设 p 与 a,b 共线）。"""
    return (min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and
            min(a[1], b[1]) <= p[1] <= max(a[1], b[1]))


def segments_intersect(p1, p2, q1, q2):
    """
    判断两条线段 P1P2 和 Q1Q2 是否相交。
    使用两次跨越测试（orientation test）。
    """
    d1 = cross_product(q1, q2, p1)
    d2 = cross_product(q1, q2, p2)
    d3 = cross_product(p1, p2, q1)
    d4 = cross_product(p1, p2, q2)

    # 一般情况：P1和P2在Q1Q2两侧，且Q1和Q2在P1P2两侧
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    # 边界情况：共线且投影区间重叠
    if d1 == 0 and on_segment(p1, q1, q2): return True
    if d2 == 0 and on_segment(p2, q1, q2): return True
    if d3 == 0 and on_segment(q1, p1, p2): return True
    if d4 == 0 and on_segment(q2, p1, p2): return True

    return False


def point_in_polygon(point, polygon):
    """
    Ray Casting 算法：判断点是否在多边形内部。
    从点向右发射水平射线，统计与多边形边的交点数。
    奇数 → 在内部，偶数 → 在外部。
    """
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        # 检查射线是否与边相交（排除水平边和端点重合）
        if ((yi > y) != (yj > y)) and \
           (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside  # 每相交一次翻转状态
        j = i
    return inside


# ============================================================================
# 第二部分：Graham Scan 凸包
# ============================================================================

def graham_scan(points):
    """
    Graham Scan 算法求凸包。
    返回凸包上的点（按逆时针顺序）。
    """
    if len(points) <= 2:
        return points[:]

    # 1. 找最左下角的点（y 最小，y 相同时 x 最小）
    p0 = min(points, key=lambda p: (p[1], p[0]))

    # 2. 按极角排序
    def polar_angle(p):
        dx, dy = p[0] - p0[0], p[1] - p0[1]
        return math.atan2(dy, dx)

    sorted_points = sorted(points, key=lambda p:
                           (polar_angle(p), distance2(p0, p)))

    # 3. 栈构建凸包
    hull = [p0]
    for p in sorted_points[1:]:
        # 如果新点与前两个点构成右转（非凸）→ 弹出中间点
        while len(hull) >= 2 and cross_product(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        hull.append(p)

    return hull


# ============================================================================
# 第三部分：最近点对（分治法）
# ============================================================================

def closest_pair(points):
    """
    分治法求最近点对的距离。
    返回 (最小距离, (点1, 点2))。

    分治步骤：
    1. 按 x 坐标排序
    2. 中线分左右
    3. 递归求左右最近距离 d
    4. 检查跨中线条带内距离 < d 的点对
    """
    px = sorted(points, key=lambda p: p[0])
    py = sorted(points, key=lambda p: p[1])

    def _closest(px_sorted):
        n = len(px_sorted)
        if n <= 3:
            # 暴力比较
            min_dist2 = float('inf')
            best_pair = None
            for i in range(n):
                for j in range(i + 1, n):
                    d2 = distance2(px_sorted[i], px_sorted[j])
                    if d2 < min_dist2:
                        min_dist2 = d2
                        best_pair = (px_sorted[i], px_sorted[j])
            return min_dist2, best_pair

        mid = n // 2
        mid_x = px_sorted[mid][0]

        # 递归
        left_d2, left_pair = _closest(px_sorted[:mid])
        right_d2, right_pair = _closest(px_sorted[mid:])

        d2 = min(left_d2, right_d2)
        best_pair = left_pair if left_d2 <= right_d2 else right_pair

        # 条带内的点：|x - mid_x| <= sqrt(d2)
        d = math.sqrt(d2)
        strip = [p for p in px_sorted if abs(p[0] - mid_x) <= d]
        strip.sort(key=lambda p: p[1])  # 按 y 坐标排序

        # 检查条带内的点对（每个点最多检查 7 个后继）
        for i in range(len(strip)):
            j = i + 1
            while j < len(strip) and (strip[j][1] - strip[i][1]) <= d:
                curr_d2 = distance2(strip[i], strip[j])
                if curr_d2 < d2:
                    d2 = curr_d2
                    best_pair = (strip[i], strip[j])
                j += 1

        return d2, best_pair

    d2, pair = _closest(px)
    return math.sqrt(d2), pair


# ============================================================================
# 第四部分：Nim 游戏求解器
# ============================================================================

def nim_solve(piles):
    """
    Nim 游戏求解器。
    返回 (先手是否能赢, 必胜第一步操作)。
    必胜第一步：将某个 pile 减少为 pile ^ xor_sum。
    """
    xor_sum = 0
    for p in piles:
        xor_sum ^= p

    if xor_sum == 0:
        return False, None  # 先手必败（前提是对手也最优）

    # 找必胜操作
    for i, p in enumerate(piles):
        target = p ^ xor_sum
        if target < p:  # 取走 (p - target) 个 → 新 piles[i] = target
            new_piles = piles[:]
            new_piles[i] = target
            return True, (f'从第{i+1}堆取走{p - target}个，剩{target}个', new_piles)

    return True, None  # 理论上总是能找到


def mex(s):
    """计算集合 s 的 mex（最小未出现的非负整数）。"""
    m = 0
    while m in s:
        m += 1
    return m


def sg_function(stones, moves):
    """
    计算 SG 函数值（对于"每次取 moves 中的某个数量"的取石子游戏）。
    stones: 当前石子数
    moves: 允许的一次操作可取走的石子数列表
    返回 SG 值和所有状态的 SG 数组。
    """
    sg = [0] * (stones + 1)
    for i in range(1, stones + 1):
        reachable_sg = set()
        for m in moves:
            if i >= m:
                reachable_sg.add(sg[i - m])
        sg[i] = mex(reachable_sg)
    return sg[stones], sg


# ============================================================================
# 第五部分：Minimax 井字棋
# ============================================================================

class TicTacToe:
    """井字棋游戏，演示 Minimax 算法。"""

    def __init__(self):
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.ai_player = 'O'
        self.human_player = 'X'

    def is_full(self):
        return all(self.board[i][j] != ' ' for i in range(3) for j in range(3))

    def check_winner(self):
        """检查是否有人获胜。返回获胜者或 None。"""
        lines = []
        # 行和列
        for i in range(3):
            lines.append([self.board[i][j] for j in range(3)])
            lines.append([self.board[j][i] for j in range(3)])
        # 对角线
        lines.append([self.board[i][i] for i in range(3)])
        lines.append([self.board[i][2 - i] for i in range(3)])
        for line in lines:
            if line[0] != ' ' and line[0] == line[1] == line[2]:
                return line[0]
        return None

    def get_empty_cells(self):
        """获取所有空格的位置列表。"""
        return [(i, j) for i in range(3) for j in range(3)
                if self.board[i][j] == ' ']

    def minimax(self, is_maximizing, depth=0):
        """
        Minimax 算法：为 AI（O）选择最优走法。
        AI (O) 是 maximizing player，人类 (X) 是 minimizing player。
        返回 (分数, 最佳走法)。
        """
        winner = self.check_winner()
        if winner == self.ai_player:
            return 10 - depth, None  # AI 赢，越早越好
        elif winner == self.human_player:
            return depth - 10, None  # 人类赢，越晚越好
        elif self.is_full():
            return 0, None  # 平局

        empty = self.get_empty_cells()
        best_move = empty[0]

        if is_maximizing:  # AI 的回合
            best_score = float('-inf')
            for i, j in empty:
                self.board[i][j] = self.ai_player
                score, _ = self.minimax(False, depth + 1)
                self.board[i][j] = ' '  # 回溯
                if score > best_score:
                    best_score = score
                    best_move = (i, j)
            return best_score, best_move
        else:  # 人类回合（对手最优）
            best_score = float('inf')
            for i, j in empty:
                self.board[i][j] = self.human_player
                score, _ = self.minimax(True, depth + 1)
                self.board[i][j] = ' '  # 回溯
                if score < best_score:
                    best_score = score
                    best_move = (i, j)
            return best_score, best_move

    def get_best_move(self):
        """获取 AI 的最佳走法。"""
        _, move = self.minimax(True)
        return move


# ============================================================================
# 第六部分：可视化
# ============================================================================

def visualize_convex_hull(points, hull):
    """可视化凸包。"""
    fig, ax = plt.subplots(figsize=(8, 8))

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.scatter(xs, ys, c='#42A5F5', s=60, zorder=2, label='Points')

    # 绘制凸包多边形（闭合）
    hull_xs = [p[0] for p in hull] + [hull[0][0]]
    hull_ys = [p[1] for p in hull] + [hull[0][1]]
    ax.plot(hull_xs, hull_ys, 'o-', color='#D32F2F', linewidth=2.5,
            markersize=8, zorder=3, label='Convex Hull')

    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('Graham Scan Convex Hull', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(_IMAGES, 'algo16_convex_hull.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[可视化] 凸包图已保存: {path}')


# ============================================================================
# 主程序
# ============================================================================

def main():
    print('=' * 60)
    print('algo16 计算几何与博弈论入门 — 演示')
    print('=' * 60)

    # --- 1. 叉积与方向判断 ---
    print('\n### 1. 向量叉积 — 方向判断 ###')
    A, B, C = (0, 0), (1, 0), (0.5, 0.8)
    print(f'  A={A}, B={B}, C={C}')
    print(f'  cross(A,B,C) = {cross_product(A, B, C):.1f} '
          f'({"左转" if cross_product(A, B, C) > 0 else "右转/共线"})')

    # --- 2. 线段相交 ---
    print('\n### 2. 线段相交检测 ###')
    tests = [
        ((0, 0), (1, 1), (0, 1), (1, 0)),   # 相交 (X型)
        ((0, 0), (0.5, 0.5), (0, 1), (1, 0)),  # 不相交 (T型接触在端点不穿)
        ((0, 0), (1, 1), (2, 2), (3, 3)),   # 共线不相交
    ]
    for p1, p2, q1, q2 in tests:
        result = segments_intersect(p1, p2, q1, q2)
        print(f'  {p1}-{p2} vs {q1}-{q2}: {"相交" if result else "不相交"}')

    # --- 3. 点在多边形内 ---
    print('\n### 3. 点在多边形内（Ray Casting）###')
    polygon = [(0, 0), (4, 0), (4, 3), (2, 4), (0, 3)]
    test_points = [(2, 1.5), (5, 2), (0, 1.5)]
    for pt in test_points:
        inside = point_in_polygon(pt, polygon)
        print(f'  点 {pt} 是否在多边形内: {inside}')

    # --- 4. Graham Scan 凸包 ---
    print('\n### 4. Graham Scan 凸包 ###')
    random.seed(42)
    points = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(30)]
    hull = graham_scan(points)
    print(f'  随机点数量: {len(points)}')
    print(f'  凸包顶点数: {len(hull)}')
    print(f'  凸包顶点: {[(round(x,1), round(y,1)) for x,y in hull]}')
    visualize_convex_hull(points, hull)

    # --- 5. 最近点对 ---
    print('\n### 5. 最近点对（分治法）###')
    small_points = [(0, 0), (1, 0), (0.5, 0.1), (2, 2), (3, 3.1)]
    dist, pair = closest_pair(small_points)
    print(f'  点集: {small_points}')
    print(f'  最近距离: {dist:.4f}, 点对: {pair}')

    # --- 6. Nim 游戏 ---
    print('\n### 6. Nim 游戏求解 ###')
    test_piles = [
        [3, 4, 5],    # 3⊕4⊕5 = 2 ≠ 0 → 先手胜
        [1, 2, 3],    # 1⊕2⊕3 = 0 → 先手败
        [2, 2],       # 2⊕2 = 0 → 先手败
    ]
    for piles in test_piles:
        can_win, strategy = nim_solve(piles)
        xor_s = 0
        for p in piles:
            xor_s ^= p
        print(f'  堆={piles}, XOR={xor_s}, 先手{"必胜" if can_win else "必败"}')
        if strategy:
            print(f'    必胜操作: {strategy[0]}')

    # --- 7. SG 函数 ---
    print('\n### 7. SG 函数（取石子游戏）###')
    moves = [1, 3, 4]  # 每次可取 1, 3 或 4 个
    for stones in range(11):
        sg_val, _ = sg_function(stones, moves)
        status = '必败 (P)' if sg_val == 0 else '必胜 (N)'
        print(f'  石子={stones} -> SG={sg_val} ({status})')

    # --- 8. Minimax 井字棋 ---
    print('\n### 8. Minimax — 井字棋 AI 走法演示 ###')
    game = TicTacToe()
    # 设置一个中间局面
    game.board = [
        ['X', ' ', ' '],
        [' ', 'O', ' '],
        [' ', ' ', 'X']
    ]
    best = game.get_best_move()
    print(f'  当前棋盘 (O=AI, X=人类):')
    for row in game.board:
        print(f'    {row}')
    print(f'  AI 最佳走法: {best}')

    print('\n' + '=' * 60)
    print('总结：叉积/凸包/最近点对/Nim/SG函数/Minimax 覆盖几何与博弈核心')
    print('=' * 60)


if __name__ == '__main__':
    main()
