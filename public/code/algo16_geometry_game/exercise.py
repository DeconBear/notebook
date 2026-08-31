# -*- coding: utf-8 -*-
"""
algo16 计算几何与博弈论入门 — 练习代码
======================================
请完成以下 TODO 任务。
"""


# ============================================================================
# TODO 1: 计算多边形面积（使用叉积）
# ============================================================================
def polygon_area(polygon):
    """
    使用 Shoelace 公式（叉积法）计算多边形面积。
    公式: Area = 0.5 * |sum(x_i * y_{i+1} - x_{i+1} * y_i)|
    顶点按逆时针或顺时针顺序给出均可。

    参数:
        polygon: list of (x, y)，按顺序排列的多边形顶点
    返回:
        多边形面积（float）
    """
    # TODO: 实现
    # n = len(polygon)
    # area = 0
    # for i in range(n):
    #     x1, y1 = polygon[i]
    #     x2, y2 = polygon[(i + 1) % n]
    #     area += x1 * y2 - x2 * y1
    # return abs(area) / 2.0
    return 0


# ============================================================================
# TODO 2: 判断两点是否在直线的同一侧
# ============================================================================
def same_side(line_start, line_end, p1, p2):
    """
    判断点 p1 和 p2 是否在直线 line_start→line_end 的同一侧。
    使用叉积的符号比较。

    返回: True 表示同一侧（或不严格地，至少一个在直线上）
    """
    # TODO: 计算 cross(line_start, line_end, p1) 和 cross(line_start, line_end, p2)
    # 如果两者的乘积 >= 0（同号或至少一个为0），则在同一侧
    return False  # TODO


# ============================================================================
# TODO 3: Nim 游戏的三堆必胜策略完整分析
# ============================================================================
def nim_analyze_all(a_max):
    """
    枚举 Nim 游戏的所有三堆局面 (a,b,c)，其中 0 <= a,b,c <= a_max。
    统计先手必胜局面的比例。

    参数:
        a_max: 每堆的最大石子数
    返回:
        必胜局面的数量、总局面数和比例
    """
    # TODO: 实现
    # win_count = 0
    # total = (a_max + 1) ** 3
    # for a, b, c in itertools.product(range(a_max+1), repeat=3):
    #     if (a ^ b ^ c) != 0:
    #         win_count += 1
    # return win_count, total, win_count / total
    return 0, 0, 0


# ============================================================================
# TODO 4: Minimax 扩展到带 Alpha-Beta 剪枝的版本
# ============================================================================
class MinimaxWithABPruning:
    """
    在已有的 Minimax 基础上添加 Alpha-Beta 剪枝。
    用于更高效的博弈树搜索。
    """
    def __init__(self):
        # 假设一个简化的游戏状态
        pass

    def minimax_ab(self, depth, alpha, beta, is_maximizing):
        """
        Minimax with Alpha-Beta pruning.

        alpha: 当前 MAX 玩家能够保证的最低分数
        beta: 当前 MIN 玩家能够保证的最高分数
        当 alpha >= beta 时剪枝。

        返回 (分数, 最佳走法)
        """
        # TODO: 实现 alpha-beta 剪枝版本的 minimax
        return 0, None


# ============================================================================
# TODO 5: 使用叉积实现"判断点是否在三角形内"
# ============================================================================
def point_in_triangle(p, a, b, c):
    """
    判断点 p 是否在三角形 abc 内。
    方法：p 必须与 a,b,c 中任意两点的连线构成的三个小三角形有相同符号的叉积。

    参数:
        p: 待判断的点 (x, y)
        a, b, c: 三角形三个顶点
    返回:
        True 如果 p 在三角形内（含边界）
    """
    # TODO: 计算三个叉积: cross(a,b,p), cross(b,c,p), cross(c,a,p)
    # 如果三者同号（都 >=0 或都 <=0），则 p 在三角形内
    return False  # TODO


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == '__main__':
    print('=' * 50)
    print('algo16 计算几何与博弈论入门 — 练习')
    print('=' * 50)

    # 测试 TODO 1: 多边形面积
    print('\n--- TODO 1: 多边形面积（Shoelace 公式）---')
    square = [(0, 0), (4, 0), (4, 3), (0, 3)]
    area = polygon_area(square)
    print(f'  矩形面积: {area} (期望: 12)')

    triangle = [(0, 0), (4, 0), (0, 3)]
    area2 = polygon_area(triangle)
    print(f'  三角形面积: {area2} (期望: 6)')

    # 测试 TODO 2: 同侧判断
    print('\n--- TODO 2: 两点是否在直线同侧 ---')
    A, B = (0, 0), (4, 0)
    p1, p2 = (2, 3), (2, -1)
    print(f'  A={A}, B={B}, p1={p1}, p2={p2} → {same_side(A, B, p1, p2)} '
          f'(期望: False, 两点在直线两侧)')

    # 测试 TODO 3: Nim 分析
    print('\n--- TODO 3: Nim 三堆必胜率 ---')
    wins, total, ratio = nim_analyze_all(5)
    print(f'  a_max=5: 必胜 {wins}/{total} = {ratio:.1%}')

    # 测试 TODO 4
    print('\n--- TODO 4: Alpha-Beta 剪枝 ---')
    print('  (待实现)')

    # 测试 TODO 5
    print('\n--- TODO 5: 点在三角形内 ---')
    tri = ((0, 0), (4, 0), (2, 3))
    test_pts = [(2, 1), (5, 1), (2, 0)]
    for pt in test_pts:
        in_tri = point_in_triangle(pt, *tri)
        print(f'  {pt} in {tri}? {in_tri}')

    print('\n提示: 请补全上方所有 TODO 函数后再运行测试。')
