# -*- coding: utf-8 -*-
"""
=== AlphaGo 练习：实现 PUCT 分数 ===
补全 puct_score。公式:
  PUCT = Q + c * P * sqrt(N_parent) / (1 + N_child)
运行: python exercise.py
"""
import math


def puct_score(q, prior, n_parent, n_child, c=1.5):
    """
    TODO: 返回 PUCT。n_parent 是父节点访问次数，n_child 是该边访问次数。
    """
    # TODO: 用正文里的 PUCT 公式替换下一行
    return 0.0


def _run_checks():
    unvisited = puct_score(q=0.0, prior=0.3, n_parent=100, n_child=0, c=1.5)
    visited = puct_score(q=0.2, prior=0.3, n_parent=100, n_child=20, c=1.5)
    expected_u = 1.5 * 0.3 * math.sqrt(100) / (1 + 0)
    assert abs(unvisited - expected_u) < 1e-6, f'未访问边应只有探索项, 得到 {unvisited}'
    expected = 0.2 + 1.5 * 0.3 * math.sqrt(100) / 21
    assert abs(visited - expected) < 1e-6, f'访问过的边 Q+U, 得到 {visited}'
    assert unvisited > visited, '同样先验下，未访问边的 PUCT 应更高（鼓励探索）'
    print('PUCT 练习通过。')


if __name__ == '__main__':
    _run_checks()
