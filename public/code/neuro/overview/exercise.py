# -*- coding: utf-8 -*-
"""
=== 导论练习 ===
把下列现象标到正确的尺度：membrane / synapse / circuit / connectome / neuroai
运行: python exercise.py
"""

PAIRS = [
    ('动作电位的钠钾门控', 'membrane'),
    ('STDP 改突触权重', 'synapse'),
    ('E–I 网络出现同步齐射', 'circuit'),
    ('FlyWire 给出谁连谁', 'connectome'),
    ('CNN 层与腹侧流做 RSA', 'neuroai'),
]


def label_scale(phenomenon):
    """返回 'membrane' | 'synapse' | 'circuit' | 'connectome' | 'neuroai'。"""
    # TODO: 按现象选择尺度
    raise NotImplementedError


def _check():
    for name, ans in PAIRS:
        got = label_scale(name)
        assert got == ans, f'{name}: 期望 {ans}，得到 {got}'
    print('通过：尺度标注正确。')


if __name__ == '__main__':
    _check()
