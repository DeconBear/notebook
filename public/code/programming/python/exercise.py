# -*- coding: utf-8 -*-
"""
=== Python 基础 练习 ===
完成 Greeter.hello，以及 mean_of_evens。
运行: python exercise.py
"""


class Greeter:
    def __init__(self, name: str):
        self.name = name

    def hello(self) -> str:
        # TODO: 返回「你好, {name}」（中文逗号+空格）
        raise NotImplementedError


def mean_of_evens(xs):
    """只对偶数求平均；没有偶数时返回 0.0。"""
    # TODO
    raise NotImplementedError


def _check():
    g = Greeter('世界')
    assert g.hello() == '你好, 世界', g.hello()
    g2 = Greeter('PyTorch')
    assert g2.hello() == '你好, PyTorch'
    assert g is not g2, '两个实例必须是不同对象'
    assert mean_of_evens([1, 2, 3, 4]) == 3.0
    assert mean_of_evens([1, 3]) == 0.0
    print('练习通过。类是图纸：Greeter 一份，g 和 g2 是两台机器。')


if __name__ == '__main__':
    _check()
