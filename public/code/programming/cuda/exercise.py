# -*- coding: utf-8 -*-
"""
=== CUDA 练习 ===
实现 choose_device()：有 CUDA 用 cuda，否则 cpu。不要写死 'cuda'。
运行: python exercise.py
"""
import torch


def choose_device() -> torch.device:
    # TODO: 返回 torch.device('cuda') 或 torch.device('cpu')
    raise NotImplementedError


def to_numpy_safe(t: torch.Tensor):
    """任意设备上的 Tensor → CPU numpy。"""
    # TODO
    raise NotImplementedError


def _check():
    d = choose_device()
    assert isinstance(d, torch.device)
    if torch.cuda.is_available():
        assert d.type == 'cuda'
    else:
        assert d.type == 'cpu'
    x = torch.tensor([1.0, 2.0])
    arr = to_numpy_safe(x)
    assert arr.tolist() == [1.0, 2.0]
    print('练习通过。device =', d, '  没有 GPU 时必须仍返回 cpu。')


if __name__ == '__main__':
    _check()
