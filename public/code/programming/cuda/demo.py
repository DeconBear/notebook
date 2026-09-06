# -*- coding: utf-8 -*-
"""
=== CUDA 与设备 ===
探测 CPU / CUDA / MPS，搬一次张量，有 GPU 时对比一次矩阵乘。
没有 GPU 也会正常结束。
运行: python demo.py
"""
import os
import time
import torch
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)


def pick_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def demo_move():
    device = pick_device()
    print('选用 device =', device)
    print('torch.cuda.is_available() =', torch.cuda.is_available())
    x_cpu = torch.randn(4, 4)
    x = x_cpu.to(device)
    y = x @ x
    y_back = y.detach().cpu()
    print('运算在', y.device, '  搬回 CPU 后', y_back.device)
    return device


def demo_timing(device):
    n = 512
    a = torch.randn(n, n)
    b = torch.randn(n, n)
    t0 = time.perf_counter()
    _ = a @ b
    t_cpu = time.perf_counter() - t0
    times = {'CPU': t_cpu}
    if device.type == 'cuda':
        ac, bc = a.to(device), b.to(device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = ac @ bc
        torch.cuda.synchronize()
        times['CUDA'] = time.perf_counter() - t0
        print('512 矩阵乘  CPU %.4fs  CUDA %.4fs（含少量启动开销）' % (times['CPU'], times['CUDA']))
    else:
        print('无 CUDA，只报 CPU 矩阵乘 %.4fs（小矩阵看不出加速）' % t_cpu)

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.bar(list(times.keys()), list(times.values()), color=['#5B8FF9', '#61DDAA'][:len(times)])
    ax.set_ylabel('秒')
    ax.set_title('同一句 a@b，设备不同（无 GPU 则只有 CPU）')
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'prog-cuda-device.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)


if __name__ == '__main__':
    dev = demo_move()
    demo_timing(dev)
