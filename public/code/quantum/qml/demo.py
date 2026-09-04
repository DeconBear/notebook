# -*- coding: utf-8 -*-
"""
=== 量子机器学习 ===
1) 同任务经典基线：附带 16x16、数字 3 vs 6 的小 CNN（必跑）
2) 原对照路线：torchvision MNIST 上的 LeNet-5，28x28、数字 0 vs 1（下载失败则跳过）
3) VQNet 混合模型：若已安装 pyvqnet，做一次前向冒烟；完整训练请运行 train.py
运行: python demo.py
VQNet 核心与数据结构收编自独立示例工程 qml-mnist-classify（MIT，现随本章发布）。
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from lenet import train_lenet

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
_DATA_DIR = os.path.join(_SCRIPT_DIR, 'dataset')
os.makedirs(_IMAGES_DIR, exist_ok=True)
torch.manual_seed(42)
np.random.seed(42)


def load_npz(split):
    name = 'mnist_train_1000_16_16.npz' if split == 'train' else 'mnist_test_200_16_16.npz'
    blob = np.load(os.path.join(_DATA_DIR, name))
    x = blob['data'].astype(np.float32)
    if x.max() > 1.0:
        x = x / 255.0
    y = blob['label'].astype(np.int64)
    return x, y


class TinyCNN(nn.Module):
    """与量子混合模型同一套 16x16 / 3 vs 6 数据上的小卷积基线。"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc = nn.Linear(16 * 4 * 4, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        return self.fc(x.flatten(1))


def train_classical(epochs=8):
    x_tr, y_tr = load_npz('train')
    x_te, y_te = load_npz('test')
    train_ds = TensorDataset(torch.from_numpy(x_tr).unsqueeze(1), torch.from_numpy(y_tr))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    model = TinyCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    hist = []
    for ep in range(epochs):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for xb, yb in train_loader:
            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += float(loss.detach()) * len(yb)
            correct += int((logits.argmax(1) == yb).sum())
            total += len(yb)
        hist.append(loss_sum / total)
        print(f'[16x16 CNN] epoch {ep+1}/{epochs}  acc={correct/total:.3f}  loss={hist[-1]:.4f}')
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(x_te).unsqueeze(1)).argmax(1).numpy()
    test_acc = float((pred == y_te).mean())
    print(f'[16x16 CNN] 测试准确率 {test_acc:.3f}（与量子混合模型同一任务）')
    return hist, test_acc, x_te, y_te, pred


def try_vqnet_smoke():
    try:
        import pyvqnet  # noqa: F401
    except ImportError:
        print('-' * 60)
        print('未检测到 pyvqnet，跳过 VQNet 混合模型（这是预期行为）。')
        print('VQNet 不是 pip 默认依赖。安装见：')
        print('  https://qcloud.originqc.com.cn/zh/programming/VQNet')
        print('完整训练/评估脚本：本目录 train.py / eval.py')
        print('-' * 60)
        return False
    sys.path.insert(0, _SCRIPT_DIR)
    os.chdir(_SCRIPT_DIR)
    from qml_core import QuantumImageClassifier, get_default_spec, load_test_dataset
    from pyvqnet import QTensor

    spec = get_default_spec()
    model = QuantumImageClassifier(spec.model)
    x_test, y_test = load_test_dataset(_SCRIPT_DIR)
    xb = QTensor(x_test[:8])
    logits = model(xb)
    print('[VQNet] 冒烟通过：8 个测试样本前向输出形状', getattr(logits, 'shape', type(logits)))
    print('[VQNet] 完整训练: python train.py   评估: python eval.py')
    return True


def _tile(images, k=12):
    tiles = [images[i] for i in range(min(k, len(images)))]
    while len(tiles) < 12:
        tiles.append(np.zeros_like(tiles[0]))
    return np.concatenate(
        [np.concatenate(tiles[0:6], axis=1), np.concatenate(tiles[6:12], axis=1)],
        axis=0,
    )


def visualize(cnn, lenet_pack):
    n = 2 if lenet_pack is None else 3
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.7))
    if n == 2:
        axes = list(axes)

    hist, test_acc, x_te, y_te, pred = cnn
    axes[0].plot(hist, color='#2563eb')
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('交叉熵')
    axes[0].set_title(f'16x16 CNN  3vs6  acc={test_acc:.2f}')

    axes[1].axis('off')
    axes[1].set_title('同任务样张（真值→预测）')
    axes[1].imshow(_tile(x_te), cmap='gray')
    labels = [f'{int(y)}->{int(p)}' for y, p in zip(y_te[:12], pred[:12])]
    axes[1].set_xlabel('  '.join(labels[:6]) + '\n' + '  '.join(labels[6:]))

    if lenet_pack is not None:
        lhist, lacc, lx, ly, lp = lenet_pack
        axes[2].axis('off')
        axes[2].set_title(f'LeNet-5  0vs1  acc={lacc:.2f}')
        axes[2].imshow(_tile(lx), cmap='gray')
        ll = [f'{int(y)}->{int(p)}' for y, p in zip(ly[:12], lp[:12])]
        axes[2].set_xlabel('  '.join(ll[:6]) + '\n' + '  '.join(ll[6:]))

    fig.tight_layout()
    path = os.path.join(_IMAGES_DIR, 'qml_classical_baseline.png')
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f'已保存 {path}')


if __name__ == '__main__':
    cnn_pack = train_classical()
    lenet_pack = train_lenet()
    visualize(cnn_pack, lenet_pack)
    try_vqnet_smoke()
    print('完成：同任务小 CNN + LeNet-5 对照；量子混合模型视 pyvqnet 而定。')
