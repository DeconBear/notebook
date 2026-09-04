# -*- coding: utf-8 -*-
"""
=== 经典 LeNet-5 对照 ===
结构对齐原独立示例：28×28 灰度、二分类 0 vs 1、Conv-Pool-FC。
数据改用 torchvision MNIST，不保存 PNG。
下载失败时返回 None，demo 跳过这一路。
运行: 由 demo.py 调用；也可 python lenet.py
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT = os.path.join(_SCRIPT_DIR, '_mnist_cache')


class LeNet5(nn.Module):
    """原示例 LeNet-5：1→6→16 卷积 + 400→120→84→2。"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Sequential(nn.Linear(16 * 5 * 5, 120), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(120, 84), nn.ReLU())
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)


def _filter_digits(dataset, digits=(0, 1)):
    idx = [i for i, y in enumerate(dataset.targets.tolist()) if y in digits]
    subset = Subset(dataset, idx)

    class Relabel(torch.utils.data.Dataset):
        def __len__(self):
            return len(subset)

        def __getitem__(self, i):
            img, y = subset[i]
            return img, digits.index(int(y))

    return Relabel()


def load_mnist_01(max_train=2500, max_test=500):
    """返回 (train_loader, test_loader, test_images_np, test_labels_np)；失败则 (None,)*4。"""
    tfm = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    try:
        os.makedirs(_DATA_ROOT, exist_ok=True)
        train_full = torchvision.datasets.MNIST(_DATA_ROOT, train=True, download=True, transform=tfm)
        test_full = torchvision.datasets.MNIST(_DATA_ROOT, train=False, download=True, transform=tfm)
    except Exception as exc:
        print(f'[LeNet] MNIST 下载失败（{exc}），跳过 28×28 对照。')
        return None, None, None, None

    train_ds = _filter_digits(train_full)
    test_ds = _filter_digits(test_full)
    g = torch.Generator().manual_seed(42)
    if max_train and len(train_ds) > max_train:
        perm = torch.randperm(len(train_ds), generator=g)[:max_train]
        train_ds = Subset(train_ds, perm.tolist())
    if max_test and len(test_ds) > max_test:
        perm = torch.randperm(len(test_ds), generator=g)[:max_test]
        test_ds = Subset(test_ds, perm.tolist())

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    xs, ys = [], []
    for img, y in test_ds:
        xs.append(img.squeeze(0).numpy())
        ys.append(int(y))
    x_te = np.stack(xs, axis=0)
    y_te = np.array(ys, dtype=np.int64)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    print(f'[LeNet] 0/1 子集  train={len(train_ds)}  test={len(test_ds)}')
    return train_loader, test_loader, x_te, y_te


def train_lenet(epochs=5):
    loaders = load_mnist_01()
    train_loader, test_loader, x_te, y_te = loaders
    if train_loader is None:
        return None
    model = LeNet5()
    opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    hist = []
    for ep in range(epochs):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for xb, yb in train_loader:
            opt.zero_grad()
            logits = model(xb)
            loss = nn.functional.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += float(loss.detach()) * len(yb)
            correct += int((logits.argmax(1) == yb).sum())
            total += len(yb)
        hist.append(loss_sum / max(total, 1))
        print(f'[LeNet-5] epoch {ep+1}/{epochs}  acc={correct/total:.3f}  loss={hist[-1]:.4f}')
    model.eval()
    pred = []
    with torch.no_grad():
        for xb, _ in test_loader:
            pred.append(model(xb).argmax(1).cpu().numpy())
    pred = np.concatenate(pred)
    test_acc = float((pred == y_te).mean())
    print(f'[LeNet-5] 测试准确率 {test_acc:.3f}（28x28，数字 0 vs 1）')
    return hist, test_acc, x_te, y_te, pred


if __name__ == '__main__':
    train_lenet()
