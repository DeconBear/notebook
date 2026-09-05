---
title: "量子机器学习 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 量子机器学习 — demo.py 代码详解

<a href="/notebook/code/quantum/qml/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/quantum/qml/code
python demo.py
```

默认路径**不训练量子电路**。必跑：同目录 `dataset/*.npz` 上的 16×16、数字 3 vs 6 小 CNN。可选：`from lenet import train_lenet` 的 28×28、0 vs 1 LeNet-5（MNIST 下载失败则跳过）。`pyvqnet` 未装时打印说明并返回 `False`（退出码仍 0）。完整量子训练是同目录 `train.py` / `eval.py`，不是本文件。图：`qml_classical_baseline.png`。

## 代码逐段详解

### 第1步：`load_npz` — 与量子混合模型同一套数据

```python
name = 'mnist_train_1000_16_16.npz' if split == 'train' else 'mnist_test_200_16_16.npz'
x = blob['data'].astype(np.float32)
if x.max() > 1.0:
    x = x / 255.0
y = blob['label'].astype(np.int64)
```

- **文件名写死**：1000 张训练、200 张测试，边长 16。没有 torchvision 下载这一路。
- **`max()>1` 才除 255**：已在 $[0,1]$ 的数组不要再除。
- 标签是 0/1（任务是 3 vs 6 的二分类编码），后面 `CrossEntropy` 要 `int64`。

---

### 第2步：`TinyCNN` — 两层池化接到 2 类

```python
x = F.relu(F.max_pool2d(self.conv1(x), 2))
x = F.relu(F.max_pool2d(self.conv2(x), 2))
return self.fc(x.flatten(1))
```

- **`Conv2d(1,8,3,padding=1)`**：16×16 尺寸不变，再 `max_pool2d(...,2)` → 8×8。第二层 8→16 通道再池化 → **4×4**。所以全连接是 `Linear(16*4*4, 2)`。
- **`flatten(1)`**：从第 1 维起展平，留下 batch。等价 `view(B,-1)`。
- 没有 Dropout、没有 BN。8 epoch、Adam `1e-3`、batch 64。

`unsqueeze(1)`：NumPy `(N,16,16)` 加通道维变成 `(N,1,16,16)`，`Conv2d` 才吃得下。

---

### 第3步：`train_classical` 记损失与测试准确率

```python
logits = model(xb)
loss = F.cross_entropy(logits, yb)
loss_sum += float(loss.detach()) * len(yb)
correct += int((logits.argmax(1) == yb).sum())
```

- **`cross_entropy`**：内部 Softmax + NLL。标签是类下标，不是 one-hot。
- **`loss.detach() * len(yb)`**：把「batch 平均损失」还原成总和，epoch 结束再 `/ total`，最后一小批大小不同时才公平。
- **`argmax(1)`**：沿类别维。测试循环包在 `eval()` + `no_grad()`。返回 `(hist, test_acc, x_te, y_te, pred)` 给可视化。

这是量子混合模型的**经典基线**，不是「量子准确率」。

---

### 第4步：`try_vqnet_smoke` — 缺库就跳过

```python
try:
    import pyvqnet
except ImportError:
    print('未检测到 pyvqnet，跳过 VQNet 混合模型（这是预期行为）。')
    return False
```

装上之后才 `from qml_core import QuantumImageClassifier, get_default_spec, load_test_dataset`，取 8 个测试样本 `QTensor(x_test[:8])` 做一次 `model(xb)`，打印 `shape`。**没有** `loss.backward()`。`os.chdir(_SCRIPT_DIR)` 是为了让 `qml_core` 找到相对数据路径。

不要把冒烟当成训练曲线。

---

### 第5步：`_tile` 与 `visualize`

```python
return np.concatenate(
    [np.concatenate(tiles[0:6], axis=1), np.concatenate(tiles[6:12], axis=1)],
    axis=0,
)
```

12 张样例：上 6 张横向拼，下 6 张再横向拼，最后纵向拼成 2×6 接触印。不足 12 张用全零垫。`imshow(..., cmap='gray')`。横轴文字是 `'真值->预测'`。

子图数：`lenet_pack is None` 时 2 列（损失曲线 + 16×16 样张）；LeNet 成功则 3 列，第三列是 0 vs 1 的接触印和准确率。`train_lenet` 定义在 `lenet.py`，下载失败返回 `None`，demo 仍保存 CNN 那两格。

`TensorDataset(torch.from_numpy(x_tr).unsqueeze(1), torch.from_numpy(y_tr))`：NumPy 进 Tensor 后再包数据集。`DataLoader(..., shuffle=True)` 每个 epoch 打乱；测试不用 Loader，整份 `x_te` 一次前向。

`visualize` 里 `if n == 2: axes = list(axes)`：两子图时 `subplots` 返回的 `axes` 已经是数组，转成 list 只为和三子图分支同一套 `axes[i]` 下标。`labels[:6]` / `[6:]` 两行 xlabel，对应接触印上下两排。

入口顺序：`train_classical` → `train_lenet` → `visualize` → `try_vqnet_smoke`。种子 `torch.manual_seed(42)` 与 `np.random.seed(42)` 都设了；CNN 可复现，LeNet 若走下载路径还依赖 torchvision 打乱。

---

### 关键概念速查表

| 概念 | 直觉 | 代码 |
|------|------|------|
| 同任务数据 | 16×16，3 vs 6 | `load_npz` / `dataset/*.npz` |
| `TinyCNN` | 两层 conv+pool → 2 类 | `Linear(16*4*4, 2)` |
| `unsqueeze(1)` | 加通道 | `Conv2d` 输入 |
| 交叉熵 | 分类 | `F.cross_entropy` |
| LeNet 对照 | 28×28，0 vs 1 | `train_lenet()`（可跳过） |
| VQNet | 可选依赖 | `try_vqnet_smoke` |
| `_tile` | 2×6 接触印 | `concatenate` |
| 量子训练 | 不在 demo 主路径 | `train.py` / `eval.py` |

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/quantum/qml/code/demo.py`
