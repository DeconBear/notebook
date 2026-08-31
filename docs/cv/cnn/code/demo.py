# -*- coding: utf-8 -*-
"""
s10 CNN 核心原理 demo：从零实现 Conv2d 和 MaxPool2d
===================================================
使用纯 NumPy 实现 im2col 卷积、最大池化，并在 MNIST 上构建简单 CNN。
展示：卷积可视化、特征图可视化、感受野计算、参数量对比。

运行方式：python demo.py（从 s10_cnn_fundamentals/code/ 目录运行）
依赖：numpy, matplotlib（可选：scikit-learn 用于加载 MNIST）
"""

import numpy as np
from typing import Tuple, Optional
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免 GUI 依赖
import matplotlib.pyplot as plt
matplotlib.rcParams['axes.unicode_minus'] = False
import os

# 图片保存目录：固定为本章节的 images/ 目录（相对于本脚本的 ../images/）
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)

# ============================================================
# 第 1 部分：加载 MNIST 数据
# ============================================================

def load_mnist(data_dir: str = "../data") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    加载 MNIST 数据集（优先使用 sklearn，否则用本地 npz 或在线下载）

    参数:
        data_dir: 数据缓存目录
    返回:
        (X_train, y_train, X_test, y_test): 训练/测试图像和标签
    """
    os.makedirs(data_dir, exist_ok=True)
    cache_path = os.path.join(data_dir, "mnist.npz")

    # 尝试从缓存加载
    if os.path.exists(cache_path):
        print(f"从缓存加载 MNIST: {cache_path}")
        data = np.load(cache_path)
        return data["X_train"], data["y_train"], data["X_test"], data["y_test"]

    # 尝试使用 sklearn
    try:
        from sklearn.datasets import fetch_openml
        print("正在从 OpenML 下载 MNIST 数据集...")
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto")
        X = X.astype(np.float32) / 255.0  # 归一化到 [0, 1]
        y = y.astype(np.int64)

        # 划分训练/测试集（MNIST 前 60000 训练，后 10000 测试）
        X_train, X_test = X[:60000], X[60000:]
        y_train, y_test = y[:60000], y[60000:]

        # 缓存到本地
        np.savez_compressed(cache_path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        return X_train, y_train, X_test, y_test

    except ImportError:
        pass

    # 最后尝试从网络下载原始 MNIST
    try:
        from urllib import request
        import gzip

        def _download_mnist_files():
            """下载并解析原始 MNIST 二进制文件"""
            base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
            files = {
                "train_images": "train-images-idx3-ubyte.gz",
                "train_labels": "train-labels-idx1-ubyte.gz",
                "test_images": "t10k-images-idx3-ubyte.gz",
                "test_labels": "t10k-labels-idx1-ubyte.gz",
            }
            result = {}
            for key, fname in files.items():
                fpath = os.path.join(data_dir, fname)
                if not os.path.exists(fpath):
                    print(f"下载 {fname}...")
                    request.urlretrieve(base_url + fname, fpath)

                with gzip.open(fpath, "rb") as f:
                    if "labels" in key:
                        # 标签文件：跳过 8 字节头部
                        result[key] = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
                    else:
                        # 图像文件：跳过 16 字节头部
                        result[key] = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 784)

            return result

        data = _download_mnist_files()
        X_train = data["train_images"].astype(np.float32) / 255.0
        y_train = data["train_labels"].astype(np.int64)
        X_test = data["test_images"].astype(np.float32) / 255.0
        y_test = data["test_labels"].astype(np.int64)

        np.savez_compressed(cache_path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        return X_train, y_train, X_test, y_test

    except Exception as e:
        raise RuntimeError(f"无法加载 MNIST 数据集: {e}。请安装 scikit-learn: pip install scikit-learn") from e


# ============================================================
# 第 2 部分：Im2Col —— 卷积的矩阵乘法实现
# ============================================================

class Im2Col:
    """
    Im2Col + Col2Im 工具类

    Im2Col 将图像中每一个"卷积核覆盖的小块"展开成矩阵的一列（或一行），
    从而把卷积转化为矩阵乘法，利用高度优化的 BLAS/GEMM 库加速。
    """

    @staticmethod
    def im2col(x: np.ndarray, kernel_h: int, kernel_w: int,
               stride: int = 1, pad: int = 0) -> np.ndarray:
        """
        将图像/特征图张量展开为列矩阵（im2col）

        输入 x 形状为 (N, C, H, W)，其中：
            N: batch size（样本数）
            C: 输入通道数
            H: 输入高度（行数）
            W: 输入宽度（列数）

        参数:
            x: 输入张量，形状 (N, C, H, W)
            kernel_h: 卷积核高度
            kernel_w: 卷积核宽度
            stride: 步长
            pad: 填充大小

        返回:
            cols: 展开后的矩阵，形状 (N, C * kernel_h * kernel_w, H_out * W_out)
                  每一列（在最后一维）是一个卷积位置的展开 patch
        """
        N, C, H, W = x.shape

        # ---------- 计算输出尺寸 ----------
        H_out = (H + 2 * pad - kernel_h) // stride + 1  # 输出高度
        W_out = (W + 2 * pad - kernel_w) // stride + 1  # 输出宽度

        # ---------- Padding：在 H 和 W 维度前后各加 pad 层 0 ----------
        if pad > 0:
            # np.pad: ((前,后), (前,后)) 针对 H, W 维度
            x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)),
                              mode='constant', constant_values=0)
        else:
            x_padded = x

        # ---------- 使用 as_strided 高效提取所有 patches ----------
        # as_strided 通过修改步长（stride）来"查看"同一个内存区域的不同视角，
        # 避免了显式循环，速度极快但需要小心使用
        shape = (N, C, H_out, W_out, kernel_h, kernel_w)
        # 原始 strides 乘以 stride 参数实现跳跃采样
        strides = (
            x_padded.strides[0],           # N 维度步长
            x_padded.strides[1],           # C 维度步长
            x_padded.strides[2] * stride,  # H 维度跳 S 行
            x_padded.strides[3] * stride,  # W 维度跳 S 列
            x_padded.strides[2],           # 卷积核内部 H 步长（不跳）
            x_padded.strides[3],           # 卷积核内部 W 步长（不跳）
        )

        # 创建一个"视图"（不复制数据），然后 reshape 为 im2col 格式
        patches = np.lib.stride_tricks.as_strided(
            x_padded, shape=shape, strides=strides
        )
        # patches 形状: (N, C, H_out, W_out, kernel_h, kernel_w)
        # 转置并 reshape 为 (N, C*kernel_h*kernel_w, H_out*W_out)
        cols = patches.transpose(0, 1, 4, 5, 2, 3).reshape(
            N, C * kernel_h * kernel_w, H_out * W_out
        )
        return cols

    @staticmethod
    def col2im(cols: np.ndarray, x_shape: Tuple[int, ...],
               kernel_h: int, kernel_w: int, stride: int = 1, pad: int = 0) -> np.ndarray:
        """
        将列矩阵还原为图像形状（col2im，用于反向传播）

        参数:
            cols: 列矩阵，形状 (N, C*k_h*k_w, H_out*W_out)
            x_shape: 原始输入形状 (N, C, H, W)
            kernel_h, kernel_w: 卷积核尺寸
            stride: 步长
            pad: 填充大小

        返回:
            x: 还原后的张量，形状 (N, C, H + 2*pad, W + 2*pad)
        """
        N, C, H, W = x_shape
        H_padded, W_padded = H + 2 * pad, W + 2 * pad
        H_out = (H + 2 * pad - kernel_h) // stride + 1
        W_out = (W + 2 * pad - kernel_w) // stride + 1

        # 将 cols reshape 为 patches 形状
        patches_shape = (N, C, kernel_h, kernel_w, H_out, W_out)
        patches = cols.reshape(patches_shape).transpose(0, 1, 4, 5, 2, 3)
        # patches 形状: (N, C, H_out, W_out, kernel_h, kernel_w)

        # 还原为 padded 图像（累加模式，因为每个像素可能被多个 patch 覆盖）
        x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
        for h in range(H_out):
            for w in range(W_out):
                h_start = h * stride
                w_start = w * stride
                x_padded[:, :, h_start:h_start+kernel_h, w_start:w_start+kernel_w] += \
                    patches[:, :, h, w, :, :]

        # 去掉 padding 部分
        if pad > 0:
            return x_padded[:, :, pad:-pad, pad:-pad]
        return x_padded


# ============================================================
# 第 3 部分：Conv2d —— 从零实现的卷积层
# ============================================================

class Conv2d:
    """
    二维卷积层（NumPy 实现，通过 im2col 加速）

    前向传播流程：输入 → im2col → 矩阵乘法 → reshape → 加偏置 → 输出
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = True):
        """
        初始化卷积层参数

        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数（卷积核数量）
            kernel_size: 卷积核大小（正方形，如 3 表示 3×3）
            stride: 步长
            padding: 填充大小
            bias: 是否使用偏置
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = bias

        # ---------- 参数初始化：He 初始化 ----------
        # 使用 He 初始化（适用于 ReLU），标准差 = sqrt(2 / fan_in)
        fan_in = in_channels * kernel_size * kernel_size  # 每个卷积核的输入连接数
        scale = np.sqrt(2.0 / fan_in)
        # 权重形状: (out_channels, in_channels, k_h, k_w)
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        # 偏置: 每个输出通道一个
        self.b = np.zeros(out_channels, dtype=np.float32) if bias else None

        # ---------- 缓存前向传播中间值（供反向传播使用）----------
        self.cache = {}  # 存储 x_cols, x_shape 等

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        卷积层前向传播

        参数:
            x: 输入张量，形状 (N, C_in, H, W)

        返回:
            out: 输出特征图，形状 (N, C_out, H_out, W_out)
        """
        N, C_in, H, W = x.shape
        assert C_in == self.in_channels, f"输入通道 {C_in} != 期望 {self.in_channels}"

        # ---------- 步骤 1: Im2Col ----------
        # 将每个卷积窗口展开为列向量，得到 (N, C_in*k*k, H_out*W_out)
        x_cols = Im2Col.im2col(x, self.kernel_size, self.kernel_size,
                                self.stride, self.padding)

        # ---------- 步骤 2: 权重展开 ----------
        # 将卷积核从 (C_out, C_in, k, k) 展开为 (C_out, C_in*k*k)
        W_col = self.W.reshape(self.out_channels, -1)

        # ---------- 步骤 3: 矩阵乘法 ----------
        # (N, C_in*k*k, H_out*W_out) @ (C_in*k*k, C_out) → 需要调整
        # 实际实现：对每个 batch 做 W_col @ x_cols[i]
        # 更高效的方式：
        # x_cols: (N, C_in*k*k, H_out*W_out)
        # W_col:  (C_out, C_in*k*k)
        # out_cols: (N, C_out, H_out*W_out) = W_col @ x_cols (广播到 batch)
        # 使用 np.einsum 或广播实现
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        # out_cols 形状: (N, C_out, H_out * W_out)
        out_cols = W_col @ x_cols  # 自动广播: (C_out, C_in*k*k) @ (N, C_in*k*k, H_out*W_out)

        # ---------- 步骤 4: Reshape + 加偏置 ----------
        out = out_cols.reshape(N, self.out_channels, H_out, W_out)
        if self.use_bias:
            # 偏置形状 (C_out,) → 广播到 (1, C_out, 1, 1) → (N, C_out, H_out, W_out)
            out += self.b.reshape(1, -1, 1, 1)

        # ---------- 缓存中间值 ----------
        self.cache["x_shape"] = x.shape
        self.cache["x_cols"] = x_cols

        return out

    @property
    def param_count(self) -> int:
        """返回该层的参数量"""
        count = self.W.size
        if self.use_bias:
            count += self.b.size
        return count


# ============================================================
# 第 4 部分：MaxPool2d —— 从零实现的最大池化层
# ============================================================

class MaxPool2d:
    """
    二维最大池化层（NumPy 实现）

    在 kernel_size × kernel_size 的窗口内取最大值，保留最显著的特征。
    """

    def __init__(self, kernel_size: int = 2, stride: int = 2):
        """
        初始化池化层

        参数:
            kernel_size: 池化窗口大小（通常 2）
            stride: 池化步长（通常 2，与 kernel_size 相同实现不重叠池化）
        """
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = {}  # 存储 argmax 索引（供反向传播使用）

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        最大池化前向传播

        参数:
            x: 输入特征图，形状 (N, C, H, W)

        返回:
            out: 池化后的特征图，形状 (N, C, H_out, W_out)
        """
        N, C, H, W = x.shape
        k = self.kernel_size
        s = self.stride

        # 计算输出尺寸
        H_out = (H - k) // s + 1
        W_out = (W - k) // s + 1

        # ---------- 使用 im2col 高效实现 ----------
        # 将 x 展开为 patches: (N, C, H_out, W_out, k, k)
        # 使用 as_strided 技巧
        shape = (N, C, H_out, W_out, k, k)
        strides = (x.strides[0], x.strides[1],
                   x.strides[2] * s, x.strides[3] * s,
                   x.strides[2], x.strides[3])
        patches = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
        # patches 形状: (N, C, H_out, W_out, k, k)

        # 在最后两个维度 (k, k) 上取最大值
        out = patches.max(axis=(4, 5))  # 沿 k, k 维度取 max
        # out 形状: (N, C, H_out, W_out)

        # ---------- 缓存 argmax（用于反向传播）----------
        # argmax 返回展平后的索引，记录每个窗口中最大值的位置
        patches_flat = patches.reshape(N, C, H_out, W_out, -1)  # 展平 k*k
        self.cache["argmax"] = patches_flat.argmax(axis=4)  # 形状 (N, C, H_out, W_out)
        self.cache["x_shape"] = x.shape

        return out


# ============================================================
# 第 5 部分：ReLU —— 激活函数
# ============================================================

class ReLU:
    """ReLU 激活函数 f(x) = max(0, x)"""

    def __init__(self):
        self.cache = {}  # 存储 x>0 的 mask

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        ReLU 前向传播

        参数:
            x: 任意形状的张量
        返回:
            out: max(0, x) 逐元素
        """
        self.cache["mask"] = x > 0  # 记录哪些位置 >0
        return x * self.cache["mask"]


# ============================================================
# 第 6 部分：全连接层（用于分类器）
# ============================================================

class Linear:
    """全连接层（仿射变换）"""

    def __init__(self, in_features: int, out_features: int):
        """
        初始化全连接层

        参数:
            in_features: 输入维度
            out_features: 输出维度
        """
        scale = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(in_features, out_features) * scale  # (D_in, D_out)
        self.b = np.zeros(out_features, dtype=np.float32)

        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播 y = x @ W + b

        参数:
            x: 输入，形状 (N, D_in)
        返回:
            out: 输出，形状 (N, D_out)
        """
        self.cache["x"] = x
        return x @ self.W + self.b

    @property
    def param_count(self) -> int:
        """返回参数量"""
        return self.W.size + self.b.size


# ============================================================
# 第 7 部分：Softmax + 交叉熵
# ============================================================

def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Softmax 函数（数值稳定版本）

    参数:
        logits: 形状 (N, C)，每行是一个样本的各类别分数
    返回:
        probs: 形状 (N, C)，每行和为 1 的概率分布
    """
    # 减去最大值防止指数溢出
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / exp_vals.sum(axis=1, keepdims=True)


def cross_entropy_loss(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    交叉熵损失

    参数:
        probs: softmax 输出的概率分布，形状 (N, C)
        labels: 真实标签索引，形状 (N,) 每个值是 0~C-1 的整数
    返回:
        loss: 平均交叉熵损失
    """
    N = probs.shape[0]
    # 只取正确类别对应的概率，取 -log
    correct_probs = probs[np.arange(N), labels]
    # 加小值防止 log(0)
    return -np.mean(np.log(correct_probs + 1e-8))


def compute_accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    """计算分类准确率"""
    preds = probs.argmax(axis=1)  # 预测类别
    return (preds == labels).mean()


# ============================================================
# 第 8 部分：SimpleCNN —— 构建简单 CNN 模型
# ============================================================

class SimpleCNN:
    """
    简单 CNN 模型：Conv → ReLU → Pool → Conv → ReLU → Pool → Flatten → FC → Softmax

    架构:
        - Conv2d(in=1, out=8, k=3, padding=1)  → ReLU → MaxPool2d(2,2)
        - Conv2d(in=8, out=16, k=3, padding=1) → ReLU → MaxPool2d(2,2)
        - Linear(16*7*7, 10) → Softmax
    """

    def __init__(self):
        # ---------- 构建网络层 ----------
        # 第 1 个卷积块
        self.conv1 = Conv2d(in_channels=1, out_channels=8, kernel_size=3,
                            stride=1, padding=1)  # 输出: (8, 28, 28)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2d(kernel_size=2, stride=2)  # 输出: (8, 14, 14)

        # 第 2 个卷积块
        self.conv2 = Conv2d(in_channels=8, out_channels=16, kernel_size=3,
                            stride=1, padding=1)  # 输出: (16, 14, 14)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2d(kernel_size=2, stride=2)  # 输出: (16, 7, 7)

        # 分类器
        self.fc = Linear(in_features=16 * 7 * 7, out_features=10)

        # 存储每一层的特征图（用于可视化）
        self.feature_maps = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        CNN 前向传播

        参数:
            x: 输入图像，形状 (N, 1, 28, 28)
        返回:
            probs: 类别概率，形状 (N, 10)
        """
        self.feature_maps = {}

        # ---------- Block 1: Conv → ReLU → Pool ----------
        x = self.conv1.forward(x)
        self.feature_maps["conv1"] = x  # 记录卷积输出
        x = self.relu1.forward(x)
        self.feature_maps["relu1"] = x  # 记录激活输出
        x = self.pool1.forward(x)
        self.feature_maps["pool1"] = x  # 记录池化输出

        # ---------- Block 2: Conv → ReLU → Pool ----------
        x = self.conv2.forward(x)
        self.feature_maps["conv2"] = x
        x = self.relu2.forward(x)
        self.feature_maps["relu2"] = x
        x = self.pool2.forward(x)
        self.feature_maps["pool2"] = x  # 形状: (N, 16, 7, 7)

        # ---------- Flatten + FC → Softmax ----------
        N = x.shape[0]
        x_flat = x.reshape(N, -1)  # 展平: (N, 16*7*7)
        logits = self.fc.forward(x_flat)  # (N, 10)
        probs = softmax(logits)  # (N, 10)
        return probs

    @property
    def total_params(self) -> int:
        """返回总参数量"""
        return self.conv1.param_count + self.conv2.param_count + self.fc.param_count


# ============================================================
# 第 9 部分：可视化工具
# ============================================================

def visualize_kernels(conv_layer: Conv2d, save_path: str):
    """
    将卷积核可视化为小图像

    参数:
        conv_layer: 训练好的 Conv2d 层
        save_path: 保存路径
    """
    kernels = conv_layer.W  # 形状 (C_out, C_in, k, k)
    C_out, C_in, k, _ = kernels.shape

    fig, axes = plt.subplots(C_in, C_out, figsize=(C_out * 1.2, C_in * 1.2))
    if C_in == 1 and C_out == 1:
        axes = np.array([[axes]])
    elif C_in == 1:
        axes = axes.reshape(1, -1)
    elif C_out == 1:
        axes = axes.reshape(-1, 1)

    for ic in range(C_in):
        for oc in range(C_out):
            ax = axes[ic, oc]
            kernel = kernels[oc, ic]  # 单个 2D 卷积核
            im = ax.imshow(kernel, cmap="RdBu_r", vmin=-np.abs(kernel).max(),
                           vmax=np.abs(kernel).max())
            ax.set_xticks([])
            ax.set_yticks([])
            if ic == 0:
                ax.set_title(f"ch{oc}", fontsize=8)
            if oc == 0:
                ax.set_ylabel(f"in{ic}", fontsize=8)

    plt.suptitle(f"Kernel Visualization (C_in={C_in}, C_out={C_out})", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  [可视化] 卷积核已保存到 {save_path}")


def visualize_feature_maps(feature_maps: dict, save_prefix: str, sample_idx: int = 0):
    """
    逐层可视化特征图

    参数:
        feature_maps: SimpleCNN.forward() 记录的字典 {层名: 特征图 (N,C,H,W)}
        save_prefix: 文件名前缀
        sample_idx: 要可视化的样本索引
    """
    for layer_name, fm in feature_maps.items():
        # fm 形状: (N, C, H, W)
        N, C, H, W = fm.shape
        ncols = min(8, C)  # 最多显示 8 个通道
        nrows = (C + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.5))
        axes = np.atleast_2d(axes)  # 确保是 2D 数组

        for c in range(C):
            row, col = c // ncols, c % ncols
            ax = axes[row, col]
            ax.imshow(fm[sample_idx, c], cmap="viridis")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"ch{c}", fontsize=7)

        # 关闭多余的子图
        for c in range(C, nrows * ncols):
            row, col = c // ncols, c % ncols
            axes[row, col].axis("off")

        plt.suptitle(f"{layer_name} Feature Map (sample {sample_idx})", fontsize=11)
        plt.tight_layout()
        fname = f"{save_prefix}_feat_{layer_name}.png"
        plt.savefig(fname, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"  [可视化] {layer_name} 特征图已保存到 {fname}")


def compute_receptive_field(layers: list, verbose: bool = True) -> int:
    """
    计算给定 CNN 架构的最终感受野大小

    参数:
        layers: 层配置列表，每个元素为 (kernel_size, stride) 的元组
        verbose: 是否打印每一步的中间结果

    返回:
        rf: 最后一层在原始输入上的感受野大小

    示例:
        >>> compute_receptive_field([(3,1), (2,2), (3,1), (2,2)])
        >>> # Conv3→Pool2→Conv3→Pool2 的感受野
    """
    rf = 1  # 初始感受野（输入层）
    cum_stride = 1  # 累积步长（到原始输入的缩放因子）

    if verbose:
        print("\n  感受野逐层变化：")
        print(f"  {'层':<8} {'核大小':<8} {'步长':<8} {'感受野':<8}")

    for i, (k, s) in enumerate(layers):
        rf = rf + (k - 1) * cum_stride  # 递推公式
        cum_stride *= s  # 累积步长更新

        if verbose:
            print(f"  Layer{i+1:<4} {k:<8} {s:<8} {rf:<8}")

    return rf


# ============================================================
# 第 10 部分：训练与演示
# ============================================================

def train_and_demo():
    """主函数：加载数据 → 训练 CNN → 可视化"""
    print("=" * 60)
    print("s10 CNN 核心原理 Demo")
    print("从零实现 Conv2d、MaxPool2d，在 MNIST 上训练简单 CNN")
    print("=" * 60)

    # ---------- 1. 加载数据 ----------
    print("\n[1/6] 加载 MNIST 数据集...")
    X_train, y_train, X_test, y_test = load_mnist()
    # 调整形状: (N, 784) → (N, 1, 28, 28)
    X_train = X_train.reshape(-1, 1, 28, 28)
    X_test = X_test.reshape(-1, 1, 28, 28)
    print(f"  训练集: {X_train.shape}, 标签: {y_train.shape}")
    print(f"  测试集: {X_test.shape}, 标签: {y_test.shape}")

    # ---------- 2. 构建模型 ----------
    print("\n[2/6] 构建 SimpleCNN 模型...")
    model = SimpleCNN()
    print(f"  模型结构: Conv(1→8,3×3)→ReLU→Pool(2)→Conv(8→16,3×3)→ReLU→Pool(2)→FC(784→10)")
    print(f"  总参数量: {model.total_params:,}")

    # 对比全连接网络
    fc_params = 28 * 28 * 128 + 128 * 64 + 64 * 10  # 一个简单的 3 层 MLP
    print(f"  等效全连接网络参数量（估算）: ~{fc_params:,}")
    print(f"  参数节省倍数: {fc_params / model.total_params:.1f}x")

    # ---------- 3. 训练（少量 epoch，仅用于演示）----------
    print("\n[3/6] 开始训练（demo 模式，仅 3 个 epoch）...")
    batch_size = 32
    n_epochs = 3
    n_train = X_train.shape[0]
    n_batches = n_train // batch_size
    lr = 0.01

    train_losses = []
    for epoch in range(n_epochs):
        # 随机打乱训练数据
        indices = np.random.permutation(n_train)
        epoch_loss = 0.0

        for b in range(n_batches):
            batch_idx = indices[b * batch_size:(b + 1) * batch_size]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]

            # 前向传播
            probs = model.forward(X_batch)
            loss = cross_entropy_loss(probs, y_batch)
            epoch_loss += loss

            # 手工实现 SGD（简化版，仅更新 FC 和 Conv 的 W, b）
            # 计算 softmax + cross-entropy 的梯度
            N = X_batch.shape[0]
            dlogits = probs.copy()
            dlogits[np.arange(N), y_batch] -= 1  # softmax 交叉熵的梯度
            dlogits /= N  # 除以 batch size

            # FC 层反向传播（手动链式法则）
            x_flat = model.fc.cache["x"]  # (N, 16*7*7)
            dW_fc = x_flat.T @ dlogits  # (16*7*7, 10)
            db_fc = dlogits.sum(axis=0)  # (10,)
            dx_flat = dlogits @ model.fc.W.T  # (N, 16*7*7)

            # 更新 FC 参数
            model.fc.W -= lr * dW_fc
            model.fc.b -= lr * db_fc

            # Conv 层的梯度（简化处理：只更新 conv2，conv1 近似通过）
            # 实际完整的反向传播需要实现 conv 的反向，这里仅做演示
            # 对于 demo 用途，仅更新 FC 层即可看到训练效果

            # 更新 conv2（近似梯度）
            d_pool2 = dx_flat.reshape(N, 16, 7, 7)  # reshape 回特征图形状
            # 反池化（简化：用最近邻上采样近似）
            d_relu2 = np.repeat(np.repeat(d_pool2, 2, axis=2), 2, axis=3)[:, :, :14, :14]
            d_relu2 *= model.relu2.cache["mask"]  # ReLU 反向

            # 完整反向传播会计算 conv2 的梯度，这里用简化的 SGD 近似
            # 增量更新 conv2 权重
            x_cols2 = model.conv2.cache["x_cols"]  # (N, C_in*k*k, H_out*W_out)
            d_out_cols2 = d_relu2.reshape(N, 16, -1)  # (N, 16, H_out*W_out)
            # dW = d_out_cols2 @ x_cols2.T as batch matmul, then sum over N
            # d_out_cols2: (N, C_out, H_out*W_out), x_cols2: (N, C_in*k*k, H_out*W_out)
            # x_cols2.T over the last two dims: (N, H_out*W_out, C_in*k*k)
            # batch matmul: (N, C_out, H_out*W_out) @ (N, H_out*W_out, C_in*k*k) -> (N, C_out, C_in*k*k)
            dW_col2 = (d_out_cols2 @ x_cols2.transpose(0, 2, 1)).sum(axis=0)  # (C_out, C_in*k*k)
            dW_conv2 = dW_col2.reshape(model.conv2.W.shape) / N
            model.conv2.W -= lr * 0.1 * dW_conv2  # 小学习率更新

            # 同样近似更新 conv1
            if model.conv1.cache.get("x_cols") is not None:
                # Proper conv2 backward to compute d_pool1 (dX of conv2)
                # d_out_cols2: (N, 16, 196), need dX = W^T @ d_out via im2col
                # batch matmul: (N, 196, 16) @ (16, 72) = (N, 196, 72) -> transpose -> (N, 72, 196)
                W_col2 = model.conv2.W.reshape(model.conv2.out_channels, -1)
                dX_cols2 = d_out_cols2.transpose(0, 2, 1) @ W_col2  # (N, 196, 72)
                dX_cols2 = dX_cols2.transpose(0, 2, 1)  # (N, 72, 196)
                d_pool1 = Im2Col.col2im(dX_cols2, (N, 8, 14, 14),
                                        model.conv2.kernel_size, model.conv2.kernel_size,
                                        model.conv2.stride, model.conv2.padding)
                # d_pool1.shape: (N, 8, 14, 14) — now correct channel count
                # Naive unpool pool1
                d_relu1 = np.repeat(np.repeat(d_pool1, 2, axis=2), 2, axis=3)[:, :, :28, :28]
                d_relu1 *= model.relu1.cache["mask"]
                x_cols1 = model.conv1.cache["x_cols"]
                d_out_cols1 = d_relu1.reshape(N, 8, -1)
                dW_col1 = (d_out_cols1 @ x_cols1.transpose(0, 2, 1)).sum(axis=0)
                dW_conv1 = dW_col1.reshape(model.conv1.W.shape) / N
                model.conv1.W -= lr * 0.01 * dW_conv1  # 更小的学习率

        epoch_loss /= n_batches
        train_losses.append(epoch_loss)
        print(f"  Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.4f}")

    # ---------- 4. 测试模型 ----------
    print("\n[4/6] 测试模型...")
    test_probs = model.forward(X_test[:1000])  # 取 1000 张测试
    test_acc = compute_accuracy(test_probs, y_test[:1000])
    print(f"  测试准确率 (1000 样本): {test_acc:.2%}")

    # ---------- 5. 可视化 ----------
    print("\n[5/6] 生成可视化...")
    output_dir = _IMAGES_DIR

    # 5a. 可视化第一个输入图像
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(X_test[0, 0], cmap="gray")
    ax.set_title(f"Input Image - True Label: {y_test[0]}", fontsize=12)
    ax.axis("off")
    out_path = os.path.join(output_dir, "demo_input.png")
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  [可视化] 输入图像已保存到 {out_path}")

    # 5b. 可视化卷积核
    viz_path = os.path.join(output_dir, "demo_kernels_conv1.png")
    visualize_kernels(model.conv1, viz_path)

    # 5c. 可视化各层特征图
    # 重新做一次前向传播以记录特征图
    _ = model.forward(X_test[:1])
    visualize_feature_maps(model.feature_maps, os.path.join(output_dir, "demo"), sample_idx=0)

    # ---------- 6. 感受野计算 ----------
    print("\n[6/6] 计算感受野...")
    # SimpleCNN 的层序列：Conv3→Pool2→Conv3→Pool2
    layer_config = [(3, 1), (2, 2), (3, 1), (2, 2)]
    rf = compute_receptive_field(layer_config)
    print(f"\n  最终感受野: {rf}×{rf}")
    print(f"  这意味最后一层（Pool2 之后）的每个神经元能看到原始")
    print(f"  输入图像上 {rf}×{rf} 的区域，约占 28×28 图像的({rf/28:.1%})")

    # ---------- 7. 参数量对比总结 ----------
    print("\n" + "=" * 60)
    print("参数量对比总结:")
    print(f"  Conv1 (1→8, 3×3): {model.conv1.param_count:,} 个参数")
    print(f"  Conv2 (8→16, 3×3): {model.conv2.param_count:,} 个参数")
    print(f"  FC (784→10):       {model.fc.param_count:,} 个参数")
    print(f"  CNN 总计:          {model.total_params:,} 个参数")
    print(f"  等效 3层 MLP:      ~{fc_params:,} 个参数")
    print(f"  参数减少:          {fc_params / model.total_params:.1f} 倍")
    print("=" * 60)
    print(f"\nDemo 完成！查看 {_IMAGES_DIR} 目录下的可视化结果。")


if __name__ == "__main__":
    train_and_demo()
