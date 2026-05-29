# -*- coding: utf-8 -*-
"""
===============================================================================
s01_ai_overview/code/demo.py — 感知机从零实现
===============================================================================
本演示从零开始（仅使用 NumPy）实现一个完整的感知机（Perceptron）模型，
涵盖数据生成、模型定义、训练、预测和可视化五个环节。

通过本演示，你将理解：
  1. 感知机的数学模型：ŷ = sign(w·x + b)，其中 sign 为阶跃函数
  2. 感知机学习算法：对于每个误分类样本，使用规则 w ← w + η·y·x 更新权重
  3. 感知机的几何意义：寻找一个超平面 w·x + b = 0 来分离两类数据
  4. 感知机仅对线性可分数据保证收敛

感知机是神经网络的基本单元，也是深度学习的起点。
理解感知机的工作机制，是理解更复杂模型的基础。

作者：learn-ai 项目
日期：2025
===============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False

# 图片保存目录：固定为本章节的 images/ 目录（相对于本脚本的 ../images/）
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)


# ============================================================================
# 第一部分：生成合成数据
# ============================================================================

def generate_linearly_separable_data(n_samples: int = 100, random_seed: int = 42):
    """
    生成线性可分的二分类数据集。

    在二维平面上生成两个类别的点，它们可以被一条直线完全分开。
    这样可以保证感知机算法能够收敛。

    参数:
        n_samples: int, 每类样本的数量（总共 2*n_samples 个点）
        random_seed: int, 随机种子，保证结果可复现

    返回:
        X: np.ndarray, 形状 (2*n_samples, 2)，特征矩阵（x1, x2 坐标）
        y: np.ndarray, 形状 (2*n_samples,)，标签（-1 或 +1）
    """
    np.random.seed(random_seed)  # 固定随机种子，保证每次运行结果一致

    # 第一类数据：从均值 [2, 2] 的正态分布中采样，标签为 +1
    X_pos = np.random.randn(n_samples, 2) + np.array([2.0, 2.0])  # 形状 (100, 2)
    y_pos = np.ones(n_samples)  # 标签全为 1，形状 (100,)

    # 第二类数据：从均值 [-2, -2] 的正态分布中采样，标签为 -1
    X_neg = np.random.randn(n_samples, 2) + np.array([-2.0, -2.0])  # 形状 (100, 2)
    y_neg = -np.ones(n_samples)  # 标签全为 -1，形状 (100,)

    # 沿第 0 维（行方向）拼接两类数据
    X = np.vstack([X_pos, X_neg])  # 形状 (200, 2)
    y = np.hstack([y_pos, y_neg])  # 形状 (200,)

    # 随机打乱数据顺序，避免训练时先看到一类再看到另一类
    shuffle_idx = np.random.permutation(len(y))  # 生成随机排列的索引
    X = X[shuffle_idx]  # 按随机索引重排特征
    y = y[shuffle_idx]  # 按随机索引重排标签

    return X, y


# ============================================================================
# 第二部分：感知机模型
# ============================================================================

class Perceptron:
    """
    感知机分类器。

    感知机是最简单的神经网络模型。它由一个线性组合器和一个阶跃激活函数组成。
    模型预测：ŷ = sign(w·x + b)，其中 sign(z) = 1 if z >= 0 else -1

    学习方法（感知机学习算法）：
        - 遍历每个训练样本 (x_i, y_i)
        - 如果预测结果 ŷ_i 与真实标签 y_i 不一致（即 y_i * (w·x_i + b) <= 0）：
            - 更新权重：w ← w + η * y_i * x_i
            - 更新偏置：b ← b + η * y_i
        - 重复直到所有样本分类正确（或达到最大迭代次数）

    属性:
        w: np.ndarray, 权重向量，形状 (n_features,)
        b: float, 偏置项
        losses: list, 每轮训练后的误分类样本数（损失记录）
    """

    def __init__(self, learning_rate: float = 0.01, max_epochs: int = 1000):
        """
        初始化感知机模型。

        参数:
            learning_rate: float, 学习率 η，控制每次更新的步长
            max_epochs: int, 最大训练轮数，防止数据不可分时无限循环
        """
        self.learning_rate = learning_rate  # 学习率 η，控制权重更新的幅度
        self.max_epochs = max_epochs  # 最大迭代次数，防止死循环
        self.w = None  # 权重向量，训练时初始化
        self.b = None  # 偏置项，训练时初始化
        self.losses = []  # 记录每轮 epoch 的误分类样本数

    def _activation(self, z: np.ndarray) -> np.ndarray:
        """
        阶跃激活函数（Step Function）。

        对于输入 z，输出 +1（z >= 0）或 -1（z < 0）。
        这是感知机使用的激活函数，也是最简单的激活函数。

        参数:
            z: np.ndarray, 线性组合结果 w·x + b

        返回:
            np.ndarray, 激活后的输出，值为 +1 或 -1
        """
        return np.where(z >= 0, 1, -1)  # 大于等于 0 输出 1，否则输出 -1

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        训练感知机模型。

        使用感知机学习算法迭代更新权重，直到所有样本分类正确
        或达到最大训练轮数。

        感知机收敛定理：如果数据是线性可分的，感知机算法一定能在有限步内收敛。

        参数:
            X: np.ndarray, 形状 (n_samples, n_features)，训练数据特征
            y: np.ndarray, 形状 (n_samples,)，训练数据标签（取值 -1 或 +1）
        """
        n_samples, n_features = X.shape  # n_samples: 样本数, n_features: 特征数

        # 用 Xavier 初始化权重（小随机数），偏置初始化为 0
        self.w = np.random.randn(n_features) * 0.01  # 小随机数初始化权重
        self.b = 0.0  # 偏置初始化为 0
        self.losses = []  # 清空损失记录

        for epoch in range(self.max_epochs):
            n_errors = 0  # 记录本轮 epoch 的误分类样本数

            # 遍历每一个训练样本
            for i in range(n_samples):
                x_i = X[i]  # 第 i 个样本的特征向量，形状 (n_features,)
                y_i = y[i]  # 第 i 个样本的真实标签

                # 计算线性组合: z = w·x_i + b
                z = np.dot(self.w, x_i) + self.b  # 标量，w 和 x_i 的点积 + 偏置

                # 判断是否误分类: 如果 y_i * z <= 0，说明预测和真实标签不一致
                # 因为正确的预测应该是 y_i 和 z 同号（都正或都负）
                if y_i * z <= 0:
                    # 感知机更新规则（核心公式！）
                    # w ← w + η * y_i * x_i：沿着正确方向移动权重的方向
                    # 直观理解：如果 y_i=+1，就把 w 往 x_i 方向推；
                    #           如果 y_i=-1，就把 w 往 x_i 反方向推
                    self.w += self.learning_rate * y_i * x_i
                    self.b += self.learning_rate * y_i  # 偏置也同步更新
                    n_errors += 1  # 累计误分类数

            # 记录本轮的误分类数（作为损失指标）
            self.losses.append(n_errors)

            # 如果本轮没有误分类样本，说明已经完全分开了，提前结束训练
            if n_errors == 0:
                print(f"感知机在第 {epoch + 1} 轮收敛！所有样本分类正确。")
                break

        # 训练结束后打印最终结果
        print(f"训练完成。共 {epoch + 1} 轮。")
        print(f"最终权重 w = {self.w}, 偏置 b = {self.b:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        对新数据进行预测。

        对输入 X 中的每个样本，计算 w·x + b，然后通过阶跃函数输出类别。

        参数:
            X: np.ndarray, 形状 (n_samples, n_features)，待预测的特征

        返回:
            np.ndarray, 形状 (n_samples,)，预测的类别标签（+1 或 -1）
        """
        # 计算线性组合 z = w·x + b（矩阵形式，可批量处理）
        z = np.dot(X, self.w) + self.b  # 形状 (n_samples,)，每个样本一个得分
        return self._activation(z)  # 通过阶跃函数得到最终类别

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        计算决策函数值（未经激活函数处理的原始得分）。

        用于绘制决策边界和计算点到超平面的距离。

        参数:
            X: np.ndarray, 形状 (n_samples, n_features)

        返回:
            np.ndarray, 形状 (n_samples,)，每个样本的原始得分 w·x + b
        """
        return np.dot(X, self.w) + self.b  # 返回未经 sign 处理的原始得分


# ============================================================================
# 第三部分：可视化
# ============================================================================

def plot_decision_boundary(perceptron: Perceptron, X: np.ndarray, y: np.ndarray):
    """
    绘制数据点和感知机的决策边界。

    在二维平面上画出所有数据点（不同颜色表示不同类别），
    以及感知机学到的分界线（超平面 w·x + b = 0）。

    参数:
        perceptron: Perceptron, 训练好的感知机模型
        X: np.ndarray, 形状 (n_samples, 2)，训练数据
        y: np.ndarray, 形状 (n_samples,)，训练标签
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))  # 创建 1 行 2 列的子图

    # ---- 子图 1：决策边界 ----
    ax = axes[0]

    # 绘制正类样本（y=+1），用红色圆点
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='red', marker='o',
               edgecolors='k', s=60, label='Class +1', alpha=0.7)
    # 绘制负类样本（y=-1），用蓝色三角
    ax.scatter(X[y == -1, 0], X[y == -1, 1], c='blue', marker='s',
               edgecolors='k', s=60, label='Class -1', alpha=0.7)

    # 获取当前坐标轴范围，用于绘制决策边界线
    x_min, x_max = ax.get_xlim()

    # 从权重计算决策边界的斜率和截距
    # 决策边界方程: w1*x1 + w2*x2 + b = 0
    # 变形为: x2 = -(w1/w2)*x1 - (b/w2)
    w1, w2 = perceptron.w[0], perceptron.w[1]  # 提取两个权重分量
    b_val = perceptron.b  # 偏置
    slope = -w1 / w2  # 斜率 = -w1/w2
    intercept = -b_val / w2  # 截距 = -b/w2

    # 生成 x1 坐标点，用于画线
    x1_line = np.linspace(x_min, x_max, 100)  # 100 个等间距点
    x2_line = slope * x1_line + intercept  # 对应的 x2 坐标

    ax.plot(x1_line, x2_line, 'g-', linewidth=2, label='Decision Boundary w*x+b=0')

    # 绘制法向量 w 的箭头（垂直于决策边界，指向正类方向）
    # 取决策边界上的一点作为箭头起点
    center_x1 = np.mean(x1_line)  # 决策边界中点的 x1 坐标
    center_x2 = slope * center_x1 + intercept  # 对应的 x2 坐标
    ax.arrow(center_x1, center_x2,
             perceptron.w[0] * 0.5, perceptron.w[1] * 0.5,
             head_width=0.15, head_length=0.15, fc='purple', ec='purple',
             label='Weight Vector w')

    ax.set_xlabel('Feature x1', fontsize=12)  # x 轴标签
    ax.set_ylabel('Feature x2', fontsize=12)  # y 轴标签
    ax.set_title('Perceptron Decision Boundary', fontsize=14)  # 子图标题
    ax.legend(loc='upper left', fontsize=8)  # 显示图例
    ax.grid(True, alpha=0.3)  # 添加半透明网格
    ax.set_aspect('equal')  # 设置等比例坐标轴

    # ---- 子图 2：训练损失曲线 ----
    ax = axes[1]
    ax.plot(range(1, len(perceptron.losses) + 1), perceptron.losses,
            'b-o', markersize=4, linewidth=1.5)  # 蓝色圆点线
    ax.set_xlabel('Epoch', fontsize=12)  # x 轴标签
    ax.set_ylabel('Misclassified Samples', fontsize=12)  # y 轴标签
    ax.set_title('Misclassifications During Training', fontsize=14)  # 子图标题
    ax.grid(True, alpha=0.3)  # 添加半透明网格

    plt.tight_layout()  # 自动调整子图间距
    plt.savefig(os.path.join(_IMAGES_DIR, 'perceptron_results.png'), dpi=150, bbox_inches='tight')  # 保存图片
    plt.show()  # 显示图片
    print(f"\n图片已保存为 {os.path.join(_IMAGES_DIR, 'perceptron_results.png')}")


# ============================================================================
# 第四部分：主程序
# ============================================================================

def main():
    """
    主函数：串联数据生成、模型训练、评估和可视化的完整流程。
    """
    print("=" * 60)
    print("感知机从零实现 — s01_ai_overview")
    print("=" * 60)

    # 1. 生成线性可分的合成数据
    print("\n[步骤 1] 生成线性可分数据...")
    X, y = generate_linearly_separable_data(n_samples=100, random_seed=42)
    print(f"数据形状: X={X.shape}, y={y.shape}")  # X: (200, 2), y: (200,)
    print(f"类别分布: +1 有 {np.sum(y == 1)} 个, -1 有 {np.sum(y == -1)} 个")

    # 2. 创建感知机模型并训练
    print("\n[步骤 2] 创建感知机模型并训练...")
    perceptron = Perceptron(learning_rate=0.1, max_epochs=500)  # 学习率 0.1
    perceptron.fit(X, y)  # 在训练数据上拟合模型

    # 3. 评估模型准确率
    print("\n[步骤 3] 评估模型...")
    y_pred = perceptron.predict(X)  # 对训练数据进行预测
    accuracy = np.mean(y_pred == y)  # 计算准确率 = 预测正确的比例
    print(f"训练集准确率: {accuracy:.2%}")  # 打印准确率百分比

    # 4. 可视化决策边界和训练过程
    print("\n[步骤 4] 可视化决策边界...")
    plot_decision_boundary(perceptron, X, y)

    # 5. 测试感知机在几个样本点上的预测
    print("\n[步骤 5] 测试几个样本点的预测...")
    test_points = np.array([
        [2.0, 2.0],   # 应该被预测为 +1（正类区域）
        [-2.0, -2.0], # 应该被预测为 -1（负类区域）
        [0.0, 0.0],   # 决策边界附近，预测结果取决于模型学到什么
        [3.0, -1.0],  # 边界测试
    ])
    for i, point in enumerate(test_points):
        pred = perceptron.predict(point.reshape(1, -1))[0]  # 预测单个点
        score = perceptron.decision_function(point.reshape(1, -1))[0]  # 原始得分
        print(f"  点 ({point[0]:.1f}, {point[1]:.1f}) → "
              f"预测: {'+1' if pred > 0 else '-1'}, 得分: {score:.4f}")

    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


# 运行主函数
if __name__ == '__main__':
    main()
