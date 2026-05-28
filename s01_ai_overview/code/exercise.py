# -*- coding: utf-8 -*-
"""
===============================================================================
s01_ai_overview/code/exercise.py — 感知机练习
===============================================================================
本练习文件中，感知机模型的核心部分被替换为了 TODO 注释。
你的任务是完成这些 TODO，使感知机能够正确训练和预测。

练习目标：
  1. 理解感知机的激活函数（阶跃函数）的数学形式
  2. 掌握感知机权重更新的核心规则
  3. 完成预测方法，将训练好的模型用于分类

提示：
  - 感知机的激活函数：sign(z) = 1 (z >= 0) 或 -1 (z < 0)
  - 权重更新规则：w ← w + η * y_i * x_i（当样本被误分类时）
  - 预测：对每个样本 x，计算 sign(w·x + b)

运行方式：
  python exercise.py

如果你正确实现了代码，你应该看到：
  - 训练过程中的误分类数逐渐减少到 0
  - 训练集准确率应为 100%（因为是线性可分数据）
  - 可视化图显示清晰的决策边界
===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def generate_linearly_separable_data(n_samples=100, random_seed=42):
    """
    生成线性可分的二分类数据集。

    参数:
        n_samples: int, 每类样本数
        random_seed: int, 随机种子

    返回:
        X: np.ndarray, 形状 (2*n_samples, 2)
        y: np.ndarray, 形状 (2*n_samples,)，标签为 -1 或 +1
    """
    np.random.seed(random_seed)
    # 生成正类数据（均值在 [2, 2]，标签 +1）
    X_pos = np.random.randn(n_samples, 2) + np.array([2.0, 2.0])
    y_pos = np.ones(n_samples)
    # 生成负类数据（均值在 [-2, -2]，标签 -1）
    X_neg = np.random.randn(n_samples, 2) + np.array([-2.0, -2.0])
    y_neg = -np.ones(n_samples)
    # 合并并打乱
    X = np.vstack([X_pos, X_neg])
    y = np.hstack([y_pos, y_neg])
    shuffle_idx = np.random.permutation(len(y))
    return X[shuffle_idx], y[shuffle_idx]


class PerceptronExercise:
    """
    感知机分类器（练习版）。

    部分关键方法留空，需要你完成 TODO 实现。

    属性:
        w: np.ndarray, 权重向量
        b: float, 偏置
        learning_rate: float, 学习率 η
        max_epochs: int, 最大训练轮数
        losses: list, 每轮的误分类样本数
    """

    def __init__(self, learning_rate=0.01, max_epochs=1000):
        """初始化模型参数。"""
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.w = None
        self.b = None
        self.losses = []

    # ======================================================================
    # TODO 1: 实现激活函数
    # ======================================================================
    # 感知机使用阶跃函数（Step Function）作为激活函数。
    # 输入 z 是一个数值或数组，你需要返回：
    #   +1 当 z >= 0
    #   -1 当 z < 0
    #
    # 提示：使用 np.where() 可以优雅地实现向量化操作。
    # 示例：np.where(condition, value_if_true, value_if_false)
    # ======================================================================
    def _activation(self, z):
        """
        阶跃激活函数。

        参数:
            z: np.ndarray, 输入值

        返回:
            np.ndarray, 激活后输出（+1 或 -1）
        """
        # TODO: 实现阶跃函数
        # 如果 z >= 0，返回 1；否则返回 -1
        # 请用一行代码完成
        pass  # <-- 替换为你的代码

    # ======================================================================
    # TODO 2: 实现预测方法
    # ======================================================================
    # 预测方法需要做两件事：
    #   1. 计算线性组合 z = w·x + b（对所有样本同时计算）
    #   2. 将 z 通过激活函数得到类别标签
    #
    # 提示：使用 np.dot(X, self.w) + self.b 进行批量计算
    # ======================================================================
    def predict(self, X):
        """
        对样本 X 进行类别预测。

        参数:
            X: np.ndarray, 形状 (n_samples, n_features)

        返回:
            np.ndarray, 形状 (n_samples,)，预测类别（+1 或 -1）
        """
        # TODO: 实现预测
        # 步骤 1: 计算线性组合 z = w·X + b
        # 步骤 2: 通过激活函数得到最终类别
        pass  # <-- 替换为你的代码

    # ======================================================================
    # TODO 3: 实现感知机权重更新规则
    # ======================================================================
    # 在下面的 fit() 方法中，找到 TODO 3 标记的位置。
    # 当样本 (x_i, y_i) 被误分类时（即 y_i * (w·x_i + b) <= 0），
    # 你需要按以下规则更新权重和偏置：
    #   self.w ← self.w + η * y_i * x_i
    #   self.b ← self.b + η * y_i
    #
    # 直觉解释：
    #   - 如果 y_i=+1 但预测为 -1：把 w 往 x_i 的正方向推
    #   - 如果 y_i=-1 但预测为 +1：把 w 往 x_i 的反方向推
    # ======================================================================
    def fit(self, X, y):
        """
        训练感知机模型。

        参数:
            X: np.ndarray, 形状 (n_samples, n_features)
            y: np.ndarray, 形状 (n_samples,)，标签为 -1 或 +1
        """
        n_samples, n_features = X.shape

        # 初始化权重和偏置
        self.w = np.random.randn(n_features) * 0.01
        self.b = 0.0
        self.losses = []

        for epoch in range(self.max_epochs):
            n_errors = 0

            for i in range(n_samples):
                x_i = X[i]
                y_i = y[i]

                # 计算线性得分 z = w·x_i + b
                z = np.dot(self.w, x_i) + self.b

                # 检查是否误分类
                if y_i * z <= 0:
                    # ==================================================
                    # TODO 3: 在这里实现感知机的权重更新规则
                    # ==================================================
                    # 提示：你需要更新 self.w 和 self.b
                    # 公式：w ← w + η * y_i * x_i
                    #       b ← b + η * y_i
                    # 其中 η 是 self.learning_rate
                    # ==================================================
                    pass  # <-- 替换为你的代码

                    n_errors += 1

            self.losses.append(n_errors)

            if n_errors == 0:
                print(f"感知机在第 {epoch + 1} 轮收敛！所有样本分类正确。")
                break

        print(f"训练完成。共 {epoch + 1} 轮。")
        print(f"最终权重 w = {self.w}, 偏置 b = {self.b:.4f}")

    def decision_function(self, X):
        """计算决策函数值（原始得分 w·x+b）。"""
        return np.dot(X, self.w) + self.b


def plot_results(model, X, y):
    """绘制决策边界和训练损失曲线。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 子图 1：决策边界
    ax = axes[0]
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='red', marker='o',
               edgecolors='k', s=60, label='类别 +1', alpha=0.7)
    ax.scatter(X[y == -1, 0], X[y == -1, 1], c='blue', marker='s',
               edgecolors='k', s=60, label='类别 -1', alpha=0.7)

    x_min, x_max = ax.get_xlim()
    w1, w2 = model.w[0], model.w[1]
    b_val = model.b
    x1_line = np.linspace(x_min, x_max, 100)
    if abs(w2) > 1e-10:  # 避免除零
        x2_line = -(w1 / w2) * x1_line - (b_val / w2)
        ax.plot(x1_line, x2_line, 'g-', linewidth=2, label='决策边界')

    ax.set_xlabel('特征 x1', fontsize=12)
    ax.set_ylabel('特征 x2', fontsize=12)
    ax.set_title('感知机决策边界', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 子图 2：损失曲线
    ax = axes[1]
    ax.plot(range(1, len(model.losses) + 1), model.losses,
            'b-o', markersize=4, linewidth=1.5)
    ax.set_xlabel('Epoch（训练轮数）', fontsize=12)
    ax.set_ylabel('误分类样本数', fontsize=12)
    ax.set_title('训练过程中的误分类数变化', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('perceptron_exercise_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n图片已保存为 perceptron_exercise_results.png")


def main():
    """主函数：测试你的实现是否正确。"""
    print("=" * 60)
    print("感知机练习 — 请完成代码中的 TODO 标记")
    print("=" * 60)

    # 1. 生成数据
    print("\n[1] 生成线性可分数据...")
    X, y = generate_linearly_separable_data(n_samples=100, random_seed=42)
    print(f"数据形状: X={X.shape}, y={y.shape}")
    print(f"类别分布: +1 有 {np.sum(y == 1)} 个, -1 有 {np.sum(y == -1)} 个")

    # 2. 训练模型
    print("\n[2] 训练感知机...")
    model = PerceptronExercise(learning_rate=0.1, max_epochs=500)
    model.fit(X, y)

    # 3. 评估
    print("\n[3] 评估模型...")
    y_pred = model.predict(X)
    accuracy = np.mean(y_pred == y)
    print(f"训练集准确率: {accuracy:.2%}")

    if accuracy == 1.0:
        print("✓ 完美！你的感知机实现正确，所有样本分类正确。")
    elif accuracy > 0.5:
        print("⚠ 你的感知机已经学到了一些规律，但还没有完全收敛。请检查 TODO 实现。")
    else:
        print("✗ 准确率太低。请检查你的 TODO 实现是否正确。")

    # 4. 可视化
    print("\n[4] 可视化结果...")
    plot_results(model, X, y)

    print("\n" + "=" * 60)
    print("练习结束！")
    print("=" * 60)


if __name__ == '__main__':
    main()
