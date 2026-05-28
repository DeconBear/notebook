# -*- coding: utf-8 -*-
"""
===============================================================================
s02_linear_regression/code/exercise.py — 线性回归练习
===============================================================================
本练习文件中，线性回归的核心部分被替换为了 TODO 注释。
你需要完成以下任务：

练习目标：
  1. 计算 MSE 损失函数及其梯度
  2. 在梯度下降循环中更新权重和偏置
  3. 实现并比较 mini-batch GD 和 full-batch GD（Bonus）

提示：
   - MSE 损失: J(w,b) = (1/n) * Σ (ŷ_i - y_i)²
   - ∂J/∂w = (2/n) * Σ (ŷ_i - y_i) * x_i
   - ∂J/∂b = (2/n) * Σ (ŷ_i - y_i)
   - 更新规则: w ← w - η * ∂J/∂w, b ← b - η * ∂J/∂b
   - Mini-batch: 每次随机取 batch_size 个样本计算梯度

运行方式：
  python exercise.py
===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def generate_regression_data(n_samples=100, noise_std=3.0, random_seed=42):
    """生成 y = 2x + 5 + noise 的合成数据。"""
    np.random.seed(random_seed)
    X = np.random.uniform(low=0.0, high=10.0, size=n_samples)
    noise = np.random.randn(n_samples) * noise_std
    y = 2.0 * X + 5.0 + noise
    return X, y


class LinearRegressionExercise:
    """
    线性回归模型（练习版）。

    你需要完成损失计算、梯度计算和参数更新的实现。
    """

    def __init__(self, learning_rate=0.01, max_epochs=500):
        """初始化模型。"""
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.w = None
        self.b = None
        self.loss_history = []

    def predict(self, X):
        """预测：ŷ = w * X + b。"""
        return self.w * X + self.b

    # ======================================================================
    # TODO 1: 实现 MSE 损失的计算
    # ======================================================================
    # MSE (Mean Squared Error) 的定义：
    #   J(w, b) = (1/n) * Σ_{i=1}^{n} (ŷ_i - y_i)²
    #
    # 其中：
    #   - ŷ_i = w * x_i + b 是第 i 个样本的预测值
    #   - y_i 是第 i 个样本的真实值
    #   - n 是样本总数
    #
    # 提示：先用 self.predict(X) 得到所有预测值，
    #       然后计算每个预测值与真实值的平方差，最后取平均。
    #       可以使用 np.mean() 简化代码。
    # ======================================================================
    def _compute_loss(self, X, y):
        """
        计算 MSE 损失。

        参数:
            X: np.ndarray, 特征 (n_samples,)
            y: np.ndarray, 真实值 (n_samples,)

        返回:
            float, MSE 损失值
        """
        # TODO: 实现 MSE 损失计算
        # 步骤 1: 计算预测值 ŷ
        # 步骤 2: 计算每个样本的平方误差 (ŷ - y)²
        # 步骤 3: 对所有样本取平均
        pass  # <-- 替换为你的代码

    # ======================================================================
    # TODO 2: 实现梯度的计算
    # ======================================================================
    # MSE 损失对 w 和 b 的偏导数（梯度）：
    #   ∂J/∂w = (2/n) * Σ_{i=1}^{n} (ŷ_i - y_i) * x_i
    #   ∂J/∂b = (2/n) * Σ_{i=1}^{n} (ŷ_i - y_i)
    #
    # 其中 ŷ_i = w * x_i + b 是预测值，n 是样本数。
    #
    # 提示：先计算误差向量 errors = ŷ - y，
    #       然后分别求 dw 和 db。
    #       np.sum() 可以用于求和。
    # ======================================================================
    def _compute_gradients(self, X, y):
        """
        计算损失函数对 w 和 b 的梯度。

        参数:
            X: np.ndarray, 特征
            y: np.ndarray, 真实值

        返回:
            dw: float, ∂J/∂w
            db: float, ∂J/∂b
        """
        # TODO: 实现梯度计算
        # 步骤 1: 计算预测值 ŷ 和误差 errors = ŷ - y
        # 步骤 2: 计算 dw = (2/n) * Σ errors * x_i
        # 步骤 3: 计算 db = (2/n) * Σ errors
        pass  # <-- 替换为你的代码

    def fit(self, X, y, verbose=True):
        """梯度下降训练。"""
        self.w = np.random.randn() * 0.1
        self.b = np.random.randn() * 0.1
        self.loss_history = []

        for epoch in range(self.max_epochs):
            # 计算当前损失
            loss = self._compute_loss(X, y)
            self.loss_history.append(loss)

            # 计算梯度
            dw, db = self._compute_gradients(X, y)

            # ==============================================================
            # TODO 3: 使用梯度下降更新参数
            # ==============================================================
            # 梯度下降更新规则:
            #   w ← w - η * ∂J/∂w
            #   b ← b - η * ∂J/∂b
            #
            # 其中 η 是 self.learning_rate
            # 注意：这里是 -= 不是 +=，因为我们沿梯度反方向走
            # ==============================================================
            # TODO: 在这里实现参数更新
            pass  # <-- 替换为你的代码（两行：更新 self.w 和 self.b）

            # 判断是否收敛
            if len(self.loss_history) > 1:
                if abs(self.loss_history[-2] - loss) < 1e-6:
                    if verbose:
                        print(f"第 {epoch+1} 轮收敛！损失变化 < 1e-6")
                    break

            if verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1:4d}: loss={loss:.6f}, w={self.w:.4f}, b={self.b:.4f}")

        if verbose:
            print(f"\n训练完成，共 {epoch+1} 轮")
            print(f"学到的参数: w = {self.w:.4f}, b = {self.b:.4f} (真实值: w=2.0, b=5.0)")


# ==============================================================================
# TODO 4 (Bonus): 实现 Mini-batch 梯度下降
# ==============================================================================
# Mini-batch 梯度下降是批量 GD 和随机 GD 的折中方案：
# 每次不是用全部数据也不是用 1 个数据，而是随机取一小批（如 32 个）来计算梯度。
#
# 算法步骤:
#   1. 随机打乱数据顺序
#   2. 将数据分成若干个小批次（batches）
#   3. 对每个批次:
#       a. 用这个批次的数据计算梯度（公式同上，但只用 batch 内的样本）
#       b. 更新参数
#   4. 所有批次处理完毕 = 1 个 epoch
# ==============================================================================

class MiniBatchLinearRegression:
    """
    使用 Mini-batch 梯度下降的线性回归。

    与 full-batch GD 的区别：
      - Full-batch: 每个 epoch 用全部数据计算一次梯度
      - Mini-batch: 每个 epoch 将数据分成多个 batch，每个 batch 更新一次参数
    """

    def __init__(self, learning_rate=0.01, batch_size=32, max_epochs=200):
        """初始化。"""
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.w = None
        self.b = None
        self.loss_history = []

    def predict(self, X):
        return self.w * X + self.b

    def fit(self, X, y, verbose=True):
        """
        使用 Mini-batch 梯度下降训练。

        参数:
            X, y: 训练数据
            verbose: 是否打印日志
        """
        self.w = np.random.randn() * 0.1
        self.b = np.random.randn() * 0.1
        self.loss_history = []
        n = len(y)

        for epoch in range(self.max_epochs):
            # ==============================================================
            # TODO 4 (Bonus): 实现 mini-batch 的数据划分
            # ==============================================================
            # 步骤:
            #   a. 生成随机排列的索引 shuffle_idx = np.random.permutation(n)
            #   b. 用这些索引打乱 X 和 y
            #   c. 遍历数据，每次取 batch_size 个样本
            #   d. 对每个 batch 计算梯度并更新参数
            #
            # 提示：
            #   - range(0, n, self.batch_size) 可以按 batch 大小步进
            #   - X_batch = X_shuffled[start:start + self.batch_size]
            #   - 梯度公式与 full-batch 相同，只是用 batch 内数据计算
            # ==============================================================

            # TODO: 在这里实现 mini-batch 梯度下降的逻辑
            # 伪代码：
            #   1. shuffle_indices = np.random.permutation(n)
            #   2. X_shuffled = X[shuffle_indices]
            #   3. y_shuffled = y[shuffle_indices]
            #   4. for start in range(0, n, self.batch_size):
            #          end = start + self.batch_size
            #          X_batch = X_shuffled[start:end]
            #          y_batch = y_shuffled[start:end]
            #          用 X_batch, y_batch 计算梯度并更新参数
            pass  # <-- 替换为你的代码

            # 记录整个 epoch 的损失（用全部数据评估）
            total_loss = np.mean((self.predict(X) - y) ** 2)
            self.loss_history.append(total_loss)

            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1:4d}: loss={total_loss:.6f}, "
                      f"w={self.w:.4f}, b={self.b:.4f}")

        if verbose:
            print(f"\nMini-batch 训练完成，共 {epoch+1} 轮")
            print(f"学到的参数: w = {self.w:.4f}, b = {self.b:.4f}")


def main():
    """主函数：测试你的实现。"""
    print("=" * 60)
    print("线性回归练习 — 请完成代码中的 TODO 标记")
    print("=" * 60)

    # 1. 生成数据
    print("\n[1] 生成数据...")
    X, y = generate_regression_data(n_samples=200, random_seed=42)
    print(f"真实参数: w_true = 2.0, b_true = 5.0")

    # 2. 测试你的 Full-batch GD 实现
    print("\n[2] 测试 Full-batch 梯度下降...")
    model = LinearRegressionExercise(learning_rate=0.01, max_epochs=500)
    model.fit(X, y)

    # 3. 评估
    if model.w is not None:
        w_err = abs(model.w - 2.0)
        b_err = abs(model.b - 5.0)
        print(f"\n参数误差: |w - w_true| = {w_err:.4f}, |b - b_true| = {b_err:.4f}")
        if w_err < 0.1 and b_err < 0.5:
            print("✓ 参数接近真实值，你的实现基本正确！")
        else:
            print("⚠ 参数与真实值有较大偏差，请检查 TODO 实现。")

    # 4. Bonus: 测试 mini-batch GD
    print("\n[3] 测试 Mini-batch 梯度下降 (Bonus)...")
    model_mb = MiniBatchLinearRegression(learning_rate=0.01, batch_size=32, max_epochs=200)
    model_mb.fit(X, y)

    # 5. 可视化对比
    print("\n[4] 可视化结果...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：数据散点和拟合直线
    ax = axes[0]
    ax.scatter(X, y, c='steelblue', alpha=0.6, s=30, edgecolors='white', linewidth=0.3)
    X_line = np.linspace(X.min(), X.max(), 200)

    if model.w is not None and model.b is not None:
        ax.plot(X_line, model.predict(X_line), 'r-', linewidth=2,
                label=f'Full-batch GD: ŷ={model.w:.2f}x+{model.b:.2f}')
    if model_mb.w is not None and model_mb.b is not None:
        ax.plot(X_line, model_mb.predict(X_line), 'g--', linewidth=2,
                label=f'Mini-batch GD: ŷ={model_mb.w:.2f}x+{model_mb.b:.2f}')

    ax.set_xlabel('特征 x', fontsize=12)
    ax.set_ylabel('目标值 y', fontsize=12)
    ax.set_title('线性回归拟合结果对比', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 右图：损失曲线对比
    ax = axes[1]
    if len(model.loss_history) > 0:
        ax.plot(range(1, len(model.loss_history) + 1), model.loss_history,
                'r-', linewidth=1.5, label='Full-batch GD')
    if len(model_mb.loss_history) > 0:
        ax.plot(range(1, len(model_mb.loss_history) + 1), model_mb.loss_history,
                'g-', linewidth=1.5, label='Mini-batch GD')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE 损失', fontsize=12)
    ax.set_title('损失曲线对比', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig('linear_regression_exercise_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("图片已保存为 linear_regression_exercise_results.png")

    print("\n" + "=" * 60)
    print("练习结束！")
    print("=" * 60)


if __name__ == '__main__':
    main()
