# -*- coding: utf-8 -*-
"""
ml09 降维与特征工程 — 演示代码
===============================
功能：从零实现 PCA（SVD 方法），对比 PCA/t-SNE/LDA 在 MNIST 和合成数据上的降维效果，
      展示特征工程 pipeline（分箱、交互特征、目标编码）。

每个函数都有中文 docstring，每行逻辑代码都有中文注释。
运行方式：在 ml09_dimensionality_reduction/ 目录下执行 python code/demo.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
from sklearn.datasets import load_digits, make_classification
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE

_HERE = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_HERE, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)


# ============================================================================
# 第一部分：PCA 从零实现（SVD 方法）
# ============================================================================

class PCA:
    """
    主成分分析（PCA）——从零实现。

    使用 SVD（奇异值分解）对中心化后的数据矩阵进行降维。

    数学推导：
        1. 数据中心化：X_centered = X - mean(X)
        2. SVD 分解：X_centered = U Σ V^T
        3. V 的前 k 列 = 主成分方向（协方差矩阵的特征向量）
        4. 投影：Z = X_centered @ V_k

    参数:
        n_components: 保留的主成分数量
    """

    def __init__(self, n_components=2):
        self.n_components = n_components
        self.mean_ = None          # 训练数据的均值，用于 center
        self.components_ = None    # 主成分方向，shape (n_components, d)
        self.explained_variance_ratio_ = None  # 各主成分的方差解释率

    def fit(self, X):
        """计算主成分方向"""
        # 步骤 1：数据中心化
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_  # (n, d)

        # 步骤 2：SVD 分解
        # SVD: X_centered = U @ Sigma @ Vt
        # Vt 的行 = 右奇异向量 = X_centered 的协方差矩阵的特征向量
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # 步骤 3：取前 k 个右奇异向量作为主成分方向
        self.components_ = Vt[:self.n_components]  # (k, d)

        # 步骤 4：计算方差解释率
        # 特征值 = S^2 / (n - 1)，方差解释率 = lambda_j / sum(lambda)
        eigenvalues = S ** 2 / (X.shape[0] - 1)  # (n_components,)
        total_var = np.sum(eigenvalues) if len(S) > self.n_components else np.sum(eigenvalues)
        # 注意：需要所有奇异值的平方和，但 full_matrices=False 只返回前 min(n,d) 个
        # 用数据方差做近似
        total_var_fallback = X_centered.var(axis=0).sum()
        self.explained_variance_ratio_ = eigenvalues / total_var_fallback

    def transform(self, X):
        """将数据投影到主成分空间"""
        X_centered = X - self.mean_
        return X_centered @ self.components_.T  # (n, k)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# ============================================================================
# 第二部分：可视化
# ============================================================================

def plot_pca_explained_variance(X, title='PCA Explained Variance'):
    """绘制 PCA 方差解释率条形图 + 累积折线图"""
    n_components = min(50, X.shape[0], X.shape[1])

    # 用 sklearn 的 PCA 获得完整的方差解释率
    from sklearn.decomposition import PCA as SkPCA
    pca = SkPCA(n_components=n_components)
    pca.fit(X)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # 柱状图：单独解释率
    bars = ax1.bar(range(1, n_components + 1), pca.explained_variance_ratio_,
                   alpha=0.7, color='steelblue', label='Individual')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_ylim(0, max(pca.explained_variance_ratio_) * 1.2)

    # 折线图：累积解释率
    ax2 = ax1.twinx()
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    ax2.plot(range(1, n_components + 1), cumsum, 'ro-', markersize=4,
             linewidth=2, label='Cumulative')
    ax2.set_ylabel('Cumulative Explained Variance Ratio')
    ax2.set_ylim(0, 1.05)

    # 标注 90% 和 95% 阈值
    for threshold, color in [(0.90, 'green'), (0.95, 'orange')]:
        n_needed = np.argmax(cumsum >= threshold) + 1
        ax2.axhline(y=threshold, color=color, linestyle='--', alpha=0.7, linewidth=1.5)
        ax2.annotate(f'{threshold*100:.0f}% → {n_needed} PCs', xy=(n_needed, threshold),
                     xytext=(n_needed + 3, threshold + 0.03),
                     fontsize=10, color=color,
                     arrowprops=dict(arrowstyle='->', color=color))

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(_IMAGES_DIR, 'ml09-05-pca-variance.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[保存] {path}')


def plot_pca_vs_tsne_vs_lda(X, y, title_prefix=''):
    """在同一数据上对比 PCA、t-SNE、LDA 的降维效果"""
    n_classes = len(np.unique(y))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    method_info = [
        ('PCA', 'PCA (unsupervised)'),
        ('t-SNE', 't-SNE (perplexity=30)'),
        (f'LDA ({n_classes-1}D)', 'LDA (supervised)'),
    ]

    for ax, (method, subtitle) in zip(axes, method_info):
        if method == 'PCA':
            X_trans = PCA(n_components=2).fit_transform(X)
        elif method == 't-SNE':
            tsne = TSNE(n_components=2, perplexity=30, random_state=42)
            X_trans = tsne.fit_transform(X)
        else:  # LDA
            # LDA 最多降到 C-1 维，需要至少 2 类
            if n_classes >= 3:
                lda = LDA(n_components=2)
                X_trans = lda.fit_transform(X, y)
            else:
                lda = LDA(n_components=1)
                X_trans_1d = lda.fit_transform(X, y)
                X_trans = np.hstack([X_trans_1d, np.zeros_like(X_trans_1d)])  # 补零填充到 2D

        scatter = ax.scatter(X_trans[:, 0], X_trans[:, 1], c=y, cmap='tab10',
                             s=15, alpha=0.7, edgecolors='none')
        ax.set_title(f'{subtitle}\n{title_prefix}', fontsize=11)
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    path = os.path.join(_IMAGES_DIR, 'ml09-06-pca-tsne-lda-comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[保存] {path}')


# ============================================================================
# 第三部分：特征工程 Demo
# ============================================================================

def demo_feature_engineering():
    """
    展示一个完整的特征工程 pipeline：
    原始数据 → 分箱 → 交互特征 → 目标编码 → 标准化 → PCA → 模型
    """
    print('\n[特征工程 Demo]')

    # 创建模拟数据：房价预测任务
    np.random.seed(42)
    n_samples = 500

    # 原始特征
    area = np.random.gamma(shape=2, scale=50, size=n_samples)     # 面积（偏态分布）
    rooms = np.random.randint(1, 6, size=n_samples)                # 房间数
    age = np.random.randint(0, 50, size=n_samples)                 # 房龄
    district = np.random.choice(['A', 'B', 'C', 'D'], size=n_samples, p=[0.3, 0.3, 0.2, 0.2])

    # 目标：房价（非线性组合）
    price = (area * 2.5 + rooms * 15 - age * 0.8
             + np.where(district == 'A', 50, 0)
             + np.where(district == 'B', 20, 0)
             + area * rooms * 0.3  # 交互效应
             + np.random.randn(n_samples) * 10)

    # 构建 DataFrame
    import pandas as pd
    df = pd.DataFrame({'area': area, 'rooms': rooms, 'age': age, 'district': district})
    X_original = df.copy()

    # ---- 特征 1: 分箱（Age Binning） ----
    # 将房龄分为"新/中/旧"三个箱子，赋予线性模型非线性能力
    binner = KBinsDiscretizer(n_bins=3, encode='onehot-dense', strategy='uniform')
    age_binned = binner.fit_transform(df[['age']])

    # ---- 特征 2: 交互特征 ----
    # 面积×房间数 → 总面积
    df['area_x_rooms'] = df['area'] * df['rooms']
    # 每房间面积
    df['area_per_room'] = df['area'] / df['rooms']

    # ---- 特征 3: 目标编码（Target Encoding） ----
    # 对高基数类别特征 'district' 做目标编码
    global_mean = price.mean()
    target_encoding = {}
    alpha = 10  # 平滑参数

    for d in df['district'].unique():
        mask = df['district'] == d
        n_d = mask.sum()
        cat_mean = price[mask].mean()
        # 经验贝叶斯收缩: 样本少时向全局均值收缩
        target_encoding[d] = (n_d * cat_mean + alpha * global_mean) / (n_d + alpha)

    df['district_te'] = df['district'].map(target_encoding)

    # ---- 组合特征矩阵 ----
    X_fe = np.column_stack([
        df[['area', 'rooms', 'age']].values,  # 原始连续特征
        age_binned,                            # 分箱特征（3列 one-hot）
        df[['area_x_rooms', 'area_per_room']].values,  # 交互特征
        df['district_te'].values[:, None],     # 目标编码
    ])
    y = price

    # ---- 训练模型并对比 ----
    X_train, X_test, y_train, y_test = train_test_split(
        X_fe, y, test_size=0.2, random_state=42)

    # 仅用原始特征
    X_train_orig = df[['area', 'rooms', 'age']].values
    X_test_orig = X_test[:, :3]

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_orig_scaled = scaler.fit_transform(X_train_orig) if StandardScaler().fit_transform(X_train_orig).shape[1] > 0 else X_train_orig
    # 重新标准化原始特征
    scaler2 = StandardScaler()
    X_train_orig_scaled = scaler2.fit_transform(X_train_orig)
    X_test_orig_scaled = scaler2.transform(X_test_orig)

    # 逻辑回归（转为二分类：高于中位数 vs 低于中位数）
    y_median = np.median(y)
    y_train_bin = (y_train > y_median).astype(int)
    y_test_bin = (y_test > y_median).astype(int)

    # 模型 1: 仅用原始特征
    lr1 = LogisticRegression(max_iter=1000)
    lr1.fit(X_train_orig_scaled, y_train_bin)
    acc1 = accuracy_score(y_test_bin, lr1.predict(X_test_orig_scaled))

    # 模型 2: 使用特征工程后的特征
    lr2 = LogisticRegression(max_iter=1000)
    lr2.fit(X_train_scaled, y_train_bin)
    acc2 = accuracy_score(y_test_bin, lr2.predict(X_test_scaled))

    # 模型 3: 特征工程 + PCA 降维
    pca = PCA(n_components=5)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    lr3 = LogisticRegression(max_iter=1000)
    lr3.fit(X_train_pca, y_train_bin)
    acc3 = accuracy_score(y_test_bin, lr3.predict(X_test_pca))

    print(f'  模型1 (原始特征):      Accuracy={acc1:.4f} (特征数={X_train_orig.shape[1]})')
    print(f'  模型2 (特征工程):      Accuracy={acc2:.4f} (特征数={X_train.shape[1]})')
    print(f'  模型3 (特征工程+PCA):  Accuracy={acc3:.4f} (特征数=5)')
    print(f'  特征工程贡献:         +{acc2-acc1:.4f}')

    return X_fe, y


# ============================================================================
# 第四部分：主程序
# ============================================================================

def main():
    print('=' * 60)
    print('ml09 降维与特征工程 — 演示代码')
    print('=' * 60)

    # 1. PCA 从零实现验证
    print('\n[1/5] PCA 从零实现（SVD 方法）...')
    from sklearn.datasets import make_blobs
    X_syn, _ = make_blobs(n_samples=200, n_features=10, random_state=42)
    X_scaled = StandardScaler().fit_transform(X_syn)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    print(f'  原始维度: {X_scaled.shape[1]} → 降维后: {X_pca.shape[1]}')
    print(f'  前2主成分方差解释率: {pca.explained_variance_ratio_}')
    print(f'  累积解释率: {pca.explained_variance_ratio_.sum():.4f}')

    # 2. PCA 方差解释率图
    print('[2/5] PCA 方差解释率分析...')
    plot_pca_explained_variance(X_scaled, title='PCA Explained Variance (Synthetic 10D Data)')

    # 3. 加载 MNIST digits 子集，对比 PCA / t-SNE / LDA
    print('[3/5] 加载 digits 数据，对比降维方法...')
    digits = load_digits()
    X_digits = digits.data
    y_digits = digits.target

    # 只取前 5 类以加速 t-SNE
    mask = y_digits < 5
    X_sub = X_digits[mask]
    y_sub = y_digits[mask]
    X_sub_scaled = StandardScaler().fit_transform(X_sub)

    print(f'  数据: {X_sub.shape[0]} 样本, {X_sub.shape[1]} 特征, {len(np.unique(y_sub))} 类')
    plot_pca_vs_tsne_vs_lda(X_sub_scaled, y_sub, title_prefix='Digits (subset 5 classes)')

    # 4. 特征工程 Demo
    print('[4/5] 特征工程 Pipeline Demo...')
    demo_feature_engineering()

    # 5. 使用 sklearn LDA 进行分类前置降维
    print('\n[5/5] LDA 降维 + 分类实验...')
    X_clf, y_clf = make_classification(n_samples=500, n_features=20, n_informative=8,
                                        n_classes=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X_clf, y_clf, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 对比：无降维 vs PCA vs LDA
    for name, reducer in [('None (20D)', None),
                           ('PCA (5D)', PCA(n_components=5)),
                           ('LDA (2D)', LDA(n_components=2))]:
        if reducer is None:
            X_tr, X_te = X_train_s, X_test_s
        else:
            if name.startswith('LDA'):
                X_tr = reducer.fit_transform(X_train_s, y_train)
                X_te = reducer.transform(X_test_s)
            else:
                X_tr = reducer.fit_transform(X_train_s)
                X_te = reducer.transform(X_test_s)

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_tr, y_train)
        acc = accuracy_score(y_test, clf.predict(X_te))
        print(f'  {name:>12s}: 测试准确率={acc:.4f}')

    print('\n完成！所有图表已保存到 images/ 目录。')


if __name__ == '__main__':
    main()
