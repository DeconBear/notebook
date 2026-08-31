# -*- coding: utf-8 -*-
"""
===============================================================================
ml03_naive_bayes/code/demo.py — 朴素贝叶斯与贝叶斯网络简介
===============================================================================
本演示实现三种朴素贝叶斯变体并进行对比实验：
  1. GaussianNB: 连续特征分类（从零实现）
  2. MultinomialNB: 离散计数特征分类 / 文本垃圾邮件检测
  3. 拉普拉斯平滑的效果展示
  4. 与 sklearn 的 GaussianNB / MultinomialNB 对比
  5. 三类变体在不同数据集上的性能比较

通过本演示，你将理解：
  - 条件独立假设如何将 O(d^2) 参数的协方差矩阵降为 O(d)
  - 三种朴素贝叶斯变体的适用场景
  - 拉普拉斯平滑如何防止零概率
  - 为什么"朴素"的假设在实际效果中出奇地好

作者：notebook 项目
日期：2025
===============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB as SkGaussianNB
from sklearn.naive_bayes import MultinomialNB as SkMultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
matplotlib.rcParams['axes.unicode_minus'] = False

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)

def _save_path(filename):
    return os.path.join(_IMAGES_DIR, filename)


# ============================================================================
# 第一部分: 自定义 GaussianNB 实现
# ============================================================================

class GaussianNB:
    """
    高斯朴素贝叶斯分类器。

    假设: P(x_i | class j) ~ N(mu_{ji}, sigma^2_{ji})
    即每个特征在每个类别下服从独立的一维高斯分布。

    与 ml02 完整高斯贝叶斯的区别：这里的协方差矩阵是对角的
    (Sigma = diag(sigma1^2, ..., sigma_d^2)), 大幅减少参数。

    参数:
        var_smoothing: float, 方差平滑项，防止方差为零
    """

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
        self.classes_ = None
        self.priors_ = None
        self.means_ = None
        self.vars_ = None

    def fit(self, X, y):
        """估计各类别的先验概率、均值和方差。"""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.priors_ = np.zeros(n_classes)
        self.means_ = np.zeros((n_classes, n_features))
        self.vars_ = np.zeros((n_classes, n_features))

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.priors_[idx] = len(X_c) / len(X)
            self.means_[idx] = np.mean(X_c, axis=0)
            # 方差: 使用有偏估计 (除以 n) 加上平滑项
            self.vars_[idx] = np.var(X_c, axis=0) + self.var_smoothing

        return self

    def _log_prob(self, X):
        """计算 log P(x_i | class j) for all i, j."""
        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        # log_joint: (n_samples, n_classes)
        log_joint = np.zeros((n_samples, n_classes))

        for idx in range(n_classes):
            # 对数一维高斯密度
            # log N(x|mu, sigma^2) = -0.5 * [log(2*pi*sigma^2) + (x-mu)^2/sigma^2]
            mu = self.means_[idx]
            var = self.vars_[idx]

            log_lik = -0.5 * (
                np.log(2 * np.pi * var) +
                ((X - mu) ** 2) / var
            )  # (n_samples, n_features)

            # 条件独立假设: log P(x|class) = sum_i log P(x_i|class)
            log_lik_sum = np.sum(log_lik, axis=1)  # (n_samples,)

            log_joint[:, idx] = log_lik_sum + np.log(self.priors_[idx])

        return log_joint

    def predict(self, X):
        """预测: 选 log_joint 最大的类别。"""
        X = np.asarray(X, dtype=np.float64)
        log_joint = self._log_prob(X)
        return self.classes_[np.argmax(log_joint, axis=1)]

    def predict_proba(self, X):
        """返回后验概率。使用 log-sum-exp 技巧保证数值稳定。"""
        from scipy.special import logsumexp
        X = np.asarray(X, dtype=np.float64)
        log_joint = self._log_prob(X)
        log_evidence = logsumexp(log_joint, axis=1, keepdims=True)
        log_posteriors = log_joint - log_evidence
        return np.exp(log_posteriors)


# ============================================================================
# 第二部分: 自定义 MultinomialNB 实现
# ============================================================================

class MultinomialNB:
    """
    多项朴素贝叶斯分类器。

    适用于非负整数特征（如词频计数、TF-IDF）。
    假设每个特征在给定类别下服从多项分布。

    P(x_i | class j) = (N_{ji} + alpha) / (N_j + alpha * n_features)

    N_{ji}: 类别 j 中特征 i 的总计数
    N_j: 类别 j 中所有特征的总计数

    参数:
        alpha: float, 拉普拉斯平滑参数 (0 = MLE, 1 = Laplace)
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes_ = None
        self.priors_ = None
        self.feature_log_prob_ = None  # log P(x_i|class)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.priors_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.priors_[idx] = len(X_c) / len(X)

            # 特征计数: sum over samples of each feature
            count_c = np.sum(X_c, axis=0)  # (n_features,)
            total_count = np.sum(count_c)

            # 带平滑的概率: (count + alpha) / (total + alpha * n_features)
            prob_c = (count_c + self.alpha) / (total_count + self.alpha * n_features)
            self.feature_log_prob_[idx] = np.log(prob_c)

        return self

    def predict(self, X):
        """预测 = sum_i x_i * log P(x_i|class) + log prior"""
        X = np.asarray(X, dtype=np.float64)
        # log_joint = X @ feature_log_prob_^T + log prior
        log_joint = X @ self.feature_log_prob_.T + np.log(self.priors_)
        return self.classes_[np.argmax(log_joint, axis=1)]


# ============================================================================
# 第三部分：数据生成辅助
# ============================================================================

def generate_spam_data():
    """生成模拟的垃圾邮件文本数据集（用于 MultinomialNB 演示）。"""
    np.random.seed(42)
    n_samples = 500

    # 垃圾邮件高频词
    spam_words = ['free', 'win', 'click', 'offer', 'prize', 'money', 'now', 'buy', 'limited', 'exclusive']
    # 正常邮件高频词
    ham_words = ['meeting', 'report', 'project', 'team', 'schedule', 'update', 'review', 'plan', 'discuss', 'work']

    vocabulary = spam_words + ham_words
    n_features = len(vocabulary)

    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        is_spam = np.random.rand() < 0.4  # 40% 垃圾邮件
        y[i] = 1 if is_spam else 0

        if is_spam:
            # 垃圾邮件: 前 10 个词出现概率高
            word_probs = np.array([0.3] * 10 + [0.02] * 10)
        else:
            # 正常邮件: 后 10 个词出现概率高
            word_probs = np.array([0.02] * 10 + [0.3] * 10)

        # 泊松采样模拟词频
        X[i] = np.random.poisson(lam=word_probs * 5)

    return X, y, vocabulary


# ============================================================================
# 第四部分：可视化函数
# ============================================================================

def plot_gaussian_nb_boundary():
    """绘制 GaussianNB 的决策边界（对比完整高斯贝叶斯的线性/二次边界）。"""
    np.random.seed(42)
    X, y = make_classification(n_samples=300, n_features=2,
                                n_redundant=0, n_clusters_per_class=2,
                                random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 自定义 GaussianNB
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    # sklearn GaussianNB
    sk_gnb = SkGaussianNB()
    sk_gnb.fit(X_train, y_train)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # 绘制决策边界
    for ax, model, name in zip(axes, [gnb, sk_gnb],
                                ['Custom GaussianNB (NumPy)', 'sklearn GaussianNB']):
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                            np.linspace(y_min, y_max, 200))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict(grid).reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='k',
                  cmap=plt.cm.RdYlBu, s=40)
        ax.set_xlabel('Feature 1', fontsize=11)
        ax.set_ylabel('Feature 2', fontsize=11)
        ax.set_title(f'{name}\n(Test Acc: {accuracy_score(y_test, model.predict(X_test)):.3f})',
                     fontsize=12, fontweight='bold')

    plt.tight_layout()
    fp = _save_path('gaussian_nb_boundary.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Done] GaussianNB boundary saved to {fp}")


def plot_laplace_smoothing_effect():
    """展示拉普拉斯平滑对 MultinomialNB 特征概率的影响。"""
    np.random.seed(42)
    # 生成一个极端的小样本数据集
    n_features = 6
    # class 0: 特征 0,1 大量出现; class 1: 特征 4,5 大量出现
    # 特征 2,3 几乎不出现 → 需要平滑
    X = np.array([
        [10, 8, 0, 0, 1, 1],
        [8, 7, 1, 0, 0, 2],
        [12, 9, 0, 0, 2, 0],
        [0, 1, 0, 1, 9, 8],
        [1, 0, 1, 0, 10, 7],
        [0, 0, 0, 1, 8, 9],
    ])
    y = np.array([0, 0, 0, 1, 1, 1])

    alphas = [0.001, 1.0, 5.0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, alpha in zip(axes, alphas):
        nb = MultinomialNB(alpha=alpha)
        nb.fit(X, y)

        x_pos = np.arange(n_features)
        width = 0.35

        bars0 = ax.bar(x_pos - width/2, np.exp(nb.feature_log_prob_[0]), width,
                       label='Class 0', color='steelblue', alpha=0.7)
        bars1 = ax.bar(x_pos + width/2, np.exp(nb.feature_log_prob_[1]), width,
                       label='Class 1', color='coral', alpha=0.7)

        ax.set_xlabel('Feature Index', fontsize=11)
        ax.set_ylabel('P(feature | class)', fontsize=11)
        ax.set_title(f'alpha = {alpha}\n' +
                    ('No smoothing (near-MLE)' if alpha < 0.01 else
                     'Laplace Smoothing' if alpha == 1.0 else
                     'Strong Smoothing'),
                     fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.legend(fontsize=9)

    plt.tight_layout()
    fp = _save_path('laplace_smoothing.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Done] Laplace smoothing saved to {fp}")


def plot_spam_detection_demo():
    """垃圾邮件检测演示: MultinomialNB vs sklearn。"""
    X, y, vocab = generate_spam_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 自定义 MultinomialNB (alpha=1, Laplace)
    custom_mnb = MultinomialNB(alpha=1.0)
    custom_mnb.fit(X_train, y_train)

    # sklearn MultinomialNB
    sk_mnb = SkMultinomialNB(alpha=1.0)
    sk_mnb.fit(X_train, y_train)

    custom_acc = accuracy_score(y_test, custom_mnb.predict(X_test))
    sk_acc = accuracy_score(y_test, sk_mnb.predict(X_test))

    print(f"\n[Spam Detection Results]")
    print(f"  Custom MultinomialNB Accuracy: {custom_acc:.4f}")
    print(f"  sklearn MultinomialNB Accuracy:  {sk_acc:.4f}")

    # 可视化 log 概率差异
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 垃圾邮件 vs 正常邮件的特征 log 概率
    for idx, (ax, class_name) in enumerate(zip(axes, ['Ham (Class 0)', 'Spam (Class 1)'])):
        log_probs = custom_mnb.feature_log_prob_[idx]
        probs = np.exp(log_probs)
        # 找 top-5 特征
        top5_idx = np.argsort(probs)[-5:][::-1]
        colors = ['coral' if vocab[i] in ['free', 'win', 'click', 'offer', 'prize'] else 'steelblue'
                  for i in range(len(vocab))]

        x_pos = np.arange(len(vocab))
        ax.bar(x_pos, probs, color=colors, alpha=0.7)
        ax.set_xlabel('Word Index', fontsize=11)
        ax.set_ylabel('P(word | class)', fontsize=11)
        ax.set_title(f'Feature Probabilities — {class_name}', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos[::2])
        ax.set_xticklabels([vocab[i] for i in range(0, len(vocab), 2)],
                          rotation=45, ha='right', fontsize=8)

    plt.tight_layout()
    fp = _save_path('spam_detection_probs.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Done] spam detection saved to {fp}")


def plot_model_comparison():
    """对比 GaussianNB vs MultinomialNB 在不同数据集上的表现。"""
    np.random.seed(42)
    n_runs = 10

    # 场景 1: 连续特征（GaussianNB 应该更好）
    gaussian_scores = []
    multinomial_scores = []

    for run in range(n_runs):
        X, y = make_classification(n_samples=300, n_features=5,
                                    n_informative=3, n_redundant=1,
                                    random_state=run)
        # MultinomialNB 需要非负特征
        X_mn = X - X.min(axis=0) + 0.1

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=run)
        X_tr_mn, X_te_mn, _, _ = train_test_split(X_mn, y, test_size=0.3, random_state=run)

        gnb = GaussianNB()
        gnb.fit(X_tr, y_tr)
        gaussian_scores.append(accuracy_score(y_te, gnb.predict(X_te)))

        mnb = MultinomialNB(alpha=1.0)
        mnb.fit(X_tr_mn, y_tr)
        multinomial_scores.append(accuracy_score(y_te_mn, mnb.predict(X_te_mn)))

    # 场景 2: 计数数据（MultinomialNB 应该更好）
    count_gaussian_scores = []
    count_multinomial_scores = []

    for run in range(n_runs):
        X, y = make_classification(n_samples=300, n_features=5,
                                    n_informative=3, random_state=run)
        X = np.abs(X) * 10  # 转为非负计数
        X = X.astype(int)

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=run)

        gnb = GaussianNB()
        gnb.fit(X_tr, y_tr)
        count_gaussian_scores.append(accuracy_score(y_te, gnb.predict(X_te)))

        mnb = MultinomialNB(alpha=1.0)
        mnb.fit(X_tr, y_tr)
        count_multinomial_scores.append(accuracy_score(y_te, mnb.predict(X_te_mn)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 箱线图
    data1 = [gaussian_scores, multinomial_scores]
    axes[0].boxplot(data1, labels=['GaussianNB', 'MultinomialNB'])
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Dataset: Continuous Features\n(GaussianNB expected to win)',
                      fontsize=12, fontweight='bold')

    data2 = [count_gaussian_scores, count_multinomial_scores]
    axes[1].boxplot(data2, labels=['GaussianNB', 'MultinomialNB'])
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Dataset: Count Data Features\n(MultinomialNB expected to win)',
                      fontsize=12, fontweight='bold')

    plt.tight_layout()
    fp = _save_path('model_type_comparison.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Done] model comparison saved to {fp}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 70)
    print("ml03_naive_bayes/demo.py — 朴素贝叶斯与贝叶斯网络")
    print("=" * 70)

    print("\n[1/4] 绘制 GaussianNB 决策边界...")
    plot_gaussian_nb_boundary()

    print("\n[2/4] 展示拉普拉斯平滑效果...")
    plot_laplace_smoothing_effect()

    print("\n[3/4] 垃圾邮件检测演示...")
    plot_spam_detection_demo()

    print("\n[4/4] 对比 GaussianNB vs MultinomialNB...")
    plot_model_comparison()

    print("\n" + "=" * 70)
    print("全部可视化完成! 图像保存在:", _IMAGES_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
