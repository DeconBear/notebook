# -*- coding: utf-8 -*-
"""
===============================================================================
ml02_bayesian_decision/code/demo.py — 贝叶斯决策理论
===============================================================================
本演示通过实现高斯贝叶斯分类器，全面展示：
  1. 贝叶斯定理：先验概率、似然函数、后验概率的关系
  2. 三种协方差矩阵假设下的判别函数与决策面
  3. 最小错误率决策 vs 最小风险决策
  4. ROC 曲线、AUC 和 EER 的计算与可视化
  5. 与 sklearn 的 GaussianNB / LDA / QDA 对比

通过本演示，你将理解：
  - 贝叶斯决策如何将分类问题转化为概率推理
  - 协方差矩阵假设如何影响决策边界形态
  - 不对称损失如何改变最优决策
  - ROC/AUC 对分类器性能的评估含义

作者：learn-ai 项目
日期：2025
===============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.special import logsumexp
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import roc_curve, auc, roc_auc_score
matplotlib.rcParams['axes.unicode_minus'] = False

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)

def _save_path(filename):
    return os.path.join(_IMAGES_DIR, filename)


# ============================================================================
# 第一部分：高斯贝叶斯分类器（从零实现）
# ============================================================================

class GaussianBayesClassifier:
    """
    高斯贝叶斯分类器：假设每个类别的类条件密度为多元高斯分布。

    支持三种协方差类型：
      - 'isotropic': Sigma_j = sigma^2 * I（各向同性，各类别共享）→ 最近均值分类器
      - 'shared': Sigma_j = Sigma（各类别共享同一个协方差矩阵）→ 类似 LDA
      - 'full': 各类别有自己独立的协方差矩阵 → 类似 QDA

    参数:
        cov_type: str, 'isotropic', 'shared', 或 'full'
    """

    def __init__(self, cov_type='full'):
        self.cov_type = cov_type
        self.priors_ = None
        self.means_ = None
        self.covs_ = None
        self.classes_ = None

    def fit(self, X, y):
        """
        参数估计:
          1. priors: P(omega_j) = n_j / n
          2. means: mu_j = (1/n_j) * sum(x_i in class j)
          3. covs: 根据 cov_type 计算 Sigma_j
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.priors_ = np.zeros(n_classes)
        self.means_ = np.zeros((n_classes, n_features))

        # 收集每个类别的均值和样本
        class_data = {}
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            class_data[c] = X_c
            self.priors_[idx] = len(X_c) / len(X)
            self.means_[idx] = np.mean(X_c, axis=0)

        # 根据 cov_type 估计协方差
        if self.cov_type == 'isotropic':
            # 各类别共享各向同性协方差: sigma^2 * I
            total_var = 0.0
            total_n = 0
            for idx, c in enumerate(self.classes_):
                X_c = class_data[c]
                mean_c = self.means_[idx]
                total_var += np.sum((X_c - mean_c) ** 2)
                total_n += len(X_c)
            sigma2 = total_var / (total_n * n_features)
            self.covs_ = [sigma2 * np.eye(n_features) for _ in range(n_classes)]

        elif self.cov_type == 'shared':
            # 各类别共享同一个协方差矩阵 (pooled)
            pooled_cov = np.zeros((n_features, n_features))
            for idx, c in enumerate(self.classes_):
                X_c = class_data[c]
                mean_c = self.means_[idx]
                pooled_cov += (X_c - mean_c).T @ (X_c - mean_c)
            pooled_cov /= len(X)
            self.covs_ = [pooled_cov for _ in range(n_classes)]

        elif self.cov_type == 'full':
            # 各类别有自己独立的协方差矩阵
            self.covs_ = []
            for idx, c in enumerate(self.classes_):
                X_c = class_data[c]
                mean_c = self.means_[idx]
                # 使用无偏估计，加上微小正则化防止奇异
                cov_c = np.cov(X_c.T, bias=False)
                cov_c += 1e-6 * np.eye(n_features)
                self.covs_.append(cov_c)

        return self

    def _log_likelihood(self, X):
        """计算每个样本属于每个类别的 log P(x | omega_j)。"""
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        d = X.shape[1]
        log_lik = np.zeros((n_samples, n_classes))

        for idx in range(n_classes):
            mu = self.means_[idx]
            cov = self.covs_[idx]

            # 计算 cov 的逆和行列式的对数
            # 对于 isotropic 情况: cov = sigma^2 * I
            if self.cov_type == 'isotropic':
                sigma2 = cov[0, 0]
                # log N(x|mu, sigma^2 I) = -(d/2)log(2*pi) - (d/2)log(sigma^2)
                #                            - (1/(2*sigma^2)) * ||x-mu||^2
                log_det = d * np.log(sigma2)
                # 马氏距离 = ||x-mu||^2 / sigma^2
                diff = X - mu
                mahal = np.sum(diff ** 2, axis=1) / sigma2
            else:
                # 用伪逆和 slogdet 保证数值稳定性
                try:
                    cov_inv = np.linalg.pinv(cov)
                    sign, log_det = np.linalg.slogdet(cov)
                except Exception:
                    cov_inv = np.linalg.pinv(cov)
                    log_det = 0.0

                diff = X - mu
                # 马氏距离: (x-mu)^T Sigma^{-1} (x-mu)
                # 向量化: sum_j (diff @ cov_inv)_j * diff_j
                mahal = np.sum((diff @ cov_inv) * diff, axis=1)

            # log 高斯密度
            log_gauss = -0.5 * (d * np.log(2 * np.pi) + log_det + mahal)
            log_lik[:, idx] = log_gauss

        return log_lik

    def predict_proba(self, X):
        """返回后验概率 P(omega_j | x)。"""
        X = np.asarray(X, dtype=np.float64)
        log_lik = self._log_likelihood(X)
        log_prior = np.log(self.priors_)

        # log joint = log P(x|omega_j) + log P(omega_j)
        log_joint = log_lik + log_prior

        # log evidence = log sum_j exp(log_joint_j)
        # 使用 logsumexp 保证数值稳定性
        log_evidence = logsumexp(log_joint, axis=1, keepdims=True)

        # log posterior = log_joint - log_evidence
        log_posteriors = log_joint - log_evidence
        return np.exp(log_posteriors)

    def predict(self, X):
        """预测: 选择后验概率最大的类别。"""
        log_lik = self._log_likelihood(X)
        log_prior = np.log(self.priors_)
        log_joint = log_lik + log_prior
        best_idx = np.argmax(log_joint, axis=1)
        return self.classes_[best_idx]


# ============================================================================
# 第二部分：最小风险决策的演示
# ============================================================================

def minimum_risk_predict(posteriors, loss_matrix):
    """
    最小风险决策: 对每个样本，选择使条件风险最小的行动。

    R(alpha_i | x) = sum_j loss_matrix[i, j] * P(omega_j | x)
    alpha* = argmin_i R(alpha_i | x)

    参数:
        posteriors: (n_samples, n_classes), P(omega_j | x)
        loss_matrix: (n_classes, n_classes), loss_matrix[i, j] = cost of
                     predicting class i when true class is j
    返回:
        predictions: (n_samples,)
    """
    # 条件风险: (n_samples, n_classes) = posteriors @ loss_matrix.T
    # 简化版本: 每个样本的每个决策的风险
    n_samples, n_classes = posteriors.shape
    risks = np.zeros((n_samples, n_classes))

    for i in range(n_classes):
        for j in range(n_classes):
            risks[:, i] += loss_matrix[i, j] * posteriors[:, j]

    return np.argmin(risks, axis=1)


# ============================================================================
# 第三部分：可视化函数
# ============================================================================

def plot_decision_boundaries_cov_types():
    """
    对比三种协方差假设下的决策边界。
    """
    np.random.seed(42)
    # 生成三类数据，各类别有不同协方差结构
    n_samples = 150

    # 类别 0: 圆形分布
    X0 = np.random.randn(n_samples, 2) * 1.0 + np.array([-4, 0])
    # 类别 1: 拉长 + 旋转的椭圆
    rot = np.array([[1.5, 0.8], [0.8, 0.5]])
    X1 = np.random.randn(n_samples, 2) @ rot + np.array([2, 2])
    # 类别 2: 另一个方向的椭圆
    rot2 = np.array([[0.8, -0.8], [-0.8, 1.5]])
    X2 = np.random.randn(n_samples, 2) @ rot2 + np.array([0, -3])

    X = np.vstack([X0, X1, X2])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples), 2 * np.ones(n_samples)])

    cov_types = ['isotropic', 'shared', 'full']
    titles = [
        'Isotropic Covariance\n(Same spherical for all classes)',
        'Shared Covariance\n(Same ellipsoid for all classes)',
        'Full Covariance\n(Each class has own ellipsoid)'
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for ax, cov_type, title in zip(axes, cov_types, titles):
        clf = GaussianBayesClassifier(cov_type=cov_type)
        clf.fit(X, y)

        # 管网格预测
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                            np.linspace(y_min, y_max, 200))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = clf.predict(grid).reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
        ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdYlBu, s=30)
        # 绘制各类别的均值和协方差椭圆
        for idx in range(3):
            mu = clf.means_[idx]
            ax.scatter(mu[0], mu[1], c='lime', marker='X', s=200, edgecolors='black', linewidths=1.5)

        ax.set_xlabel('Feature 1', fontsize=11)
        ax.set_ylabel('Feature 2', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')

    plt.tight_layout()
    fp = _save_path('cov_type_boundaries.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Done] covariance type boundaries saved to {fp}")


def plot_risk_decision_demo():
    """
    演示最小风险决策：非对称损失矩阵导致不同的决策边界。
    """
    np.random.seed(42)
    X, y = make_blobs(n_samples=300, centers=2, cluster_std=[2.0, 1.5],
                      center_box=(-5, 5), random_state=42)
    X = StandardScaler().fit_transform(X)

    # 训练高斯贝叶斯分类器
    clf = GaussianBayesClassifier(cov_type='full')
    clf.fit(X, y)
    posteriors = clf.predict_proba(X)

    # 损失矩阵设置（疾病诊断场景）
    # class 0 = 良性, class 1 = 恶性
    # loss[0,0]=0 (良性→良性), loss[0,1]=5 (良性→恶性, 漏诊代价高!)
    # loss[1,0]=1 (恶性→良性, 误报代价低), loss[1,1]=0 (恶性→恶性)
    loss_symmetric = np.array([[0, 1],
                               [1, 0]])  # 0-1 loss = 最小错误率

    loss_asymmetric = np.array([[0, 5],
                                [1, 0]])  # 漏诊代价 5 倍于误报

    pred_sym = minimum_risk_predict(posteriors, loss_symmetric)
    pred_asym = minimum_risk_predict(posteriors, loss_asymmetric)

    # 绘制
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    titles = [
        'Minimum Error Rate Decision\n(Symmetric 0-1 Loss)',
        'Minimum Risk Decision\n(FN cost = 5x FP cost)'
    ]
    preds = [pred_sym, pred_asym]

    for ax, title, pred in zip(axes, titles, preds):
        ax.scatter(X[:, 0], X[:, 1], c=pred, cmap=plt.cm.RdYlBu, edgecolors='k', s=40, alpha=0.8)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='gray',
                  s=40, alpha=0.2, marker='s')
        ax.set_xlabel('Feature 1', fontsize=11)
        ax.set_ylabel('Feature 2', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')

    plt.tight_layout()
    fp = _save_path('risk_decision_comparison.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Done] risk decision saved to {fp}")


def plot_roc_auc_eer():
    """
    绘制 ROC 曲线，计算 AUC 和 EER。
    """
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=20,
                                n_informative=8, n_redundant=2,
                                random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 训练多个模型
    models = {
        'Gaussian NB (Isotropic)': GaussianBayesClassifier(cov_type='isotropic'),
        'Gaussian NB (Full)': GaussianBayesClassifier(cov_type='full'),
        'sklearn GNB': GaussianNB(),
        'sklearn LDA': LinearDiscriminantAnalysis(),
        'sklearn QDA': QuadraticDiscriminantAnalysis(),
    }

    fig, ax = plt.subplots(figsize=(8, 7))

    for name, model in models.items():
        model.fit(X_train, y_train)

        # 获取预测概率（对于我们的自定义模型用 predict_proba）
        if hasattr(model, 'predict_proba'):
            y_score = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_score = model.decision_function(X_test)
        else:
            continue

        # ROC 曲线
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)

        # 绘制
        ax.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC={roc_auc:.3f})')

        # EER: 找到 FPR ≈ FNR 的点
        fnr = 1 - tpr
        eer_idx = np.argmin(np.abs(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
        ax.scatter(fpr[eer_idx], tpr[eer_idx], s=80, zorder=5)

    # 对角线 (随机猜测)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random (AUC=0.5)')
    ax.fill_between([0, 1], [0, 1], [1, 1], alpha=0.05, color='green')

    ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
    ax.set_ylabel('True Positive Rate (TPR / Recall)', fontsize=12)
    ax.set_title('ROC Curves — Multiple Bayesian Classifiers\n(Dots mark EER points)',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    plt.tight_layout()
    fp = _save_path('roc_auc_multi_model.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Done] ROC/AUC plot saved to {fp}")


def plot_posterior_3d():
    """
    一维特征的贝叶斯决策三维可视化（后验概率面）。
    """
    np.random.seed(42)
    # 两类高斯分布，一维特征
    n_samples = 100
    X0 = np.random.randn(n_samples) * 1.0 - 2.0  # class 0
    X1 = np.random.randn(n_samples) * 1.5 + 1.5  # class 1

    X = np.hstack([X0, X1]).reshape(-1, 1)
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])

    clf = GaussianBayesClassifier(cov_type='full')
    clf.fit(X, y)

    # 计算后验概率
    x_line = np.linspace(-8, 8, 500).reshape(-1, 1)
    posteriors = clf.predict_proba(x_line)
    likelihood_x = clf._log_likelihood(x_line)
    likelihood = np.exp(likelihood_x)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图: 似然函数和先验
    ax = axes[0]
    prior0 = clf.priors_[0]
    prior1 = clf.priors_[1]
    ax.plot(x_line, likelihood[:, 0] * prior0, 'b-', linewidth=2,
            label=f'P(x|Class 0) * P(Class 0) = likelihood*prior')
    ax.plot(x_line, likelihood[:, 1] * prior1, 'r-', linewidth=2,
            label=f'P(x|Class 1) * P(Class 1)')
    ax.fill_between(x_line.ravel(), 0, likelihood[:, 0] * prior0, alpha=0.3, color='blue')
    ax.fill_between(x_line.ravel(), 0, likelihood[:, 1] * prior1, alpha=0.3, color='red')
    # 决策边界: 两曲线交点
    ax.set_xlabel('Feature x', fontsize=12)
    ax.set_ylabel('Likelihood * Prior', fontsize=12)
    ax.set_title('Class-conditional Densities * Priors\n(Decision boundary at intersection)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)

    # 右图: 后验概率
    ax = axes[1]
    ax.plot(x_line, posteriors[:, 0], 'b-', linewidth=2, label='P(Class 0 | x)')
    ax.plot(x_line, posteriors[:, 1], 'r-', linewidth=2, label='P(Class 1 | x)')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    # 标注决策点: P(Class 0 | x) = P(Class 1 | x) = 0.5
    ax.fill_between(x_line.ravel(), 0, posteriors[:, 0], alpha=0.3, color='blue')
    ax.fill_between(x_line.ravel(), 0, posteriors[:, 1], alpha=0.3, color='red')
    ax.set_xlabel('Feature x', fontsize=12)
    ax.set_ylabel('Posterior Probability', fontsize=12)
    ax.set_title('Posterior Probabilities\n(P(Class 0|x) + P(Class 1|x) = 1 everywhere)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)

    plt.tight_layout()
    fp = _save_path('bayesian_posterior_demo.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Done] posterior probability demo saved to {fp}")


def plot_sklearn_comparison():
    """
    对比自定义实现与 sklearn 的 GaussianNB / LDA / QDA。
    """
    np.random.seed(42)
    X, y = make_classification(n_samples=300, n_features=2,
                                n_redundant=0, n_clusters_per_class=1,
                                random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 自定义模型
    custom = GaussianBayesClassifier(cov_type='full')
    custom.fit(X_train, y_train)

    # sklearn 模型
    sk_gnb = GaussianNB()
    sk_gnb.fit(X_train, y_train)
    sk_qda = QuadraticDiscriminantAnalysis(reg_param=1e-6)
    sk_qda.fit(X_train, y_train)

    from sklearn.metrics import accuracy_score
    custom_acc = accuracy_score(y_test, custom.predict(X_test))
    gnb_acc = accuracy_score(y_test, sk_gnb.predict(X_test))
    qda_acc = accuracy_score(y_test, sk_qda.predict(X_test))

    print(f"\n[Comparison] Test Accuracy:")
    print(f"  Custom Gaussian Bayes (full): {custom_acc:.4f}")
    print(f"  sklearn GaussianNB:           {gnb_acc:.4f}")
    print(f"  sklearn QDA:                  {qda_acc:.4f}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 70)
    print("ml02_bayesian_decision/demo.py — 贝叶斯决策理论")
    print("=" * 70)

    print("\n[1/5] 对比三种协方差假设的决策边界...")
    plot_decision_boundaries_cov_types()

    print("\n[2/5] 展示最小风险决策...")
    plot_risk_decision_demo()

    print("\n[3/5] 绘制 ROC 曲线 & 计算 AUC/EER...")
    plot_roc_auc_eer()

    print("\n[4/5] 一维后验概率可视化...")
    plot_posterior_3d()

    print("\n[5/5] 与 sklearn 对比验证...")
    plot_sklearn_comparison()

    print("\n" + "=" * 70)
    print("全部可视化完成! 图像保存在:", _IMAGES_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
