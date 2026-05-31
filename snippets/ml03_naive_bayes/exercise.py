# -*- coding: utf-8 -*-
"""
===============================================================================
ml03_naive_bayes/code/exercise.py — 朴素贝叶斯练习
===============================================================================
本练习文件中，你需要完成以下任务：

练习目标：
  1. 实现 GaussianNB 的对数似然计算
  2. 实现 MultinomialNB 的拉普拉斯平滑特征概率估计
  3. 实现 Laplace 平滑参数自动选择的交叉验证（Bonus）

提示：
  - 对数一维高斯密度: log N(x|mu, sigma^2) = -0.5*[log(2*pi*sigma^2) + (x-mu)^2/sigma^2]
  - 条件独立: log P(x|class) = sum_i log P(x_i|class)
  - 带平滑的多项分布概率: P(w_i|class) = (count_i + alpha) / (total + alpha * n_features)
  - MLE (alpha=0): 零计数导致 log(0) = -inf, 分类失效

运行方式：
  python exercise.py
===============================================================================
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification


# ============================================================================
# 任务 1: 实现 GaussianNB 的对数联合概率计算 (约 12 行)
# ============================================================================

def gaussian_nb_log_joint(X, means, vars_, priors):
    """
    计算 log P(x|class_j) + log P(class_j) (对数联合概率)。

    参数:
        X: (n, d), 特征矩阵
        means: (C, d), 各类别各特征的均值
        vars_: (C, d), 各类别各特征的方差
        priors: (C,), 先验概率
    返回:
        log_joint: (n, C), log P(x_i|class_j) + log P(class_j)
    """
    n_classes = len(priors)
    n_samples = X.shape[0]
    log_joint = np.zeros((n_samples, n_classes))

    # TODO: 对每个类别，计算对数联合概率
    # 步骤:
    # 1. for each class j:
    #    a. 计算对数似然: log_lik = -0.5 * [log(2*pi*var[j]) + (X - mean[j])^2 / var[j]]
    #       这会得到形状 (n, d) 的矩阵
    #    b. 条件独立假设: 将 log_lik 按 axis=1 求和 → (n,)
    #    c. log_joint[:, j] = log_lik_sum + log(priors[j])
    # 2. 返回 log_joint
    # --- BEGIN YOUR CODE ---
    for j in range(n_classes):
        mu_j = means[j]
        var_j = vars_[j]
        # 对数一维高斯: -0.5 * [log(2*pi*sigma^2) + (x-mu)^2/sigma^2]
        log_lik = -0.5 * (np.log(2 * np.pi * var_j) + ((X - mu_j) ** 2) / var_j)
        # 条件独立: sum over features
        log_lik_sum = np.sum(log_lik, axis=1)
        log_joint[:, j] = log_lik_sum + np.log(priors[j])
    # --- END YOUR CODE ---
    return log_joint


# 测试函数（验证高斯 NB 实现）
def test_gaussian_nb():
    """测试 GaussianNB 在简单数据上的分类。"""
    print("[测试 1] GaussianNB 对数联合概率...")
    np.random.seed(42)
    X, y = make_classification(n_samples=100, n_features=3,
                                n_informative=2, n_redundant=0,
                                random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    classes = np.unique(y_train)
    n_classes = len(classes)
    n_features = X_train.shape[1]

    means = np.zeros((n_classes, n_features))
    vars_ = np.zeros((n_classes, n_features))
    priors = np.zeros(n_classes)

    for idx, c in enumerate(classes):
        X_c = X_train[y_train == c]
        priors[idx] = len(X_c) / len(X_train)
        means[idx] = np.mean(X_c, axis=0)
        vars_[idx] = np.var(X_c, axis=0) + 1e-9

    # 计算对数联合概率
    log_joint = gaussian_nb_log_joint(X_test, means, vars_, priors)
    predictions = np.argmax(log_joint, axis=1)

    acc = np.mean(predictions == y_test)
    assert acc > 0.5, f"Accuracy too low: {acc:.3f}"
    print(f"  [PASS] GaussianNB 测试准确率: {acc:.3f}")


# ============================================================================
# 任务 2: 实现 MultinomialNB 的特征概率估计 (约 8 行)
# ============================================================================

def multinomial_nb_fit(X, y, alpha=1.0):
    """
    估计 MultinomialNB 的参数（带拉普拉斯平滑）。

    参数:
        X: (n, d), 非负整数特征矩阵（如词频）
        y: (n,), 标签
        alpha: float, 平滑参数
    返回:
        priors: (C,), 先验概率
        feature_log_prob: (C, d), log P(feature i | class j)
    """
    classes = np.unique(y)
    n_classes = len(classes)
    n_features = X.shape[1]

    priors = np.zeros(n_classes)
    feature_log_prob = np.zeros((n_classes, n_features))

    # TODO: 对每个类别，估计特征概率
    # 步骤:
    # 1. for each class j:
    #    a. X_c = X[y == c], 取出该类别的样本
    #    b. priors[j] = len(X_c) / len(X)
    #    c. count_c = sum(X_c, axis=0), 对每个特征求和得到特征总计数
    #    d. total = sum(count_c), 所有特征的总计数
    #    e. prob_c = (count_c + alpha) / (total + alpha * n_features)  # Laplace 平滑!
    #    f. feature_log_prob[j] = log(prob_c)
    # 2. 返回 priors 和 feature_log_prob
    # --- BEGIN YOUR CODE ---
    for idx, c in enumerate(classes):
        X_c = X[y == c]
        priors[idx] = len(X_c) / len(X)
        count_c = np.sum(X_c, axis=0)
        total = np.sum(count_c)
        prob_c = (count_c + alpha) / (total + alpha * n_features)
        feature_log_prob[idx] = np.log(prob_c)
    # --- END YOUR CODE ---
    return priors, feature_log_prob


def test_multinomial_nb():
    """测试 MultinomialNB 的拉普拉斯平滑。"""
    print("[测试 2] MultinomialNB 拉普拉斯平滑...")

    # 极端小样本: 有些特征在某类别中从未出现
    X = np.array([[5, 3, 0, 0], [4, 2, 1, 0], [0, 1, 4, 5], [1, 0, 3, 6]])
    y = np.array([0, 0, 1, 1])

    # alpha=0: MLE (零概率会导致 log(-inf) = 无效)
    priors, flp_mle = multinomial_nb_fit(X, y, alpha=0.0001)
    # alpha=1: Laplace 平滑
    priors, flp_laplace = multinomial_nb_fit(X, y, alpha=1.0)

    # 验证: Laplace 平滑后，所有概率 > 0
    assert np.all(np.isfinite(flp_laplace)), "Laplace 平滑后出现无效概率"
    print(f"  [PASS] MLE 下某些概率: {np.exp(flp_mle)}")
    print(f"  [PASS] Laplace 平滑后所有概率: {np.exp(flp_laplace)}")

    # 验证: 无平滑（alpha=0）时，未出现的特征概率约等于 0
    probs_mle = np.exp(flp_mle)
    assert probs_mle[0, 2] < 1e-4 or probs_mle[0, 3] < 1e-4, \
        "无平滑时未出现的特征概率应趋近于 0"


# ============================================================================
# 任务 3 (Bonus): 交叉验证选择最佳平滑参数 (约 12 行)
# ============================================================================

def select_alpha_cv(X, y, alphas, n_folds=3):
    """
    用 K-Fold 交叉验证选择最佳拉普拉斯平滑参数 alpha。

    参数:
        X: (n, d), 非负整数特征
        y: (n,), 标签
        alphas: list, 候选 alpha 值
        n_folds: int, 折数
    返回:
        best_alpha: float
        cv_scores: dict, {alpha: mean_accuracy}
    """
    n = len(X)
    fold_size = n // n_folds
    cv_scores = {}

    # TODO: 实现交叉验证
    # 步骤:
    # 1. for each alpha:
    #    2. 将数据分为 n_folds 折
    #    3. for each fold:
    #       a. val_idx = fold 的索引, train_idx = 其余索引（用 np.setdiff1d）
    #       b. priors, flp = multinomial_nb_fit(X[train_idx], y[train_idx], alpha)
    #       c. log_joint = X[val_idx] @ flp.T + log(priors)
    #       d. pred = argmax(log_joint, axis=1), 计算准确率
    #    4. cv_scores[alpha] = mean accuracy across folds
    # 5. 返回最佳 alpha
    # --- BEGIN YOUR CODE ---
    for alpha in alphas:
        fold_accs = []
        for fold in range(n_folds):
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size if fold < n_folds - 1 else n
            val_idx = np.arange(val_start, val_end)
            train_idx = np.setdiff1d(np.arange(n), val_idx)

            priors, flp = multinomial_nb_fit(X[train_idx], y[train_idx], alpha)
            log_joint = X[val_idx] @ flp.T + np.log(priors)
            pred = np.argmax(log_joint, axis=1)
            fold_accs.append(np.mean(pred == y[val_idx]))

        cv_scores[alpha] = np.mean(fold_accs)
    # --- END YOUR CODE ---

    best_alpha = max(cv_scores, key=cv_scores.get)
    return best_alpha, cv_scores


def test_alpha_selection():
    """测试 alpha 选择的交叉验证。"""
    print("[测试 3 (Bonus)] 交叉验证选择 alpha...")

    np.random.seed(42)
    X, y = make_classification(n_samples=100, n_features=5,
                                n_informative=3, random_state=42)
    X = np.abs(X) * 10
    X = X.astype(int)

    alphas = [0.01, 0.1, 1.0, 5.0]
    best_alpha, cv_scores = select_alpha_cv(X, y, alphas, n_folds=3)

    assert best_alpha in alphas, f"最佳 alpha 不在候选列表中: {best_alpha}"
    print(f"  [PASS] 最佳 alpha = {best_alpha}, scores = {cv_scores}")


if __name__ == "__main__":
    print("=" * 60)
    print("ml03_naive_bayes exercise.py — 朴素贝叶斯练习")
    print("=" * 60)

    try:
        test_gaussian_nb()
        test_multinomial_nb()
        test_alpha_selection()
        print("\n" + "=" * 60)
        print("所有测试通过!")
        print("=" * 60)
    except Exception as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
