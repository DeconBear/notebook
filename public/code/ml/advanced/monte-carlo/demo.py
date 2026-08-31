# -*- coding: utf-8 -*-
"""
ml10 蒙特卡洛方法 — 演示代码
=============================
功能：蒙特卡洛 π 估计、重要性采样对比、Metropolis-Hastings 采样 2D 高斯混合、
      Gibbs 采样二元正态分布。

每个函数都有中文 docstring，每行逻辑代码都有中文注释。
运行方式：在 ml10_monte_carlo/ 目录下执行 python code/demo.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
from scipy.stats import multivariate_normal, norm

_HERE = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_HERE, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)


# ============================================================================
# 第一部分：蒙特卡洛估计 π
# ============================================================================

def estimate_pi_mc(N=10000):
    """
    用蒙特卡洛方法估计 π。

    原理：在边长为 2 的正方形 [-1,1]^2 中均匀撒点，
         落在单位圆内的点比例 × 4 = π 的估计。

    数学：
        π ≈ 4 * (#{ (x,y): x^2 + y^2 ≤ 1 }) / N
    """
    # 从均匀分布 U(-1, 1) 采样 N 个 2D 点
    points = np.random.uniform(-1, 1, size=(N, 2))  # (N, 2)
    # 判断每个点是否在单位圆内: x^2 + y^2 ≤ 1
    inside = (points[:, 0]**2 + points[:, 1]**2) <= 1  # 布尔数组
    # 估计 π
    pi_est = 4.0 * inside.sum() / N
    return points, inside, pi_est


def plot_pi_estimation(points, inside, N, pi_est):
    """绘制 π 的蒙特卡洛估计可视化"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左图：散点图
    ax = axes[0]
    ax.scatter(points[inside, 0], points[inside, 1], c='coral', s=1,
               alpha=0.6, label='Inside Circle')
    ax.scatter(points[~inside, 0], points[~inside, 1], c='steelblue', s=1,
               alpha=0.6, label='Outside Circle')
    # 画单位圆
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=1.5)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.set_title(f'Monte Carlo Pi Estimation (N={N})\n'
                 f'Estimate = {pi_est:.6f} (True = {np.pi:.6f})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(markerscale=5, fontsize=9)

    # 右图：估计值随 N 的收敛
    ax2 = axes[1]
    cumulative = 4.0 * np.cumsum(inside) / np.arange(1, N + 1)
    ax2.plot(np.arange(1, N + 1), cumulative, 'b-', linewidth=1, alpha=0.7)
    ax2.axhline(y=np.pi, color='r', linestyle='--', linewidth=1.5, label=f'True π = {np.pi:.6f}')
    ax2.set_xlabel('Number of Samples N')
    ax2.set_ylabel('Estimated π')
    ax2.set_title('Convergence of MC Estimate')
    ax2.legend()
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(_IMAGES_DIR, 'ml10-04-pi-estimation.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[保存] {path}')


# ============================================================================
# 第二部分：重要性采样 —— 估计尾部概率
# ============================================================================

def importance_sampling_demo():
    """
    重要性采样 vs 普通 MC 估计 P(X > 5)，X ~ N(0, 1)。

    普通 MC：从 N(0,1) 采样 10000 次，记录 >5 的频率。方差极大。
    重要性采样：从 N(5,1) 采样，用权重 p(x)/q(x) 修正。
    """
    N = 10000
    true_prob = 1.0 - norm.cdf(5)  # 真实尾部概率 ≈ 2.87e-7

    # ---- 方法 1: 普通 MC ----
    samples_mc = np.random.randn(N)
    mc_estimates = np.cumsum(samples_mc > 5) / np.arange(1, N + 1)

    # ---- 方法 2: 重要性采样 ----
    # 提议分布: q(x) = N(5, 1)，即从 N(0,1) 偏移 5
    samples_is = np.random.randn(N) + 5  # ~ N(5, 1)
    # 重要性权重: p(x)/q(x) = phi(x)/phi(x-5)
    # p(x) = N(x; 0, 1), q(x) = N(x; 5, 1)
    # w(x) = exp(-0.5*x^2) / exp(-0.5*(x-5)^2) = exp(-0.5*(x^2 - (x-5)^2))
    log_weights = -0.5 * (samples_is**2 - (samples_is - 5)**2)
    weights = np.exp(log_weights)
    # 加权平均: (1/N) sum w(x_i) * I(x_i > 5)
    is_estimates = np.cumsum(weights * (samples_is > 5)) / np.arange(1, N + 1)

    # ---- 可视化 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：提议分布和采样点
    ax = axes[0]
    x_grid = np.linspace(-2, 10, 500)
    ax.plot(x_grid, norm.pdf(x_grid, 0, 1), 'b-', linewidth=2, label='p(x) ~ N(0,1)')
    ax.plot(x_grid, norm.pdf(x_grid, 5, 1), 'r--', linewidth=2, label='q(x) ~ N(5,1)')
    ax.axvline(x=5, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.fill_between(x_grid[x_grid >= 5], 0, norm.pdf(x_grid[x_grid >= 5], 0, 1),
                    alpha=0.15, color='blue', label='Target region x>5')
    ax.set_xlabel('x')
    ax.set_ylabel('Density')
    ax.set_title('Target Distribution vs Proposal Distribution')
    ax.legend()

    # 右图：两个估计器的收敛对比
    ax2 = axes[1]
    ax2.plot(np.arange(1, N + 1), mc_estimates, 'b-', linewidth=1, alpha=0.7,
             label='Standard MC')
    ax2.plot(np.arange(1, N + 1), is_estimates, 'r-', linewidth=1, alpha=0.7,
             label='Importance Sampling')
    ax2.axhline(y=true_prob, color='gray', linestyle='--', linewidth=1.5,
                label=f'True prob = {true_prob:.2e}')
    ax2.set_xlabel('Number of Samples N')
    ax2.set_ylabel('Estimated P(X > 5)')
    ax2.set_title('Importance Sampling vs Standard MC')
    ax2.set_xscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 打印统计信息
    print(f'\n  真实尾部概率: {true_prob:.2e}')
    print(f'  普通 MC (N={N}): 命中 {sum(samples_mc > 5)} 次, 估计值={mc_estimates[-1]:.2e}')
    print(f'  重要性采样 (N={N}): 估计值={is_estimates[-1]:.2e}')
    # 计算方差
    mc_var = np.var(samples_mc > 5) / N
    is_var = np.var(weights * (samples_is > 5)) / N
    print(f'  MC 方差: {mc_var:.2e}')
    print(f'  IS 方差: {is_var:.2e} (方差缩减比: {mc_var/is_var:.1f}x)')

    plt.tight_layout()
    path = os.path.join(_IMAGES_DIR, 'ml10-05-importance-sampling-comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[保存] {path}')


# ============================================================================
# 第三部分：Metropolis-Hastings 采样 2D 高斯混合
# ============================================================================

def target_log_pdf(x):
    """
    目标分布的对数概率密度: 2D 高斯混合。

    p(x) = 0.5 * N(x | mu1, sigma1) + 0.5 * N(x | mu2, sigma2)
    """
    mu1 = np.array([-2, -2])
    mu2 = np.array([2, 2])
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])

    log_p1 = multivariate_normal.logpdf(x, mean=mu1, cov=cov)
    log_p2 = multivariate_normal.logpdf(x, mean=mu2, cov=cov)

    # log-sum-exp trick 防数值溢出
    log_p_max = np.maximum(log_p1, log_p2)
    log_p = log_p_max + np.log(0.5 * np.exp(log_p1 - log_p_max)
                                + 0.5 * np.exp(log_p2 - log_p_max))
    return log_p


def metropolis_hastings(n_iter=5000, proposal_std=1.0, random_state=42):
    """
    Metropolis-Hastings 算法采样 2D 高斯混合分布。

    算法：
        for t = 1, ..., n_iter:
            1. 提议: x' ~ N(x^{(t)}, proposal_std^2 * I)
            2. 计算接受率 α = min(1, p(x')/p(x^{(t)}))
            3. 以概率 α 接受，否则保留当前

    参数:
        n_iter: 迭代次数
        proposal_std: 提议分布（随机游走）的标准差
        random_state: 随机种子
    """
    rng = np.random.RandomState(random_state)
    d = 2  # 2D 空间
    samples = np.zeros((n_iter, d))
    accepted = np.zeros(n_iter, dtype=bool)

    # 初始化：从均匀分布中选起点
    x_current = rng.uniform(-4, 4, size=d)
    log_p_current = target_log_pdf(x_current.reshape(1, -1)).item()

    for t in range(n_iter):
        # 步骤 1：提议 x' = x + ε, ε ~ N(0, proposal_std^2 I)
        x_proposal = x_current + rng.randn(d) * proposal_std
        log_p_proposal = target_log_pdf(x_proposal.reshape(1, -1)).item()

        # 步骤 2：计算接受率（对称提议 q(x'|x)=q(x|x')，所以只比 p(x')/p(x)）
        log_alpha = log_p_proposal - log_p_current
        alpha = min(1.0, np.exp(log_alpha))

        # 步骤 3：接受/拒绝
        if rng.rand() < alpha:
            x_current = x_proposal
            log_p_current = log_p_proposal
            accepted[t] = True

        samples[t] = x_current

    return samples, accepted


def plot_mh_results(samples, accepted, n_burnin=1000):
    """绘制 MH 采样结果"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 左上：目标分布的等高线 + 采样点（burn-in 后的）
    ax = axes[0, 0]
    xx, yy = np.meshgrid(np.linspace(-6, 6, 100), np.linspace(-6, 6, 100))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    zz = np.exp(target_log_pdf(grid)).reshape(100, 100)
    ax.contour(xx, yy, zz, levels=8, cmap='Greys', alpha=0.5)

    post_samples = samples[n_burnin:]
    ax.scatter(post_samples[::10, 0], post_samples[::10, 1],  # thinning: 每10步取一个
               c='coral', s=3, alpha=0.5, label='Thinned samples')
    ax.plot(samples[:n_burnin, 0], samples[:n_burnin, 1], 'b-', linewidth=0.5, alpha=0.4,
            label='Burn-in path')
    ax.plot(samples[0, 0], samples[0, 1], 'go', markersize=8, label='Start')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('MH Sampling: 2D Gaussian Mixture')
    ax.legend(fontsize=8)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)

    # 右上：x1 的 trace plot
    ax2 = axes[0, 1]
    ax2.plot(range(n_burnin + len(post_samples)), samples[:, 0], 'b-', linewidth=0.5, alpha=0.8)
    ax2.axvline(x=n_burnin, color='red', linestyle='--', linewidth=1.5, label='Burn-in end')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('x1')
    ax2.set_title('Trace Plot of x1')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 左下：x1 的边缘分布直方图
    ax3 = axes[1, 0]
    ax3.hist(post_samples[:, 0], bins=50, density=True, color='coral', alpha=0.7,
             edgecolor='white', linewidth=0.5)
    ax3.set_xlabel('x1')
    ax3.set_ylabel('Density')
    ax3.set_title('Marginal Distribution of x1 (post-burn-in)')

    # 右下：接受率时序
    ax4 = axes[1, 1]
    window = 200
    cum_accept_rate = np.convolve(accepted, np.ones(window)/window, mode='valid')
    ax4.plot(range(window-1, len(accepted)), cum_accept_rate, 'g-', linewidth=1)
    ax4.axhline(y=accepted.mean(), color='gray', linestyle='--', linewidth=1,
                label=f'Overall acceptance: {accepted.mean():.3f}')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Acceptance Rate (rolling mean)')
    ax4.set_title('MH Acceptance Rate Over Time')
    ax4.legend()
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(_IMAGES_DIR, 'ml10-06-metropolis-hastings-results.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[保存] {path}')

    print(f'  MH 接受率: {accepted.mean():.4f} (理想范围: 0.2-0.5)')
    print(f'  有效样本数 (post-burn-in): {len(post_samples)}')


# ============================================================================
# 第四部分：Gibbs 采样二元正态分布
# ============================================================================

def gibbs_sampling_bivariate_normal(n_iter=5000, rho=0.8):
    """
    Gibbs 采样二元正态分布。

    目标分布：
        [x1, x2] ~ N([0, 0], [[1, ρ], [ρ, 1]])

    条件分布：
        x1 | x2 ~ N(ρ * x2, 1 - ρ^2)
        x2 | x1 ~ N(ρ * x1, 1 - ρ^2)
    """
    rng = np.random.RandomState(42)
    samples = np.zeros((n_iter, 2))

    # 初始化
    x1, x2 = 0.0, 0.0

    for t in range(n_iter):
        # Gibbs 步骤 1: 从 p(x1 | x2) 采样
        # x1 | x2 ~ N(rho * x2, 1 - rho^2)
        x1 = rng.normal(rho * x2, np.sqrt(1 - rho**2))

        # Gibbs 步骤 2: 从 p(x2 | x1) 采样
        # x2 | x1 ~ N(rho * x1, 1 - rho^2)
        x2 = rng.normal(rho * x1, np.sqrt(1 - rho**2))

        samples[t] = [x1, x2]

    return samples


def plot_gibbs_results(samples, rho, n_burnin=500):
    """绘制 Gibbs 采样结果"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 左上：散点图 + 等高线
    ax = axes[0, 0]
    post = samples[n_burnin:]
    ax.scatter(post[:, 0], post[:, 1], c='teal', s=2, alpha=0.4)
    # 真实等高线
    xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    cov = np.array([[1, rho], [rho, 1]])
    dist = multivariate_normal(mean=[0, 0], cov=cov)
    zz = dist.pdf(np.column_stack([xx.ravel(), yy.ravel()])).reshape(100, 100)
    ax.contour(xx, yy, zz, levels=6, cmap='Reds', alpha=0.5, linewidths=1.5)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title(f'Gibbs Sampling: Bivariate Normal (ρ={rho})')
    ax.set_aspect('equal')

    # 右上：x1 的 trace plot（前 500 步）
    ax2 = axes[0, 1]
    ax2.plot(range(min(500, n_iter)), samples[:500, 0], 'b-', linewidth=0.8)
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax2.axvline(x=n_burnin, color='red', linestyle='--', linewidth=1.5, label=f'Burn-in={n_burnin}')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('x1')
    ax2.set_title('Trace Plot of x1 (first 500 iterations)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 左下：x1 自相关函数
    ax3 = axes[1, 0]
    n_lags = 50
    x1_series = post[:, 0]
    autocorr = [1.0]
    mean = x1_series.mean()
    var = x1_series.var()
    for lag in range(1, n_lags + 1):
        corr = np.mean((x1_series[:-lag] - mean) * (x1_series[lag:] - mean)) / var
        autocorr.append(corr)
    ax3.stem(range(len(autocorr)), autocorr, linefmt='b-', markerfmt='bo', basefmt=' ')
    ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax3.set_xlabel('Lag')
    ax3.set_ylabel('Autocorrelation')
    ax3.set_title(f'Autocorrelation of x1 (ρ={rho})')
    ax3.grid(True, alpha=0.3)

    # 右下：样本协方差 vs 真实协方差
    ax4 = axes[1, 1]
    sample_cov = np.cov(post.T)
    true_cov = np.array([[1, rho], [rho, 1]])
    labels = ['Var(x1)', 'Cov(x1,x2)', 'Var(x2)']
    true_vals = [true_cov[0,0], true_cov[0,1], true_cov[1,1]]
    sample_vals = [sample_cov[0,0], sample_cov[0,1], sample_cov[1,1]]
    x_pos = np.arange(3)
    width = 0.35
    ax4.bar(x_pos - width/2, true_vals, width, label='True', color='gray', alpha=0.7)
    ax4.bar(x_pos + width/2, sample_vals, width, label='Gibbs Estimate', color='teal', alpha=0.7)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(labels)
    ax4.set_ylabel('Value')
    ax4.set_title('Covariance Recovery')
    ax4.legend()

    plt.tight_layout()
    path = os.path.join(_IMAGES_DIR, 'ml10-07-gibbs-sampling.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[保存] {path}')

    # 有效样本量（ESS）估计
    n = len(post)
    max_lag = min(100, n - 1)
    acf_sum = 0
    for lag in range(1, max_lag):
        corr = np.corrcoef(post[:-lag, 0], post[lag:, 0])[0, 1]
        acf_sum += corr
    ESS = n / (1 + 2 * acf_sum) if (1 + 2 * acf_sum) > 0 else n
    print(f'  Gibbs: 估计 ESS(有效样本量) ≈ {ESS:.0f} / {n} (ρ={rho})')


# ============================================================================
# 第五部分：主程序
# ============================================================================

def main():
    print('=' * 60)
    print('ml10 蒙特卡洛方法 — 演示代码')
    print('=' * 60)

    # 1. 蒙特卡洛估计 π
    print('\n[1/4] 蒙特卡洛估计 π...')
    N_pi = 5000
    points, inside, pi_est = estimate_pi_mc(N=N_pi)
    print(f'  MC 估计 π = {pi_est:.6f} (真实值 = {np.pi:.6f}, 误差 = {abs(pi_est - np.pi):.6f})')
    plot_pi_estimation(points, inside, N_pi, pi_est)

    # 2. 重要性采样
    print('\n[2/4] 重要性采样 vs 普通 MC...')
    importance_sampling_demo()

    # 3. Metropolis-Hastings
    print('\n[3/4] Metropolis-Hastings 采样 2D 高斯混合...')
    samples_mh, accepted_mh = metropolis_hastings(n_iter=5000, proposal_std=1.2)
    plot_mh_results(samples_mh, accepted_mh, n_burnin=1000)

    # 4. Gibbs 采样
    print('\n[4/4] Gibbs 采样二元正态分布...')
    samples_gibbs = gibbs_sampling_bivariate_normal(n_iter=5000, rho=0.8)
    plot_gibbs_results(samples_gibbs, rho=0.8, n_burnin=500)

    print('\n完成！所有图表已保存到 images/ 目录。')


if __name__ == '__main__':
    main()
