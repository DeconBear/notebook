#pragma once
// 常见 pmf / pdf。不依赖 Eigen。阶乘用循环，n 不要太大。
#include <cmath>

inline long long n_choose_k(int n, int k) {
    if (k < 0 || k > n)
        return 0;
    if (k > n - k)
        k = n - k;
    long long r = 1;
    for (int i = 1; i <= k; ++i)
        r = r * (n - k + i) / i;
    return r;
}

inline double binomial_pmf(int k, int n, double p) {
    return static_cast<double>(n_choose_k(n, k)) *
           std::pow(p, k) * std::pow(1.0 - p, n - k);
}

inline double poisson_pmf(int k, double lam) {
    double logp = -lam + k * std::log(lam);
    for (int i = 2; i <= k; ++i)
        logp -= std::log(static_cast<double>(i));
    return std::exp(logp);
}

inline double gaussian_pdf(double x, double mu, double sigma) {
    const double pi = 3.141592653589793;
    double z = (x - mu) / sigma;
    return std::exp(-0.5 * z * z) / (sigma * std::sqrt(2.0 * pi));
}

inline double exponential_pdf(double x, double lam) {
    if (x < 0.0)
        return 0.0;
    return lam * std::exp(-lam * x);
}
