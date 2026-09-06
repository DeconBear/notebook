#pragma once
// 伯努利 / 高斯的极大似然。不依赖 Eigen。
#include <cmath>
#include <cstddef>

inline double bernoulli_mle(const int* x, std::size_t n) {
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        s += static_cast<double>(x[i]);
    return s / static_cast<double>(n);
}

inline void gaussian_mle(const double* x, std::size_t n, double& mu, double& sigma2) {
    mu = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        mu += x[i];
    mu /= static_cast<double>(n);
    sigma2 = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double d = x[i] - mu;
        sigma2 += d * d;
    }
    sigma2 /= static_cast<double>(n);  // MLE 除以 n，不是 n-1
}
