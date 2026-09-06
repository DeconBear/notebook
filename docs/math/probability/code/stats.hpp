#pragma once
// 样本均值 / 无偏方差。不依赖 Eigen。
#include <cmath>
#include <cstddef>

inline double sample_mean(const double* x, std::size_t n) {
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        s += x[i];
    return s / static_cast<double>(n);
}

// unbiased=true 时除以 n-1（样本方差）；否则除以 n（总体 / MLE）。
inline double sample_var(const double* x, std::size_t n, bool unbiased = true) {
    double mu = sample_mean(x, n);
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double d = x[i] - mu;
        s += d * d;
    }
    double den = unbiased ? static_cast<double>(n - 1) : static_cast<double>(n);
    return s / den;
}

inline double sample_std(const double* x, std::size_t n, bool unbiased = true) {
    return std::sqrt(sample_var(x, n, unbiased));
}
