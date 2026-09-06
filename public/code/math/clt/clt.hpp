#pragma once
// 最小 LCG + 指数分布抽样，用来看样本均值。不依赖 Eigen。
#include <cmath>
#include <cstdint>

struct Lcg {
    std::uint32_t state;
    explicit Lcg(std::uint32_t seed = 42) : state(seed) {}
    // [0,1) 均匀。minstd 一类的乘同余。
    double uniform() {
        state = static_cast<std::uint32_t>(state * 1103515245u + 12345u);
        return (state >> 8) * (1.0 / 16777216.0);
    }
    // Exp(lam)：-log(U)/lam。U 避开 0。
    double exponential(double lam) {
        double u = uniform();
        if (u < 1e-12)
            u = 1e-12;
        return -std::log(u) / lam;
    }
};

inline double mean_of_n(Lcg& rng, int n, double lam) {
    double s = 0.0;
    for (int i = 0; i < n; ++i)
        s += rng.exponential(lam);
    return s / static_cast<double>(n);
}
