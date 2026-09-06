#pragma once
// 幂迭代：找模最大的特征值。对称矩阵上效果最好。不依赖 Eigen。
#include <array>
#include <cmath>

using Vec2 = std::array<double, 2>;
using Mat2 = std::array<std::array<double, 2>, 2>;

inline double dot2(Vec2 a, Vec2 b) {
    return a[0] * b[0] + a[1] * b[1];
}

inline double norm2(Vec2 v) {
    return std::sqrt(dot2(v, v));
}

inline Vec2 mat2_apply(Mat2 A, Vec2 v) {
    return Vec2{{
        A[0][0] * v[0] + A[0][1] * v[1],
        A[1][0] * v[0] + A[1][1] * v[1],
    }};
}

// 返回 Rayleigh 商 v^T A v / v^T v（假定 ||v||=1 时就是 v^T A v）。
inline double power_iteration(Mat2 A, Vec2& v, int n_iter = 20) {
    double n = norm2(v);
    v[0] /= n;
    v[1] /= n;
    double lam = 0.0;
    for (int k = 0; k < n_iter; ++k) {
        Vec2 Av = mat2_apply(A, v);
        lam = dot2(v, Av);
        double nv = norm2(Av);
        v[0] = Av[0] / nv;
        v[1] = Av[1] / nv;
    }
    return lam;
}
