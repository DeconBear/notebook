#pragma once
// 最小 SO(3)：hat / Rodrigues exp。编译: g++ -std=c++17 demo.cpp -o so3_demo
#include <array>
#include <cmath>

inline std::array<std::array<double, 3>, 3> hat(double x, double y, double z) {
    return {{{0, -z, y}, {z, 0, -x}, {-y, x, 0}}};
}

inline std::array<std::array<double, 3>, 3> so3_exp(double wx, double wy, double wz) {
    const double th = std::sqrt(wx * wx + wy * wy + wz * wz);
    auto K = hat(wx, wy, wz);
    std::array<std::array<double, 3>, 3> R{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    auto add = [&](double s, const auto& M) {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) R[i][j] += s * M[i][j];
    };
    auto mul = [](const auto& A, const auto& B) {
        std::array<std::array<double, 3>, 3> C{};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 3; ++k) C[i][j] += A[i][k] * B[k][j];
        return C;
    };
    if (th < 1e-12) {
        add(1.0, K);
        return R;
    }
    add(std::sin(th) / th, K);
    add((1 - std::cos(th)) / (th * th), mul(K, K));
    return R;
}
