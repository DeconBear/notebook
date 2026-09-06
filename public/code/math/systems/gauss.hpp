#pragma once
// 3×3 高斯消元（部分主元）。不依赖 Eigen。
#include <array>
#include <cmath>
#include <algorithm>

using Mat3 = std::array<std::array<double, 3>, 3>;
using Vec3 = std::array<double, 3>;

inline int gauss_rank(Mat3 A, double eps = 1e-10) {
    const int n = 3;
    int rank = 0;
    int col = 0;
    for (int row = 0; row < n && col < n;) {
        int piv = row;
        for (int i = row + 1; i < n; ++i)
            if (std::fabs(A[i][col]) > std::fabs(A[piv][col]))
                piv = i;
        if (std::fabs(A[piv][col]) < eps) {
            ++col;
            continue;
        }
        std::swap(A[row], A[piv]);
        double pv = A[row][col];
        for (int j = col; j < n; ++j)
            A[row][j] /= pv;
        for (int i = 0; i < n; ++i) {
            if (i == row)
                continue;
            double f = A[i][col];
            for (int j = col; j < n; ++j)
                A[i][j] -= f * A[row][j];
        }
        ++rank;
        ++row;
        ++col;
    }
    return rank;
}

// 成功则把解写入 x 并返回 true；奇异则 false。
inline bool gauss_solve(Mat3 A, Vec3 b, Vec3& x, double eps = 1e-10) {
    const int n = 3;
    for (int col = 0; col < n; ++col) {
        int piv = col;
        for (int i = col + 1; i < n; ++i)
            if (std::fabs(A[i][col]) > std::fabs(A[piv][col]))
                piv = i;
        if (std::fabs(A[piv][col]) < eps)
            return false;
        std::swap(A[col], A[piv]);
        std::swap(b[col], b[piv]);
        double pv = A[col][col];
        for (int j = col; j < n; ++j)
            A[col][j] /= pv;
        b[col] /= pv;
        for (int i = 0; i < n; ++i) {
            if (i == col)
                continue;
            double f = A[i][col];
            for (int j = col; j < n; ++j)
                A[i][j] -= f * A[col][j];
            b[i] -= f * b[col];
        }
    }
    x = b;
    return true;
}
