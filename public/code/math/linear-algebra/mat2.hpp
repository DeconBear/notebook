#pragma once
// 2×2 手写矩阵：不依赖 Eigen。只为把 Ax 和 det 写清楚。
#include <cmath>

struct Mat2 {
    double a[2][2];
};

inline Mat2 mat2(double a00, double a01, double a10, double a11) {
    return Mat2{{{a00, a01}, {a10, a11}}};
}

inline Mat2 mat2_mul(Mat2 A, Mat2 B) {
    Mat2 C{};
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            C.a[i][j] = A.a[i][0] * B.a[0][j] + A.a[i][1] * B.a[1][j];
    return C;
}

inline void mat2_apply(Mat2 A, double x, double y, double& ox, double& oy) {
    ox = A.a[0][0] * x + A.a[0][1] * y;
    oy = A.a[1][0] * x + A.a[1][1] * y;
}

inline double mat2_det(Mat2 A) {
    return A.a[0][0] * A.a[1][1] - A.a[0][1] * A.a[1][0];
}

// 逆时针旋转，与 demo.py 的 rotation_matrix 同一约定。
inline Mat2 mat2_rotation(double theta) {
    double c = std::cos(theta);
    double s = std::sin(theta);
    return mat2(c, -s, s, c);
}
