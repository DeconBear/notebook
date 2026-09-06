#include "mat2.hpp"
#include <cstdio>

int main() {
    const double pi = 3.141592653589793;
    auto R = mat2_rotation(pi / 2.0);
    double x = 1.0, y = 0.0, ox, oy;
    mat2_apply(R, x, y, ox, oy);
    std::printf("R(90 deg) * [1, 0] = [%.6f, %.6f]\n", ox, oy);
    std::printf("det(R) = %.6f  (rotation => 1)\n", mat2_det(R));

    auto A = mat2(2.0, 1.0, 0.0, 3.0);
    std::printf("det([[2,1],[0,3]]) = %.6f  (expect 6)\n", mat2_det(A));
    return 0;
}
