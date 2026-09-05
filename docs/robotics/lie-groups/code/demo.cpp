#include "so3.hpp"
#include <cstdio>

int main() {
    auto R = so3_exp(0.3, -0.1, 0.8);
    double det =
        R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1]) -
        R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0]) +
        R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]);
    std::printf("det(R) = %.6f  (should be 1)\n", det);
    return 0;
}
