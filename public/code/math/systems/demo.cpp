#include "gauss.hpp"
#include <cstdio>

int main() {
    Mat3 A{};
    A[0] = {{2.0, 1.0, -1.0}};
    A[1] = {{-3.0, -1.0, 2.0}};
    A[2] = {{-2.0, 1.0, 2.0}};
    Vec3 b = {{8.0, -11.0, -3.0}};
    Vec3 x{};
    bool ok = gauss_solve(A, b, x);
    std::printf("solve ok=%d  x=[%.6f, %.6f, %.6f]  (expect 2, 3, -1)\n",
                int(ok), x[0], x[1], x[2]);
    std::printf("rank(A)=%d\n", gauss_rank(A));

    Mat3 B{};
    B[0] = {{1.0, 2.0, 3.0}};
    B[1] = {{2.0, 4.0, 6.0}};
    B[2] = {{1.0, 1.0, 1.0}};
    std::printf("rank(B)=%d  (row2=2*row1 => expect 2)\n", gauss_rank(B));
    return 0;
}
