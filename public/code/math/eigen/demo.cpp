#include "power.hpp"
#include <cstdio>

int main() {
    Mat2 A{};
    A[0] = {{3.0, 1.0}};
    A[1] = {{1.0, 2.0}};
    Vec2 v = {{1.0, 0.0}};
    double lam = power_iteration(A, v, 25);
    std::printf("power iter  lambda=%.6f  v=[%.6f, %.6f]\n", lam, v[0], v[1]);
    std::printf("(numpy eig of [[3,1],[1,2]] largest ~ 3.618)\n");
    return 0;
}
