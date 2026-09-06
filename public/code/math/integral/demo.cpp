#include "quad.hpp"
#include <cstdio>

static double square(double x) { return x * x; }

int main() {
    Poly p{};
    p.n = 2;
    p.c[2] = 1.0;  // x^2
    Poly P = poly_int(p);  // x^3/3
    double exact = poly_eval(P, 1.0) - poly_eval(P, 0.0);
    std::printf("poly  int_0^1 x^2 dx = %.6f  (exact 1/3=%.6f)\n", exact, 1.0 / 3.0);

    const int ns[3] = {4, 16, 64};
    for (int k = 0; k < 3; ++k) {
        int n = ns[k];
        double t = trapezoid(square, 0.0, 1.0, n);
        std::printf("trap n=%2d  %.6f  err=%.2e\n", n, t, t - 1.0 / 3.0);
    }
    return 0;
}
