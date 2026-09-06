#include "dual.hpp"
#include <cstdio>

static double cube_minus_2x(double x) { return x * x * x - 2.0 * x; }

int main() {
    const double x0 = 2.0;
    Dual x = Dual::var(x0);
    Dual f = x * x * x - 2.0 * x;
    std::printf("autodiff  f(x)=x^3-2x at x=2\n");
    std::printf("  f=%.6f  f'=%.6f  (exact 4 and 10)\n", f.v, f.d);

    Poly p{};
    p.n = 3;
    p.c[1] = -2.0;
    p.c[3] = 1.0;  // x^3 - 2x
    Poly dp = poly_diff(p);
    std::printf("poly coeff  p(2)=%.6f  p'(2)=%.6f\n",
                poly_eval(p, x0), poly_eval(dp, x0));

    std::printf("central diff h=1e-6  f'=%.6f\n",
                central_diff(cube_minus_2x, x0, 1e-6));

    Dual y = Dual::var(0.0);
    Dual g = sin(y) * exp(y);
    std::printf("autodiff  sin(x)exp(x) at 0  g=%.6f  g'=%.6f  (exact 0 and 1)\n",
                g.v, g.d);
    return 0;
}
