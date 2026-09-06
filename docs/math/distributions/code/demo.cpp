#include "dist.hpp"
#include <cstdio>

int main() {
    std::printf("Binom(n=10,p=0.3) P(K=3)=%.6f\n", binomial_pmf(3, 10, 0.3));
    std::printf("Poisson(lam=3) P(K=3)=%.6f\n", poisson_pmf(3, 3.0));
    std::printf("N(0,1) phi(0)=%.6f  (expect ~0.3989)\n", gaussian_pdf(0.0, 0.0, 1.0));
    std::printf("Exp(lam=2) f(0)=%.6f  (expect 2)\n", exponential_pdf(0.0, 2.0));
    return 0;
}
