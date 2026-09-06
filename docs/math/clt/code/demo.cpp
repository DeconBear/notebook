#include "clt.hpp"
#include <cstdio>

int main() {
    const double lam = 1.0;  // Exp(1)，均值 1、方差 1
    const int n = 30;
    const int n_rep = 2000;
    Lcg rng(42);
    double acc = 0.0;
    double acc2 = 0.0;
    for (int r = 0; r < n_rep; ++r) {
        double m = mean_of_n(rng, n, lam);
        acc += m;
        acc2 += m * m;
    }
    double mu_hat = acc / n_rep;
    double var_hat = acc2 / n_rep - mu_hat * mu_hat;
    std::printf("Exp(1) sample-mean n=%d, %d repeats\n", n, n_rep);
    std::printf("E[bar X] ~ %.4f  (theory 1)\n", mu_hat);
    std::printf("Var(bar X) ~ %.4f  (theory 1/n=%.4f)\n", var_hat, 1.0 / n);
    return 0;
}
