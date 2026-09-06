#include "mle.hpp"
#include <cstdio>

int main() {
    const int coins[] = {1, 0, 1, 1, 0, 1, 1, 1, 0, 1};
    std::size_t n = 10;
    std::printf("Bernoulli MLE p_hat=%.6f  (7/10=0.7)\n", bernoulli_mle(coins, n));

    const double x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double mu, s2;
    gaussian_mle(x, 5, mu, s2);
    std::printf("Gaussian MLE  mu=%.6f  sigma2=%.6f  (3 and 2)\n", mu, s2);
    return 0;
}
