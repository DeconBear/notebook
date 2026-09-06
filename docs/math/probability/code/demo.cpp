#include "stats.hpp"
#include <cstdio>

int main() {
    const double x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    const std::size_t n = 5;
    std::printf("mean=%.6f  (expect 3)\n", sample_mean(x, n));
    std::printf("s^2 unbiased=%.6f  (expect 2.5)\n", sample_var(x, n, true));
    std::printf("MLE  /n     =%.6f  (expect 2.0)\n", sample_var(x, n, false));
    return 0;
}
