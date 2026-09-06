#include "fk2r.hpp"
#include <cstdio>

int main() {
    double x, y;
    fk_2r(1.0, 0.7, 0.0, 0.0, x, y);
    std::printf("theta=0  (x,y)=(%.6f, %.6f)  expect (1.7, 0)\n", x, y);
    fk_2r(1.0, 0.7, 0.4, -0.7, x, y);
    std::printf("theta=(0.4,-0.7)  (x,y)=(%.6f, %.6f)\n", x, y);
    return 0;
}
