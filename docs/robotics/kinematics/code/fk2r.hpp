#pragma once
// 平面 2R 正运动学。不依赖 Eigen。
#include <cmath>

inline void fk_2r(double l1, double l2, double th1, double th2,
                  double& x, double& y) {
    x = l1 * std::cos(th1) + l2 * std::cos(th1 + th2);
    y = l1 * std::sin(th1) + l2 * std::sin(th1 + th2);
}
