#pragma once
// 多项式逐项积分 + 梯形法则。不依赖 Eigen。
#include <cmath>

constexpr int POLY_MAX = 8;
struct Poly {
    int n = 0;
    double c[POLY_MAX + 1]{};
};

inline double poly_eval(const Poly& p, double x) {
    double s = 0.0;
    double xp = 1.0;
    for (int i = 0; i <= p.n; ++i) {
        s += p.c[i] * xp;
        xp *= x;
    }
    return s;
}

// ∫ p 取原函数常数项为 0：x^{k} → x^{k+1}/(k+1)
inline Poly poly_int(const Poly& p) {
    Poly q{};
    q.n = p.n + 1;
    if (q.n > POLY_MAX)
        q.n = POLY_MAX;
    q.c[0] = 0.0;
    for (int i = 0; i <= p.n && i + 1 <= POLY_MAX; ++i)
        q.c[i + 1] = p.c[i] / static_cast<double>(i + 1);
    return q;
}

// 复合梯形：∫_a^b f ≈ (h/2)*(f(a)+f(b)) + h*中间点
inline double trapezoid(double (*f)(double), double a, double b, int n) {
    double h = (b - a) / static_cast<double>(n);
    double s = 0.5 * (f(a) + f(b));
    for (int i = 1; i < n; ++i)
        s += f(a + i * h);
    return s * h;
}
