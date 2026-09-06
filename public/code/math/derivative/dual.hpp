#pragma once
// 对偶数：一次求值同时得到 f 和 f'。不依赖 Eigen。
#include <cmath>

struct Dual {
    double v = 0.0;  // 函数值
    double d = 0.0;  // 对自变量的导数
    static Dual var(double x) { return Dual{x, 1.0}; }   // 自变量：导数种子 = 1
    static Dual cnst(double x) { return Dual{x, 0.0}; }  // 常数：导数 = 0
};

inline Dual operator+(Dual a, Dual b) { return Dual{a.v + b.v, a.d + b.d}; }
inline Dual operator-(Dual a, Dual b) { return Dual{a.v - b.v, a.d - b.d}; }
inline Dual operator-(Dual a) { return Dual{-a.v, -a.d}; }
inline Dual operator+(Dual a, double b) { return a + Dual::cnst(b); }
inline Dual operator+(double a, Dual b) { return Dual::cnst(a) + b; }
inline Dual operator-(Dual a, double b) { return a - Dual::cnst(b); }
inline Dual operator-(double a, Dual b) { return Dual::cnst(a) - b; }

inline Dual operator*(Dual a, Dual b) {
    return Dual{a.v * b.v, a.v * b.d + a.d * b.v};
}
inline Dual operator*(Dual a, double b) { return a * Dual::cnst(b); }
inline Dual operator*(double a, Dual b) { return Dual::cnst(a) * b; }

inline Dual operator/(Dual a, Dual b) {
    return Dual{a.v / b.v, (a.d * b.v - a.v * b.d) / (b.v * b.v)};
}
inline Dual operator/(Dual a, double b) { return a / Dual::cnst(b); }
inline Dual operator/(double a, Dual b) { return Dual::cnst(a) / b; }

inline Dual sin(Dual x) { return Dual{std::sin(x.v), std::cos(x.v) * x.d}; }
inline Dual cos(Dual x) { return Dual{std::cos(x.v), -std::sin(x.v) * x.d}; }
inline Dual exp(Dual x) {
    double e = std::exp(x.v);
    return Dual{e, e * x.d};
}

// 多项式 p(x)=c[0]+c[1]x+...+c[n]x^n。n 为次数。
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

inline Poly poly_diff(const Poly& p) {
    Poly q{};
    if (p.n <= 0)
        return q;
    q.n = p.n - 1;
    for (int i = 1; i <= p.n; ++i)
        q.c[i - 1] = static_cast<double>(i) * p.c[i];
    return q;
}

// 中心差分：黑盒数值导数。h 太小会吃掉有效数字。
inline double central_diff(double (*f)(double), double x, double h = 1e-6) {
    return (f(x + h) - f(x - h)) / (2.0 * h);
}
