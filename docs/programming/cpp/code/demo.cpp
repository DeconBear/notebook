#include "vec2.hpp"
#include <cstdio>

int main() {
    Vec2 a{3.0, 4.0};
    Vec2 b{1.0, 0.0};
    Vec2 s = a + b * 2.0;
    std::printf("a=(%.1f, %.1f)  ||a||^2=%.1f\n", a.x, a.y, norm2(a));
    std::printf("a + 2b = (%.1f, %.1f)\n", s.x, s.y);
    const Vec2& r = a;  // 只读引用，不拷贝
    std::printf("const ref still (%.1f, %.1f)\n", r.x, r.y);
    return 0;
}
