---
title: "C++ 基础 — demo.cpp"
---

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# C++ 基础 — demo.cpp / vec2.hpp

<a href="/notebook/code/programming/cpp/demo.cpp" target="_blank" download>Download demo.cpp</a>
<a href="/notebook/code/programming/cpp/vec2.hpp" target="_blank" download>Download vec2.hpp</a>

## 运行方式

```bash
cd docs/programming/cpp/code
g++ -std=c++17 demo.cpp -o cpp_demo
./cpp_demo
```

`vec2.hpp` 是 header-only：`#pragma once` + `inline` 运算符。`demo.cpp` 只 `#include` 再打印 \(\|a\|^2=25\)、`a+2b`。

对照数学章：`g++ -std=c++17 demo.cpp` 同一套路，不必装 Eigen。

## 源码位置

`docs/programming/cpp/code/demo.cpp`  
`docs/programming/cpp/code/vec2.hpp`
