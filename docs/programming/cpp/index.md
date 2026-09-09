---
title: "C++ 基础"
order: 20
---
# C++ 基础：为了读懂本仓库的 `*.hpp`

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

> 数学章和李群用 **header-only** 的手写 C++：一个 `.hpp` 里既有声明也有实现，`g++ -std=c++17 demo.cpp` 就能编，**不必安装 Eigen**。本章只覆盖读那些文件够用的语法。Python 默认语言见 [上一章](/programming/python/)；深度学习回 [PyTorch](/programming/pytorch/)。

C++ 在本仓库里不是第二门课，是「把二维向量、对偶数、李群指数写成能编译的公式」。读 `.hpp` 时盯三件事：头文件怎样避免重复定义、`Vec2&` 改的是不是原对象、模板在编译期复印出哪一版函数。

## 一、编译一条命令

```bash
cd docs/programming/cpp/code
g++ -std=c++17 demo.cpp -o cpp_demo
./cpp_demo          # Windows 上是 cpp_demo.exe
```

- `demo.cpp` 是**翻译单元**（真正被编译的那份源）。
- `#include "vec2.hpp"` 在预处理阶段把头文件**粘进来**。
- `-std=c++17`：本仓库按 C++17 写（结构化绑定、`if constexpr` 用得少，但统一这个标准）。
- 没有 `main` 的 `.hpp` 不能单独链接成程序。

Windows 上若命令不存在，需要安装 MinGW-w64 或 MSVC，并把 `g++` 放进 `PATH`。编不过时先看报错第一行的文件名和行号。

## 二、头文件在防什么

```cpp
#pragma once          // 这份头只被包含一次，避免重复定义
struct Vec2 {
    double x = 0.0;
    double y = 0.0;
};
inline Vec2 operator+(Vec2 a, Vec2 b) { return Vec2{a.x + b.x, a.y + b.y}; }
```

- `struct`：一堆字段打成一种类型。默认字段公开（`class` 默认私有，本仓库演示常用 `struct`）。
- `inline`：函数写在头文件里时，多个 `.cpp` 包含同一份也不报「重复定义」。header-only 几乎都要 `inline`（模板除外，模板本身按需实例化）。
- `#pragma once` 等价于传统 include guard：`#ifndef FOO_HPP` … `#endif`。

**不要**在头文件里写 `using namespace std;`：会污染所有包含它的翻译单元。

::: details 逐步说明：`#include` 之后编译器看见的是什么（点击展开）

预处理先把 `#include "vec2.hpp"` 换成文件内容。若 A.hpp 含 B.hpp，B.hpp 又含 A.hpp，没有 `#pragma once` 就会把 `struct Vec2` 粘两遍，链接器或编译器报 redefinition。

`inline` 的日常含义：允许这个函数的定义出现在多个翻译单元里，链接时合并成一份。header-only 把实现写在头里，每个 `demo.cpp` 都会得到一份拷贝，没有 `inline`（或模板）就会 multiple definition。

**值 vs 引用数字例。** `Vec2 a{1, 0}`。`void f(Vec2 v){ v.x = 9; }` 之后 `a.x` 仍是 1，因为 `v` 是副本。`void g(Vec2& v){ v.x = 9; }` 之后 `a.x==9`。`const Vec2&` 既不拷贝大对象，又禁止 `v.x=9`。数学章的 `operator*` 常返回新的 `Vec2`（值），不改入参。

模板 `square<T>`：见到 `square(3)` 生成 `int square(int)`；`square(3.0)` 生成 `double` 版。这是编译期多态，不是 Python 那种运行时鸭子类型。本仓库为少依赖，许多 `.hpp` 直接写 `double`，不逼你先懂模板。

:::

## 三、值、引用、const

```cpp
void scale(Vec2& v, double k) { v.x *= k; v.y *= k; }   // 引用：改的是原对象
double n2(const Vec2& v) { return v.x * v.x + v.y * v.y; }  // 只读引用，不拷贝
```

- `Vec2 v;`：值，拷贝语义。函数参数若写 `Vec2 v`，进来的是副本。
- `Vec2&`：引用，别名，没有空引用。
- `const Vec2&`：只读别名，大对象别拷贝、又不想被改。
- `*` 指针本章不展开；读 `*.hpp` 几乎全是值和引用。

## 四、模板：一份代码，多种类型

[导数章](/math/derivative/) 的对偶数是 `struct Dual`。更一般地，模板让「对 `double` 和 `float` 用同一套加法」：

```cpp
template <typename T>
T square(T x) { return x * x; }
```

编译器见到 `square(3)` 就生成 `int` 版，见到 `square(3.0)` 就生成 `double` 版。数学章为了少依赖，常常**不**上模板，直接写 `double`。

可选的 Eigen 片段只出现在各章 `code-demo.md` 里，仓库**不附带** Eigen 源码。

## 五、和 Python 对一下

| Python | C++（本仓库风格） |
|--------|-------------------|
| `a = [1.0, 2.0]` | `double a[2] = {1.0, 2.0};` 或手写 `Vec2` |
| `def f(x):` | `double f(double x) { ... }` |
| `class` + `__init__` | `struct` + 构造函数（演示里常用聚合初始化 `Vec2{1, 2}`） |
| 解释执行 | 先编译再跑 |
| `nn.Linear` 那种运行时对象 | 编译期类型；深度学习用 Python 调 PyTorch |

下一章：[PyTorch 张量](/programming/pytorch/)。

## 📥 Code

| File | View | Download |
|------|------|----------|
| demo.cpp / vec2.hpp | [Open](./code-demo) | <a href="/notebook/code/programming/cpp/demo.cpp" target="_blank" download>Download demo.cpp</a> |
| exercise.py | [Open](./code-exercise) | <a href="/notebook/code/programming/cpp/exercise.py" target="_blank" download>Download</a> |
