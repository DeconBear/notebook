---
title: "李群与李代数 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 李群与李代数 — Python / C++ 代码详解

<a href="/notebook/code/robotics/lie-groups/demo.py" target="_blank" download>Download demo.py</a>
<a href="/notebook/code/robotics/lie-groups/so3.hpp" target="_blank" download>Download so3.hpp</a>
<a href="/notebook/code/robotics/lie-groups/demo.cpp" target="_blank" download>Download demo.cpp</a>

## 运行方式

Python：

```bash
cd docs/robotics/lie-groups/code
python demo.py
```

C++（header-only，只需能编译 C++17 的 `g++`）：

```bash
cd docs/robotics/lie-groups/code
g++ -std=c++17 demo.cpp -o so3_demo
./so3_demo
```

Windows 若没有 `./`，直接 `so3_demo.exe`。不要加 `-l` 任何库。`so3.hpp` 必须和 `demo.cpp` 同一目录（`#include "so3.hpp"`）。

Python 出图 `so3_exp.png`：灰点原立方体，橙点旋转后，蓝箭头为 $\omega$。C++ 只印 `det(R)`，应约为 $1$。两边对同一向量 $\omega=(0.3,-0.1,0.8)$ 做 Rodrigues。

## 代码逐段详解（Python）

### 第1步：`hat` — $\mathbb{R}^3\to\mathfrak{so}(3)$

$$
\hat\omega
=
\begin{pmatrix}
0 & -\omega_z & \omega_y \\
\omega_z & 0 & -\omega_x \\
-\omega_y & \omega_x & 0
\end{pmatrix}
$$

```python
def hat(w):
    """R^3 → so(3)：叉乘矩阵。"""
    x, y, z = w
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
```

- **`x, y, z = w`**：解包长度为 3 的序列。少一个数会 `ValueError`。
- **$(0,2)$ 位置是 $+\omega_y$**：练习 `hat_02` 就是这一格。符号写反则 $\exp$ 转反方向。
- **反对称**：`hat(w).T` 应等于 `-hat(w)`。`ω × p = hat(ω) @ p`。

---

### 第2步：`so3_exp` — Rodrigues

$\theta=\|\omega\|$，$K=\hat\omega$（此处 **$K$ 不是单位**反对称阵）：

$$
R=I+\frac{\sin\theta}{\theta}K+\frac{1-\cos\theta}{\theta^2}K^2
$$

```python
def so3_exp(w):
    th = np.linalg.norm(w)
    K = hat(w)
    if th < 1e-10:
        return np.eye(3) + K
    return np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / (th * th) * (K @ K)
```

- **小角度**：$\sin\theta/\theta\to 1$，公式趋向 $I+K$。阈值 $10^{-10}$ 避免除零；一阶 $I+K$ 对微小 $\omega$ 够用。
- **`K @ K` 必须矩阵乘**：`K * K` 是逐元素平方，不是 $\hat\omega^2$。
- **`th * th`**：分母 $\theta^2$。不要写成 `th ** 2` 也行，本文件用乘法。
- **与「单位轴 + 角」教材**：若 $\omega=\theta n$、$n$ 单位，则 $K=\theta\hat n$，代入后与 $I+\sin\theta\,\hat n+(1-\cos\theta)\hat n^2$ 相同。

---

### 第3步：`so3_log` — 轴角

```python
def so3_log(R):
    """主值轴角；输入须为 SO(3)，分别处理小角和接近 π 的退化。"""
    c = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    vee = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0],
                    R[1, 0] - R[0, 1]])
    sin_th = 0.5 * np.linalg.norm(vee)
    th = np.arctan2(sin_th, c)
    if th < 1e-7:
        # θ/(2 sinθ) → 1/2；保留微小旋转，不能直接返回零。
        return 0.5 * vee
    if np.pi - th < 1e-6:
        # 对称部分沿转轴的特征值为 1，另两个为 cosθ。
        # 这避免了在 π 附近除以几乎为零的 sinθ。
        _, axes = np.linalg.eigh(0.5 * (R + R.T))
        axis = axes[:, -1]
        if sin_th > 1e-12:
            if np.dot(axis, vee) < 0.0:
                axis = -axis
        elif axis[np.argmax(np.abs(axis))] < 0.0:
            axis = -axis  # 精确 π 时 ±轴等价，固定一种符号。
        return th * axis
    return (th / (2.0 * sin_th)) * vee
```

- **$\cos\theta=(\mathrm{tr}R-1)/2$**：$\mathrm{SO}(3)$ 的标准提取。`clip` 把迹的浮点误差限制在合法余弦范围。实现用 `atan2(sin_th, c)` 同时利用正弦与余弦恢复主值角。
- **轴 $n$ 来自反对称部分**：$(R-R^\top)$ 的独立三元除以 $2\sin\theta$。下标 `R[2,1]-R[1,2]` 对应 $x$ 分量。
- **小角度**：用 `0.5 * vee` 保留一阶旋转量，避免迹舍入为 3 时丢掉微小旋转。转角用 `atan2(sin_th, c)`，比仅用 `arccos` 更稳。
- **接近 $\pi$**：从对称部分的最大特征值对应特征向量恢复转轴，不再除以趋零的 $\sin\theta$。精确 $\pi$ 时 $\pm n$ 等价；应检查 `exp(log(R)) ≈ R`，不要强求轴的符号唯一。

主程序核验：

```python
    w = np.array([0.3, -0.1, 0.8])
    R = so3_exp(w)
    w2 = so3_log(R)
    print('R^T R ≈ I', np.round(R.T @ R, 6))
    print('det R    ', np.linalg.det(R))
```

`w2` 应接近 `w`。`R.T @ R` 应接近单位阵（打印到 6 位）。`det` 应接近 $+1$（反射会是 $-1$，那不在 $\mathrm{SO}(3)$）。

---

### 第4步：立方体可视化

```python
    cube = np.array([[1, 1, 1], [1, 1, -1], ...], dtype=float) * 0.4
    ax = fig.add_subplot(111, projection='3d')
    rot = (so3_exp(w) @ cube.T).T
    ax.quiver(0, 0, 0, w[0], w[1], w[2], color='#2980B9', lw=2)
```

- **8 个 $(\pm1,\pm1,\pm1)$ 顶点**再缩放 $0.4$，只为好看，与公式无关。
- **`R @ cube.T`**：`cube` 是 `(8,3)`，矩阵乘要 `(3,3)@(3,8)`，再 `.T` 回去。写成 `cube @ R.T` 等价。
- **`projection='3d'`**：需要 `mpl_toolkits`（随 matplotlib 来）。`quiver` 从原点画 $\omega$，几何上是转轴方向，长度是转角。

---

## 代码逐段详解（C++）

### `so3.hpp`：与 Python 同一公式

```cpp
inline std::array<std::array<double, 3>, 3> hat(double x, double y, double z) {
    return {{{0, -z, y}, {z, 0, -x}, {-y, x, 0}}};
}
```

- **`#pragma once`**：防止头文件被 include 两次。
- **`inline`**：函数定义在头文件里，多个翻译单元链接时不重复定义。
- **`std::array<std::array<double,3>,3>`**：3×3，不用自己写 `double[3][3]` 当返回类型（C 数组不能直接返回得那么干净）。
- **三层花括号**：外层初始化 `array`，中层三行，内层三个 `double`。`hat` 的 $(0,2)$ 仍是 `y`。

```cpp
inline std::array<std::array<double, 3>, 3> so3_exp(double wx, double wy, double wz) {
    const double th = std::sqrt(wx * wx + wy * wy + wz * wz);
    auto K = hat(wx, wy, wz);
    std::array<std::array<double, 3>, 3> R{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
```

- **`std::sqrt`**：来自 `<cmath>`。没有 NumPy 的 `norm`。
- **`R` 先设成 $I$**，再往上加 $\frac{\sin\theta}{\theta}K$ 与 $\frac{1-\cos\theta}{\theta^2}K^2$。

两个局部 lambda：

```cpp
    auto add = [&](double s, const auto& M) {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) R[i][j] += s * M[i][j];
    };
    auto mul = [](const auto& A, const auto& B) {
        std::array<std::array<double, 3>, 3> C{};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 3; ++k) C[i][j] += A[i][k] * B[k][j];
        return C;
    };
```

- **`[&]`**：`add` 捕获外层 `R` 的引用，才能原地累加。`mul` 无捕获，纯函数。
- **`C{}`**：值初始化为全 0，再做三重循环矩阵乘。这就是 Python 的 `K @ K`。
- **小角度** `th < 1e-12`：`add(1.0, K)` 即 $I+K$，与 Python $10^{-10}$ 同思路、阈值略不同，对 demo 的 $\omega$ 无影响。

```cpp
    add(std::sin(th) / th, K);
    add((1 - std::cos(th)) / (th * th), mul(K, K));
    return R;
```

没有 `so3_log`：C++ 侧只演示 $\exp$ 落到 $\det=1$。

---

### `demo.cpp`：行列式

```cpp
#include "so3.hpp"
#include <cstdio>

int main() {
    auto R = so3_exp(0.3, -0.1, 0.8);
    double det =
        R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1]) -
        R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0]) +
        R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]);
    std::printf("det(R) = %.6f  (should be 1)\n", det);
    return 0;
}
```

- **引号 include `"so3.hpp"`**：先搜当前目录。
- **3×3 行列式展开**：按第一行余子式。没有 Eigen。应打印 `1.000000` 附近。
- **`%.6f`**：六位小数。`return 0` 表示成功。

`g++ -std=c++17 demo.cpp -o so3_demo`：`-std=c++17` 因为用了 `auto` lambda；`-o` 指定可执行文件名。不要编译 `so3.hpp` 本身。

---

### 关键概念速查表

| 概念 | 数学 / 直觉 | 代码 |
|------|-------------|------|
| `hat` | 叉乘矩阵 | `hat(w)` / `hat(x,y,z)` |
| $(0,2)$ | $\omega_y$ | 练习 `hat_02` |
| Rodrigues | $I+\mathrm{sinc}K+\cdots K^2$ | `so3_exp` |
| 小 $\theta$ | $I+K$ | `th < 1e-10` |
| `K @ K` | 矩阵平方 | Python `@`；C++ `mul` |
| $\log$ | 轴角 | 仅 Python `so3_log` |
| `trace` | $(\mathrm{tr}R-1)/2=\cos\theta$ | `np.trace` |
| $\det R$ | 应为 $+1$ | `np.linalg.det` / 手写展开 |
| 立方体 | $Rp$ | `R @ cube.T` |
| 编译 | C++17 | `g++ -std=c++17 demo.cpp -o so3_demo` |

## 源码位置

clone 后打开（相对仓库根目录）：

- `docs/robotics/lie-groups/code/demo.py`
- `docs/robotics/lie-groups/code/so3.hpp`
- `docs/robotics/lie-groups/code/demo.cpp`
