---
title: "李群与李代数"
order: 40
---
# 李群与李代数：旋转为什么不能当向量加

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

> 三维旋转矩阵 $R$ 满足 $R^\top R=I$ 且 $\det R=1$，全体记作 $\mathrm{SO}(3)$。两个旋转的「平均」或「一点点增量」都不能把九个数字当 $\mathbb{R}^9$ 加减——会走出合法姿态。正确的小量住在单位元处的切空间 **$\mathfrak{so}(3)$**（反对称矩阵），再用 **指数映射** 送回 $\mathrm{SO}(3)$。本章 Python 与 C++ 实现同一套 Rodrigues；平面刚体的「转 + 移」见 [旋量](/robotics/screw/)。

---

## 一、$\mathrm{SO}(3)$ 是流形，$\mathfrak{so}(3)$ 是切空间

单位元 $I$ 是「什么都不转」。在 $I$ 旁边，合法的无穷小位移是反对称的：$\hat\omega^\top=-\hat\omega$。三维反对称矩阵恰好三个自由量，与轴角向量 $\omega\in\mathbb{R}^3$ 一一对应（`hat`）：

$$
\hat\omega
=
\begin{pmatrix}
0 & -\omega_z & \omega_y \\
\omega_z & 0 & -\omega_x \\
-\omega_y & \omega_x & 0
\end{pmatrix}.
$$

叉乘 $\omega\times p=\hat\omega\, p$。指数映射把切空间里的有限长向量变成一个真旋转：

$$
R=\exp(\hat\omega)\in\mathrm{SO}(3).
$$

![SO(3) 与 so(3)](./images/rob-04-so3.png)

> **图解说明**：左：球面示意旋转流形，切平面是 $\mathfrak{so}(3)$，曲线箭头是 $\exp$。右：$\omega$ 变成 $\hat\omega$，Rodrigues 给出绕轴转角。总结：群上的点是姿态，代数上的向量是「转多少」。

量子信息里的幺正群 $\mathrm{U}(n)$ 是同一类故事（群 + 代数 + 指数），只是矩阵换成复数；见 [量子信息全景](/quantum/overview/)。线性代数复习见 [向量与矩阵](/math/linear-algebra/)。

---

## 二、Rodrigues 公式（代码用的形式）

令 $\theta=\|\omega\|$，$K=\hat\omega$（**不是**单位反对称阵）。则

$$
\exp(\hat\omega)
=
I + \frac{\sin\theta}{\theta}K + \frac{1-\cos\theta}{\theta^2}K^2
\qquad(\theta\to 0\text{ 时退回 }I+K).
$$

教材若写单位轴 $\hat n$ 与转角 $\theta$，则 $K=\theta\hat n$，两种写法等价。对数映射反向：

$$
\theta=\arccos\frac{\mathrm{tr}(R)-1}{2},\quad
n=\frac{1}{2\sin\theta}\begin{pmatrix}R_{32}-R_{23}\\ R_{13}-R_{31}\\ R_{21}-R_{12}\end{pmatrix},
\quad \omega=\theta n.
$$

$\theta\approx 0$ 时 $\log$ 返回 $0$，避免除零。

---

## 三、Python 与 C++ 对照

同一组 $\omega=(0.3,-0.1,0.8)$：

- Python `so3_exp` / `so3_log` 验证 $\log(\exp(\omega))\approx\omega$、$R^\top R\approx I$、$\det R=1$，并把立方体顶点转过去；
- `so3.hpp` 里 `hat` / `so3_exp` 是同样的 Rodrigues；`demo.cpp` 只印 $\det(R)$。

编译（在 `docs/robotics/lie-groups/code/`）：

```bash
g++ -std=c++17 demo.cpp -o so3_demo
```

头文件是 header-only，不必再链库。数值应与 Python 同一 $\omega$ 下 $\det\approx 1$。

---

## 四、代码在做什么

`demo.py` 打印往返误差与正交性，三维散点图 `so3_exp.png`：灰点原立方体，橙点 $Rp$，蓝箭头是 $\omega$ 轴。

![指数映射旋转立方体](./images/so3_exp.png)

---

## 五、小结

| 概念 | 一句话 |
|------|--------|
| $\mathrm{SO}(3)$ | 合法三维旋转矩阵 |
| $\mathfrak{so}(3)$ | 反对称；$3$ 个数 $= \omega$ |
| `hat` | $\mathbb{R}^3\to$ 叉乘矩阵 |
| $\exp/\log$ | 切空间 $\leftrightarrow$ 群 |
| 下游 | 位姿滤波、IMU、旋量 $\mathrm{SE}(3)$ |

> 下一章 [机构学](/robotics/mechanisms/) 先离开矩阵，看闭链怎么动。把转动与平移合成螺旋见 [旋量代数](/robotics/screw/)。DH 链上的 $R$ 块正是 $\mathrm{SO}(3)$ 里的点，见 [建模](/robotics/modeling/)。

## 📥 Code

| File | View | Download |
|------|------|----------|
| demo.py | [Open](./code-demo) | <a href="/notebook/code/robotics/lie-groups/demo.py" target="_blank" download>Download</a> |
| so3.hpp | — | <a href="/notebook/code/robotics/lie-groups/so3.hpp" target="_blank" download>Download</a> |
| demo.cpp | — | <a href="/notebook/code/robotics/lie-groups/demo.cpp" target="_blank" download>Download</a> |
| exercise.py | [Open](./code-exercise) | <a href="/notebook/code/robotics/lie-groups/exercise.py" target="_blank" download>Download</a> |

## 参考

1. Murray, Li, Sastry, *A Mathematical Introduction to Robotic Manipulation*
2. Solà, “Quaternion kinematics for the error-state Kalman filter”（$\exp/\log$ 工程笔记）
