---
title: "algo16 计算几何与博弈论入门 — exercise.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# algo16 计算几何与博弈论入门 — exercise.py 练习指南

<a href="/notebook/code/algorithms/topics/geometry-game/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过五个练习巩固计算几何和博弈论基础：多边形面积、同侧判断、Nim 胜负率分析、Alpha-Beta 剪枝和三角形内点判断。

## 预备知识

- 叉积的有向面积公式和 Shoelace 公式
- 线段相交的跨越测试原理
- Nim 游戏 XOR 判定
- Minimax 博弈树搜索
- Alpha-Beta 剪枝概念

## 任务清单

### 任务1：多边形面积（Shoelace 公式）

$$Area = \frac{1}{2} \left| \sum_{i=1}^{n} (x_i y_{i+1} - x_{i+1} y_i) \right|$$

其中 $(x_{n+1}, y_{n+1}) = (x_1, y_1)$ 是闭合的。

**几何直觉**：Shoelace 公式将多边形划分为顶点与原点构成的三角形，累加有向面积。

### 任务2：同侧判断

- 计算 `cross(line_start, line_end, p1)` 和 `cross(line_start, line_end, p2)`。
- 两者乘积 ≥ 0 → 同侧（或不严格地跨在直线上）。
- 乘积 < 0 → 异侧。

### 任务3：Nim 胜负率分析

- 枚举所有 `(a, b, c)` 三堆组合
- 统计 XOR ≠ 0 的比例
- 有趣的事实：在三堆 Nim 中，必败态（XOR=0）的比例随着 `a_max` 增大趋近于 0

### 任务4：Alpha-Beta 剪枝

- 维护 alpha（MAX 已知最佳下界）和 beta（MIN 已知最佳上界）
- 当 `alpha >= beta` 时，当前分支不可能被选用 → 剪枝

### 任务5：点在三角形内

- 计算三个叉积：`cross(a,b,p)`, `cross(b,c,p)`, `cross(c,a,p)`
- 三者同号 → p 在三角形内

## 提示

1. Shoelace 公式中 `abs()` 确保返回正值面积。
2. 同侧判断注意叉积为 0（点在直线上）的边界情况。
3. Nim 分析用 `itertools.product` 高效枚举。
4. 三角形内点判断用同号法，比射线法更简洁。

<<< @/algorithms/topics/geometry-game/code/exercise.py
