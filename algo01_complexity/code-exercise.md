---
title: "algo01 复杂度分析与渐进记号 — exercise.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# algo01 复杂度分析与渐进记号 — exercise.py 练习指南

<a href="../code/algo01_complexity/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过四个递进的任务，掌握复杂度分析的核心技能：代码复杂度推断、均摊分析实现、主定理应用、势能法分析。

## 预备知识

在开始练习前，确保你已经理解了以下概念（参见 [demo.py 代码详解](./code-demo) 中的详细解释）：

- 大 O 记号的含义：忽略常数和低阶项，只关注增长率
- 常见循环模式的复杂度：单层循环 $O(n)$、嵌套循环 $O(n^2)$、每次减半的循环 $O(\log n)$
- 均摊分析的基本思想：贵的操作被便宜的操作摊薄
- 主定理的三种情形及其适用条件

## 任务清单

### 任务1：分析代码复杂度

分析 `func_a` 到 `func_e` 五个函数的大 O 复杂度，将答案填入 TODO 注释中。

**提示**：
- **func_a**：两个独立的循环，都是 $O(n)$ → 总复杂度
- **func_b**：外层 $n$ 次，内层 $j$ 每次 $\times 2$（即 $\log_2 n$ 次） → 总复杂度
- **func_c**：递推式 $T(n) = 2T(n/2) + O(1)$ → 应用主定理情形 1
- **func_d**：外层 while 循环变量 $i$ 每次 $\div 2$，内层循环 $n$ 次 → 总复杂度
- **func_e**：外层 $n$ 次，每次内层做二分查找 $O(\log n)$ → 总复杂度

### 任务2：模拟动态数组均摊分析

在 `SimulatedDynamicArray.append()` 方法中完成扩容逻辑：

```
当 size == capacity 时：
  1. capacity *= 2
  2. total_copies += size  (复制旧元素)
当 size < capacity 时：
  直接追加，不扩容
```

**验证标准**：运行 `task2_simulate()` 后：
- 最终均摊代价（total_copies / size）应 $\leq 2.0$
- 程序输出 `✅ 均摊代价验证通过！`

### 任务3：主定理练习

分析五个递推式，确定 $a, b, \log_b(a)$ 以及属于主定理的哪种情形。

| 递推式 | $a$ | $b$ | $\log_b a$ | $f(n)$ | 情形 | 结果 |
|-------|-----|-----|-----------|--------|------|------|
| $T(n)=2T(n/2)+n$ | 2 | 2 | 1 | $n$ | 2 | $\Theta(n \log n)$ |
| $T(n)=T(n/2)+1$ | 1 | 2 | 0 | $1$ | 2 | $\Theta(\log n)$ |
| $T(n)=3T(n/2)+n^2$ | 3 | 2 | ~1.585 | $n^2$ | 3 | $\Theta(n^2)$ |
| $T(n)=4T(n/2)+n$ | 4 | 2 | 2 | $n$ | 1 | $\Theta(n^2)$ |
| $T(n)=T(2n/3)+1$ | 1 | 3/2 | 0 | $1$ | 2 | $\Theta(\log n)$ |

### 任务4（Bonus）：势能法验证

使用势能函数 $\Phi = 2 \times size - capacity$ 计算每次 append 的均摊代价。

**核心公式**：
$$\hat{c}_i = c_i + \Phi(D_i) - \Phi(D_{i-1})$$

其中 $c_i$ 是第 $i$ 次操作的实际代价，$\Phi(D_i)$ 是操作后数据结构的势能。

**期望结果**：每次 append 的均摊代价 $\hat{c}_i$ 恒为常数 3（无论是否发生扩容）。

**势能法的直觉**：
- 当数组不满时：$\Phi$ 增加（储存势能），抵消了实际 $c_i=1$ → 均摊后变 3
- 当数组满需要扩容时：$\Phi$ 大幅减少（释放势能），抵消了实际 $c_i=1+size$ → 均摊后还是 3

## 运行与验证

```bash
cd algo01_complexity/code
python exercise.py
```

如果你的实现正确，应该看到：
- 任务1的分析与提示一致
- 任务2的均摊代价 $\leq 2.0$
- 任务4的势能法均摊代价恒为 3
- 所有 assert 断言通过

## 完整代码

<<< @/snippets/algo01_complexity/exercise.py
