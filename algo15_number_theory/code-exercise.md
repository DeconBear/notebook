---
title: "algo15 数论与组合数学 — exercise.py"
---

# algo15 数论与组合数学 — exercise.py 练习指南

<a href="../code/algo15_number_theory/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过四个练习巩固数论核心概念：欧拉函数、线性筛扩展、大数快速幂和容斥原理应用。

## 预备知识

- 欧拉函数的定义和质因子分解公式：$\varphi(n) = n \cdot \prod_{p|n} (1 - 1/p)$
- 线性筛中每个合数只被最小质因子筛一次的原理
- 二进制快速幂的实现（取模版本和大数版本相同）
- 容斥原理的集合交并转换

## 任务清单

### 任务1：欧拉函数 `euler_phi(n)`

- **公式**：$\varphi(n) = n \times \prod_{p|n} (1 - \frac{1}{p})$。
- **实现方法**：在质因子分解的过程中计算。对每个质因子 $p$，先除尽 $p$，然后 `result -= result // p`。

### 任务2：批量求欧拉函数 `euler_phi_range(n)`

- **线性筛扩展**：当发现素数 $p$ 时，$\varphi(p) = p - 1$。
- 当筛掉合数 `i*p` 时：
  - 若 `i % p == 0`（p 最小质因子）：$\varphi(i \cdot p) = \varphi(i) \cdot p$
  - 否则：$\varphi(i \cdot p) = \varphi(i) \cdot (p - 1)$

### 任务3：大数快速幂 `fast_pow_big(x, n)`

- 与取模版本结构完全相同，只是去掉 `% mod`。
- Python 自带大整数支持，可以计算非常大的精确值。

### 任务4：容斥互质计数 `count_coprimes_up_to(n, m)`

- **步骤1**：分解 $m$ 的质因子集合 $P$。
- **步骤2**：用容斥原理计算 $1 \sim n$ 中至少被 $P$ 中某个质因子整除的数的个数。
- **步骤3**：$n$ 减去上一步结果，即为与 $m$ 互质的数的个数。

## 提示

1. 欧拉函数中注意质因子分解后用 `result -= result // p` 而非 `result *= (1 - 1/p)`（整数运算避免浮点）。
2. 批量欧拉函数是线性筛的经典扩展之一。
3. 容斥枚举子集使用位运算 `mask` 遍历。

<<< @/snippets/algo15_number_theory/exercise.py
