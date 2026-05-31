---
title: "algo15 数论与组合数学 — demo.py"
---

# algo15 数论与组合数学 — demo.py 代码详解

<a href="../code/algo15_number_theory/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd algo15_number_theory/code
python demo.py
```

## 代码逐段详解

### 第1步：二进制快速幂

```python
def fast_pow(x, n, mod):
    result = 1
    x = x % mod
    while n > 0:
        if n & 1:
            result = (result * x) % mod
        x = (x * x) % mod
        n >>= 1
    return result
```

**原理**：将指数 $n$ 写成二进制。例如 $n=13 = 1101_2$，则 $x^{13} = x^8 \cdot x^4 \cdot x^1$。算法通过不断平方 $x$ 来获得 $x^2, x^4, x^8, \dots$，并在二进制位为 1 时乘入结果。

**运算步骤**（$2^{13}$）：
| n (binary) | n&1 | result | x |
|-------------|-----|--------|---|
| 1101 | 1 | 2 | 4 |
| 110 | 0 | 2 | 16 |
| 11 | 1 | 32 | 256 |
| 1 | 1 | 8192 | — |

Result = 8192 = $2^{13}$。

### 第2步：矩阵快速幂求斐波那契

```python
A = [[1, 1], [1, 0]]
An = mat_pow(A, n - 1, mod)
F_n = An[0][0]
```

矩阵形式的斐波那契：
$$\begin{bmatrix} F_{n} \\ F_{n-1} \end{bmatrix} = \begin{bmatrix} 1 & 1 \\ 1 & 0 \end{bmatrix}^{n-1} \begin{bmatrix} 1 \\ 0 \end{bmatrix}$$

矩阵快速幂在 $O(\log n)$ 时间内完成，比递推 $O(n)$ 快得多。对于 $n=10^{18}$ 级别的查询，这是标准做法。

### 第3步：欧拉线性筛

```python
for i in range(2, n + 1):
    if is_prime[i]:
        primes.append(i)
    for p in primes:
        if i * p > n: break
        is_prime[i * p] = False
        if i % p == 0: break  # ← 关键行
```

**关键行的含义**：当 `i % p == 0` 时，`p` 是 `i` 的最小质因子。那么 `i * p` 的最小质因子也是 `p`。如果继续用更大的质数 `p_next` 筛 `i * p_next`，那么 `i * p_next` 的最小质因子还是 `p`（因为 `p` 整除 `i`），但 `p_next > p`，这就不是用最小质因子筛了——会造成重复。

**举例**：`i=6, p=2` 时 `6%2==0`，筛掉 12 后 break。如果用 `p=3` 继续筛，`6*3=18` 的最小质因子是 2（18=2*9），用 3 筛是重复的。实际上 18 会在 `i=9, p=2` 时被正确筛掉。

### 第4步：中国剩余定理

$$x = \sum_i a_i \cdot M_i \cdot (M_i^{-1} \bmod m_i) \bmod M$$

其中 $M = \prod m_i, M_i = M / m_i$。

**示例**：$x \equiv 2 \pmod{3}, x \equiv 3 \pmod{5}, x \equiv 2 \pmod{7}$：
- $M = 105$, $M_1=35, M_2=21, M_3=15$
- $35^{-1} \bmod 3 = 2$, $21^{-1} \bmod 5 = 1$, $15^{-1} \bmod 7 = 1$
- $x = 2\cdot 35 \cdot 2 + 3\cdot 21 \cdot 1 + 2\cdot 15 \cdot 1 = 140 + 63 + 30 = 233 \equiv 23 \pmod{105}$

### 第5步：组合数 O(1) 查询

预处理阶乘 `fact[n]` 和逆阶乘 `inv_fact[n]`：

```python
nCr(n, r) = fact[n] * inv_fact[r] % mod * inv_fact[n-r] % mod
```

逆阶乘的计算方法：先求 `inv_fact[n] = mod_inverse_fermat(fact[n], mod)`，然后从后往前递推：`inv_fact[i-1] = inv_fact[i] * i % mod`。

## 关键概念速查表

| 概念 | 公式/方法 | 代码位置 |
|------|----------|---------|
| 快速幂 | 二进制分解 | `fast_pow()` |
| 费马逆元 | $a^{mod-2}$ | `mod_inverse_fermat()` |
| 扩展欧几里得 | $ax+by=\gcd(a,b)$ | `ext_gcd()` |
| 矩阵快速幂 | Fib = [[1,1],[1,0]]^n | `fibonacci_mat()` |
| 埃筛 | $i^2$ 开始标记 | `sieve_eratosthenes()` |
| 线性筛 | 最小质因子 break | `linear_sieve()` |
| CRT | $\sum a_i M_i M_i^{-1}$ | `crt()` |
| 组合数 | $n!/(k!(n-k)!)$ | `nCr()` |
| 错位排列 | $D_n = (n-1)(D_{n-1}+D_{n-2})$ | `derangement()` |

## 完整代码

<<< @/snippets/algo15_number_theory/demo.py
