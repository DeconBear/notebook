# -*- coding: utf-8 -*-
"""
algo15 数论与组合数学 — 演示代码
================================
功能：模逆元（费马+扩展欧几里得）、CRT、快速幂+矩阵快速幂、
      素数筛（埃筛+线筛）、GCD/扩展欧几里得、
      组合数（阶乘+逆元）、容斥原理、期望线性性。

每个函数都有中文 docstring，每行逻辑代码都有中文注释。
运行方式：在 algo15_number_theory/ 目录下执行 python code/demo.py
"""

import os
import math
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False

_HERE = os.path.dirname(os.path.abspath(__file__))
_IMAGES = os.path.join(_HERE, '..', 'images')
os.makedirs(_IMAGES, exist_ok=True)


# ============================================================================
# 第一部分：快速幂与模逆元
# ============================================================================

def fast_pow(x, n, mod):
    """二进制快速幂：计算 (x^n) % mod，O(log n)。"""
    result = 1
    x = x % mod  # 先取模，防止溢出
    while n > 0:
        if n & 1:  # 当前二进制位为 1 → 乘入结果
            result = (result * x) % mod
        x = (x * x) % mod  # 平方
        n >>= 1  # 右移一位
    return result


def mod_inverse_fermat(a, mod):
    """
    用费马小定理求模逆元：a^{-1} ≡ a^{mod-2} (mod mod)。
    要求 mod 为质数，且 gcd(a, mod) = 1。
    """
    return fast_pow(a, mod - 2, mod)


def ext_gcd(a, b):
    """
    扩展欧几里得算法：求解 ax + by = gcd(a, b)。
    返回 (gcd, x, y)。
    """
    if b == 0:
        return a, 1, 0  # gcd(a, 0) = a, ax + 0*y = a → x=1, y=0
    g, x1, y1 = ext_gcd(b, a % b)
    # 回代：x = y1, y = x1 - (a//b) * y1
    x = y1
    y = x1 - (a // b) * y1
    return g, x, y


def mod_inverse_extgcd(a, mod):
    """用扩展欧几里得求模逆元：解 ax ≡ 1 (mod mod)。"""
    g, x, y = ext_gcd(a, mod)
    if g != 1:
        return None  # 逆元不存在
    return (x % mod + mod) % mod  # 确保结果在 [0, mod) 内


# ============================================================================
# 第二部分：矩阵快速幂（斐波那契数列）
# ============================================================================

def mat_mul(A, B, mod):
    """2x2 矩阵乘法。"""
    return [
        [(A[0][0]*B[0][0] + A[0][1]*B[1][0]) % mod,
         (A[0][0]*B[0][1] + A[0][1]*B[1][1]) % mod],
        [(A[1][0]*B[0][0] + A[1][1]*B[1][0]) % mod,
         (A[1][0]*B[0][1] + A[1][1]*B[1][1]) % mod]
    ]


def mat_pow(A, n, mod):
    """矩阵快速幂：计算 A^n % mod。"""
    # 单位矩阵
    result = [[1, 0], [0, 1]]
    base = A
    while n > 0:
        if n & 1:
            result = mat_mul(result, base, mod)
        base = mat_mul(base, base, mod)
        n >>= 1
    return result


def fibonacci_mat(n, mod=10**9+7):
    """
    用矩阵快速幂求斐波那契数列第 n 项。
    Fib = [[1,1],[1,0]]^(n-1) * [F1, F0]
    """
    if n <= 1:
        return n
    A = [[1, 1], [1, 0]]
    An = mat_pow(A, n - 1, mod)
    return An[0][0]  # F_n


# ============================================================================
# 第三部分：素数筛法
# ============================================================================

def sieve_eratosthenes(n):
    """埃拉托色尼筛法：O(n log log n)。"""
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            # 从 i*i 开始，因为更小的倍数已被筛过
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(n + 1) if is_prime[i]]


def linear_sieve(n):
    """
    欧拉线性筛：O(n)，每个合数只被其最小质因子筛一次。
    关键 if i % p == 0: break
    """
    primes = []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False

    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)
        for p in primes:
            if i * p > n:
                break
            is_prime[i * p] = False
            if i % p == 0:  # p 是 i 的最小质因子 → 该 break 了
                break
    return primes


# ============================================================================
# 第四部分：中国剩余定理（CRT）
# ============================================================================

def crt(remainders, moduli):
    """
    中国剩余定理（CRT）：求解 x ≡ r_i (mod m_i)，m_i 两两互质。
    返回最小非负解 x。
    """
    M = 1
    for m in moduli:
        M *= m

    x = 0
    for r, m in zip(remainders, moduli):
        Mi = M // m
        inv = mod_inverse_extgcd(Mi, m)  # Mi 在模 m 下的逆元
        if inv is None:
            return None
        x = (x + r * Mi * inv) % M

    return x


# ============================================================================
# 第五部分：组合数
# ============================================================================

def precompute_factorials(n, mod):
    """预处理阶乘和逆阶乘，用于 O(1) 组合数查询。mod 需为质数。"""
    fact = [1] * (n + 1)
    inv_fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = fact[i - 1] * i % mod
    inv_fact[n] = mod_inverse_fermat(fact[n], mod)
    for i in range(n - 1, -1, -1):
        inv_fact[i] = inv_fact[i + 1] * (i + 1) % mod
    return fact, inv_fact


def nCr(n, r, fact, inv_fact, mod):
    """组合数 C(n, r) % mod。"""
    if r < 0 or r > n:
        return 0
    return fact[n] * inv_fact[r] % mod * inv_fact[n - r] % mod


def nPr(n, r, fact, inv_fact, mod):
    """排列数 P(n, r) % mod。"""
    if r < 0 or r > n:
        return 0
    return fact[n] * inv_fact[n - r] % mod


def precompute_comb_pascal(n, mod):
    """
    用 Pascal 三角形递推组合数 C[n][k]。
    C[n][k] = C[n-1][k-1] + C[n-1][k]
    适用于 n <= 2000。
    """
    C = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        C[i][0] = C[i][i] = 1
        for j in range(1, i):
            C[i][j] = (C[i - 1][j - 1] + C[i - 1][j]) % mod
    return C


# ============================================================================
# 第六部分：容斥原理 — 错位排列
# ============================================================================

def derangement(n, mod=None):
    """
    错位排列数 D_n：n 个元素没有元素在原位的排列数。
    递推：D_0=1, D_1=0, D_n=(n-1)*(D_{n-1}+D_{n-2})
    """
    if n == 0:
        return 1
    if n == 1:
        return 0
    d0, d1 = 1, 0  # D_0, D_1
    for i in range(2, n + 1):
        d2 = (i - 1) * (d1 + d0)
        if mod:
            d2 %= mod
        d0, d1 = d1, d2
    return d1


def inclusion_exclusion_count_coprimes(n, primes):
    """
    容斥原理：计算 1~n 中至少被给定素数列表中某个素数整除的数的个数。
    例：primes=[2,3,5] → 被 2 或 3 或 5 整除的数的个数。
    """
    k = len(primes)
    total = 0
    for mask in range(1, 1 << k):
        prod = 1
        bits = 0
        for i in range(k):
            if mask & (1 << i):
                prod *= primes[i]
                bits += 1
        count = n // prod
        if bits % 2 == 1:
            total += count   # 奇数个集合 → 加
        else:
            total -= count   # 偶数个集合 → 减
    return total


# ============================================================================
# 第七部分：可视化筛法对比
# ============================================================================

def visualize_sieve_comparison(n=100):
    """可视化埃筛和线筛的对比（素数的分布）。"""
    erat_primes = set(sieve_eratosthenes(n))
    linear_primes = set(linear_sieve(n))

    fig, ax = plt.subplots(figsize=(14, 3))

    # 显示 1~n 的数，素数高亮
    for i in range(1, n + 1):
        color = '#4CAF50' if i in erat_primes else '#E0E0E0'
        ax.bar(i, 1, color=color, edgecolor='none', width=0.8)

    ax.set_xlim(0, n + 1)
    ax.set_ylim(0, 1.5)
    ax.set_xlabel('Number', fontsize=12)
    ax.set_yticks([])
    ax.set_title(f'Primes up to {n} (total: {len(erat_primes)})',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(_IMAGES, 'algo15_sieve_primes.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[可视化] 素数分布图已保存: {path}')


# ============================================================================
# 主程序
# ============================================================================

def main():
    print('=' * 60)
    print('algo15 数论与组合数学 — 演示')
    print('=' * 60)

    MOD = 10**9 + 7

    # --- 1. 快速幂 ---
    print('\n### 1. 快速幂 ###')
    print(f'  3^13 = {3**13}')
    print(f'  快速幂 3^13 mod {MOD} = {fast_pow(3, 13, MOD)}')
    print(f'  快速幂 2^64 mod {MOD} = {fast_pow(2, 64, MOD)}')

    # --- 2. 模逆元 ---
    print('\n### 2. 模逆元 ###')
    a = 7
    inv_f = mod_inverse_fermat(a, MOD)
    inv_e = mod_inverse_extgcd(a, MOD)
    print(f'  {a}^-1 mod {MOD}: 费马={inv_f}, 扩展GCD={inv_e}')
    print(f'  验证: {a} * {inv_f} mod {MOD} = {(a * inv_f) % MOD}')

    # --- 3. 矩阵快速幂 — 斐波那契 ---
    print('\n### 3. 矩阵快速幂 — 斐波那契数列 ###')
    for n in [0, 1, 5, 10, 20, 50]:
        print(f'  Fib({n}) = {fibonacci_mat(n)}')

    # --- 4. 素数筛 ---
    print('\n### 4. 素数筛法 ###')
    n_test = 100
    erat = sieve_eratosthenes(n_test)
    linear = linear_sieve(n_test)
    print(f'  1-{n_test} 内素数数量: 埃筛={len(erat)}, 线筛={len(linear)}')
    print(f'  前 10 个素数: {erat[:10]}')
    visualize_sieve_comparison(n_test)

    # --- 5. GCD & 扩展欧几里得 ---
    print('\n### 5. GCD & 扩展欧几里得 ###')
    a, b = 30, 18
    g, x, y = ext_gcd(a, b)
    print(f'  gcd({a}, {b}) = {g}, 解: {a}*({x}) + {b}*({y}) = {a*x+b*y}')

    # --- 6. 中国剩余定理 ---
    print('\n### 6. 中国剩余定理 ###')
    remainders = [2, 3, 2]
    moduli = [3, 5, 7]
    crt_result = crt(remainders, moduli)
    print(f'  x ≡ {remainders[0]} (mod {moduli[0]}), ≡ {remainders[1]} (mod {moduli[1]}), ≡ {remainders[2]} (mod {moduli[2]})')
    print(f'  解: x = {crt_result}')
    for r, m in zip(remainders, moduli):
        print(f'  验证: {crt_result} % {m} = {crt_result % m} (期望 {r})')

    # --- 7. 组合数 ---
    print('\n### 7. 组合数 ###')
    fact, inv_fact = precompute_factorials(20, MOD)
    for n in [5, 10]:
        for k in [0, 2, n]:
            c = nCr(n, k, fact, inv_fact, MOD)
            print(f'  C({n}, {k}) = {c}')

    # --- 8. 容斥原理 — 错位排列 ---
    print('\n### 8. 容斥原理 & 错位排列 ###')
    for n in range(6):
        print(f'  D_{n} = {derangement(n)}')
    # 容斥：1~100 中能被 2,3,5 整除的数
    cnt = inclusion_exclusion_count_coprimes(100, [2, 3, 5])
    print(f'  1~100 中能被 2/3/5 整除的数: {cnt}')

    print('\n' + '=' * 60)
    print('总结：数论工具箱 — 模运算/快速幂/筛法/GCD/组合数/CRT/容斥')
    print('=' * 60)


if __name__ == '__main__':
    main()
