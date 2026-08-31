# -*- coding: utf-8 -*-
"""
===============================================================================
as01_ai4s_overview/code/exercise.py — AI4S 全景练习
===============================================================================
本练习文件中，你需要完成以下任务：

练习目标：
  1. 实现二阶中心差分近似二阶导数（PDE 残差计算的基础）
  2. 实现 PDE 残差函数，并用它给"候选解"打分
  3. 实现一个"制造解"（method of manufactured solutions）生成器（Bonus）

提示：
  - 二阶中心差分: u''(x_i) ≈ (u_{i-1} - 2*u_i + u_{i+1}) / dx^2
  - PDE 残差: r(x) = -u''(x) - f(x)，衡量候选解违反方程的程度
  - 制造解: 先选定 u(x)，再反推 f(x) = -u''(x)，这样就有了精确的验证基准

运行方式：
  python exercise.py
===============================================================================
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


# ============================================================================
# 辅助函数（已实现，可直接使用）
# ============================================================================

def u_true(x):
    """真解: u(x) = sin(pi * x)"""
    return np.sin(np.pi * x)


def f_true(x):
    """对应的真实源项: f(x) = pi^2 * sin(pi * x)"""
    return (np.pi ** 2) * np.sin(np.pi * x)


# ============================================================================
# 任务 1: 实现二阶中心差分 (约 5 行)
# ============================================================================

def second_derivative_fd(u_vals, dx):
    """
    用二阶中心差分近似二阶导数。

    公式: u''(x_i) ≈ (u_{i-1} - 2*u_i + u_{i+1}) / dx^2

    参数:
        u_vals: (n,) 数组，函数在等距网格点上的取值
        dx: float，网格间距
    返回:
        d2u: (n,) 数组，内部点为二阶导数近似值，两端边界点填 0
             （边界点没有左右邻居，这里简化处理为 0，不影响残差比较的结论）
    """
    n = len(u_vals)
    d2u = np.zeros(n)
    # TODO: 完成以下步骤
    # 1. 对内部点 i=1..n-2 (对应数组切片 [1:-1])，计算:
    #    d2u[1:-1] = (u_vals[:-2] - 2*u_vals[1:-1] + u_vals[2:]) / dx^2
    # --- BEGIN YOUR CODE ---
    d2u[1:-1] = (u_vals[:-2] - 2 * u_vals[1:-1] + u_vals[2:]) / (dx ** 2)
    # --- END YOUR CODE ---
    # ^^^ TODO: 请删除上面的实现代码，自己重写 ^^^
    return d2u


# ============================================================================
# 任务 2: 实现 PDE 残差函数 (约 5 行)
# ============================================================================

def pde_residual(u_vals, x_grid, f_func):
    """
    计算一维 Poisson 方程 -u''(x) = f(x) 的残差:
        r(x) = -u''(x) - f(x)

    参数:
        u_vals: (n,) 候选解在网格点上的取值
        x_grid: (n,) 网格坐标
        f_func: 可调用对象，f_func(x) 返回源项 f(x)
    返回:
        r: (n,) 残差数组（边界两个点因为二阶差分不准确，可以忽略）
    """
    # TODO: 完成以下步骤
    # 1. 计算网格间距 dx = x_grid[1] - x_grid[0]
    # 2. 调用 second_derivative_fd 得到 d2u
    # 3. 计算 f_vals = f_func(x_grid)
    # 4. 返回 r = -d2u - f_vals
    # --- BEGIN YOUR CODE ---
    dx = x_grid[1] - x_grid[0]
    d2u = second_derivative_fd(u_vals, dx)
    f_vals = f_func(x_grid)
    r = -d2u - f_vals
    # --- END YOUR CODE ---
    # ^^^ TODO: 请删除上面的实现代码，自己重写 ^^^
    return r


# ============================================================================
# 任务 3 (Bonus): 实现"制造解"生成器 (约 10 行)
# ============================================================================

def manufacture_solution(k, n=201):
    """
    制造解方法: 给定一个正整数 k，构造真解 u(x) = sin(k * pi * x)，
    并解析地推导出对应的源项 f(x) = -u''(x) = (k*pi)^2 * sin(k*pi*x)。

    这是验证任何数值 PDE 求解器（包括 as02 的 PINN）正确性的标准技巧：
    因为我们知道精确解，可以直接计算误差，而不需要"真实世界"的标注数据。

    参数:
        k: int，正弦函数的模式数（频率）
        n: int，网格点数量
    返回:
        x: (n,) 网格坐标，范围 [0, 1]
        u: (n,) 真解取值
        f: (n,) 对应源项取值
    """
    # TODO: 完成以下步骤
    # 1. 用 np.linspace(0, 1, n) 生成网格 x
    # 2. u = sin(k * pi * x)
    # 3. f = (k * pi)^2 * sin(k * pi * x)   [ 因为 u'' = -(k*pi)^2 * sin(k*pi*x) ]
    # --- BEGIN YOUR CODE ---
    x = np.linspace(0, 1, n)
    u = np.sin(k * np.pi * x)
    f = (k * np.pi) ** 2 * np.sin(k * np.pi * x)
    # --- END YOUR CODE ---
    # ^^^ TODO: 请删除上面的实现代码，自己重写 ^^^
    return x, u, f


# ============================================================================
# 验证代码
# ============================================================================

def test_second_derivative():
    """验证二阶差分: 对 u=x^2, u''=2 (常数)。"""
    print("[测试 1] 二阶中心差分...")
    x = np.linspace(0, 1, 101)
    dx = x[1] - x[0]
    u = x ** 2
    d2u = second_derivative_fd(u, dx)
    # 内部点应接近 2.0
    assert np.allclose(d2u[5:-5], 2.0, atol=1e-6), f"二阶差分不正确: {d2u[50]}"
    print(f"  [PASS] 二阶差分正确! u''(x)≈{d2u[50]:.6f} (期望 2.0)")


def test_pde_residual():
    """验证 PDE 残差: 真解的残差应接近 0，错解的残差应明显偏离 0。"""
    print("[测试 2] PDE 残差...")
    x = np.linspace(0, 1, 201)
    u = u_true(x)
    r = pde_residual(u, x, f_true)
    rms = np.sqrt(np.mean(r[2:-2] ** 2))
    assert rms < 1e-2, f"真解残差应接近 0，实际 RMS={rms}"
    print(f"  [PASS] 真解残差 RMS = {rms:.6f} (接近 0)")

    # 错误解: 振幅减半
    u_wrong = 0.5 * u_true(x)
    r_wrong = pde_residual(u_wrong, x, f_true)
    rms_wrong = np.sqrt(np.mean(r_wrong[2:-2] ** 2))
    assert rms_wrong > rms * 10, "错解的残差应明显大于真解"
    print(f"  [PASS] 错解残差 RMS = {rms_wrong:.6f} (明显更大)")


def test_manufacture_solution():
    """验证制造解: 不同 k 值下 u, f 应满足解析关系，且残差应接近 0。"""
    print("[测试 3 (Bonus)] 制造解生成器...")
    for k in [1, 2, 3]:
        x, u, f = manufacture_solution(k, n=301)
        r = pde_residual(u, x, lambda xx: (k * np.pi) ** 2 * np.sin(k * np.pi * xx))
        rms = np.sqrt(np.mean(r[2:-2] ** 2))
        assert rms < 0.5, f"k={k} 时残差过大: {rms}"
        print(f"  [PASS] k={k}: 残差 RMS = {rms:.6f}")


def visualize_manufactured_solutions():
    """额外可视化: 画出 k=1,2,3 三个制造解，帮助理解频率对曲率的影响。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    for k in [1, 2, 3]:
        x, u, _ = manufacture_solution(k)
        ax.plot(x, u, linewidth=2, label=f'k={k}: sin({k}πx)')
    ax.set_xlabel('x')
    ax.set_ylabel('u(x)')
    ax.set_title('制造解示例: 不同频率 k 对应不同曲率')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('manufactured_solutions_preview.png', dpi=120)
    plt.close(fig)
    print("  (可选) 预览图已保存: manufactured_solutions_preview.png")


if __name__ == "__main__":
    print("=" * 60)
    print("as01_ai4s_overview exercise.py — AI4S 全景练习")
    print("=" * 60)

    try:
        test_second_derivative()
        test_pde_residual()
        test_manufacture_solution()
        visualize_manufactured_solutions()
        print("\n" + "=" * 60)
        print("所有测试通过!")
        print("=" * 60)
    except Exception as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
