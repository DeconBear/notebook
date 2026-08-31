# -*- coding: utf-8 -*-
"""
s06 反向传播与链式法则 — 练习代码
================================
请完成以下 TODO 任务，加深对反向传播和自动微分的理解。

每个 TODO 都有详细的中文指示和预期输出描述。
建议先阅读 README.md 并运行 demo.py，再尝试独立补全代码。
"""

import sys
import os
# 导入 demo.py 中的 Value 类
sys.path.insert(0, os.path.dirname(__file__))


# 将 demo.py 中的 Value 类复制到此处，便于独立练习
# 如果你已经运行过 demo.py，可以直接 from demo import Value
import math
from typing import Tuple, Set, List


# ============================================================================
# 复制 Value 类基础（包含 __init__, __add__, __mul__, __pow__, 辅助方法）
# 你需要在下面补全 tanh, truediv 的 backward 以及 backward() 方法
# ============================================================================

class Value:
    """自动微分引擎的基本节点（练习版本）"""

    def __init__(self, data: float, _children: Tuple = (), _op: str = ""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "仅支持数值指数"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def relu(self):
        out = Value(max(0.0, self.data), (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        x = self.data
        if x >= 0:
            s = 1.0 / (1.0 + math.exp(-x))
        else:
            exp_x = math.exp(x)
            s = exp_x / (1.0 + exp_x)
        out = Value(s, (self,), 'Sigmoid')

        def _backward():
            self.grad += out.data * (1 - out.data) * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return other + (-self)

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # ================================================================
    # TODO 1: 实现 tanh 激活函数的反向传播
    # ================================================================
    def tanh(self):
        """
        Tanh 激活函数：tanh(x) = (e^x - e^{-x}) / (e^x + e^{-x})

        前向传播：计算 tanh(self.data) 并创建新的 Value 节点
        反向传播：tanh 的导数是 1 - tanh^2(x)
                  即 self.grad += (1 - out.data^2) * out.grad

        提示:
          1. 使用 math.tanh 计算前向值
          2. 导数公式：d(tanh(x))/dx = 1 - tanh^2(x) = 1 - out.data^2
          3. 参考上面 sigmoid 的实现方式
        """
        # TODO: 实现前向传播
        t = None  # ← TODO: 计算 math.tanh(self.data)

        # TODO: 创建输出节点
        out = None  # ← TODO: Value(t, (self,), 'Tanh')

        # TODO: 定义反向传播闭包
        def _backward():
            pass  # ← TODO: self.grad += (1 - out.data ** 2) * out.grad

        out._backward = _backward
        return out

    # ================================================================
    # TODO 2: 实现除法的反向传播
    # ================================================================
    def __truediv__(self, other):
        """
        除法操作: self / other

        可以分解为: self * (other ** -1)
        即 a / b = a * b^{-1}

        局部导数:
          ∂(a/b)/∂a = 1/b
          ∂(a/b)/∂b = -a / b^2

        提示:
          1. 可以利用现有的 __mul__ 和 __pow__ 来实现
          2. return self * (other ** -1)
          3. 这个一行就可以实现！因为 ** -1 和 * 已经定义了 backward
        """
        # TODO: 利用已有的运算实现除法
        return None  # ← TODO: return self * other ** -1

    def __rtruediv__(self, other):
        """右侧除法：other / self"""
        return None  # ← TODO: return other * self ** -1

    # ================================================================
    # TODO 3: 实现 backward() 方法（拓扑排序 + 逆序遍历）
    # ================================================================
    def backward(self):
        """
        执行反向传播。

        算法步骤：
          1. 拓扑排序：DFS 遍历计算图，记录节点顺序
          2. 设置根节点梯度为 1.0 (∂L/∂L = 1)
          3. 按拓扑逆序依次调用每个节点的 _backward()

        提示:
          - 使用 visited 集合避免重复访问
          - build_topo 函数做后序遍历（子节点先入列表）
          - 逆序遍历: for node in reversed(topo)
        """
        # TODO: 拓扑排序
        topo = []        # ← TODO: 存放拓扑排序结果
        visited = set()  # ← TODO: 记录已访问节点

        def build_topo(v):
            """
            DFS 后序遍历，构建拓扑排序。
            子节点先入列表，父节点后入。
            """
            pass  # ← TODO: 实现 build_topo
            # 提示:
            # if v not in visited:
            #     visited.add(v)
            #     for child in v._prev:
            #         build_topo(child)
            #     topo.append(v)

        # TODO: 调用 build_topo(self)
        build_topo(None)  # ← TODO: 改为 build_topo(self)

        # TODO: 设置根节点梯度
        # self.grad = 1.0

        # TODO: 按拓扑逆序调用 _backward
        # for node in reversed(topo):
        #     node._backward()

    def zero_grad(self):
        """梯度清零"""
        visited = set()

        def _zero(v):
            if v not in visited:
                visited.add(v)
                v.grad = 0.0
                for child in v._prev:
                    _zero(child)

        _zero(self)


# ============================================================================
# TODO 3(续): 使用自动微分求 f(x) = x² + 3x 的最小值
# ============================================================================
def find_minimum():
    """
    使用自动微分和梯度下降，找到 f(x) = x² + 3x 的最小值。

    解析解: f'(x) = 2x + 3 = 0 → x = -1.5
    最小值: f(-1.5) = -2.25

    任务:
      1. 初始化 x = Value(5.0)
      2. 循环 30 步:
         a. 构造 loss = x² + 3x（使用 Value 的运算）
         b. 清零梯度
         c. 反向传播
         d. 更新 x.data -= 0.1 * x.grad
      3. 打印最终结果

    提示:
      - x² 可以写成 x * x 或 x ** 2
      - 3x 可以写成 x * Value(3.0) 或 Value(3.0) * x
      - 每次迭代前记得 x.zero_grad()
    """
    print("=" * 60)
    print("TODO 3: 梯度下降求 f(x)=x²+3x 最小值")
    print("=" * 60)

    # TODO: 初始化 x
    x = None  # ← TODO: x = Value(5.0)

    if x is None:
        print("  TODO 未完成，请补全 find_minimum 函数")
        return

    learning_rate = 0.1
    n_steps = 30

    print(f"\n  初始 x = {x.data:.1f}")
    print(f"  理论最优: x = -1.5, f(x) = -2.25")

    for step in range(n_steps):
        # TODO: 构造损失 f(x) = x² + 3x
        loss = None  # ← TODO: x ** 2 + Value(3.0) * x  或者  x * x + x * Value(3.0)

        # TODO: 清零梯度
        # x.zero_grad()

        # TODO: 反向传播
        # loss.backward()

        # TODO: 梯度下降更新
        # x.data -= learning_rate * x.grad

        if step % 10 == 0 or step == n_steps - 1:
            print(f"    Step {step:2d}: x = {x.data:.4f}, f(x) = {loss.data:.4f}, grad = {x.grad:.4f}")

    print(f"\n  最终: x = {x.data:.4f} (理论: -1.5), f(x) = {x.data**2 + 3*x.data:.4f} (理论: -2.25)")


# ============================================================================
# 测试函数
# ============================================================================

def test_tanh():
    """测试 tanh 反向传播"""
    print("=" * 60)
    print("TODO 1 测试: tanh 反向传播")
    print("=" * 60)

    x = Value(0.5)
    y = x.tanh()

    if y is None:
        print("  TODO 未完成，请补全 tanh 方法")
        return

    y.backward()
    expected = 1 - y.data ** 2  # tanh 导数公式
    print(f"  tanh(0.5) = {y.data:.6f}")
    print(f"  ∂tanh/∂x = {x.grad:.6f}")
    print(f"  预期值   = {expected:.6f}")
    print(f"  匹配: {abs(x.grad - expected) < 1e-6}")

    # 测试负值
    x2 = Value(-1.0)
    y2 = x2.tanh()
    y2.backward()
    expected2 = 1 - y2.data ** 2
    print(f"\n  tanh(-1.0) = {y2.data:.6f}")
    print(f"  ∂tanh/∂x = {x2.grad:.6f}")
    print(f"  预期值   = {expected2:.6f}")
    print(f"  匹配: {abs(x2.grad - expected2) < 1e-6}")
    print()


def test_division():
    """测试除法反向传播"""
    print("=" * 60)
    print("TODO 2 测试: 除法反向传播")
    print("=" * 60)

    a = Value(6.0)
    b = Value(2.0)
    c = a / b  # c = 6/2 = 3

    if c is None:
        print("  TODO 未完成，请补全 __truediv__ 方法")
        return

    c.backward()
    # ∂(a/b)/∂a = 1/b = 1/2 = 0.5
    # ∂(a/b)/∂b = -a/b² = -6/4 = -1.5
    expected_da = 1.0 / b.data
    expected_db = -a.data / (b.data ** 2)
    print(f"  {a.data}/{b.data} = {c.data}")
    print(f"  ∂c/∂a = {a.grad:.4f} (预期: {expected_da:.4f})")
    print(f"  ∂c/∂b = {b.grad:.4f} (预期: {expected_db:.4f})")
    print(f"  ∂c/∂a 匹配: {abs(a.grad - expected_da) < 1e-6}")
    print(f"  ∂c/∂b 匹配: {abs(b.grad - expected_db) < 1e-6}")

    # 测试右侧除法
    a2 = Value(10.0)
    b2 = Value(3.0)
    c2 = a2 / b2  # 10/3

    rt_div = (Value(5.0) / a2) if hasattr(a2, '__rtruediv__') else None
    print(f"\n  右侧除法 5/{a2.data}: ", end="")
    if rt_div is not None:
        print(f"{rt_div.data:.4f}")
    else:
        print("TODO 未完成")
    print()


def test_minimum_search():
    """测试梯度下降求最小值"""
    find_minimum()
    print()


# ============================================================================
# 主程序
# ============================================================================
if __name__ == "__main__":
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║   s06 反向传播与链式法则 — 动手练习                        ║")
    print("║   请依次完成 TODO 1, 2, 3                                   ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    test_tanh()
    test_division()
    test_minimum_search()

    print("=" * 60)
    print("所有测试完成！请检查输出结果。")
    print("如有未通过的测试，请回到对应的 TODO 部分补全代码。")
    print()
    print("提示: 完成后的 tanh, truediv 和 backward 方法应该与")
    print("      demo.py 中的实现产生相同的结果。")
    print("=" * 60)
