# -*- coding: utf-8 -*-
"""
s06 反向传播与链式法则 — 演示代码
==================================
功能：从零实现一个微型自动微分引擎（mini autograd），
      模拟 PyTorch 的 .backward() 核心逻辑。

每个 Value 节点记录：
  - data: 该节点的数值
  - grad: 该节点累积的梯度
  - _backward: 该节点的局部反向传播函数
  - _prev: 该节点的前驱节点（用于拓扑排序）

运行方式：在 s06_backprop_chain_rule/ 目录下执行 python code/demo.py
"""

import math
from typing import Set, List, Tuple


# ============================================================================
# 第一部分：Value 类 —— 自动微分的核心
# ============================================================================

class Value:
    """
    自动微分引擎中的基本计算节点。

    每个 Value 对象代表计算图中的一个节点，它存储：
    - data: 数值（标量）
    - grad: 累积的梯度（默认初始化为 0）
    - _backward: 局部反向传播函数（闭包）
    - _prev: 该节点的直接前驱节点集合

    灵感来源：Andrej Karpathy 的 micrograd 项目
    """

    def __init__(self, data: float, _children: Tuple = (), _op: str = ""):
        """
        初始化一个 Value 节点。

        参数:
            data: 节点的数值
            _children: 前驱节点元组（用于构建计算图）
            _op: 产生此节点的操作名称（如 "+", "*", "ReLU" 等）
        """
        self.data = data                # 存储数值
        self.grad = 0.0                 # 梯度初始化为 0（反向传播时累加）
        self._backward = lambda: None   # 默认的反向传播函数（叶子节点无需操作）
        self._prev = set(_children)     # 前驱节点集合（用于拓扑排序）
        self._op = _op                  # 操作名称（用于可视化）

    # ---- 基本算术运算 ----

    def __add__(self, other):
        """
        加法操作: self + other

        局部梯度: ∂(self+other)/∂self = 1, ∂(self+other)/∂other = 1
        反向传播: 上游梯度原样传递给两个输入（加法门）
        """
        # 如果 other 不是 Value，先转换为 Value（支持 Value + 数值）
        other = other if isinstance(other, Value) else Value(other)
        # 创建结果节点
        out = Value(self.data + other.data, (self, other), '+')

        # 定义加法门的局部反向传播
        def _backward():
            # 加法门：梯度原样传递
            self.grad += 1.0 * out.grad   # ∂out/∂self = 1
            other.grad += 1.0 * out.grad  # ∂out/∂other = 1

        out._backward = _backward  # 绑定反向传播函数
        return out

    def __mul__(self, other):
        """
        乘法操作: self * other

        局部梯度: ∂(self*other)/∂self = other, ∂(self*other)/∂other = self
        反向传播: 梯度交换（乘以对方的 data 值）
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # 乘法门：梯度交换（gradient switcheroo）
            self.grad += other.data * out.grad   # ∂out/∂self = other.data
            other.grad += self.data * out.grad   # ∂out/∂other = self.data

        out._backward = _backward
        return out

    def __pow__(self, other):
        """
        幂运算: self ** other (仅支持整数幂)

        局部梯度: ∂(x^n)/∂x = n * x^(n-1)
        """
        assert isinstance(other, (int, float)), "仅支持数值指数"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            # 幂函数的求导: d(x^n)/dx = n * x^(n-1)
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        """取负操作: -self 等价于 self * (-1)"""
        return self * -1

    def __sub__(self, other):
        """减法操作: self - other 等价于 self + (-other)"""
        return self + (-other)

    def __truediv__(self, other):
        """
        除法操作: self / other

        利用了: a / b = a * (b ** -1)
        局部梯度:
          ∂(a/b)/∂a = 1/b
          ∂(a/b)/∂b = -a / b^2
        """
        return self * other ** -1

    def __radd__(self, other):
        """右侧加法：other + self"""
        return self + other

    def __rmul__(self, other):
        """右侧乘法：other * self"""
        return self * other

    def __rsub__(self, other):
        """右侧减法：other - self"""
        return other + (-self)

    def __rtruediv__(self, other):
        """右侧除法：other / self"""
        return other * self ** -1

    # ---- 激活函数 ----

    def relu(self):
        """
        ReLU 激活函数: max(0, self)

        局部梯度: 1 if self.data > 0 else 0
        反向传播: 梯度门控——正区间通过，负区间截断为 0
        """
        out = Value(max(0.0, self.data), (self,), 'ReLU')

        def _backward():
            # ReLU 的局部导数：正区间为 1，负区间为 0
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        """
        Sigmoid 激活函数: 1 / (1 + e^{-x})

        局部梯度: σ(x) * (1 - σ(x))
        反向传播: 可以利用前向输出直接计算导数
        """
        # 数值稳定的 sigmoid 实现
        x = self.data
        # 对于非常大的负值，exp(-x) 会溢出，因此需要特殊情况处理
        if x >= 0:
            s = 1.0 / (1.0 + math.exp(-x))
        else:
            exp_x = math.exp(x)
            s = exp_x / (1.0 + exp_x)

        out = Value(s, (self,), 'Sigmoid')

        def _backward():
            # sigmoid 导数: σ(x)(1 - σ(x)) = out.data * (1 - out.data)
            self.grad += out.data * (1 - out.data) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        """
        Tanh 激活函数: (e^x - e^{-x}) / (e^x + e^{-x})

        局部梯度: 1 - tanh^2(x)
        反向传播: 利用前向输出直接计算导数
        """
        t = math.tanh(self.data)
        out = Value(t, (self,), 'Tanh')

        def _backward():
            # tanh 导数: 1 - tanh^2(x) = 1 - out.data^2
            self.grad += (1 - out.data ** 2) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        """
        指数函数: e^{self}

        局部梯度: e^{self} = out.data
        反向传播: 梯度 = 节点自身的值
        """
        out = Value(math.exp(self.data), (self,), 'exp')

        def _backward():
            # exp 的导数就是它自己: d(e^x)/dx = e^x
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    # ---- 反向传播主函数 ----

    def backward(self):
        """
        执行反向传播：从当前节点（通常是损失）出发，
        按拓扑逆序遍历计算图，依次调用每个节点的 _backward()。

        算法步骤：
          1. 拓扑排序：找到从根节点到当前节点的所有路径上的节点
          2. 将当前节点的梯度设为 1.0（因为 ∂L/∂L = 1）
          3. 按拓扑逆序调用每个节点的 _backward()
        """
        # ---- 步骤 1: 拓扑排序（DFS 实现） ----
        topo = []      # 存储拓扑排序结果
        visited = set()  # 记录已访问节点

        def build_topo(v: Value):
            """深度优先搜索，构建拓扑排序"""
            if v not in visited:
                visited.add(v)              # 标记为已访问
                for child in v._prev:       # 递归访问所有前驱节点
                    build_topo(child)
                topo.append(v)              # 后序遍历：子节点在前，父节点在后

        build_topo(self)  # 从当前节点开始拓扑排序

        # ---- 步骤 2: 初始化梯度 ----
        self.grad = 1.0  # ∂L/∂L = 1（损失对自身的梯度恒为 1）

        # ---- 步骤 3: 按拓扑逆序调用 backward ----
        for node in reversed(topo):  # 逆序遍历拓扑排序（从输出到输入）
            node._backward()         # 调用该节点的局部反向传播

    # ---- 工具方法 ----

    def zero_grad(self):
        """将所有节点的梯度清零（每次反向传播前调用）"""
        # 遍历计算图中的所有节点
        visited = set()

        def _zero(v):
            if v not in visited:
                visited.add(v)
                v.grad = 0.0  # 梯度清零
                for child in v._prev:
                    _zero(child)

        _zero(self)

    def __repr__(self):
        """格式化输出：显示数值和梯度"""
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"


# ============================================================================
# 第二部分：计算图可视化
# ============================================================================

def trace_graph(root: Value) -> Tuple[List[Value], Set[Tuple[Value, Value]]]:
    """
    从根节点出发，追踪整个计算图的所有节点和边。

    参数:
        root: 计算图的根节点（通常是输出/损失）

    返回:
        nodes: 所有节点的列表
        edges: 边的集合，每条边是 (parent, child) 元组
    """
    nodes = []       # 存储所有节点
    edges = set()    # 存储所有边
    visited = set()  # 记录已访问节点

    def build(v: Value):
        if v not in visited:
            visited.add(v)
            nodes.append(v)         # 记录节点
            for child in v._prev:   # 遍历所有子节点
                edges.add((child, v))  # 记录边：child → v
                build(child)

    build(root)
    return nodes, edges


def print_computation_graph(root: Value):
    """
    以文本形式打印计算图的结构和每个节点的梯度信息。

    参数:
        root: 计算图的根节点
    """
    nodes, edges = trace_graph(root)

    print("\n" + "=" * 70)
    print("【计算图结构】")
    print("=" * 70)

    # 按拓扑序打印（叶子节点在前）
    # 计算每个节点的"深度"（从叶子节点起的最长路径）
    depth = {}

    def compute_depth(v: Value) -> int:
        """递归计算节点的深度"""
        if v not in depth:
            if len(v._prev) == 0:
                depth[v] = 0  # 叶子节点深度为 0
            else:
                depth[v] = 1 + max(compute_depth(c) for c in v._prev)
        return depth[v]

    for node in nodes:
        compute_depth(node)

    # 按深度排序打印
    sorted_nodes = sorted(nodes, key=lambda v: depth.get(v, 0))

    print(f"\n{'深度':<6} {'操作':<10} {'数据值':<18} {'梯度':<18} {'输入'}")
    print("-" * 80)

    for node in sorted_nodes:
        d = depth.get(node, 0)
        op = node._op if node._op else "input"
        inputs_str = ", ".join([f"id={id(c) % 1000:03d}" for c in node._prev])
        if not node._prev:
            inputs_str = "(叶子节点)"
        print(f"{d:<6} {op:<10} {node.data:<18.6f} {node.grad:<18.8f} {inputs_str}")

    print("-" * 80)
    print(f"总节点数: {len(nodes)}, 总边数: {len(edges)}")
    print("=" * 70)


# ============================================================================
# 第三部分：神经网络模块
# ============================================================================

class Neuron:
    """
    单个神经元：w·x + b，后接激活函数

    参数:
        nin: 输入维度（特征数量）
        activation: 激活函数类型 ("relu", "sigmoid", "tanh", "linear")
    """

    def __init__(self, nin: int, activation: str = "relu"):
        import random
        # He 初始化：权重从 N(0, sqrt(2/nin)) 采样
        self.w = [Value(random.uniform(-1, 1) * math.sqrt(2.0 / nin)) for _ in range(nin)]
        self.b = Value(0.0)  # 偏置初始化为 0
        self.activation = activation

    def __call__(self, x: List[Value]) -> Value:
        """
        前向传播: activation(w·x + b)

        参数:
            x: 输入列表，长度必须等于 nin

        返回:
            神经元的输出 Value 节点
        """
        # 计算加权和 w·x + b
        # sum(w_i * x_i for each i) + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)

        # 应用激活函数
        if self.activation == "relu":
            out = act.relu()
        elif self.activation == "sigmoid":
            out = act.sigmoid()
        elif self.activation == "tanh":
            out = act.tanh()
        elif self.activation == "linear":
            out = act
        else:
            raise ValueError(f"不支持的激活函数: {self.activation}")

        return out

    def parameters(self) -> List[Value]:
        """返回该神经元的所有参数"""
        return self.w + [self.b]


class Layer:
    """
    神经网络中的一层：包含多个神经元

    参数:
        nin: 输入维度
        nout: 输出维度（该层的神经元数量）
        activation: 激活函数类型
    """

    def __init__(self, nin: int, nout: int, activation: str = "relu"):
        self.neurons = [Neuron(nin, activation) for _ in range(nout)]

    def __call__(self, x: List[Value]) -> List[Value]:
        """
        前向传播：对输入 x，每个神经元独立计算，输出列表

        参数:
            x: 输入列表

        返回:
            该层所有神经元的输出列表
        """
        return [n(x) for n in self.neurons]

    def parameters(self) -> List[Value]:
        """返回该层的所有参数"""
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params


class MLP:
    """
    多层感知机：按顺序堆叠多个 Layer

    参数:
        nin: 输入维度
        nouts: 每层输出维度的列表，如 [4, 4, 1] 表示两隐藏层+输出层
        activations: 每层激活函数的列表
    """

    def __init__(self, nin: int, nouts: List[int], activations: List[str]):
        # 构建层列表
        sizes = [nin] + nouts  # 如 [3, 4, 4, 1]
        self.layers = []
        for i in range(len(nouts)):
            self.layers.append(Layer(sizes[i], sizes[i + 1], activations[i]))

    def __call__(self, x: List[Value]) -> Value:
        """
        前向传播：数据依次经过每一层

        参数:
            x: 输入列表

        返回:
            最终输出节点（最后一层第一个神经元的输出，用于回归）
        """
        for layer in self.layers:
            x = layer(x)  # 将本层输出作为下层输入
        return x[0]  # 输出层取第一个神经元

    def parameters(self) -> List[Value]:
        """返回网络中所有可训练参数"""
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params


# ============================================================================
# 第四部分：演示程序
# ============================================================================

def demo_basic_expression():
    """演示 1: 基本表达式的计算图和反向传播 (a*b + c) * d"""
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 10 + "演示 1: 基本表达式 (a*b + c) * d 的反向传播" + " " * 16 + "║")
    print("╚" + "═" * 68 + "╝")

    # ---- 前向传播 ----
    a = Value(2.0)  # a = 2
    b = Value(3.0)  # b = 3
    c = Value(4.0)  # c = 4
    d = Value(5.0)  # d = 5

    # 构建表达式: f = (a*b + c) * d
    e = a * b        # e = 2*3 = 6
    f = e + c        # f = 6+4 = 10
    L = f * d        # L = 10*5 = 50

    print(f"\n表达式: L = (a*b + c) * d")
    print(f"具体值: L = ({a.data}*{b.data} + {c.data}) * {d.data} = {L.data}")

    # ---- 反向传播 ----
    L.backward()  # 自动计算所有节点的梯度

    print(f"\n反向传播后的梯度:")
    print(f"  ∂L/∂a = {a.grad:.1f}  (预期: d * b = {d.data:.0f} * {b.data:.0f} = {d.data * b.data:.0f})")
    print(f"  ∂L/∂b = {b.grad:.1f}  (预期: d * a = {d.data:.0f} * {a.data:.0f} = {d.data * a.data:.0f})")
    print(f"  ∂L/∂c = {c.grad:.1f}  (预期: d = {d.data:.0f})")
    print(f"  ∂L/∂d = {d.grad:.1f}  (预期: a*b + c = {a.data*b.data + c.data:.0f})")

    # ---- 打印计算图 ----
    print_computation_graph(L)

    # ---- 手动验证 ----
    print("\n【手动验证】")
    # ∂L/∂a = d * (∂(a*b)/∂a) = d * b = 5*3 = 15
    expected_da = d.data * b.data
    print(f"  ∂L/∂a: 计算值={a.grad:.1f}, 预期值={expected_da:.1f}, 匹配={abs(a.grad - expected_da) < 1e-6}")
    # ∂L/∂b = d * a = 5*2 = 10
    expected_db = d.data * a.data
    print(f"  ∂L/∂b: 计算值={b.grad:.1f}, 预期值={expected_db:.1f}, 匹配={abs(b.grad - expected_db) < 1e-6}")
    # ∂L/∂c = d * 1 = 5
    expected_dc = d.data
    print(f"  ∂L/∂c: 计算值={c.grad:.1f}, 预期值={expected_dc:.1f}, 匹配={abs(c.grad - expected_dc) < 1e-6}")
    # ∂L/∂d = a*b + c = 10
    expected_dd = a.data * b.data + c.data
    print(f"  ∂L/∂d: 计算值={d.grad:.1f}, 预期值={expected_dd:.1f}, 匹配={abs(d.grad - expected_dd) < 1e-6}")


def demo_activation_functions():
    """演示 2: 激活函数及其反向传播"""
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 10 + "演示 2: 激活函数的反向传播 (ReLU, Sigmoid, Tanh)" + " " * 14 + "║")
    print("╚" + "═" * 68 + "╝")

    x = Value(1.5)

    # ---- ReLU ----
    print(f"\n--- ReLU ---")
    a_relu = x.relu()
    a_relu.backward()
    print(f"  x={x.data:.1f}, ReLU(x)={a_relu.data:.4f}, ∂ReLU/∂x={x.grad:.4f} (预期: 1.0, 因为 x>0)")

    # ---- Sigmoid ----
    x.zero_grad()  # 清空梯度，准备下一个测试
    print(f"\n--- Sigmoid ---")
    a_sig = x.sigmoid()
    a_sig.backward()
    expected_sig_grad = a_sig.data * (1 - a_sig.data)
    print(f"  x={x.data:.1f}, Sigmoid(x)={a_sig.data:.6f}, ∂Sigmoid/∂x={x.grad:.6f}")
    print(f"  预期导数: σ(x)(1-σ(x)) = {expected_sig_grad:.6f}")

    # ---- Tanh ----
    x.zero_grad()
    print(f"\n--- Tanh ---")
    a_tanh = x.tanh()
    a_tanh.backward()
    expected_tanh_grad = 1 - a_tanh.data ** 2
    print(f"  x={x.data:.1f}, Tanh(x)={a_tanh.data:.6f}, ∂Tanh/∂x={x.grad:.6f}")
    print(f"  预期导数: 1-tanh²(x) = {expected_tanh_grad:.6f}")

    # ---- ReLU 在负值的情况 ----
    x_neg = Value(-1.5)
    print(f"\n--- ReLU (负输入) ---")
    a_relu_neg = x_neg.relu()
    a_relu_neg.backward()
    print(f"  x={x_neg.data:.1f}, ReLU(x)={a_relu_neg.data:.1f}, ∂ReLU/∂x={x_neg.grad:.1f} (预期: 0.0, 因为 x<0)")


def demo_fanout():
    """演示 3: Fan-out —— 梯度累积"""
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 10 + "演示 3: Fan-Out 梯度累积             " + " " * 27 + "║")
    print("╚" + "═" * 68 + "╝")

    # ---- 构建 fan-out 示例: L = (2x) * (x + 3) ----
    x = Value(2.0)
    # 路径 1: x → *2 → u
    u = x * 2    # u = 2x = 4
    # 路径 2: x → +3 → v
    v = x + 3    # v = x+3 = 5
    # 两条路径汇合: u * v = L
    L = u * v    # L = 4*5 = 20

    print(f"\n表达式: L = (2x) * (x + 3)")
    print(f"当 x=2: u=2x={u.data}, v=x+3={v.data}, L={L.data}")

    L.backward()

    # 验证梯度
    # ∂L/∂x = ∂L/∂u · ∂u/∂x + ∂L/∂v · ∂v/∂x
    #        = v · 2        + u · 1
    #        = 5 · 2        + 4 · 1
    #        = 10 + 4 = 14
    expected_grad = v.data * 2 + u.data * 1
    print(f"\n反向传播结果:")
    print(f"  ∂L/∂x = {x.grad:.1f}")
    print(f"  预期值 = v·2 + u·1 = {v.data:.0f}·2 + {u.data:.0f}·1 = {expected_grad:.0f}")
    print(f"  分解: 路径1贡献 = {v.data * 2:.0f} (∂L/∂u · ∂u/∂x = v · 2)")
    print(f"        路径2贡献 = {u.data:.0f} (∂L/∂v · ∂v/∂x = u · 1)")
    print(f"        总和 = {v.data * 2 + u.data:.0f}")

    print(f"\n  梯度匹配: {abs(x.grad - expected_grad) < 1e-6}")


def demo_mini_neural_network():
    """演示 4: 使用 mini autograd 训练一个小神经网络"""
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 10 + "演示 4: 小神经网络完整训练 (正向+反向+更新)" + " " * 16 + "║")
    print("╚" + "═" * 68 + "╝")

    # ---- 创建一个简单的 MLP: 2输入 → 4隐藏 → 1输出 ----
    model = MLP(2, [4, 1], ["relu", "linear"])
    print(f"\n网络结构: 输入=2, 隐藏层=4(ReLU), 输出=1(线性)")
    print(f"总参数量: {len(model.parameters())}")

    # ---- 生成简单的训练数据 ----
    # 目标函数: y = 3*x1^2 - 2*x2 + 1
    xs = [
        [Value(1.0), Value(2.0)],
        [Value(2.0), Value(1.0)],
        [Value(0.0), Value(3.0)],
        [Value(3.0), Value(0.0)],
    ]
    ys = [3*1*1 - 2*2 + 1, 3*2*2 - 2*1 + 1, 3*0*0 - 2*3 + 1, 3*3*3 - 2*0 + 1]
    # 目标值: [0, 11, -5, 28]

    print(f"\n训练数据:")
    for i, (xi, yi) in enumerate(zip(xs, ys)):
        print(f"  样本{i+1}: x={[v.data for v in xi]}, y={yi}")

    # ---- 训练循环 ----
    learning_rate = 0.01
    n_epochs = 100

    print(f"\n开始训练 (lr={learning_rate}, epochs={n_epochs})...")

    for epoch in range(n_epochs):
        # ---- 前向传播 ----
        y_preds = [model(x) for x in xs]  # 对每个样本进行预测

        # ---- 计算损失 (MSE) ----
        # L = (1/N) * Σ (y_pred - y_true)²
        losses = [(yp - Value(y_true)) ** 2 for yp, y_true in zip(y_preds, ys)]
        total_loss = sum(losses[1:], losses[0])  # 对损失求和
        # 注意：这里没有除以 N，实际训练时除以 N 不影响优化方向

        # ---- 反向传播 ----
        # 先清零所有参数的梯度
        for p in model.parameters():
            p.grad = 0.0
        total_loss.backward()  # 反向传播，计算所有梯度

        # ---- 参数更新 (梯度下降) ----
        for p in model.parameters():
            p.data -= learning_rate * p.grad  # θ := θ - α·∇L

        # ---- 打印训练进度 ----
        if epoch % 20 == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch:3d}: loss = {total_loss.data:.4f}, "
                  f"预测 = {[f'{yp.data:.2f}' for yp in y_preds]}")

    print(f"\n训练完成！最终预测 vs 目标:")
    for i, (xi, yi) in enumerate(zip(xs, ys)):
        yp = model(xi)
        print(f"  样本{i+1}: 预测={yp.data:.2f}, 目标={yi}, 误差={abs(yp.data - yi):.2f}")


def demo_gradient_descent_1d():
    """演示 5: 用自动微分做梯度下降，求 f(x) = x² + 3x 的最小值"""
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 8 + "演示 5: 自动微分 + 梯度下降求 f(x)=x²+3x 最小值" + " " * 14 + "║")
    print("╚" + "═" * 68 + "╝")

    x = Value(5.0)  # 初始值 x=5
    learning_rate = 0.1
    n_steps = 20

    print(f"\nf(x) = x² + 3x, 初始 x = {x.data:.1f}")
    print(f"解析解: f'(x)=2x+3=0 → x = -1.5")
    print(f"最小值: f(-1.5) = (-1.5)² + 3(-1.5) = 2.25 - 4.5 = -2.25")

    print(f"\n梯度下降过程 (lr={learning_rate}):")
    print(f"{'步骤':<6} {'x':<12} {'f(x)':<12} {'梯度':<12}")

    for step in range(n_steps):
        # 构造表达式
        loss = x * x + Value(3.0) * x  # f(x) = x² + 3x

        # 清零梯度
        x.grad = 0.0

        # 反向传播
        loss.backward()

        # 打印当前状态
        if step % 5 == 0 or step == n_steps - 1:
            print(f"{step:<6} {x.data:<12.4f} {loss.data:<12.6f} {x.grad:<12.4f}")

        # 参数更新
        x.data -= learning_rate * x.grad

    print(f"\n最终结果: x ≈ {x.data:.4f} (理论最优: -1.5)")
    print(f"最小函数值: f({x.data:.4f}) ≈ {x.data**2 + 3*x.data:.4f} (理论: -2.25)")


# ============================================================================
# 主程序入口
# ============================================================================

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║     s06 反向传播与链式法则 — 从零实现 mini autograd 引擎        ║")
    print("║     灵感来源: Andrej Karpathy 的 micrograd                       ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # 依次运行所有演示
    demo_basic_expression()       # 基本表达式和链式法则
    demo_activation_functions()   # 激活函数的反向传播
    demo_fanout()                 # Fan-out 梯度累积
    demo_mini_neural_network()    # 完整神经网络训练
    demo_gradient_descent_1d()    # 自动微分做优化

    print("\n" + "=" * 70)
    print("【总结】")
    print("=" * 70)
    print("  ✓ 实现了 Value 类，支持自动微分的核心操作")
    print("  ✓ 构建了计算图，按拓扑逆序执行反向传播")
    print("  ✓ 验证了链式法则、局部梯度规则和梯度累积")
    print("  ✓ 用 mini autograd 完成了完整的神经网络训练流程")
    print("  ✓ 核心原理与 PyTorch/TensorFlow 的 autograd 一致")
    print("=" * 70)
