# -*- coding: utf-8 -*-
"""
ml13 概率图模型基础 — 演示代码
===============================
功能：实现简单贝叶斯网络的枚举推理和变量消除（Variable Elimination），
      展示链状因子图上的信念传播，以及 d-分离的可视化。

每个函数都有中文 docstring，每行逻辑代码都有中文注释。
运行方式：在 ml13_probabilistic_graphical_models/ 目录下执行 python code/demo.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
from itertools import product

_HERE = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_HERE, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)


# ============================================================================
# 第一部分：贝叶斯网络的定义
# ============================================================================

class BayesianNetwork:
    """
    贝叶斯网络（离散变量）。

    结构：
        parents: dict {node: list of parent nodes}
        cpt: dict {node: 条件概率表（多维数组）}

    联合分布：
        P(X_1,...,X_n) = ∏_i P(X_i | Pa(X_i))
    """

    def __init__(self):
        self.parents = {}
        self.cpt = {}
        self.nodes = []
        self.cardinalities = {}  # 每个变量的取值数量

    def add_node(self, name, cardinality, parents=None, cpt=None):
        """添加节点及其 CPT"""
        self.nodes.append(name)
        self.cardinalities[name] = cardinality
        self.parents[name] = parents if parents else []
        if cpt is not None:
            self.cpt[name] = np.array(cpt)

    def get_cpt_value(self, node, assignment):
        """
        从 CPT 中查找 P(node = val | Pa = pa_vals)。

        参数:
            node: 节点名
            assignment: 字典 {变量名: 取值}，必须包含 node 和其所有父节点
        """
        # 获取 node 和其父节点的取值索引
        idx = tuple(assignment[p] for p in [node] + self.parents[node])
        # 多维数组索引，第一个维度是 node 的取值
        # CPT 形状: (|X_node|, |X_pa1|, |X_pa2|, ...)
        # 索引: [node_val, pa1_val, pa2_val, ...]
        cpt_arr = self.cpt[node]
        return cpt_arr[idx]

    def joint_probability(self, assignment):
        """
        计算联合概率 P(X_1,...,X_n) = ∏ P(X_i | Pa(X_i))

        参数:
            assignment: 字典 {变量名: 取值}
        """
        prob = 1.0
        for node in self.nodes:
            prob *= self.get_cpt_value(node, assignment)
        return prob

    def enumerate_all(self, query, evidence=None):
        """
        枚举推理：通过遍历所有可能的变量取值组合，
        计算 P(query | evidence)。

        参数:
            query: 查询变量名
            evidence: 证据字典 {var: value}（可选）
        返回:
            查询变量的概率分布（列表，长度 = cardinality）
        """
        if evidence is None:
            evidence = {}

        cardinality = self.cardinalities[query]
        query_probs = np.zeros(cardinality)

        # 获取所有变量的取值空间
        all_nodes = self.nodes
        # 生成所有可能的赋值组合
        node_ranges = [range(self.cardinalities[n]) for n in all_nodes]

        for values in product(*node_ranges):
            assignment = dict(zip(all_nodes, values))

            # 检查是否与 evidence 一致
            consistent = all(assignment[var] == val for var, val in evidence.items())
            if not consistent:
                continue

            prob = self.joint_probability(assignment)
            q_val = assignment[query]
            query_probs[q_val] += prob

        # 归一化
        if query_probs.sum() > 0:
            query_probs /= query_probs.sum()
        return query_probs


# ============================================================================
# 第二部分：构建示例贝叶斯网络（Sprinkler 模型）
# ============================================================================

def build_sprinkler_network():
    """
    经典 Sprinkler 贝叶斯网络：

        Cloudy (C) → Sprinkler (S) → WetGrass (W)
              ↓                            ↑
              └──────────────────────────┘
                        Rain (R) → WetGrass (W)

    变量都是二值的 (0=no, 1=yes)
    """
    bn = BayesianNetwork()

    # P(Cloudy)
    bn.add_node('Cloudy', 2, cpt=[0.5, 0.5])

    # P(Sprinkler | Cloudy)
    bn.add_node('Sprinkler', 2, parents=['Cloudy'],
                cpt=[
                    [0.5, 0.1],  # P(S=0 | C): C=0 → 0.5, C=1 → 0.1
                    [0.5, 0.9],  # P(S=1 | C): C=0 → 0.5, C=1 → 0.9
                ])

    # P(Rain | Cloudy)
    bn.add_node('Rain', 2, parents=['Cloudy'],
                cpt=[
                    [0.8, 0.2],  # P(R=0 | C): C=0 → 0.8, C=1 → 0.2
                    [0.2, 0.8],  # P(R=1 | C): C=0 → 0.2, C=1 → 0.8
                ])

    # P(WetGrass | Sprinkler, Rain)
    bn.add_node('WetGrass', 2, parents=['Sprinkler', 'Rain'],
                cpt=[
                    [[1.0, 0.1], [0.1, 0.01]],  # P(W=0 | S, R)
                    [[0.0, 0.9], [0.9, 0.99]],  # P(W=1 | S, R)
                ])

    return bn


def demo_sprinkler_queries(bn):
    """展示 Sprinkler 网络上的多种推理查询"""
    print('\n[Sprinkler 贝叶斯网络推理]')

    # 查询 1: P(Cloudy) — 无证据
    p_c = bn.enumerate_all('Cloudy')
    print(f'  P(Cloudy): {p_c}')

    # 查询 2: P(Rain | WetGrass=1) — 知道草湿了，下雨的概率
    p_r_given_w = bn.enumerate_all('Rain', evidence={'WetGrass': 1})
    print(f'  P(Rain=1 | WetGrass=1) = {p_r_given_w[1]:.4f}')

    # 查询 3: P(Sprinkler | WetGrass=1) — 对比洒水器的概率
    p_s_given_w = bn.enumerate_all('Sprinkler', evidence={'WetGrass': 1})
    print(f'  P(Sprinkler=1 | WetGrass=1) = {p_s_given_w[1]:.4f}')

    # 查询 4: Explaining away — P(Rain=1 | WetGrass=1, Sprinkler=1)
    p_r_given_ws = bn.enumerate_all('Rain', evidence={'WetGrass': 1, 'Sprinkler': 1})
    print(f'  P(Rain=1 | WetGrass=1, Sprinkler=1) = {p_r_given_ws[1]:.4f}')
    print(f'  (解释消除: 知道洒水器开了 → 雨的概率下降，因为洒水器"解释"了草湿)')

    return bn


# ============================================================================
# 第三部分：变量消除（Variable Elimination）
# ============================================================================

class Factor:
    """概率因子 —— 变量消除的基本操作单元"""

    def __init__(self, variables, values):
        """
        参数:
            variables: 变量名列表
            values: 多维数组，values[a,b,c] = f(var1=a, var2=b, var3=c)
        """
        self.variables = list(variables)
        self.values = np.array(values, dtype=np.float64)

    def multiply(self, other):
        """两个因子的乘积（变量集取并集，值逐元素相乘）"""
        # 找到公共变量和独有变量
        common = [v for v in self.variables if v in other.variables]
        new_vars = self.variables + [v for v in other.variables if v not in self.variables]

        # 用广播实现：扩展两个因子到相同的形状（union of variables）
        shape_self = []
        shape_other = []
        for v in new_vars:
            if v in self.variables:
                shape_self.append(self.values.shape[self.variables.index(v)])
            else:
                shape_self.append(1)
            if v in other.variables:
                shape_other.append(other.values.shape[other.variables.index(v)])
            else:
                shape_other.append(1)

        expanded_self = self.values.reshape(shape_self)
        expanded_other = other.values.reshape(shape_other)

        result_values = expanded_self * expanded_other
        return Factor(new_vars, result_values)

    def marginalize(self, variable):
        """对指定变量求和（消除该变量）"""
        if variable not in self.variables:
            return self
        axis = self.variables.index(variable)
        new_vars = [v for v in self.variables if v != variable]
        new_values = self.values.sum(axis=axis)
        return Factor(new_vars, new_values)

    def normalize(self):
        """归一化：使得所有值之和为 1"""
        total = self.values.sum()
        if total > 0:
            self.values /= total


def variable_elimination(bn, query, evidence=None):
    """
    变量消除算法：计算 P(query | evidence)。

    参数:
        bn: BayesianNetwork 实例
        query: 查询变量
        evidence: 证据字典

    返回:
        查询变量的概率分布
    """
    if evidence is None:
        evidence = {}

    # 消除顺序：除 query 和 evidence 外的所有变量
    elim_order = [n for n in bn.nodes if n != query and n not in evidence]

    # 构建初始因子列表（每个节点的 CPT 转换因子）
    factors = []
    for node in bn.nodes:
        vars_in_factor = [node] + bn.parents[node]
        # 从 CPT 中提取因子值
        cpt = bn.cpt[node]
        # 如果 node 在 evidence 中，先切片（固定证据值）
        cpt_copy = cpt.copy()
        if node in evidence:
            # 取第 evidence[node] 个切片
            cpt_copy = cpt_copy[evidence[node]]
            vars_in_factor = bn.parents[node]
        factors.append(Factor(vars_in_factor, cpt_copy))

    # 对其他 evidence 节点的因子进行切片
    for factor in factors:
        for ev_var, ev_val in evidence.items():
            if ev_var in factor.variables and ev_var != list(factor.variables)[0]:
                idx = factor.variables.index(ev_var)
                factor.values = np.take(factor.values, ev_val, axis=idx)
                if factor.values.ndim > 1:
                    # 简化：如果维度已经消除，调整 shape
                    pass

    # 变量消除循环
    for var in elim_order:
        # 找到所有包含 var 的因子
        relevant = [f for f in factors if var in f.variables]
        if not relevant:
            continue

        # 乘积
        product_factor = relevant[0]
        for f in relevant[1:]:
            product_factor = product_factor.multiply(f)

        # 消除 var（求和）
        product_factor = product_factor.marginalize(var)

        # 替换因子列表
        factors = [f for f in factors if var not in f.variables]
        factors.append(product_factor)

    # 所有剩余因子的乘积 = P(query, evidence)
    result = factors[0]
    for f in factors[1:]:
        result = result.multiply(f)

    result.normalize()
    return result.values.flatten()


def demo_variable_elimination(bn):
    """对比枚举法和变量消除的结果"""
    print('\n[变量消除 vs 枚举法]')

    # 查询: P(Cloudy | WetGrass=1)
    p_enum = bn.enumerate_all('Cloudy', evidence={'WetGrass': 1})
    p_ve = variable_elimination(bn, 'Cloudy', evidence={'WetGrass': 1})

    print(f'  枚举法:      P(Cloudy | WetGrass=1) = {p_enum}')
    print(f'  变量消除法:  P(Cloudy | WetGrass=1) = {p_ve}')
    print(f'  等价: {np.allclose(p_enum, p_ve, atol=1e-10)}')


# ============================================================================
# 第四部分：链状图上的信念传播
# ============================================================================

def belief_propagation_chain(potentials, evidence=None, n_iter=10):
    """
    在链状因子图上执行信念传播（BP）。

    链: x1 — f1 — x2 — f2 — x3 — ... — x_n

    势函数:
        f_i(x_i, x_{i+1}) = 转移"兼容度"（非归一化）

    参数:
        potentials: 势函数列表 [(n_i, n_{i+1}) 形状数组]
        evidence: 证据字典 {node_idx: value}（可选）
        n_iter: 迭代次数

    返回:
        marginals: 每个变量的边缘分布
    """
    n_vars = len(potentials) + 1
    card = [potentials[0].shape[0]]  # 每个变量的取值数
    for p in potentials:
        card.append(p.shape[1])

    # 初始化消息（均匀分布）
    fwd_messages = [np.ones(c) for c in card]   # 前向消息
    bwd_messages = [np.ones(c) for c in card]   # 后向消息

    for it in range(n_iter):
        # 前向传递：x_i → f_i → x_{i+1}
        for i in range(n_vars - 1):
            # 消息从 x_i 收集后传给因子 f_i，再传给 x_{i+1}
            msg_from_x = fwd_messages[i].copy()
            if evidence is not None and i in evidence:
                msg_from_x *= 0
                msg_from_x[evidence[i]] = 1.0

            # 因子 f_i 的消息：Σ_{x_i} f_i(x_i, x_{i+1}) * msg(x_i)
            psi = potentials[i]  # (n_i, n_{i+1})
            msg_to_next = (msg_from_x[:, np.newaxis] * psi).sum(axis=0)
            msg_to_next /= msg_to_next.sum() + 1e-10  # 归一化
            fwd_messages[i+1] = msg_to_next

        # 后向传递：x_{i+1} → f_i → x_i
        for i in range(n_vars - 2, -1, -1):
            msg_from_next = bwd_messages[i+1].copy()
            if evidence is not None and (i+1) in evidence:
                msg_from_next *= 0
                msg_from_next[evidence[i+1]] = 1.0

            psi = potentials[i]  # (n_i, n_{i+1})
            msg_to_prev = (psi * msg_from_next[np.newaxis, :]).sum(axis=1)
            msg_to_prev /= msg_to_prev.sum() + 1e-10
            bwd_messages[i] = msg_to_prev

    # 计算边缘分布: P(x_i) ∝ fwd_msg[i] * bwd_msg[i]
    marginals = []
    for i in range(n_vars):
        marg = fwd_messages[i] * bwd_messages[i]
        if evidence is not None and i in evidence:
            marg *= 0
            marg[evidence[i]] = 1.0
        marg /= marg.sum() + 1e-10
        marginals.append(marg)

    return marginals, fwd_messages, bwd_messages


def demo_belief_propagation():
    """链状图上的 BP 演示"""
    print('\n[链状图信念传播]')

    # 3 变量链 x1 — f1 — x2 — f2 — x3
    # 势函数 f1 和 f2 定义了相邻变量的"兼容度"
    potentials = [
        np.array([[0.9, 0.1], [0.2, 0.8]]),  # f1(x1, x2): 倾向同值
        np.array([[0.8, 0.2], [0.1, 0.9]]),  # f2(x2, x3): 倾向同值
    ]

    # 无证据
    marginals, fwd, bwd = belief_propagation_chain(potentials)
    print(f'  无证据时边缘分布:')
    for i, m in enumerate(marginals):
        print(f'    P(x{i+1}) = {m}')

    # 带证据: x2 = 1
    marginals_ev, _, _ = belief_propagation_chain(potentials, evidence={1: 1})
    print(f'  有证据 x2=1 时边缘分布:')
    for i, m in enumerate(marginals_ev):
        print(f'    P(x{i+1} | x2=1) = {m}')

    return marginals


# ============================================================================
# 第五部分：d-分离可视化
# ============================================================================

def plot_d_separation():
    """绘制 d-分离的三种基本结构"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    structures = [
        {
            'title': 'Chain: A → B → C',
            'nodes': [(0, 0.5, 'A'), (0.5, 0.5, 'B'), (1, 0.5, 'C')],
            'edges': [(0, 1), (1, 2)],
            'observed': [1],
            'indep': (0, 2),
            'note': 'A ⊥ C | B',
        },
        {
            'title': 'Fork: A ← B → C',
            'nodes': [(0, 0.5, 'A'), (0.5, 0.7, 'B'), (1, 0.5, 'C')],
            'edges': [(1, 0), (1, 2)],
            'observed': [1],
            'indep': (0, 2),
            'note': 'A ⊥ C | B',
        },
        {
            'title': 'Collider: A → B ← C',
            'nodes': [(0, 0.7, 'A'), (0.5, 0.5, 'B'), (1, 0.7, 'C')],
            'edges': [(0, 1), (2, 1)],
            'observed': [],
            'indep': (0, 2),
            'note': 'A ⊥ C (when B unobs.)',
        },
    ]

    for ax, struct in zip(axes, structures):
        # 画边
        for e in struct['edges']:
            i, j = e
            xi, yi, _ = struct['nodes'][i]
            xj, yj, _ = struct['nodes'][j]
            ax.annotate('', xy=(xj, yj), xytext=(xi, yi),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=2))
        # 画节点
        for idx, (x, y, name) in enumerate(struct['nodes']):
            color = 'lightcoral' if idx in struct['observed'] else 'lightblue'
            edge = 'darkred' if idx in struct['observed'] else 'navy'
            circle = plt.Circle((x, y), 0.08, color=color, ec=edge, linewidth=2)
            ax.add_patch(circle)
            ax.text(x, y, name, ha='center', va='center', fontsize=11, fontweight='bold')

        # 标注独立性
        i, j = struct['indep']
        xi, yi, _ = struct['nodes'][i]
        xj, yj, _ = struct['nodes'][j]
        mx, my = (xi + xj) / 2, (yi + yj) / 2
        ax.plot([xi + 0.1, xj - 0.1], [yi + 0.1, yj + 0.1], 'r--', linewidth=1.5, alpha=0.5)

        ax.text(0.5, 0.05, struct['note'], transform=ax.transAxes, ha='center',
                fontsize=11, style='italic', color='darkred')
        ax.set_title(struct['title'], fontsize=12, fontweight='bold')
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(0.3, 0.9)
        ax.axis('off')

    plt.suptitle('d-Separation: Three Fundamental Structures', fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(_IMAGES_DIR, 'ml13-03-dseparation-diagram.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[保存] {path}')


# ============================================================================
# 第六部分：主程序
# ============================================================================

def main():
    print('=' * 60)
    print('ml13 概率图模型基础 — 演示代码')
    print('=' * 60)

    # 1. 构建 Sprinkler 贝叶斯网络
    print('\n[1/4] 构建 Sprinkler 贝叶斯网络...')
    bn = build_sprinkler_network()

    # 2. 枚举推理查询
    print('[2/4] 枚举推理...')
    demo_sprinkler_queries(bn)

    # 3. 变量消除 vs 枚举法
    print('[3/4] 变量消除...')
    demo_variable_elimination(bn)

    # 4. 链状图上的信念传播
    print('[4/4] 链状图信念传播...')
    demo_belief_propagation()

    # 5. d-分离可视化
    print('[5/5] d-分离可视化...')
    plot_d_separation()

    print('\n完成！所有图表已保存到 images/ 目录。')


if __name__ == '__main__':
    main()
