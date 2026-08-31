# -*- coding: utf-8 -*-
"""
ml13 概率图模型基础 — 练习代码
===============================
请完成以下 TODO 任务，巩固对贝叶斯网络推理和信念传播的理解。

每个 TODO 都有详细的指示和预期输出描述。
建议先阅读 index.md 和 code-demo.md，再尝试独立补全代码。
"""

import numpy as np


# ============================================================================
# TODO 1: 计算贝叶斯网络的联合概率
# ============================================================================
def compute_joint_probability(cpts, assignment):
    """
    计算贝叶斯网络的联合概率。

    联合分布: P(X1,...,Xn) = ∏_i P(X_i | Pa(X_i))

    参数:
        cpts: 列表，每个元素是一个函数 cpt_i(assignment) → 概率值
        assignment: 字典 {var_name: value}

    返回:
        联合概率（浮点数）
    """
    prob = 1.0
    for cpt_fn in cpts:
        # TODO: 对每个节点，调用其 CPT 函数，乘积累加
        prob = None  # ← TODO: prob * cpt_fn(assignment)
    return prob


def test_joint_prob():
    """测试联合概率计算"""
    print("=" * 60)
    print("TODO 1 测试: 联合概率计算")
    print("=" * 60)

    # 简单两变量网络: P(A) P(B|A)
    # P(A=0)=0.4, P(A=1)=0.6
    # P(B=0|A=0)=0.7, P(B=0|A=1)=0.2
    def cpt_A(assign):
        return 0.4 if assign['A'] == 0 else 0.6

    def cpt_B(assign):
        if assign['A'] == 0:
            return 0.7 if assign['B'] == 0 else 0.3
        else:
            return 0.2 if assign['B'] == 0 else 0.8

    cpts = [cpt_A, cpt_B]
    prob = compute_joint_probability(cpts, {'A': 0, 'B': 0})

    if prob is None:
        print("  TODO 1 未完成，请补全 compute_joint_probability 函数")
    else:
        print(f"  P(A=0, B=0) = {prob:.4f} (期望: 0.4 * 0.7 = 0.28)")
        assert abs(prob - 0.28) < 1e-6, f"期望 0.28, 得到 {prob}"


# ============================================================================
# TODO 2: 判断 d-分离
# ============================================================================
def is_d_separated(edges, X, Y, Z, debug=False):
    """
    判断在给定 Z 的条件下，X 是否 d-分离于 Y。

    简化版本：只处理三个节点的链/分叉/汇合结构。

    参数:
        edges: 有向边列表 [(from, to), ...]
        X, Y: 两个节点名（字符串）
        Z: 给定观测的节点集（列表）
        debug: 是否打印调试信息

    返回:
        True 如果 X 和 Y 在给定 Z 下条件独立
    """
    # 如果 X 和 Y 之间有直接边，不独立
    if (X, Y) in edges or (Y, X) in edges:
        return False

    # TODO: 检查 X 和 Y 是否通过中间节点 B 连接
    # 找到 X 和 Y 的公共邻居（可能经过的中间节点）
    for b in set([e[0] for e in edges] + [e[1] for e in edges]):
        if b == X or b == Y:
            continue

        # TODO: 检查是否 X → B → Y (链式)
        is_chain = None  # ← TODO: (X, b) in edges and (b, Y) in edges

        # TODO: 检查是否 X ← B → Y (分叉)
        is_fork = None   # ← TODO: (b, X) in edges and (b, Y) in edges

        # TODO: 检查是否 X → B ← Y (汇合)
        is_collider = None  # ← TODO: (X, b) in edges and (Y, b) in edges

        if is_chain or is_fork:
            # 链式和分叉：如果 B 被观测则阻塞
            if b in Z:
                return True
        elif is_collider:
            # 汇合：如果 B 不被观测且 B 的后代不被观测，则阻塞
            if b not in Z:
                return True

    return False


def test_d_separation():
    """测试 d-分离判断"""
    print("\n" + "=" * 60)
    print("TODO 2 测试: d-分离判断")
    print("=" * 60)

    # 链式 A → B → C
    edges_chain = [('A', 'B'), ('B', 'C')]

    result1 = is_d_separated(edges_chain, 'A', 'C', ['B'])
    print(f"  Chain A→B→C, Z={{B}}: {'独立' if result1 else '不独立'} (期望: 独立)")

    result2 = is_d_separated(edges_chain, 'A', 'C', [])
    print(f"  Chain A→B→C, Z={{}}:  {'独立' if result2 else '不独立'} (期望: 不独立)")

    # 汇合 A → B ← C
    edges_collider = [('A', 'B'), ('C', 'B')]
    result3 = is_d_separated(edges_collider, 'A', 'C', [])
    print(f"  Collider A→B←C, Z={{}}: {'独立' if result3 else '不独立'} (期望: 独立)")

    result4 = is_d_separated(edges_collider, 'A', 'C', ['B'])
    print(f"  Collider A→B←C, Z={{B}}: {'独立' if result4 else '不独立'} (期望: 不独立)")


# ============================================================================
# TODO 3: 实现信念传播的单步消息传递
# ============================================================================
def bp_message_from_factor(psi, msg_in):
    """
    计算从因子到变量的消息（和积算法的核心操作）。

    因子到变量 x_j 的消息：
        μ_{f→x_j}(x_j) = Σ_{x_i} ψ(x_i, x_j) · μ_{x_i→f}(x_i)

    参数:
        psi: 势函数（因子），shape (n_i, n_j)
        msg_in: 来自变量 x_i 的消息，shape (n_i,)

    返回:
        msg_out: 发送给变量 x_j 的消息，shape (n_j,)
    """
    # TODO: 对 x_i 求和
    # μ(x_j) = Σ_{x_i} ψ(x_i, x_j) · msg(x_i)
    # 提示: 将 msg_in 扩展为 (n_i, 1)，与 psi 逐元素相乘，再沿 axis=0 求和
    msg_out = None  # ← TODO: (msg_in[:, np.newaxis] * psi).sum(axis=0)

    # 归一化
    if msg_out.sum() > 0:
        msg_out = msg_out / msg_out.sum()

    return msg_out


def test_bp_message():
    """测试 BP 消息"""
    print("\n" + "=" * 60)
    print("TODO 3 测试: BP 消息传递")
    print("=" * 60)

    psi = np.array([[0.9, 0.1], [0.2, 0.8]])  # 倾向同值的势函数
    msg_in = np.array([0.5, 0.5])              # 均匀先行消息

    msg_out = bp_message_from_factor(psi, msg_in)

    if msg_out is None:
        print("  TODO 3 未完成，请补全 bp_message_from_factor 函数")
    else:
        print(f"  输入消息: {msg_in}")
        print(f"  势函数 ψ:\n{psi}")
        print(f"  输出消息: {msg_out}")
        print(f"  消息和为 1: {abs(msg_out.sum() - 1.0) < 1e-6}")


# ============================================================================
# TODO 4: 用变量消除法计算 P(query | evidence)
# ============================================================================
def simple_variable_elimination(factors, query_var, evidence, elim_order):
    """
    简化的变量消除算法（因子为列表形式）。

    参数:
        factors: 因子列表，每个是 (variables, values) 元组
                 variables: 变量名列表
                 values: numpy 数组
        query_var: 查询变量名
        evidence: 证据字典 {var: value}
        elim_order: 变量消除顺序（列表）

    返回:
        查询变量的后验概率分布（numpy 数组，已归一化）
    """
    # TODO: 步骤 1 — 对 evidence 中的变量进行切片（固定取值）
    simplified = []
    for vars_list, vals in factors:
        vals_copy = vals.copy()
        # 如果 evidence 中有该变量，对对应维度做切片
        for ev_var, ev_val in evidence.items():
            if ev_var in vars_list:
                axis = vars_list.index(ev_var)
                vals_copy = None  # ← TODO: np.take(vals_copy, ev_val, axis=axis)
                vars_list = [v for v in vars_list if v != ev_var]  # 从变量列表中移除
        simplified.append((vars_list, vals_copy))

    # TODO: 步骤 2 — 按 elim_order 逐个消除变量
    for var in elim_order:
        # 找到所有包含 var 的因子
        relevant = [(v, arr) for v, arr in simplified if var in v]
        if not relevant:
            continue

        # 乘积所有相关因子
        result_arr = relevant[0][1]
        result_vars = list(relevant[0][0])
        for vars2, arr2 in relevant[1:]:
            # 简单相乘（假设维度对齐）
            result_arr = result_arr * arr2

        # TODO: 对 var 求和（消除）
        axis = result_vars.index(var)
        result_arr = None  # ← TODO: result_arr.sum(axis=axis)
        result_vars.remove(var)

        # 更新因子列表
        simplified = [(v, arr) for v, arr in simplified if var not in v]
        simplified.append((result_vars, result_arr))

    # TODO: 步骤 3 — 所有剩余因子的乘积并归一化
    final = simplified[0][1]
    for _, arr in simplified[1:]:
        final = final * arr

    # 归一化
    if final.sum() > 0:
        final = final / final.sum()

    return final.flatten()


def test_variable_elim():
    """测试变量消除"""
    print("\n" + "=" * 60)
    print("TODO 4 测试: 变量消除")
    print("=" * 60)

    # 简单链: P(A) P(B|A) — 查询 P(B)
    fA = (['A'], np.array([0.4, 0.6]))     # P(A)
    fB = (['A', 'B'], np.array([[0.7, 0.3], [0.2, 0.8]]))  # P(B|A)
    factors = [fA, fB]

    # 查询 P(B)
    result = simple_variable_elimination(factors, 'B', {}, ['A'])

    if result is None:
        print("  TODO 4 未完成，请补全 simple_variable_elimination 函数")
    else:
        print(f"  P(B) = {result}")
        # P(B=0) = 0.4*0.7 + 0.6*0.2 = 0.4
        # P(B=1) = 0.4*0.3 + 0.6*0.8 = 0.6
        print(f"  期望: P(B=0)=0.4, P(B=1)=0.6")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == '__main__':
    print("ml13 概率图模型基础 — 练习代码")
    print("请完成所有 TODO 标记的代码")
    print()

    test_joint_prob()
    test_d_separation()
    test_bp_message()
    test_variable_elim()

    print("\n全部测试完成！")
