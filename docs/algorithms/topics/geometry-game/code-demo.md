---
title: "algo16 计算几何与博弈论入门 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# algo16 计算几何与博弈论入门 — demo.py 代码详解

<a href="/notebook/code/algorithms/topics/geometry-game/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/algorithms/topics/geometry-game/code
python demo.py
```

## 代码逐段详解

### 第1步：叉积 — 计算几何的万能工具

```python
def cross_product(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
```

**几何直觉**：cross > 0 表示从 OA 到 OB 是逆时针旋转（左转），< 0 是顺时针旋转（右转），= 0 是共线。

**关键应用**：
- 判断三点是左转还是右转 → Graham Scan 的核心
- 线段相交检测 → 跨越测试
- 有向面积 → 三角剖分求面积

### 第2步：线段相交 — 两次跨越测试

```python
d1 = cross_product(q1, q2, p1)  # P1 在 Q1Q2 的哪一侧？
d2 = cross_product(q1, q2, p2)  # P2 在 Q1Q2 的哪一侧？
# 如果 d1 和 d2 异号（一正一负），说明 P1P2 跨越了 Q1Q2
# 反之亦然：检查 Q1Q2 是否跨越 P1P2
```

**跨越测试的直觉**：两条线段相交，当且仅当每条线段的两个端点都在另一条线段所在直线的两侧。这是一种"你在我的左右，我在你的上下"的双向约束。

### 第3步：Graham Scan

```python
while len(hull) >= 2 and cross_product(hull[-2], hull[-1], p) <= 0:
    hull.pop()
```

**核心逻辑**：用栈维护凸包的"凸"性。当新加入的点与栈顶两点构成"右转"（或共线）时，说明栈顶点不是凸包的顶点——它位于前一个点和当前点的连线内侧——需要弹出。

**为什么 `<= 0` 而非 `< 0`？** 共线情况（= 0）时保留最近的点（更外围），所以也弹出共线点。

### 第4步：最近点对（分治法）

分治策略的巧妙之处在合并步骤：

1. 已经知道左右两半的最近距离 $d$（的最小值）
2. 对于跨中线的点对，只检查距离中线 $\leq d$ 的"条带"内的点
3. 条带内的点按 y 坐标排序后，每个点只需要检查上方不多于 7 个点

**为什么是 7 个点？** 因为条带的半宽为 $d$，在 $d \times 2d$ 的矩形区域内，任何两点距离都 $\geq d$（否则 $d$ 不是最近距离）。这种几何约束下，矩形内最多放 8 个点（排列在格点上），但每个点只需检查接下来的 7 个就够了。

### 第5步：Nim 游戏 — Bouton 定理

```python
def nim_solve(piles):
    xor_sum = 0
    for p in piles:
        xor_sum ^= p
    if xor_sum == 0:
        return False, None  # 先手必败
    # 找必胜操作: 将 pile 减少为 pile ^ xor_sum
```

**Bouton 定理的简洁性令人叹服**：一个 XOR 运算就解决了 Nim 游戏的胜负判断。定理的关键在于：
1. 全 0 状态 XOR=0 → 必败
2. 从 XOR≠0 总能一步走到 XOR=0（必胜操作存在）
3. 从 XOR=0，任何操作都会破坏（对手重新获得必胜态）

### 第6步：SG 函数

```python
def sg_function(stones, moves):
    sg = [0] * (stones + 1)
    for i in range(1, stones + 1):
        reachable_sg = set()
        for m in moves:
            if i >= m:
                reachable_sg.add(sg[i - m])
        sg[i] = mex(reachable_sg)
    return sg[stones], sg
```

对于取石子游戏（每次可取 1, 3, 4 个）：
- SG(0) = mex({}) = 0（终态必败）
- SG(1) = mex({SG(0)}) = mex({0}) = 1
- SG(2) = mex({SG(1)}) = mex({1}) = 0
- SG(3) = mex({SG(2), SG(0)}) = mex({0, 0}) = 1
- SG(4) = mex({SG(3), SG(1), SG(0)}) = mex({1, 1, 0}) = 2

### 第7步：Minimax 井字棋

井字棋只有 $3^9 \approx 20000$ 种状态，Minimax 可以在瞬间搜索完整棵树。AI 永远不输（最优博弈结果是平局）。

## 关键概念速查表

| 概念 | 公式/方法 | 代码位置 |
|------|----------|---------|
| 叉积 | $(x_1 y_2 - x_2 y_1)$ | `cross_product()` |
| 点积 | $(x_1 x_2 + y_1 y_2)$ | `dot_product()` |
| 线段相交 | 两次跨越测试 | `segments_intersect()` |
| Ray Casting | 水平射线奇偶性 | `point_in_polygon()` |
| Graham Scan | 极角排序 + 栈 | `graham_scan()` |
| 最近点对 | 分治 + 条带扫描 | `closest_pair()` |
| Nim 判定 | $a_1 \oplus \cdots \oplus a_n$ | `nim_solve()` |
| SG 函数 | mex + 递推 | `sg_function()` |
| Minimax | 递归博弈树 | `TicTacToe.minimax()` |


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/algorithms/topics/geometry-game/code/demo.py`
