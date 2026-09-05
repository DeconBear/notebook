---
title: "机构学 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 机构学 — demo.py 代码详解

<a href="/notebook/code/robotics/mechanisms/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/robotics/mechanisms/code
python demo.py
```

CPU、NumPy 即可。一张图 `fourbar.png`：偶联点（连杆中点）扫出的封闭曲线，并叠一帧 $\theta\approx 0.7$ 的四杆快照。杆长机架 / 曲柄 / 连杆 / 摇杆 $=1.4,\;0.45,\;1.1,\;0.9$。无仿真引擎：每步用两圆相交解环闭合。

## 代码逐段详解

### 第1步：杆长

```python
L0, L1, L2, L3 = 1.4, 0.45, 1.1, 0.9
```

$L_0$ 是机架 $O$ 到 $D$。曲柄最短，满足 Grashof，预期曲柄整周转、摇杆摆动。改短 $L_2$ 可能出现某段 $\theta$ 无交点，`coupler_point` 返回 `None`，轨迹缺口。

---

### 第2步：`coupler_point` — 两圆相交

曲柄端 $A=L_1(\cos\theta,\sin\theta)$，摇杆地铰 $D=(L_0,0)$。求 $B$ 使 $|B-A|=L_2$、$|B-D|=L_3$。

```python
def coupler_point(theta):
    A = np.array([L1 * np.cos(theta), L1 * np.sin(theta)])
    D = np.array([L0, 0.0])
    d = D - A
    dist = np.linalg.norm(d)
    if dist < 1e-9 or dist > L2 + L3 or dist < abs(L2 - L3):
        return None, None
    a = (L2 ** 2 - L3 ** 2 + dist ** 2) / (2 * dist)
    h = np.sqrt(max(L2 ** 2 - a * a, 0.0))
    mid = A + a * d / dist
    n = np.array([-d[1], d[0]]) / dist
    B = mid + h * n
    return A, B
```

- **`dist`**：两圆心距离。大于 $L_2+L_3$ 或小于 $|L_2-L_3|$ 则不相交（三角形不等式）。
- **`dist < 1e-9`**：圆心重合，方向 $d$ 没定义。
- **`a`**：从 $A$ 沿 $AD$ 走到公弦垂足的有向距离。余弦定理的一维版。
- **`h`**：半弦长。`max(...,0)` 吃掉一点点负的浮点。
- **`n = (-d_y, d_x)/dist`**：$d$ 逆时针转 $90^\circ$ 的单位法向。`B = mid + h n` 是叉积为正的那一支装配；另一支是 `-h`（本函数不返回）。
- **返回 `None, None`**：调用方 `continue` 跳过该 $\theta$，不要拿去画折线。

这不是数值优化，是解析几何。闭链 IK 在平面四杆上刚好有这套圆交。

---

### 第3步：Gruebler 打印

```python
    N = 3 * 1 + 4  # 演示 Gruebler：平面 N=4, J1=4 → M=3(4-1)-2*4=1
    print(f'Gruebler: M=3(N-1)-2 J1 = 3(4-1)-2*4 = {3 * (4 - 1) - 2 * 4}  (期望 1)')
```

- **公式** $M=3(N-1)-2J_1$，含机架 $N=4$、$J_1=4$ → $1$。练习 `gruebler_planar(4, 4)` 必须返回 `1`。
- **变量 `N = 3*1+4`**：只是注释性算术，后面打印并没有用这个 `N`。不要改成活动构件数 $3$ 却仍套 $(N-1)$，会得到错误自由度。

---

### 第4步：扫一圈并取快照

```python
    thetas = np.linspace(0, 2 * np.pi, 180)
    curve = []
    snapshot = None
    for th in thetas:
        A, B = coupler_point(th)
        if B is None:
            continue
        P = 0.5 * (A + B)  # 连杆中点当「偶联点」
        curve.append(P)
        if snapshot is None and abs(th - 0.7) < 0.05:
            snapshot = (th, A, B, P)
```

- **180 个 $\theta$**：封闭曲线够光滑。`linspace` 含端点，$0$ 与 $2\pi$ 是同一姿态，首尾点重合无妨。
- **`P = 0.5*(A+B)`**：连杆中点。真机构偶联点可以焊在连杆任意处，轨迹形状会变，自由度不变。
- **快照窗口** `|θ-0.7|<0.05`：只存第一次命中，避免同一弧上重复。`0.7\,\mathrm{rad}` 只为构图，不是特殊奇异角。
- **`curve = np.array(curve)`**：之后 `curve[:,0]` 画紫线。

作图把 $O\to A\to B\to D\to O$ 连成一圈，黑点 $O$、$D$ 在水平机架上。`set_aspect('equal')` 否则矩形看起来像平行四边形。

---

### 关键概念速查表

| 概念 | 数学 / 直觉 | 代码 |
|------|-------------|------|
| Gruebler | $3(N-1)-2J_1$ | 打印式；练习同公式 |
| 曲柄端 | $L_1(\cos\theta,\sin\theta)$ | `A = ...` |
| 环闭合 | 两圆交 | `coupler_point` |
| 不相交 | 三角形不等式 | `return None, None` |
| 装配分支 | $\pm h n$ | 只取 `+h` |
| 法向 | $d$ 转 $90^\circ$ | `n = [-d[1], d[0]]` |
| 偶联点 | 连杆固连点 | `P = 0.5*(A+B)` |
| `is None` | 跳过死点 | `continue` |
| 快照 | 一帧杆件 | `abs(th-0.7)<0.05` |

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/robotics/mechanisms/code/demo.py`
