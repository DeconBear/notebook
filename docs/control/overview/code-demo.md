---
title: "控制论导论 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 控制论导论 — demo.py 代码详解

<a href="/notebook/code/control/overview/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/control/overview/code
python demo.py
```

CPU、NumPy 即可。一张图 `overview_loop.png`：左是质量-弹簧的开环 vs 比例闭环，右是二阶系统四个阻尼比 $\zeta$ 的阶跃。没有 `scipy.signal`，植物用半隐式欧拉往前走。

## 代码逐段详解

### 第1步：植物参数

```python
M, C, KSPRING = 1.0, 0.4, 2.0
DT = 0.01
T_END = 8.0
R = 1.0
KP = 8.0
WN = 2.0
```

- **`KSPRING`**：弹簧刚度。后面 PID 增益用 `KP`，避免和弹簧挤同一个 `K`。
- **`R`**：参考位置。开环稳态力必须是 `R * KSPRING` 才能抵弹簧，否则质量会被拉回 $0$。
- **`WN`**：右图二阶系统的自然频率 $\omega_n$，四个 $\zeta$ 共用，只改阻尼。

`np.random.seed(42)` 写了，但本脚本无随机数，轨迹完全确定。

---

### 第2步：`mass_spring` — 开环力 vs $u=K_p e$

$$
m\ddot x + c\dot x + kx = u
$$

```python
def mass_spring(open_loop=True):
    n = int(T_END / DT)
    x, v = 0.0, 0.0
    xs, ts = [], []
    for i in range(n):
        if open_loop:
            u = R * KSPRING
        else:
            u = KP * (R - x)
        a = (u - C * v - KSPRING * x) / M
        v = v + DT * a
        x = x + DT * v
        xs.append(x)
        ts.append(i * DT)
    return np.array(ts), np.array(xs)
```

- **先更新 `v` 再更新 `x`**：半隐式欧拉，比两处都用旧速度更稳一点。
- **开环 `u = R * KSPRING`**：稳态对，暂态全交给 $m,c,k$ 自己晃。
- **闭环只乘比例**：没有积分，有弹簧时末端往往仍差一点——这是导论故意留的缺口，[PID 章](/control/classical/pid/)用 $K_i$ 啃掉它。

语法：`n = int(T_END / DT)` 把秒数变成步数；`xs.append(x)` 先用 Python 列表，最后 `np.array` 一次转，避免循环里反复扩数组。

---

### 第3步：`second_order_step` — 标准二阶

$$
\ddot y + 2\zeta\omega_n\dot y + \omega_n^2 y = \omega_n^2 r
$$

改写成 $\ddot y = \omega_n^2(r-y)-2\zeta\omega_n\dot y$，和弹簧植物是同一类积分。

```python
a = wn ** 2 * (R - y) - 2.0 * zeta * wn * v
```

- **`zeta < 0`**：阻尼项变成加油，能量往上长，曲线发散。右图把它画出来，是为了让「右半平面极点」在时域里看得见。
- **`wn ** 2`**：Python 的幂是 `**`，不是 `^`（`^` 是按位异或）。

---

### 第4步：画图与打印

```python
zetas = [(1.4, '过阻尼 ζ=1.4'), (0.5, '欠阻尼 ζ=0.5'),
         (0.0, '无阻尼 ζ=0'), (-0.15, '负阻尼 ζ=-0.15')]
```

元组列表：第一个元素给积分，第二个给图例。`y.max()` 是 NumPy 数组的峰值，欠阻尼会大于 $r$，过阻尼不会。

图存到章节 `images/overview_loop.png`：路径由 `__file__` 往上拼，从哪启动脚本都写到对的地方。

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/control/overview/code/demo.py`
