---
title: "PID 控制 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# PID 控制 — demo.py 代码详解

<a href="/notebook/code/control/classical/pid/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/control/classical/pid/code
python demo.py
```

CPU、NumPy 即可。一张图 `pid_step.png`：左是位置阶跃，右是控制力。四条曲线：开环、$K_p=8$ 的 P、$K_p=8,K_d=4$ 的 PD、再加 $K_i=3$ 的 PID。植物是 $m\ddot x+c\dot x+kx=u$，参考 $r=1$。没有拉普拉斯、没有 `scipy.signal`，积分与 PID 都手写。

## 代码逐段详解

### 第1步：植物参数与步长

```python
M, C, K = 1.0, 0.4, 2.0
DT = 0.01
T_END = 8.0
R = 1.0  # 阶跃目标位置
```

- **`M, C, K`**：质量、阻尼、刚度。与弹簧力 $Kx$ 共用字母 `K`，后面 PID 增益用小写 `kp,ki,kd`，避免撞名。
- **`DT = 0.01`**：积分步长。太大欧拉会漂；太小只是变慢。$8\,\mathrm{s}$ 仿真 → `n = int(T_END / DT)` 步。
- **`R`**：参考位置，不是现代控制那章的代价矩阵。开环稳态力用 `R * K`（这里的 `K` 是弹簧）。

`np.random.seed(42)` 写了，但本脚本无随机数，轨迹完全确定。

---

### 第2步：`plant_step` — 半隐式欧拉

$$
\ddot x = (u - c\dot x - kx)/m
$$

```python
def plant_step(x, v, u):
    """欧拉积分：ẍ = (u - c v - k x) / m。"""
    a = (u - C * v - K * x) / M
    v = v + DT * a
    x = x + DT * v
    return x, v
```

- **先更新 `v` 再更新 `x`**：用的是新速度。这是半隐式（辛）欧拉，比「两处都用旧速度」的显式欧拉更稳一点，仍不是 RK4。
- **语法 `C * v`**：标量乘法。状态不是数组，是两个 `float`，调试时比向量短。
- **为什么不把弹簧写进控制器？** 植物自己含 $kx$。开环必须提供 `u = R*K` 才能在稳态抵消它；闭环则让 PID 自己长出力。

---

### 第3步：`simulate` — 开环 vs PID

```python
def simulate(kp=0.0, ki=0.0, kd=0.0, open_loop=False):
    n = int(T_END / DT)
    xs, us, ts = [], [], []
    x, v, integ, e_prev = 0.0, 0.0, 0.0, 0.0
    for i in range(n):
        t = i * DT
        if open_loop:
            u = R * K  # 稳态力刚好抵弹簧，但暂态全靠植物
        else:
            e = R - x
            integ += e * DT
            de = (e - e_prev) / DT
            u = kp * e + ki * integ + kd * de
            e_prev = e
        x, v = plant_step(x, v, u)
        xs.append(x)
        us.append(u)
        ts.append(t)
    return np.array(ts), np.array(xs), np.array(us)
```

- **默认增益全 0**：若忘了传 `kp` 又没开开环，`u` 恒为 0，质量会被弹簧拉回原点——那是第三条对照，本脚本没画。
- **`open_loop`**：不看 $e$，力一步到位。暂态是欠阻尼二阶系统的自由响应（外力相当于改平衡点）。
- **`e = R - x`**：负反馈。写成 `x - R` 会正反馈，曲线向外炸。
- **`integ += e * DT`**：矩形积分。没有抗饱和（anti-windup）；`T_END` 短、`ki` 不大，尚可。
- **`de = (e - e_prev) / DT`**：后向差分。第一步 `e_prev=0`，若 $x_0=0$ 则 $e=1$，微分项会冲一下——真实 D 还常加低通，这里故意裸着，好对照曲线。
- **`**kw` 调用**：`main` 里 `dict(kp=8.0)` 解包进关键字参数，开环那条只传 `open_loop=True`。

**为什么积分用位置误差而不是力误差？** PID 的「静差」指 $x$ 到不了 $r$。弹簧会在 $x=r$ 时仍要 $u=kr$，纯 P 的 $u=k_p(r-x)$ 在 $x\to r$ 时 $u\to 0$，稳态方程 $0=kr$ 矛盾，所以必留静差。I 项把这份力慢慢攒出来。

---

### 第4步：超调与静差

```python
def overshoot(x):
    return float(max(0.0, np.max(x) - R) / R * 100.0)

def ss_error(x):
    return float(abs(x[-1] - R))
```

- **超调**：峰值超出 $r$ 的百分比。从未超过则 $0$，避免负超调干扰读数。
- **`x[-1]`**：最后一样本当「稳态」。严格应收尾段平均；8 秒对这组参数够用。
- **`float(...)`**：`np.max` 可能是 numpy 标量，打印时统一成 Python `float`。

---

### 第5步：四条对照与双轴图

```python
cases = [
    ('开环', dict(open_loop=True)),
    ('P  kp=8', dict(kp=8.0)),
    ('PD kp=8 kd=4', dict(kp=8.0, kd=4.0)),
    ('PID kp=8 ki=3 kd=4', dict(kp=8.0, ki=3.0, kd=4.0)),
]
```

同一植物、同一 $r$，只改控制器。标签字符串里写出增益，图例才读得懂。

```python
t, x, u = simulate(**kw)
axes[0].plot(t, x, lw=2, label=name)
axes[1].plot(t, u, lw=1.4, label=name)
```

- **`**kw`**：字典拆成 `simulate(open_loop=True)` 或 `simulate(kp=8.0, kd=4.0)`。`ki` 缺省 0。
- **`lw`**：线宽。位置更粗，力更细，避免右图糊成一块。
- **`axhline(R, ...)`**：参考水平线。没有它很难判断静差。

保存到章节 `images/pid_step.png`（脚本相对路径 `../images`），不是 CWD。

---

### 关键概念速查表

| 概念 | 数学 / 直觉 | 代码 |
|------|-------------|------|
| 植物 | $m\ddot x+c\dot x+kx=u$ | `plant_step` |
| 半隐式欧拉 | 先 $v$ 后 $x$ | `v = v + DT*a` |
| 开环 | $u=kr$ | `u = R * K` |
| 误差 | $e=r-x$ | `e = R - x` |
| PID | $K_p e+K_i\int e+K_d\dot e$ | `kp*e + ki*integ + kd*de` |
| 超调 | $(\max x-r)/r$ | `overshoot` |
| 静差 | $\|x(\infty)-r\|$ | `ss_error` |
| `**kw` | 字典变关键字 | `simulate(**kw)` |

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/control/classical/pid/code/demo.py`
