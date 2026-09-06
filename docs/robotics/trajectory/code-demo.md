---
title: "轨迹规划 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 轨迹规划 — demo.py 代码详解

<a href="/notebook/code/robotics/trajectory/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/robotics/trajectory/code
python demo.py
```

一张图 `traj_joint.png`：关节角、角速度模、FK 末端路径。虚线 = 关节线性插值；实线 = 起停速度为 0 的三次。CPU、NumPy 即可。

## 代码逐段详解

### 第1步：线性插值

```python
def lerp(q0, q1, t, T):
    s = t / T
    return q0 + s * (q1 - q0)
```

$s\in[0,1]$ 是归一化时间。向量加法在 **NumPy 数组** 上逐关节进行。角速度恒为 $(q_1-q_0)/T$，所以 `|q̇|` 是水平线。

### 第2步：smoothstep 三次

$$
q(s)=q_0+s^2(3-2s)\,(q_1-q_0)
$$

```python
a = s * s * (3.0 - 2.0 * s)
return q0 + a * (q1 - q0)
```

对 $s$ 求导：$da/ds=6s-6s^2$，再除以 $T$ 变成对 $t$ 的导数。$s=0,1$ 时 $da/ds=0$，所以端点速度为 0。$s=0.5$ 时 $da/ds=1.5$，峰值角速度是线性情形的 $1.5$ 倍——demo 打印的两个 `|q̇|` 应对上这个倍数。

练习 `smoothstep(s)` 就是 `s*s*(3-2*s)`。$s=0.5$ 应回到 $0.5$（对称）。

### 第3步：用 FK 看手

```python
ee_lin = np.array([fk(*q) for q in q_lin])
```

两条手路径都弯，而且不重合：时间分配不同，非线性 FK 走出不同的参数曲线。只有笛卡尔规划才能强迫手走直线——本文件不做 IK。

`scatter` 黑圆 / 黑方块是手的起终点，应对齐两条曲线的端点。

## 源码位置

`docs/robotics/trajectory/code/demo.py`
