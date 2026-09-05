---
title: "机器人运动学 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 机器人运动学 — demo.py 代码详解

<a href="/notebook/code/robotics/kinematics/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/robotics/kinematics/code
python demo.py
```

CPU、NumPy 即可。一张图 `arm_2r.png`：左是工作空间采样 + 同一目标的肘上/肘下构型，右是 $\det J$ 随 $\theta_2$ 过零。杆长 $\ell_1=1$、$\ell_2=0.7$，目标 $(1.1,0.6)$。无物理引擎，纯几何。

## 代码逐段详解

### 第1步：正运动学 `fk`

$$
x=\ell_1\cos\theta_1+\ell_2\cos(\theta_1+\theta_2),\quad
y=\ell_1\sin\theta_1+\ell_2\sin(\theta_1+\theta_2)
$$

```python
def fk(th1, th2):
    x = L1 * np.cos(th1) + L2 * np.cos(th1 + th2)
    y = L1 * np.sin(th1) + L2 * np.sin(th1 + th2)
    return np.array([x, y])
```

- **`th1 + th2`**：第二杆相对基座的绝对角，不是相对角再转一次坐标系。
- **返回长度为 2 的数组**：后面 `fk(*th)` 用星号把 `(th1,th2)` 拆进两个参数。

`joints` 额外给出肘点，供画折线：

```python
def joints(th1, th2):
    p0 = np.array([0.0, 0.0])
    p1 = np.array([L1 * np.cos(th1), L1 * np.sin(th1)])
    p2 = fk(th1, th2)
    return p0, p1, p2
```

$p_1$ 只用 $\theta_1$，与第二杆无关。

---

### 第2步：`ik` — 余弦定理 + 肘号

```python
def ik(x, y, elbow='up'):
    r2 = x * x + y * y
    c2 = (r2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    c2 = np.clip(c2, -1.0, 1.0)
    s2 = np.sqrt(max(0.0, 1.0 - c2 * c2))
    if elbow == 'down':
        s2 = -s2
    th2 = np.arctan2(s2, c2)
    k1 = L1 + L2 * c2
    k2 = L2 * s2
    th1 = np.arctan2(y, x) - np.arctan2(k2, k1)
    return th1, th2
```

- **`r2 = x*x + y*y`**：不用 `hypot` 再平方，少一次开方。余弦定理要的是 $r^2$。
- **`clip(c2, -1, 1)`**：目标略超出工作空间时，浮点会让 $|\cos\theta_2|>1$，`sqrt` 变 NaN。夹紧等于「投影到边界」，不是严格无解检测。
- **`s2 = ±sqrt(1-c2²)`**：肘上默认正正弦，肘下取负。`elbow='up'/'down'` 是字符串，不是布尔，调用 `ik(*target, elbow=name)`。
- **`arctan2(s2, c2)`**：用正弦余弦一起定象限。只 `arccos(c2)` 得不到负 $\theta_2$。
- **`th1 = atan2(y,x) - atan2(k2,k1)`**：$k_1=\ell_1+\ell_2\cos\theta_2$、$k_2=\ell_2\sin\theta_2$ 是把第二杆折进「等效第一杆」后的虚部/实部。先对准目标方位，再扣掉这截偏角。

**为什么 IK 之后立刻 FK？** 终端打印 `FK=p` 应回到 `target`。对不上就是肘号或 `atan2` 参数顺序写反（`atan2` 是 $(y,x)$ 不是 $(x,y)$）。

---

### 第3步：`jacobian`

$$
J=
\begin{pmatrix}
-\ell_1\sin\theta_1-\ell_2\sin(\theta_1+\theta_2) & -\ell_2\sin(\theta_1+\theta_2)\\
\ell_1\cos\theta_1+\ell_2\cos(\theta_1+\theta_2) & \ell_2\cos(\theta_1+\theta_2)
\end{pmatrix}
$$

就是把 `fk` 对 $\theta$ 求导。`s12,c12` 先算好，避免写四遍 `th1+th2`。

```python
    return np.array([
        [-L1 * s1 - L2 * s12, -L2 * s12],
        [L1 * c1 + L2 * c12, L2 * c12],
    ])
```

平面 2R 有 $\det J=\ell_1\ell_2\sin\theta_2$。右图并不用闭式，而是 `np.linalg.det(jacobian(0.4, t))`，让你看见「矩阵算出来的行列式」在 $\theta_2=0,\pm180^\circ$ 过零。

---

### 第4步：主程序 — 两解、采样、奇异扫描

```python
    target = np.array([1.1, 0.6])
    for name in ('up', 'down'):
        th = ik(*target, elbow=name)
        p = fk(*th)
        J = jacobian(*th)
        ...
        print(f'肘{name}: θ={np.rad2deg(th)} deg  FK={p}  detJ={np.linalg.det(J):.3f}')
```

- **`*target`**：把数组拆成 `ik(1.1, 0.6, elbow=...)`。
- **`np.rad2deg`**：打印用度，计算全程弧度。`sin`/`cos` 吃度会错。

工作空间：

```python
    th1s = np.linspace(-np.pi, np.pi, 80)
    th2s = np.linspace(-2.4, 2.4, 60)
    ...
        for a in th1s[::4]:
            for b in th2s[::3]:
                pts.append(fk(a, b))
```

- **`θ_2` 不到 $\pm\pi$**：避免画成完全折叠的稠密一团，仍覆盖主要圆环。
- **`[::4]`**：每隔 4 个取一个，散点别太密。这是可达采样不是边界提取。

左图 `plot([p0[0], p1[0], p2[0]], ...)` 把三个点连成折线。`scatter(*target)` 把 $x,y$ 拆进 `scatter(x,y)`。`set_aspect('equal')` 否则手臂比例畸变，肘形看起来假。

右图：

```python
    th2_line = np.linspace(-np.pi, np.pi, 200)
    dets = [np.linalg.det(jacobian(0.4, t)) for t in th2_line]
    axes[1].plot(np.rad2deg(th2_line), dets, color='#8E44AD')
```

固定 $\theta_1=0.4$ 只为让 $J$ 的第一列不是零姿态；过零位置仍由 $\theta_2$ 决定。列表推导把 200 个标量 `det` 收齐。`axhline(0)` 标奇异。

---

### 关键概念速查表

| 概念 | 数学 / 直觉 | 代码 |
|------|-------------|------|
| FK | 角 → $(x,y)$ | `fk` |
| 肘点 | 第一杆末端 | `joints` |
| IK 余弦 | $\cos\theta_2=(r^2-\ell_1^2-\ell_2^2)/(2\ell_1\ell_2)$ | `c2 = ...` |
| 肘号 | $\sin\theta_2$ 的符号 | `elbow == 'down'` |
| `arctan2` | 双参数定象限 | `th2`,`th1` |
| `clip` | 防 $\cos$ 越界 | 工作空间边缘 |
| $J$ | $\partial(x,y)/\partial\theta$ | `jacobian` |
| 奇异 | $\det J=0$ | 右图过零 |
| `*th` | 元组拆参数 | `fk(*th)` |
| `rad2deg` | 只用于显示 | 打印与右图横轴 |

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/robotics/kinematics/code/demo.py`
