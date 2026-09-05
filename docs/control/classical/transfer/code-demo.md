---
title: "传递函数与时域响应 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 传递函数与时域响应 — demo.py 代码详解

<a href="/notebook/code/control/classical/transfer/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/control/classical/transfer/code
python demo.py
```

CPU、NumPy 即可。一张图 `tf_step.png`：同一 $\omega_n=3$，四个 $\zeta$ 的单位阶跃。微分方程是 $G(s)$ 的时域原形，不用符号拉普拉斯。

## 代码逐段详解

### 第1步：常数

```python
WN = 3.0
DT = 0.005
T_END = 6.0
R = 1.0
```

- **`WN`**：$\omega_n$。越大振荡越密（欠阻尼时）。四个 $\zeta$ 共用，才能公平对比「只改阻尼」。
- **`DT = 0.005`**：比导论更小。欠阻尼 $\zeta=0.15$ 晃得快，步长大了峰值会偏。

---

### 第2步：积标准二阶

$$
\ddot y = \omega_n^2(r-y) - 2\zeta\omega_n\dot y
$$

```python
def step_response(zeta, wn=WN):
    n = int(T_END / DT)
    y, v = 0.0, 0.0
    ys = []
    for _ in range(n):
        a = wn ** 2 * (R - y) - 2.0 * zeta * wn * v
        v = v + DT * a
        y = y + DT * v
        ys.append(y)
    t = np.arange(n) * DT
    return t, np.array(ys)
```

- **`np.arange(n) * DT`**：长度为 $n$ 的时间轴，等价于导论里循环里 `ts.append`。
- **默认参数 `wn=WN`**：调用时可改自然频率；本 demo 不改。
- 零初始 $y=\dot y=0$，对应传递函数「零状态响应」。

---

### 第3步：超调百分比

```python
def overshoot_pct(y):
    return float(max(0.0, np.max(y) - R) / R * 100.0)
```

- **`np.max(y)`**：全程峰值。过阻尼峰值就是末端，超调被 `max(0.0, …)` 截成 $0$。
- **`float(...)`**：NumPy 标量转成 Python `float`，打印更干净。
- 欠阻尼解析近似 $\sigma\approx\exp(-\zeta\pi/\sqrt{1-\zeta^2})$，代码故意不用，避免「公式对了但积分错了」被公式掩盖。

`rf'$\zeta={z}$'`：matplotlib 图例里的数学模式；`rf` 是 raw+f-string，反斜杠不用加倍。

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/control/classical/transfer/code/demo.py`
