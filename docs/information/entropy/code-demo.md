---
title: "熵与条件熵 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 熵与条件熵 — demo.py 代码详解

<a href="/notebook/code/information/entropy/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/information/entropy/code
python demo.py
```

CPU、NumPy 即可。一张图 `entropy_joint.png`：左是 $2\times 2$ 联合热力，右是 $H(X),H(Y),H(X,Y),H(X\mid Y),I$ 五根柱。

## 代码逐段详解

### 第1步：从联合表拆边缘

```python
Pxy = Pxy / Pxy.sum()
px = Pxy.sum(axis=1)
py = Pxy.sum(axis=0)
```

- **行是 $x$、列是 $y$**。`axis=1` 对每行求和 → $p(x)$；`axis=0` 对每列求和 → $p(y)$。
- 先归一化，避免表里的数加起来不是 $1$。

---

### 第2步：条件熵是差，不是再扫一遍

$$
H(X\mid Y)=H(X,Y)-H(Y)
$$

```python
hx_given_y = hxy - hy
ixy = hx - hx_given_y
```

不必对每个 $y$ 再算 $H(X\mid Y=y)$ 再平均——链规则已经给了同一结果。互信息是 $H(X)$ 减掉这个剩余。

`entropy_bits` 对联合用 `Pxy.ravel()` 把 4 个格子当成一个 4 元分布，因为 $H(X,Y)$ 就是那 4 个质量的熵。

链规则误差应在 $10^{-15}$ 量级，只是浮点。

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/information/entropy/code/demo.py`
