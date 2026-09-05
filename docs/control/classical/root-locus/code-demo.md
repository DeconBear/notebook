---
title: "根轨迹 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 根轨迹 — demo.py 代码详解

<a href="/notebook/code/control/classical/root-locus/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/control/classical/root-locus/code
python demo.py
```

CPU、NumPy 即可。一张图 `root_locus.png`：横轴实部、纵轴虚部，颜色是 $K$。植物 $G(s)=1/(s(s+1)(s+3))$，特征多项式 $s^3+4s^2+3s+K$。

## 代码逐段详解

### 第1步：`np.roots` 解特征多项式

$$
s^3+4s^2+3s+K=0
$$

```python
def closed_loop_roots(k):
    """s³ + 4s² + 3s + K = 0 的三个根。"""
    return np.roots([1.0, 4.0, 3.0, k])
```

- **`np.roots(coeff)`**：系数从**最高次**到常数项。`[1, 4, 3, k]` 就是 $1\cdot s^3+4s^2+3s+k$。
- 返回复数 `np.complex128` 数组。实根的虚部是 $0$（可能残留 $10^{-16}$）。
- 不把 $G(s)$ 写成有理函数对象：展开一次就够。

---

### 第2步：扫描 $K$，记下稳定边界

```python
ks = np.linspace(0.0, 40.0, 81)
...
        if np.all(np.real(rts) < 0):
            last_stable_k = k
```

- **`np.linspace(0, 40, 81)`**：$K$ 均匀 81 个点。根轨迹在 $K$ 大时变得稀疏，这是扫点法的代价；Evans 规则画的是连续曲线。
- **`np.all(np.real(rts) < 0)`**：三个根实部都为负才算渐近稳定。`np.real` 取实部；`np.all` 是「每一个都」。
- 网格是离散的，打印的「最大稳定 $K$」是网格上的近似，不是精确 Routh 边界。

内层 `for r in rts` 把三个根都丢进散点列表。同一 $K$ 的三个点颜色相同。

---

### 第3步：散点 + 开环极点

```python
sc = ax.scatter(reals, imags, c=colors, s=18, cmap='viridis', linewidths=0)
ax.plot([0, -1, -3], [0, 0, 0], 'rx', ms=10, mew=2, label='开环极点 K=0')
```

- **`c=colors`**：每个点一个 $K$，配合 `cmap` 做成色条。
- **`'rx'`**：红色叉，正是控制教材里开环极点的记号。`ms` 点大小，`mew` 叉的线宽。
- **`set_aspect('equal')`**：否则虚轴方向会被压扁，共轭对称看不出来。

`np.round(closed_loop_roots(k), 3)`：打印时留三位，复数照样 round。

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/control/classical/root-locus/code/demo.py`
