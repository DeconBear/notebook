---
title: "非线性与李雅普诺夫 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 非线性与李雅普诺夫 — demo.py 代码详解

<a href="/notebook/code/control/modern/nonlinear/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/control/modern/nonlinear/code
python demo.py
```

CPU、NumPy 即可。一张图 `lyapunov_pendulum.png`：左是 $\theta(t)$，右是 $V(t)$。两条轨迹同一初值 $\theta=1.2,\omega=0$，只改阻尼 $b$。没有小角度 $\sin\theta\approx\theta$，完整非线性。

## 代码逐段详解

### 第1步：能量

$$
V=\tfrac12 m(\ell\omega)^2 + mgl(1-\cos\theta)
$$

```python
def energy(theta, omega):
    kinetic = 0.5 * M * (L * omega) ** 2
    potential = M * G * L * (1.0 - np.cos(theta))
    return kinetic + potential
```

- **$(1-\cos\theta)$**：最低点势能为 $0$，倒立 $\theta=\pi$ 时势能为 $2mgl$。不要用 $-\cos\theta$ 还不加常数——$V$ 可以差一个常数，但打印「是不是在减」会难看。
- **`np.cos`**：$\theta$ 是标量时返回标量；后面若传入数组也会逐元素算。

原点 $V(0,0)=0$，别处为正：正定。这就是「碗底」。

---

### 第2步：非线性 ODE

$$
\dot\omega = -\frac{g}{\ell}\sin\theta - \frac{b}{m\ell^2}\omega
$$

```python
wdot = -(G / L) * np.sin(th) - (b / (M * L ** 2)) * w
w = w + DT * wdot
th = th + DT * w
```

- **先更新 $\omega$ 再更新 $\theta$**：又是半隐式欧拉。无阻尼时 $V$ 仍会因数值耗散略降，这是积分器误差，不是物理阻尼。步长 `DT=0.01` 把泄漏压在可接受范围。
- **`b=0` vs `b=0.6`**：解析上 $\dot V=-b\omega^2$。代码不计算 $\dot V$，只把 $V(t)$ 画出来让你看见单调。

`L ** 2`：幂。分母 $m\ell^2$ 是转动惯量。

---

### 第3步：读图

无阻尼 $\theta$ 几乎等幅晃（略衰减来自欧拉）；有阻尼振幅收拢。右图 $V$：灰线接近水平，橙线往下走。若你把 $b$ 改成负数，$V$ 会往上爬——那是在给摆加油，李雅普诺夫条件破了。

`TH0, W0 = 1.2, 0.0` 是弧度，大约 $69^\circ$，小角度近似已经不准，这正是要用 $V$ 而不是极点的原因。

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/control/modern/nonlinear/code/demo.py`
