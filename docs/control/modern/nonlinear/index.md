---
title: "非线性与李雅普诺夫"
order: 40
---
# 李雅普诺夫：不必线性化也能谈稳定

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

> 极点、Routh、Bode 都要求线性（或先在工作点线性化）。单摆 $\ddot\theta+(g/\ell)\sin\theta=0$ 不是线性的。李雅普诺夫的办法是找一个能量函数 $V$：**原点最低、沿轨迹不增加**，就能断定稳定，而不用把微分方程解出来。线性系统里「左半平面极点」是这件事的特例。状态方程见 [状态空间](/control/modern/state-space/)。

---

## 一、能量碗

单摆（向下为 $\theta=0$），带枢轴粘滞阻尼 $b>0$：

$$
m\ell^2\ddot\theta + b\dot\theta + mg\ell\sin\theta = 0.
$$

取

$$
V(\theta,\dot\theta)=\tfrac12 m(\ell\dot\theta)^2 + m g \ell(1-\cos\theta).
$$

$V\ge 0$，仅在 $(\theta,\dot\theta)=(0,0)$（模 $2\pi$）为 $0$。沿轨迹

$$
\dot V = -b\dot\theta^2 \le 0.
$$

无阻尼 $b=0$ 时 $\dot V=0$，能量守恒，相图是闭合轨道；有阻尼时 $V$ 严格下降（除非已经停住），轨迹螺旋进原点。

![李雅普诺夫能量碗](./images/ctrl-lyapunov.png)

> **图解说明**：左是单摆；中是 $V$ 这只碗，轨迹只能往低处走；右是无阻尼闭轨 vs 有阻尼螺旋。不必把 $\theta(t)$ 写成公式。

这叫**李雅普诺夫稳定 / 渐近稳定**的充分条件（还要 $V$ 正定、径向无界等技术假设；本课用物理能量，条件自然满足）。倒立摆在上平衡点 $V$ 不是正定的——那里要另找 $V$，或先线性化再 [LQR](/control/modern/lqr/)。

---

## 二、代码在做什么

`demo.py` 积两组单摆：无阻尼 vs 有阻尼，同一初值。左图 $\theta(t)$，右图 $V(t)$。无阻尼 $V$ 几乎水平（欧拉会有一点数值泄漏）；有阻尼 $V$ 单调下降，$\theta$ 衰减振荡。

![单摆能量](./images/lyapunov_pendulum.png)

终端打印末端 $V$。阻尼那条应明显更小。相图不画了，以免和右图能量重复；你若改 `demo.py` 把 $(\theta,\dot\theta)$ 撒上去，闭轨 vs 螺旋会更直观。

---

## 三、小结

| 概念 | 一句话 |
|------|--------|
| $V$ | 能量碗：远点高、原点最低 |
| $\dot V\le 0$ | 沿轨迹不爬升 ⇒ 稳定 |
| $\dot V<0$ | 还往下漏 ⇒ 渐近稳定 |
| 线性特例 | 左半平面极点 ⇔ 存在二次 $V=x^\top Px$ |
| 下游 | 机器人非线性动力学见 [动力学](/robotics/dynamics/) |

> 现代控制四章到此。回到 [控制论导论](/control/overview/) 或接 [运动学](/robotics/kinematics/)。世界模型里的「想象滚动」是另一条不要求 $V$ 的规划路。

## 📥 Code

| File | View | Download |
|------|------|----------|
| demo.py | [Open](./code-demo) | <a href="/notebook/code/control/modern/nonlinear/demo.py" target="_blank" download>Download</a> |
| exercise.py | [Open](./code-exercise) | <a href="/notebook/code/control/modern/nonlinear/exercise.py" target="_blank" download>Download</a> |

## 参考

1. Khalil, *Nonlinear Systems*（李雅普诺夫定理）
2. Åström & Murray, *Feedback Systems*
