---
title: "非线性与李雅普诺夫 — exercise.py"
---

# 非线性与李雅普诺夫 — 练习

<a href="/notebook/code/control/modern/nonlinear/exercise.py" target="_blank" download>Download exercise.py</a>

实现 `pendulum_V(theta, omega, m, l, g) = 0.5*m*(l*omega)**2 + m*g*l*(1 - cos(theta))`。原点应为 $0$；纯转动动能为 $\tfrac12 m(\ell\omega)^2$；倒立且静止时势能为 $2mgl$。

```bash
cd docs/control/modern/nonlinear/code
python exercise.py
```

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/control/modern/nonlinear/code/exercise.py`
