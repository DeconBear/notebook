---
title: "PPO：近端策略优化 — exercise.py"
---

# PPO：近端策略优化 — 练习

<a href="/notebook/code/nn-decision/rl/ppo/exercise.py" target="_blank" download>Download exercise.py</a>

两处 TODO：`clipped_surrogate`（正文 $L^{\mathrm{CLIP}}$）和 `compute_gae`（从后往前累加 $\delta_t$）。$\lambda=1$ 时应得到 Monte Carlo 回报；$\lambda=0$ 应退回单步 TD。

```bash
cd docs/nn-decision/rl/ppo/code
python exercise.py
```


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/nn-decision/rl/ppo/code/exercise.py`
