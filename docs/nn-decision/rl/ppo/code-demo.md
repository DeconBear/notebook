---
title: "PPO：近端策略优化 — demo.py"
---

# PPO：近端策略优化 — demo.py 代码详解

<a href="/notebook/code/nn-decision/rl/ppo/demo.py" target="_blank" download>Download demo.py</a>

在「走廊平衡」玩具 MDP 上对比 REINFORCE 与 PPO-Clip + GAE。重点看 `ppo-02-clip-curves.png`：$\hat{A}>0$ 时目标在 $1+\varepsilon$ 封顶。训练曲线里 PPO 的回报应更稳；右图 clip fraction 是比率越出盒子的比例。

```bash
cd docs/nn-decision/rl/ppo/code
python demo.py
```

## 完整代码

<<< @/nn-decision/rl/ppo/code/demo.py
