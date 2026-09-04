---
title: "GRPO：组相对策略优化 — exercise.py"
---

# GRPO：组相对策略优化 — 练习

<a href="/notebook/code/nn-decision/rl/grpo/exercise.py" target="_blank" download>Download exercise.py</a>

`group_advantages` 做组内 z-score，全相同则返回 0；`grpo_clipped_objective` 与 [PPO 练习](/nn-decision/rl/ppo/code-exercise) 是同一套 $\min$ + $\mathrm{clip}$，只是一条输出的所有 token 共用同一个 $\hat{A}_i$。

```bash
cd docs/nn-decision/rl/grpo/code
python exercise.py
```

## 完整代码

<<< @/nn-decision/rl/grpo/code/exercise.py
