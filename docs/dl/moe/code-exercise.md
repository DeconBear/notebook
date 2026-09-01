---
title: "混合专家 MoE — exercise.py"
---

# 混合专家 MoE — 练习

<a href="/notebook/code/dl/moe/exercise.py" target="_blank" download>Download exercise.py</a>

实现 `topk_gate(probs, k)`：选出概率最大的 $k$ 个专家，并把这 $k$ 个概率重新归一化成门控权重。

```bash
cd docs/dl/moe/code
python exercise.py
```

## 完整代码

<<< @/dl/moe/code/exercise.py
