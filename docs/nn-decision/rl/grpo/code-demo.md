---
title: "GRPO：组相对策略优化 — demo.py"
---

# GRPO：组相对策略优化 — demo.py 代码详解

<a href="/notebook/code/nn-decision/rl/grpo/demo.py" target="_blank" download>Download demo.py</a>

口算题玩具：每道 $a+b$ 采 $G$ 个答案，组内 z-score 当优势，再套 PPO 的 clip。对比无基线 REINFORCE。右图是「一组全对/全错被跳过」的比例——GRPO 没组内方差就没有梯度。

```bash
cd docs/nn-decision/rl/grpo/code
python demo.py
```

## 完整代码

<<< @/nn-decision/rl/grpo/code/demo.py
