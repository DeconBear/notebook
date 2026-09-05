---
title: "AlphaGo：自我对弈与 MCTS — exercise.py"
---

# AlphaGo：自我对弈与 MCTS — 练习

<a href="/notebook/code/nn-decision/rl/alphago/exercise.py" target="_blank" download>Download exercise.py</a>

补全 `puct_score`：$\mathrm{PUCT}=Q + c\cdot P\cdot\sqrt{N_{\mathrm{parent}}}/(1+N_{\mathrm{child}})$。未访问边（$N=0$）应只靠探索项，同样先验下比已访问边更高。

```bash
cd docs/nn-decision/rl/alphago/code
python exercise.py
```


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/nn-decision/rl/alphago/code/exercise.py`
