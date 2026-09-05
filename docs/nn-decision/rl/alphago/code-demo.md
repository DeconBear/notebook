---
title: "AlphaGo：自我对弈与 MCTS — demo.py"
---

# AlphaGo：自我对弈与 MCTS — demo.py 代码详解

<a href="/notebook/code/nn-decision/rl/alphago/demo.py" target="_blank" download>Download demo.py</a>

井字棋上的 PUCT-MCTS：均匀先验 + 随机滚出，不训练深度网络。看两件事——空盘根节点的访问次数 $N$（应对应「中心格更常被搜」），以及执 X 对随机 / 中心贪心的胜率。

```bash
cd docs/nn-decision/rl/alphago/code
python demo.py
```

运行后会在 `images/` 写出正文用的四张示意图，以及 `mcts_tic_tac_toe.png`。


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/nn-decision/rl/alphago/code/demo.py`
