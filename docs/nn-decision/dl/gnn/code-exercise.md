---
title: "图神经网络 GNN — exercise.py"
---

# 图神经网络 — 练习

<a href="/notebook/code/nn-decision/dl/gnn/exercise.py" target="_blank" download>Download exercise.py</a>

1. `gcn_normalize(A)`：对称归一化 \(\tilde D^{-1/2}(A+I)\tilde D^{-1/2}\)。
2. `dst_softmax(scores, dst, n)`：每个目标节点上对入射边做 softmax（GAT）。

```bash
cd docs/nn-decision/dl/gnn/code
python exercise.py
```

## 源码位置

`docs/nn-decision/dl/gnn/code/exercise.py`
