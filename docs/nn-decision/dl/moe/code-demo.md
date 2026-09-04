---
title: "混合专家 MoE — demo.py"
---

# 混合专家 MoE — demo.py 代码详解

<a href="/notebook/code/nn-decision/dl/moe/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/nn-decision/dl/moe/code
python demo.py
```

玩具复现三件事：Top-2 路由、有/无负载均衡时的专家使用率、以及 MoE 分类边界。重点看左图柱状是否被均衡损失「摊平」。

## 完整代码

<<< @/nn-decision/dl/moe/code/demo.py
