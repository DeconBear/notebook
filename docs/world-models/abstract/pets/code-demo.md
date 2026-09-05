---
title: "PETS — demo.py"
---

# PETS — demo.py 代码详解

<a href="/notebook/code/world-models/abstract/pets/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/world-models/abstract/pets/code
python demo.py
```

玩具一维质点（看 CEM 收缩）+ **倒立摆直立稳定**（概率 MLP 集成 + TS∞ + CEM-MPC，不依赖 Gymnasium）。跑完应看到 `pets_cem_mpc.png` 与 `pets_pendulum.png`。CPU 约 1–2 分钟。


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/world-models/abstract/pets/code/demo.py`
