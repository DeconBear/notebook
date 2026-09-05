---
title: "LeWM — demo.py"
---

# LeWM — demo.py 代码详解

<a href="/notebook/code/world-models/abstract/lewm/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/world-models/abstract/lewm/code
python demo.py
```

二维质点（MSE + 高斯代理正则）+ **倒立摆状态嵌入**（随机投影 SIGReg + 对准直立嵌入的 CEM；火柴杆只做可视化）。跑完应看到 `lewm_cem_mpc.png` 与 `lewm_pendulum.png`。

## 完整代码

<<< @/world-models/abstract/lewm/code/demo.py
