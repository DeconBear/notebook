---
title: "wm03 Dreamer — demo.py"
---

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# Dreamer — demo.py 代码详解

<a href="/notebook/code/world-models/abstract/dreamer/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/world-models/abstract/dreamer/code
python demo.py
```

先跑倒立摆上的想象 Actor-Critic（`dreamer_pendulum.png`），再短跑离散走廊对照图。与 PETS / LeWM 同一套摆物理；不是完整 DreamerV3。CPU 约 2–3 分钟。

<<< @/world-models/abstract/dreamer/code/demo.py
