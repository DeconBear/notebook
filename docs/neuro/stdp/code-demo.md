---
title: "Hebb 与 STDP — demo.py"
---

# Hebb 与 STDP — demo.py 代码详解

<a href="/notebook/code/neuro/stdp/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/neuro/stdp/code
python demo.py
```

## 在讲什么

画出 pairwise STDP 的指数学习窗，再让一对突触以固定 $\Delta t=+10\,\mathrm{ms}$ 重复配对，看权重被推向上界。

## 完整代码

<<< @/neuro/stdp/code/demo.py
