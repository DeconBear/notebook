---
title: "信道编码 — exercise.py"
---

# 信道编码 — 练习

<a href="/notebook/code/information/channel-coding/exercise.py" target="_blank" download>Download exercise.py</a>

实现 `syndrome_pos(s0, s1, s2) = s0 + 2*s1 + 4*s2`。在至多一位翻转的假设下，三个校验比特合成 1-index 错误位置；$0$ 表示无错。一般情况下 $0$ 只代表校验通过，不能排除多位错误。

```bash
cd docs/information/channel-coding/code
python exercise.py
```

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/information/channel-coding/code/exercise.py`
