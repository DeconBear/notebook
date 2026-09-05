---
title: "率失真 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 率失真 — demo.py 代码详解

<a href="/notebook/code/information/rate-distortion/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/information/rate-distortion/code
python demo.py
```

CPU、NumPy 即可。一张图 `rd_binary.png`：公平比特、汉明失真下的 $R(D)=1-h_2(D)$。

## 代码逐段详解

### 第1步：与 BSC 容量同一函数

```python
def R_D(D):
    D = np.asarray(D, dtype=float)
    out = np.zeros_like(D, dtype=float)
    mask = D < 0.5
    out[mask] = 1.0 - h2(D[mask])
    return out
```

$D=0.5$ 时 $h_2=1$，$R=0$，掩码写成 `< 0.5` 再把边界显式填 0，避免 $1-1$ 的浮点噪声。$h_2$ 与 [香农章](/information/shannon/code-demo) 同一个 `clip` 技巧。

公式对偶：BSC 容量是「信道已经以 $p$ 翻转」；这里是「你愿意以 $D$ 翻转」。

标几个 $D$ 只是读图，不是仿真最优码。

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/information/rate-distortion/code/demo.py`
