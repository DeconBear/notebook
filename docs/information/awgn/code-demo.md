---
title: "高斯信道 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 高斯信道 — demo.py 代码详解

<a href="/notebook/code/information/awgn/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/information/awgn/code
python demo.py
```

CPU、NumPy 即可。一张图 `awgn_cap.png`：左是容量 vs SNR(dB)，右是二进制 $\pm\sqrt{P}$ 加噪后的 $Y$。

## 代码逐段详解

### 第1步：dB 与线性 SNR

```python
def snr_db_to_lin(db):
    return 10.0 ** (np.asarray(db, dtype=float) / 10.0)

def capacity_bits(snr_lin):
    return 0.5 * np.log2(1.0 + snr_lin)
```

功率比用十进制对数：$10\,\mathrm{dB}\Rightarrow \mathrm{SNR}=10$。容量公式里的对数是 $\log_2$。$\mathrm{SNR}=0$ 时 $C=0$；$\mathrm{SNR}=1$（0 dB）时 $C=1/2$；$\mathrm{SNR}=3$ 时 $C=1$。

`**` 是幂，不要写成 `^`。

---

### 第2步：散点不是容量实现

右图输入只有两个点 $\pm\sqrt{P}$，不是容量公式假定的高斯 $X$。它只显示「噪声云有多糊」。横向加了一点抖动，否则同一 SNR 的点会叠成一条竖线。

`N_NOISE=1` 固定，SNR 全靠 $P$ 调，避免同时改两个量。

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/information/awgn/code/demo.py`
