---
title: "信道编码 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 信道编码 — demo.py 代码详解

<a href="/notebook/code/information/channel-coding/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/information/channel-coding/code
python demo.py
```

CPU、NumPy 即可。一张图 `hamming_ber.png`：$p$ 从小到大，未编码 4 比特 vs Hamming(7,4) 译码后的数据误比特率。`np.random.seed(42)` 可复现。每点 4000 组，大约一两秒。

## 代码逐段详解

### 第1步：编码 — 校验覆盖哪些位

1-index 位置 $1,2,4$ 是校验，$3,5,6,7$ 是数据。0-index 就是 `c[0],c[1],c[3]` 与 `c[2],c[4],c[5],c[6]`。

```python
c[0] = c[2] ^ c[4] ^ c[6]  # 覆盖 1,3,5,7
c[1] = c[2] ^ c[5] ^ c[6]  # 覆盖 2,3,6,7
c[3] = c[4] ^ c[5] ^ c[6]  # 覆盖 4,5,6,7
```

`^` 是按位异或，0/1 上等于模 2 加。偶校验：覆盖集合里 1 的个数为偶。

---

### 第2步：校验子合成位置

```python
pos = s0 + 2 * s1 + 4 * s2
if pos:
    r[pos - 1] ^= 1
```

三个比特当二进制整数，正好是 1-index 错误位置。`pos==0` 表示三个校验都过。翻那位再取出数据位 `r[[2,4,5,6]]`。两个错误时会指错位置——Hamming 只保证纠 1。

---

### 第3步：BSC 与 BER

```python
flips = np.random.rand(*bits.shape) < p
return bits ^ flips.astype(int)
```

`rand < p` 得到布尔翻转掩码。未编码的 BER 就是每位独立出错的 $p$（大数据下）。Hamming 的 BER 是译码后再比 4 个数据位。

Python 循环 4000 次编码对 CPU 足够；不必向量化成 $(N,7)$，可读性优先。

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/information/channel-coding/code/demo.py`
