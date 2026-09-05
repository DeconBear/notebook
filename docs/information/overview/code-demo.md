---
title: "信息论导论 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 信息论导论 — demo.py 代码详解

<a href="/notebook/code/information/overview/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/information/overview/code
python demo.py
```

CPU、NumPy 即可。一张图 `overview_entropy.png`：四种分布的熵柱状图。对数是 `np.log2`（bit），不是 `np.log`（nat）。

## 代码逐段详解

### 第1步：熵，丢掉数值零

$$
H(p)=-\sum_i p_i\log_2 p_i
$$

```python
def entropy_bits(p, eps=1e-15):
    p = np.asarray(p, dtype=float)
    p = p[p > eps]
    return float(-np.sum(p * np.log2(p)))
```

- **`p[p > eps]`**：布尔下标。$p=0$ 时 $p\log p$ 的极限是 $0$，但 `log2(0)` 会得到 `-inf`。丢掉零项最干净。
- **`np.log2`**：底 2。公平硬币 `[0.5,0.5]` 应得正好 $1$。
- 确定性 `[1,0]` 只剩一项 $1\cdot\log_2 1=0$。

---

### 第2步：nat → bit

```python
def nats_to_bits(h_nats):
    return h_nats / np.log(2.0)
```

`np.log` 是 $\ln$。$H_{\mathrm{bit}}=H_{\mathrm{nat}}/\ln 2$。打印 `1 nat ≈ 1.4427 bit`。机器学习损失常用 nat，本领域画容量用 bit。

---

### 第3步：四根柱

四面均匀 $p_i=1/4$，$H=\log_2 4=2$。偏硬币 $0.1/0.9$ 介于 $0$ 与 $1$ 之间。柱状图只是把四个标量并排，不是信道仿真。

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/information/overview/code/demo.py`
