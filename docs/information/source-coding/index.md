---
title: "信源编码"
order: 30
---
# 信源编码：频繁符号为什么要短码

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

> 无噪声时，压缩的极限是熵：平均码长 $L\ge H$。Huffman 是一种前缀码，把概率大的符号放在短分支上。容量与噪声留给 [香农](/information/shannon/) 和 [信道编码](/information/channel-coding/)；允许主动失真见 [率失真](/information/rate-distortion/)。

---

## 一、前缀码与 Kraft

前缀码：没有任何码字是另一个码字的前缀，才能即时解码（不必等结束符）。平均长度

$$
L=\sum_i p_i \ell_i \ge H(p),
$$

等号在 $p_i$ 都是 $2^{-\ell_i}$ 时可以逼近（香农第一定理：分组够长时 $L\to H$）。

![Huffman 树](./images/info-huffman.png)

> **图解说明**：概率大的叶子离根近；平均码长 $L$ 贴在熵 $H$ 上方。这是可变长度，不是固定 $\log_2|\mathcal{X}|$。

---

## 二、Huffman 怎么长树

反复把当前最轻的两棵子树合并，左 $0$ 右 $1$。四人字母表 $\{A,B,C,D\}$ 概率 $0.4,0.3,0.2,0.1$ 是教材常客。得到的 $L$ 应满足 $H\le L<H+1$。

---

## 三、代码在做什么

`demo.py` 用堆实现 Huffman，打印码本、$H$、$L$，并画各符号的概率 vs 码长。没有画二叉树图形（示意图在上面），只验证不等式。

![Huffman 码长](./images/huffman_len.png)

$A$ 最短，$D$ 最长。$L-H$ 是这张短表上「没分成长块」留下的冗余。

---

## 四、小结

| 概念 | 一句话 |
|------|--------|
| 前缀码 | 码字互不为前缀，可即时解 |
| $L\ge H$ | 压不进熵以下（无失真） |
| Huffman | 贪心合并最轻两棵树 |
| 下游 | 有噪改走 [信道编码](/information/channel-coding/) |

> 下一章 [信道编码](/information/channel-coding/)。

## 📥 Code

| File | View | Download |
|------|------|----------|
| demo.py | [Open](./code-demo) | <a href="/notebook/code/information/source-coding/demo.py" target="_blank" download>Download</a> |
| exercise.py | [Open](./code-exercise) | <a href="/notebook/code/information/source-coding/exercise.py" target="_blank" download>Download</a> |

## 参考

1. Cover & Thomas, 第 5 章（Huffman）
2. MacKay, *ITILA*
