---
title: "经典控制"
---

# 经典控制：传递函数这门语言

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

> 经典控制看的是一个（或少数几个）误差标量，工具是拉普拉斯域的 $G(s)$：时域阶跃、根轨迹、Bode 图说的是同一套极点。PID 是把 $e$ 拆成三路再合成 $u$。状态向量与矩阵增益留给 [现代控制](/control/modern/)。

![闭环反馈回路](./images/ctrl-01-feedback-loop.png)

> **图解说明**：参考 $r$ 减输出 $y$ 得误差 $e$，控制器 $C$ 出 $u$，植物 $P$ 出 $y$，传感器把 $y$ 送回。经典四章都围着这张图转。

| 章 | 在问什么 |
|----|----------|
| **[传递函数与时域响应](/control/classical/transfer/)** | $G(s)$ 与二阶阻尼 $\zeta$：过阻尼 / 欠阻尼 / 振荡 |
| **[PID 控制](/control/classical/pid/)** | $K_p,K_i,K_d$ 各管什么；质量-弹簧-阻尼阶跃 |
| **[根轨迹](/control/classical/root-locus/)** | 增益 $K$ 变时，闭环极点在 $s$ 平面怎么走 |
| **[频域分析](/control/classical/frequency/)** | $G(j\omega)$、幅频 / 相频、相位裕度直觉 |

建议顺序：传递函数 → PID（手上先有一条会动的回路）→ 根轨迹 → Bode。跳去现代控制从 [状态空间](/control/modern/state-space/) 进。
