---
title: "s19 强化学习入门：MDP 与 Q-Learning — demo.py"
---

# s19 强化学习入门：MDP 与 Q-Learning — demo.py

<a href="../code/s19_rl_qlearning/demo.py" target="_blank" download>Download demo.py</a>

## What This Code Does

从零实现 10x10 GridWorld 环境和 Q-Learning 表格方法 Agent，演示 TD 更新公式 Q(s,a) += alpha * (r + gamma * max Q(s',a') - Q(s,a)) 的迭代收敛过程，并进行 epsilon 衰减策略（快速/中等/慢速）和学习率（0.05-0.5）的消融对比实验。

## Run It

```bash
cd s19_rl_qlearning/code
python demo.py
```

## Key Output

- 环境: 10x10 网格，起点 (0,0)，终点 (9,9)，3 个陷阱位于对角线 (3,3)/(5,5)/(7,7)，每步惩罚 -0.1
- Agent 配置: alpha=0.1, gamma=0.95, epsilon_init=1.0->min=0.01, decay=0.995
- 训练 2000 episode 后收敛，打印最优路径长度和最终 100 episode 平均奖励
- Epsilon 对比实验: 快速衰减(0.99) 早利用但可能次优，中等(0.995) 均衡，慢速(0.999) 探索充分但收敛慢

<<< @/snippets/s19_rl_qlearning/demo.py
