---
title: "s20 深度强化学习：DQN 与策略梯度 — demo.py"
---

# s20 深度强化学习：DQN 与策略梯度 — demo.py

<a href="../code/s20_deep_rl/demo.py" target="_blank" download>Download demo.py</a>

## What This Code Does

在 OpenAI Gymnasium 的 CartPole-v1 环境上从零实现并对比两种经典深度 RL 算法：DQN（Deep Q-Network，含经验回放缓冲区 + 目标网络固定 + epsilon 衰减）和 REINFORCE（Monte Carlo 策略梯度，含回报标准化降方差）。训练 500 episode 后可视化奖励曲线、DQN 损失曲线和 REINFORCE 策略决策边界。

## Run It

```bash
cd s20_deep_rl/code
python demo.py
```

## Key Output

- 环境: CartPole-v1，状态 4 维（位置/速度/角度/角速度），动作 2 个（左推/右推），最高分 500
- DQN 超参数: lr=0.001, gamma=0.99, epsilon 1.0->0.01, buffer=10000, batch=64, target_update=100
- REINFORCE 超参数: lr=0.001, gamma=0.99
- 训练 500 episode 后输出两种算法最后 100 episode 平均奖励和达到 475+ 的次数

<<< @/snippets/s20_deep_rl/demo.py
