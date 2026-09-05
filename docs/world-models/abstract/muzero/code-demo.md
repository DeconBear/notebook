---
title: "wm04 MuZero — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# MuZero — demo.py 代码详解

<a href="/notebook/code/world-models/abstract/muzero/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/world-models/abstract/muzero/code
python demo.py
```

秒级。本文件**不是**可训练的 MuZero：没有表示网络 $h$、动力学 $g$、预测头 $f$ 的梯度，也没有完整 MCTS。它只做两件事：画 $h/g/f$ 框图；在长度为 7 的一维「抓棋子」上对比贪心 vs 一步看清距离的搜索。要看真正的树搜索循环（选择/扩展/回传），见 [AlphaGo 章](/nn-decision/rl/alphago/code-demo)。

## 代码逐段详解

### 第1步：三头框图 `draw_muzero`

```python
boxes = [(0.5, 'h: obs→latent'), (3.5, 'g: latent,action→next'), (6.5, 'f: policy,value,reward')]
ax.add_patch(FancyBboxPatch((x, 1.3), 2.5, 1.5, boxstyle='round,pad=0.04', ...))
```

MuZero 相对「有模拟器的 AlphaZero」：搜索树里的边**不调用**环境，只调用学到的 $g$（隐状态转移）和 $f$（策略、价值、即时奖励）。$h$ 只在根节点把观测压成隐状态。本函数用 Matplotlib 补丁画三个盒子，箭头表示数据流。`axis('off')` 去掉坐标轴。`FancyBboxPatch` 的 `(x,y)` 是左下角，宽 2.5、高 1.5。

---

### 第2步：一维捕猎 — 环境规则写在循环里

```python
goal = 6
s = 0
for t in range(12):
    ...
    s = int(np.clip(s + a, 0, goal))
    total += 1.0 if s == goal else -0.05
    if s == goal:
        break
```

格子 `0…6`，目标在 6。动作是位移 $-1/0/+1$。到终点 +1 并结束，否则每步 −0.05（催你快走）。`np.clip` 防走出界。**没有**单独的 `Env` 类：规则短，嵌在评估函数里更易读。

`value_of` 对每种策略跑 300 局取平均。起点固定 $s=0$，随机性几乎没有（两种策略都是确定性的），重复只是为了和「若以后加噪声」同一结构。

---

### 第3步：贪心为什么其实已经够好（以及搜索在比什么）

```python
if policy_name == 'greedy':
    a = 1 if s < goal else 0
else:
    cand = [s - 1, s, s + 1]
    cand = [c for c in cand if 0 <= c <= goal]
    a = max(cand, key=lambda c: -abs(c - goal)) - s
```

贪心：没到就 +1。搜索式：在合法后继里选 **$|c-\mathrm{goal}|$ 最小**的格子，动作 = 新位置减旧位置。

在这条线上，贪心每步 +1，和「减小距离」**一样**。柱状图两者应接近——这是故意的玩具：让你看见「搜索 = 在候选后继上最大化某个价值」，而不是吹嘘搜索碾压。真 MuZero 的价值来自 $f$ 的 $v$，候选来自 $g$ 展开的树，深度远大于 1。

**语法 `max(cand, key=lambda c: -abs(c - goal))`**：`key` 指定比较依据；负距离 = 距离越小越大。`[c for c in cand if ...]` 过滤非法格。

**语法 `int(np.clip(...))`**：`clip` 返回数组/标量 dtype 可能是 float，`int` 变回格子索引。

---

### 第4步：柱状图

```python
ax.bar(names, vals, color=['#95A5A6', '#27AE60'])
```

灰 = 贪心，绿 = 搜索式。`tight_layout` 防标题被裁。图写入 `muzero_search_compare.png`。

---

### 和第5步：和正文 MuZero 损失的对应（代码里没有，读到这里要对上）

完整 MuZero 在真实轨迹上展开 $K$ 步想象，对齐：

- 策略：$p$ vs MCTS 改进后的 $\pi$
- 价值：$v$ vs 折现回报
- 奖励：$r$ vs 环境即时奖励

本 demo **没有**这三项 `loss`。若你只看到框图和柱子，不要以为已经实现了 $h,g,f$ 训练。下一步可对照 [RSSM](/world-models/abstract/rssm/code-demo) 的「隐状态滚动」+ AlphaGo 的 MCTS。

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/world-models/abstract/muzero/code/demo.py`
