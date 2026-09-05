---
title: "wm08 LLM 世界模型与路径对比 — exercise.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# wm08 LLM 世界模型与路径对比 — exercise.py 练习指南

<a href="/notebook/code/world-models/symbolic/llm-sim/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

1. `encode`：文本 → 定长 id（PAD=0）
2. `TextWorldModel.forward`：state/action 嵌入均值池化 → MLP → 字符 logits
3. `lookup_transition`：查真实转移表，作为推演对照

## 关键实现提示

```python
# encode
ids = [stoi.get(c, 0) for c in text][:max_len]
ids += [0] * (max_len - len(ids))

# forward
s = self.embed(state_ids).mean(dim=1)
a = self.embed(action_ids).mean(dim=1)
h = self.mlp(torch.cat([s, a], dim=1))
return self.head(h).view(-1, self.max_state_len, self.vocab_size)

# lookup
return STATES[TRANSITIONS.get((STATES.index(s), ACTIONS.index(a)), STATES.index(s))]
```

## 完成后的验证

运行后推演至少 2/3 步匹配真实下一状态，即视为通过。


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/world-models/symbolic/llm-sim/code/exercise.py`
