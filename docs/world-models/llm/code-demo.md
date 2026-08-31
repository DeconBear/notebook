---
title: "wm08 LLM 世界模型与路径对比 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# wm08 LLM 世界模型与路径对比 — demo.py 代码详解

<a href="/notebook/code/world-models/llm/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/world-models/llm/code
python demo.py
```

## 代码逐段详解

### 第1步：八路径对比图

`plot_path_comparison()` 输出 `wm08-01-path-compare.png`：左侧为 Dreamer / JEPA / Genie / LLM 四条代表路径的雷达图，右侧为 wm01–wm08 在六维能力上的分组柱状图。评分是教学向相对尺度。

### 第2步：文本世界转移表

5 个状态 × 5 个动作，用字典 `TRANSITIONS[(s,a)] -> s'` 定义合法转移；非法动作则状态不变。这是"可验证的真实世界动力学"，用来对照模型预测。

### 第3步：字符级 TextWorldModel

```python
s = embed(state_ids).mean(1)
a = embed(action_ids).mean(1)
logits = head(mlp(cat([s, a]))).view(B, max_state_len, vocab_size)
```

用均值池化把变长字符序列压成向量，再预测下一状态每个字符位置的分布。这是 LLM 下一状态预测的**最小玩具近似**。

### 第4步：多步推演

按「拿起钥匙 → 开门 → 出门 → 放下钥匙」执行，打印真实 vs 预测，并导出训练曲线与结果表。

## 完整代码

<<< @/world-models/llm/code/demo.py
