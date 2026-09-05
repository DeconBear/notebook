---
title: "符号世界模型导论 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 符号世界模型导论 — demo.py 代码详解

<a href="/notebook/code/world-models/symbolic/overview/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/world-models/symbolic/overview/code
python demo.py
```

秒级，无 PyTorch。同一把「钥匙—门」微型世界：规则执行器 vs 只记训练转移的查找表。换一把未见过的钥匙后，只有规则还能出门。

## 代码逐段详解

### 第1步：状态是字典，不是张量

```python
s = dict(state)
loc, has_key, door, key_id = s["loc"], s["has_key"], s["door"], s["key_id"]
```

符号世界模型把状态写成**有名字的字段**。`dict(state)` 浅拷贝，避免 `step` 改到调用者手里的原字典。**语法 `s["loc"]`**：用字符串当下标；拼错键会 `KeyError`，这是故意的——字段集是契约。

---

### 第2步：`step_rules` — 定律写在 `if` 里

```python
if action == "pickup" and loc == "room" and not has_key:
    s["has_key"] = True
elif action == "unlock":
    if has_key and key_id and door == "locked":
        s["door"] = "open"
elif action == "go_out":
    if door == "open":
        s["loc"] = "outside"
```

- **pickup**：只在房间且没钥匙时捡起。条件失败则状态不变（非法动作被忽略，不是报错）。
- **unlock**：`has_key and key_id` — `key_id` 非空就视为匹配。注释写明「任意非空 id 都匹配」：泛化靠的是**谓词**「有钥匙」，不是记忆字符串 `"iron"`。
- **go_out**：门必须已开。规则组合：pickup → unlock → go_out 才能达成目标。

**设计**：没有神经网络。可执行规则 = 世界模型。换 `key_id` 不改代码，定律仍成立。

---

### 第3步：查找表怎么「学」

```python
def train_lookup(episodes):
    table = {}
    for s, a in episodes:
        key = (s["loc"], s["has_key"], s["door"], s["key_id"], a)
        table[key] = step_rules(s, a)
    return table
```

键是**冻结后的元组**（字典不能当 dict 的 key，因为可变、不可哈希）。值是规则走一步的下一状态——查找表其实是在**抄规则的轨迹**，不是另学一套物理。

训练集只见过 `key_id="iron"` 的三步计划，表里只有这三条键。

```python
if key in table:
    return dict(table[key])
return dict(state)  # 没见过：保持原状
```

未见过的 `(state, action)` 当成「什么都没发生」。这是死记硬背的典型失败：不是随机胡编，而是**静默不变**，更像「幻觉成空操作」。

**语法 `key in table`**：字典成员检测，平均 O(1)。

---

### 第4步：`rollout` 与成功判定

```python
def rollout(step_fn, start, actions):
    s = dict(start)
    traj = [dict(s)]
    for a in actions:
        s = step_fn(s, a)
        traj.append(dict(s))
    return traj
```

`step_fn` 可以是 `step_rules`，也可以是 `lambda st, a: step_lookup(table, st, a)`。同一条动作序列、两套动力学，才能公平对比。

`goal_reached`：终点 `loc=="outside"` **且** `door=="open"`。只走到 outside 但门还锁着不算（本规则下也走不出去）。

---

### 第5步：三组测试在拆哪种子泛化

| 用例 | 规则 | 查找表 |
|------|------|--------|
| 训练钥匙 iron | 成功 | 成功（表里有这些键） |
| 新钥匙 gold，从空手开始 | 成功（定律不看名字） | 失败：`("room", False, "locked", "gold", "pickup")` 不在表里，pickup 空操作，后面全卡死 |
| 已持 gold 直接开 | 成功 | 仍可能失败：没见过 gold 的 unlock 键 |

柱状图 `int(True)=1`：成功画成 1。`bar` 两组错开 `±w/2`。

`matplotlib.use("Agg")`：无显示器也能存 png（服务器/CI）。

---

### 和第6步：和神经网络世界模型的关系

RSSM/Dreamer 学的是连续函数近似；本章强调：**组合泛化**可以来自显式规则。查找表 ≈ 过拟合的转移记忆。LLM 世界模型介于两者之间，见 [llm-sim](/world-models/symbolic/llm-sim/code-demo)。

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/world-models/symbolic/overview/code/demo.py`
