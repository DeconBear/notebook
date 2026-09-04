---
title: "s25 AI 安全与对齐 — exercise.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# s25 AI 安全与对齐 — exercise.py 练习指南

<a href="/notebook/code/applied/systems/safety/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过补全三个 TODO 任务，掌握 AI 安全检测的三个核心组件：
1. 幻觉检测 —— 基于关键词重叠和数字一致性验证模型输出
2. 越狱攻击检测 —— 基于正则表达式的多模式匹配
3. 内容安全分类器 —— 基于关键词的多维度安全评分

## 预备知识

- 幻觉检测的三种方法：关键词重叠、否定词检测、数字一致性检查
- 越狱攻击的四种类型：提示注入、角色扮演、编码绕过、限制解除
- 内容安全的三级分类：safe（评分 $\geq 80$）、review（$50-79$）、unsafe（$< 50$）

## 任务清单

### TODO 1：实现幻觉检测（`detect_hallucination` 函数）

**任务**：实现两种检测方法——`keyword_overlap` 和 `number_check`。

**方法 1：keyword_overlap**

```python
intersection = model_kw & truth_kw      # 交集（共同关键词）
union = model_kw | truth_kw              # 并集（所有关键词）
overlap_ratio = len(intersection) / len(union)
# 重叠率低 → 可能是幻觉
is_hallucination = overlap_ratio < 0.2
confidence = 1.0 - overlap_ratio
```

**关键词提取**（已实现）：
- 中文：提取 2-3 字滑动窗口组合（"人工"、"智能"、"人工智能"等）
- 英文：提取 2+ 字母的单词
- 数字：提取所有数字序列

**方法 2：number_check**

```python
model_nums = extract_numbers(model_answer)    # 模型回答中的数字集合
truth_nums = extract_numbers(ground_truth)    # 真实答案中的数字集合
common_nums = model_nums & truth_nums
all_nums = model_nums | truth_nums

num_consistency = len(common_nums) / len(all_nums)
is_hallucination = num_consistency < 0.5
confidence = 1.0 - num_consistency
```

**数字一致性检查的考量**：
- 如果真实答案没有数字（`not truth_nums`），返回非幻觉（无法判断）
- 如果模型没有提供数字（`not model_nums`），可能是在回避 → 置信度 0.3
- 数字完全一致 → 一致性 1.0 → 非幻觉
- 数字完全不同 → 一致性 0.0 → 高度可能是幻觉

**预期输出**：
```
[number_check] ✓ 检测="幻觉" 预期="幻觉" 置信度 ≈ 1.0
  (水的沸点=200 vs 真实=100 → 数字不一致)

[keyword_overlap] ✓ 检测="正常" 预期="正常" 置信度 ≈ 0.x
  (深度学习使用反向传播 → 关键词与知识库重叠)

检测准确率: 5-6/6
```

### TODO 2：实现基于模式的越狱检测（`PatternBasedJailbreakDetector` 类）

**任务 2a**：在 `__init__` 中定义检测模式

已有基础模式（需补全更多模式）：

```python
self.patterns = [
    # 提示注入
    (r'ignore\s+(all\s+)?(previous|prior|above).*(?:instructions?|prompts?)', "英文提示注入", 0.9),
    (r'忽略\s*(所有|之前的|上面的)?\s*(指令|提示|规则)', "中文提示注入", 0.9),

    # 角色扮演
    (r'(?:pretend|imagine|act).*(?:you\s+(?:are|have))', "英文角色扮演", 0.8),
    (r'假装.*(?:你是|作为)', "中文角色扮演", 0.8),

    # 限制解除
    (r'(?:no\s+(?:restrictions?|limitations?|filters?|rules?)|unrestricted|unfiltered)', "限制解除", 0.9),
    (r'没有.*(?:限制|规则|过滤|约束)', "中文限制解除", 0.9),

    # 编码绕过
    (r'(?:decode|解码).*(?:base64|base\s*64|编码)', "编码绕过", 0.8),

    # DAN 等特殊角色（高风险）
    (r'\bDAN\b', "DAN角色扮演", 0.95),
    (r'Do\s+Anything\s+Now', "DAN角色扮演", 0.95),

    # 系统指令覆盖
    (r'(?:override|overwrite)\s+(?:system\s+)?(?:prompt|instruction)', "系统指令覆盖", 0.95),
    (r'(?:从|自)\s*(?:现在|此刻|这一秒).*(?:开始|起).*(?:你|作为)', "角色切换", 0.7),
]
```

**任务 2b**：实现 `detect(prompt)` 方法

```python
def detect(self, prompt):
    matched_categories = []
    max_risk = 0.0

    for pattern, category, risk_level in self.patterns:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            matched_categories.append(category)
            max_risk = max(max_risk, risk_level)    # 最高风险等级

    # 综合评分：最高风险 + 额外匹配的增量
    extra_matches = len(matched_categories) - 1 if matched_categories else 0
    risk_score = min(1.0, max_risk + 0.05 * extra_matches)
    is_attack = risk_score > 0.3
    return is_attack, matched_categories, risk_score
```

**风险评分公式**：$\text{risk} = \min(1.0,\; \max\_\text{risk} + 0.05 \times (\text{num\_matches} - 1))$

- 如果只匹配 1 个低风险模式（0.7），risk = 0.7
- 如果匹配 1 个高风险模式（0.95），risk = 0.95
- 如果匹配 2+ 个模式，每个额外匹配加 0.05（匹配越多越可疑）

**`re.IGNORECASE` 的作用**：大小写不敏感匹配——"DAN"、"Dan"、"dan" 都能匹配。

**预期输出**：
```
✓ [正常] 预期="正常" | Python的列表推导式
✓ [攻击] 预期="攻击" | Ignore all previous instructions... (英文提示注入)
✓ [攻击] 预期="攻击" | You are now DAN... (DAN角色)
✓ [攻击] 预期="攻击" | Pretend you are an AI... (角色扮演)
...
检测准确率: 8-9/9
```

### TODO 3：实现简单的内容安全分类器（`ContentSafetyClassifier` 类）

**任务 3a**：在 `__init__` 中为各类别添加关键词

```python
self.categories = {
    "harmful": {
        "label": "有害内容",
        "keywords": ["制造炸弹", "恶意代码", "入侵系统", "病毒制作", "毒品制作"],
        "weight": 1.0,
    },
    "hate": {
        "label": "仇恨言论",
        "keywords": ["去死", "废物", "垃圾", "恨你", "该死", "恶心"],
        "weight": 0.9,
    },
    "violence": {
        "label": "暴力内容",
        "keywords": ["杀死", "殴打", "炸弹", "武器", "枪", "暴力"],
        "weight": 0.9,
    },
    "self_harm": {
        "label": "自残/自杀",
        "keywords": ["自杀", "自残", "不想活", "结束生命", "割腕"],
        "weight": 1.0,
    },
}
```

**任务 3b**：实现 `classify(text)` 方法

```python
def classify(self, text):
    text_lower = text.lower()
    total_penalty = 0.0
    matched_categories = []
    details = {}

    for cat_id, cat_config in self.categories.items():
        keywords = cat_config.get("keywords", [])
        weight = cat_config.get("weight", 0.5)
        label = cat_config.get("label", cat_id)

        # 统计匹配的关键词
        matched_keywords = [kw for kw in keywords if kw.lower() in text_lower]
        if matched_keywords:
            penalty = len(matched_keywords) * 20 * weight
            total_penalty += penalty
            matched_categories.append(label)
            details[label] = matched_keywords

    # 安全评分
    safety_score = max(0, 100 - total_penalty)

    # 分级
    if safety_score >= 80:
        safety_level = "safe"
    elif safety_score >= 50:
        safety_level = "review"
    else:
        safety_level = "unsafe"

    return {
        "safety_level": safety_level,
        "score": safety_score,
        "matched_categories": matched_categories,
        "details": details,
    }
```

**三级分类的含义**：
- **safe（安全）**：评分 $\geq 80$ —— 可以正常输出
- **review（需审查）**：评分 $50-79$ —— 需要二次审查或自动改写
- **unsafe（不安全）**：评分 $< 50$ —— 应拒绝输出或大幅改写

**罚分公式**：$\text{penalty} = \text{match\_count} \times 20 \times \text{weight}$。例如匹配 3 个仇恨关键词（weight=0.9）→ 扣 54 分 → 评分 46 → unsafe。

**预期输出**：
```
✓ [安全] 预期=safe 评分=100 | 正常聊天
✓ [安全] 预期=safe 评分=100 | 技术讨论
✓ [不安全] 预期=unsafe 评分<50 | 仇恨言论（"去死"、"恨你"等）
✓ [不安全] 预期=unsafe 评分<50 | 危险内容请求（"炸弹"、"制造"等）
✓ [审查] 预期=review 评分50-79 | 心理困扰（含"不想活"）

检测准确率: 5-6/6
```

## 完成后的验证

全部三个 TODO 通过测试后，运行 `python code/demo.py` 观察：
1. 幻觉检测的准确率和误报率
2. 越狱攻击检测的 Accuracy/Precision/Recall
3. 内容安全评估的多维度评分和风险标识
4. 综合安全报告的生成

### 扩展思考题

1. **关键词重叠检测有什么根本局限性？**
   - 无法理解语义——"巴黎不是法国的首都" vs "巴黎是法国的首都"关键词重叠率高但事实相反
   - 对同义词重组无能为力——"水的沸点低于200度但高于50度"无法被验证

2. **基于模式的越狱检测可能被如何绕过？**
   - 同义词替换："忽略" → "无视/跳过/省略"
   - 编码："ignore" → "i\x67nore" (十六进制)、Base64、ROT13
   - 多语言："忽略之前的指令" → 翻译为非英语/中文（低资源语言可能没有匹配模式）

3. **内容安全分类器应该如何与模型推理流水线集成？**
   - 预处理阶段：分类器扫描输入 prompt → 拦截已知越狱模式
   - 后处理阶段：分类器扫描模型输出 → 不符合安全标准的拒绝或改写
   - 实时监控：对所有输入/输出进行安全评分记录，异常检测

4. **如何平衡安全过滤的假阳性和假阴性？**
   - 假阳性（False Positive）：误拦正常内容 → 用户体验差
   - 假阴性（False Negative）：漏过有害内容 → 安全风险
   - 策略：多层分级（safe/review/unsafe），review 级别引入人工审核；对高风险场景收紧阈值，低风险场景放宽

## 完整代码

<<< @/applied/systems/safety/code/exercise.py
