---
title: "s23 RAG 与 AI Agent — exercise.py"
---

# s23 RAG 与 AI Agent — exercise.py 练习指南

<a href="../code/s23_rag_agent/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过补全三个 TODO 任务，掌握 RAG 和 ReAct Agent 的核心实现：
1. 带重叠的文本切分 —— RAG 索引的第一步
2. 向量相似度搜索 —— RAG 检索的核心
3. ReAct Agent 输出解析 —— Agent 推理链的文本理解

## 预备知识

- RAG 索引阶段：文档 → 句子切分 → chunk 组装（含重叠）→ 嵌入 → 向量库
- 余弦相似度：$\cos(\mathbf{q}, \mathbf{d}_i) = \frac{\mathbf{q} \cdot \mathbf{d}_i}{\|\mathbf{q}\| \cdot \|\mathbf{d}_i\|}$
- ReAct 循环格式：`Thought → Action → Observation → ... → Final Answer`

## 任务清单

### TODO 1：实现带重叠的文本切分（`chunk_text_with_overlap` 函数）

**任务**：将长文本切分为具有固定大小和重叠的文本块。

**实现步骤**：

1. **句子分割**：
   ```python
   raw_sentences = re.split(r'(?<=[。！？.!?\n])', text)
   # (?<=...) 是后顾断言 —— 在标点符号之后切分，保留标点在前一句中
   sentences = [s.strip() for s in raw_sentences if s.strip()]
   ```

2. **句子组装为 chunk**：
   ```python
   chunks = []
   current_chunk = ""
   for sentence in sentences:
       if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
           chunks.append(current_chunk.strip())
           # 取重叠部分作为新 chunk 的开头
           overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
           current_chunk = overlap_text + sentence
       else:
           current_chunk += sentence
   ```

3. **添加最后一个 chunk**：
   ```python
   if current_chunk.strip():
       chunks.append(current_chunk.strip())
   ```

**关键考量**：
- **重叠 `overlap=40`**：相邻 chunk 共享 40 个字符，防止关键信息被切在边界
- **边界情况**：一个句子超过 `chunk_size`` 时怎么办？代码应正确处理超长句子
- **空文本**：输入空字符串时返回空列表

**预期输出**：
```
原始文本长度: ~200 字符
切分为 2-3 个块
相邻块之间有重叠（第二个 chunk 包含第一个 chunk 的结尾）
无空块 ✓
```

### TODO 2：实现向量相似度搜索

**任务 2a**：实现 `cosine_similarity_matrix(query_vec, doc_matrix)`

**数学**：$\text{sim}_i = \frac{\mathbf{q} \cdot \mathbf{d}_i}{\|\mathbf{q}\| \cdot \|\mathbf{d}_i\|}$

**实现步骤**：
```python
# 1. 归一化查询向量
query_norm = np.linalg.norm(query_vec)
query_normalized = query_vec / query_norm

# 2. 归一化文档矩阵（每行一个文档向量）
doc_norms = np.linalg.norm(doc_matrix, axis=1)     # (N,)
doc_norms = np.where(doc_norms < 1e-10, 1, doc_norms)  # 避免除零
doc_normalized = doc_matrix / doc_norms[:, np.newaxis]  # (N, d)

# 3. 归一化后内积 = 余弦相似度
similarities = doc_normalized @ query_normalized    # (N,)
```

**注意**：`doc_norms[:, np.newaxis]` 将 (N,) 变为 (N, 1)，实现逐行归一化。

**任务 2b**：实现 `top_k_search(query_vec, doc_matrix, doc_texts, top_k)`

```python
similarities = cosine_similarity_matrix(query_vec, doc_matrix)
top_indices = np.argsort(-similarities)[:top_k]     # 降序
results = [(doc_texts[idx], similarities[idx]) for idx in top_indices]
```

**预期输出**：
```
文档 0 (深度学习) 获得最高相似度 ✓
Top-3 搜索结果按相似度降序排列
```

### TODO 3：解析 ReAct Agent 的输出（`parse_react_output` 函数）

**任务**：从 LLM 的原始输出中解析出 `Thought`、`Action`、`Observation` 和 `Final Answer`。

**实现步骤**：

1. **提取 Thought**：
   ```python
   thought_match = re.search(r'Thought:\s*(.+?)(?=\n(?:Action|Observation|Final)|$)',
                             raw_output, re.IGNORECASE | re.DOTALL)
   # (?=...) 是前瞻断言 —— 匹配到下一个 Action/Observation/Final 标签之前
   # re.DOTALL 使 . 也匹配换行符
   ```

2. **提取 Action**：
   ```python
   action_match = re.search(r'Action:\s*(.+?)(?:\n|$)', raw_output, re.IGNORECASE)
   if action_match:
       action_text = action_match.group(1).strip()
       # 解析 "tool_name(input)" 格式
       tool_match = re.match(r'(\w+)\((.+)\)', action_text)
       if tool_match:
           parsed["action"] = tool_match.group(1)
           parsed["action_input"] = tool_match.group(2)
       else:
           # 尝试 "tool_name: input" 格式
           alt_match = re.match(r'(\w+):?\s*(.+)', action_text)
           if alt_match:
               parsed["action"] = alt_match.group(1)
               parsed["action_input"] = alt_match.group(2)
   ```

3. **提取 Observation**：
   ```python
   obs_match = re.search(r'Observation:\s*(.+?)(?=\n(?:Thought|Action|Final)|$)',
                         raw_output, re.IGNORECASE | re.DOTALL)
   ```

4. **提取 Final Answer**：
   ```python
   final_match = re.search(r'Final Answer:\s*(.+?)$', raw_output, re.IGNORECASE | re.DOTALL)
   if final_match:
       parsed["final_answer"] = final_match.group(1).strip()
       parsed["is_final"] = True
   ```

5. **结束信号检测**（没有显式 Final Answer 但有隐含信号）：
   ```python
   end_markers = ["总结", "综上", "答案是", "结论", "最终"]
   if any(m in parsed["thought"] for m in end_markers):
       parsed["final_answer"] = parsed["thought"]
       parsed["is_final"] = True
   ```

**正则表达式注意事项**：
- `re.IGNORECASE`：大小写不敏感（"Thought" 和 "thought" 都能匹配）
- `re.DOTALL`：`.` 匹配包括换行符在内的任意字符，允许跨行内容
- 前瞻断言 `(?=...)`：匹配到但不消耗字符，用于精确定位边界

**预期输出**：
```
测试 1: 标准 Action 格式
  Thought 正确 ✓  Action 正确 ✓  Action Input 正确 ✓

测试 2: Final Answer 格式
  正确识别为最终回答 ✓

测试 3: 多步推理格式
  能从多步输出中提取关键信息
```

## 完成后的验证

全部三个 TODO 通过测试后，运行 `python code/demo.py` 观察：
1. RAG 检索能否返回与查询最相关的文档片段
2. RAG vs 纯 LLM 回答的差异（事实准确性 vs 流畅度）
3. ReAct Agent 的 Thought→Action→Observation 完整推理链

## 完整代码

<<< @/snippets/s23_rag_agent/exercise.py
