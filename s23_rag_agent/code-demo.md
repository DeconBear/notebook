---
title: "s23 RAG 与 AI Agent — demo.py"
---

# s23 RAG 与 AI Agent — demo.py

<a href="../code/s23_rag_agent/demo.py" target="_blank" download>Download demo.py</a>

## What This Code Does

从零构建完整的 RAG（检索增强生成）系统：文本切分 -> embedding 向量化 -> 向量存储与检索 -> 结合 LLM 生成回答。同时实现 ReAct Agent 的 Thought-Action-Observation 推理循环，对比 RAG vs 非 RAG 的回答质量差异。内置中文知识库涵盖 AI、Python、机器学习等主题，所有高级依赖均可选回退。

## Run It

```bash
cd s23_rag_agent/code
python demo.py
```

## Key Output

- 环境检测报告：sentence-transformers / ChromaDB / OpenAI API 等依赖的可用性状态
- 文档语料库统计：内置 10+ 篇中文知识文档的加载数量与总字数
- RAG 检索 Top-K 展示：对每个查询显示最相关的文档片段及相似度分数
- RAG vs 非 RAG 回答对比：有检索增强的回答包含事实依据
- ReAct Agent 步骤日志：Thought -> Action -> Observation 的完整推理链

<<< @/snippets/s23_rag_agent/demo.py
