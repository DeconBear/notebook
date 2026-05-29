# s23 RAG 与 AI Agent -- 代码说明与运行报告

## 程序做了什么
从零构建完整的 RAG（检索增强生成）系统：文本切分 -> embedding 向量化 -> 向量存储与检索 -> 结合 LLM 生成回答。同时实现 ReAct Agent 的 Thought-Action-Observation 推理循环，对比 RAG vs 非 RAG 的回答质量差异。内置中文知识库涵盖 AI、Python、机器学习等主题，所有高级依赖均可选回退。

## 运行方法
```bash
cd s23_rag_agent/code
python demo.py
```

## 运行结果

### 输出摘要（命令行文本，无生成图片）
- 环境检测报告：sentence-transformers / ChromaDB / OpenAI API 等依赖的可用性状态
- 文档语料库统计：内置 10+ 篇中文知识文档的加载数量与总字数
- 文档切分结果：按 chunk_size 切分后的段落数量
- RAG 检索 Top-K 展示：对每个查询显示最相关的文档片段及相似度分数
- RAG vs 非 RAG 回答对比：有检索增强的回答包含事实依据，无非 RAG 的回答可能产生幻觉
- ReAct Agent 步骤日志：Thought（分析当前状态） -> Action（调用工具如搜索/计算） -> Observation（观察结果）的完整推理链
- 各部分的最终总结与关键指标汇总

### 图片资源: 概念图解
本章无 demo 生成的图片，所有结果以命令行文本形式展示。以下为配套概念图：
- `23-01-rag-pipeline.png` -- RAG 完整流程：文档加载 -> 切分 -> 向量化 -> 存储 -> 检索 -> 增强生成
- `23-02-vector-similarity-search.png` -- 向量相似度检索原理：查询向量与文档向量的余弦相似度/KNN 搜索
- `23-03-react-agent-loop.png` -- ReAct Agent 循环：Thought -> Action -> Observation 的迭代推理流程
- `23-04-agent-architecture.png` -- AI Agent 通用架构：感知 -> 推理 -> 规划 -> 工具调用 -> 执行的完整框架

## 代码结构
- `check_environment()` -- 检测 embedding 模型/ChromaDB/OpenAI 可用性，标记回退方案
- `load_document_corpus()` -- 加载内置的 10+ 篇中文知识文档（AI/ML/Python 等主题）
- `split_documents()` -- 按段落/句子进行文档切分，支持可配置 chunk_size
- `class SimpleEmbedder` -- embedding 抽象层：优先 sentence-transformers（语义嵌入），回退 TF-IDF（词频统计）
- `class VectorStore` -- 向量存储抽象层：优先 ChromaDB（持久化），回退内存字典 + 余弦相似度
- `class RAGPipeline` -- RAG 完整流程：检索 -> 构造含上下文 prompt -> LLM 生成回答
- `class ReActAgent` -- ReAct Agent：实现 Thought-Action-Observation 循环推理
- `compare_rag_vs_no_rag()` -- 并排对比有/无检索增强的回答质量
- `main()` -- 主流程：环境检测 -> 文档加载 -> 切分 -> RAG 演示 -> ReAct Agent 演示 -> 对比总结

## 运行环境
- Python 依赖: numpy, scikit-learn（基础）；sentence-transformers, chromadb, openai（可选增强）
- 硬件需求: CPU 即可（使用 TF-IDF 回退方案时极小内存，约 50MB）
- 预计运行时间: ~10-30 秒（使用回退方案），~1-2 分钟（使用全量依赖）
- 注：安装 sentence-transformers 后首次运行会自动下载 embedding 模型（约 100-400MB）
