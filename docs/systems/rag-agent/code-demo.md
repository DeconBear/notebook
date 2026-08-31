---
title: "s23 RAG 与 AI Agent — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# s23 RAG 与 AI Agent — demo.py 代码详解

<a href="/notebook/code/systems/rag-agent/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/systems/rag-agent/code
python demo.py
```

**依赖说明**：代码设计了多层回退机制。
- 有 `sentence-transformers` → 语义嵌入（384 维稠密向量）
- 无 `sentence-transformers` → TF-IDF 稀疏向量回退（384 维词频特征）
- 有 `chromadb` → 向量数据库存储（支持 ANN 搜索）
- 无 `chromadb` → 内存字典+Numpy 存储（支持精确余弦搜索）
- 有 `OPENAI_API_KEY` → 真实 GPT 生成回答
- 无 API Key → 模拟回答（展示 RAG 流程而不依赖外部 API）

## 代码逐段详解

### 第1步：环境检测 — 多级回退设计

```python
def check_environment():
    # 检测 sentence-transformers（语义嵌入）
    try:
        from sentence_transformers import SentenceTransformer
        HAS_EMBEDDING_MODEL = True
    except ImportError:
        print("使用 TF-IDF 回退方案")

    # 检测 ChromaDB（向量数据库）
    try:
        import chromadb
        HAS_CHROMADB = True
    except ImportError:
        print("使用内存字典回退方案")
```

**为什么设计多级回退**：初学者可能只有 Python + NumPy，没有 GPU，也没有下载大型模型。回退机制确保每个人都能运行 RAG 的核心逻辑，感受检索增强的效果。TF-IDF 虽不如语义嵌入精确，但在关键词匹配场景下可工作。

### 第2步：文档语料库 — 模拟知识库

内置 7 篇中文知识文档，涵盖多个主题领域：
- 人工智能概论（机器学习、深度学习、LLM）
- Python 编程语言（特性、应用、框架）
- 气候变化与环境（全球变暖、巴黎协定）
- 深度学习基础（架构、反向传播、Transformer）
- RAG 技术详解（三阶段、向量数据库、优势）
- 健康饮食指南（营养素、膳食建议）
- 太阳系与天文知识（行星、太阳）

**设计考量**：语料库涵盖的主题与测试查询形成覆盖关系，确保有相关文档可供检索。同时也包含不相关的主题，测试检索系统能否准确区分。

### 第3步：文本切分（Chunking）— RAG 的关键预处理

**为什么需要切分**：
- LLM 的上下文窗口有限，不能一次性塞入整篇文档
- 检索需要细粒度的匹配——返回整篇文档不如返回相关段落
- 太小的 chunk 丢失上下文，太大的 chunk 降低检索精度

**切分策略**（Recursive Character Splitting）：

```python
def split_text_into_chunks(text, chunk_size=300, chunk_overlap=50):
    # 1. 按句子分隔符初步分割
    separators = [r'\n\n', r'\n', r'[。！？!?]', r'[，,；;]']
    # 按优先级从高到低尝试切分

    # 2. 将句子组装成不超过 chunk_size 的块
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # 重叠：从当前 chunk 末尾取 overlap 字符作为新 chunk 的开头
            overlap_text = current_chunk[-chunk_overlap:]
            current_chunk = overlap_text + sentence
        else:
            current_chunk += sentence
```

**重叠的重要性**：`chunk_overlap=50` 确保了相邻 chunk 之间共享 50 个字符的上下文。这避免了一个关键句子恰好被切在两块的边界上，导致语义断裂。

**Chunk 大小选择**：`chunk_size=300` 字符（约 150-200 个中文 token）。太小（如 50）只包含一两个词，没有意义；太大（如 1000）相当于半篇文档，检索不精确。

### 第4步：嵌入模型 — 文本到向量

```python
class EmbeddingModel:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # 优先使用 sentence-transformers 语义模型
        if HAS_EMBEDDING_MODEL:
            self._model = SentenceTransformer(model_name)  # 384 维
            self._is_semantic = True
        else:
            # TF-IDF 回退：384 维词频向量
            self._tfidf_vectorizer = TfidfVectorizer(max_features=384)

    def encode(self, texts):
        if self._is_semantic:
            embeddings = self._model.encode(texts)
            # L2 归一化：使内积 = 余弦相似度
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms
            return embeddings
        else:
            return self._encode_tfidf(texts)
```

**为什么选择 `all-MiniLM-L6-v2`**：
- 只有 384 维（vs OpenAI 的 1536 维），计算效率高
- 在 MTEB 基准上表现优秀，适合语义搜索
- 轻量（~80MB），CPU 上也能运行

**为什么做 L2 归一化**：归一化后 $\|\mathbf{v}\| = 1$，两个向量的余弦相似度等于内积：$\cos(\mathbf{a},\mathbf{b}) = \mathbf{a} \cdot \mathbf{b}$。这使后续的向量搜索可以用矩阵乘法高效计算。

**TF-IDF 的局限性**：TF-IDF 基于词频统计，无法捕获语义相似性——"深度学习"和"神经网络"虽语义相关但词形完全不同，TF-IDF 无法识别这种关联。语义嵌入模型能解决这个问题。

### 第5步：向量存储 — 支持两种后端

```python
class SimpleVectorStore:
    def search(self, query_embedding, top_k=5):
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)

        if self.use_chromadb:
            # ChromaDB 后端：调用 ANN 搜索
            chroma_results = self._chroma_collection.query(
                query_embeddings=[query_embedding.flatten().tolist()],
                n_results=min(top_k, len(self._documents))
            )
            # ChromaDB 返回 cosine distance = 1 - cosine_similarity
            for doc, dist, meta in zip(...):
                score = 1.0 - dist
        else:
            # 内存后端：精确余弦相似度计算
            emb_matrix = np.stack(self._embeddings, axis=0)  # (N, d)
            similarities = emb_matrix @ query_embedding.flatten()  # (N,)
            top_indices = np.argsort(-similarities)[:top_k]
```

**为什么内积等于余弦相似度**：所有向量已 L2 归一化（$\|\mathbf{v}\| = 1$），所以 $\mathbf{q} \cdot \mathbf{d}_i = \|\mathbf{q}\| \cdot \|\mathbf{d}_i\| \cdot \cos(\mathbf{q}, \mathbf{d}_i) = \cos(\mathbf{q}, \mathbf{d}_i)$。

**内存后端的优缺点**：
- 优点：无需安装额外库，精确搜索（非近似）
- 缺点：$O(Nd)$ 复杂度，百万级向量时无法实时响应
- ChromaDB 使用 HNSW 近似最近邻索引，在百万级向量上可达到毫秒级响应

### 第6步：RAG 系统 — 集成索引、检索与生成

```python
class RAGSystem:
    def index_documents(self, documents, chunk_size=300, chunk_overlap=50):
        for doc in documents:
            chunks = split_text_into_chunks(content, chunk_size, chunk_overlap)
            for chunk in chunks:
                enriched_chunk = f"[{title}] {chunk}"    # 标题前缀增强检索
                all_chunks.append(enriched_chunk)

        embeddings = self.embedding_model.encode(all_chunks)
        self.vector_store.add(all_chunks, embeddings, all_metadata)

    def retrieve(self, query, top_k=3):
        query_embedding = self.embedding_model.encode([query])[0]
        results = self.vector_store.search(query_embedding, top_k)
        return results

    def build_rag_prompt(self, query, retrieved_docs):
        # 构造结构化上下文
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            title = doc["metadata"].get("title", "未知来源")
            context_parts.append(f"[来源 {i}: {title}]\n{doc['document']}")
        context_text = "\n\n---\n\n".join(context_parts)

        # 构建增强 prompt
        prompt = f"""你是一个基于知识库的问答助手。请仅根据以下提供的参考资料回答问题。
如果参考资料中没有足够的信息来回答，请明确说"根据现有资料无法确定"。

=== 参考资料 ===
{context_text}

=== 用户问题 ===
{query}

请用中文回答，并在可能的情况下引用具体的来源。"""
        return prompt
```

**Prompt 设计的关键元素**：
1. **角色设定**："基于知识库的问答助手"——限制模型只使用提供的资料
2. **边界声明**："根据现有资料无法确定"——让模型学会承认知识边界，减少幻觉
3. **来源标注**：`[来源 i: title]`——实现可溯源，用户能验证信息出处
4. **结构化分隔**：`---` 将不同来源明确分开

**为什么用标题前缀**：在 chunk 前加 `[人工智能概论]` 等标题，让嵌入模型在向量空间中更好地建立 chunk 与其来源文档的关联，同时帮助 LLM 理解每个片段的语境。

### 第7步：LLM 调用 — 真实 API vs 模拟回答

```python
def call_llm(prompt, model="gpt-3.5-turbo"):
    if HAS_OPENAI:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": "你是一个知识库问答助手..."},
                      {"role": "user", "content": prompt}],
            temperature=0.3,   # 低温度 = 更确定、更少随机性
            max_tokens=500
        )
        return response.choices[0].message.content
    else:
        return simulate_llm_response(prompt)
```

**`temperature=0.3` 的考量**：RAG 场景中我们希望模型忠实于提供的资料，而非发挥创造性。低温度减少随机性，使模型更倾向于选择概率最高的 token，从而减少偏离参考资料的风险。

**模拟回答的作用**：从 prompt 中提取数据构造展示性回答——这不产生真正的 LLM 质量，但完整展示了 RAG 流程的端到端效果。

### 第8步：ReAct Agent — 推理与行动的循环

**ReAct（Yao et al., 2022）的核心循环**：

```
Thought → Action → Observation → Thought → ... → Final Answer
```

```python
class ReActAgent:
    def run(self, query):
        for step in range(1, self.max_steps + 1):
            # Thought: 根据当前状态推理
            thought = self._generate_thought(query, step)

            # 判断是否已到最终回答
            if self._is_final_answer(thought):
                return self._extract_final_answer(thought)

            # Action: 解析并执行工具调用
            tool_name, tool_input = self._parse_action(thought)
            observation = self._execute_tool(tool_name, tool_input)

            # 记录历史
            self.history.append({
                "step": step, "thought": thought,
                "action": f"{tool_name}({tool_input})",
                "observation": observation
            })
```

**工具集设计**（模拟）：
- `calculator(expression)`：安全计算数学表达式，只允许基本运算符号
- `search(query)`：模拟知识检索，返回预定义的搜索结果
- `weather(city)`：模拟天气 API，返回预定义的城市天气

**ReAct 的关键设计**：
1. **`max_steps=5` 上限**：防止 Agent 陷入无限循环（工具调用失败后反复重试）
2. **`_parse_action`** 根据 Thought 中的关键词（"天气"→weather, "计算"→calculator, "搜索"→search）路由到对应工具
3. **`_is_final_answer`** 检测 Thought 中的结束标记（"最终回答"、"可以回答"、"答案是"等）

**一个完整的 ReAct 示例**：

> 用户："北京今天天气怎么样？需要带伞吗？"
> - **Thought 1**：需要查询天气。提取城市名并使用天气工具。
> - **Action**: weather("北京")
> - **Observation**: 温度 22°C，降雨概率 10%
> - **Thought 2**：已获取天气数据。温度适宜，降雨概率极低。
> - **Final Answer**: 北京今天 22°C，降雨概率仅 10%，不需要带伞。

### 第9步：RAG vs 非 RAG 对比实验

```python
def demo_rag_with_vs_without(rag_system, query):
    # RAG 模式：检索 → 构建增强prompt → LLM生成
    retrieved = rag_system.retrieve(query, top_k=3)
    rag_prompt = rag_system.build_rag_prompt(query, retrieved)
    rag_answer = call_llm(rag_prompt)

    # 纯 LLM 模式：直接回答问题
    no_rag_prompt = f"请回答以下问题：{query}"
    no_rag_answer = call_llm(no_rag_prompt)
```

**预期差异**：
- RAG 回答：基于检索到的具体文档，可溯源，事实准确但可能不够流畅
- 纯 LLM 回答：依赖参数化记忆，可能过时或不准确（如关于 2025 年的信息），但语言更流畅

### 关键概念速查表

| 概念 | 一句话解释 | 代码位置 |
|------|-----------|---------|
| RAG 三阶段 | 索引（文档→chunk→嵌入→向量库）→检索→生成 | `RAGSystem` 类 |
| Chunking | 将长文档切为重叠块，chunk_size=300, overlap=50 | `split_text_into_chunks()` |
| 语义嵌入 | 384 维稠密向量，用 L2 归一化后内积=余弦相似度 | `EmbeddingModel.encode()` |
| 向量搜索 | 余弦相似度排序取 top-k | `SimpleVectorStore.search()` |
| 增强 Prompt | 将检索文档作为上下文注入 prompt | `build_rag_prompt()` |
| ReAct 循环 | Thought→Action→Observation 的推理-行动交替 | `ReActAgent.run()` |
| 工具集 | Agent 可调用的外部功能（搜索、计算器、天气 API） | `MockTools` 类 |
| 多级回退 | 语义嵌入→TF-IDF，ChromaDB→内存字典，LLM→模拟回答 | `check_environment()` |

## 完整代码

<<< @/systems/rag-agent/code/demo.py
