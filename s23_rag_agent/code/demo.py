# -*- coding: utf-8 -*-
"""
s23 RAG 与 AI Agent — 演示代码
===============================
功能：
  1. 从零构建完整的 RAG 系统（文本切分 → 嵌入 → 向量存储 → 检索 → 生成）
  2. 实现简单的 ReAct Agent（Thought → Action → Observation 循环）
  3. 对比 RAG vs 非 RAG 回答的差异

每个函数都有中文 docstring，每行逻辑代码都有中文注释。
运行方式：在 s23_rag_agent/ 目录下执行 python code/demo.py

依赖：pip install numpy scikit-learn sentence-transformers chromadb openai
注意：
  - 如果没有 sentence-transformers，会使用简单的 TF-IDF 回退方案
  - 如果没有 OpenAI API key，LLM 部分将使用模拟输出演示流程
"""

import os
import sys
import re
import time
import warnings
import json
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

warnings.filterwarnings("ignore")

# ============================================================================
# 第 0 部分：环境检测与配置
# ============================================================================

# 全局标志
HAS_EMBEDDING_MODEL = False
HAS_CHROMADB = False
HAS_OPENAI = False

def check_environment():
    """检测可用依赖，启用对应的功能。"""
    global HAS_EMBEDDING_MODEL, HAS_CHROMADB, HAS_OPENAI

    print("[环境检测]")

    # 检测 embedding 模型
    try:
        from sentence_transformers import SentenceTransformer
        HAS_EMBEDDING_MODEL = True
        print("  ✓ sentence-transformers 可用 — 使用语义嵌入")
    except ImportError:
        print("  ✗ sentence-transformers 不可用 — 使用 TF-IDF 回退方案")
        print("    安装以获得更好的效果: pip install sentence-transformers")

    # 检测 ChromaDB
    try:
        import chromadb
        HAS_CHROMADB = True
        print("  ✓ ChromaDB 可用 — 使用向量数据库")
    except ImportError:
        print("  ✗ ChromaDB 不可用 — 使用内存字典回退方案")
        print("    安装: pip install chromadb")

    # 检测 OpenAI
    HAS_OPENAI = os.environ.get("OPENAI_API_KEY") is not None
    if HAS_OPENAI:
        print("  ✓ OpenAI API key 已配置 — 使用 GPT 生成回答")
    else:
        print("  ✗ 未检测到 OPENAI_API_KEY — LLM 将使用模拟输出")
        print("    设置环境变量: set OPENAI_API_KEY=your-key")


# ============================================================================
# 第 1 部分：文档语料库
# ============================================================================

def load_document_corpus() -> List[Dict[str, str]]:
    """
    加载内置的中文知识文档。这些文档模拟了知识库中的内容，
    涵盖了多个主题领域。

    返回:
        documents: 文档列表，每个文档包含 title 和 content
    """
    documents = [
        {
            "title": "人工智能概论",
            "content": """
            人工智能（Artificial Intelligence，简称AI）是计算机科学的一个重要分支，旨在创建能够模拟、
            延伸和扩展人类智能的理论、方法、技术和应用系统。AI的研究领域包括机器学习、自然语言处理、
            计算机视觉、语音识别、专家系统和机器人学等。

            机器学习是AI的核心技术之一。它让计算机系统能够通过经验自动改进性能，而无需显式编程。
            深度学习是机器学习的一个子集，使用多层人工神经网络来处理复杂的模式识别任务。

            近年来，大语言模型（LLM）如GPT、Claude和Gemini的出现，标志着AI发展的新纪元。
            这些模型通过在海量文本数据上训练，展现出了令人惊叹的语言理解和生成能力。
            """,
        },
        {
            "title": "Python编程语言",
            "content": """
            Python是由Guido van Rossum于1991年创建的高级编程语言。它以简洁清晰的语法和强大的
            可读性而闻名，是目前世界上最流行的编程语言之一。

            Python的主要特点包括：
            1. 简洁易读的语法，使用缩进来定义代码块
            2. 动态类型系统，无需声明变量类型
            3. 丰富的标准库和第三方库生态系统（如NumPy、Pandas、PyTorch）
            4. 跨平台兼容性，支持Windows、macOS和Linux
            5. 多范式支持，包括面向对象、函数式和过程式编程

            Python在数据科学、Web开发、自动化脚本、人工智能和科学计算等领域有广泛应用。
            特别是PyTorch和TensorFlow这两个深度学习框架，都以Python作为主要接口语言。
            """,
        },
        {
            "title": "气候变化与环境",
            "content": """
            气候变化是指地球气候系统的长期统计特性发生显著变化。自工业革命以来，人类活动导致的
            温室气体排放急剧增加，主要包括二氧化碳（CO2）、甲烷（CH4）和氧化亚氮（N2O）。

            全球平均气温已经比工业化前水平上升了约1.1°C，导致了一系列严重后果：
            - 海平面上升：冰川和冰盖融化导致全球海平面以每年约3.3毫米的速度上升
            - 极端天气事件增多：热浪、干旱、洪水和飓风的频率和强度都在增加
            - 生态系统破坏：许多物种面临灭绝风险，珊瑚礁大规模白化
            - 农业生产受影响：气候变化改变作物生长周期和适宜种植区域

            为应对气候变化，国际社会于2015年签署了《巴黎协定》，目标是将全球升温控制在2°C以内，
            并努力限制在1.5°C以内。这需要各国大幅减少碳排放，转向可再生能源。
            """,
        },
        {
            "title": "深度学习基础",
            "content": """
            深度学习（Deep Learning）是机器学习的一个分支，使用多层神经网络来学习数据的层次化
            表示。其灵感来源于人脑神经元的结构和功能。

            反向传播算法（Backpropagation）是训练深度神经网络的核心算法。它通过链式法则计算
            损失函数对每个参数的梯度，然后使用梯度下降等优化器更新参数：
            θ_new = θ_old - η * ∇L(θ)

            常见的深度学习架构包括：
            - 卷积神经网络（CNN）：擅长处理图像和空间数据
            - 循环神经网络（RNN）和LSTM：擅长处理序列数据
            - Transformer：基于自注意力机制，是目前LLM的基础架构
            - 生成对抗网络（GAN）：用于生成新的数据样本
            - 扩散模型（Diffusion Model）：图像生成的最新范式

            Transformer架构由Google在2017年的论文"Attention Is All You Need"中提出，
            它抛弃了RNN的序列计算，完全依赖自注意力机制，实现了高效的并行训练。
            """,
        },
        {
            "title": "RAG技术详解",
            "content": """
            检索增强生成（Retrieval-Augmented Generation，RAG）是一种将信息检索与文本生成
            相结合的技术，旨在提升大语言模型回答的准确性和时效性。

            RAG的工作流程分为三个阶段：
            1. 索引阶段（离线）：将知识库文档切割成小块（chunks），用嵌入模型编码为向量，
               存入向量数据库。
            2. 检索阶段（在线）：将用户查询编码为向量，在向量数据库中搜索最相关的K个文档块。
            3. 生成阶段：将检索到的文档块作为上下文注入Prompt，让LLM基于真实信息生成回答。

            RAG的核心优势：
            - 减少幻觉：模型回答基于检索到的真实文档
            - 知识可更新：只需更新知识库，无需重新训练模型
            - 可溯源：可以标注回答的信息来源
            - 成本低：无需微调大模型即可扩展知识

            向量数据库（如Chroma、FAISS、Milvus）是RAG的关键基础设施，它们支持高效的
            近似最近邻（ANN）搜索，可以在毫秒级别从百万级向量中找到最相似的结果。
            """,
        },
        {
            "title": "健康饮食指南",
            "content": """
            均衡的营养摄入是维持健康的基础。人体需要六大营养素：碳水化合物、蛋白质、脂肪、
            维生素、矿物质和水。

            中国居民膳食指南（2022）建议：
            1. 食物多样，谷类为主：每天摄入12种以上食物，每周25种以上
            2. 多吃蔬果、奶类、大豆：每天摄入300-500克蔬菜，200-350克水果
            3. 适量吃鱼、禽、蛋、瘦肉：每周吃鱼280-525克
            4. 少盐少油，控糖限酒：每天食盐不超过5克，烹调油25-30克
            5. 吃动平衡，健康体重：每周至少进行5天中等强度身体活动

            研究表明，地中海饮食模式（富含橄榄油、蔬菜、水果、全谷物、鱼类）对心血管健康
            有显著益处，可以降低心脏病和中风的风险。
            """,
        },
        {
            "title": "太阳系与天文知识",
            "content": """
            太阳系由太阳和围绕它运行的所有天体组成，包括八颗行星、它们的卫星、矮行星
            （如冥王星）、小行星和彗星。

            八颗行星按距离太阳由近到远依次是：
            1. 水星（Mercury）：距离太阳最近，没有大气层，表面温度极大变化
            2. 金星（Venus）：最热的行星，浓厚的二氧化碳大气造成强烈的温室效应
            3. 地球（Earth）：已知唯一存在生命的天体，拥有液态水
            4. 火星（Mars）：被称为"红色星球"，拥有太阳系最高的山——奥林帕斯山
            5. 木星（Jupiter）：太阳系最大的行星，质量是其他所有行星总和的2.5倍
            6. 土星（Saturn）：以壮观的环系统闻名，主要由冰和岩石碎片组成
            7. 天王星（Uranus）：侧躺着自转的冰巨星
            8. 海王星（Neptune）：太阳系风速最快的行星，风速可达2100公里/小时

            太阳是一颗中等大小的恒星（G型主序星），其质量占太阳系总质量的99.86%。
            太阳核心的温度高达约1500万摄氏度，通过核聚变将氢转化为氦，释放巨大能量。
            """,
        },
    ]
    return documents


# ============================================================================
# 第 2 部分：文本切分（Chunking）
# ============================================================================

def split_text_into_chunks(
    text: str,
    chunk_size: int = 300,
    chunk_overlap: int = 50
) -> List[str]:
    """
    将长文本切分为固定大小的块（chunk），块之间有重叠以保证语义连续性。

    切割策略：
    1. 优先在句号、换行等自然分隔处切割
    2. 如果句子本身太长，则在字符级别切割
    3. 相邻块之间有 chunk_overlap 个字符的重叠

    参数:
        text: 输入的长文本
        chunk_size: 每个块的近似最大字符数
        chunk_overlap: 相邻块之间的重叠字符数

    返回:
        chunks: 文本块列表
    """
    # 步骤 1: 按段落/句子分隔符初步分割
    # 用正则表达式匹配中英文句子分隔符
    separators = [r'\n\n', r'\n', r'[。！？!?]', r'[，,；;]', r'\s+']
    sentences = []
    remaining = text.strip()

    # 简单按句号、换行等切分
    for sep_pattern in separators:
        if not remaining:
            break
        parts = re.split(f'({sep_pattern})', remaining)
        new_sentences = []
        for part in parts:
            if re.match(sep_pattern, part):
                if new_sentences:
                    new_sentences[-1] += part  # 标点附加到前一句
            else:
                part = part.strip()
                if part:
                    new_sentences.append(part)
        if new_sentences:
            sentences = new_sentences
            remaining = ""

    if not sentences and remaining.strip():
        sentences = [remaining.strip()]

    # 步骤 2: 将句子组装成 chunk
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        # 如果加上当前句子后超过 chunk_size，则保存当前 chunk 并开始新的
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # 保留重叠部分：从当前 chunk 末尾取 overlap 字符
            overlap_text = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else current_chunk
            current_chunk = overlap_text + sentence
        else:
            current_chunk += sentence

    # 添加最后一个 chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# ============================================================================
# 第 3 部分：嵌入模型（支持多种后端）
# ============================================================================

class EmbeddingModel:
    """
    文本嵌入模型封装。优先使用 sentence-transformers，回退到 TF-IDF。

    TF-IDF 回退方案说明：
    TF-IDF 是一种基于词频的稀疏向量表示，虽然不如语义嵌入精确，
    但在关键词匹配场景下仍可工作。它不需要 GPU，也不需要下载模型。
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        初始化嵌入模型。

        参数:
            model_name: sentence-transformers 模型名称
        """
        self.model_name = model_name
        self._model = None
        self._tfidf_vectorizer = None
        self._tfidf_matrix = None
        self._is_semantic = False

        if HAS_EMBEDDING_MODEL:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"[嵌入] 加载语义模型: {model_name}")
                self._model = SentenceTransformer(model_name)
                self._is_semantic = True
                # 获取嵌入维度
                test_emb = self._model.encode(["test"])
                print(f"  嵌入维度: {test_emb.shape[1]}")
            except Exception as e:
                print(f"[嵌入] 语义模型加载失败: {e}，回退到 TF-IDF")
                self._model = None
                self._is_semantic = False
        else:
            print("[嵌入] 使用 TF-IDF 词频向量（回退方案）")

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        将文本列表编码为向量矩阵。

        参数:
            texts: 文本字符串列表

        返回:
            embeddings: 向量矩阵，shape (len(texts), embedding_dim)
        """
        if self._is_semantic and self._model is not None:
            # 使用语义嵌入
            embeddings = self._model.encode(texts, show_progress_bar=False)
            # L2 归一化
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # 避免除零
            embeddings = embeddings / norms
            return embeddings
        else:
            # TF-IDF 回退
            return self._encode_tfidf(texts)

    def _encode_tfidf(self, texts: List[str]) -> np.ndarray:
        """
        使用 TF-IDF 进行文本向量化（回退方案）。

        参数:
            texts: 文本列表
        返回:
            TF-IDF 向量矩阵
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        # 如果还没创建 vectorizer 或这是新文档，重新 fit
        if self._tfidf_vectorizer is None:
            self._tfidf_vectorizer = TfidfVectorizer(
                max_features=384,        # 限制最大特征数
                token_pattern=r'(?u)\b\w+\b',  # 匹配中文和英文词
            )
            # 用当前 texts 训练 vectorizer
            self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(texts).toarray()
        else:
            self._tfidf_matrix = self._tfidf_vectorizer.transform(texts).toarray()

        # L2 归一化
        norms = np.linalg.norm(self._tfidf_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return self._tfidf_matrix / norms


# ============================================================================
# 第 4 部分：向量存储
# ============================================================================

class SimpleVectorStore:
    """
    简单的向量存储（支持 ChromaDB 和内存字典两种后端）。

    ChromaDB 后端：适合生产环境，支持持久化和高效搜索。
    内存字典后端：适合演示和学习，不需要额外安装。
    """

    def __init__(self, use_chromadb: bool = False, collection_name: str = "rag_demo"):
        """
        初始化向量存储。

        参数:
            use_chromadb: 是否使用 ChromaDB 向量数据库
            collection_name: 集合名称（仅 ChromaDB）
        """
        self.use_chromadb = use_chromadb and HAS_CHROMADB
        self._chroma_client = None
        self._chroma_collection = None

        # 内存存储（回退方案）
        self._documents: List[str] = []        # 存储文档文本
        self._embeddings: List[np.ndarray] = [] # 存储嵌入向量
        self._metadata: List[dict] = []         # 存储元数据

        if self.use_chromadb:
            try:
                import chromadb
                self._chroma_client = chromadb.Client()
                # 如果 collection 已存在则先删除
                try:
                    self._chroma_client.delete_collection(collection_name)
                except Exception:
                    pass
                self._chroma_collection = self._chroma_client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}  # 使用余弦距离
                )
                print(f"[向量库] 使用 ChromaDB (collection: {collection_name})")
            except Exception as e:
                print(f"[向量库] ChromaDB 初始化失败: {e}，使用内存存储")
                self.use_chromadb = False
        else:
            print("[向量库] 使用内存字典存储")

    def add(self, documents: List[str], embeddings: np.ndarray, metadata: List[dict] = None):
        """
        将文档和对应的嵌入向量添加到存储中。

        参数:
            documents: 文档文本列表
            embeddings: 对应的嵌入向量矩阵 (N, d)
            metadata: 可选的元数据列表
        """
        if metadata is None:
            metadata = [{}] * len(documents)

        if self.use_chromadb:
            # ChromaDB 后端：一次性添加所有文档
            ids = [f"doc_{i + len(self._documents)}" for i in range(len(documents))]
            self._chroma_collection.add(
                embeddings=embeddings.tolist(),
                documents=documents,
                metadatas=metadata,
                ids=ids
            )
        else:
            # 内存后端：逐条添加
            for doc, emb, meta in zip(documents, embeddings, metadata):
                self._documents.append(doc)
                self._embeddings.append(emb / np.linalg.norm(emb))  # 归一化
                self._metadata.append(meta)

        # 同时更新内存索引（保持两种后端同步）
        for doc, emb, meta in zip(documents, embeddings, metadata):
            if not self.use_chromadb:
                continue  # 已在上面添加
            self._documents.append(doc)
            self._embeddings.append(emb / np.linalg.norm(emb))
            self._metadata.append(meta)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        搜索与查询嵌入最相似的 top_k 个文档。

        参数:
            query_embedding: 查询向量 (d,)，应已 L2 归一化
            top_k: 返回的文档数量

        返回:
            results: 列表，每项包含 {"document": str, "score": float, "metadata": dict}
        """
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)

        if self.use_chromadb:
            # ChromaDB 搜索
            chroma_results = self._chroma_collection.query(
                query_embeddings=[query_embedding.flatten().tolist()],
                n_results=min(top_k, len(self._documents))
            )
            results = []
            if chroma_results["documents"] and chroma_results["documents"][0]:
                for doc, dist, meta in zip(
                    chroma_results["documents"][0],
                    chroma_results["distances"][0],
                    chroma_results["metadatas"][0]
                ):
                    # ChromaDB 的 cosine distance = 1 - cosine_similarity
                    score = 1.0 - dist
                    results.append({"document": doc, "score": score, "metadata": meta})
            return results
        else:
            # 内存搜索：计算余弦相似度
            if not self._embeddings:
                return []

            # 将嵌入矩阵转换为 (N, d) 的 numpy 数组
            emb_matrix = np.stack(self._embeddings, axis=0)  # (N, d)

            # 计算余弦相似度
            similarities = emb_matrix @ query_embedding.flatten()  # (N,)

            # 获取 top_k 索引
            top_indices = np.argsort(-similarities)[:top_k]

            # 构建结果
            results = []
            for idx in top_indices:
                if idx < len(self._documents):
                    results.append({
                        "document": self._documents[idx],
                        "score": float(similarities[idx]),
                        "metadata": self._metadata[idx] if idx < len(self._metadata) else {}
                    })
            return results

    def size(self) -> int:
        """返回存储中的文档数量。"""
        return len(self._documents)

    def reset(self):
        """清空所有数据。"""
        self._documents = []
        self._embeddings = []
        self._metadata = []
        if self._chroma_client is not None:
            self._chroma_client.reset()


# ============================================================================
# 第 5 部分：构建 RAG 系统
# ============================================================================

class RAGSystem:
    """
    完整的 RAG（检索增强生成）系统。

    RAG 流水线：
    1. 加载文档 → 文本切分 → 生成嵌入 → 存入向量库（索引阶段）
    2. 接收查询 → 嵌入查询 → 检索 top-k 文档 → 构建增强 prompt → LLM 生成（查询阶段）
    """

    def __init__(self, embedding_model: EmbeddingModel, vector_store: SimpleVectorStore):
        """
        初始化 RAG 系统。

        参数:
            embedding_model: 文本嵌入模型
            vector_store: 向量存储后端
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def index_documents(self, documents: List[Dict[str, str]],
                        chunk_size: int = 300, chunk_overlap: int = 50):
        """
        索引阶段：处理文档并存入向量数据库。

        参数:
            documents: 文档列表，每项包含 title 和 content
            chunk_size: 每个文本块的最大字符数
            chunk_overlap: 块之间的重叠字符数
        """
        print(f"\n[RAG索引] 开始处理 {len(documents)} 篇文档...")

        all_chunks = []
        all_metadata = []

        for doc in documents:
            title = doc["title"]
            content = doc["content"]

            # 切分文档为块
            chunks = split_text_into_chunks(content, chunk_size, chunk_overlap)

            for i, chunk in enumerate(chunks):
                # 在每块前面加上标题，帮助检索定位
                enriched_chunk = f"[{title}] {chunk}"
                all_chunks.append(enriched_chunk)
                all_metadata.append({
                    "title": title,
                    "chunk_index": i,
                    "chunk_total": len(chunks)
                })

        print(f"  共切分为 {len(all_chunks)} 个文本块 (chunk_size={chunk_size}, overlap={chunk_overlap})")

        if not all_chunks:
            print("  警告：没有生成任何块")
            return

        # 生成嵌入
        print(f"  正在生成嵌入向量...")
        embeddings = self.embedding_model.encode(all_chunks)
        print(f"  嵌入矩阵形状: {embeddings.shape}")

        # 存入向量库
        self.vector_store.add(all_chunks, embeddings, all_metadata)
        print(f"  ✓ 索引完成！向量库中共 {self.vector_store.size()} 条记录")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        检索阶段：根据查询找到最相关的文档块。

        参数:
            query: 用户查询文本
            top_k: 返回的文档数量

        返回:
            results: 最相关的文档块列表
        """
        # 生成查询嵌入
        query_embedding = self.embedding_model.encode([query])[0]

        # 向量搜索
        results = self.vector_store.search(query_embedding, top_k=top_k)
        return results

    def build_rag_prompt(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        构建 RAG 增强的 prompt，将检索到的文档作为上下文注入。

        参数:
            query: 原始用户查询
            retrieved_docs: 检索到的相关文档列表

        返回:
            prompt: 完整的增强 prompt 字符串
        """
        # 构造上下文部分
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            title = doc["metadata"].get("title", "未知来源")
            context_parts.append(f"[来源 {i}: {title}]\n{doc['document']}")

        context_text = "\n\n---\n\n".join(context_parts)

        # 构造完整 prompt
        prompt = f"""你是一个基于知识库的问答助手。请仅根据以下提供的参考资料回答问题。
如果参考资料中没有足够的信息来回答，请明确说"根据现有资料无法确定"。

=== 参考资料 ===
{context_text}

=== 用户问题 ===
{query}

请用中文回答，并在可能的情况下引用具体的来源。"""
        return prompt


# ============================================================================
# 第 6 部分：LLM 调用
# ============================================================================

def simulate_llm_response(prompt: str) -> str:
    """
    模拟 LLM 回答（当无法访问真实 LLM 时的回退方案）。

    这不是真正的 LLM，而是基于 prompt 中检索到的文档内容拼接出模拟回答。
    目的是展示 RAG 流程，而非生成高质量回答。

    参数:
        prompt: RAG 增强后的 prompt

    返回:
        response: 模拟的回答文本
    """
    # 从 prompt 中提取关键信息构建模拟回答
    # 提取用户问题
    query_match = re.search(r'=== 用户问题 ===\s*\n(.+)', prompt)
    query = query_match.group(1).strip() if query_match else "未知问题"

    # 提取来源标题
    sources = re.findall(r'\[来源 \d+: (.+?)\]', prompt)
    source_list = "、".join(set(sources)) if sources else "参考资料"

    # 从第一个检索到的文档中提取前 200 字符作为「回答」
    doc_match = re.search(r'\[来源 1: .+?\]\n(.+?)(?:\n---|\Z)', prompt, re.DOTALL)
    if doc_match:
        snippet = doc_match.group(1).strip()[:300]
    else:
        snippet = "根据提供的参考资料，这个问题可以从多个角度来理解。"

    response = f"""[模拟回答 - 基于 RAG 检索结果]

关于您的问题「{query}」，根据检索到的资料（来源：{source_list}）：

{snippet}

---
注意：这是一个模拟回答。连接真实 LLM（如 OpenAI API 或本地 Qwen 模型）
可以获得更高质量的回答。请设置 OPENAI_API_KEY 环境变量以启用真实 LLM。"""
    return response


def call_llm(prompt: str, model: str = "gpt-3.5-turbo") -> str:
    """
    调用 LLM（优先 OpenAI API，否则使用模拟回答）。

    参数:
        prompt: 输入 prompt
        model: OpenAI 模型名称

    返回:
        response: LLM 的回答
    """
    if HAS_OPENAI:
        try:
            from openai import OpenAI
            client = OpenAI()  # 自动从环境变量读取 API key

            print("  [LLM] 调用 OpenAI API...")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一个基于知识库的问答助手，请用中文回答。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,   # 较低的温度以获得更确定的回答
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"  [LLM] OpenAI API 调用失败: {e}")
            print(f"  [LLM] 回退到模拟回答")

    return simulate_llm_response(prompt)


# ============================================================================
# 第 7 部分：ReAct Agent
# ============================================================================

class MockTools:
    """
    Agent 的模拟工具集。

    在真实场景中，这些工具会调用实际的 API、搜索引擎、计算器等。
    这里用模拟实现来展示 ReAct 循环的概念。
    """

    @staticmethod
    def calculator(expression: str) -> str:
        """
        计算器工具：安全地计算数学表达式。

        参数:
            expression: 数学表达式字符串，如 "2 + 3 * 4"
        返回:
            计算结果字符串
        """
        try:
            # 安全地计算表达式（仅限基本数学运算）
            allowed_chars = set("0123456789+-*/().%^ ")
            if not all(c in allowed_chars for c in expression):
                return "错误：表达式包含不允许的字符"
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            return f"计算错误：{e}"

    @staticmethod
    def search(query: str) -> str:
        """
        搜索工具：模拟知识检索。

        参数:
            query: 搜索查询
        返回:
            搜索结果（模拟）
        """
        # 模拟搜索返回结果
        mock_results = {
            "北京": "北京是中国的首都，地处华北平原北部。",
            "天气": "今天天气晴朗，气温 22°C，湿度 45%。",
            "python": "Python 是一种高级编程语言，由 Guido van Rossum 创建。",
            "AI": "人工智能（AI）是计算机科学的分支，研究智能机器的构建。",
        }
        for key, value in mock_results.items():
            if key.lower() in query.lower():
                return f"搜索结果：「{key}」— {value}"
        return f"搜索结果：未找到与「{query}」直接相关的内容。请尝试更具体的关键词。"

    @staticmethod
    def weather(city: str) -> str:
        """
        天气工具：查询城市天气（模拟）。

        参数:
            city: 城市名称
        返回:
            天气信息字符串
        """
        mock_weather = {
            "北京": "北京今日天气：晴，温度 22°C，湿度 45%，风力 3级，降雨概率 10%",
            "上海": "上海今日天气：多云转小雨，温度 25°C，湿度 70%，风力 2级，降雨概率 65%",
            "广州": "广州今日天气：雷阵雨，温度 28°C，湿度 85%，风力 4级，降雨概率 90%",
        }
        return mock_weather.get(city, f"{city}的天气数据暂不可用，请检查城市名称。")


class ReActAgent:
    """
    ReAct (Reasoning + Acting) Agent 实现。

    核心循环：
    Thought (思考) → Action (行动) → Observation (观察) → Thought → ... → Final Answer

    Agent 通过分析 LLM 输出的 Thought/Action 模式来决定下一步操作。
    """

    def __init__(self, tools: MockTools, max_steps: int = 5):
        """
        初始化 ReAct Agent。

        参数:
            tools: 可用的工具集
            max_steps: 最大执行步数（防止无限循环）
        """
        self.tools = tools
        self.max_steps = max_steps
        self.history: List[Dict[str, str]] = []  # 记录执行历史

    def run(self, query: str) -> str:
        """
        运行 ReAct 循环处理用户查询。

        参数:
            query: 用户的问题

        返回:
            answer: Agent 的最终回答
        """
        self.history = []
        print(f"\n{'=' * 60}")
        print(f"[ReAct Agent] 处理查询: {query}")
        print(f"{'=' * 60}")

        for step in range(1, self.max_steps + 1):
            print(f"\n--- Step {step} ---")

            # ---- Thought: 根据历史和当前状态推理 ----
            thought = self._generate_thought(query, step)
            print(f"  Thought: {thought}")

            # ---- 判断是否到达最终回答 ----
            if self._is_final_answer(thought):
                final_answer = self._extract_final_answer(thought)
                self.history.append({"step": step, "type": "final", "content": final_answer})
                print(f"  Final Answer: {final_answer}")
                return final_answer

            # ---- Action: 解析并执行工具调用 ----
            action = self._parse_action(thought)
            if action is None:
                # 无法解析动作，当作最终回答
                self.history.append({"step": step, "type": "final", "content": thought})
                return thought

            tool_name, tool_input = action
            print(f"  Action: {tool_name}({tool_input})")

            # ---- Observation: 执行工具并获得结果 ----
            observation = self._execute_tool(tool_name, tool_input)
            print(f"  Observation: {observation}")

            self.history.append({
                "step": step,
                "type": "action",
                "thought": thought,
                "action": f"{tool_name}({tool_input})",
                "observation": observation
            })

        # 超过最大步数，返回当前状态
        return f"处理超时（已执行 {self.max_steps} 步）。基于已有信息：{self.history}"

    def _generate_thought(self, query: str, step: int) -> str:
        """
        生成 Agent 的「思考」步骤。

        在实际系统中，这会调用 LLM 来生成思考和行动计划。
        这里使用基于规则的模拟来展示 ReAct 循环。

        参数:
            query: 用户原始问题
            step: 当前步骤数

        返回:
            thought: Agent 的思考文本
        """
        # 基于查询关键词选择工具策略
        if step == 1:
            if any(kw in query for kw in ["天气", "weather", "气温", "下雨"]):
                return "我需要查询天气信息。让我提取城市名称并使用天气工具。"
            elif any(kw in query for kw in ["计算", "算", "等于", "+", "-", "*"]):
                return "这是一个计算问题。我需要提取数学表达式并使用计算器。"
            elif any(kw in query for kw in ["什么是", "定义", "解释", "介绍", "知识"]):
                return "这是一个知识类问题。我需要使用搜索工具查找相关信息。"
            else:
                return "我需要先搜索相关信息来理解这个问题。"

        # Step 2+: 基于之前的观察决定下一步
        if step >= 2 and self.history:
            last_obs = self.history[-1].get("observation", "")
            if "未找到" in last_obs or "不可用" in last_obs:
                return "前一步没有获得足够信息。让我尝试更广泛的搜索或换个关键词。"
            elif "天气" in query and "温度" in last_obs:
                if "伞" in query or "带伞" in query:
                    return "我已有天气数据（温度和降雨概率）。现在判断是否需要带伞。"
                else:
                    return "已获取天气信息，可以给出最终回答。"
            else:
                return "已获取足够信息，可以给出最终回答。"

        return "已收集完必要信息，现在给出最终回答。"

    def _parse_action(self, thought: str) -> Optional[Tuple[str, str]]:
        """
        从 Thought 中解析出要执行的动作。

        参数:
            thought: Agent 的思考文本

        返回:
            (工具名称, 工具参数) 或 None（表示这是最终回答）
        """
        if "天气" in thought:
            return ("weather", "北京")
        elif "计算" in thought:
            return ("calculator", "3 * 15 + 28")
        elif "搜索" in thought:
            return ("search", "人工智能")
        elif "最终回答" in thought or "给出" in thought or "可以回答" in thought:
            return None
        else:
            return ("search", "通用查询")

    def _is_final_answer(self, thought: str) -> bool:
        """判断思考文本是否表示这是最终回答。"""
        final_markers = ["最终回答", "给出答案", "可以回答", "答案是", "结论是"]
        return any(marker in thought for marker in final_markers)

    def _extract_final_answer(self, thought: str) -> str:
        """从最终思考中提取答案。如果之前有观察结果，则整合。"""
        # 整合历史观察信息
        if self.history:
            last_obs = self.history[-1].get("observation", "")
            if last_obs:
                return f"根据查询结果：{last_obs}\n综上，这是基于可用信息的回答。"
        return thought

    def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        """
        执行指定的工具调用。

        参数:
            tool_name: 工具名称
            tool_input: 工具输入参数

        返回:
            observation: 工具执行结果
        """
        tool_map = {
            "calculator": self.tools.calculator,
            "search": self.tools.search,
            "weather": self.tools.weather,
        }

        tool_func = tool_map.get(tool_name)
        if tool_func is None:
            return f"错误：未知工具 '{tool_name}'。可用工具：{list(tool_map.keys())}"

        try:
            return tool_func(tool_input)
        except Exception as e:
            return f"工具执行错误：{e}"


# ============================================================================
# 第 8 部分：可视化与结果展示
# ============================================================================

def print_separator(title: str):
    """打印格式化的分隔标题。"""
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def demo_rag_with_vs_without(rag_system: RAGSystem, query: str):
    """
    对比展示：使用 RAG vs 不使用 RAG 的回答差异。

    参数:
        rag_system: 已索引的 RAG 系统
        query: 测试查询
    """
    print_separator(f"RAG 对比实验 — 查询: 「{query}」")

    # --- RAG 模式 ---
    print(f"\n  [RAG 模式 — 有知识库支持]")
    retrieved = rag_system.retrieve(query, top_k=3)

    print(f"  检索到的 {len(retrieved)} 个文档块：")
    for i, doc in enumerate(retrieved, 1):
        title = doc["metadata"].get("title", "未知")
        score = doc["score"]
        snippet = doc["document"][:100].replace('\n', ' ')
        print(f"    {i}. [{title}] (相似度: {score:.3f})")
        print(f"       片段: {snippet}...")

    # 用 RAG prompt 调用 LLM
    rag_prompt = rag_system.build_rag_prompt(query, retrieved)
    rag_answer = call_llm(rag_prompt)

    print(f"\n  [RAG 回答]")
    print(f"  {rag_answer[:400]}...")

    # --- 无 RAG 模式 ---
    print(f"\n  [无 RAG 模式 — 纯 LLM，无知识库]")
    no_rag_prompt = f"请回答以下问题：{query}\n请用中文回答。"
    no_rag_answer = call_llm(no_rag_prompt)

    print(f"\n  [纯 LLM 回答]")
    print(f"  {no_rag_answer[:400]}...")

    print(f"\n  [对比总结]")
    print(f"  RAG 回答：基于 {len(retrieved)} 篇具体文档，可溯源，事实性更强")
    print(f"  纯 LLM 回答：依赖模型参数化知识，可能过时或不准确，但更流畅")


# ============================================================================
# 第 9 部分：主程序
# ============================================================================

def main():
    """
    主程序：构建 RAG 系统 + ReAct Agent 并运行演示。

    流程：
    1. 检测环境
    2. 加载文档语料库
    3. 构建 RAG 索引
    4. 演示 RAG 检索
    5. 对比 RAG vs 非 RAG
    6. 演示 ReAct Agent
    7. 可视化总结
    """
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 8 + "s23 RAG 与 AI Agent — 从零构建完整系统" + " " * 18 + "║")
    print("║" + " " * 8 + "检索增强生成 · ReAct 智能体 · 效果对比" + " " * 20 + "║")
    print("╚" + "═" * 68 + "╝")

    # ---- 步骤 1: 环境检测 ----
    check_environment()

    # ---- 步骤 2: 加载文档 ----
    print_separator("步骤 1: 加载文档语料库")
    documents = load_document_corpus()
    print(f"  加载了 {len(documents)} 篇文档:")
    for doc in documents:
        print(f"    - {doc['title']} ({len(doc['content'])} 字符)")

    # ---- 步骤 3: 初始化嵌入模型和向量库 ----
    print_separator("步骤 2: 初始化嵌入模型和向量库")

    # 创建嵌入模型
    embedder = EmbeddingModel(model_name="all-MiniLM-L6-v2")

    # 创建向量存储
    vector_store = SimpleVectorStore(use_chromadb=HAS_CHROMADB, collection_name="s23_demo")

    # 创建 RAG 系统
    rag = RAGSystem(embedding_model=embedder, vector_store=vector_store)

    # ---- 步骤 4: 索引文档 ----
    print_separator("步骤 3: 构建 RAG 索引")
    rag.index_documents(documents, chunk_size=300, chunk_overlap=50)

    # ---- 步骤 5: 演示检索 ----
    test_queries = [
        "什么是深度学习？",
        "Python有哪些主要特点？",
        "气候变化对地球有什么影响？",
        "健康饮食应该注意什么？",
        "太阳系有哪些行星？",
    ]

    print_separator("步骤 4: RAG 检索演示")

    for query in test_queries[:3]:  # 只演示前 3 个
        print(f"\n  查询: 「{query}」")
        results = rag.retrieve(query, top_k=2)
        for i, doc in enumerate(results, 1):
            title = doc["metadata"].get("title", "?")
            score = doc["score"]
            print(f"  {i}. [{title}] (相关性: {score:.3f})")

    # ---- 步骤 6: RAG vs 非 RAG 对比 ----
    print_separator("步骤 5: RAG vs 纯 LLM 对比")
    demo_rag_with_vs_without(rag, "深度学习中的反向传播算法是如何工作的？")

    # ---- 步骤 7: ReAct Agent 演示 ----
    print_separator("步骤 6: ReAct Agent 演示")

    tools = MockTools()
    agent = ReActAgent(tools=tools, max_steps=4)

    # 演示 1: 天气查询（多步推理）
    print("\n  --- ReAct 演示 1: 天气查询 ---")
    answer1 = agent.run("北京今天天气怎么样？需要带伞吗？")

    # 演示 2: 知识查询
    print("\n  --- ReAct 演示 2: 知识搜索 ---")
    agent2 = ReActAgent(tools=tools, max_steps=3)
    answer2 = agent2.run("什么是人工智能？")

    # 演示 3: 计算任务
    print("\n  --- ReAct 演示 3: 计算任务 ---")
    agent3 = ReActAgent(tools=tools, max_steps=3)
    answer3 = agent3.run("帮我算一下 3乘以15加28等于多少？")

    # ---- 步骤 8: 可视化 Agent 执行历史 ----
    print_separator("步骤 7: Agent 执行历史总结")

    # 画出 ReAct 循环的痕迹
    print("\n  ReAct 循环轨迹 (演示1 - 天气查询):")
    print(f"  {'─' * 50}")
    for entry in agent.history:
        step = entry["step"]
        if entry["type"] == "action":
            print(f"  Step {step}:")
            print(f"    Thought:     {entry['thought'][:60]}...")
            print(f"    Action:      {entry['action']}")
            print(f"    Observation: {entry['observation'][:60]}...")
        else:
            print(f"  Step {step}:")
            print(f"    Final Answer: {entry['content'][:80]}...")
    print(f"  {'─' * 50}")

    # ---- 最终总结 ----
    print("\n" + "=" * 70)
    print("【s23 总结】")
    print("=" * 70)
    print("  ✓ 理解了 RAG 的三阶段流水线（索引 → 检索 → 生成）")
    print("  ✓ 实现了文本切分、向量嵌入、相似度搜索、向量存储")
    print("  ✓ 体验了 RAG vs 非 RAG 的回答差异")
    print("  ✓ 理解了 ReAct Agent 的 Thought → Action → Observation 循环")
    print()
    print("  RAG 让 LLM 的回答扎根于真实数据")
    print("  Agent 让 LLM 从「对话工具」进化为「行动主体」")
    print("  两者结合 = 既能获取知识、又能执行任务的智能系统")
    print("=" * 70)


if __name__ == "__main__":
    main()
