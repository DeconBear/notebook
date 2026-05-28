# -*- coding: utf-8 -*-
"""
s23 RAG 与 AI Agent — 练习代码
===============================
请完成以下 TODO 任务，巩固对 RAG 和 AI Agent 的理解。

每个 TODO 都有详细的指示和预期输出描述。
建议先阅读 README.md，再尝试独立补全代码。
"""

import re
import numpy as np
from typing import List, Tuple, Optional, Dict


# ============================================================================
# TODO 1: 实现带重叠的文本切分函数
# ============================================================================
def chunk_text_with_overlap(
    text: str,
    chunk_size: int = 200,
    overlap: int = 40
) -> List[str]:
    """
    将长文本切分为重叠的文本块。

    切割策略：
    1. 按句子边界（句号、换行等）初步分割
    2. 将句子组装成不超过 chunk_size 的块
    3. 相邻块之间有 overlap 个字符的重叠

    参数:
        text: 输入的长文本
        chunk_size: 每个块的最大字符数
        overlap: 重叠字符数

    返回:
        chunks: 文本块列表

    提示:
        1. 首先用正则或简单的 split 按句子分割（注意中英文标点）
        2. 遍历句子，累积到 current_chunk
        3. 当 len(current_chunk) + len(sentence) > chunk_size 时：
           a. 保存 current_chunk
           b. 从 current_chunk 末尾取 overlap 字符作为新 current_chunk 的开头
           c. 添加当前句子
        4. 注意处理超长句子（一个句子就超过 chunk_size）的情况
    """
    if not text or not text.strip():
        return []

    # TODO: 步骤 1 — 清理文本并将文本按句子切分
    # 提示: 用正则表达式 re.split 按中英文标点分句
    # 中文标点: 。！？\n
    # 英文标点: .!?\n
    text = text.strip()

    # 使用正则表达式按句子分割
    # 提示: re.split(r'(?<=[。！？.!?\n])', text)
    raw_sentences = None  # ← TODO: 实现句子分割

    # 过滤空句子并去除首尾空白
    sentences = []  # ← TODO: 遍历 raw_sentences，strip 后非空的加入

    if not sentences:
        return [text]  # 如果无法分句，返回整个文本作为一个块

    # TODO: 步骤 2 — 将句子组装成重叠的 chunk
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # TODO: 判断当前块加上新句子后是否超过 chunk_size
        # 如果超过且 current_chunk 不为空，则保存 current_chunk
        # 然后用 current_chunk 末尾的 overlap 字符 + 当前句子 构成新块
        pass  # ← TODO: 实现组装逻辑

    # TODO: 步骤 3 — 添加最后一个未保存的块
    pass  # ← TODO

    return chunks


# ---- 测试 TODO 1 ----
def test_text_chunking():
    """测试带重叠的文本切分功能。"""
    print("=" * 60)
    print("TODO 1 测试: 文本切分函数")
    print("=" * 60)

    # 测试文本：包含多个段落和句子
    test_text = """
    人工智能是计算机科学的重要分支。它旨在创建能够模拟人类智能的系统。
    机器学习是AI的核心技术之一，让计算机从数据中学习规律。

    深度学习使用多层神经网络来处理复杂的模式识别任务。
    近年来，大语言模型如GPT和Claude的出现，标志着AI发展的新纪元。
    这些模型在文本理解和生成方面展现出惊人的能力。
    """

    chunks = chunk_text_with_overlap(test_text, chunk_size=150, overlap=30)

    if chunks is None or len(chunks) == 0:
        print("  TODO 未完成，请补全 chunk_text_with_overlap 函数")
    else:
        print(f"\n  原始文本长度: {len(test_text)} 字符")
        print(f"  切分为 {len(chunks)} 个块 (chunk_size=150, overlap=30):")
        for i, chunk in enumerate(chunks):
            print(f"\n  Chunk {i+1} ({len(chunk)} 字符):")
            print(f"    {chunk[:120]}{'...' if len(chunk) > 120 else ''}")

        # 检查重叠
        if len(chunks) >= 2:
            last_of_first = chunks[0][-30:]
            # 检查第二个 chunk 是否包含第一个的结尾部分
            found_overlap = last_of_first[:15] in chunks[1]
            if found_overlap:
                print(f"\n  ✓ 检测到块间重叠，文本连续性得到保持")
            else:
                print(f"\n  ✗ 相邻块之间缺少重叠，可能导致语义断裂")

        # 检查是否有空块
        empty_chunks = [c for c in chunks if not c.strip()]
        if empty_chunks:
            print(f"  ✗ 发现 {len(empty_chunks)} 个空块")
        else:
            print(f"  ✓ 无空块")

        # 检查是否有严重的句子截断
        for i, chunk in enumerate(chunks):
            if not chunk.endswith(('。', '！', '？', '.', '!', '?', '\n')):
                print(f"  ⚠ Chunk {i+1} 末可能截断了句子 (末尾字符: '{chunk[-3:]}')")

    print()


# ============================================================================
# TODO 2: 实现向量相似度搜索
# ============================================================================
def cosine_similarity_matrix(
    query_vec: np.ndarray,
    doc_matrix: np.ndarray
) -> np.ndarray:
    """
    计算查询向量与文档矩阵中每个向量的余弦相似度。

    公式: cos_sim(q, d_i) = (q · d_i) / (||q|| * ||d_i||)

    参数:
        query_vec: 查询向量，shape (d,)
        doc_matrix: 文档向量矩阵，shape (N, d)

    返回:
        similarities: 余弦相似度数组，shape (N,)

    提示:
        1. 使用 np.dot 或 @ 计算点积
        2. 使用 np.linalg.norm 计算 L2 范数
        3. 注意广播维度
    """
    # TODO: 实现余弦相似度计算

    # 步骤 1: 归一化查询向量
    query_norm = np.linalg.norm(query_vec)  # 查询向量的 L2 范数
    if query_norm < 1e-10:
        return np.zeros(doc_matrix.shape[0])
    query_normalized = None  # ← TODO: query_vec / query_norm

    # 步骤 2: 归一化文档矩阵（每行一个文档向量）
    doc_norms = None  # ← TODO: np.linalg.norm(doc_matrix, axis=1)
    doc_norms = np.where(doc_norms < 1e-10, 1, doc_norms)  # 避免除零
    doc_normalized = None  # ← TODO: doc_matrix / doc_norms[:, np.newaxis]

    # 步骤 3: 计算点积（归一化后的点积 = 余弦相似度）
    similarities = None  # ← TODO: doc_normalized @ query_normalized

    return similarities


def top_k_search(
    query_vec: np.ndarray,
    doc_matrix: np.ndarray,
    doc_texts: List[str],
    top_k: int = 3
) -> List[Tuple[str, float]]:
    """
    在文档向量库中搜索与查询最相似的 top-k 个文档。

    参数:
        query_vec: 查询向量，shape (d,)
        doc_matrix: 文档向量矩阵，shape (N, d)
        doc_texts: 文档文本列表，与 doc_matrix 的行对应
        top_k: 返回的文档数量

    返回:
        results: [(文档文本, 相似度分数), ...] 按分数降序排列

    提示:
        1. 调用上面的 cosine_similarity_matrix 计算相似度
        2. 用 np.argsort 排序（注意降序）
        3. 取前 top_k 个结果
    """
    if len(doc_texts) == 0:
        return []

    # TODO: 实现 top-k 搜索

    # 步骤 1: 计算查询与所有文档的余弦相似度
    similarities = None  # ← TODO: 调用 cosine_similarity_matrix

    # 步骤 2: 获取相似度最高的 top_k 个索引
    # 使用 np.argsort(-similarities) 实现降序
    top_indices = None  # ← TODO

    # 步骤 3: 构建结果列表
    results = []
    # TODO: 遍历 top_indices 的前 top_k 个，每个元素为 (doc_texts[idx], similarities[idx])
    # 确保 idx 不超过 len(doc_texts)

    return results


# ---- 测试 TODO 2 ----
def test_similarity_search():
    """测试相似度搜索的实现。"""
    print("=" * 60)
    print("TODO 2 测试: 向量相似度搜索")
    print("=" * 60)

    np.random.seed(42)
    N, d = 8, 32  # 8 个文档，32 维嵌入

    # 创建模拟文档嵌入
    doc_embs = np.random.randn(N, d).astype(np.float32)

    # 人为让文档 0 和 1 更接近查询（模拟它们与查询更相关）
    doc_embs[0] = np.random.randn(d) * 0.2 + np.array([1.0] * 4 + [0.0] * (d - 4))
    doc_embs[1] = np.random.randn(d) * 0.3 + np.array([0.8] * 4 + [0.0] * (d - 4))

    # 创建查询向量（与文档 0 相关）
    query_vec = np.random.randn(d) * 0.2 + np.array([1.0] * 4 + [0.0] * (d - 4))

    # 模拟文档文本
    doc_texts = [
        "深度学习使用多层神经网络进行模式识别，包括CNN和Transformer等架构。",
        "反向传播算法通过链式法则计算梯度来更新神经网络参数。",
        "气候变化导致全球平均气温上升，极端天气事件增多。",
        "Python是一种广泛使用的高级编程语言，具有简洁的语法。",
        "太阳系的八大行星包括水星、金星、地球、火星等。",
        "健康饮食建议每天摄入多种蔬果和适量蛋白质。",
        "RAG技术将信息检索与文本生成相结合来提升回答准确性。",
        "AI Agent可以自主使用工具并执行多步骤任务。",
    ]

    # 测试余弦相似度
    similarities = cosine_similarity_matrix(query_vec, doc_embs)

    if similarities is None:
        print("  TODO 未完成，请补全 cosine_similarity_matrix 函数")
    else:
        print(f"\n  余弦相似度计算结果:")
        for i, sim in enumerate(similarities):
            print(f"    文档 {i}: {sim:.4f}  {doc_texts[i][:50]}...")

        # 验证：文档 0 应该有最高的相似度
        best_idx = np.argmax(similarities)
        if best_idx == 0:
            print(f"\n  ✓ 文档 0 (深度学习) 获得了最高相似度: {similarities[0]:.4f}")
        else:
            print(f"\n  ⚠ 最高相似度是文档 {best_idx}，预期是文档 0")

    # 测试 top-k 搜索
    results = top_k_search(query_vec, doc_embs, doc_texts, top_k=3)
    if results is None or len(results) == 0:
        print("\n  TODO 未完成，请补全 top_k_search 函数")
    else:
        print(f"\n  Top-{len(results)} 搜索结果:")
        for rank, (text, score) in enumerate(results, 1):
            print(f"    {rank}. [{score:.4f}] {text[:60]}...")

    print()


# ============================================================================
# TODO 3: 解析 ReAct Agent 的输出格式
# ============================================================================
def parse_react_output(raw_output: str) -> Dict[str, Optional[str]]:
    """
    解析 ReAct Agent 的输出，提取 Thought、Action 和 Observation 等信息。

    ReAct 输出格式示例:
        Thought: 我需要查询天气信息
        Action: weather_api("北京")
        Observation: 温度15°C, 降雨概率80%

    或者:
        Thought: 我已获得足够信息
        Final Answer: 今天15°C,有80%降雨概率,建议带伞

    参数:
        raw_output: LLM 的原始输出文本

    返回:
        parsed: 字典，包含以下可能的键:
            - "thought": 思考内容 (str 或 None)
            - "action": 动作名称 (str 或 None)
            - "action_input": 动作输入 (str 或 None)
            - "observation": 观察结果 (str 或 None)
            - "final_answer": 最终回答 (str 或 None)
            - "is_final": 是否为最终回答 (bool)

    提示:
        1. 使用正则表达式匹配 "Thought:"、"Action:"、"Final Answer:" 等模式
        2. Action 的格式可能是 "tool_name(input)" 或 "tool_name: input"
        3. 注意大小写不敏感和首尾空白
        4. 如果匹配到 "Final Answer"，将 is_final 设为 True
    """
    # TODO: 实现 ReAct 输出解析

    parsed = {
        "thought": None,
        "action": None,
        "action_input": None,
        "observation": None,
        "final_answer": None,
        "is_final": False,
    }

    # 步骤 1: 提取 Thought
    # 提示: 使用 re.search(r'Thought:\s*(.+?)(?=\n(?:Action|Observation|Final)|$)', raw_output, re.IGNORECASE | re.DOTALL)
    thought_match = None  # ← TODO
    if thought_match:
        parsed["thought"] = thought_match.group(1).strip()

    # 步骤 2: 提取 Action
    # 提示: 匹配 "Action: tool_name(input)" 或 "Action: tool_name"
    action_match = None  # ← TODO: re.search(r'Action:\s*(.+?)(?:\n|$)', raw_output, re.IGNORECASE)
    if action_match:
        action_text = action_match.group(1).strip()
        # TODO: 尝试解析 tool_name(input) 格式
        # 提示: re.match(r'(\w+)\((.+)\)', action_text) 或 re.match(r'(\w+):?\s*(.+)', action_text)
        pass  # ← TODO: 设置 parsed["action"] 和 parsed["action_input"]

    # 步骤 3: 提取 Observation
    obs_match = None  # ← TODO: 匹配 Observation
    if obs_match:
        parsed["observation"] = obs_match.group(1).strip()

    # 步骤 4: 提取 Final Answer
    final_match = None  # ← TODO: 匹配 Final Answer
    if final_match:
        parsed["final_answer"] = final_match.group(1).strip()
        parsed["is_final"] = True

    # 步骤 5: 检查是否应该是最终回答（某些隐含信号）
    # 如果没有任何 Action 但有实质内容，可能是最终回答
    if not parsed["action"] and not parsed["final_answer"] and parsed["thought"]:
        # 检查 thought 中是否有结束信号
        end_markers = ["总结", "综上", "答案是", "结论", "最终"]
        if any(m in parsed["thought"] for m in end_markers):
            parsed["final_answer"] = parsed["thought"]
            parsed["is_final"] = True

    return parsed


# ---- 测试 TODO 3 ----
def test_react_parsing():
    """测试 ReAct 输出解析功能。"""
    print("=" * 60)
    print("TODO 3 测试: ReAct Agent 输出解析")
    print("=" * 60)

    # 测试用例 1: 标准格式（有 Action）
    test_output_1 = """
Thought: 我需要查询北京今天的天气。用户还想知道是否需要带伞。
Action: weather_api("北京")
Observation: 温度15°C, 降雨概率80%, 湿度65%
"""

    result1 = parse_react_output(test_output_1)
    if result1 is None or all(v is None for v in result1.values() if v != "is_final"):
        print("  TODO 未完成，请补全 parse_react_output 函数")
    else:
        print("\n  测试 1: 标准 Action 格式")
        print(f"    Thought: {result1.get('thought', 'N/A')[:60]}...")
        print(f"    Action: {result1.get('action')} ({result1.get('action_input')})")
        print(f"    Observation: {result1.get('observation', 'N/A')[:60]}...")
        print(f"    Is Final: {result1.get('is_final')}")

        # 验证
        checks = []
        if result1.get("thought") and "天气" in result1["thought"]:
            checks.append("✓ Thought 正确")
        else:
            checks.append("✗ Thought 错误")
        if result1.get("action") == "weather_api":
            checks.append("✓ Action 正确")
        else:
            checks.append(f"✗ Action 错误 (got: {result1.get('action')})")
        if result1.get("action_input") == "北京":
            checks.append("✓ Action Input 正确")
        else:
            checks.append(f"✗ Action Input 错误 (got: {result1.get('action_input')})")
        for check in checks:
            print(f"    {check}")

    # 测试用例 2: 最终回答格式
    test_output_2 = """
Thought: 我已经获得了天气信息。15°C偏凉，降雨概率80%很高。建议带伞并穿外套。
Final Answer: 北京今天气温15°C，降雨概率80%。天气偏凉且大概率下雨，强烈建议您带伞，最好也穿一件薄外套。
"""

    result2 = parse_react_output(test_output_2)
    if result2 is not None and not all(v is None for v in result2.values() if v != "is_final"):
        print("\n  测试 2: Final Answer 格式")
        print(f"    Thought: {result2.get('thought', 'N/A')[:60]}...")
        print(f"    Final Answer: {result2.get('final_answer', 'N/A')[:80]}...")
        print(f"    Is Final: {result2.get('is_final')}")

        if result2.get("is_final") and result2.get("final_answer"):
            print(f"    ✓ 正确识别为最终回答")
        else:
            print(f"    ✗ 未能正确识别最终回答")

    # 测试用例 3: 多步推理（Action + 继续思考）
    test_output_3 = """
Thought: 这是一个复杂的多步问题。首先搜索相关资料。
Action: search("人工智能的定义")
Observation: 人工智能是计算机科学的分支，研究如何创建智能机器。

Thought: 基于搜索结果，我可以给出完整的回答了。
Final Answer: 人工智能（AI）是计算机科学的一个重要分支，旨在创建能够模拟人类智能的系统。它包括机器学习、深度学习、自然语言处理等子领域。
"""

    result3 = parse_react_output(test_output_3)
    if result3 is not None and not all(v is None for v in result3.values() if v != "is_final"):
        print("\n  测试 3: 多步推理格式")
        print(f"    Step 1 - Thought: {result3.get('thought', 'N/A')[:60]}...")
        print(f"    Step 1 - Action: {result3.get('action')}")
        print(f"    Is Final: {result3.get('is_final')}")
        # 注意：parse_react_output 只返回最后一次匹配的 thought 和 action
        # 多步解析需要更复杂的逻辑（如按步骤切分）

    print()


# ============================================================================
# 主程序
# ============================================================================
if __name__ == "__main__":
    print("\n╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "s23 RAG 与 AI Agent — 动手练习" + " " * 18 + "║")
    print("║" + " " * 6 + "请依次完成 TODO 1, 2, 3" + " " * 26 + "║")
    print("╚" + "═" * 58 + "╝\n")

    test_text_chunking()
    test_similarity_search()
    test_react_parsing()

    print("=" * 60)
    print("所有测试完成！请检查输出结果。")
    print("如有未通过的测试，请回到对应的 TODO 部分补全代码。")
    print()
    print("提示：")
    print("  TODO 1: 文本切分 — 理解 RAG 索引的第一步")
    print("  TODO 2: 相似度搜索 — 理解 RAG 检索的核心")
    print("  TODO 3: ReAct 输出解析 — 理解 Agent 的工作方式")
    print()
    print("扩展思考：")
    print("  1. 如果 chunk_size 太大或太小，对检索有什么影响？")
    print("  2. 余弦相似度和点积相似度在什么情况下等价？")
    print("  3. ReAct Agent 如何避免陷入无限循环？")
    print("=" * 60)
