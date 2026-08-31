# -*- coding: utf-8 -*-
"""
s22 多模态模型 — 练习代码
=========================
请完成以下 TODO 任务，巩固对多模态学习和 CLIP 的理解。

每个 TODO 都有详细的指示和预期输出描述。
建议先阅读 README.md，再尝试独立补全代码。
"""

import numpy as np
from typing import List, Tuple


# ============================================================================
# TODO 1: 实现 InfoNCE 对比损失
# ============================================================================
def infonce_loss(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    temperature: float = 0.07
) -> Tuple[float, np.ndarray]:
    """
    实现 CLIP 使用的 InfoNCE 对比损失。

    数学公式：
        L_image = -1/N * Σ_i log( exp(S_ii/τ) / Σ_j exp(S_ij/τ) )
        L_text  = -1/N * Σ_i log( exp(S_ii/τ) / Σ_j exp(S_ji/τ) )
        L_CLIP  = (L_image + L_text) / 2

    其中 S_ij = I_i · T_j 是图像嵌入和文本嵌入的余弦相似度。

    参数:
        image_embeddings: 图像嵌入向量矩阵，shape (N, d)，已 L2 归一化
        text_embeddings:  文本嵌入向量矩阵，shape (N, d)，已 L2 归一化
        temperature: 温度参数 τ，控制 softmax 分布锐度

    返回:
        loss: InfoNCE 损失的标量值
        similarity_matrix: 余弦相似度矩阵 S, shape (N, N)

    提示：
        1. 先计算相似度矩阵 S = image_embeddings @ text_embeddings.T
        2. 由于向量已归一化，内积即为余弦相似度
        3. 将 S 除以 τ 得到 logits
        4. 对每行做 softmax：使用 exp + sum 的方式
        5. 对角线元素（S[i,i]）是匹配对，应作为正样本
        6. 分别从图像和文本两个方向计算损失，最后取平均
        7. 使用 np.exp 和 np.sum 时注意 axis 参数
    """
    N = image_embeddings.shape[0]  # batch size

    # TODO: 步骤 1 — 计算相似度矩阵 S (shape: N×N)
    # 提示: 使用矩阵乘法 @ ，因为向量已归一化
    S = None  # ← TODO: S = image_embeddings @ text_embeddings.T

    # TODO: 步骤 2 — 计算 logits = S / temperature
    # 提示: 除以 τ 使分布更尖锐，匹配对的优势更明显
    logits = None  # ← TODO

    # TODO: 步骤 3 — 图像方向的损失 L_image
    # 对于每一行 i:
    #   - 分子 = exp(logits[i, i])
    #   - 分母 = Σ_j exp(logits[i, j])
    #   - loss_i = -log(分子 / 分母)
    # 最终 L_image = mean(loss_i)
    L_image = None  # ← TODO

    # TODO: 步骤 4 — 文本方向的损失 L_text (对称)
    # 对于每一列 j:
    #   - 分子 = exp(logits[j, j])  (注意: S[j,j] 在矩阵转置后)
    #   - 分母 = Σ_i exp(logits[i, j])
    #   - loss_j = -log(分子 / 分母)
    # 最终 L_text = mean(loss_j)
    L_text = None  # ← TODO

    # TODO: 步骤 5 — 总损失 = 两者的平均
    loss = None  # ← TODO

    return loss, S


# ---- 测试 TODO 1 ----
def test_infonce_loss():
    """测试 InfoNCE 损失的实现。"""
    print("=" * 60)
    print("TODO 1 测试: InfoNCE 对比损失")
    print("=" * 60)

    # 测试数据：3 对嵌入向量 (N=3, d=4)
    np.random.seed(42)
    N, d = 3, 4

    # 创建并归一化图像嵌入
    img_embs = np.random.randn(N, d).astype(np.float64)
    img_embs = img_embs / np.linalg.norm(img_embs, axis=1, keepdims=True)

    # 创建并归一化文本嵌入
    txt_embs = np.random.randn(N, d).astype(np.float64)
    txt_embs = txt_embs / np.linalg.norm(txt_embs, axis=1, keepdims=True)

    loss, S = infonce_loss(img_embs, txt_embs)

    if loss is None:
        print("  TODO 未完成，请补全 infonce_loss 函数")
    else:
        print(f"\n  相似度矩阵 S (3×3):")
        for i in range(N):
            row_str = "    ".join([f"{S[i, j]:+.4f}" for j in range(N)])
            print(f"    {row_str}")

        print(f"\n  InfoNCE 损失: {loss:.6f}")

        # 验证损失是否在合理范围内
        # 随机情况下，损失应接近 -log(1/N)
        expected_random = -np.log(1.0 / N)
        print(f"  随机情况下的理论值: -log(1/{N}) = {expected_random:.4f}")
        print(f"  (实际值可能因随机初始化而不同)")

        # 附加验证：完美对齐的情况
        identity_S = np.eye(N)
        logits_p = identity_S / 0.07
        numerator = np.exp(np.diag(logits_p))
        denominator = np.sum(np.exp(logits_p), axis=1)
        perfect_loss = -np.mean(np.log(numerator / denominator))
        print(f"  完美对齐时的理论最小值: {perfect_loss:.6f}")
        print(f"  损失范围: [{perfect_loss:.4f}, {expected_random:.4f}]")
        print(f"  损失越小 = 图文对齐越好")

    print()


# ============================================================================
# TODO 2: 实现余弦相似度计算与图文匹配
# ============================================================================
def cosine_similarity(
    vec_a: np.ndarray,
    vec_b: np.ndarray
) -> np.ndarray:
    """
    计算两组向量之间的余弦相似度。

    公式：
        cos(a, b) = (a · b) / (||a|| · ||b||)

    参数:
        vec_a: 第一组向量，shape (M, d)
        vec_b: 第二组向量，shape (N, d)

    返回:
        similarity: 相似度矩阵，shape (M, N)，其中 similarity[i, j] = cos(vec_a[i], vec_b[j])

    提示：
        1. 先计算 a 和 b 各自的 L2 范数（np.linalg.norm with axis）
        2. 计算点积矩阵 = vec_a @ vec_b.T
        3. 除以范数的外积得到余弦相似度
        4. 注意广播维度
    """
    # TODO: 实现余弦相似度计算

    # 步骤 1: 计算点积矩阵
    # shape: (M, N)
    dot_product = None  # ← TODO

    # 步骤 2: 计算 vec_a 每个向量的 L2 范数
    # shape: (M,)
    norm_a = None  # ← TODO

    # 步骤 3: 计算 vec_b 每个向量的 L2 范数
    # shape: (N,)
    norm_b = None  # ← TODO

    # 步骤 4: 计算余弦相似度
    # dot_product / (norm_a[:, None] * norm_b[None, :])
    similarity = None  # ← TODO

    return similarity


def find_best_match(
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    candidate_labels: List[str],
    top_k: int = 3
) -> List[Tuple[str, float]]:
    """
    在候选嵌入中找到与查询嵌入最匹配的前 top_k 个。

    用于场景：
    - 给定图片嵌入，找到最匹配的文本描述
    - 给定文本嵌入，找到最匹配的图片

    参数:
        query_embedding: 查询向量，shape (d,)
        candidate_embeddings: 候选向量矩阵，shape (N, d)
        candidate_labels: 候选向量的标签列表
        top_k: 返回前 k 个最佳匹配

    返回:
        [(标签, 相似度), ...] 按相似度降序排列

    提示：
        1. 将 query_embedding 扩展为 (1, d) 以匹配 cosine_similarity 的输入要求
        2. 调用上面的 cosine_similarity 函数
        3. 用 np.argsort 排序（-相似度 实现降序）
        4. 取前 top_k 个
    """
    # TODO: 实现跨模态搜索

    # 步骤 1: 将查询向量 reshape 为 (1, d)
    query_reshaped = None  # ← TODO: query_embedding.reshape(1, -1)

    # 步骤 2: 计算查询与所有候选的余弦相似度
    # 调用上面实现的 cosine_similarity 函数
    similarities = None  # ← TODO, shape: (1, N)

    # 步骤 3: 展平为 1D 数组 (N,)
    similarities_flat = None  # ← TODO

    # 步骤 4: 按相似度降序排序，取前 top_k 个
    # 使用 np.argsort(-similarities_flat)
    top_indices = None  # ← TODO

    # 步骤 5: 返回 (标签, 相似度) 列表
    results = None  # ← TODO

    return results


# ---- 测试 TODO 2 ----
def test_similarity_search():
    """测试余弦相似度计算和跨模态搜索。"""
    print("=" * 60)
    print("TODO 2 测试: 余弦相似度与图文匹配")
    print("=" * 60)

    np.random.seed(123)
    d = 16  # 嵌入维度

    # 模拟不同类别的嵌入
    # 狗类：图像 + 文本嵌入
    dog_img = np.random.randn(d) * 0.3 + np.array([1.0] * 4 + [0.0] * 12)
    dog_text1 = dog_img * 0.9 + np.random.randn(d) * 0.1
    dog_text2 = dog_img * 0.85 + np.random.randn(d) * 0.15

    # 猫类
    cat_img = np.random.randn(d) * 0.3 + np.array([0.5] * 4 + [0.8] * 4 + [0.0] * 8)
    cat_text = cat_img * 0.9 + np.random.randn(d) * 0.1

    # 汽车类
    car_img = np.random.randn(d) * 0.3 + np.array([0.0] * 8 + [1.0] * 4 + [0.0] * 4)
    car_text = car_img * 0.9 + np.random.randn(d) * 0.1

    # ---- 测试余弦相似度 ----
    img_embs = np.stack([dog_img, cat_img, car_img])  # shape: (3, 16)
    txt_embs = np.stack([dog_text1, dog_text2, cat_text, car_text])  # shape: (4, 16)

    # 归一化
    img_embs = img_embs / np.linalg.norm(img_embs, axis=1, keepdims=True)
    txt_embs = txt_embs / np.linalg.norm(txt_embs, axis=1, keepdims=True)

    sim_matrix = cosine_similarity(img_embs, txt_embs)

    if sim_matrix is None:
        print("  TODO 未完成，请补全 cosine_similarity 函数")
    else:
        print(f"\n  余弦相似度矩阵 ({img_embs.shape[0]}张图片 × {txt_embs.shape[0]}段文本):")
        print(f"  {'':>12}", end="")
        labels = ["狗文1", "狗文2", "猫文", "车文"]
        for lbl in labels:
            print(f"{lbl:>10}", end="")
        print()
        img_labels = ["狗图像", "猫图像", "车图像"]
        for i, lbl in enumerate(img_labels):
            print(f"  {lbl:<10}", end="")
            for j in range(txt_embs.shape[0]):
                print(f"{sim_matrix[i, j]:10.4f}", end="")
            print()

        # 检查：狗图像应该最匹配狗文本
        if sim_matrix[0, 0] > sim_matrix[0, 2] and sim_matrix[0, 0] > sim_matrix[0, 3]:
            print(f"\n  ✓ 狗图像正确匹配了狗文本（相似度: {sim_matrix[0,0]:.4f}）")
        else:
            print(f"\n  ✗ 匹配结果异常，请检查余弦相似度实现")

    # ---- 测试 find_best_match ----
    text_labels = ["一只金毛犬", "一只可爱的狗", "一只橘猫", "一辆红色的汽车"]
    results = find_best_match(dog_img / np.linalg.norm(dog_img),
                              txt_embs, text_labels, top_k=3)

    if results is None:
        print("\n  TODO 未完成，请补全 find_best_match 函数")
    else:
        print(f"\n  查询: 狗的图像嵌入")
        print(f"  Top-{len(results)} 匹配文本:")
        for rank, (label, score) in enumerate(results, 1):
            print(f"    {rank}. 「{label}」 - 相似度: {score:.4f}")

    print()


# ============================================================================
# TODO 3: 构建简单的 CLIP 图像搜索引擎
# ============================================================================
class SimpleImageSearchEngine:
    """
    基于 CLIP 嵌入的简单图像搜索引擎。

    工作原理：
    1. 索引阶段：将图库中的所有图像用 CLIP 编码，存储嵌入向量
    2. 搜索阶段：将查询文本用 CLIP 编码，在嵌入库中找最相似的图像
    """

    def __init__(self, embedding_dim: int = 512):
        """
        初始化搜索引擎。

        参数:
            embedding_dim: 嵌入向量的维度
        """
        self.embedding_dim = embedding_dim
        # 存储所有索引图像的嵌入
        self.image_embeddings: List[np.ndarray] = []
        # 存储每张图像的元数据（路径、标签等）
        self.image_metadata: List[dict] = []
        # 存储文本嵌入缓存（避免重复编码相同查询）
        self.text_cache: dict = {}

    def add_image(self, embedding: np.ndarray, metadata: dict) -> None:
        """
        向索引中添加一张图像。

        参数:
            embedding: 图像的 CLIP 嵌入向量，shape (d,)，应为 L2 归一化向量
            metadata: 图像的元数据字典，如 {"path": "...", "label": "狗", "id": 1}
        """
        # TODO: 将 embedding 和 metadata 添加到对应的列表中
        # 提示：
        #   1. 确保 embedding 是 L2 归一化的（调用 _normalize）
        #   2. 追加到 self.image_embeddings 和 self.image_metadata
        normalized_emb = self._normalize(embedding)  # L2 归一化
        # ← TODO: 将归一化后的向量和元数据加入存储

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        """
        L2 归一化向量。

        参数:
            vec: 输入向量，shape (d,)
        返回:
            归一化后的向量，L2 范数 = 1
        """
        norm = np.linalg.norm(vec)  # 计算 L2 范数
        if norm < 1e-10:
            return vec  # 避免除零
        return vec / norm

    def search_by_text(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        min_similarity: float = 0.0
    ) -> List[dict]:
        """
        根据文本嵌入搜索最匹配的图像。

        参数:
            query_embedding: 查询文本的 CLIP 嵌入向量，shape (d,)
            top_k: 返回前 k 个结果
            min_similarity: 最低相似度阈值，低于此值的结果将被过滤

        返回:
            results: 列表，每项为 {"metadata": ..., "similarity": float}
                    按相似度降序排列

        提示：
            1. 如果索引为空，返回空列表
            2. 将 query_embedding 归一化
            3. 用 cosine_similarity 或内积计算与所有图像嵌入的相似度
            4. 排序，取 top_k，过滤低于 min_similarity 的结果
            5. 返回包含 metadata 和 similarity 的字典列表
        """
        # TODO: 实现文本搜索图像功能

        if len(self.image_embeddings) == 0:
            return []  # 索引为空

        # 步骤 1: 归一化查询向量
        query_normalized = self._normalize(query_embedding)  # shape: (d,)

        # 步骤 2: 将索引嵌入堆叠为矩阵 (N, d)
        emb_matrix = None  # ← TODO: np.stack(self.image_embeddings, axis=0)

        # 步骤 3: 计算查询与所有索引嵌入的余弦相似度
        # 由于向量已归一化，直接用内积即可
        similarities = None  # ← TODO: emb_matrix @ query_normalized

        # 步骤 4: 排序并取 top_k
        # 使用 np.argsort(-similarities)，注意 similarities 是 1D 数组
        top_indices = None  # ← TODO

        # 步骤 5: 构建结果列表
        results = []
        # TODO: 遍历 top_indices，如果 similarity >= min_similarity 则加入 results
        # 每项格式: {"metadata": self.image_metadata[idx], "similarity": float(similarities[idx])}

        return results

    def search_by_image(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[dict]:
        """
        根据图像嵌入搜索最相似的图像（以图搜图）。

        参数和返回值同 search_by_text。
        """
        # 以图搜图与文搜图使用相同的向量检索逻辑
        return self.search_by_text(query_embedding, top_k)

    def get_index_size(self) -> int:
        """返回当前索引中的图像数量。"""
        return len(self.image_embeddings)

    def reset(self) -> None:
        """清空索引。"""
        self.image_embeddings = []
        self.image_metadata = []


# ---- 测试 TODO 3 ----
def test_image_search_engine():
    """测试简化版 CLIP 图像搜索引擎。"""
    print("=" * 60)
    print("TODO 3 测试: 简单图像搜索引擎")
    print("=" * 60)

    np.random.seed(99)
    d = 16

    # 构造模拟图像嵌入
    engine = SimpleImageSearchEngine(embedding_dim=d)

    # 创建 10 张模拟图像的嵌入（4个类别）
    mock_images = [
        # 狗类（索引 0-2）
        (np.random.randn(d) * 0.2 + np.array([1.0] * 4 + [0.0] * 12), {"class": "dog", "id": 1, "name": "金毛犬.jpg"}),
        (np.random.randn(d) * 0.2 + np.array([1.0] * 4 + [0.0] * 12), {"class": "dog", "id": 2, "name": "哈士奇.jpg"}),
        (np.random.randn(d) * 0.2 + np.array([1.0] * 4 + [0.0] * 12), {"class": "dog", "id": 3, "name": "柯基.jpg"}),
        # 猫类（索引 3-5）
        (np.random.randn(d) * 0.2 + np.array([0.0] * 4 + [1.0] * 4 + [0.0] * 8), {"class": "cat", "id": 4, "name": "橘猫.jpg"}),
        (np.random.randn(d) * 0.2 + np.array([0.0] * 4 + [1.0] * 4 + [0.0] * 8), {"class": "cat", "id": 5, "name": "英短.jpg"}),
        (np.random.randn(d) * 0.2 + np.array([0.0] * 4 + [1.0] * 4 + [0.0] * 8), {"class": "cat", "id": 6, "name": "布偶.jpg"}),
        # 汽车类（索引 6-8）
        (np.random.randn(d) * 0.2 + np.array([0.0] * 8 + [1.0] * 4 + [0.0] * 4), {"class": "car", "id": 7, "name": "跑车.jpg"}),
        (np.random.randn(d) * 0.2 + np.array([0.0] * 8 + [1.0] * 4 + [0.0] * 4), {"class": "car", "id": 8, "name": "SUV.jpg"}),
        (np.random.randn(d) * 0.2 + np.array([0.0] * 8 + [1.0] * 4 + [0.0] * 4), {"class": "car", "id": 9, "name": "轿车.jpg"}),
        # 食物类（索引 9）
        (np.random.randn(d) * 0.2 + np.array([0.0] * 12 + [1.0] * 4), {"class": "food", "id": 10, "name": "披萨.jpg"}),
    ]

    # 检查 add_image 是否实现
    engine.add_image(mock_images[0][0], mock_images[0][1])
    if len(engine.image_embeddings) == 0:
        print("  TODO 未完成，请补全 SimpleImageSearchEngine.add_image 方法")
        # 手动添加所有图片以测试其他方法
        for emb, meta in mock_images:
            engine.image_embeddings.append(emb / np.linalg.norm(emb))
            engine.image_metadata.append(meta)
    else:
        # 添加所有图片
        for emb, meta in mock_images:
            engine.add_image(emb, meta)

    # 创建模拟的文本查询嵌入（狗的文本描述）
    query_text_emb = np.random.randn(d) * 0.2 + np.array([1.0] * 4 + [0.0] * 12)
    query_text_emb = engine._normalize(query_text_emb)

    results = engine.search_by_text(query_text_emb, top_k=5)

    if results is None or len(results) == 0:
        print("\n  TODO 未完成，请补全 search_by_text 方法")
    else:
        print(f"\n  索引规模: {engine.get_index_size()} 张图片")
        print(f"\n  查询: 狗的文本描述")
        print(f"  Top-{len(results)} 搜索结果:")
        print(f"  {'排名':<6} {'相似度':<10} {'图片名':<15} {'类别'}")
        print(f"  {'─' * 45}")
        for rank, result in enumerate(results, 1):
            name = result["metadata"].get("name", "unknown")
            cls = result["metadata"].get("class", "unknown")
            sim = result["similarity"]
            print(f"  {rank:<6} {sim:<10.4f} {name:<15} {cls}")

        # 验证：排名靠前的结果应该是狗类
        if results[0]["metadata"]["class"] == "dog":
            print(f"\n  ✓ 搜索引擎正确返回了狗类图片作为最佳匹配")
        else:
            print(f"\n  ✗ 搜索结果异常，最佳匹配应该是狗类图片")

    # 测试以图搜图
    print(f"\n  查询: 以图搜图（用第一张狗图片搜索相似图片）")
    query_img_emb = engine.image_embeddings[0]
    img_results = engine.search_by_image(query_img_emb, top_k=3)
    if img_results:
        for rank, result in enumerate(img_results, 1):
            name = result["metadata"].get("name", "unknown")
            sim = result["similarity"]
            print(f"    {rank}. {name} — 相似度: {sim:.4f}")
        if img_results[0]["metadata"]["id"] == mock_images[0][1]["id"]:
            print(f"    ✓ 以图搜图返回了查询图像本身（最高相似度 ≈ 1.0）")

    print()


# ============================================================================
# 主程序
# ============================================================================
if __name__ == "__main__":
    print("\n╔" + "═" * 58 + "╗")
    print("║" + " " * 12 + "s22 多模态模型 — 动手练习" + " " * 20 + "║")
    print("║" + " " * 6 + "请依次完成 TODO 1, 2, 3" + " " * 26 + "║")
    print("╚" + "═" * 58 + "╝\n")

    test_infonce_loss()
    test_similarity_search()
    test_image_search_engine()

    print("=" * 60)
    print("所有测试完成！请检查输出结果。")
    print("如有未通过的测试，请回到对应的 TODO 部分补全代码。")
    print()
    print("提示：")
    print("  TODO 1: InfoNCE 损失 — 理解 CLIP 的训练目标")
    print("  TODO 2: 余弦相似度 — 理解跨模态匹配的数学基础")
    print("  TODO 3: 图像搜索引擎 — 将理论转化为实际应用")
    print("=" * 60)
