# -*- coding: utf-8 -*-
"""
s14 文本表示 — 练习题
==============================================
请补全以下 TODO 部分，完成后运行验证。
"""

import numpy as np
import math
import torch
import torch.nn.functional as F

# ============================================================
# 练习 1：实现 TF-IDF 的 IDF 计算
# ============================================================

def compute_idf(tokenized_docs: list) -> dict:
    """
    TODO: 实现 IDF 计算
    IDF(w) = log(N / df(w))
    其中 N 是总文档数，df(w) 是包含词 w 的文档数

    参数：
        tokenized_docs: 文档列表，每个文档是词的列表
    返回：
        dict: {词: IDF值}
    """
    N = len(tokenized_docs)
    df = {}
    for doc in tokenized_docs:
        for word in set(doc):  # 每篇文档中每个词只计一次
            df[word] = df.get(word, 0) + 1

    # TODO: 计算 IDF
    # 公式: idf[word] = log(N / df[word])
    # 使用 math.log 计算自然对数，加平滑避免除零
    idf = {}
    # ===== 你的代码在这里 =====
    # 提示：
    #   for word, doc_freq in df.items():
    #       idf[word] = math.log(N / doc_freq)
    #       # 可选：使用平滑 (N+1)/(doc_freq+1) + 1
    # ==========================
    return idf


# 测试数据
test_docs = [
    ["机器", "学习", "是", "人工智能", "的", "分支"],
    ["深度", "学习", "需要", "大量", "数据"],
    ["机器", "学习", "模型", "需要", "训练"],
]
print("[练习1] 你的 IDF 计算结果:")
print(compute_idf(test_docs))
# 预期："的"的 IDF 应该最高（只出现在1篇文档中），"学习"的 IDF 较低（出现在3篇文档中）


# ============================================================
# 练习 2：实现负采样损失函数
# ============================================================

def negative_sampling_loss(
    v_center: torch.Tensor,  # (batch, embed_dim) — 中心词向量
    u_pos: torch.Tensor,     # (batch, embed_dim) — 正样本上下文向量
    u_neg: torch.Tensor,     # (batch, num_neg, embed_dim) — 负样本向量
) -> torch.Tensor:
    """
    TODO: 实现 Skip-gram 负采样损失函数

    损失公式:
    L = -[log σ(v_center · u_pos)] - Σ[log σ(-v_center · u_neg_i)]
    其中 σ 是 sigmoid 函数，· 是向量点积

    参数：
        v_center: 中心词向量 (batch, d)
        u_pos: 正样本上下文向量 (batch, d)
        u_neg: 负样本向量 (batch, K, d)
    返回：
        loss: 标量损失值
    """
    # TODO: 实现损失计算
    # 步骤：
    #   1. 计算正样本得分 pos_score = sum(v_center * u_pos, dim=1) → (batch,)
    #   2. 用 F.logsigmoid(pos_score) 计算正样本 log 概率
    #   3. 计算负样本得分 neg_score = batch_mm(u_neg, v_center) → (batch, K)
    #   4. 用 F.logsigmoid(-neg_score) 计算负样本 log 概率
    #   5. 总损失 = -(pos_loss.mean() + neg_loss.sum(dim=1).mean())
    # ===== 你的代码在这里 =====
    loss = torch.tensor(0.0)
    # ==========================
    return loss


# 测试负采样损失
batch_size, embed_dim, num_neg = 4, 5, 3
v_test = torch.randn(batch_size, embed_dim)
u_pos_test = torch.randn(batch_size, embed_dim)
u_neg_test = torch.randn(batch_size, num_neg, embed_dim)
print(f"\n[练习2] 你的负采样损失: {negative_sampling_loss(v_test, u_pos_test, u_neg_test).item():.4f}")


# ============================================================
# 练习 3：计算余弦相似度
# ============================================================

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    TODO: 实现两个向量的余弦相似度

    cos(v1, v2) = (v1 · v2) / (||v1|| × ||v2||)

    参数：
        v1, v2: numpy 数组
    返回：
        余弦相似度，范围 [-1, 1]
    """
    # TODO: 实现
    # 步骤：
    #   1. 计算点积 dot = np.dot(v1, v2)
    #   2. 计算范数 norm = ||v1|| × ||v2||
    #   3. 返回 dot / norm（注意处理 norm=0 的情况）
    # ===== 你的代码在这里 =====
    return 0.0
    # ==========================


# 测试余弦相似度
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.0, 2.0, 3.0])  # 完全相同 → sim = 1.0
c = np.array([-1.0, -2.0, -3.0])  # 完全相反 → sim = -1.0
d = np.array([1.0, 0.0, 0.0])  # 正交部分

print(f"\n[练习3] 余弦相似度测试:")
print(f"  cos(a, b) = {cosine_similarity(a, b):.4f}  (期望: 1.0000)")
print(f"  cos(a, c) = {cosine_similarity(a, c):.4f}  (期望: -1.0000)")
print(f"  cos(a, d) = {cosine_similarity(a, d):.4f}  (期望: 0.2673)")

print("\n所有练习测试完成！请对比 demo.py 查看参考实现。")
