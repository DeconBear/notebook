# -*- coding: utf-8 -*-
"""
s17 预训练范式 — 练习题
==============================================
请补全以下 TODO 部分，完成后运行验证。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random


# ============================================================
# 练习 1：为 BERT 构建分类头并计算损失
# ============================================================

class BertForClassification(nn.Module):
    """
    TODO: 构建 BERT 分类模型
    在预训练 BERT 之上添加分类头
    """

    def __init__(self, bert_backbone: nn.Module, num_labels: int):
        """
        参数：
            bert_backbone: 预训练的 BERT 模型（不含分类头）
            num_labels: 分类类别数
        """
        super().__init__()
        self.bert = bert_backbone
        hidden_size = self.bert.config.hidden_size  # BERT 的隐藏维度，通常 768
        # TODO: 创建一个线性层作为分类头
        #   输入维度: hidden_size (取 [CLS] token 的向量)
        #   输出维度: num_labels
        # ===== 你的代码在这里 =====
        self.classifier = None  # 替换为 nn.Linear(hidden_size, num_labels)
        # ==========================

    def forward(self, input_ids, attention_mask, labels=None):
        """
        前向传播：提取 [CLS] 向量 → 分类头 → 计算损失

        参数：
            input_ids: token 索引 (batch, seq_len)
            attention_mask: 注意力掩码 (batch, seq_len)
            labels: 真实标签 (batch,), 可选
        返回：
            如果 labels 不为 None: (loss, logits)
            否则: logits
        """
        # 获取 BERT 的输出
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # outputs.last_hidden_state: (batch, seq_len, hidden_size)
        # TODO: 取出 [CLS] token 的向量（第一个 token）
        # ===== 你的代码在这里 =====
        cls_embedding = None  # outputs.last_hidden_state[:, 0, :]
        # ==========================

        # TODO: 通过分类头得到 logits
        # ===== 你的代码在这里 =====
        logits = None  # self.classifier(cls_embedding)
        # ==========================

        loss = None
        if labels is not None:
            # TODO: 用 CrossEntropyLoss 计算损失
            # ===== 你的代码在这里 =====
            pass  # loss = F.cross_entropy(logits, labels)
            # ==========================
            return loss, logits
        return logits


print("[练习1] BERT 分类头结构已定义（需要实际的 bert_backbone 才能测试）")
print("        核心步骤: [CLS]向量 → Linear(hidden, num_labels) → CrossEntropy")


# ============================================================
# 练习 2：实现简单的 MLM：随机遮盖 15% 的 token
# ============================================================

def random_mask_tokens(
    input_ids: torch.Tensor,    # (batch, seq_len) — token id 序列
    vocab_size: int,            # 词汇表大小
    mask_token_id: int,         # [MASK] token 的 id
    special_tokens: set = None, # 特殊 token 集合，不应被遮盖
    mask_prob: float = 0.15,    # 遮盖比例
) -> tuple:
    """
    TODO: 实现 MLM 的随机遮盖
    模仿 BERT 的策略:
      - 15% 的 token 被选中
      - 选中的 token 中: 80% 替换为 [MASK], 10% 替换为随机 token, 10% 保持原样
      - 特殊 token ([CLS], [SEP], [PAD]) 不应被遮盖

    参数：
        input_ids: 原始 token 序列
        vocab_size: 词汇表大小
        mask_token_id: [MASK] token 的 id
        special_tokens: 特殊 token id 集合
        mask_prob: 被选中遮盖的概率
    返回：
        masked_input_ids: 遮盖后的 token 序列
        labels: 真实标签，非遮盖位置为 -100（在 CrossEntropyLoss 中被忽略）
    """
    if special_tokens is None:
        special_tokens = set()
    batch_size, seq_len = input_ids.shape

    # 创建标签：默认全部为 -100（损失计算时忽略）
    labels = torch.full_like(input_ids, -100)

    # TODO: 实现遮盖逻辑
    # 步骤：
    #   1. 创建概率矩阵，形状 (batch, seq_len)，值为随机均匀分布
    #   2. 找出需要遮盖的位置：
    #      - 随机概率 < mask_prob
    #      - 不是特殊 token
    #   3. 对这些位置：
    #      - 生成 [0, 1) 随机数
    #      - < 0.8: 替换为 [MASK]
    #      - < 0.9: 替换为随机 token id
    #      - >= 0.9: 保持原样
    #   4. labels 中，被选中的位置填入原始 token id
    # ===== 你的代码在这里 =====
    masked_input_ids = input_ids.clone()
    # 概率矩阵
    prob_matrix = torch.rand(input_ids.shape)
    # 找出需要遮盖的 mask 位置
    # ...
    # ==========================

    return masked_input_ids, labels


# 测试 MLM 遮盖
test_ids = torch.tensor([[101, 123, 456, 789, 102, 0, 0]])  # [CLS]=101, [SEP]=102, [PAD]=0
mask_token = 103
special = {101, 102, 0}

try:
    masked_ids, labels = random_mask_tokens(test_ids, 10000, mask_token, special, mask_prob=1.0)
    print(f"\n[练习2] MLM 遮盖测试:")
    print(f"  原始:     {test_ids.tolist()[0]}")
    print(f"  遮盖后:   {masked_ids.tolist()[0]}")
    print(f"  标签:     {labels.tolist()[0]}")
    print(f"  注意: 101([CLS]), 102([SEP]), 0([PAD]) 不应被遮盖")
except Exception as e:
    print(f"\n[练习2] 未完成实现: {e}")


# ============================================================
# 练习 3：比较多义词在不同上下文中的 BERT 嵌入
# ============================================================

def compare_polysemous_embeddings(
    sentence_pairs: list,  # [(句1, 句2), ...] 每组包含同一个多义词在不同上下文中的句子
    get_bert_embedding,    # 函数: sentence → embedding vector
) -> list:
    """
    TODO: 比较多义词在不同上下文中的 BERT 嵌入

    参数：
        sentence_pairs: 句子对列表
        get_bert_embedding: 获取 BERT 嵌入的函数
    返回：
        similarities: 每组句对的余弦相似度列表
    """
    # TODO: 实现
    # 步骤：
    #   1. 对每组句对，调用 get_bert_embedding 获取每个句子的嵌入
    #   2. 计算余弦相似度 F.cosine_similarity(v1, v2, dim=0)
    #   3. 比较同义词在不同上下文中和不同词在相同上下文中的相似度
    # ===== 你的代码在这里 =====
    similarities = []
    # ==========================
    return similarities


# 模拟测试
print(f"\n[练习3] 多义词嵌入对比（需要实际的 BERT 模型才能测试）")
print("         预期: '苹果很好吃'中的'苹果' ≠ '苹果发布了新手机'中的'苹果'")
print("         BERT 的上下文嵌入让同一词在不同语境中有不同的向量表示")

print("\n所有练习测试完成！请对比 demo.py 查看参考实现。")
print("""
提示:
  - BERT 分类: [CLS] token 聚合了全句信息 → Linear → softmax
  - MLM 遮盖: 15% 选中的 token 中 80%→[MASK], 10%→随机, 10%→不变
  - 上下文嵌入: BERT 给同一词在不同上下文中不同的向量
""")
