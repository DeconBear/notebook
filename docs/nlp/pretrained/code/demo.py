# -*- coding: utf-8 -*-
"""
s17 预训练范式 demo：BERT 微调 + GPT-2 生成对比
================================================
本文件使用 HuggingFace transformers 库，展示预训练模型的核心用法：
  任务1：BERT 文本分类微调（中文情感分析）
  任务2：BERT 掩码预测（MLM 能力展示）
  任务3：GPT-2 文本生成（对比 BERT 无法生成）

运行方式：在 s17_pretrained_models 目录下执行 `python code/demo.py`
依赖：torch, transformers, datasets, matplotlib
注意：首次运行会在 models/ 目录下下载模型文件（约400MB）
"""

import numpy as np
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# GPU 自动检测
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")
if DEVICE.type == 'cpu':
    print("（未检测到 GPU，使用 CPU 运行。如有 GPU，请安装 CUDA 版 PyTorch 以获得加速）")

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    pipeline,
)

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMAGES = os.path.join(_HERE, '..', 'images')
os.makedirs(_IMAGES, exist_ok=True)

# 设置模型下载目录
os.environ.setdefault('HF_HOME', os.path.join(os.path.dirname(__file__), '..', 'models'))

# ============================================================
# 第一部分：BERT 文本分类微调
# ============================================================

# 中文情感分析数据集（少量示例数据，用于演示微调流程）
train_data = [
    ("这个产品质量非常好，我很满意", 1),
    ("客服态度特别好，问题很快就解决了", 1),
    ("物流很快，包装很精美，好评", 1),
    ("性价比超级高，推荐大家购买", 1),
    ("已经用了好几个月了，质量很稳定", 1),
    ("颜色很好看，大小也合适，非常满意", 1),
    ("功能强大，操作也很简单方便", 1),
    ("比实体店便宜多了，正品无疑，会回购", 1),
    ("材质很好，手感不错，下次还会来买", 1),
    ("发货很快，商品完好无损，很满意", 1),
    ("款式新颖，做工也很好，值得购买", 1),
    ("味道很好，生产日期很新鲜，好评", 1),
    ("这个产品质量太差了，用了两天就坏了", 0),
    ("客服态度恶劣，完全不解决问题", 0),
    ("发货特别慢，等了一周才到，非常失望", 0),
    ("和描述完全不符，图片严重美化，上当了", 0),
    ("用了不到一个月就出问题，质量太差", 0),
    ("包装简陋，收到的时候已经破损了", 0),
    ("一点都不好用，操作复杂，不值这个价钱", 0),
    ("有异味，不敢用，退货还特别麻烦", 0),
    ("刚买完就降价了，还不给保价差评", 0),
    ("做工粗糙，细节处理不到位，不满意", 0),
    ("安装说明书写得太烂，完全看不懂", 0),
    ("噪音很大，严重影响使用体验，差评", 0),
]

# 测试数据
eval_data = [
    ("这个东西真的很好用，我太喜欢了", 1),
    ("物流太慢了，商品还有破损，不推荐", 0),
    ("整体还不错，价格合理，推荐购买", 1),
    ("完全不如描述的那么好，不建议买", 0),
    ("质量没话说，会推荐给朋友的", 1),
    ("售后态度太差了，再也不会买了", 0),
]


class SentimentDataset(Dataset):
    """情感分析 PyTorch 数据集"""

    def __init__(self, data: List[Tuple[str, int]], tokenizer, max_len: int = 128):
        """
        参数：
            data: (文本, 标签) 列表
            tokenizer: BERT tokenizer
            max_len: 最大序列长度
        """
        self.texts = [item[0] for item in data]
        self.labels = [item[1] for item in data]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        # 使用 tokenizer 编码文本
        encoding = self.tokenizer(
            text,
            truncation=True,           # 超过 max_len 则截断
            padding='max_length',      # 不足 max_len 则填充
            max_length=self.max_len,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),       # (max_len,)
            'attention_mask': encoding['attention_mask'].squeeze(0),  # (max_len,)
            'labels': torch.tensor(label, dtype=torch.long),
        }


print("=" * 60)
print("[BERT 文本分类] 加载 bert-base-chinese 模型...")
print("=" * 60)

# 使用最小的可用模型以加速演示（CPU 友好）
model_name = "prajjwal1/bert-tiny"  # 最小 BERT 变体（约 4MB，2层，128维）
_HAS_REAL_MODEL = False

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_cls = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    bert_cls = bert_cls.to(DEVICE)
    _HAS_REAL_MODEL = True
    print(f"[模型] 成功加载: {model_name}")
except Exception as e:
    print(f"[警告] 加载 {model_name} 失败: {e}")
    # 尝试备用：bert-base-chinese
    try:
        model_name = "bert-base-chinese"
        print("[备用] 尝试加载 bert-base-chinese...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        bert_cls = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        bert_cls = bert_cls.to(DEVICE)
        _HAS_REAL_MODEL = True
        print(f"[模型] 成功加载: {model_name}")
    except Exception as e2:
        print(f"[警告] 所有模型下载均失败 ({e2})")
        print("[回退] 使用本地简化模型进行演示...")
        # 回退：创建一个微型随机 BERT 风格模型
        class TinyFallbackClassifier(nn.Module):
            def __init__(self, vocab_size=1000, num_labels=2):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, 32)
                self.encoder = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model=32, nhead=2, dim_feedforward=64, batch_first=True),
                    num_layers=2
                )
                self.classifier = nn.Linear(32, num_labels)
                self.config = type('obj', (object,), {'hidden_size': 32})()
                self.device = DEVICE

            def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
                x = self.embedding(input_ids)
                x = self.encoder(x)
                x = x.mean(dim=1)
                logits = self.classifier(x)
                loss = F.cross_entropy(logits, labels) if labels is not None else None
                return type('obj', (object,), {'loss': loss, 'logits': logits})()

        try:
            tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        except Exception:
            tokenizer = None
        if tokenizer is None:
            # 无 tokenizer 可用，创建一个最简的字符级编码器
            class SimpleCharTokenizer:
                def __init__(self):
                    self.vocab = {c: i for i, c in enumerate('abcdefghijklmnopqrstuvwxyz0123456789')}
                def __call__(self, text, truncation=True, padding='max_length', max_length=128, return_tensors='pt', **kwargs):
                    ids = [self.vocab.get(c, 0) for c in text.lower()[:max_length]]
                    ids = ids + [0] * (max_length - len(ids))
                    return {'input_ids': torch.tensor([ids]), 'attention_mask': torch.tensor([[1]*len(ids)])}
            tokenizer = SimpleCharTokenizer()
        bert_cls = TinyFallbackClassifier().to(DEVICE)
        print(f"[回退] 使用 TinyFallbackClassifier（仅用于演示流程，不是真实 BERT）")

print(f"[模型] 参数量: {sum(p.numel() for p in bert_cls.parameters()):,}")
print(f"[数据] 训练样本: {len(train_data)}, 验证样本: {len(eval_data)}")

# 准备数据集
train_dataset = SentimentDataset(train_data, tokenizer)
eval_dataset = SentimentDataset(eval_data, tokenizer)

# 训练参数（少量 epoch 演示微调）
training_args = TrainingArguments(
    output_dir="./bert_sentiment_checkpoints",  # 模型保存路径
    num_train_epochs=4,                         # 微调 epoch 数（预训练模型只需少量 epoch）
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_strategy="epoch",                # 每个 epoch 评估一次
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=5,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",                           # 不上报到 wandb 等平台
)

# 定义评估指标
def compute_metrics(eval_pred):
    """计算准确率"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}


if _HAS_REAL_MODEL:
    trainer = Trainer(
        model=bert_cls,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    print("\n[微调] 开始训练...（使用预训练 BERT + 少量情感标注数据）")
    trainer.train()

    # 评估
    eval_results = trainer.evaluate()
    print(f"\n[评估] 验证集准确率: {eval_results.get('eval_accuracy', 'N/A'):.4f}")
else:
    print("\n[微调] 跳过（使用回退模型，无需训练）")
    # 使用回退模型进行简单训练
    if DEVICE.type == 'cpu':
        n_fallback_epochs = 1
        n_fallback_samples = 8  # CPU 模式：仅用 8 条样本快速演示
        train_dataset_small = SentimentDataset(train_data[:n_fallback_samples], tokenizer)
        print("[配置] CPU 模式：使用轻量参数快速演示（回退模型 1 epoch, 8 样本）。GPU 模式下将使用完整训练配置。")
    else:
        n_fallback_epochs = 3
        train_dataset_small = train_dataset
    bert_cls.train()
    import torch.optim as optim
    optimizer_ft = optim.Adam(bert_cls.parameters(), lr=0.01)
    for epoch in range(n_fallback_epochs):
        total_loss = 0.0
        for batch in DataLoader(train_dataset_small, batch_size=4, shuffle=True):
            input_ids = batch['input_ids'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            optimizer_ft.zero_grad()
            out = bert_cls(input_ids, labels=labels)
            out.loss.backward()
            optimizer_ft.step()
            total_loss += out.loss.item()
        print(f"  Epoch {epoch+1}: loss={total_loss/len(train_dataset_small)*4:.4f}")
    print("[回退训练] 完成（此训练无法达到预训练BERT的效果，仅用于演示流程）")

# 预测新样本
print("\n[预测] 对新样本进行情感分类:")
test_texts = [
    "这个东西简直太好用了，完全超出预期",
    "质量好坏不说，光是等了一周就不想买了",
    "性价比还不错，但也算不上特别惊艳",
]
for text in test_texts:
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
    device = bert_cls.device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = bert_cls(**inputs).logits
    probs = F.softmax(logits, dim=-1)
    pred = torch.argmax(logits, dim=-1).item()
    label_str = "正面 👍" if pred == 1 else "负面 👎"
    print(f"  文本: {text}")
    print(f"  预测: {label_str} (正面概率: {probs[0][1].item():.3f}, 负面概率: {probs[0][0].item():.3f})")
    print()

# ============================================================
# 第二部分：BERT MLM 掩码预测
# ============================================================

print("=" * 60)
print("[BERT MLM] 掩码预测能力展示")
print("=" * 60)

# 加载专用的 MLM 模型
print("[模型] 加载 bert-base-chinese MLM...")
try:
    mlm_tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    mlm_model = AutoModelForMaskedLM.from_pretrained("bert-base-chinese")
    mlm_model = mlm_model.to(DEVICE)
    print("[模型] 加载成功")
except Exception as e:
    print(f"[警告] 加载失败: {e}, 跳过 MLM 演示")
    mlm_model = None

if mlm_model is not None:
    # 使用 pipeline 简化调用
    mlm_pipeline = pipeline(
        "fill-mask",
        model=mlm_model,
        tokenizer=mlm_tokenizer,
        device=0 if DEVICE.type == 'cuda' else -1,
    )

    # 测试 MLM：在不同上下文中预测 [MASK] 的词
    mlm_examples = [
        "今天天气真[MASK]，适合出去郊游。",
        "这个手机拍照效果很[MASK]，我非常满意。",
        "深度学习是人工智能的一个重要[MASK]。",
        "他每天坚持[MASK]身体，所以非常健康。",
        "这家餐厅的菜品味道[MASK]，价格也合理。",
    ]

    for text in mlm_examples:
        results = mlm_pipeline(text, top_k=3)
        print(f"\n  原文: {text}")
        print(f"  Top-3 预测:")
        for r in results:
            print(f"    [{r['score']:.3f}] {r['token_str']} → {r['sequence']}")


# ============================================================
# 第三部分：GPT-2 文本生成（对比 BERT）
# ============================================================

print("\n" + "=" * 60)
print("[GPT-2 文本生成] 对比 BERT 的生成能力")
print("=" * 60)

# 加载 GPT-2 中文模型
gpt_model_name = "uer/gpt2-chinese-cluecorpussmall"
print(f"[模型] 加载 {gpt_model_name}...")

try:
    gpt_tokenizer = AutoTokenizer.from_pretrained(gpt_model_name)
    gpt_model = AutoModelForCausalLM.from_pretrained(gpt_model_name)
    gpt_model = gpt_model.to(DEVICE)
    # 设置 pad_token
    if gpt_tokenizer.pad_token is None:
        gpt_tokenizer.pad_token = gpt_tokenizer.eos_token
    print("[模型] GPT-2 加载成功")
    print(f"[模型] GPT-2 参数量: {sum(p.numel() for p in gpt_model.parameters()):,}")

    # 文本生成
    prompts = [
        "人工智能的发展历程可以追溯到",
        "今天天气很好，我决定去",
        "在深度学习中，神经网络的训练过程",
    ]

    for prompt in prompts:
        inputs = gpt_tokenizer(prompt, return_tensors='pt')
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        # 生成文本
        outputs = gpt_model.generate(
            **inputs,
            max_new_tokens=30,         # 最多生成 30 个新 token
            temperature=0.8,           # 温度参数控制随机性
            do_sample=True,             # 使用采样而非贪心解码
            top_p=0.9,                  # nucleus sampling
            repetition_penalty=1.1,     # 抑制重复
            pad_token_id=gpt_tokenizer.pad_token_id,
        )
        generated = gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n  提示: {prompt}")
        print(f"  GPT-2: {generated}")

    # 对比：尝试让 BERT 生成文本
    print("\n[对比] BERT 能否生成文本？")
    print("  BERT 是 Encoder-only 架构，使用双向注意力，无法自回归生成。")
    print("  尝试将 BERT 用于生成 → 缺乏因果注意力+自回归训练 → 输出无意义的乱码。")
    print("  这就是为什么：理解任务用 BERT，生成任务用 GPT。")

except Exception as e:
    print(f"[警告] GPT-2 加载失败: {e}")
    print("  (这是正常的，可以跳过此部分）")
    print("  核心理解：BERT=biderectional→理解，GPT=causal→生成")

# ============================================================
# 第四部分：嵌入可视化（BERT 上下文嵌入对比 word2vec）
# ============================================================

print("\n" + "=" * 60)
print("[上下文嵌入] BERT vs word2vec — 多义词对比")
print("=" * 60)

if mlm_model is not None:
    # 展示 BERT 的上下文相关嵌入
    # 同一词在不同上下文中会有不同的向量表示
    test_sentences = [
        "我喜欢吃苹果，特别是红富士苹果",
        "苹果公司发布了最新的iPhone手机",
        "我在超市买了三个苹果",
        "苹果的股价今天上涨了百分之五",
    ]

    for sentence in test_sentences:
        inputs = mlm_tokenizer(sentence, return_tensors='pt')
        with torch.no_grad():
            # 获取 BERT 的隐藏状态（取最后一层的 [CLS] 或特定 token）
            outputs = mlm_model.base_model(**inputs)
            # 取最后一个隐藏层的所有 token 向量
            last_hidden = outputs.last_hidden_state  # (1, seq_len, 768)
            # 找到"苹果"token 的位置
            tokens = mlm_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            # 打印句子和"苹果"的 token 位置
            apple_positions = [i for i, t in enumerate(tokens) if '苹' in t]
            print(f"\n  句子: {sentence}")
            print(f"  Token 序列: {tokens}")
            if len(apple_positions) >= 2:
                # 计算两个"苹果"嵌入的余弦相似度
                v1 = last_hidden[0, apple_positions[0]]
                v2 = last_hidden[0, apple_positions[1]]
                sim = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
                print(f'  句子内"苹果"相似度: {sim:.4f}')
    print("\n  关键观察：同一句中两个'苹果'的上下文嵌入高度相似")
    print("  （因为它们都在'水果'上下文中）")

# ============================================================
# 第五部分：总结
# ============================================================

print("\n" + "=" * 60)
print("[总结] BERT 与 GPT 的核心差异")
print("=" * 60)
print("""
  架构差异:
    BERT: Encoder-only, 双向注意力, 适合"理解"任务
    GPT:  Decoder-only, 因果注意力, 适合"生成"任务

  训练目标:
    BERT: MLM (Masked Language Model) — 预测被遮盖的词
    GPT:  CLM (Causal Language Model) — 预测下一个词

  预训练-微调范式:
    一次大规模预训练 → 添加简单任务头 → 快速微调 → 部署

  BERT 的弱点:
    无法生成连贯文本 — 因为它不是自回归的
    固定长度输入 — 最大512 tokens (原始版本)
    [MASK] 在微调时不存在 — 预训练与微调之间有gap

  GPT 的优势:
    天然适合生成文本 — 从左到右逐个输出
    随着规模增大涌现新能力 — In-context learning, CoT
    统一的输入输出格式 — 所有任务都是"续写文本"

  下一章 s18: 大语言模型的 Scaling Law, 涌现能力, 以及对齐技术
""")
print("所有 demo 运行完成！")
