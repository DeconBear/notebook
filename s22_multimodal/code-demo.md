---
title: "s22 多模态模型 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# s22 多模态模型 — demo.py 代码详解

<a href="../code/s22_multimodal/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd s22_multimodal/code
python demo.py
```

**注意**：首次运行会自动下载 CLIP ViT-B/32 模型（约 600MB）。如果网络较慢，demo 4（InfoNCE 对比损失演示）不依赖模型下载，会优先运行。

## 代码逐段详解

### 第1步：导入库 — 每个库做什么

```python
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
# CLIPProcessor: 同时处理图像和文本的预处理器（缩放、裁剪、归一化图像；分词文本）
# CLIPModel: OpenAI 的预训练 CLIP 模型（ViT-B/32 版本）
# CLIPTokenizer: CLIP 使用的文本分词器（BPE 编码）

from sklearn.decomposition import PCA
# PCA: 将 512 维嵌入降到 2 维，可视化「语义相似的图文在空间中靠近」
```

**关键依赖**：PyTorch、transformers、Pillow、scikit-learn。在没有 GPU 的 CPU 模式下，代码会设置较短的下载超时并优雅回退。

### 第2步：加载 CLIP 模型 — 双编码器架构

**CLIP（Radford et al., 2021）的架构**包含两个独立的编码器：

- **视觉编码器**（Vision Transformer ViT-B/32）：~86M 参数，将图像编码为 512 维向量 $\mathbf{I} \in \mathbb{R}^{512}$
- **文本编码器**（Transformer Decoder）：~63M 参数，将文本编码为 512 维向量 $\mathbf{T} \in \mathbb{R}^{512}$

两个编码器输出的向量**在同一空间中**，且都已做 L2 归一化（$\|\mathbf{I}\| = \|\mathbf{T}\| = 1$），因此余弦相似度退化为向量内积：

$$
\cos(\mathbf{I}, \mathbf{T}) = \frac{\mathbf{I} \cdot \mathbf{T}}{\|\mathbf{I}\| \cdot \|\mathbf{T}\|} = \mathbf{I} \cdot \mathbf{T}
$$

```python
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
model.eval()  # 评估模式，禁用 dropout
```

**为什么用 ViT-B/32**：这是在推理速度和嵌入质量之间的一个良好平衡。"B" 表示 Base 尺寸，"32" 表示 patch size 为 32——将图像切成 32×32 像素的 patch 作为 Transformer 的输入 token。

**processor 的作用**：`CLIPProcessor` 封装了图像的预处理（resize→224, center crop→224, normalize with CLIP stats）和文本的分词（BPE tokenizer, max_length=77），一步完成两种模态的编码准备。

### 第3步：零样本图像分类 — CLIP 最惊艳的能力

**核心思想**：将分类问题转化为**图文匹配问题**。传统分类器需要为每个类别收集标注数据——CLIP 只需要每个类别的**一段文字描述**。

**数学流程**：

1. 为每个候选类别构造自然语言提示：`"a photo of a {class_name}"`
2. 编码图像：$\mathbf{I} = \text{encode}_{\text{image}}(\text{img})$
3. 编码所有文本提示：$\mathbf{T}_k = \text{encode}_{\text{text}}(\text{prompt}_k)$
4. 计算相似度：$s_k = \cos(\mathbf{I}, \mathbf{T}_k)$
5. 分类结果：$\hat{y} = \arg\max_k s_k$

```python
def zero_shot_classification(model, processor, tokenizer, image_path, class_names, ...):
    img = Image.open(image_path).convert("RGB")

    # 为每个类别构造提示
    text_prompts = [f"a photo of a {name}" for name in class_names]

    with torch.no_grad():
        inputs = processor(text=text_prompts, images=img,
                          return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model(**inputs)
        # outputs.logits_per_image: CLIP 内部计算的温度缩放相似度
        probs = outputs.logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    # 排序取 top-k
    sorted_indices = np.argsort(-probs)
    results = [(class_names[i], float(probs[i])) for i in sorted_indices[:top_k]]
    return results
```

**`outputs.logits_per_image` 是什么**：CLIP 模型的 `forward` 方法内部会自动计算图像嵌入和文本嵌入的余弦相似度，乘以可学习的温度参数 $\tau$（初始化为 $\frac{1}{0.07} \approx 14.3$），然后作为 logits 返回。用 softmax 将其转为概率。

**零样本的含义**：模型从未在"金毛犬"这个特定类别上训练过，但因为它在 4 亿图文对上学习了"狗"的语义，它能将图中"犬类特征"与"a photo of a dog"的文本描述匹配——这就是跨模态泛化。

**提示模板的重要性**：CLIP 对提示格式很敏感。`"a photo of a dog"` 的准确率远高于 `"dog"`，因为训练数据中的文本通常是以自然语言段落的形式出现的。

### 第4步：图文相似度计算 — 跨模态语义搜索

这是零样本分类的一般化——不仅限于类别标签，而是任意文本描述：

```python
def compute_image_text_similarity(model, processor, tokenizer, image_path, captions, device):
    img = Image.open(image_path).convert("RGB")
    with torch.no_grad():
        inputs = processor(text=captions, images=img, return_tensors="pt", ...).to(device)
        outputs = model(**inputs)
        similarity = outputs.logits_per_image.cpu().numpy()[0]
    # 排序返回
    sorted_indices = np.argsort(-similarity)
    return [(captions[i], float(similarity[i])) for i in sorted_indices]
```

**应用场景**：
- **图像到文本**：给定图片，从候选描述中选最佳匹配（如自动图片标注）
- **文本到图像**：给定查询文本，在图库中找最佳匹配（如以文搜图）
- **跨模态检索**：用一段文字找图片，或用一张图片找相关文字

**为什么能工作**：CLIP 在 4 亿图文对上用对比学习训练，使得共享嵌入空间中 $\text{encode}_{\text{image}}(\text{dog}) \approx \text{encode}_{\text{text}}(\text{"a dog"})$。即使具体的描述组合从未在训练中出现，只要分别学过"golden retriever"和"playing in grass"，就能判断其相关性。

### 第5步：嵌入空间探索 — PCA 可视化

**核心思想**：用 PCA 将 512 维 CLIP 嵌入降到 2 维，直观展示「语义相似 = 向量相近」：

```python
from sklearn.decomposition import PCA
embeddings_matrix = np.stack(all_embeddings, axis=0)  # (N, 512)
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings_matrix)  # (N, 2)
```

**可视化设计**：
- **圆形标记（o）**= 图像嵌入 —— 每个类别一张图片
- **方形标记（s）**= 文本嵌入 —— 每个类别多个文本描述
- **虚线连接**= 同一类别的图文对 —— 展示"跨模态对齐"
- **不同颜色**= 不同语义类别（狗=红, 猫=橙, 车=蓝, 食物=绿）

**预期观察**：
- 同一类别的图文嵌入**聚在一起**（如"狗"的图片靠近"a dog"的文本）
- 不同类别的嵌入**彼此分离**（"狗"和"车"的向量相距较远）
- 相关类别在更高层次上靠近（"狗"和"猫"都是动物，比"汽车"更靠近）

**嵌入距离分析**：代码还打印了余弦相似度矩阵的部分预览，用 L2（高相似度 >0.8）、L1（中等 0.4-0.8）、L0（低 <0.4）三级标注。

### 第6步：InfoNCE 对比损失 — CLIP 的数学核心

**这是不需要模型就能运行的纯数学演示**，用 NumPy 从头实现 CLIP 的训练目标。

**双编码器输出**（L2 归一化后）的相似度矩阵：

$$
S_{ij} = \mathbf{I}_i \cdot \mathbf{T}_j \quad \in \mathbb{R}^{N \times N}
$$

**图像方向的 InfoNCE 损失**（给定图像，从 N 个文本中找出正确的一个）：

$$
\mathcal{L}_{\text{image}} = -\frac{1}{N}\sum_{i=1}^{N} \log\frac{\exp(S_{ii}/\tau)}{\sum_{j=1}^{N}\exp(S_{ij}/\tau)}
$$

**文本方向的 InfoNCE 损失**（给定文本，从 N 个图像中找出正确的一个）：

$$
\mathcal{L}_{\text{text}} = -\frac{1}{N}\sum_{i=1}^{N} \log\frac{\exp(S_{ii}/\tau)}{\sum_{j=1}^{N}\exp(S_{ji}/\tau)}
$$

**总损失**（对称对比学习）：

$$
\mathcal{L}_{\text{CLIP}} = \frac{1}{2}(\mathcal{L}_{\text{image}} + \mathcal{L}_{\text{text}})
$$

```python
# 图像方向损失
logits_image = S / tau                           # 温度缩放
numerator = np.exp(np.diag(logits_image))         # 正样本 (匹配对)
denominator = np.sum(np.exp(logits_image), axis=1)  # 所有样本
loss_per_image = -np.log(numerator / denominator)
loss_image = np.mean(loss_per_image)

# 文本方向损失（对称）
logits_text = S.T / tau
numerator_text = np.exp(np.diag(logits_text))
denominator_text = np.sum(np.exp(logits_text), axis=1)
loss_per_text = -np.log(numerator_text / denominator_text)
loss_text = np.mean(loss_per_text)

# 总损失
loss_clip = 0.5 * (loss_image + loss_text)
```

**温度参数 $\tau = 0.07$ 的作用**：$\tau$ 控制 softmax 分布的**锐度**。小 $\tau$ → 分布尖锐 → 提高对正负样本的区分力，但也使训练更不稳定。CLIP 将 $\tau$ 设为可学习参数，初始值 $\exp(\tau) \approx 0.07$。

**为什么是双向的**：对称损失确保信息在两个方向流动——图像编码器学习区分不同的文本描述，文本编码器也学习区分不同的图像。这使嵌入空间对两种模态都是"对齐"的。

**极端场景分析**：
- **完美对齐**（$S = I_N$，即单位矩阵）：$\mathcal{L} \approx -\log(\frac{\exp(1/\tau)}{\exp(1/\tau) + (N-1)\exp(0)})$，随 N 增大而增大
- **完全随机**：$\mathcal{L} \approx -\log(1/N) = \log N$，这是 InfoNCE 损失在随机情况下的理论上限

### 第7步：损失函数行为分析

代码还计算了两个极端场景：
1. **完美匹配**（$S$ = 单位矩阵）：损失接近理论最小值——分类器 100% 确定
2. **完全随机**（$S \approx 0$ 矩阵）：损失 ≈ $-\log(1/N) = \log N$——每个样本是正样本的概率为 $1/N$

**对比学习的关键洞察**：InfoNCE 将"对齐图文"转化为一个 N 选 1 的分类问题。batch size N 越大，负样本越多，分类任务越难，学到的表示越有区分力——这就是为什么 CLIP 的 batch size 高达 32768。

### 关键概念速查表

| 概念 | 一句话解释 | 代码位置 |
|------|-----------|---------|
| 双编码器架构 | 图像编码器(ViT) + 文本编码器(Transformer)，输出同维向量 | `load_clip_model()` |
| L2 归一化 | $\|\mathbf{I}\| = \|\mathbf{T}\| = 1$，使内积=余弦相似度 | 模型内部自动完成 |
| 零样本分类 | 比较图像与各类别文本描述的余弦相似度 | `zero_shot_classification()` |
| InfoNCE 损失 | 对称对比学习：图→文 + 文→图 | `demo_contrastive_loss()` |
| 温度参数 $\tau$ | 控制 softmax 锐度，$\tau=0.07$ 使匹配对优势明显 | `tau = 0.07` |
| 共享嵌入空间 | $\text{encode}_{\text{image}}(\text{dog}) \approx \text{encode}_{\text{text}}(\text{"a dog"})$ | PCA 可视化 |
| 提示模板 | `"a photo of a {class}"` 比 `"{class}"` 效果好得多 | `prompt_template` |

## 完整代码

<<< @/snippets/s22_multimodal/demo.py
