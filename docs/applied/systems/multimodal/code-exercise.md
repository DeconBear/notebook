---
title: "s22 多模态模型 — exercise.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# s22 多模态模型 — exercise.py 练习指南

<a href="/notebook/code/applied/systems/multimodal/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过补全三个 TODO 任务，掌握多模态学习的三个核心组件：
1. InfoNCE 对比损失 —— CLIP 的训练目标
2. 余弦相似度与跨模态检索 —— 图文匹配的数学基础
3. 简单图像搜索引擎 —— 将理论转化为应用

## 预备知识

- CLIP 的对称 InfoNCE 损失：$\mathcal{L} = \frac{1}{2}(\mathcal{L}_{\text{image}} + \mathcal{L}_{\text{text}})$
- 图像方向：$\mathcal{L}_{\text{image}} = -\frac{1}{N}\sum_i \log\frac{\exp(S_{ii}/\tau)}{\sum_j \exp(S_{ij}/\tau)}$
- 文本方向：$\mathcal{L}_{\text{text}} = -\frac{1}{N}\sum_i \log\frac{\exp(S_{ii}/\tau)}{\sum_j \exp(S_{ji}/\tau)}$
- 余弦相似度：$\cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \cdot \|\mathbf{b}\|}$

## 任务清单

### TODO 1：实现 InfoNCE 对比损失（`infonce_loss` 函数）

**任务**：输入 L2 归一化的图像和文本嵌入，输出对称对比损失。

**实现步骤**：
1. `S = image_embeddings @ text_embeddings.T` —— 计算相似度矩阵 (N, N)
2. `logits = S / temperature` —— 温度缩放
3. 图像方向：$\text{num} = \exp(\text{diag}(\text{logits}))$，$\text{denom} = \sum_j \exp(\text{logits}[i, j])$
   ```python
   numerator_image = np.exp(np.diag(logits))           # (N,)
   denominator_image = np.sum(np.exp(logits), axis=1)  # (N,)
   L_image = -np.mean(np.log(numerator_image / denominator_image))
   ```
4. 文本方向（对称）：对 `logits.T` 做同样操作
   ```python
   logits_text = S.T / temperature
   numerator_text = np.exp(np.diag(logits_text))
   denominator_text = np.sum(np.exp(logits_text), axis=1)
   L_text = -np.mean(np.log(numerator_text / denominator_text))
   ```
5. `loss = (L_image + L_text) / 2`

**关键理解**：
- **对角线** $S_{ii}$ 是匹配的图文对（正样本），每行的其他元素是负样本
- **对称设计**确保两个编码器都学习对齐——图→文和文→图两个方向
- **$\tau$ 的作用**：$\tau=0.07$ 很小，$\exp(S_{ii}/0.07)$ 远大于 $\exp(S_{ij}/0.07)$，增强了正样本的优势

**预期输出**：
```
随机情况下的理论值: -log(1/3) = 1.0986
完美对齐时的理论最小值: 接近 0（取决于 τ）
损失范围: [约0.001, 约1.099]
损失越小 → 图文对齐越好
```

### TODO 2：实现余弦相似度与跨模态匹配

**任务 2a**：实现 `cosine_similarity(vec_a, vec_b)` —— 返回 (M, N) 相似度矩阵。

**数学**：$\text{sim}[i, j] = \frac{\mathbf{a}_i \cdot \mathbf{b}_j}{\|\mathbf{a}_i\| \cdot \|\mathbf{b}_j\|}$

**实现步骤**：
1. `dot_product = vec_a @ vec_b.T` —— 点积矩阵 (M, N)
2. `norm_a = np.linalg.norm(vec_a, axis=1)` —— 每行的 L2 范数 (M,)
3. `norm_b = np.linalg.norm(vec_b, axis=1)` —— 每行的 L2 范数 (N,)
4. `similarity = dot_product / (norm_a[:, None] * norm_b[None, :])` —— 广播除

**注意**：`norm_a[:, None]` 将 (M,) 变为 (M, 1)，`norm_b[None, :]` 将 (N,) 变为 (1, N)，相乘得到 (M, N)。

**任务 2b**：实现 `find_best_match(query_embedding, candidate_embeddings, candidate_labels, top_k)`

**实现步骤**：
1. `query_reshaped = query_embedding.reshape(1, -1)` —— (d,) → (1, d)
2. `similarities = cosine_similarity(query_reshaped, candidate_embeddings)` —— (1, N)
3. `similarities_flat = similarities.flatten()` —— (N,)
4. `top_indices = np.argsort(-similarities_flat)[:top_k]` —— 降序取前 k
5. 构造 `[(candidate_labels[i], similarities_flat[i]) for i in top_indices]`

**预期输出**：
```
余弦相似度矩阵: 狗图像→狗文1 (最高) > 狗图像→猫文 (低)
狗图像正确匹配了狗文本 ✓

Top-3 匹配文本:
  1. 「一只金毛犬」相似度最高
  2. 「一只可爱的狗」次之
  3. 「一只橘猫」相似度明显更低
```

### TODO 3：构建简单的 CLIP 图像搜索引擎（`SimpleImageSearchEngine` 类）

**任务 3a**：实现 `add_image(embedding, metadata)`

```python
def add_image(self, embedding, metadata):
    normalized_emb = self._normalize(embedding)    # L2 归一化
    self.image_embeddings.append(normalized_emb)    # 存入向量列表
    self.image_metadata.append(metadata)            # 存入元数据列表
```

**任务 3b**：实现 `search_by_text(query_embedding, top_k, min_similarity)`

**实现步骤**：
1. `query_normalized = self._normalize(query_embedding)` —— (d,)
2. `emb_matrix = np.stack(self.image_embeddings, axis=0)` —— (N, d)
3. `similarities = emb_matrix @ query_normalized` —— 内积=余弦相似度 (N,)
4. `top_indices = np.argsort(-similarities)[:top_k]`
5. 构造结果，过滤掉 `< min_similarity` 的：
   ```python
   results = []
   for idx in top_indices:
       if similarities[idx] >= min_similarity:
           results.append({"metadata": self.image_metadata[idx],
                          "similarity": float(similarities[idx])})
   ```

**关键设计**：
- 向量已 L2 归一化，因此 `emb_matrix @ query_normalized` 直接得到余弦相似度
- `np.argsort(-similarities)` 实现降序排序
- `min_similarity` 阈值过滤不相关结果

**预期输出**：
```
索引规模: 10 张图片
查询: 狗的文本描述
Top-5 搜索结果:
  金毛犬.jpg (dog) - 相似度最高
  哈士奇.jpg (dog) - 接近
  柯基.jpg (dog) - 接近
  ... (其他类别图片相似度更低)

以图搜图：
  查询图片本身获得最高相似度（≈ 1.0）✓
```

## 完成后的验证

全部三个 TODO 通过测试后，如果安装了 CLIP 模型和 sklearn，运行 `python code/demo.py` 观察：
1. 零样本分类如何不需要任何训练就识别图像
2. 图文相似度排序是否正确匹配语义
3. PCA 可视化中图文嵌入的空间分布


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/applied/systems/multimodal/code/exercise.py`
