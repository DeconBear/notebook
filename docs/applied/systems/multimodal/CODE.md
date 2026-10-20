# s22 多模态模型 -- 代码说明与运行报告

## 程序做了什么
加载 OpenAI 预训练的 CLIP ViT-B/32 模型（首次运行自动下载约 600MB），演示零样本图像分类（无需目标任务微调，比较给定候选类别；不保证任意类别都正确）、图文相似度计算（图像与多段文本描述的温度缩放余弦相似度分数）、以及 CLIP 嵌入空间的 PCA 降维可视化（展示图像和文本在共享 512 维空间中的分布关系）。

## 运行方法
```bash
cd docs/applied/systems/multimodal/code
python demo.py
```

## 运行结果

### 设计输出（不是本次完整模型运行记录）
- 环境自动检测 CUDA/MPS/CPU 设备并据此选择运行模式
- 零样本分类：对测试图像输出 Top-5 预测类别及概率分数
- 图文相似度：计算图像与多段文本描述之间的温度缩放余弦相似度分数并排序打印
- PCA 可视化：将图像和文本嵌入向量降至 2D 并绘制在同一共享空间中
- 如依赖未安装，给出优雅提示并跳过对应部分（不影响纯数学/概念演示）

### 生成图表

#### 图表 1: CLIP 嵌入空间 PCA 投影
> **结果勘误**：现存 `embedding_space_pca.png` 中四类图像点完全重叠，不能证明图文语义对齐。仓库四张样本图实际上是同一纯灰占位图，与这一异常一致；该旧图暂不展示。请先提供并核验真实样本，再重新运行。PCA 投影也不保证保留所有高维距离关系。

#### 图片资源: 概念图解
- `22-01-clip-architecture.png` -- CLIP 双编码器架构（图像编码器 ViT + 文本编码器 Transformer）与对比学习训练过程
- `22-02-clip-zero-shot.png` -- CLIP 零样本分类流程：将类别名编码为文本嵌入，与图像嵌入计算余弦相似度
- `22-03-llava-architecture.png` -- LLaVA 等多模态大模型架构：视觉编码器 + 投影层 + LLM 的端到端设计
- `22-04-multimodal-embedding-space.png` -- 多模态嵌入空间概念图：图文在统一语义空间中对齐的示意

#### 测试图像
`images/samples/` 下的 golden_retriever.jpg、orange_cat.jpg、pizza.jpg、red_car.jpg 当前均为同一张 224×224 纯灰占位图，并非文件名所描述的真实物体。代码会尝试重新下载并校验后替换这些文件；失败时保留旧文件但不纳入可用样本。不能将占位图的分类或检索结果作为演示结论。替换样本时还应核对物体类别、颜色、来源与复用许可。

## 代码结构
- `check_environment()` -- 检测 PyTorch/transformers 可用性及设备类型（CUDA/MPS/CPU）
- `load_clip_model()` -- 加载 CLIPModel + CLIPProcessor + CLIPTokenizer
- `zero_shot_classification()` -- 零样本分类：类别名 -> 文本嵌入 -> 余弦相似度 -> Top-K 预测
- `compute_image_text_similarity()` -- 计算图像与任意文本描述之间的相似度分数矩阵
- `demo_embedding_space()` -- PCA 降维并可视化图像和文本嵌入在 2D 空间中的分布
- `download_sample_images()` -- 从配置的 Wikimedia 地址获取样本，校验后原子替换无效缓存
- `main()` -- 主流程：环境检测 -> 纯 NumPy 对比损失演示 -> 模型加载 -> 零样本分类 -> 图文相似度 -> PCA 可视化

## 运行环境
- Python 依赖: torch, torchvision, transformers, matplotlib, pillow, scikit-learn
- 硬件需求: CPU 即可运行（GPU 推荐以获得实时推理速度）
- 预计运行时间: ~1-2 分钟（含首次模型下载则 ~5-10 分钟，取决于网络速度）
- 模型大小: CLIP ViT-B/32 约 600MB

## 本次验证范围（2026-10-20）

- 执行 `python -m unittest discover -s tests -p test_multimodal_samples.py`：11 项测试通过。
- 测试全程使用本地内存图片和 mock 网络，覆盖缓存命中、缺失下载、无效缓存修复、错误响应、断网、部分下载、重试、体积与时间预算、替换失败和单个样本失败隔离。
- 未下载 CLIP 权重，未重新运行真实图片分类、检索或 PCA；没有新增模型实测结果。
- 尚待核验真实照片内容、来源与复用许可，再替换四个占位 JPG 并重新生成 PCA 图。旧 PCA 图仍不可作为有效演示结论。
