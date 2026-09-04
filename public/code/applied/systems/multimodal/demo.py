# -*- coding: utf-8 -*-
"""
s22 多模态模型 — 演示代码
=========================
功能：加载预训练的 CLIP 模型，展示零样本图像分类、图文相似度计算、
      以及嵌入空间探索（PCA/t-SNE 可视化）。

每个函数都有中文 docstring，每行逻辑代码都有中文注释。
运行方式：在 s22_multimodal/ 目录下执行 python code/demo.py

依赖：pip install torch torchvision transformers matplotlib pillow scikit-learn
注意：首次运行会自动下载 CLIP 模型（约 600MB），请确保网络连接。
"""

import os
import sys
import warnings
import numpy as np
from typing import List, Tuple, Dict, Optional

# 抑制非关键警告
warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_IMAGES = os.path.join(_HERE, '..', 'images')
os.makedirs(_IMAGES, exist_ok=True)

# ============================================================================
# 第 1 部分：环境检测与模型加载
# ============================================================================

# 全局变量：标记是否成功加载了 CLIP
CLIP_AVAILABLE = False
_device = "cpu"  # 默认使用 CPU

def check_environment() -> Tuple[bool, str]:
    """
    检测运行环境中是否安装了必要的依赖包。

    返回:
        (是否就绪, 设备类型字符串)
    """
    try:
        import torch
        # GPU 自动检测
        device_obj = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        device = device_obj.type
        print(f"[环境] 使用设备: {device_obj}")
        if device_obj.type == 'cuda':
            print(f"[环境] 检测到 CUDA GPU: {torch.cuda.get_device_name(0)}")
        elif device_obj.type == 'mps':
            print("[环境] 检测到 Apple Silicon MPS 加速")
        else:
            print("[环境] 未检测到 GPU，使用 CPU（推理速度较慢但可用）")
    except ImportError:
        print("[警告] 未安装 PyTorch，请执行: pip install torch torchvision")
        return False, "cpu"

    try:
        import transformers
        print(f"[环境] transformers 版本: {transformers.__version__}")
    except ImportError:
        print("[警告] 未安装 transformers，请执行: pip install transformers")
        return False, device

    return True, device


def load_clip_model(device: str = "cpu"):
    """
    加载 CLIP ViT-B/32 模型和对应的预处理器。

    CLIP ViT-B/32 是一个轻量级变体：
    - 视觉编码器：Vision Transformer (ViT-B/32)，约 86M 参数
    - 文本编码器：Transformer，约 63M 参数
    - 嵌入维度：512

    参数:
        device: 运行设备 ("cpu", "cuda", "mps")

    返回:
        (model, processor, tokenizer) 三元组
        如果加载失败，返回 (None, None, None)
    """
    global CLIP_AVAILABLE
    print(f"\n[模型加载] 正在加载 CLIP ViT-B/32 模型（约 600MB，首次运行需下载）...")

    # CPU 模式下设置较短的下载超时，避免长时间等待
    if device == "cpu":
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
        print("[配置] CPU 模式：下载超时设为 30 秒。如网络较慢，模型下载可能跳过，但不影响纯数学演示。")

    try:
        from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer

        # 加载 CLIP 模型 — 使用 ViT-B/32 作为视觉骨干
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        model.eval()  # 切换到评估模式，禁用 dropout 等训练行为

        # 加载预处理器 — 负责图像缩放、裁剪、归一化
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # 加载分词器 — 将文本转换为 token ID
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        print(f"  ✓ CLIP ViT-B/32 模型加载成功")
        print(f"  ✓ 模型参数总量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        print(f"  ✓ 嵌入维度: {model.config.projection_dim}")
        print(f"  ✓ 运行设备: {device}")

        CLIP_AVAILABLE = True
        return model, processor, tokenizer

    except Exception as e:
        print(f"\n  [跳过] CLIP 模型加载失败: {type(e).__name__}")
        if device == "cpu":
            print(f"  (CPU 模式下网络下载可能较慢，已跳过。可提前下载模型到本地以启用视觉演示)")
        else:
            print(f"  可能的原因：网络连接、磁盘空间、或 transformers 版本问题")
        return None, None, None


# ============================================================================
# 第 2 部分：零样本图像分类
# ============================================================================

def download_sample_images() -> List[str]:
    """
    下载用于演示的样本图片。如果本地已有则跳过下载。

    返回:
        图片文件路径的列表
    """
    import urllib.request
    from io import BytesIO
    from PIL import Image

    # 创建图片存储目录
    os.makedirs(os.path.join(_IMAGES, "samples"), exist_ok=True)
    image_paths = []

    # 演示图片：从网络下载或生成简单的纯色分类图片
    sample_sources = [
        ("golden_retriever", "https://upload.wikimedia.org/wikipedia/commons/thumb/9/93/Golden_Retriever_Carlos_%2810581910556%29.jpg/320px-Golden_Retriever_Carlos_%2810581910556%29.jpg"),
        ("orange_cat", "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/320px-Cat_November_2010-1a.jpg"),
        ("red_car", "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Panamera_4_E-Hybrid_%28MSP17%29.jpg/320px-Panamera_4_E-Hybrid_%28MSP17%29.jpg"),
        ("pizza", "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a3/Eq_it-na_pizza-margherita_sep2005_sml.jpg/320px-Eq_it-na_pizza-margherita_sep2005_sml.jpg"),
    ]

    for name, url in sample_sources:
        save_path = os.path.join(_IMAGES, "samples", f"{name}.jpg")
        image_paths.append(save_path)

        # 如果文件已存在，跳过下载
        if os.path.exists(save_path):
            print(f"  图片已存在: {save_path}")
            continue

        try:
            print(f"  下载图片: {name}...")
            urllib.request.urlretrieve(url, save_path)
        except Exception as e:
            print(f"  下载失败 ({name}): {e}")
            # 如果下载失败，创建一个纯色占位图片
            img = Image.new('RGB', (224, 224), color=(100, 100, 100))
            img.save(save_path)
            print(f"  已创建占位图片: {save_path}")

    return image_paths


def zero_shot_classification(
    model,
    processor,
    tokenizer,
    image_path: str,
    class_names: List[str],
    prompt_template: str = "a photo of a {}",
    device: str = "cpu",
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """
    使用 CLIP 进行零样本图像分类。

    工作原理：
    1. 将每个类别名称填入提示模板（如 "a photo of a dog"）
    2. 用文本编码器获取每个提示的嵌入
    3. 用图像编码器获取图像的嵌入
    4. 计算图像嵌入与每个文本嵌入的余弦相似度
    5. 相似度最高的类别即为分类结果

    参数:
        model: CLIP 模型
        processor: 图像预处理器
        tokenizer: 文本分词器
        image_path: 待分类图像的路径
        class_names: 候选类别名称列表
        prompt_template: 文本提示模板，用 {} 占位类别名称
        device: 运行设备
        top_k: 返回前 k 个最可能的类别

    返回:
        [(类别名称, 置信度分数), ...] 按分数降序排列
    """
    from PIL import Image
    import torch

    # ---- 步骤 1: 加载并预处理图像 ----
    img = Image.open(image_path).convert("RGB")  # 确保 RGB 格式

    # ---- 步骤 2: 构建所有类别的文本提示 ----
    # 为每个类别创建一个自然语言提示
    text_prompts = [prompt_template.format(name) for name in class_names]

    # ---- 步骤 3: 编码图像和文本 ----
    with torch.no_grad():  # 禁用梯度计算，节省显存和加快推理
        # 使用 processor 同时处理图像和文本（自动做 padding、截断等）
        inputs = processor(
            text=text_prompts,
            images=img,
            return_tensors="pt",  # 返回 PyTorch 张量
            padding=True,         # 对文本做 padding 到相同长度
            truncation=True       # 截断过长文本
        ).to(device)

        # 模型前向传播，获取图像和文本的嵌入
        outputs = model(**inputs)
        image_embedding = outputs.image_embeds  # shape: (1, 512)
        text_embeddings = outputs.text_embeds   # shape: (N_classes, 512)

        # ---- 步骤 4: 计算余弦相似度 ----
        # CLIP 的输出已经做了 L2 归一化，所以内积等于余弦相似度
        # logits_per_image: (1, N_classes)，值越大表示越匹配
        logits_per_image = outputs.logits_per_image  # 这是 logit-scaled 的相似度

        # 用 softmax 将相似度转换为概率分布
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    # ---- 步骤 5: 排序并返回 top-k 结果 ----
    # 按概率降序排列，取前 top_k 个
    sorted_indices = np.argsort(-probs)  # 负号实现降序
    results = [(class_names[i], float(probs[i])) for i in sorted_indices[:top_k]]

    return results


def demo_zero_shot_classification(model, processor, tokenizer, device: str):
    """
    演示 1：零样本图像分类

    使用 CLIP 对样本图片进行分类，展示不需要任何训练即可识别物体的能力。
    """
    print("\n" + "=" * 70)
    print("【演示 1】CLIP 零样本图像分类")
    print("=" * 70)

    # 定义候选类别
    class_names = ["dog", "cat", "car", "pizza", "house", "airplane",
                   "bicycle", "bird", "flower", "tree", "horse", "fish"]

    # 获取样本图片
    print("\n准备样本图片...")
    image_paths = download_sample_images()

    for img_path in image_paths:
        if not os.path.exists(img_path):
            continue

        print(f"\n{'─' * 50}")
        print(f"图片: {os.path.basename(img_path)}")

        # 执行零样本分类
        results = zero_shot_classification(
            model, processor, tokenizer,
            image_path=img_path,
            class_names=class_names,
            prompt_template="a photo of a {}",
            device=device,
            top_k=5
        )

        # 打印分类结果
        print("\n  Top-5 预测:")
        print(f"  {'类别':<15} {'置信度':<12} {'柱状图'}")
        print(f"  {'─' * 50}")
        for rank, (name, score) in enumerate(results, 1):
            bar = "█" * int(score * 40)  # 用方块绘制柱状图
            marker = "← 最佳匹配" if rank == 1 else ""
            print(f"  {rank}. {name:<13} {score:.4f}      {bar} {marker}")

    print(f"\n  CLIP 不需要在这些类别上专门训练即可完成分类 —— 这就是零样本学习的力量。")
    print(f"  传统分类器需要为每个类别收集数百张标注图片，而 CLIP 只需要一段文字描述。")


# ============================================================================
# 第 3 部分：图文相似度计算
# ============================================================================

def compute_image_text_similarity(
    model,
    processor,
    tokenizer,
    image_path: str,
    captions: List[str],
    device: str = "cpu"
) -> List[Tuple[str, float]]:
    """
    计算一张图像与多个文本描述之间的相似度。

    用于场景：
    - 给定一张图片，从多个候选描述中选出最匹配的
    - 给定一段文字，从多张图片中找到最符合的

    参数:
        model: CLIP 模型
        processor: 图像预处理器
        tokenizer: 文本分词器
        image_path: 图像路径
        captions: 候选文本描述列表
        device: 运行设备

    返回:
        [(描述, 相似度分数), ...] 按分数降序排列
    """
    from PIL import Image
    import torch

    # ---- 加载并编码 ----
    img = Image.open(image_path).convert("RGB")

    with torch.no_grad():
        inputs = processor(
            text=captions,
            images=img,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        outputs = model(**inputs)
        # logits_per_image 即为图像与每个文本的相似度分数
        similarity = outputs.logits_per_image.cpu().numpy()[0]

    # ---- 排序返回 ----
    sorted_indices = np.argsort(-similarity)
    results = [(captions[i], float(similarity[i])) for i in sorted_indices]

    return results


def demo_image_text_similarity(model, processor, tokenizer, device: str):
    """
    演示 2：图文相似度计算

    展示 CLIP 如何判断一段文字是否与一张图片匹配。
    """
    print("\n" + "=" * 70)
    print("【演示 2】图文相似度计算")
    print("=" * 70)

    # ---- 2a: 给定图片，排序多个描述 ----
    print("\n--- 2a: 对图片排序候选描述 ---")

    # 检查是否有可用的图片
    dog_img = "images/samples/golden_retriever.jpg"
    if os.path.exists(dog_img):
        # 构建候选描述 —— 有些正确，有些错误
        captions = [
            "a golden retriever dog playing in the grass",  # 正确
            "a dog",                                        # 正确但笼统
            "a cute puppy with golden fur",                  # 正确
            "a cat sitting on a chair",                      # 错误
            "a red car parked on the street",                # 错误
            "a delicious pizza on a table",                  # 错误
            "an animal with four legs",                       # 部分正确
        ]

        results = compute_image_text_similarity(
            model, processor, tokenizer,
            image_path=dog_img,
            captions=captions,
            device=device
        )

        print(f"  图片: golden_retriever.jpg (金毛犬)")
        print(f"\n  描述相关性排序: (分数越高越匹配)")
        print(f"  {'排名':<6} {'分数':<10} {'描述'}")
        print(f"  {'─' * 60}")
        for rank, (caption, score) in enumerate(results, 1):
            # 标记真实匹配的描述
            is_correct = "dog" in caption.lower() or "puppy" in caption.lower() or "animal" in caption.lower()
            marker = "[匹配 ✓]" if is_correct else "[不匹配 ✗]"
            print(f"  {rank:<6} {score:<10.4f} {marker} {caption}")

    # ---- 2b: 演示跨模态语义搜索 ----
    print("\n--- 2b: 文本到图像搜索模拟 ---")
    print("  给定查询文本，在多张图片中找到最佳匹配...")

    # 获取所有可用的图片
    available_images = []
    for fname in ["golden_retriever.jpg", "orange_cat.jpg", "red_car.jpg", "pizza.jpg"]:
        fpath = f"images/samples/{fname}"
        if os.path.exists(fpath):
            available_images.append(fpath)

    if len(available_images) >= 2:
        import torch
        from PIL import Image

        queries = [
            "a cute dog",
            "a fluffy cat",
            "a vehicle on the road",
            "Italian food",
        ]

        for query in queries:
            similarities = []
            for img_path in available_images:
                img = Image.open(img_path).convert("RGB")
                with torch.no_grad():
                    inputs = processor(
                        text=[query],
                        images=img,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    ).to(device)
                    outputs = model(**inputs)
                    sim = outputs.logits_per_image.cpu().numpy()[0][0]
                    similarities.append((os.path.basename(img_path), sim))

            # 排序
            similarities.sort(key=lambda x: -x[1])
            best_img, best_score = similarities[0]

            print(f"\n  查询: 「{query}」")
            for img_name, score in similarities:
                marker = "← 最佳匹配" if img_name == best_img else ""
                print(f"    {img_name:<25} {score:.4f}  {marker}")


# ============================================================================
# 第 4 部分：嵌入空间探索
# ============================================================================

def demo_embedding_space(model, processor, tokenizer, device: str):
    """
    演示 3：嵌入空间探索

    提取图像和文本的 CLIP 嵌入向量，用 PCA 投影到 2D 并可视化，
    展示「语义相似的图文在空间中靠近」这一核心特性。
    """
    print("\n" + "=" * 70)
    print("【演示 3】嵌入空间探索 — 图文语义聚类")
    print("=" * 70)

    from PIL import Image
    import torch
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt
    matplotlib.rcParams['axes.unicode_minus'] = False

    # ---- 定义类别和对应的图文样本 ----
    # 使用多个类别的图片和文本描述
    categories = {
        "Dog": {
            "images": ["images/samples/golden_retriever.jpg"],
            "texts": ["a dog", "a golden retriever", "a cute puppy"]
        },
        "Cat": {
            "images": ["images/samples/orange_cat.jpg"],
            "texts": ["a cat", "an orange cat", "a feline"]
        },
        "Car": {
            "images": ["images/samples/red_car.jpg"],
            "texts": ["a car", "a red vehicle", "an automobile"]
        },
        "Food": {
            "images": ["images/samples/pizza.jpg"],
            "texts": ["pizza", "Italian food", "a delicious meal"]
        },
    }

    # ---- 收集所有嵌入 ----
    all_embeddings = []  # 存储所有嵌入向量
    all_labels = []      # 存储每个嵌入的标签（类别名）
    all_types = []       # 存储类型：'image' 或 'text'

    for category_name, data in categories.items():
        # --- 提取图像嵌入 ---
        for img_path in data["images"]:
            if not os.path.exists(img_path):
                continue
            img = Image.open(img_path).convert("RGB")
            with torch.no_grad():
                inputs = processor(images=img, return_tensors="pt").to(device)
                image_emb = model.get_image_features(**inputs)
                # L2 归一化（确保所有向量在同一尺度上）
                image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
                all_embeddings.append(image_emb.cpu().numpy()[0])
                all_labels.append(category_name)
                all_types.append("Image")

        # --- 提取文本嵌入 ---
        for text in data["texts"]:
            with torch.no_grad():
                inputs = tokenizer(
                    text, return_tensors="pt", padding=True, truncation=True
                ).to(device)
                text_emb = model.get_text_features(**inputs)
                # L2 归一化
                text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
                all_embeddings.append(text_emb.cpu().numpy()[0])
                all_labels.append(category_name)
                all_types.append("Text")

    if len(all_embeddings) < 3:
        print("  样本不足，跳过可视化")
        return

    # ---- 用 PCA 将高维嵌入（512 维）降到 2 维 ----
    from sklearn.decomposition import PCA

    embeddings_matrix = np.stack(all_embeddings, axis=0)  # shape: (N, 512)
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings_matrix)  # shape: (N, 2)

    # 打印 PCA 的方差解释比例
    print(f"\n  PCA 降维: 512 → 2")
    print(f"  第 1 主成分解释方差: {pca.explained_variance_ratio_[0]:.1%}")
    print(f"  第 2 主成分解释方差: {pca.explained_variance_ratio_[1]:.1%}")
    print(f"  累计解释方差: {pca.explained_variance_ratio_.sum():.1%}")

    # ---- 可视化 ----
    fig, ax = plt.subplots(figsize=(12, 8))

    # 为每个类别分配颜色
    category_colors = {
        "Dog": "#E74C3C",     # red
        "Cat": "#F39C12",     # orange
        "Car": "#3498DB",     # blue
        "Food": "#27AE60",   # green
    }

    # Assign different markers for each modality
    markers = {"Image": "o", "Text": "s"}

    for category in category_colors:
        mask = [l == category for l in all_labels]
        for vtype in ["Image", "Text"]:
            vmask = [t == vtype for t in all_types]
            idxs = [i for i in range(len(all_labels))
                    if mask[i] and vmask[i]]
            if not idxs:
                continue
            points = embeddings_2d[idxs]
            ax.scatter(
                points[:, 0], points[:, 1],
                c=category_colors[category],
                marker=markers[vtype],
                s=120 if vtype == "Image" else 80,  # 图像标记稍大
                edgecolors='white',
                linewidth=1.5,
                alpha=0.85,
                label=f"{category} - {vtype}",
                zorder=5
            )

    # 为每个类别的图像-文本对画连接线（如果有匹配对）
    for category in category_colors:
        img_mask = [(l == category) and (t == "Image")
                    for l, t in zip(all_labels, all_types)]
        txt_mask = [(l == category) and (t == "Text")
                     for l, t in zip(all_labels, all_types)]
        img_idxs = [i for i, m in enumerate(img_mask) if m]
        txt_idxs = [i for i, m in enumerate(txt_mask) if m]

        if img_idxs and txt_idxs:
            img_point = embeddings_2d[img_idxs[0]]
            for txt_idx in txt_idxs:
                txt_point = embeddings_2d[txt_idx]
                ax.plot([img_point[0], txt_point[0]],
                        [img_point[1], txt_point[1]],
                        '--', color=category_colors[category],
                        alpha=0.3, linewidth=1, zorder=2)

    # 标注关键点
    for i, (x, y) in enumerate(embeddings_2d):
        # 只为文本类别做简单标注
        if all_types[i] == "Text":
            ax.annotate(
                "", xy=(x, y), xytext=(x + 0.1, y + 0.1),
                fontsize=6, alpha=0.7, ha='center'
            )

    ax.set_xlabel("Principal Component 1 (PC1)", fontsize=12)
    ax.set_ylabel("Principal Component 2 (PC2)", fontsize=12)
    ax.set_title("PCA Visualization of CLIP Embedding Space — Image-Text Semantic Clustering", fontsize=14, fontweight='bold')

    # 图例：去重
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
              loc='lower left', fontsize=8, ncol=2,
              framealpha=0.9)

    ax.grid(True, alpha=0.3, linestyle=':')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.2)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(_IMAGES, "embedding_space_pca.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  [可视化] 嵌入空间 PCA 图已保存到 images/embedding_space_pca.png")

    # ---- 打印嵌入距离分析 ----
    print(f"\n  --- 嵌入空间距离分析 ---")
    print(f"  同类图文嵌入之间的平均距离应该小于不同类间的距离")

    # 计算各嵌入向量之间的余弦距离
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(embeddings_matrix)

    print(f"\n  余弦相似度矩阵热力图预览 (部分):")
    print(f"  {'':<20}", end="")
    for i, label in enumerate(all_labels[:6]):
        print(f"{label[:8]:<10}", end="")
    print()
    for i in range(min(12, len(all_labels))):
        print(f"  {all_labels[i]:<18} {all_types[i]:<10}", end="")
        for j in range(min(6, len(all_labels))):
            val = sim_matrix[i, j]
            # 用颜色编码相似度
            if val > 0.8:
                color = "2"  # 高
            elif val > 0.4:
                color = "1"  # 中
            else:
                color = "0"  # 低
            print(f"{val:.3f} (L{color}) ", end="")
        print()
    print(f"  L2=高相似度(>0.8), L1=中等(0.4-0.8), L0=低(<0.4)")


# ============================================================================
# 第 5 部分：对比学习损失演示
# ============================================================================

def demo_contrastive_loss():
    """
    使用 NumPy 演示 CLIP 中使用的 InfoNCE 对比损失的计算过程。
    不需要模型，纯粹展示损失函数的数学原理。
    """
    print("\n" + "=" * 70)
    print("【演示 4】InfoNCE 对比损失 — 数学原理演示")
    print("=" * 70)

    # ---- 模拟一个 batch ----
    # 假设 batch_size = 4，嵌入维度 d = 8
    batch_size = 4
    d = 8

    np.random.seed(42)
    # 创建 4 个图像嵌入和 4 个文本嵌入（已经 L2 归一化）
    image_embs = np.random.randn(batch_size, d).astype(np.float32)
    text_embs = np.random.randn(batch_size, d).astype(np.float32)

    # L2 归一化每个向量
    image_embs = image_embs / np.linalg.norm(image_embs, axis=1, keepdims=True)
    text_embs = text_embs / np.linalg.norm(text_embs, axis=1, keepdims=True)

    # ---- 人为制造一些匹配效果 ----
    # 让第 i 个图像和第 i 个文本更相似（模拟匹配的图文对）
    for i in range(batch_size):
        text_embs[i] = 0.7 * image_embs[i] + 0.3 * text_embs[i]
        text_embs[i] = text_embs[i] / np.linalg.norm(text_embs[i])

    # ---- 计算余弦相似度矩阵 S ----
    # S[i, j] = I_i · T_j（因为向量已归一化，内积等于余弦相似度）
    S = image_embs @ text_embs.T  # shape: (4, 4)

    print(f"\n  相似度矩阵 S (4×4):")
    print(f"  (行=图像索引 i, 列=文本索引 j)")
    print(f"  {'':>10}", end="")
    for j in range(batch_size):
        print(f"  T_{j}  ", end="")
    print()
    for i in range(batch_size):
        print(f"  I_{i}    ", end="")
        for j in range(batch_size):
            highlight = ">" if i == j else " "
            print(f"{S[i, j]:+.3f}{highlight} ", end="")
        print()
    print(f"  (> 标记对角线 = 匹配的图文对)")

    # ---- 计算 InfoNCE 损失 ----
    # 温度参数 τ
    tau = 0.07

    # 图像方向的损失
    # logits = S / τ
    logits_image = S / tau  # 除以温度以使分布更尖锐
    # softmax 分子：匹配对
    numerator_image = np.exp(np.diag(logits_image))  # 对角线 = S[i,i]/τ
    # softmax 分母：每行所有元素
    denominator_image = np.sum(np.exp(logits_image), axis=1)
    # 逐样本损失
    loss_per_image = -np.log(numerator_image / denominator_image)
    loss_image = np.mean(loss_per_image)

    # 文本方向的损失（对称）
    logits_text = S.T / tau
    numerator_text = np.exp(np.diag(logits_text))
    denominator_text = np.sum(np.exp(logits_text), axis=1)
    loss_per_text = -np.log(numerator_text / denominator_text)
    loss_text = np.mean(loss_per_text)

    # 总损失 = 两者的平均
    loss_clip = 0.5 * (loss_image + loss_text)

    print(f"\n  InfoNCE 损失计算 (τ = {tau}):")
    print(f"  {'─' * 50}")
    print(f"  L_image: {loss_image:.6f}  (给定图像，选出正确文本)")
    print(f"  L_text:  {loss_text:.6f}  (给定文本，选出正确图像)")
    print(f"  L_CLIP:  {loss_clip:.6f}  (= (L_image + L_text) / 2)")
    print(f"  {'─' * 50}")

    # ---- 展示对比损失的行为 ----
    print(f"\n  --- 损失函数行为分析 ---")

    # 场景 1: 完美匹配（对角线 = 1，其余 = 0）
    S_perfect = np.eye(batch_size, dtype=np.float32)
    logits_p = S_perfect / tau
    loss_p = -np.log(np.exp(np.diag(logits_p)) / np.sum(np.exp(logits_p), axis=1)).mean()
    print(f"\n  极端场景分析:")
    print(f"  1) 完美对齐 (S=单位矩阵): L ≈ {loss_p:.6f}  (理论最小值)")
    print(f"     对角线相似度=1，其余=0 → 分类器 100% 确定")

    # 场景 2: 完全随机（所有相似度相等）
    S_random = np.full((batch_size, batch_size), 0.0)
    np.fill_diagonal(S_random, 0.0)
    logits_r = S_random / tau
    loss_r = -np.log(np.exp(np.diag(logits_r)) / np.sum(np.exp(logits_r), axis=1)).mean()
    print(f"  2) 完全随机 (S≈0 矩阵): L ≈ {loss_r:.6f}")
    # 理论上随机时 loss ≈ -log(1/batch_size)
    print(f"     理论值 = -log(1/N) = -log(1/{batch_size}) = {(-np.log(1.0/batch_size)):.4f}")
    print(f"     → 对比学习通过增大 batch size 提供更多负样本，提高学习难度")


# ============================================================================
# 第 6 部分：主程序
# ============================================================================

def main():
    """
    主程序：加载 CLIP 模型并运行所有演示。

    流程：
    1. 检测环境（PyTorch, transformers）
    2. 加载 CLIP ViT-B/32 模型
    3. 演示 1: 零样本图像分类
    4. 演示 2: 图文相似度计算
    5. 演示 3: 嵌入空间探索 (PCA 可视化)
    6. 演示 4: InfoNCE 对比损失演示 (纯 NumPy)
    """
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 10 + "s22 多模态模型 — CLIP 从零演示" + " " * 28 + "║")
    print("║" + " " * 6 + "零样本分类 · 图文相似度 · 嵌入空间可视化 · 对比损失" + " " * 11 + "║")
    print("╚" + "═" * 68 + "╝")

    # ---- 检测环境 ----
    ready, device = check_environment()

    # ---- 演示 4 不依赖模型，先运行 ----
    demo_contrastive_loss()

    # ---- 加载模型并运行视觉相关演示 ----
    if ready:
        model, processor, tokenizer = load_clip_model(device)

        if model is not None:
            # 演示 1: 零样本分类
            demo_zero_shot_classification(model, processor, tokenizer, device)

            # 演示 2: 图文相似度
            demo_image_text_similarity(model, processor, tokenizer, device)

            # 演示 3: 嵌入空间探索
            try:
                import sklearn
                demo_embedding_space(model, processor, tokenizer, device)
            except ImportError:
                print("\n[跳过] 演示 3 需要 scikit-learn (pip install scikit-learn)")
        else:
            print("\n[跳过] 模型加载失败，仅运行纯数学演示。")
    else:
        print("\n[跳过] 环境不满足要求，仅运行纯数学演示。")
        print("请安装依赖: pip install torch transformers pillow scikit-learn")

    # ---- 最终总结 ----
    print("\n" + "=" * 70)
    print("【s22 总结】")
    print("=" * 70)
    print("  ✓ 理解了 CLIP 的双编码器架构（图像 + 文本）")
    print("  ✓ 理解了 InfoNCE 对比损失的工作原理")
    print("  ✓ 体验了零样本图像分类 — 无需标注数据的奇迹")
    print("  ✓ 感受了共享嵌入空间中「语义相似 = 向量相近」")
    print()
    print("  CLIP 是多模态 AI 的基石：")
    print("  - 它证明了自然语言可以作为图像的监督信号")
    print("  - 它构建的共享嵌入空间是 LLaVA、DALL-E 等模型的基础")
    print("  - 零样本能力预示了 AI 从「专用工具」向「通用能力」的转变")
    print("=" * 70)


if __name__ == "__main__":
    main()
