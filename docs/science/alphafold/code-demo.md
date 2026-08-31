---
title: "as06 蛋白质结构预测与 AlphaFold — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# as06 蛋白质结构预测与 AlphaFold — demo.py 代码详解

<a href="/notebook/code/science/alphafold/demo.py" target="_blank" download>Download demo.py</a>

> ⚠️ **重要提示**：本演示是一个纯教学玩具，**不是 AlphaFold**。它只用几百行 NumPy 代码
> 重现 AlphaFold 流水线背后的统计/几何直觉（共进化信号、Outer Product Mean、
> 距离几何重建），完全不涉及真实的深度网络训练或推理。

## 运行方式

```bash
cd docs/science/alphafold/code
python demo.py
```

## 代码逐段详解

### 第1步：合成一条"真实"蛋白骨架

真实蛋白的折叠遵循复杂的物理规律，但本演示只需要一组"具有局部规律 + 长程接触"
的 3D 坐标作为教学用的 ground truth，因此用简单的几何规则手工构造：

```python
def generate_synthetic_backbone(n_residues=40, seed=42):
    coords = np.zeros((n_residues, 3))
    bond_len = 3.8    # Cα-Cα 典型键长
    ...
    for i in range(1, n_residues):
        if i < seg1:
            # α-螺旋段：绕固定轴规则旋转
            ...
        elif i < seg2:
            # 转角区域：随机扭转，制造长程接触
            ...
        else:
            # 延伸段：朝螺旋中段靠拢，制造跨段接触
            ...
    return coords
```

**关键设计**：把序列分成螺旋段、转角段、延伸段三部分，并让延伸段主动"靠拢"螺旋段——
这样生成的结构里既有局部规律（螺旋的固定周期），也有若干**长程接触**（序列上相距很远、
空间上却很近的残基对），这正是接触图预测要抓住的核心信号类型。

### 第2步：计算真实接触图

```python
def compute_contact_map(coords, threshold=8.0):
    diff = coords[:, None, :] - coords[None, :, :]   # (N, N, 3) 两两差向量
    dist = np.linalg.norm(diff, axis=-1)               # (N, N) 距离矩阵
    contact = (dist < threshold).astype(np.float32)
    # 序列近邻(|i-j|<=1)天然接触，无信息量，置0不计入评估
    for i in range(n):
        for j in range(max(0, i-1), min(n, i+2)):
            contact[i, j] = 0
    return contact, dist
```

这一步得到的 `contact` 矩阵是后续评估的"标准答案"（ground truth）——真实场景中，
这类信息只能通过实验解析结构获得，AlphaFold 要做的正是**不看真实结构、只从序列/MSA 预测它**。

### 第3步：合成多序列比对 (MSA) — 注入共进化信号

**数学背景**：如果残基 $i, j$ 在结构上接触，演化压力会让它们的突变发生"配对"——
一个位点变化后，另一个位点往往要发生补偿性突变以维持局部结构稳定。这种统计相关性
就是 **共进化（co-evolution）**。

```python
def generate_synthetic_msa(n_residues, contact_map, n_sequences=300,
                            mutation_rate=0.35, coupling_strength=0.85, seed=42):
    ancestor = rng.randint(0, ALPHABET_SIZE, size=n_residues)   # 祖先序列
    msa = np.tile(ancestor, (n_sequences, 1))

    contact_pairs = [(i, j) for i in range(n_residues) for j in range(i+1, n_residues)
                      if contact_map[i, j] > 0]

    for s in range(n_sequences):
        seq = msa[s].copy()
        # 独立随机突变
        mask = rng.random(n_residues) < mutation_rate
        seq[mask] = rng.randint(0, ALPHABET_SIZE, size=mask.sum())
        # 共进化配对突变：接触残基对联合突变（用相同偏移量）
        for (i, j) in contact_pairs:
            if rng.random() < coupling_strength * mutation_rate:
                shift = rng.randint(1, ALPHABET_SIZE)
                seq[i] = (ancestor[i] + shift) % ALPHABET_SIZE
                seq[j] = (ancestor[j] + shift) % ALPHABET_SIZE   # 配对：相同偏移=相关突变
        msa[s] = seq
    return msa
```

**关键细节**：`seq[i]` 和 `seq[j]` 使用**同一个** `shift` 值来突变——这就制造出了
"两列字符的联合分布偏离独立假设"的统计信号，也就是共进化信号的最简化版本。真实的
共进化远比这复杂（涉及氨基酸的物理化学性质、进化树结构等），但统计直觉是相通的。

### 第4步：Outer Product Mean — 从 MSA 提取耦合矩阵

这一步是对 AlphaFold2 Evoformer 中"MSA 表示 → Pair 表示"的核心操作
**Outer Product Mean** 的教学复刻：

$$
\text{joint}_{ij}[a, b] = \frac{1}{M}\sum_{m=1}^{M} \mathbb{1}[a_i^{(m)} = a] \cdot \mathbb{1}[a_j^{(m)} = b]
$$

即对每条序列 $m$，取残基 $i, j$ 处字符的 one-hot 向量做外积，再对所有序列取均值——
这正是"Outer Product Mean"字面意义上的操作。如果 $i, j$ 独立，联合分布应接近
$\text{freq}_i \otimes \text{freq}_j$；耦合强度定义为两者的偏离程度：

$$
\text{coupling}_{ij} = \left\| \text{joint}_{ij} - \text{freq}_i \otimes \text{freq}_j \right\|_F
$$

```python
def outer_product_mean_coupling(msa, n_residues):
    one_hot = np.zeros((m, n, ALPHABET_SIZE), dtype=np.float32)
    one_hot[rows, cols, msa] = 1.0                   # (M, N, K) one-hot 编码

    freq = one_hot.mean(axis=0)                       # (N, K) 单列频率 P(a_i)
    for i in range(n):
        for j in range(i + 1, n):
            joint = (one_hot[:, i, :, None] * one_hot[:, j, None, :]).mean(axis=0)  # 外积均值
            indep = np.outer(freq[i], freq[j])         # 独立假设下的期望联合频率
            score = np.linalg.norm(joint - indep)       # Frobenius 范数 = 耦合强度
            coupling[i, j] = coupling[j, i] = score
    return coupling
```

**为什么这和互信息（Mutual Information）异曲同工**：互信息衡量的正是联合分布与
独立分布之间的 KL 散度；这里用 Frobenius 范数做了一个更简单、计算更快的近似，
两者在"检测列间统计依赖"这个目的上是等价的直觉。

### 第5步：CASP 式评估指标 — precision@L

```python
def evaluate_contact_precision(pred_coupling, true_contact):
    iu = np.triu_indices(n, k=2)          # 只看上三角、排除相邻残基
    scores = pred_coupling[iu]
    labels = true_contact[iu]
    order = np.argsort(-scores)            # 按预测分数从高到低排序
    sorted_labels = labels[order]

    for frac, name in [(1.0, 'L'), (0.5, 'L/2'), (0.2, 'L/5')]:
        k = max(1, int(round(n * frac)))
        precision = sorted_labels[:k].mean()   # top-k 预测里有多少是真接触
```

**CASP 传统**：precision@L/5（只看最有把握的少数预测）往往比 precision@L（看所有可能
的预测）高得多——这符合直觉：模型最有信心的那一小部分预测通常也是最准的。

### 第6步：简化版"结构模块" — 距离几何重建 3D 坐标

真实的 Structure Module 用 Invariant Point Attention 端到端直接回归坐标。这里用一个
更朴素但足够展示核心几何直觉的方法：把耦合分数转换为目标距离约束，再用（带动量的）
梯度下降求解坐标，使实际距离尽量匹配目标距离——这本质上是经典的
**距离几何 / 应力多维标度（Stress Majorization / MDS）** 方法。

$$
\mathcal{L}(X) = \sum_{i<j} w_{ij}\left(\|x_i - x_j\| - d_{ij}^{\text{target}}\right)^2
$$

$$
\frac{\partial \mathcal{L}}{\partial x_i} = \sum_j 2 w_{ij}\left(\|x_i-x_j\| - d_{ij}^{\text{target}}\right)\frac{x_i - x_j}{\|x_i - x_j\|}
$$

```python
for restart in range(n_restarts):                  # 多次随机初始化，取损失最低的一次
    coords = rng.normal(scale=5.0, size=(n, 3))
    velocity = np.zeros_like(coords)
    for it in range(n_iters):
        diff = coords[:, None, :] - coords[None, :, :]
        dist = np.linalg.norm(diff, axis=-1) + 1e-8
        err = dist - target_dist
        coef = 2.0 * weight * err / dist
        np.fill_diagonal(coef, 0.0)
        grad = (coef[:, :, None] * diff).sum(axis=1)
        velocity = momentum * velocity - lr * grad    # 带动量的梯度下降
        coords = coords + velocity
```

**目标距离的构造规则**：
- 序列相邻残基（$|i-j|=1$）：固定目标距离 3.8Å（共价键长），权重最高
- 序列近邻（$|i-j|=2$）：目标距离 5.5Å（典型二级结构间距），弱权重
- 其他残基对：耦合分数越高，目标距离越接近 6~7Å（代表更可能接触），权重也随耦合分数增大

**多次随机重启的意义**：梯度下降容易陷入局部最优（比如把链"缠绕"成一个错误的拓扑），
多次从不同随机初始化开始、保留损失最低的一次，能在一定程度上缓解这个问题——这与真实
结构预测流程中"生成多个候选结构、按置信度/能量挑选最佳"的思路是一致的。

### 第7步：Kabsch 对齐与 RMSD

结构预测评估时，我们并不关心预测结构"整体摆在空间中的哪个位置、朝向哪个方向"，
只关心相对几何形状是否正确。因此需要先用 **Kabsch 算法** 找到最优旋转，把预测结构
对齐到真实结构上，再计算均方根偏差（RMSD）：

```python
def kabsch_align(mobile, target):
    h = mobile_c.T @ target_c                         # (3,3) 协方差矩阵
    u, s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))              # 防止反射（镜像）
    r = vt.T @ np.diag([1, 1, d]) @ u.T                  # 最优旋转矩阵
    aligned = (r @ mobile_c.T).T
    rmsd = np.sqrt(((aligned - target_c) ** 2).sum(axis=1).mean())
```

**为什么要防止反射**：SVD 分解直接得到的旋转矩阵可能带有镜像翻转（一个手性到另一个手性），
这在物理上是不允许的（蛋白质是手性分子），所以用 `det` 的符号做一次修正。

### 第8步：可视化

1. **流水线示意图**（`as06-01-alphafold-pipeline.png`）：AlphaFold 完整流程的教学示意图
2. **MSA 与接触图对比**（`contact_map_prediction.png`）：合成 MSA、真实接触图、预测耦合矩阵三者并排展示
3. **精度柱状图**（`precision_at_l.png`）：precision@L / L/2 / L/5 三个 CASP 式指标
4. **结构重建对比**（`structure_reconstruction.png`）：重建结构与真实骨架的 3D 对比（含 RMSD）

### 关键概念速查表

| 概念 | 一句话解释 | 代码位置 |
|------|-----------|---------|
| 接触图 (Contact Map) | 残基对空间距离 < 阈值即为接触，是结构预测评估的基础 | `compute_contact_map()` |
| 共进化信号 | 结构接触的残基对，突变往往配对发生 | `generate_synthetic_msa()` |
| Outer Product Mean | 计算 MSA 列间外积均值，提取共进化耦合强度 | `outer_product_mean_coupling()` |
| precision@L | CASP 式指标：预测最有信心的 top-k 对里有多少是真接触 | `evaluate_contact_precision()` |
| 距离几何重建 | 把耦合分数转为目标距离，梯度下降求 3D 坐标 | `reconstruct_structure()` |
| Kabsch 对齐 + RMSD | 消除整体旋转平移后比较两个结构的差异 | `kabsch_align()` |

## 完整代码

<<< @/science/alphafold/code/demo.py
