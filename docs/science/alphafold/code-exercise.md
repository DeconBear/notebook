---
title: "as06 蛋白质结构预测与 AlphaFold — exercise.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# as06 蛋白质结构预测与 AlphaFold — exercise.py 练习指南

<a href="/notebook/code/science/alphafold/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过补全三个 TODO 任务，理解 AlphaFold 简化流水线的三个核心计算步骤：
1. 从 3D 坐标计算接触图（结构预测评估的"标准答案"是怎么来的）
2. 用 Outer Product Mean 思想从 MSA 提取共进化耦合矩阵
3. 实现距离几何重建的梯度计算（简化版"结构模块"）

## 预备知识

在开始前请确保理解：
- 接触图定义：残基对欧氏距离小于阈值即为"接触"，序列近邻天然接触不计入评估
- 共进化直觉：结构接触的残基对，突变往往"配对"发生（一个变了，另一个跟着补偿性变化）
- Outer Product Mean：$\text{joint}_{ij} = \frac{1}{M}\sum_m \text{onehot}(a_i^{(m)}) \otimes \text{onehot}(a_j^{(m)})$，
  衡量两列字符分布是否独立
- 距离几何：$\mathcal{L}(X) = \sum_{i<j} w_{ij}(\|x_i-x_j\| - d_{ij}^{\text{target}})^2$，
  用梯度下降从目标距离约束反解 3D 坐标

## 任务清单

### TODO 1：实现接触图计算（`compute_contact_map` 函数）

**任务**：根据 3D 坐标计算 0/1 接触矩阵。

**实现步骤**：
1. 计算两两差向量：`diff = coords[:, None, :] - coords[None, :, :]`，形状 `(N, N, 3)`
2. 计算欧氏距离：`dist = np.linalg.norm(diff, axis=-1)`，形状 `(N, N)`
3. 阈值化：`contact = (dist < threshold).astype(np.float32)`
4. 屏蔽序列近邻：`|i-j| <= 1` 的位置置 0（这些残基天然共价相连，接触信息没有价值）

**关键易错点**：
- `axis=-1` 是对最后一维（3 维坐标）求范数，不要写成 `axis=0` 或 `axis=1`
- 屏蔽近邻时注意边界条件（`i=0` 和 `i=N-1` 时的范围裁剪）

**预期输出**：
```
接触矩阵中 contact[0,1] = 0（序列近邻被屏蔽）
contact[0,4] = 1（长程接触被正确识别）
```

### TODO 2：实现 Outer Product Mean 耦合矩阵（`outer_product_mean_coupling` 函数）

**任务**：从 one-hot 编码的 MSA 中提取残基对的共进化耦合强度。

**实现步骤**：
1. 单列频率：`freq = one_hot.mean(axis=0)`，形状 `(N, K)`
2. 对每一对位置 $(i,j)$：
   - 联合频率（外积均值）：`joint = (one_hot[:, i, :, None] * one_hot[:, j, None, :]).mean(axis=0)`
   - 独立假设期望值：`indep = np.outer(freq[i], freq[j])`
   - 耦合强度：`score = np.linalg.norm(joint - indep)`

**关键理解点**：
- `one_hot[:, i, :, None]` 形状是 `(M, K, 1)`，`one_hot[:, j, None, :]` 形状是 `(M, 1, K)`，
  两者相乘触发广播机制得到 `(M, K, K)`，对 `axis=0` 取均值即为外积均值
- 如果 $i, j$ 两列的字符完全独立，`joint` 应该非常接近 `indep`，耦合分数接近 0
- 如果两列强相关（比如完全同步突变），`joint` 会偏离 `indep` 很多，耦合分数显著更大

**预期输出**：
```
coupling[0,1] (强耦合对，人为构造完全同步的突变) ≈ 远大于 coupling[0,2] (独立对)
比如: coupling[0,1] ≈ 0.3-0.5，coupling[0,2] ≈ 0.05 以下
```

### TODO 3：实现距离几何梯度计算（`distance_geometry_gradient` 函数）

**任务**：计算"用坐标满足目标距离约束"这一损失函数对坐标的梯度。

**数学公式**：

$$
\mathcal{L}(X) = \sum_{i<j} w_{ij}\left(\|x_i - x_j\| - d_{ij}^{\text{target}}\right)^2
\quad\Rightarrow\quad
\frac{\partial \mathcal{L}}{\partial x_i} = \sum_j 2 w_{ij}\left(\|x_i-x_j\| - d_{ij}^{\text{target}}\right)\frac{x_i - x_j}{\|x_i - x_j\|}
$$

**实现步骤**：
1. 差向量：`diff = coords[:, None, :] - coords[None, :, :]`
2. 距离（加 epsilon 防止除零）：`dist = np.linalg.norm(diff, axis=-1) + 1e-8`
3. 残差：`err = dist - target_dist`
4. 系数：`coef = 2 * weight * err / dist`，对角线置 0
5. 梯度：`grad = (coef[:, :, None] * diff).sum(axis=1)`

**关键易错点**：
- 一定要在 `dist` 上加一个小的 epsilon，否则当两点重合时会除零报错（`nan`）
- 系数计算里的除法是逐元素的 `err / dist`，不是矩阵除法
- 别忘了 `np.fill_diagonal(coef, 0.0)`，否则每个点会对自己产生一个错误的"自梯度"

**预期输出**：
```
两点当前距离 4.0，目标距离 2.0（当前比目标远，需要拉近）
grad[0, 0] < 0，grad[1, 0] > 0
（梯度下降 x_new = x - lr*grad 后，两点会互相靠近）
```

## 完成后的验证

全部三个 TODO 通过测试后，运行 `python code/demo.py` 观察完整流水线效果：
1. 合成 MSA 与真实接触图、预测耦合矩阵的对比图
2. CASP 式 precision@L / L/2 / L/5 精度指标
3. 从耦合矩阵重建的 3D 结构与真实骨架的对比（含 RMSD）


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/science/alphafold/code/exercise.py`
