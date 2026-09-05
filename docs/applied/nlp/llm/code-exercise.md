---
title: "s18 大语言模型 — exercise.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# s18 大语言模型 — exercise.py 练习指南

<a href="/notebook/code/applied/nlp/llm/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过动手实现 Scaling Law 最优配比计算、DPO 损失函数和 LoRA 配置，建立对 LLM 核心概念的量化直觉。完成后你将能够：

1. 给定计算预算，找出 Chinchilla 最优的参数量和数据量配比
2. 独立写出 DPO 损失函数的 PyTorch 实现
3. 理解 LoRA 配置中的关键参数

## 预备知识

- **Scaling Law**：$L(N, D) = a/N^{\alpha} + b/D^{\beta} + c$，其中 $\alpha \approx 0.076$, $\beta \approx 0.095$
- **Chinchilla 最优配比**：$D \approx 20N$，训练计算量 $C \approx 6ND$
- **DPO 损失**：$-\log \sigma(\beta \cdot (\Delta_{\text{policy}} - \Delta_{\text{ref}}))$
- **LoRA**：$h = Wx + \frac{\alpha}{r} BA x$，其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$

---

## 任务清单

### 练习 1：实现 Chinchilla 最优配比计算

**目标**：给定计算预算 $C$，找到使损失最小的参数量 $N$ 和数据量 $D$。

**核心约束**：训练计算量 $C \approx 6ND$（前向传播约 $2ND$ FLOPs，反向传播约 $4ND$ FLOPs）。

**求解方法**：在给定的 $C$ 约束下，问题变为找到最小化 $L(N, D)$ 的 $N$ 和 $D$：

$$
\min_{N, D} \frac{a}{N^{\alpha}} + \frac{b}{D^{\beta}} + c \quad \text{s.t.} \quad C = 6ND
$$

使用拉格朗日乘数法可以推导出解析解，但练习中使用更直观的**对数空间网格搜索**：

```python
def find_optimal_ND(compute_budget, a=1.5, b=2.0, alpha=0.076, beta=0.095):
    # 在对数空间中搜索 N
    N_candidates = np.logspace(6, 12, 200)          # 1M → 1T 参数
    best_loss = float('inf')
    best_N, best_D = None, None

    for N in N_candidates:
        D = compute_budget / (6 * N)                 # 由约束确定 D
        if D <= 0:
            continue
        loss = a / (N ** alpha) + b / (D ** beta) + 1.0
        if loss < best_loss:
            best_loss = loss
            best_N = N
            best_D = D

    return best_N, best_D, best_loss
```

**关键步骤**：
1. 对 $N$ 做对数均匀采样（`np.logspace(6, 12, 200)`）
2. 对每个 $N$，由 $C = 6ND$ 计算 $D$（从约束推导）
3. 计算对应的损失 $L(N, D)$
4. 找到使损失最小的 $(N, D)$ 对

**检验标准**：最优配比 $D/N$ 应该约为 20（即 D=20N）。

**预期输出**：
```
[练习1] Chinchilla 最优配比:
  给定计算预算 C=5.88e+22 FLOPs
  最优参数量 N=7.00e+09 (~7.0B)
  最优数据量 D=1.40e+12 tokens (~1.40T)
  最优配比 D/N=200.0 (期望: ≈20)
```

---

### 练习 2：实现 DPO 损失函数

**目标**：补全 `dpo_loss()` 函数，实现 DPO 损失的计算。

**核心公式**：

$$
\mathcal{L}_{\text{DPO}} = -\frac{1}{N} \sum_{i=1}^{N} \log \sigma\left( \beta \cdot \left[ (\log\pi_\theta(y_w|x) - \log\pi_\theta(y_l|x)) - (\log\pi_{\text{ref}}(y_w|x) - \log\pi_{\text{ref}}(y_l|x)) \right] \right)
$$

**TODO 步骤**：

```python
def dpo_loss(pi_logps_chosen, pi_logps_rejected,
             ref_logps_chosen, ref_logps_rejected, beta=0.1):
    # 步骤 1: 计算策略模型的"偏好差异"
    pi_log_ratio = pi_logps_chosen - pi_logps_rejected
    # 步骤 2: 计算参考模型的"偏好差异"
    ref_log_ratio = ref_logps_chosen - ref_logps_rejected
    # 步骤 3: 策略模型相对于参考模型的改善
    logits = beta * (pi_log_ratio - ref_log_ratio)
    # 步骤 4: -log σ(logits)
    loss = -F.logsigmoid(logits).mean()
    return loss
```

**逐步理解**：

1. **`pi_log_ratio`**：如果策略模型正确偏好（给好回答更高的 log 概率）→ 此值为正 → 好信号
2. **`ref_log_ratio`**：参考模型的基准偏好。即使参考模型的偏好也是对的，策略模型仍可以比它做得更好
3. **`logits = beta * (pi_log_ratio - ref_log_ratio)`**：策略模型**相对于**参考模型的改善程度。这个值越大，DPO 损失越小
4. **`-F.logsigmoid(logits)`**：`logsigmoid(x) = log(1/(1+e^{-x}))`，其范围为 $(-\infty, 0)$。取负号后损失为正，当 logits 很大时损失趋近于 0

**`logsigmoid` vs `log(sigmoid(.))`**：`F.logsigmoid` 在数值上比 `torch.log(torch.sigmoid(x))` 更稳定（前者直接计算 `-softplus(-x)`，避免了 sigmoid 可能出现的数值溢出）。

**预期输出**：
```
[练习2] DPO 损失:
  好回答log P=[-1.5, -2.0, -1.8], 差回答log P=[-4.0, -5.0, -4.5]
  DPO Loss = XXXXX (期望: 一个较小的正数)
```

当策略模型的 $P(y_w|x) \gg P(y_l|x)$ 时，损失应很小（模型已经很好地学会区分好/差回答）。

---

### 练习 3：配置 LoRA 适配器

**目标**：创建 LoRA 配置字典，理解每个参数的含义。

**LoRA 的核心参数**：

| 参数 | 典型值 | 含义 |
|------|--------|------|
| `r` | 8, 16, 32, 64 | 低秩分解的秩。越大表达能力越强，但参数量也越大 |
| `lora_alpha` | 16, 32 | 缩放系数。越大 LoRA 更新的影响越大。实际缩放 = alpha/r |
| `target_modules` | ["q_proj", "v_proj"] | 对哪些模块应用 LoRA。Qwen/Llama 系列通常加在 Q 和 V 投影上 |
| `lora_dropout` | 0.0 - 0.1 | LoRA 的 dropout 概率。小数据集上可设为 0.05-0.1 |
| `bias` | "none" | LoRA 不训练 bias，保持与原始模型的 bias 一致 |
| `task_type` | "CAUSAL_LM" | 任务类型：因果语言模型（GPT 系列） |

**TODO 步骤**：

```python
def create_lora_config(r=8, lora_alpha=16.0, target_modules=None,
                       lora_dropout=0.1):
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    config = {
        "r": r,
        "lora_alpha": lora_alpha,
        "target_modules": target_modules,
        "lora_dropout": lora_dropout,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    return config
```

**常见配置建议**：
- 仅 Q+V 投影：参数量最小，效果通常够用（`["q_proj", "v_proj"]`）
- Q+K+V+O 投影：参数量翻倍，效果更好（`["q_proj", "k_proj", "v_proj", "o_proj"]`）
- 全注意力+FFN：参数量最多，但某些任务可能需要（加上 `["gate_proj", "up_proj", "down_proj"]`）

**为什么主要在 Attention 投影上加 LoRA？** 注意力层负责"查找"相关信息，任务适配的核心在于改变"模型关注什么"。FFN 层更多是"知识存储"，改变 FFN 可能导致原有知识被覆盖。但实践中，在 FFN 上加 LoRA 有时效果也很好。

**预期输出**：
```
[练习3] LoRA 配置: {'r': 16, 'lora_alpha': 32.0, 'target_modules': [...], ...}
```

---

## 三个练习的关系

| 练习 | LLM 概念 | 在 LLM 创业/应用中的位置 |
|------|---------|------------------------|
| 练习 1: Chinchilla 配比 | 如何分配算力 | 训练新模型的决策依据 |
| 练习 2: DPO 损失 | 如何对齐人类偏好 | 微调模型使其"更好用" |
| 练习 3: LoRA 配置 | 如何高效微调 | 消费级硬件上微调大模型 |

这三个概念在 LLM 产品开发中都有直接应用：决定模型规模和数据量（练习 1）、通过偏好数据优化模型行为（练习 2）、在有限算力下微调模型（练习 3）。

## 检查要点

运行 `python exercise.py`，确认：
- [ ] 练习 1 D/N 配比接近 20
- [ ] 练习 2 DPO 损失为正数，正确偏好场景损失 < 错误偏好场景
- [ ] 练习 3 LoRA 配置包含所有必需字段

完成练习后，返回 demo.py 观察这些概念的完整实现和可视化。


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/applied/nlp/llm/code/exercise.py`
