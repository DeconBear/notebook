---
title: "s18 大语言模型 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# s18 大语言模型 — demo.py 代码详解

<a href="/notebook/code/applied/nlp/llm/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/applied/nlp/llm/code
python demo.py
```

**依赖**：`numpy`, `torch`, `matplotlib`

---

## 代码逐段详解

### 第1步：Scaling Law — 语言模型损失的幂律下降

#### 1.1 Kaplan Scaling Law

Kaplan et al. (2020) 发现语言模型的测试损失 $L$ 随模型参数量 $N$ 和训练数据量 $D$ 的增长呈**幂律下降**：

$$
L(N, D) = \frac{a}{N^{\alpha}} + \frac{b}{D^{\beta}} + c
$$

其中 $c$ 是**不可约减损失**（irreducible loss）——由数据本身的熵决定，无论模型多大都无法消除。

```python
def kaplan_loss(N, D, a=1.5, b=2.0, alpha=0.076, beta=0.095, c=1.0):
    return a / (N ** alpha) + b / (D ** beta) + c
```

**逐参数解释**：

| 参数 | 值 | 含义 |
|------|-----|------|
| $N$ | 变量 | 模型参数量 |
| $D$ | 变量 | 训练数据量（token 数） |
| $\alpha \approx 0.076$ | 固定 | $L$ 对 $N$ 的幂律指数。翻倍 $N$，损失下降约 $2^{-0.076} \approx 94.9\%$ |
| $\beta \approx 0.095$ | 固定 | $L$ 对 $D$ 的幂律指数。翻倍 $D$，损失下降约 $2^{-0.095} \approx 93.6\%$ |
| $c = 1.0$ | 固定 | 不可约减损失的下界 |
| $a, b$ | 固定 | 比例系数 |

**幂律的含义**：在 log-log 图上，损失随参数量的增加呈直线下降——这意味着每将参数量翻倍，损失按**固定比例**减少。这与直觉"收益递减"一致——从 1M 到 10M 参数的效果提升远大于从 100B 到 1T。

#### 1.2 Chinchilla 最优配比

2022 年 DeepMind 的 Chinchilla 论文指出：Kaplan 的计算最优分配偏向于"模型大、数据少"，但正确的做法是**数据和参数同步增长**。

$$
D_{\text{opt}} \approx 20 \times N
$$

```python
def chinchilla_optimal_D(N):
    return 20.0 * N
```

**实际含义**：用 175B 参数的 GPT-3 应该训练约 3.5T tokens，但它只用了约 300B tokens——GPT-3 是"欠训练"的。按照 Chinchilla 最优配比，在 GPT-3 的算力预算下，训练一个 70B 参数 + 1.4T tokens 的模型（如 Chinchilla 或 LLaMA 7B）反而效果更好。

#### 1.3 可视化

代码绘制了四张子图，展示 Scaling Law 的四个维度：

- **图 1: $L(N)$ — 损失 vs 模型大小**：log-log 图上的直线，标注 GPT-1(117M)、GPT-2(1.5B)、GPT-3(175B) 的位置
- **图 2: $L(D)$ — 损失 vs 数据量**
- **图 3: $L(C)$ — 损失 vs 计算量**（$C \approx 6ND$，训练所需的 FLOPs）
- **图 4: Chinchilla 等高线**：显示不同 $(N, D)$ 组合下的损失，红色虚线标出了 $D \approx 20N$ 的最优线

```python
# GPT-3 位置：参数很大但数据不足，位于最优线右下方
ax4.scatter([1.75e11], [3e11], color='orange', s=100, marker='s')
ax4.annotate('GPT-3\n(Undertrained)', (2e11, 4e11))

# LLaMA 7B 位置：参数较小但数据充足，位于最优线附近
ax4.scatter([7e9], [1e12], color='green', s=100, marker='^')
ax4.annotate('LLaMA 7B\n(Near-optimal)', (1e10, 1.5e12))
```

---

### 第2步：涌现能力模拟 — 量变引起质变

#### 2.1 什么是涌现？

涌现（Emergence）是指：某些能力在小模型中**完全不存在**（表现为随机水平），但当模型规模跨过某个阈值后，性能**突然跃升**到接近完美的水平。

#### 2.2 Sigmoid 模型模拟涌现

代码使用 **sigmoid 函数** 来模拟涌现的"相位转变"行为：

```python
def simulate_emergence(param_sizes, task, emergent=True, threshold=1e9, noise_level=0.05):
    if emergent:
        # Sigmoid 模拟相位转变：
        # 1 / (1 + exp(-k * (log10(N) - log10(threshold))))
        accuracies = 1.0 / (1.0 + np.exp(
            -1.5 * (np.log10(param_sizes) - np.log10(threshold))
        ))
        accuracies = 0.05 + 0.85 * accuracies  # 基线 5% + 最大提升 85%
    else:
        # 非涌现：平滑线性增长
        accuracies = 0.1 + 0.8 * (np.log10(param_sizes) - 6.0) / 6.0
        accuracies = np.clip(accuracies, 0.1, 0.95)
    accuracies += np.random.normal(0, noise_level, len(param_sizes))  # 加噪声
    return np.clip(accuracies, 0.0, 1.0)
```

**Sigmoid 函数的妙用**：当 $\log_{10}(N) \ll \log_{10}(\text{threshold})$ 时，指数 $e^{-\text{大正数}} \to 0$，准确率接近随机基线 5%。当 $\log_{10}(N) \gg \log_{10}(\text{threshold})$ 时，$e^{-\text{大负数}} \to \infty$，准确率跃升到 90%。这就是涌现的数学模拟——在阈值附近有一个陡峭的跃迁。

#### 2.3 模拟的六类任务

| 任务 | 涌现？ | 涌现阈值 | 表现特征 |
|------|--------|---------|---------|
| 3 位数加减法 | 是 | ~8B | 8B 前随机，8B 后 ~90% |
| 多语言翻译 | 是 | ~10B | 训练数据中无平行语料 |
| Chain-of-Thought (CoT) | 是 | ~60B | 能"一步步思考" |
| 指令遵循 | 是 | ~30B | 理解并执行自然语言指令 |
| 情感分析 | 否 | — | 平滑增长，小模型也能做 |
| 词性标注 | 否 | — | 平滑增长 |

**涌现 vs 非涌现的本质区别**：非涌现任务（如情感分析）的准确率从小模型到大模型一直是平滑上升的——因为情感极性判断所需的基础语言能力在小模型中已存在，规模增大只是提升了精确度。而涌现任务涉及**技能的组合**（如多语言翻译 = 语言理解 + 生成 + 跨语言对齐），小模型中这些子技能不足以组合成一个新能力，直到模型足够大时才能"涌现"出来。

---

### 第3步：DPO — 直接偏好优化

#### 3.1 为什么需要 DPO？

RLHF（Reinforcement Learning from Human Feedback）是让 LLM 与人类偏好对齐的标准方法，但它需要：
1. 单独训练一个奖励模型
2. 用 PPO 做强化学习优化（训练不稳定，超参数敏感）
3. 整个过程需要四个模型同时运行（策略模型、参考模型、奖励模型、价值模型）

DPO（Rafailov et al., 2023）提出了一种更简洁的方案：**直接从偏好数据优化策略，不需要独立的奖励模型**。

#### 3.2 DPO 损失函数

DPO 利用了一个关键的数学洞察："语言模型本身隐含地就是一个奖励模型"。由此推导出的损失函数为：

$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)} \right) \right]
$$

化简后的实现：

```python
def dpo_loss(pi_logps_chosen, pi_logps_rejected,
             ref_logps_chosen, ref_logps_rejected, beta=0.1):
    # 策略模型：好回答 log P - 差回答 log P
    pi_diff = pi_logps_chosen - pi_logps_rejected
    # 参考模型：好回答 log P - 差回答 log P
    ref_diff = ref_logps_chosen - ref_logps_rejected
    # DPO 的隐式奖励：beta * (策略模型的偏好差异 - 参考模型的偏好差异)
    logits = beta * (pi_diff - ref_diff)
    # 二分类交叉熵损失：-log σ(logits)
    loss = -F.logsigmoid(logits).mean()
    return loss
```

**逐行解释**：

1. **`pi_diff = pi_logps_chosen - pi_logps_rejected`**：策略模型 $\pi_\theta$ 对好回答的对数概率减去对差回答的对数概率。如果策略模型正确地更偏好好的回答，这个值应该是正的（好回答的 log 概率更高）。

2. **`ref_diff = ref_logps_chosen - ref_logps_rejected`**：参考模型 $\pi_{\text{ref}}$（通常是 SFT 模型，训练时冻结）的对应差值。这提供了一个基准线。

3. **`logits = beta * (pi_diff - ref_diff)`**：策略模型相对于参考模型的偏好改善程度。$\beta$ 是 KL 惩罚系数——控制策略模型可以偏离参考模型多远。$\beta$ 越大，偏离越自由；$\beta$ 越小，策略模型越接近参考模型。

4. **`-F.logsigmoid(logits).mean()`**：标准二分类交叉熵损失。`logsigmoid` 等价于 `log(σ(x))`。因为 $\log\sigma(x)$ 本身是负的（$\sigma(x) \in (0,1)$），取负号后损失为正。

**数值示例**：代码展示了两种场景的 DPO 损失对比：

| 场景 | 策略模型表现 | DPO 损失 |
|------|-----------|---------|
| 正确偏好 | 好回答 log P=-2, 差回答 log P=-5 | 较小（模型已学会偏好） |
| 错误偏好 | 好/差回答概率接近 | 较大（模型未区分优劣） |

#### 3.3 模拟 DPO 训练

代码模拟了 50 个偏好对上 DPO 的训练过程：
- 随着训练进行，策略模型越来越好地学会区分好回答和差回答
- DPO 损失从高值逐渐下降到低值
- 这与真实 DPO 训练的行为一致

---

### 第4步：LoRA — 低秩适配

#### 4.1 LoRA 的核心思想

全参数微调一个 175B 的模型需要数百 GB 显存。LoRA（Hu et al., 2021）的核心洞见是：**模型适应新任务时，权重的更新矩阵 $\Delta W$ 是低秩的**。因此，不需要学习完整的 $\Delta W$，只需学习它的低秩分解：

$$
h = Wx + \Delta W x = Wx + BAx
$$

其中：
- $W \in \mathbb{R}^{d \times k}$：原始权重（冻结，不参与训练）
- $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$：可训练的低秩矩阵
- $r \ll \min(d, k)$：秩，通常 8-64

#### 4.2 LoRA 实现

```python
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=16.0):
        # 原始权重：冻结，不参与训练
        self.register_buffer('W', torch.randn(out_features, in_features) * 0.02)
        # LoRA 低秩矩阵：可训练
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.02)   # (r, in)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))         # (out, r)
        self.scaling = alpha / r

    def forward(self, x):
        original = x @ self.W.T                       # 原始路径（冻结）
        lora_out = x @ self.lora_A.T @ self.lora_B.T  # LoRA 路径: x → A → B
        return original + self.scaling * lora_out      # Wx + (α/r) BAx
```

**关键设计分析**：

1. **`self.lora_B` 初始化为零**：开始时 $\Delta W = B \times 0 = 0$，LoRA 路径输出全零，模型行为完全等同于原始模型。这保证了微调从预训练模型的原始性能开始。

2. **缩放因子 `alpha / r`**：$\alpha$ 控制 LoRA 更新的幅度。通常 $\alpha = 16$（当 $r=8$ 时缩放为 2）。缩放因子越大，LoRA 更新的影响力越大。

3. **`register_buffer` 存储原始权重**：buffer 不会被 optimizer 追踪（不参与梯度计算），确保原始权重在训练过程中保持冻结。

4. **参数效率**：对于一个 $4096 \times 4096$ 的全连接层：
   - 全参数训练：$16,777,216$ 参数
   - LoRA ($r=16$)：$2 \times 4096 \times 16 = 131,072$ 参数
   - 减少 $128\times$！

---

## 关键概念速查表

| 概念 | 公式/描述 | 一句话 |
|------|----------|--------|
| Kaplan Scaling Law | $L(N, D) = a/N^{\alpha} + b/D^{\beta} + c$ | 损失随参数/数据量幂律下降 |
| Chinchilla 最优 | $D_{\text{opt}} \approx 20N$ | 数据和参数需同步增长 |
| 不可约减损失 $c$ | 数据固有的最小损失 | 无论模型多大都无法消除 |
| 涌现 | Sigmoid 相位转变 | 越过阈值后能力突然跃升 |
| RLHF | SFT → Reward Model → PPO | 三段式对齐 pipeline |
| DPO | $-\log\sigma(\beta\Delta\log P)$ | 从偏好数据直接优化，无需奖励模型 |
| DPO 的 $\beta$ | KL 惩罚系数 | 控制偏离参考模型的程度 |
| LoRA | $h = Wx + \frac{\alpha}{r}BAx$ | 低秩适配，参数减少 100-1000x |
| LoRA 秩 $r$ | 通常 8-64 | 越小参数越少，但可能欠拟合 |
| LoRA 目标模块 | q_proj, v_proj, k_proj, o_proj | Qwen/Llama 系列通常在注意力投影上加 LoRA |

---

## 完整代码

<<< @/applied/nlp/llm/code/demo.py
