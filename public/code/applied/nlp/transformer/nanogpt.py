# -*- coding: utf-8 -*-
"""
nanoGPT 风格 GPT-2 训练与生成
================================
基于 Andrej Karpathy 的 nanoGPT 思路，从零实现一个 GPT-2 级别 Transformer，
在莎士比亚文本上训练并生成文本。

核心特点：
  - 完整的 GPT-2 架构：LayerNorm、因果自注意力、FFN、残差连接
  - CPU 友好：默认超参数可在 CPU 上 15 分钟内完成训练
  - 无需外部数据：内置莎士比亚剧本片段
  - 可选：放入 input.txt 使用完整数据集

运行方式：
  python nanogpt.py             # 使用内置语料训练
  python nanogpt.py --gpu       # 使用 GPU 训练（如可用）
  python nanogpt.py --generate  # 仅加载已有模型生成文本

参考：https://github.com/karpathy/nanoGPT
"""

import os
import math
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ============================================================
# 设备检测
# ============================================================
parser = argparse.ArgumentParser(description='nanoGPT: train a GPT on CPU')
parser.add_argument('--gpu', action='store_true', help='Force GPU usage')
parser.add_argument('--generate', action='store_true', help='Only generate text')
args = parser.parse_args()

if args.gpu and torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif args.gpu and torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

print(f"Device: {DEVICE}")
print("(Use --gpu to force GPU training if available)")

# ============================================================
# 内置语料（莎士比亚作品片段 + 英文文本）
# ============================================================
DEFAULT_TEXT = r"""
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
would yield us but the superfluity, while it were
wholesome, we might guess they relieved us humanely;
but they think we are too dear: the leanness that
afflicts us, the object of our misery, is as an
inventory to particularise their abundance; our
sufferance is a gain to them Let us revenge this with
our pikes, ere we become rakes: for the gods know I
speak this in hunger for bread, not in thirst for revenge.

Second Citizen:
Would you proceed especially against Caius Marcius?

All:
Against him first: he's a very dog to the commonalty.

Second Citizen:
Consider you what services he has done for his country?

First Citizen:
Very well; and could be content to give him good
report for't, but that he pays himself with being proud.

Second Citizen:
Nay, but speak not maliciously.

First Citizen:
I say unto you, what he hath done famously, he did
it to that end: though soft-conscienced men can be
content to say it was for his country, he did it to
please his mother and to be partly proud; which he
is, even unto the altitude of his virtue.

Second Citizen:
What he cannot help in his nature, you account a
vice in him. You shall not put him to a partial muster,
wherein you see not the bottom of his revenues.

First Citizen:
No more of him; I take him to be a worthy man.

MENENIUS:
What work's, my countrymen, in hand? Where go you
With bats and clubs? The matter? Speak, I pray you.

First Citizen:
Our business is not unknown to the senate; they have
had inkling this fortnight what we intend to do,
which now we'll show 'em in deeds. They say poor
suitors have strong breaths: they shall know we
have strong arms too.

MENENIUS:
Why, masters, my good friends, mine honest neighbours,
Will you undo yourselves?

First Citizen:
We cannot, sir; we are undone already.

MENENIUS:
I tell you, friends, most charitable care
Have the patricians of you. For your wants,
Your suffering in this dearth, you may as well
Strike at the heaven with your staves as lift them
Against the Roman state, whose course will on
The way it takes, cracking ten thousand curbs
Of more strong link asunder than can ever
Appear in your impediment. For the dearth,
The gods, not the patricians, make it, and
Your knees to them, not arms, must help. Alack,
You are transported by calamity
Thither where more attends you, and you slander
The helms o' the state, who care for you like fathers,
When you curse them as enemies.

First Citizen:
Care for us! True, indeed! They ne'er cared for us
yet: suffer us to famish, and their store-houses
crammed with grain; make edicts for usury, to
support usurers; repeal daily any wholesome act
established against the rich, and provide more
piercing statutes daily, to chain up and restrain
the poor. If the wars eat us not up, they will;
and there's all the love they bear us.

MENENIUS:
Either you must confess yourselves wondrous malicious,
Or be accused of folly. I shall tell you
A pretty tale: it may be you have heard it;
But, since it serves my purpose, I will venture
To stale't a little more.

First Citizen:
Well, I'll hear it, sir:
Yet you must not think to fob off our disgrace
With a tale: but, an't please you, deliver.

MENENIUS:
There was a time when all the body's members
Rebell'd against the belly, thus accused it:
That only like a gulf it did remain
I' the midst o' the body, idle and unactive,
Still cupboarding the viand, never bearing
Like labour with the rest, where the other instruments
Did see and hear, devise, instruct, walk, feel,
And, mutually participate, did minister
Unto the appetite and affection common
Of the whole body. The belly answer'd--

First Citizen:
Well, sir, what answer made the belly?

MENENIUS:
Sir, I shall tell you. With a kind of smile,
Which ne'er came from the lungs, but even thus--
For, look you, I may make the belly smile
As well as speak--it tauntingly replied
To the discontented members, the mutinous parts
That envied his receipt; even so most fitly
As you malign our senators for that
They are not such as you.

First Citizen:
Your belly's answer? What!
The kingly-crowned head, the vigilant eye,
The counsellor heart, the arm our soldier,
Our steed the leg, the tongue our trumpeter.
With other muniments and petty helps
In this our fabric, if that they--

MENENIUS:
What then?
'Fore me, this fellow speaks! What then? what then?

First Citizen:
Should by the cormorant belly be restrain'd,
Who is the sink o' the body,--

MENENIUS:
Well, what then?

First Citizen:
The former agents, if they did complain,
What could the belly answer?

MENENIUS:
I will tell you;
If you'll bestow a small--of what you have little--
Patience awhile, you'll hear the belly's answer.

First Citizen:
You're long about it.

MENENIUS:
Note me this, good friend;
Your most grave belly was deliberate,
Not rash like his accusers, and thus answer'd:
'True is it, my incorporate friends,' quoth he,
'That I receive the general food at first,
Which you do live upon; and fit it is,
Because I am the store-house and the shop
Of the whole body: but, if you do remember,
I send it through the rivers of your blood,
Even to the court, the heart, to the seat o' the brain;
And, through the cranks and offices of man,
The strongest nerves and small inferior veins
From me receive that natural competency
Whereby they live: and though that all at once,
You, my good friends,'--this says the belly, mark me,--

First Citizen:
Ay, sir; well, well.

MENENIUS:
'Though all at once cannot
See what I do deliver out to each,
Yet I can make my audit up, that all
From me do back receive the flour of all,
And leave me but the bran.' What say you to't?

First Citizen:
It was an answer: how apply you this?

MENENIUS:
The senators of Rome are this good belly,
And you the mutinous members; for examine
Their counsels and their cares, digest things rightly
Touching the weal o' the common, you shall find
No public benefit which you receive
But it proceeds or comes from them to you
And no way from yourselves. What do you think,
You, the great toe of this assembly?

First Citizen:
I the great toe! Why the great toe?

MENENIUS:
For that, being one o' the lowest, basest, poorest,
Of this most wise rebellion, thou go'st foremost:
Thou rascal, that art worst in blood to run,
Lead'st first to win some vantage.
But make you ready your stiff bats and clubs:
Rome and her rats are at the point of battle;
The one side must have bale.
"""

# 额外补充一些文本
DEFAULT_TEXT += """
HAMLET:
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die: to sleep;
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to, 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep: perchance to dream: ay, there's the rub;
For in that sleep of death what dreams may come
When we have shuffled off this mortal coil,
Must give us pause: there's the respect
That makes calamity of so long life;

KING CLAUDIUS:
O, my offence is rank it smells to heaven;
It hath the primal eldest curse upon't,
A brother's murder. Pray can I not,
Though inclination be as sharp as will:
My stronger guilt defeats my strong intent;
And, like a man to double business bound,
I stand in pause where I shall first begin,
And both neglect. What if this cursed hand
Were thicker than itself with brother's blood,
Is there not rain enough in the sweet heavens
To wash it white as snow? Whereto serves mercy
But to confront the visage of offence?
And what's in prayer but this two-fold force,
To be forestalled ere we come to fall,
Or pardon'd being down? Then I'll look up;
My fault is past. But, O, what form of prayer
Can serve my turn? 'Forgive me my foul murder?'
"""

# ============================================================
# GPT 模型定义
# ============================================================

class LayerNorm(nn.Module):
    """LayerNorm 但带有可学习的 bias（匹配 GPT-2 风格）"""
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """
    因果自注意力（带 mask，防止看到未来 token）
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # Q、K、V 的线性投影（合并为一个矩阵以提高效率）
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # 输出投影
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # 因果 mask（下三角矩阵），注册为 buffer 随模型移动设备
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # batch, sequence length, embedding dim

        # 一次性计算 Q、K、V
        qkv = self.c_attn(x)  # (B, T, 3*C)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # 拆分为多头：(B, T, C) -> (B, n_head, T, head_dim)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # 缩放点积注意力
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = F.dropout(att, p=self.dropout, training=self.training)
        y = att @ v

        # 合并多头
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    """GPT-2 的 FFN：两层线性 + GELU"""
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu   = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = config.dropout

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class Block(nn.Module):
    """一个 Transformer Block：LN -> Attention -> residual -> LN -> MLP -> residual"""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTConfig:
    """GPT 模型配置"""
    def __init__(self, vocab_size=50304, block_size=256, n_layer=6,
                 n_head=6, n_embd=384, dropout=0.0, bias=True):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias


class GPT(nn.Module):
    """完整的 GPT-2 风格语言模型"""
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token embedding + Position embedding
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 权重共享：token embedding 和输出层共享权重（减少参数量）
        self.transformer.wte.weight = self.lm_head.weight

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """GPT-2 风格的权重初始化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size, \
            f"Cannot forward sequence of length {T}, block size is {self.config.block_size}"

        # Token embeddings + Position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        # 通过所有 Transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # 最后的 LayerNorm + 输出头
        x = self.transformer.ln_f(x)

        if targets is not None:
            # 训练模式：计算交叉熵损失
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-1)
        else:
            # 推理模式：只需要最后一个位置的 logits
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        自回归生成文本。
        idx: (B, T) 的上下文 token 序列
        max_new_tokens: 生成的最大 token 数
        temperature: 温度参数（>1 更随机，<1 更确定）
        top_k: 只从概率最高的 k 个 token 中采样
        """
        for _ in range(max_new_tokens):
            # 如果序列太长，截取最后 block_size 个 token
            idx_cond = idx if idx.size(1) <= self.config.block_size \
                       else idx[:, -self.config.block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # 温度缩放

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def configure_optimizers(self, learning_rate, weight_decay, device_type):
        """
        配置优化器，区分 weight decay 参数和非 weight decay 参数。
        这是 GPT-2/3 的标准做法：对于 bias 和 LayerNorm 的参数不应用 weight decay。
        """
        # 分离需要 weight decay 的参数（权重矩阵）和不需要的（bias、LayerNorm）
        decay_params = []
        no_decay_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                if len(param.shape) >= 2 and 'ln' not in name and 'bias' not in name:
                    decay_params.append(param)
                else:
                    no_decay_params.append(param)

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]

        # 使用 AdamW（Adam + 正确的 weight decay）
        fused_available = 'fused' in [k for k in dir(torch.optim.AdamW)
                                      if 'fused' in k]  # noqa
        use_fused = fused_available and device_type == 'cuda'
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95),
                                       fused=use_fused)
        return optimizer

    @property
    def device(self):
        return next(self.parameters()).device


# ============================================================
# 数据处理
# ============================================================

def prepare_data(text, train_split=0.9):
    """从文本构建 token<->id 映射，并划分为训练/验证集"""
    # 构建字符级词汇表
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"Vocabulary size: {vocab_size} unique characters")

    # 编码映射
    stoi = {ch: i for i, ch in enumerate(chars)}  # string -> int
    itos = {i: ch for i, ch in enumerate(chars)}  # int -> string

    # 编码整个文本
    data = np.array([stoi[c] for c in text], dtype=np.int64)

    # 划分训练/验证集
    n = int(len(data) * train_split)
    train_data = data[:n]
    val_data = data[n:]
    print(f"Train tokens: {len(train_data):,}, Val tokens: {len(val_data):,}")

    return train_data, val_data, stoi, itos, vocab_size


def get_batch(train_data, val_data, batch_size, block_size, split='train'):
    """获取一个 mini-batch"""
    data = train_data if split == 'train' else val_data
    # 随机选取 batch 个起始位置
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64))
                     for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64))
                     for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y


# ============================================================
# 训练与评估
# ============================================================

@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, eval_iters=50):
    """在训练集和验证集上估计当前 loss"""
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(train_data, val_data, batch_size, block_size, split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(model, train_data, val_data, config):
    """训练 GPT 模型"""
    print(f"\n{'='*60}")
    print(f"Training GPT")
    print(f"{'='*60}")
    print(f"Layers: {config.n_layer}, Heads: {config.n_head}, "
          f"Embed dim: {config.n_embd}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {DEVICE}")

    # 配置优化器
    # CPU 模式：更小的 batch、更短训练
    if DEVICE.type == 'cpu':
        batch_size = 16
        max_iters = 500
        learning_rate = 3e-4
        eval_interval = 100
        print("CPU mode: reduced training (500 iters). Use --gpu for full training.")
    else:
        batch_size = 64
        max_iters = 5000
        learning_rate = 1e-3
        eval_interval = 500

    optimizer = model.configure_optimizers(learning_rate, weight_decay=0.1,
                                            device_type=DEVICE.type)

    # 训练循环
    model.train()
    t0 = time.time()
    best_val_loss = float('inf')

    for step in range(max_iters):
        # 学习率预热 + 余弦衰减
        if step < max_iters * 0.1:
            lr = learning_rate * (step / (max_iters * 0.1))
        else:
            progress = (step - max_iters * 0.1) / (max_iters * 0.9)
            lr = learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 获取 batch 并训练
        X, Y = get_batch(train_data, val_data, batch_size, config.block_size, 'train')
        _, loss = model(X, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 评估和打印
        if step % eval_interval == 0 or step == max_iters - 1:
            eval_result = estimate_loss(model, train_data, val_data,
                                         batch_size, config.block_size)
            elapsed = time.time() - t0
            print(f"step {step:5d}/{max_iters} | "
                  f"train loss: {eval_result['train']:.4f} | "
                  f"val loss: {eval_result['val']:.4f} | "
                  f"lr: {lr:.2e} | "
                  f"time: {elapsed:.1f}s")

            if eval_result['val'] < best_val_loss:
                best_val_loss = eval_result['val']
                torch.save(model.state_dict(), 'nanogpt_model.pt')
                print(f"  -> saved best model (val loss: {best_val_loss:.4f})")

    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to nanogpt_model.pt")


# ============================================================
# 文本生成
# ============================================================

def generate_text(model, itos, stoi, start_text="\n", num_tokens=200,
                  temperature=0.8, top_k=40):
    """使用训练好的模型生成文本"""
    model.eval()
    # 编码起始文本
    context = torch.tensor([[stoi.get(c, 0) for c in start_text]],
                           dtype=torch.long, device=DEVICE)

    print(f"\n{'='*60}")
    print(f"Generating text (temperature={temperature}, top_k={top_k})")
    print(f"{'='*60}")
    print(f"Prompt: {repr(start_text)}")
    print("-" * 40)

    # 生成
    output = model.generate(context, max_new_tokens=num_tokens,
                             temperature=temperature, top_k=top_k)
    output_text = ''.join([itos.get(i.item(), '?') for i in output[0]])

    print(output_text)
    print("-" * 40)
    return output_text


# ============================================================
# 主函数
# ============================================================

def main():
    # 尝试读取本地 input.txt（用户可以放入自己的训练数据）
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'input.txt')
    if os.path.exists(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        if len(text) < 1000:
            print(f"input.txt too small ({len(text)} chars), using built-in corpus")
            text = DEFAULT_TEXT
        else:
            print(f"Loaded input.txt: {len(text):,} characters")
    else:
        print(f"No input.txt found, using built-in Shakespeare corpus")
        text = DEFAULT_TEXT

    # 准备数据
    train_data, val_data, stoi, itos, vocab_size = prepare_data(text)

    # 创建模型配置（CPU 友好）
    model_config = GPTConfig(
        vocab_size=vocab_size,
        block_size=128,       # 上下文长度
        n_layer=4,            # Transformer 层数
        n_head=4,             # 注意力头数
        n_embd=256,           # 嵌入维度
        dropout=0.0,          # CPU 模式下通常不需要 dropout
    )

    model = GPT(model_config).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    if args.generate:
        # 仅生成模式：加载已有模型
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'nanogpt_model.pt')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            print(f"Loaded model from {model_path}")
        else:
            print("No saved model found, training from scratch first...")
            train(model, train_data, val_data, model_config)

        # 用不同 prompt 生成
        for prompt in ["\nFirst Citizen:\n", "\nHAMLET:\n", "\nMENENIUS:\n", "\n"]:
            generate_text(model, itos, stoi, start_text=prompt,
                          num_tokens=200, temperature=0.8)
    else:
        # 训练 + 生成
        train(model, train_data, val_data, model_config)

        # 用训练好的模型生成几段文本
        generate_text(model, itos, stoi, start_text="\nFirst Citizen:\n",
                      num_tokens=300, temperature=0.8)

        generate_text(model, itos, stoi, start_text="\n",
                      num_tokens=200, temperature=1.0)

        generate_text(model, itos, stoi, start_text="\nHAMLET:\n",
                      num_tokens=200, temperature=0.7)


if __name__ == '__main__':
    main()
