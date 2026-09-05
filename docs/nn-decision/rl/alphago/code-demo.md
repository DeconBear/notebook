---
title: "AlphaGo：自我对弈与 MCTS — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# AlphaGo：自我对弈与 MCTS — demo.py 代码详解

<a href="/notebook/code/nn-decision/rl/alphago/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/nn-decision/rl/alphago/code
python demo.py
```

CPU 即可。井字棋上的 PUCT-MCTS：**不训练**策略/价值网络，先验均匀、叶子随机滚出。先画四张正文示意图，再打印空盘 400 次模拟的根访问 $N$，以及执 X 对「随机 / 中心贪心」各 80 局。要对照「同一套树循环、但叶子在学出来的模型里走」，见 [MuZero code-demo](/world-models/abstract/muzero/code-demo)——那章 demo 没有完整 `select/expand/backup`，树的代码以本页为准。

## 代码逐段详解

### 第1步：棋盘约定与胜负

```python
EMPTY, X, O = 0, 1, -1
N_SIM = 200
C_PUCT = 1.5
```

格子 9 个，一维列表。`X=1`、`O=-1`，三人成线时和为 $\pm 3$。`C_PUCT` 是探索系数 $c_{\mathrm{puct}}$。

```python
def winner(b):
    for i, j, k in lines:
        s = b[i] + b[j] + b[k]
        if s == 3: return X
        if s == -3: return O
    if all(v != EMPTY for v in b):
        return 0
    return None
```

- **返回 `None`**：未结束，还可以下。
- **返回 `0`**：平局（棋盘满且没人三连）。和「空」要分开，后面 `if w is None` 才扩展/滚出。
- **`legal`**：`enumerate` 出值为 `EMPTY` 的下标。**`copy_play`**：`list(b)` 浅拷贝再改一格——树的每个节点必须有独立棋盘，原地改会把祖先也改脏。

`np.random.seed` 管 Dirichlet；`random.seed` 管 `random.choice` 滚出。两套都要钉。

---

### 第2步：`Node` — $N$、$W$、$Q$、$P$

```python
class Node:
    def __init__(self, board, player, parent=None, action=None, prior=1.0):
        self.n = 0      # 访问次数 N
        self.w = 0.0    # 累计价值 W
        self.prior = prior
        self.children = {}
        self.expanded = False
```

| 符号 | 代码 | 含义 |
|------|------|------|
| $N(s,a)$ | `child.n` | 这条边被走过多少次 |
| $W$ | `child.w` | 累计回报（从**走进该节点的棋手**看） |
| $Q=W/N$ | `q()` | 平均价值 |
| $P(s,a)$ | `prior` | 先验；本章均匀，真 AlphaGo 来自策略网 |

```python
def q(self):
    return 0.0 if self.n == 0 else self.w / self.n

def puct(self, c):
    parent_n = self.parent.n if self.parent else 1
    u = c * self.prior * math.sqrt(parent_n) / (1 + self.n)
    return self.q() + u
```

$$
\mathrm{PUCT}=Q+c_{\mathrm{puct}}\,P\,\frac{\sqrt{\sum_b N(s,b)}}{1+N(s,a)}
$$

- **$Q$**：利用——平均打分高的边优先。
- **$U$**：探索——先验 $P$ 大、自己还没被走（$N$ 小）、父亲已经很热（$\sqrt{N_{\mathrm{parent}}}$）时加分。分母 `1+n` 避免除零，也让从未访问的边 $U$ 最大。
- **根节点没有父**：`parent_n` 退回 1，根自己不算 PUCT（选择从根的**孩子**开始）。

`children` 是 `dict`：键是动作 0–8，值是子 `Node`。

---

### 第3步：`expand` — 写入均匀先验

```python
def expand(node):
    acts = legal(node.board)
    if not acts:
        return
    prior = 1.0 / len(acts)
    for a in acts:
        child_board = copy_play(node.board, a, node.player)
        node.children[a] = Node(child_board, -node.player, node, a, prior)
    node.expanded = True
```

每个合法动作建一个孩子：棋盘已经下了这一手，**轮到对方**（`player` 取负）。先验均分 $1/|\mathcal{A}|$——没有 $p_\sigma$。终局没有合法手，不扩展。

真 AlphaGo 这里会写入网络的 $P(s,a)$；Zero 还会把 $v_\theta(s)$ 当叶子评估。本章用滚出代替 $v_\theta$。

---

### 第4步：`select` — 沿 PUCT 走到叶子

```python
def select(node, c):
    while node.expanded and node.children:
        node = max(node.children.values(), key=lambda ch: ch.puct(c))
    return node
```

- **`while expanded and children`**：只要还能往下走，就在孩子里取 PUCT 最大的。撞上未扩展节点（或没有孩子）就停，这就是本轮叶子。
- **`max(..., key=lambda ch: ch.puct(c))`**：`key` 告诉 `max` 用什么当比较键。`lambda ch: ...` 是内联小函数，参数 `ch` 是每一个子节点。
- 选择**不改**棋盘，只在已有树上走指针。

---

### 第5步：`rollout` — 随机走完，返回根视角的 $z$

```python
def rollout(b, to_move, root_player):
    player = to_move
    board = list(b)
    while True:
        w = winner(board)
        if w is not None:
            if w == 0:
                return 0.0
            return 1.0 if w == root_player else -1.0
        board = copy_play(board, random.choice(legal(board)), player)
        player = -player
```

从叶子的 `to_move` 开始，均匀随机落到终局。$z\in\{+1,0,-1\}$ **一律从根玩家看**：根赢 +1，根输 −1。所以同一终局，黑先手根和白先手根会得到相反的 $z$。

井字棋状态极少，随机滚出方差还能忍。围棋上 AlphaGo 才用价值网缩短滚出；本 demo 故意不训网，好把树循环看清。

---

### 第6步：`backup` — 路径上累加 $N$、$W$

```python
def backup(node, z_root, root_player):
    while node is not None:
        node.n += 1
        if node.parent is None:
            node.w += z_root
        else:
            mover = node.parent.player
            node.w += z_root if mover == root_player else -z_root
        node = node.parent
```

从叶子一路 `parent` 爬到根。每经过一个节点 $N\mathrel{+}=1$。

**为什么 $W$ 要按「走进这个节点的棋手」存？** 选择时父节点对孩子做 `argmax Q`。父节点的 `player` 是**刚要下棋的人**，孩子是他走完之后的局面。所以孩子的 $Q$ 必须是「这个下棋人」的价值：根玩家走的边累加 $+z_{\mathrm{root}}$，对手走的边累加 $-z_{\mathrm{root}}$。根没有父，直接加 $z_{\mathrm{root}}$。

若全程只加 $z_{\mathrm{root}}$，对手节点的 $Q$ 会变成「根有多开心」——父节点（对手）就会专挑让根开心的边，树就反了。

---

### 第7步：`mcts_move` — 一棵树、若干次模拟、按 $N$ 落子

```python
root = Node(board, player)
expand(root)
noise = np.random.dirichlet([0.3] * len(acts))
ch.prior = 0.75 * ch.prior + 0.25 * float(n)
```

根上混 Dirichlet 噪声（Zero 也这么干）：避免第一次模拟把 $N$ 锁死在同一手。`[0.3]*len(acts)` 浓度小 → 噪声尖，探索更野。`0.75/0.25` 是先验与噪声的混合比。

```python
for _ in range(n_sim):
    leaf = select(root, c)
    w = winner(leaf.board)
    if w is None:
        if not leaf.expanded:
            expand(leaf)
        z = rollout(leaf.board, leaf.player, root.player)
        backup(leaf, z, root.player)
    else:
        z = 0.0 if w == 0 else (1.0 if w == root.player else -1.0)
        backup(leaf, z, root.player)
```

一轮 = 选择 →（未终局则扩展）→ 评估 → 回传。已终局的叶子不再滚出，直接从棋面算 $z$。评估在**真实棋盘**上随机走完——这是和 MuZero 的分水岭：MuZero 用学到的动力学 $g$ 在潜空间里走，叶子用 $v$ 而不是 `rollout`。本 demo 没有 $h/g/f$，所以必须有真规则和真滚出。

```python
visits = {a: ch.n for a, ch in root.children.items()}
best = max(visits, key=visits.get)
```

对局按根的访问次数 $N$ 落子，**不是** $\arg\max Q$，也不是网络瞬时先验。搜得越多的边越可信。`max(dict, key=dict.get)`：按**值**（次数）取最大键（动作）。

---

### 第8步：对手、对局、主函数

`random_move`：合法手里 `random.choice`。`greedy_center`：手写优先级，中心 4 最先，再角再边——弱而确定的基线，用来看 MCTS 是否强于「人类常识启发式」。

`play_game(x_policy, o_policy)`：双方都是 `policy(board, player) -> action`。`mcts_policy` 丢掉 visits，只返回 `best`。

`eval_match`：MCTS 执 X，对随机 / 中心贪心各 `N_GAMES` 局。`{X:0, 0:1, O:2}[w]` 把胜平负映射到长度为 3 的计数器。

`main`：空盘 `n_sim=400` 打一次根 $N$（中心格通常最高），再 `eval_match`，画 `mcts_tic_tac_toe.png`。`draw_three_networks` 等四张是正文「三件套 / 自我对弈 / MCTS 循环 / 先验炼成 $N$」——demo **没有**实现 SL/RL 网络训练，只实现了右边那棵树。

---

### 关键概念速查表

| 概念 | 数学 / 直觉 | 代码 |
|------|-------------|------|
| $Q$ | $W/N$，未访问为 0 | `Node.q` |
| PUCT | $Q+c P\sqrt{N}/(1+N)$ | `Node.puct` |
| 选择 | 沿 PUCT 走到未扩展叶子 | `select` |
| 扩展 | 合法手各建子节点，均分 $P$ | `expand` |
| 滚出 | 随机走到终局，$z$ 从根看 | `rollout` |
| 回传 | 路径 $N{+}=1$，$W$ 按下棋人变号 | `backup` |
| 落子 | $\arg\max_a N(\mathrm{root},a)$ | `max(visits, key=visits.get)` |
| Dirichlet | 根先验掺噪声 | `0.75*P + 0.25*noise` |
| vs MuZero | 真棋盘滚出 vs 潜空间 $g$+$v$ | 本页树循环 / MuZero 章框图 |

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/nn-decision/rl/alphago/code/demo.py`
