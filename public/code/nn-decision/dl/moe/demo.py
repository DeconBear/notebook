# -*- coding: utf-8 -*-
"""
=== 混合专家 MoE 最小演示 ===
合成多簇 2D 分类：Router + Top-2 线性专家，对比有/无负载均衡时的专家使用率。
运行: python demo.py
"""
import os
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)
np.random.seed(42)

N_EXPERTS = 4
TOP_K = 2
HIDDEN = 8
LR = 0.08
STEPS = 400
AUX_COEF = 0.05


def make_data(n_per=80):
    """四个高斯簇，标签 0/1 交替，鼓励不同专家负责不同区域。"""
    centers = np.array([[-1.5, -1.2], [1.6, -1.0], [-1.4, 1.5], [1.5, 1.4]])
    xs, ys = [], []
    for i, c in enumerate(centers):
        xs.append(c + 0.35 * np.random.randn(n_per, 2))
        ys.append(np.full(n_per, i % 2))
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0).astype(float)
    # 打乱
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def softmax(logits, axis=-1):
    z = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


class TinyMoE:
    """线性路由器 + N 个线性专家（输出标量 logit）。"""

    def __init__(self, n_experts=N_EXPERTS, in_dim=2):
        self.n = n_experts
        scale = 0.5
        self.W_r = np.random.randn(in_dim, n_experts) * scale
        self.b_r = np.zeros(n_experts)
        self.W_e = np.random.randn(n_experts, in_dim) * scale
        self.b_e = np.zeros(n_experts)

    def route(self, X):
        logits = X @ self.W_r + self.b_r
        probs = softmax(logits, axis=1)
        # Top-k 索引
        top_idx = np.argsort(probs, axis=1)[:, -TOP_K:]
        # 取出对应概率并重归一化
        rows = np.arange(len(X))[:, None]
        top_p = probs[rows, top_idx]
        top_p = top_p / (top_p.sum(axis=1, keepdims=True) + 1e-9)
        return probs, top_idx, top_p

    def expert_out(self, X):
        # (batch, n_experts)
        return X @ self.W_e.T + self.b_e

    def forward(self, X):
        probs, top_idx, top_p = self.route(X)
        eo = self.expert_out(X)
        rows = np.arange(len(X))[:, None]
        chosen = eo[rows, top_idx]  # (B, k)
        y_hat = (chosen * top_p).sum(axis=1)
        return y_hat, probs, top_idx, top_p

    def load_balance_loss(self, probs, top_idx):
        # f_i: 被选中频率；P_i: 平均路由概率
        B = len(probs)
        f = np.zeros(self.n)
        for k in range(TOP_K):
            for i in top_idx[:, k]:
                f[i] += 1.0
        f = f / (B * TOP_K)
        P = probs.mean(axis=0)
        return float(self.n * np.sum(f * P))

    def train_step(self, X, y, use_aux=True):
        y_hat, probs, top_idx, top_p = self.forward(X)
        # 二元交叉熵
        p = sigmoid(y_hat)
        eps = 1e-9
        loss = float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
        aux = self.load_balance_loss(probs, top_idx) if use_aux else 0.0
        total = loss + (AUX_COEF * aux if use_aux else 0.0)

        # 误差对 logit
        dlogit = (p - y) / len(y)  # (B,)

        # 专家梯度：只更新被选中的
        eo = self.expert_out(X)
        dW_e = np.zeros_like(self.W_e)
        db_e = np.zeros_like(self.b_e)
        for b in range(len(X)):
            for j in range(TOP_K):
                e = top_idx[b, j]
                w = top_p[b, j]
                dW_e[e] += dlogit[b] * w * X[b]
                db_e[e] += dlogit[b] * w

        # 路由器：对 top-k 概率用直通近似——把 dlogit * expert_out 当作对门控的信号
        # 简化：用「被选专家输出」与平均输出的差来推路由
        d_probs = np.zeros_like(probs)
        for b in range(len(X)):
            for j in range(TOP_K):
                e = top_idx[b, j]
                d_probs[b, e] += dlogit[b] * eo[b, e]
        # Softmax 雅可比的粗糙近似：d_logits ≈ d_probs - sum(d_probs*probs)
        d_logits = d_probs - (d_probs * probs).sum(axis=1, keepdims=True) * probs
        if use_aux:
            # 轻推路由概率更均匀：对 P 的梯度回传到 batch 平均
            P = probs.mean(axis=0)
            f = np.zeros(self.n)
            for k in range(TOP_K):
                for i in top_idx[:, k]:
                    f[i] += 1.0
            f = f / (len(X) * TOP_K)
            dP = AUX_COEF * self.n * f / len(X)
            d_logits += dP  # 广播到每个样本

        self.W_e -= LR * dW_e
        self.b_e -= LR * db_e
        self.W_r -= LR * (X.T @ d_logits)
        self.b_r -= LR * d_logits.mean(axis=0)
        return total, loss, aux, probs, top_idx


def expert_usage(top_idx, n_experts):
    counts = np.zeros(n_experts)
    for k in range(top_idx.shape[1]):
        for i in top_idx[:, k]:
            counts[i] += 1
    return counts / counts.sum()


def train(use_aux, X, y):
    model = TinyMoE()
    hist = []
    for t in range(STEPS):
        total, loss, aux, probs, top_idx = model.train_step(X, y, use_aux=use_aux)
        if t % 50 == 0 or t == STEPS - 1:
            usage = expert_usage(top_idx, model.n)
            hist.append((t, total, loss, aux, usage.copy()))
            print(f"aux={use_aux} step={t:3d} loss={loss:.4f} aux={aux:.4f} usage={np.round(usage, 2)}")
    return model, hist


def main():
    print('=== MoE 玩具：Top-2 + 负载均衡 ===')
    X, y = make_data()
    model_bal, hist_bal = train(True, X, y)
    model_raw, hist_raw = train(False, X, y)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    # 负载对比（最后一步）
    u_bal = hist_bal[-1][4]
    u_raw = hist_raw[-1][4]
    xs = np.arange(N_EXPERTS)
    axes[0].bar(xs - 0.2, u_raw, width=0.4, label='无均衡', color='#E8684A')
    axes[0].bar(xs + 0.2, u_bal, width=0.4, label='有均衡', color='#5AD8A6')
    axes[0].set_xticks(xs)
    axes[0].set_xlabel('专家编号')
    axes[0].set_ylabel('被选中比例')
    axes[0].set_title('专家负载')
    axes[0].legend()

    # 损失曲线
    axes[1].plot([h[0] for h in hist_raw], [h[2] for h in hist_raw], label='无均衡 CE')
    axes[1].plot([h[0] for h in hist_bal], [h[2] for h in hist_bal], label='有均衡 CE')
    axes[1].set_xlabel('step')
    axes[1].set_title('分类损失')
    axes[1].legend()

    # 决策边界（有均衡模型）
    xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    logits, _, _, _ = model_bal.forward(grid)
    zz = sigmoid(logits).reshape(xx.shape)
    axes[2].contourf(xx, yy, zz, levels=20, cmap='RdBu_r', alpha=0.7)
    axes[2].scatter(X[y == 0, 0], X[y == 0, 1], s=10, c='#5B8FF9', label='类0')
    axes[2].scatter(X[y == 1, 0], X[y == 1, 1], s=10, c='#E8684A', label='类1')
    axes[2].set_title('MoE（有均衡）决策')
    axes[2].legend(markerscale=2)
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'moe_toy_results.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)


if __name__ == '__main__':
    main()
