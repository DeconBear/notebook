# -*- coding: utf-8 -*-
"""
=== LeWM 最小演示：MSE + 高斯正则 + 潜空间 CEM ===
二维质点；观测是带噪声的位置特征。线性编码器/预测器端到端训练
（下一步嵌入 MSE + 各向同性高斯代理正则），再用 CEM 追目标嵌入。
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

DT = 0.1
DAMP = 0.3
Z_DIM = 6
HORIZON = 10
N_SAMPLE = 64
N_ELITE = 10
CEM_ITERS = 5
LAMBDA_REG = 0.15


def true_step(pos, vel, acc):
    vel = vel + DT * (acc - DAMP * vel)
    pos = pos + DT * vel
    return pos, vel


def observe(pos):
    feat = np.array([
        pos[0], pos[1], pos[0] ** 2, pos[1] ** 2,
        np.sin(pos[0]), np.cos(pos[1]), pos[0] * pos[1], 1.0,
    ])
    return feat + 0.02 * np.random.randn(8)


def sigreg_proxy(z):
    """教学近似 SIGReg：逼近零均值、单位方差（非完整 Epps–Pulley）。"""
    mu = z.mean(axis=0)
    std = z.std(axis=0) + 1e-6
    return float(np.mean(mu ** 2) + np.mean((std - 1.0) ** 2))


class LinearLeWM:
    """enc: o->z；pred: [z,a]->z'。全部可微、用解析梯度更新。"""

    def __init__(self):
        self.We = np.random.randn(8, Z_DIM) * 0.3
        self.be = np.zeros(Z_DIM)
        self.Wp = np.random.randn(Z_DIM + 2, Z_DIM) * 0.3
        self.bp = np.zeros(Z_DIM)

    def encode(self, o):
        return o @ self.We + self.be

    def predict(self, z, a):
        x = np.concatenate([z, a], axis=-1)
        return x @ self.Wp + self.bp

    def train_step(self, o, a, no, lr=0.05):
        z = self.encode(o)
        nz_tgt = self.encode(no)
        # 预测器用「停梯度」目标嵌入，避免两项互相打架过度
        nz = nz_tgt
        hat = self.predict(z, a)
        err = hat - nz
        pred_loss = float(np.mean(err ** 2))
        reg = sigreg_proxy(np.concatenate([z, nz_tgt], axis=0))

        # pred 梯度
        x = np.concatenate([z, a], axis=-1)
        self.Wp -= lr * (x.T @ err) / len(o)
        self.bp -= lr * err.mean(axis=0)

        # enc 梯度：通过 pred 反传到 z，再加上把 z/nz 拉向标准正态
        # dL/dz ≈ err @ Wp_z.T ；正则对 z 的梯度 ≈ 2*mu/B 与对 std 的弱推
        Wp_z = self.Wp[:Z_DIM]
        gz = (err @ Wp_z.T) / len(o)
        mu = z.mean(axis=0)
        gz = gz + LAMBDA_REG * (2.0 * mu) / len(o)
        self.We -= lr * (o.T @ gz)
        self.be -= lr * gz.mean(axis=0)

        # 目标侧嵌入同样拉向零均值，避免编码器塌成常数
        mu2 = nz_tgt.mean(axis=0)
        gnz = np.broadcast_to(LAMBDA_REG * (2.0 * mu2) / len(o), nz_tgt.shape)
        self.We -= lr * (no.T @ gnz)
        self.be -= lr * gnz.mean(axis=0)

        return pred_loss + LAMBDA_REG * reg, pred_loss, reg


def collect_data(n=500):
    obs, acts, next_obs = [], [], []
    for _ in range(n):
        pos = np.random.uniform(-1.5, 1.5, size=2)
        vel = np.random.uniform(-0.5, 0.5, size=2)
        acc = np.random.uniform(-2, 2, size=2)
        o = observe(pos)
        npos, _ = true_step(pos, vel, acc)
        obs.append(o)
        acts.append(acc)
        next_obs.append(observe(npos))
    return np.array(obs), np.array(acts), np.array(next_obs)


def cem_plan(model, z0, zg):
    mu = np.zeros((HORIZON, 2))
    std = np.ones((HORIZON, 2)) * 1.5
    hist = []
    for _ in range(CEM_ITERS):
        seqs = mu + std * np.random.randn(N_SAMPLE, HORIZON, 2)
        scores = []
        for s in seqs:
            z = z0.copy()
            for a in s:
                z = model.predict(z, a)
            scores.append(-np.sum((z - zg) ** 2))
        scores = np.array(scores)
        elite = seqs[np.argsort(scores)[-N_ELITE:]]
        mu = 0.7 * elite.mean(axis=0) + 0.3 * mu
        std = 0.7 * elite.std(axis=0) + 0.3 * std + 0.05
        hist.append(mu.copy())
    return mu, hist


def main():
    print('=== LeWM 玩具：两项损失 + 潜空间 CEM ===')
    model = LinearLeWM()
    obs, acts, next_obs = collect_data()
    losses = []
    for step in range(600):
        idx = np.random.randint(0, len(obs), size=64)
        loss, pl, rg = model.train_step(obs[idx], acts[idx], next_obs[idx])
        losses.append(loss)
        if step % 150 == 0:
            print(f'step {step:4d}  loss={loss:.4f}  pred={pl:.4f}  reg={rg:.4f}')

    start = np.array([-1.0, -0.8])
    goal = np.array([1.2, 1.0])
    zg = model.encode(observe(goal))

    pos, vel = start.copy(), np.zeros(2)
    traj = [pos.copy()]
    for t in range(35):
        mu, hist = cem_plan(model, model.encode(observe(pos)), zg)
        acc = np.clip(mu[0], -3, 3)
        pos, vel = true_step(pos, vel, acc)
        traj.append(pos.copy())
        if t % 5 == 0:
            print(f't={t:02d}  pos={pos}')

    traj = np.array(traj)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(losses, lw=1)
    axes[0].set_title('训练损失（MSE + 高斯正则代理）')
    axes[0].set_xlabel('step')
    axes[1].plot(traj[:, 0], traj[:, 1], 'o-', ms=3, label='轨迹')
    axes[1].scatter(*start, c='C0', s=80, label='起点')
    axes[1].scatter(*goal, c='C1', s=80, marker='*', label='目标')
    axes[1].set_title('潜空间 CEM-MPC 闭环')
    axes[1].legend()
    axes[1].set_aspect('equal', adjustable='box')
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'lewm_cem_mpc.png')
    fig.savefig(out, dpi=140)
    print('保存', out)

    _, hist = cem_plan(model, model.encode(observe(start)), zg)
    fig2, ax = plt.subplots(figsize=(5, 3.5))
    for i, m in enumerate(hist):
        ax.plot(m[:, 0], alpha=0.3 + 0.7 * i / len(hist),
                label=f'iter {i+1}' if i in (0, len(hist) - 1) else None)
    ax.set_title('CEM：动作序列均值收敛（第 0 维）')
    ax.set_xlabel('计划内时间')
    ax.legend()
    fig2.tight_layout()
    out2 = os.path.join(_IMAGES_DIR, 'lewm_cem_iters.png')
    fig2.savefig(out2, dpi=140)
    print('保存', out2)


if __name__ == '__main__':
    main()
