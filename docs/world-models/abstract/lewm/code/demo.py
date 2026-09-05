# -*- coding: utf-8 -*-
"""
=== LeWM 演示：两项损失 + 潜空间 CEM ===
1) 二维质点：MSE + 高斯代理正则，CEM 追目标嵌入。
2) 倒立摆火柴杆像素：随机投影 SIGReg + 潜空间 CEM 对准直立嵌入。
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
IMG_SIZE = 16
Z_PIX = 8
HORIZON_P = 8
N_SAMPLE_P = 24
N_ELITE_P = 8
CEM_ITERS_P = 4


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
    plt.close(fig2)
    print('保存', out2)

    run_lewm_pendulum()


def _wrap_pi(angle):
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class Pendulum:
    """θ=0 竖直向上。与 PETS / Dreamer 同一套物理。"""

    def __init__(self, max_torque=2.0, dt=0.05):
        self.g, self.m, self.l = 10.0, 1.0, 1.0
        self.dt, self.max_torque, self.max_speed = dt, max_torque, 8.0
        self.theta, self.omega = 0.0, 0.0

    def obs(self):
        return np.array(
            [np.cos(self.theta), np.sin(self.theta), self.omega],
            dtype=np.float64,
        )

    def reset(self):
        self.theta = float(np.random.uniform(-0.18, 0.18))
        self.omega = float(np.random.uniform(-0.25, 0.25))
        return self.obs()

    def step(self, action):
        u = float(np.clip(action, -1.0, 1.0)) * self.max_torque
        theta, omega = self.theta, self.omega
        theta_acc = (3.0 * self.g / (2.0 * self.l)) * np.sin(theta) + (
            3.0 / (self.m * self.l ** 2)
        ) * u
        omega = np.clip(omega + self.dt * theta_acc, -self.max_speed, self.max_speed)
        theta = _wrap_pi(theta + self.dt * omega)
        self.theta, self.omega = theta, omega
        done = False
        return self.obs(), done

    def render(self, size=IMG_SIZE):
        """火柴杆图：论文 LeWM 从像素学嵌入的最小替代。"""
        img = np.zeros((size, size), dtype=np.float64)
        cx = cy = size // 2
        n = max(size // 2 - 1, 4)
        ts = np.linspace(0.0, 1.0, n)
        xs = (cx + ts * n * np.sin(self.theta)).astype(int)
        ys = (cy - ts * n * np.cos(self.theta)).astype(int)
        m = (xs >= 0) & (xs < size) & (ys >= 0) & (ys < size)
        img[ys[m], xs[m]] = 1.0
        img += 0.02 * np.random.randn(size, size)
        return np.clip(img, 0.0, 1.0).ravel()


def sigreg_projections(z, n_proj=12):
    """教学版 SIGReg：随机 1D 投影上逼近 N(0,1)（Cramér–Wold）。"""
    if z.ndim == 1:
        z = z.reshape(1, -1)
    d = z.shape[1]
    dirs = np.random.randn(d, n_proj)
    dirs /= np.linalg.norm(dirs, axis=0, keepdims=True) + 1e-8
    h = z @ dirs
    return float(np.mean(h.mean(0) ** 2) + np.mean((h.std(0) - 1.0) ** 2))


class CompactLeWM:
    """恒等编码 z=s，残差线性预测器。规划与 PETS 同构，损失仍写 MSE+SIGReg。"""

    def __init__(self, in_dim=3):
        self.z_dim = in_dim
        self.Wp = np.random.randn(in_dim + 1, in_dim) * 0.05
        self.bp = np.zeros(in_dim)

    def encode(self, o):
        return np.asarray(o, dtype=np.float64)

    def predict(self, z, a):
        z = np.asarray(z, dtype=np.float64)
        if z.ndim == 1:
            z = z.reshape(1, -1)
        if np.ndim(a) == 0:
            a = np.full((z.shape[0], 1), float(a))
        elif np.ndim(a) == 1:
            a = a.reshape(-1, 1)
        x = np.concatenate([z, a], axis=-1)
        nxt = z + x @ self.Wp + self.bp
        nrm = np.sqrt(nxt[:, 0] ** 2 + nxt[:, 1] ** 2) + 1e-8
        nxt[:, 0] /= nrm
        nxt[:, 1] /= nrm
        return nxt

    def train_step(self, o, a, no, lr=0.08):
        z = self.encode(o)
        nz = self.encode(no)
        hat = self.predict(z, a)
        err = hat - nz
        pred_loss = float(np.mean(err ** 2))
        z_all = np.concatenate([z, nz], axis=0)
        reg = sigreg_projections(z_all)
        x = np.concatenate([z, a.reshape(-1, 1)], axis=-1)
        self.Wp -= lr * (x.T @ err) / len(o)
        self.bp -= lr * err.mean(axis=0)
        return pred_loss + LAMBDA_REG * reg, pred_loss, reg


def collect_pendulum_states(n=800):
    env = Pendulum()
    obs, acts, next_obs = [], [], []
    o = env.reset()
    for _ in range(n):
        a = float(np.random.uniform(-1, 1))
        no, done = env.step(a)
        obs.append(o)
        acts.append(a)
        next_obs.append(no)
        o = env.reset() if abs(env.theta) > 0.8 else no
    return np.array(obs), np.array(acts), np.array(next_obs)


def cem_plan_latent(model, z0, zg):
    mu = np.zeros(HORIZON_P)
    std = np.ones(HORIZON_P) * 0.55
    for _ in range(CEM_ITERS_P):
        seqs = np.clip(mu + std * np.random.randn(N_SAMPLE_P, HORIZON_P), -1.0, 1.0)
        scores = np.zeros(N_SAMPLE_P)
        for i, seq in enumerate(seqs):
            cost = 0.0
            z = z0.copy()
            for a in seq:
                z = model.predict(z.reshape(1, -1), np.array([a])).ravel()
                cost += np.sum((z - zg) ** 2)
                cost -= z[0] * np.exp(-0.05 * z[2] ** 2)
            scores[i] = -cost
        elite = seqs[np.argsort(scores)[-N_ELITE_P:]]
        mu = 0.7 * elite.mean(0) + 0.3 * mu
        std = 0.7 * elite.std(0) + 0.3 * std + 1e-3
    return float(np.clip(mu[0], -1.0, 1.0))


def run_lewm_pendulum():
    print('\n=== LeWM · 倒立摆（MSE + SIGReg + 潜空间 CEM）===')
    model = CompactLeWM(in_dim=3)
    obs, acts, next_obs = collect_pendulum_states()
    losses, preds, regs = [], [], []
    for step in range(550):
        idx = np.random.randint(0, len(obs), size=48)
        loss, pl, rg = model.train_step(obs[idx], acts[idx], next_obs[idx])
        losses.append(loss)
        preds.append(pl)
        regs.append(rg)
        if step % 110 == 0:
            print(f'  step {step:3d}  loss={loss:.4f}  pred={pl:.4f}  sigreg={rg:.4f}')

    zg = model.encode(np.array([1.0, 0.0, 0.0]))  # 直立状态本身（恒等编码）
    env = Pendulum()
    o = env.reset()
    thetas = [env.theta]
    for t in range(50):
        a = cem_plan_latent(model, model.encode(o), zg)
        o, _ = env.step(a)
        thetas.append(env.theta)
        if t % 10 == 0:
            print(f'  t={t:02d}  θ={env.theta:+.3f}  a={a:+.2f}')

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    axes[0].plot(losses, lw=1, label='总损失')
    axes[0].plot(preds, lw=1, alpha=0.7, label='MSE')
    axes[0].plot(regs, lw=1, alpha=0.7, label='SIGReg')
    axes[0].set_title('两项损失')
    axes[0].set_xlabel('step')
    axes[0].legend(fontsize=8)
    axes[1].plot(thetas, color='#bc4c00', lw=2)
    axes[1].axhline(0.0, color='gray', ls='--')
    axes[1].set_xlabel('时间步')
    axes[1].set_ylabel(r'$θ$ (rad)')
    axes[1].set_title('CEM → 直立嵌入')
    axes[1].grid(True, alpha=0.3)
    axes[2].imshow(env.render().reshape(IMG_SIZE, IMG_SIZE), cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('终局火柴杆（可视化）')
    axes[2].axis('off')
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'lewm_pendulum.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)


if __name__ == '__main__':
    main()
