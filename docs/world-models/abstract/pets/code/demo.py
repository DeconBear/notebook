# -*- coding: utf-8 -*-
"""
=== PETS 演示：CEM-MPC + 倒立摆 ===
1) 一维质点：看清 CEM 分布收缩、MPC 只执行第一拍。
2) 倒立摆：观测 [cosθ, sinθ, ω]，概率 MLP 集成 + TS∞ + CEM，
   把摆稳定在竖直向上（课上同一套物理，不依赖 Gymnasium）。
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

DT = 0.08
MASS = 1.0
DAMPING = 0.25
HORIZON = 12
N_SAMPLE = 80
N_ELITE = 12
CEM_ITERS = 6
N_ENSEMBLE = 5
N_PARTICLE = 8


def true_step(pos, vel, acc):
    """真实质点：加速度控制，带轻微过程噪声。"""
    vel = vel + DT * (acc - DAMPING * vel) / MASS
    pos = pos + DT * vel
    vel = vel + 0.01 * np.random.randn()
    return pos, vel


def fit_ensemble(transitions, n=N_ENSEMBLE):
    """每个成员对 (s,a)->ds 做带噪声的岭回归，模拟 bootstrap 分歧。"""
    models = []
    x = np.stack([t[0] for t in transitions])  # [N, 3] pos,vel,acc
    y = np.array([t[1] for t in transitions])  # [N, 2] dpos,dvel
    n_data = len(x)
    for _ in range(n):
        idx = np.random.randint(0, n_data, size=n_data)
        xb, yb = x[idx], y[idx]
        xb = np.concatenate([xb, np.ones((len(xb), 1))], axis=1)
        # 岭回归
        a = xb.T @ xb + 1e-2 * np.eye(xb.shape[1])
        w = np.linalg.solve(a, xb.T @ yb)
        resid = yb - xb @ w
        var = np.maximum(resid.var(axis=0), 1e-4)
        models.append((w, var))
    return models


def predict_step(models, pos, vel, acc, member=None):
    """PE：抽一个成员，再从高斯里采样下一步。"""
    b = np.random.randint(len(models)) if member is None else member
    w, var = models[b]
    feat = np.array([pos, vel, acc, 1.0])
    mean = feat @ w
    noise = np.random.randn(2) * np.sqrt(var)
    dpos, dvel = mean + noise
    return pos + dpos, vel + dvel, b


def rollout_return(models, pos, vel, actions, target, ts_inf=True):
    """TS：粒子平均回报。ts_inf=True 时粒子绑定成员（TS∞）。"""
    rets = []
    for p in range(N_PARTICLE):
        member = np.random.randint(len(models)) if ts_inf else None
        pp, vv = pos, vel
        r = 0.0
        for a in actions:
            pp, vv, member = predict_step(models, pp, vv, a, member if ts_inf else None)
            r -= (pp - target) ** 2 + 0.02 * a ** 2
        rets.append(r)
    return float(np.mean(rets))


def cem_plan(models, pos, vel, target):
    """对长度为 HORIZON 的动作序列做 CEM。"""
    mu = np.zeros(HORIZON)
    std = np.ones(HORIZON) * 2.5
    history = []
    for _ in range(CEM_ITERS):
        noise = np.random.randn(N_SAMPLE, HORIZON)
        seqs = mu + std * noise
        scores = np.array([rollout_return(models, pos, vel, s, target) for s in seqs])
        elite = seqs[np.argsort(scores)[-N_ELITE:]]
        mu = 0.7 * elite.mean(axis=0) + 0.3 * mu
        std = 0.7 * elite.std(axis=0) + 0.3 * std + 0.05
        history.append((mu.copy(), std.copy(), scores.max()))
    return mu, history


def collect_random(n=80):
    trans = []
    for _ in range(n):
        pos, vel = np.random.uniform(-2, 2), np.random.uniform(-1, 1)
        acc = np.random.uniform(-3, 3)
        npos, nvel = true_step(pos, vel, acc)
        trans.append((np.array([pos, vel, acc]), np.array([npos - pos, nvel - vel])))
    return trans


def _wrap_pi(angle):
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


class Pendulum:
    """θ=0 竖直向上。动作 u∈[-1,1]，力矩 = u * max_torque。"""

    def __init__(self, max_torque=2.0, dt=0.05):
        self.g, self.m, self.l = 10.0, 1.0, 1.0
        self.dt, self.max_torque, self.max_speed = dt, max_torque, 8.0
        self.theta, self.omega = 0.0, 0.0

    def obs(self):
        return np.array([np.cos(self.theta), np.sin(self.theta), self.omega], dtype=np.float64)

    def reset(self, swing_up=False):
        if swing_up:
            self.theta = _wrap_pi(np.pi + float(np.random.uniform(-0.2, 0.2)))
            self.omega = float(np.random.uniform(-1.0, 1.0))
        else:
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
        reward = float(
            np.exp(-8.0 * theta ** 2) * np.exp(-0.05 * omega ** 2)
            - 0.01 * (u / self.max_torque) ** 2
        )
        done = False  # 本 demo 只做直立附近，不把倒下当成提前终止
        return self.obs(), reward, done


def pendulum_reward(states):
    """CEM 打分：只能从观测算。cosθ=1 表示直立。"""
    cos_t, omega = states[:, 0], states[:, 2]
    return cos_t * np.exp(-0.05 * omega ** 2) - 0.05 * (1.0 - cos_t)


class ProbMLP:
    """论文成员：概率 MLP + 高斯 NLL。倒立摆闭环为赶 CPU 时间改用下方岭回归集成。"""
    def __init__(self, in_dim, out_dim, hidden=32):
        sc = 0.12
        self.W1 = np.random.randn(in_dim, hidden) * sc
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, hidden) * sc
        self.b2 = np.zeros(hidden)
        self.W_mu = np.random.randn(hidden, out_dim) * sc
        self.b_mu = np.zeros(out_dim)
        self.W_ls = np.random.randn(hidden, out_dim) * 0.01
        self.b_ls = np.full(out_dim, -1.0)

    def _fwd(self, x):
        h1 = np.maximum(x @ self.W1 + self.b1, 0.0)
        h2 = np.maximum(h1 @ self.W2 + self.b2, 0.0)
        mu = h2 @ self.W_mu + self.b_mu
        std = np.log1p(np.exp(np.clip(h2 @ self.W_ls + self.b_ls, -20, 20))) + 1e-4
        return mu, std, h1, h2

    def train_epoch(self, X, Y, lr=3e-3, batch=64):
        n = len(X)
        idx = np.random.permutation(n)
        for start in range(0, n, batch):
            b = idx[start:start + batch]
            xb, yb = X[b], Y[b]
            mu, std, h1, h2 = self._fwd(xb)
            inv = 1.0 / (std ** 2)
            dmu = (mu - yb) * inv
            dstd = 1.0 / std - (yb - mu) ** 2 * inv / std
            dls = dstd * (1.0 / (1.0 + np.exp(-(h2 @ self.W_ls + self.b_ls))))
            dh2 = dmu @ self.W_mu.T + dls @ self.W_ls.T
            dh2 *= h2 > 0
            dh1 = (dh2 @ self.W2.T) * (h1 > 0)
            self.W_mu -= lr * (h2.T @ dmu) / len(b)
            self.b_mu -= lr * dmu.mean(0)
            self.W_ls -= lr * (h2.T @ dls) / len(b)
            self.b_ls -= lr * dls.mean(0)
            self.W2 -= lr * (h1.T @ dh2) / len(b)
            self.b2 -= lr * dh2.mean(0)
            self.W1 -= lr * (xb.T @ dh1) / len(b)
            self.b1 -= lr * dh1.mean(0)


class LinearEnsemble:
    """倒立摆用岭回归 bootstrap 集成：同构 PE，CPU 上比浅层 MLP 稳。"""

    def __init__(self, n_models=5):
        self.n_models = n_models
        self.models = []

    def fit(self, states, actions, next_states):
        x = np.concatenate(
            [states, actions.reshape(-1, 1), np.ones((len(states), 1))], axis=1
        )
        y = next_states - states
        n = len(x)
        self.models = []
        for _ in range(self.n_models):
            idx = np.random.randint(0, n, size=n)
            xb, yb = x[idx], y[idx]
            w = np.linalg.solve(xb.T @ xb + 1e-2 * np.eye(xb.shape[1]), xb.T @ yb)
            resid = yb - xb @ w
            var = np.maximum(resid.var(axis=0), 1e-4)
            self.models.append((w, var))

    def sample_next(self, states, actions, model_idx):
        feat = np.concatenate(
            [states, actions.reshape(-1, 1), np.ones((len(states), 1))], axis=1
        )
        next_s = np.zeros_like(states)
        for b, (w, var) in enumerate(self.models):
            mask = model_idx == b
            if not np.any(mask):
                continue
            mean = feat[mask] @ w
            noise = np.random.randn(*mean.shape) * np.sqrt(var)
            next_s[mask] = states[mask] + mean + noise
        nrm = np.sqrt(next_s[:, 0] ** 2 + next_s[:, 1] ** 2) + 1e-8
        next_s[:, 0] /= nrm
        next_s[:, 1] /= nrm
        return next_s


def cem_plan_pendulum(ens, state, horizon=10, n_samples=24, n_elites=8,
                      n_iters=4, n_particles=8):
    B, H, P, N = ens.n_models, horizon, n_particles, n_samples
    mean, std = np.zeros(H), np.ones(H) * 0.5
    for _ in range(n_iters):
        acts = np.clip(mean + std * np.random.randn(N, H), -1.0, 1.0)
        particles = np.tile(state, (N, P, 1))
        boot = np.random.randint(0, B, size=(N, P))
        returns = np.zeros(N)
        for t in range(H):
            flat_s = particles.reshape(N * P, -1)
            flat_a = np.repeat(acts[:, t], P)
            flat_b = boot.reshape(N * P)
            ns = ens.sample_next(flat_s, flat_a, flat_b)
            particles = ns.reshape(N, P, -1)
            returns += pendulum_reward(ns).reshape(N, P).mean(axis=1)
        elite = acts[np.argsort(returns)[-n_elites:]]
        mean, std = elite.mean(0), elite.std(0) + 1e-3
    return float(np.clip(mean[0], -1.0, 1.0))


def run_pets_pendulum(n_trials=4, init_random=220, max_steps=50):
    print('\n=== PETS · 倒立摆（直立附近稳定）===')
    env = Pendulum()
    ens = LinearEnsemble()
    buf_s, buf_a, buf_ns = [], [], []
    s = env.reset()
    for _ in range(init_random):
        a = float(np.random.uniform(-1, 1))
        ns, _, done = env.step(a)
        buf_s.append(s); buf_a.append(a); buf_ns.append(ns)
        s = env.reset() if abs(env.theta) > 0.8 else ns
    returns, angles = [], []
    for trial in range(1, n_trials + 1):
        ens.fit(np.array(buf_s), np.array(buf_a), np.array(buf_ns))
        s = env.reset()
        ret = 0.0
        ep_ns = s
        for _ in range(max_steps):
            a = cem_plan_pendulum(ens, s)
            ns, r, _ = env.step(a)
            buf_s.append(s); buf_a.append(a); buf_ns.append(ns)
            ret += r
            s = ns
            ep_ns = ns
        ang = abs(float(np.arctan2(ep_ns[1], ep_ns[0])))
        returns.append(ret)
        angles.append(ang)
        print(f'  trial {trial}/{n_trials}  return={ret:6.1f}  |θ_end|={ang:.3f}  |D|={len(buf_s)}')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(range(1, len(returns) + 1), returns, 'o-', color='#1a7f37', lw=2)
    axes[0].set_xlabel('Trial'); axes[0].set_ylabel('回合回报')
    axes[0].set_title('PETS · 倒立摆平衡'); axes[0].grid(True, alpha=0.3)
    axes[1].plot(range(1, len(angles) + 1), angles, 's-', color='#bc4c00', lw=2)
    axes[1].set_xlabel('Trial'); axes[1].set_ylabel(r'$|\theta_{\mathrm{end}}|$ (rad)')
    axes[1].set_title('终局偏离直立（越低越好）'); axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'pets_pendulum.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)
    return returns, angles


def main():
    print("=== PETS：一维质点 CEM + 倒立摆 ===")
    trans = collect_random()
    models = fit_ensemble(trans)
    target = 1.5
    pos, vel = -1.2, 0.0
    traj = [pos]
    cem_stds = []

    for t in range(40):
        mu, hist = cem_plan(models, pos, vel, target)
        cem_stds.append(hist[-1][1].mean())
        acc = float(np.clip(mu[0], -4, 4))
        old_pos, old_vel = pos, vel
        pos, vel = true_step(pos, vel, acc)
        traj.append(pos)
        trans.append((np.array([old_pos, old_vel, acc]),
                      np.array([pos - old_pos, vel - old_vel])))
        if t % 8 == 7:
            models = fit_ensemble(trans[-200:])
        print(f"t={t:02d}  pos={pos:+.3f}  acc={acc:+.3f}  cem_std={cem_stds[-1]:.3f}")

    # 再跑一次 CEM 只为可视化分布收缩
    _, hist = cem_plan(models, -1.0, 0.0, target)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(traj, marker='o', ms=3, label='位置')
    axes[0].axhline(target, color='C1', ls='--', label='目标')
    axes[0].set_xlabel('时间步')
    axes[0].set_ylabel('位置')
    axes[0].set_title('MPC 只执行 CEM 第一拍')
    axes[0].legend()
    xs = np.arange(HORIZON)
    for i, (mu, std, _) in enumerate(hist):
        axes[1].plot(xs, mu, alpha=0.3 + 0.7 * i / len(hist), label=f'iter {i+1}' if i in (0, len(hist)-1) else None)
        if i == len(hist) - 1:
            axes[1].fill_between(xs, mu - std, mu + std, color='C0', alpha=0.15)
    axes[1].set_xlabel('计划内时间')
    axes[1].set_ylabel('动作均值')
    axes[1].set_title('CEM：均值收敛、方差收缩')
    axes[1].legend()
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'pets_cem_mpc.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)

    run_pets_pendulum()


if __name__ == '__main__':
    main()
