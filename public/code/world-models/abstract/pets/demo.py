# -*- coding: utf-8 -*-
"""
=== PETS 最小演示：集成动力学 + CEM-MPC ===
一维质点追踪目标。用 B 个线性高斯模型组成集成，CEM 搜索动作序列，
MPC 只执行第一拍。输出 CEM 分布收缩图与闭环轨迹。
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


def main():
    print("=== PETS 玩具：CEM-MPC 追踪目标 ===")
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
    print('保存', out)


if __name__ == '__main__':
    main()
