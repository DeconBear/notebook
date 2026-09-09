# -*- coding: utf-8 -*-
"""Draw a correct RSSM one-step architecture (PlaNet Fig.2c + demo.py)."""
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "images",
    "wm02-01-rssm-architecture.png",
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

C_GRU = "#ECEFF1"
C_ENC = "#E8EAF6"
C_PRIOR = "#E8F5E9"
C_POST = "#E3F2FD"
C_SAMP = "#FFF8E1"
C_DEC = "#FFF3E0"
C_KL = "#F3E5F5"
C_WARN = "#FFEBEE"
C_H = "#1565C0"
C_S = "#E65100"
C_O = "#6A1B9A"
C_A = "#00897B"
C_EDGE = "#37474F"
C_TRAIN = "#1565C0"
C_IMAG = "#2E7D32"


class Box:
    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h

    @property
    def cx(self):
        return self.x + self.w / 2

    @property
    def cy(self):
        return self.y + self.h / 2

    def top(self, dx=0.0):
        return (self.cx + dx, self.y + self.h)

    def bot(self, dx=0.0):
        return (self.cx + dx, self.y)

    def left(self, dy=0.0):
        return (self.x, self.cy + dy)

    def right(self, dy=0.0):
        return (self.x + self.w, self.cy + dy)


def add_box(ax, x, y, w, h, text, fc, ec=C_EDGE, lw=1.5, fs=8.4, radius=0.10, tc="#1A1A1A"):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.018,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2,
    )
    ax.add_patch(p)
    ax.text(
        x + w / 2, y + h / 2, text, ha="center", va="center",
        fontsize=fs, color=tc, zorder=3, linespacing=1.25,
    )
    return Box(x, y, w, h)


def add_round(ax, cx, cy, w, h, text, fc, ec, fs=10.5, tc="white", lw=1.7):
    p = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.24",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3,
    )
    ax.add_patch(p)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            color=tc, zorder=4, fontweight="bold")
    return Box(cx - w / 2, cy - h / 2, w, h)


def arrow(ax, p1, p2, color=C_EDGE, ls="-", lw=1.55):
    ax.add_patch(
        FancyArrowPatch(
            p1, p2, arrowstyle="-|>", mutation_scale=11,
            linewidth=lw, linestyle=ls, color=color, zorder=1,
            shrinkA=0.5, shrinkB=0.5,
        )
    )


def elbow(ax, pts, color=C_EDGE, ls="-", lw=1.45):
    """Orthogonal polyline; arrow only on the last segment."""
    for i in range(len(pts) - 2):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        ax.plot([x0, x1], [y0, y1], color=color, lw=lw, ls=ls, zorder=1, solid_capstyle="round")
    arrow(ax, pts[-2], pts[-1], color=color, ls=ls, lw=lw)


def badge(ax, x, y, n, color):
    ax.add_patch(plt.Circle((x, y), 0.16, facecolor=color, edgecolor="white",
                            linewidth=1.1, zorder=5))
    ax.text(x, y, str(n), ha="center", va="center", fontsize=7.5,
            color="white", fontweight="bold", zorder=6)


def panel_frame(ax, x, y, w, h, title, color):
    ax.add_patch(Rectangle((x, y), w, h, facecolor="#FFFFFF",
                           edgecolor=color, linewidth=1.8, zorder=0))
    ax.add_patch(Rectangle((x, y + h - 0.50), w, 0.50, facecolor=color,
                           edgecolor=color, linewidth=0, zorder=1))
    ax.text(x + w / 2, y + h - 0.25, title, ha="center", va="center",
            fontsize=11, color="white", fontweight="bold", zorder=2)


def draw():
    fig, ax = plt.subplots(figsize=(16.8, 10.6), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 16.8)
    ax.set_ylim(0, 10.6)
    ax.axis("off")

    ax.text(
        8.4, 10.32,
        "RSSM 一步计算顺序（PlaNet Fig.2c + 本章 demo.py）",
        ha="center", va="center", fontsize=14.5, fontweight="bold", color="#212121",
    )
    ax.text(
        8.4, 9.98,
        r"顺序固定：GRU 更新 $h_t$  →  先验/后验只出 $(\mu,\sigma)$  →  采样得到 $s_t$  →  解码。"
        r"本步 $s_t$ 绝不进入本步 GRU。",
        ha="center", va="center", fontsize=9.2, color="#546E7A",
    )

    # ===================== LEFT: TRAIN =====================
    Lx, Ly, Lw, Lh = 0.22, 0.78, 8.05, 9.00
    panel_frame(ax, Lx, Ly, Lw, Lh, "训练 / 滤波（看观测）    demo.py · forward", C_TRAIN)

    s_prev = add_round(ax, 1.85, 8.95, 1.28, 0.46, r"$s_{t-1}$", C_S, C_S)
    a_prev = add_round(ax, 3.55, 8.95, 1.28, 0.46, r"$a_{t-1}$", C_A, C_A)
    o_t = add_round(ax, 7.15, 8.95, 1.15, 0.46, r"$o_t$", C_O, C_O)
    ax.text(2.70, 9.35, "上一拍留下", ha="center", fontsize=7.8, color="#607D8B")
    ax.text(7.15, 9.35, "当前观测", ha="center", fontsize=7.8, color="#607D8B")

    gru = add_box(
        ax, 1.05, 7.48, 3.70, 0.98,
        "①  GRUCell\n"
        + r"input $=[s_{t-1};\,a_{t-1}]$，内部 $h_{t-1}\!\to\! h_t$",
        C_GRU, fs=8.2,
    )
    badge(ax, 0.98, 8.36, 1, C_TRAIN)
    h_t = add_round(ax, 2.90, 7.08, 1.18, 0.42, r"$h_t$", C_H, C_H, fs=11)

    arrow(ax, s_prev.bot(), (s_prev.cx, gru.y + gru.h), C_S)
    arrow(ax, a_prev.bot(), (a_prev.cx, gru.y + gru.h), C_A)
    arrow(ax, gru.bot(), h_t.top(), C_H, lw=1.7)

    prior = add_box(
        ax, 0.55, 5.55, 2.55, 0.92,
        "②  先验  " + r"$p(s_t\mid h_t)$" + "\n只看 $h_t$，不看观测",
        C_PRIOR, fs=8.0, ec="#2E7D32",
    )
    badge(ax, 0.48, 6.37, 2, "#2E7D32")
    mup = add_box(ax, 0.70, 4.95, 2.25, 0.42, r"$(\mu_p,\ \sigma_p)$", C_PRIOR, fs=9.2, ec="#2E7D32")

    enc = add_box(
        ax, 5.55, 7.42, 2.40, 0.78,
        "③  编码器\n" + r"$e_t=\mathrm{enc}(o_t)$",
        C_ENC, fs=8.2, ec="#3949AB",
    )
    badge(ax, 5.48, 8.10, 3, "#3949AB")
    e_t = add_box(ax, 5.75, 6.82, 2.00, 0.40, r"$e_t$", C_ENC, fs=9.5, ec="#3949AB")
    ax.text(6.75, 6.58, "像素：CNN　本章 demo：恒等", ha="center", fontsize=6.8, color="#5C6BC0")

    post = add_box(
        ax, 4.85, 5.20, 3.10, 0.95,
        "④  后验  " + r"$q(s_t\mid h_t,e_t)$" + "\n"
        + r"$\mathrm{concat}(h_t,\ e_t)\ \to\ (\mu_q,\sigma_q)$",
        C_POST, fs=8.0, ec=C_TRAIN,
    )
    badge(ax, 4.78, 6.05, 4, C_TRAIN)
    muq = add_box(ax, 5.10, 4.60, 2.60, 0.42, r"$(\mu_q,\ \sigma_q)$", C_POST, fs=9.2, ec=C_TRAIN)

    # h_t → prior (down-left, stay right of gutter)
    elbow(ax, [h_t.left(), (1.82, h_t.cy), (1.82, prior.y + prior.h)], C_H)
    arrow(ax, prior.bot(), mup.top(), "#2E7D32")

    arrow(ax, o_t.bot(), enc.top(), C_O)
    arrow(ax, enc.bot(), e_t.top(), "#3949AB")
    arrow(ax, e_t.bot(), post.top(), "#3949AB")
    # h_t → posterior
    elbow(ax, [h_t.right(), (4.55, h_t.cy), (4.55, post.cy), post.left()], C_H)

    kl = add_box(ax, 3.15, 4.72, 1.55, 0.62, "KL$(q\\,\\|\\,p)$\n只比参数", C_KL, fs=7.6, ec="#7B1FA2")
    arrow(ax, mup.right(), kl.left(dy=0.08), "#7B1FA2", ls="--", lw=1.25)
    arrow(ax, muq.left(), kl.right(dy=-0.08), "#7B1FA2", ls="--", lw=1.25)

    samp = add_box(
        ax, 2.35, 3.15, 3.95, 0.85,
        "⑤  从后验采样（训练走这条）\n"
        + r"$s_t=\mu_q+\sigma_q\,\varepsilon,\ \ \varepsilon\sim\mathcal{N}(0,I)$",
        C_SAMP, fs=8.1, ec="#F9A825",
    )
    badge(ax, 2.28, 3.90, 5, "#F9A825")
    s_t = add_round(ax, 4.32, 2.72, 1.18, 0.42, r"$s_t$", C_S, C_S, fs=11)

    elbow(ax, [muq.bot(), (muq.cx, samp.cy), (samp.x + samp.w, samp.cy)], C_TRAIN)
    arrow(ax, samp.bot(), s_t.top(), C_S, lw=1.7)

    dec = add_box(
        ax, 1.55, 1.55, 5.50, 0.55,
        r"⑥  解码器　$\hat o_t=\mathrm{dec}(h_t,\,s_t)$　（吃采样后的 $s_t$，不是 $\mu$）",
        C_DEC, fs=8.2, ec="#EF6C00",
    )
    badge(ax, 1.48, 2.02, 6, "#EF6C00")
    ohat = add_round(ax, 4.32, 1.08, 1.18, 0.40, r"$\hat o_t$", "#EF6C00", "#EF6C00", fs=10.5)

    arrow(ax, s_t.bot(), dec.top(), C_S, lw=1.7)
    arrow(ax, dec.bot(), ohat.top(), "#EF6C00")

    # h_t → decoder via left gutter (x=0.42, outside prior)
    elbow(
        ax,
        [h_t.left(), (0.42, h_t.cy), (0.42, dec.cy), dec.left()],
        C_H, ls="--", lw=1.35,
    )
    ax.text(0.42, 3.55, r"$h_t$", fontsize=7.5, color=C_H, ha="center", va="bottom")

    ax.text(
        7.05, 2.72, "下一拍才把\n$s_t$ 当作 $s_{t-1}$\n送进 GRU",
        ha="center", va="center", fontsize=7.4, color=C_S,
        bbox=dict(boxstyle="round,pad=0.22", fc="#FFF3E0", ec=C_S, lw=1.0),
    )
    arrow(ax, s_t.right(), (6.35, 2.72), C_S, ls="--", lw=1.15)

    ax.text(
        4.25, 0.88,
        r"训练时先验仍要算（给 KL），但 $s_t$ 不从先验采样。",
        ha="center", fontsize=7.4, color="#546E7A",
    )

    # ===================== RIGHT: IMAGINE =====================
    Rx, Ry, Rw, Rh = 8.50, 0.78, 8.05, 9.00
    panel_frame(ax, Rx, Ry, Rw, Rh, "想象 / 开环（不看观测）    demo.py · imagine", C_IMAG)

    s_prev_r = add_round(ax, 10.55, 8.95, 1.28, 0.46, r"$s_{t-1}$", C_S, C_S)
    a_prev_r = add_round(ax, 12.35, 8.95, 1.28, 0.46, r"$a_{t-1}$", C_A, C_A)
    ax.text(11.45, 9.35, "上一拍留下", ha="center", fontsize=7.8, color="#607D8B")

    ax.text(
        15.35, 8.95, "没有 $o_t$\n无编码器 / 后验 / KL",
        ha="center", va="center", fontsize=7.6, color="#C62828",
        bbox=dict(boxstyle="round,pad=0.28", fc=C_WARN, ec="#C62828", lw=1.15),
    )

    gru_r = add_box(
        ax, 9.70, 7.48, 4.50, 0.98,
        "①  GRUCell（与训练同一套权重）\n"
        + r"input $=[s_{t-1};\,a_{t-1}]$，内部 $h_{t-1}\!\to\! h_t$",
        C_GRU, fs=8.2,
    )
    badge(ax, 9.63, 8.36, 1, C_IMAG)
    h_tr = add_round(ax, 11.95, 7.08, 1.18, 0.42, r"$h_t$", C_H, C_H, fs=11)

    arrow(ax, s_prev_r.bot(), (s_prev_r.cx, gru_r.y + gru_r.h), C_S)
    arrow(ax, a_prev_r.bot(), (a_prev_r.cx, gru_r.y + gru_r.h), C_A)
    arrow(ax, gru_r.bot(), h_tr.top(), C_H, lw=1.7)

    prior_r = add_box(
        ax, 10.05, 5.35, 3.80, 0.95,
        "②  先验  " + r"$p(s_t\mid h_t)$" + "\n"
        + r"输出 $(\mu_p,\sigma_p)$ —— 想象时这就是 $s$ 的来源",
        C_PRIOR, fs=8.1, ec="#2E7D32",
    )
    badge(ax, 9.98, 6.20, 2, "#2E7D32")
    mup_r = add_box(ax, 10.55, 4.72, 2.80, 0.42, r"$(\mu_p,\ \sigma_p)$", C_PRIOR, fs=9.5, ec="#2E7D32")

    arrow(ax, h_tr.bot(), prior_r.top(), C_H, lw=1.7)
    arrow(ax, prior_r.bot(), mup_r.top(), "#2E7D32")

    samp_r = add_box(
        ax, 9.90, 3.15, 4.10, 1.00,
        "⑤  从先验采样（想象走这条）\n"
        + r"$s_t=\mu_p+\sigma_p\,\varepsilon$"
        + "\n本章评估实现用均值：" + r"$s_t=\mu_p$",
        C_SAMP, fs=8.0, ec="#F9A825",
    )
    badge(ax, 9.83, 4.05, 5, "#F9A825")
    s_tr = add_round(ax, 11.95, 2.72, 1.18, 0.42, r"$s_t$", C_S, C_S, fs=11)

    arrow(ax, mup_r.bot(), samp_r.top(), "#2E7D32")
    arrow(ax, samp_r.bot(), s_tr.top(), C_S, lw=1.7)

    dec_r = add_box(
        ax, 9.85, 1.55, 4.20, 0.55,
        r"⑥  解码器　$\hat o_t=\mathrm{dec}(h_t,\,s_t)$",
        C_DEC, fs=8.6, ec="#EF6C00",
    )
    badge(ax, 9.78, 2.02, 6, "#EF6C00")
    ohat_r = add_round(ax, 11.95, 1.08, 1.18, 0.40, r"$\hat o_t$", "#EF6C00", "#EF6C00", fs=10.5)

    arrow(ax, s_tr.bot(), dec_r.top(), C_S, lw=1.7)
    arrow(ax, dec_r.bot(), ohat_r.top(), "#EF6C00")

    # h_t → decoder via right gutter (x=15.55), away from the red callout at 15.35,8.95
    elbow(
        ax,
        [h_tr.right(), (15.55, h_tr.cy), (15.55, dec_r.cy), dec_r.right()],
        C_H, ls="--", lw=1.35,
    )
    ax.text(15.55, 4.20, r"$h_t$", fontsize=7.5, color=C_H, ha="center", va="bottom")

    ax.text(
        15.22, 2.72, "下一拍才把\n$s_t$ 当作 $s_{t-1}$\n送进 GRU",
        ha="center", va="center", fontsize=7.4, color=C_S,
        bbox=dict(boxstyle="round,pad=0.22", fc="#FFF3E0", ec=C_S, lw=1.0),
    )
    arrow(ax, s_tr.right(), (14.52, 2.72), C_S, ls="--", lw=1.15)

    ax.text(
        12.52, 0.88,
        r"跳过 ③ 编码器、④ 后验：闭眼只靠历史 + 动作往前滚。",
        ha="center", fontsize=7.4, color="#546E7A",
    )

    # ===================== BOTTOM RULES =====================
    ax.add_patch(Rectangle((0.22, 0.06), 16.33, 0.62, facecolor="#FFF8E1",
                           edgecolor="#F9A825", linewidth=1.15, zorder=0))
    ax.text(
        8.4, 0.37,
        r"连线规则：GRU 外部只有 $s_{t-1},a_{t-1}$（$h_{t-1}$ 在 Cell 内部）。"
        r"先验只吃 $h_t$。后验只吃 $(h_t,e_t)$。采样只吃 $(\mu,\sigma,\varepsilon)$，$h_t$ 不进采样。"
        r"解码器吃 $(h_t,s_t)$。$s_t$ 不连回本步 GRU / 先验 / 后验。",
        ha="center", va="center", fontsize=7.7, color="#5D4037",
    )

    fig.savefig(OUT, dpi=150, facecolor="white", bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    draw()
