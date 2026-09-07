# -*- coding: utf-8 -*-
"""PDE 残差：错解形态靠近真解，残差掉到 0。"""
from __future__ import annotations

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, DashedLine, Scene, VGroup, ValueTracker, always_redraw, linear

from common import BLUE, GRAY, GREEN, INK, ORANGE, RED, label, polyline

XS = np.linspace(0.0, 1.0, 120)
U_TRUE = np.sin(np.pi * XS)
U_WRONG = 4.0 * XS * (1.0 - XS)
F = (np.pi ** 2) * np.sin(np.pi * XS)


def residual(u):
    # 二阶差分
    h = XS[1] - XS[0]
    r = np.zeros_like(u)
    r[1:-1] = -(u[2:] - 2 * u[1:-1] + u[:-2]) / h ** 2 - F[1:-1]
    r[0] = r[1]
    r[-1] = r[-2]
    return r


class ResidualMorph(Scene):
    def construct(self):
        self.add(
            label("形状像，不等于满足方程", 26, BLUE).to_edge(UP, buff=0.16),
            label("橙：4x(1-x) 错解 → 绿：sin(πx) 真解。下栏是 PDE 残差 −u''−f。", 15, "#4A5568").to_edge(DOWN, buff=0.12),
        )
        top = np.array([-5.6, 0.35, 0.0])
        bot = np.array([-5.6, -2.85, 0.0])
        w, h1, h2 = 9.2, 2.35, 1.85
        self.add(
            label("u(x)", 16, INK).move_to([-5.95, 2.4, 0]),
            label("残差 r(x)", 16, INK).move_to([-5.7, -0.55, 0]),
            polyline([top, top + [w, 0, 0], top + [w, h1, 0], top + [0, h1, 0], top], GRAY, 1.4),
            polyline([bot, bot + [w, 0, 0], bot + [w, h2, 0], bot + [0, h2, 0], bot], GRAY, 1.4),
        )
        # 真解浅线
        def pack(y, bl, hh, ymin, ymax):
            return [
                bl + np.array([(XS[i]) * w, (float(y[i]) - ymin) / (ymax - ymin) * hh, 0.0]) for i in range(len(XS))
            ]

        true_pts = pack(U_TRUE, top, h1, -0.05, 1.15)
        self.add(polyline(true_pts, GREEN, 2.0, 0.3))
        self.add(DashedLine(bot + [0, h2 / 2, 0], bot + [w, h2 / 2, 0], color=GRAY, dash_length=0.08))

        s = ValueTracker(0.0)

        def draw():
            t = s.get_value()
            u = (1 - t) * U_WRONG + t * U_TRUE
            r = residual(u)
            g = VGroup()
            g.add(polyline(pack(u, top, h1, -0.05, 1.15), ORANGE if t < 0.85 else GREEN, 3.5))
            rmax = 40.0
            g.add(polyline(pack(r, bot, h2, -rmax, rmax), RED, 3.0))
            g.add(label(f"混合 {t:.2f}", 18, INK).to_edge(RIGHT, buff=0.4).shift(UP * 2.15))
            return g

        self.add(always_redraw(draw))
        self.wait(0.15)
        self.play(s.animate.set_value(1.0), run_time=6.5, rate_func=linear)
        self.wait(0.8)
