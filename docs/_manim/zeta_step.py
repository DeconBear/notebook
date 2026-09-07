# -*- coding: utf-8 -*-
"""ζ 变了，二阶阶跃从过阻尼走到欠阻尼。"""
from __future__ import annotations

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, DashedLine, Scene, VGroup, ValueTracker, always_redraw, linear

from common import BLUE, GRAY, GREEN, INK, ORANGE, RED, label, polyline, prefix

WN, DT, T_END, R = 3.0, 0.005, 6.0, 1.0
ZETAS = [1.5, 1.0, 0.4, 0.15]
COLS = [GRAY, GREEN, BLUE, RED]
NAMES = ["ζ=1.5 过阻尼", "ζ=1 临界", "ζ=0.4 欠阻尼", "ζ=0.15 更欠"]


def step_response(zeta):
    n = int(T_END / DT)
    y, v = 0.0, 0.0
    ys = np.empty(n)
    for i in range(n):
        a = WN ** 2 * (R - y) - 2.0 * zeta * WN * v
        v = v + DT * a
        y = y + DT * v
        ys[i] = y
    return ys


class ZetaStep(Scene):
    def construct(self):
        self.add(
            label("阻尼比 ζ：阶跃会不会晃", 26, BLUE).to_edge(UP, buff=0.18),
            label("G(s)=ωn²/(s²+2ζωn s+ωn²)，ωn=3。虚线 = r=1。", 15, "#4A5568").to_edge(DOWN, buff=0.14),
        )
        bl = np.array([-5.7, -2.35, 0.0])
        w, h = 8.4, 4.4
        self.add(
            polyline([bl, bl + [w, 0, 0], bl + [w, h, 0], bl + [0, h, 0], bl], GRAY, 1.5),
            DashedLine(bl + [0, h / 2.2, 0], bl + [w, h / 2.2, 0], color=GRAY, dash_length=0.1),
        )
        series = []
        for z, c in zip(ZETAS, COLS):
            ys = step_response(z)
            pts = [
                bl + np.array([(i / (len(ys) - 1)) * w, float(np.clip(ys[i] / 1.85, 0, 1)) * h, 0])
                for i in range(len(ys))
            ]
            self.add(polyline(pts, c, 1.4, 0.2))
            series.append((pts, c))
        for i, (name, c) in enumerate(zip(NAMES, COLS)):
            self.add(label(name, 16, c).to_edge(RIGHT, buff=0.28).shift(UP * (2.15 - 0.42 * i)))

        s = ValueTracker(0.0)

        def draw():
            g = VGroup()
            for pts, c in series:
                g.add(prefix(pts, s.get_value(), c, 3.3))
            return g

        self.add(always_redraw(draw))
        self.wait(0.12)
        self.play(s.animate.set_value(1.0), run_time=6.8, rate_func=linear)
        self.wait(0.7)
