# -*- coding: utf-8 -*-
"""PID：同一质量-弹簧，P / PD / PID 阶跃。"""
from __future__ import annotations

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    DashedLine,
    Rectangle,
    Scene,
    VGroup,
    ValueTracker,
    always_redraw,
    linear,
)

from common import BLUE, GRAY, GREEN, ORANGE, RED, label, polyline, prefix

M, C, K = 1.0, 0.4, 2.0
DT, T_END, R = 0.01, 8.0, 1.0


def plant_step(x, v, u):
    a = (u - C * v - K * x) / M
    v = v + DT * a
    return x + DT * v, v


def simulate(kp=0.0, ki=0.0, kd=0.0):
    n = int(T_END / DT)
    xs = np.empty(n)
    x, v, integ, e_prev = 0.0, 0.0, 0.0, 0.0
    for i in range(n):
        e = R - x
        integ += e * DT
        de = (e - e_prev) / DT
        u = kp * e + ki * integ + kd * de
        e_prev = e
        x, v = plant_step(x, v, u)
        xs[i] = x
    return xs


class PIDStep(Scene):
    def construct(self):
        self.add(
            label("同一植物，三种纠偏", 28, BLUE).to_edge(UP, buff=0.16),
            label("质量-弹簧-阻尼  m=1, c=0.4, k=2。虚线 = 目标 r=1。", 15, "#4A5568").to_edge(DOWN, buff=0.12),
        )
        cases = [
            ("P  留静差", simulate(kp=8.0), RED, -4.35),
            ("PD  刹住超调", simulate(kp=8.0, kd=4.0), ORANGE, 0.0),
            ("PID  积掉残差", simulate(kp=8.0, ki=3.0, kd=4.0), GREEN, 4.35),
        ]
        plots, carts = [], []
        for name, xs, color, cx in cases:
            self.add(label(name, 18, color).move_to([cx, 2.55, 0]))
            bl = np.array([cx - 1.55, 0.38, 0.0])
            w, h = 3.1, 1.5
            self.add(
                polyline(
                    [bl, bl + [w, 0, 0], bl + [w, h, 0], bl + [0, h, 0], bl],
                    GRAY,
                    1.5,
                ),
                DashedLine(bl + [0, 0.72 * h, 0], bl + [w, 0.72 * h, 0], color=GRAY, dash_length=0.08),
            )
            pts = [
                bl + np.array([(i / (len(xs) - 1)) * w, float(np.clip(xs[i] / 1.5, 0, 1)) * h, 0.0])
                for i in range(len(xs))
            ]
            self.add(polyline(pts, color, 1.3, 0.2))
            plots.append((pts, color))
            rail = np.array([cx, -1.52, 0.0])
            self.add(polyline([rail + LEFT * 1.25, rail + RIGHT * 1.25], GRAY, 3))
            tgt = rail + RIGHT * 0.95
            self.add(DashedLine(tgt + UP * 0.32, tgt + DOWN * 0.18, color=GRAY, dash_length=0.06))
            carts.append((xs, color, rail))

        s = ValueTracker(0.0)

        def draw():
            t = s.get_value()
            g = VGroup()
            for pts, color in plots:
                g.add(prefix(pts, t, color, 3.2))
            for xs, color, rail in carts:
                x = float(xs[min(int(t * (len(xs) - 1)), len(xs) - 1)])
                pos = rail + RIGHT * (x * 0.95)
                g.add(
                    Rectangle(
                        width=0.42,
                        height=0.32,
                        color=color,
                        fill_color=color,
                        fill_opacity=0.9,
                        stroke_width=0,
                    ).move_to(pos + UP * 0.16)
                )
            return g

        self.add(always_redraw(draw))
        self.wait(0.15)
        self.play(s.animate.set_value(1.0), run_time=7.2, rate_func=linear)
        self.wait(0.7)
