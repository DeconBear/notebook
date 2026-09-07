# -*- coding: utf-8 -*-
"""梯度下降：步长太大振荡 vs 合适下山。"""
from __future__ import annotations

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, Dot, Scene, VGroup, ValueTracker, always_redraw, linear

from common import BLUE, GRAY, GREEN, INK, ORANGE, RED, label, polyline, prefix


def loss_grad(w):
    return np.array([2 * (w[0] - 1.0), 0.5 * (w[1] + 0.5)])


def gd(w0, eta, steps):
    w = np.array(w0, dtype=float)
    traj = [w.copy()]
    for _ in range(steps):
        w = w - eta * loss_grad(w)
        traj.append(w.copy())
    return np.array(traj)


class GradDesc(Scene):
    def construct(self):
        w0 = np.array([-1.5, 2.0])
        big = gd(w0, 0.95, 18)
        ok = gd(w0, 0.15, 28)
        self.add(
            label("学习率：太大振荡，合适才下山", 26, BLUE).to_edge(UP, buff=0.18),
            label("L=(w1-1)² + 0.25(w2+0.5)²。星号 = 最优 (1, -0.5)。", 15, "#4A5568").to_edge(DOWN, buff=0.14),
        )
        o = np.array([-0.3, -0.15, 0.0])
        # 参数平面：w1 in [-2.2,2.4], w2 in [-2,2.4]
        def P(w):
            return o + np.array([(w[0] + 0.2) * 1.55, (w[1] - 0.2) * 1.35, 0.0])

        self.add(
            polyline([P([-2.2, 0]), P([2.3, 0])], GRAY, 2),
            polyline([P([0, -2.1]), P([0, 2.3])], GRAY, 2),
        )
        # 等高椭圆
        for c in (0.4, 1.2, 2.4, 4.0, 6.5):
            th = np.linspace(0, 2 * np.pi, 80)
            # (w1-1)^2 + 0.25(w2+0.5)^2 = c
            pts = []
            for a in th:
                dw1 = np.sqrt(c) * np.cos(a)
                dw2 = np.sqrt(c / 0.25) * np.sin(a)
                pts.append(P([1 + dw1, -0.5 + dw2]))
            pts.append(pts[0])
            self.add(polyline(pts, BLUE, 1.5, 0.35))
        self.add(Dot(P([1, -0.5]), radius=0.09, color=INK))
        self.add(label("最优", 14, INK).next_to(P([1, -0.5]), DOWN, buff=0.08))

        path_big = [P(w) for w in big]
        path_ok = [P(w) for w in ok]
        self.add(polyline(path_big, RED, 1.4, 0.2), polyline(path_ok, GREEN, 1.4, 0.2))
        self.add(
            label("η=0.95 过大", 18, RED).to_edge(RIGHT, buff=0.4).shift(UP * 2.15),
            label("η=0.15 合适", 18, GREEN).to_edge(RIGHT, buff=0.4).shift(UP * 1.7),
        )

        s = ValueTracker(0.0)

        def draw():
            t = s.get_value()
            g = VGroup()
            g.add(prefix(path_big, t, RED, 3.2), prefix(path_ok, t, GREEN, 3.2))
            i_b = min(int(t * (len(path_big) - 1)), len(path_big) - 1)
            i_o = min(int(t * (len(path_ok) - 1)), len(path_ok) - 1)
            g.add(Dot(path_big[i_b], radius=0.08, color=RED), Dot(path_ok[i_o], radius=0.08, color=GREEN))
            return g

        self.add(always_redraw(draw))
        self.wait(0.15)
        self.play(s.animate.set_value(1.0), run_time=6.5, rate_func=linear)
        self.wait(0.7)
