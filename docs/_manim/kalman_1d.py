# -*- coding: utf-8 -*-
"""卡尔曼：预测胀、测量缩。"""
from __future__ import annotations

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, DashedLine, Dot, Line, Rectangle, Scene, VGroup, ValueTracker, always_redraw, linear

from common import BLUE, GRAY, GREEN, INK, ORANGE, RED, label, polyline, prefix

np.random.seed(42)
N, Q, Rmeas = 48, 0.04, 0.25


def run_kf():
    x, xhat, P = 0.0, 0.0, 1.0
    xs, zs, xhs, Ps = [x], [], [xhat], [P]
    for _ in range(N):
        x = x + np.random.normal(0.0, np.sqrt(Q))
        z = x + np.random.normal(0.0, np.sqrt(Rmeas))
        P_pred = P + Q
        kg = P_pred / (P_pred + Rmeas)
        xhat = xhat + kg * (z - xhat)
        P = (1.0 - kg) * P_pred
        xs.append(x)
        zs.append(z)
        xhs.append(xhat)
        Ps.append(P)
    return np.array(xs), np.array(zs), np.array(xhs), np.array(Ps)


class KalmanBand(Scene):
    def construct(self):
        xs, zs, xhs, Ps = run_kf()
        self.add(
            label("卡尔曼：预测把云撑大，测量再捏回去", 26, BLUE).to_edge(UP, buff=0.16),
            label("黑=真值，灰点=测量，绿=估计。带子 = ±2√P。", 15, "#4A5568").to_edge(DOWN, buff=0.12),
        )
        bl = np.array([-5.6, -0.35, 0.0])
        w, h = 11.2, 3.35
        ymin, ymax = -2.2, 2.2
        self.add(
            Line(bl, bl + [w, 0, 0], color=GRAY, stroke_width=2),
            Line(bl, bl + [0, h, 0], color=GRAY, stroke_width=2),
            label("k", 14, GRAY).next_to(bl + [w, 0, 0], RIGHT, buff=0.05),
            label("x", 14, GRAY).next_to(bl + [0, h, 0], UP, buff=0.05),
        )

        def Y(val):
            return (val - ymin) / (ymax - ymin) * h

        def X(i, n=len(xs) - 1):
            return (i / n) * w

        true_pts = [bl + np.array([X(i), Y(xs[i]), 0]) for i in range(len(xs))]
        est_pts = [bl + np.array([X(i), Y(xhs[i]), 0]) for i in range(len(xhs))]
        self.add(polyline(true_pts, INK, 1.5, 0.2), polyline(est_pts, GREEN, 1.5, 0.2))

        s = ValueTracker(0.0)

        def draw():
            n = len(xs) - 1
            k = int(s.get_value() * n)
            g = VGroup()
            g.add(prefix(true_pts, s.get_value(), INK, 2.8), prefix(est_pts, s.get_value(), GREEN, 3.0))
            for i in range(k):
                g.add(Dot(bl + np.array([X(i + 1), Y(zs[i]), 0]), radius=0.035, color=GRAY, fill_opacity=0.7))
            # 当前不确定带子
            sig = 2.0 * np.sqrt(max(Ps[k], 1e-6))
            cx = bl + np.array([X(k), Y(xhs[k]), 0])
            half = sig / (ymax - ymin) * h
            g.add(
                Line(cx + [0, half, 0], cx + [0, -half, 0], color=GREEN, stroke_width=8, stroke_opacity=0.35),
                Dot(bl + np.array([X(k), Y(xs[k]), 0]), radius=0.07, color=INK),
                Dot(cx, radius=0.07, color=GREEN),
            )
            if k > 0:
                g.add(Dot(bl + np.array([X(k), Y(zs[k - 1]), 0]), radius=0.055, color=ORANGE))
            return g

        self.add(always_redraw(draw))
        self.wait(0.12)
        self.play(s.animate.set_value(1.0), run_time=7.8, rate_func=linear)
        self.wait(0.6)
