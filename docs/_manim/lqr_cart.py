# -*- coding: utf-8 -*-
"""LQR：双积分器小车，无控制飞走 vs 拉回原点。"""
from __future__ import annotations

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, DashedLine, Rectangle, Scene, VGroup, ValueTracker, always_redraw, linear

from common import BLUE, GRAY, GREEN, INK, RED, label, polyline, prefix

DT, N = 0.05, 80
A = np.array([[1.0, DT], [0.0, 1.0]])
B = np.array([[0.5 * DT ** 2], [DT]])
Q = np.diag([4.0, 0.2])
R = np.array([[0.15]])


def dare_lqr():
    P = Q.copy()
    for _ in range(200):
        BtP = B.T @ P
        P = Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.solve(R + BtP @ B, BtP @ A)
    return np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)


def simulate(use_lqr):
    K = dare_lqr()
    x = np.array([-1.2, 0.8])
    xs = [x.copy()]
    for _ in range(N):
        u = float((-K @ x).item()) if use_lqr else 0.0
        x = A @ x + B.ravel() * u
        xs.append(x.copy())
    return np.array(xs)


class LQRCart(Scene):
    def construct(self):
        xs0, xs1 = simulate(False), simulate(True)
        self.add(
            label("LQR：把双积分器拉回原点", 26, BLUE).to_edge(UP, buff=0.18),
            label("ẍ = u。左：u=0 飞走；右：u=-Kx 一次给完增益。", 15, "#4A5568").to_edge(DOWN, buff=0.14),
        )
        panels = [("无控制", xs0, RED, -3.5), ("LQR  u=-Kx", xs1, GREEN, 3.5)]
        rails, plots = [], []
        for name, xs, color, cx in panels:
            self.add(label(name, 20, color).move_to([cx, 2.5, 0]))
            rail = np.array([cx, 0.55, 0.0])
            self.add(polyline([rail + LEFT * 2.4, rail + RIGHT * 2.4], GRAY, 4))
            self.add(DashedLine(rail + UP * 0.55, rail + DOWN * 0.25, color=GRAY, dash_length=0.08))
            self.add(label("0", 14, GRAY).next_to(rail, DOWN, buff=0.12))
            rails.append((xs, color, rail))
            bl = np.array([cx - 2.3, -2.35, 0.0])
            w, h = 4.6, 1.45
            self.add(polyline([bl, bl + [w, 0, 0], bl + [w, h, 0], bl + [0, h, 0], bl], GRAY, 1.4))
            pos = xs[:, 0]
            pmin, pmax = -2.2, 2.2
            pts = [
                bl + np.array([(i / (len(pos) - 1)) * w, (pos[i] - pmin) / (pmax - pmin) * h, 0])
                for i in range(len(pos))
            ]
            self.add(polyline(pts, color, 1.3, 0.22))
            plots.append((pts, color))

        s = ValueTracker(0.0)

        def draw():
            t = s.get_value()
            g = VGroup()
            for pts, color in plots:
                g.add(prefix(pts, t, color, 3.0))
            for xs, color, rail in rails:
                i = min(int(t * (len(xs) - 1)), len(xs) - 1)
                p = float(np.clip(xs[i, 0], -2.0, 2.0))
                g.add(
                    Rectangle(width=0.7, height=0.38, color=color, fill_color=color, fill_opacity=0.9, stroke_width=0).move_to(
                        rail + RIGHT * (p * 1.05) + UP * 0.22
                    )
                )
            return g

        self.add(always_redraw(draw))
        self.wait(0.12)
        self.play(s.animate.set_value(1.0), run_time=6.8, rate_func=linear)
        self.wait(0.7)
