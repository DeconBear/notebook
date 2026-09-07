# -*- coding: utf-8 -*-
"""雅可比：伸直时两列速度变成平行。"""
from __future__ import annotations

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, Arrow, Scene, VGroup, ValueTracker, always_redraw, linear

from common import BLUE, GRAY, GREEN, INK, ORANGE, RED, arm_2r, ground, label

L1, L2 = 1.0, 0.7


class JacobianSing(Scene):
    def construct(self):
        self.add(
            label("雅可比列 = 各关节给手的速度方向", 24, BLUE).to_edge(UP, buff=0.16),
            label("伸直时两支箭头平行，det J=0，少了一个可动方向。", 15, "#4A5568").to_edge(DOWN, buff=0.12),
        )
        o = LEFT * 2.6 + DOWN * 0.55
        scale = 2.15
        self.add(ground(o, 0.9))
        s = ValueTracker(0.0)
        # θ2: 1.15 -> 0.02  伸直
        th1 = 0.55

        def jac(th1, th2):
            s1, c1 = np.sin(th1), np.cos(th1)
            s12, c12 = np.sin(th1 + th2), np.cos(th1 + th2)
            return np.array(
                [
                    [-L1 * s1 - L2 * s12, -L2 * s12],
                    [L1 * c1 + L2 * c12, L2 * c12],
                ]
            )

        def draw():
            t = s.get_value()
            th2 = 1.15 * (1 - t) + 0.02 * t
            arm, ee = arm_2r(o, scale, th1, th2, L1, L2)
            J = jac(th1, th2)
            g = VGroup(arm)
            # 列向量画在末端
            for col, color, name in ((0, ORANGE, "J1 基座"), (1, GREEN, "J2 肘")):
                v = J[:, col]
                nrm = np.linalg.norm(v) + 1e-9
                v = 0.85 * v / nrm
                end = ee + scale * 0.55 * np.array([v[0], v[1], 0.0])
                g.add(Arrow(ee, end, buff=0, color=color, stroke_width=5, max_tip_length_to_length_ratio=0.18))
            det = abs(np.linalg.det(J))
            g.add(label(f"θ2={th2:.2f}   |det J|={det:.2f}", 20, INK).to_edge(RIGHT, buff=0.4).shift(UP * 2.0))
            if det < 0.12:
                g.add(label("奇异：两箭头平行", 20, RED).to_edge(RIGHT, buff=0.4).shift(UP * 1.45))
            else:
                g.add(label("两列指向不同", 20, GREEN).to_edge(RIGHT, buff=0.4).shift(UP * 1.45))
            return g

        self.add(always_redraw(draw))
        self.wait(0.2)
        self.play(s.animate.set_value(1.0), run_time=6.5, rate_func=linear)
        self.wait(0.8)
