# -*- coding: utf-8 -*-
"""特征值：单位圆被 A 拉成椭圆，特征向不转弯。"""
from __future__ import annotations

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, Arrow, Dot, Line, Scene, VGroup, ValueTracker, always_redraw, linear

from common import BLUE, GRAY, GREEN, INK, RED, label, polyline

A = np.array([[3.0, 1.0], [1.0, 2.0]])
W, Q = np.linalg.eigh(A)


class EigenStretch(Scene):
    def construct(self):
        self.add(
            label("特征向量：只被拉长，不转弯", 26, BLUE).to_edge(UP, buff=0.18),
            label("A = [[3,1],[1,2]]。红/绿轴 = 特征向。其它向量都被拧走。", 15, "#4A5568").to_edge(DOWN, buff=0.14),
        )
        o = np.array([-0.2, -0.2, 0.0])
        scale = 0.72
        self.add(
            Line(o + LEFT * 3.2, o + RIGHT * 3.2, color=GRAY, stroke_width=2),
            Line(o + DOWN * 2.4, o + UP * 2.4, color=GRAY, stroke_width=2),
        )

        angs = np.linspace(0, 2 * np.pi, 28, endpoint=False)
        circle = np.stack([np.cos(angs), np.sin(angs)], axis=1)

        def apply(t):
            M = (1.0 - t) * np.eye(2) + t * A
            return (M @ circle.T).T

        s = ValueTracker(0.0)

        def grid():
            t = s.get_value()
            pts = apply(t)
            g = VGroup()
            poly = [o + scale * np.array([p[0], p[1], 0.0]) for p in np.vstack([pts, pts[:1]])]
            g.add(polyline(poly, BLUE, 2.5, 0.85))
            for p in pts:
                end = o + scale * np.array([p[0], p[1], 0.0])
                g.add(Line(o, end, color=GRAY, stroke_width=1.5, stroke_opacity=0.45))
            # 特征向
            cols = [RED, GREEN]
            for i, c in enumerate(cols):
                v = Q[:, i]
                M = (1.0 - t) * np.eye(2) + t * A
                w = M @ v
                end = o + scale * np.array([w[0], w[1], 0.0])
                g.add(Arrow(o, end, buff=0, stroke_width=5, color=c, max_tip_length_to_length_ratio=0.12))
            g.add(label(f"t={t:.2f}  I→A", 18, INK).to_edge(RIGHT, buff=0.45).shift(UP * 2.2))
            return g

        self.add(always_redraw(grid))
        self.wait(0.2)
        self.play(s.animate.set_value(1.0), run_time=6.5, rate_func=linear)
        self.wait(0.9)
