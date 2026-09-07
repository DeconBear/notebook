# -*- coding: utf-8 -*-
"""导数：割线窗口收窄，贴成切线。"""
from __future__ import annotations

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, DashedLine, Dot, Line, Scene, VGroup, ValueTracker, always_redraw, linear

from common import BLUE, GRAY, GREEN, INK, ORANGE, RED, label, polyline


def f(x):
    return x ** 3 - 2.0 * x


def fp(x):
    return 3.0 * x ** 2 - 2.0


A0 = 1.2


class SecantTangent(Scene):
    def construct(self):
        self.add(
            label("割线贴成切线：窗口越小，斜率越稳", 26, BLUE).to_edge(UP, buff=0.18),
            label("f(x)=x³-2x，在 x=1.2 处。橙=割线，绿=切线。", 15, "#4A5568").to_edge(DOWN, buff=0.14),
        )
        o = np.array([-4.6, -0.4, 0.0])
        sx, sy = 2.35, 0.85
        xs = np.linspace(-0.3, 2.25, 120)
        curve = [o + np.array([(x + 0.3) * sx, f(x) * sy, 0.0]) for x in xs]
        self.add(
            Line(o, o + RIGHT * 6.2, color=GRAY, stroke_width=2),
            Line(o + DOWN * 0.2, o + UP * 2.8, color=GRAY, stroke_width=2),
            polyline(curve, BLUE, 3.5),
        )
        pa = o + np.array([(A0 + 0.3) * sx, f(A0) * sy, 0.0])
        self.add(Dot(pa, radius=0.07, color=INK))

        # 切线
        xline = np.array([0.2, 2.15])
        tpts = [o + np.array([(x + 0.3) * sx, (f(A0) + fp(A0) * (x - A0)) * sy, 0.0]) for x in xline]
        self.add(DashedLine(tpts[0], tpts[1], color=GREEN, dash_length=0.12, stroke_width=2.5, stroke_opacity=0.35))

        s = ValueTracker(0.0)
        h0, h1 = 0.85, 0.04

        def sec():
            u = s.get_value()
            h = h0 * (1 - u) + h1 * u
            x2 = A0 + h
            p2 = o + np.array([(x2 + 0.3) * sx, f(x2) * sy, 0.0])
            slope = (f(x2) - f(A0)) / h
            g = VGroup(
                Line(pa, p2, color=ORANGE, stroke_width=5),
                Dot(p2, radius=0.07, color=ORANGE),
                label(f"h={h:.2f}   割线斜率 {slope:.2f}   切线 {fp(A0):.2f}", 18, INK).move_to([1.3, 2.55, 0]),
            )
            # 延长割线
            xl = np.array([0.15, 2.2])
            q0 = o + np.array([(xl[0] + 0.3) * sx, (f(A0) + slope * (xl[0] - A0)) * sy, 0.0])
            q1 = o + np.array([(xl[1] + 0.3) * sx, (f(A0) + slope * (xl[1] - A0)) * sy, 0.0])
            g.add(Line(q0, q1, color=ORANGE, stroke_width=2, stroke_opacity=0.45))
            if u > 0.92:
                g.add(Line(tpts[0], tpts[1], color=GREEN, stroke_width=5))
            return g

        self.add(always_redraw(sec))
        self.wait(0.15)
        self.play(s.animate.set_value(1.0), run_time=6.8, rate_func=linear)
        self.wait(0.8)
