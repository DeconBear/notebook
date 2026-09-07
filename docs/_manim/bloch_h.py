# -*- coding: utf-8 -*-
"""布洛赫球：H 把 |0> 转到赤道，测量再塌到南北极。"""
from __future__ import annotations

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, Arrow, Circle, DashedLine, Dot, Line, Scene, VGroup, ValueTracker, always_redraw, linear

from common import BLUE, GRAY, GREEN, INK, ORANGE, RED, label


def bloch(theta, phi=0.0):
    # 侧视：x 向右，z 向上（y 忽略，H 走 xz 平面）
    x = np.sin(theta) * np.cos(phi)
    z = np.cos(theta)
    return np.array([x, z])


class BlochH(Scene):
    def construct(self):
        self.add(
            label("H 门：从北极走到赤道，测量才塌缩", 26, BLUE).to_edge(UP, buff=0.16),
            label("|0> 在北极。H|0>=|+> 在 +x。测 Z 以 50/50 回南北极。", 15, "#4A5568").to_edge(DOWN, buff=0.12),
        )
        o = np.array([-2.4, -0.15, 0.0])
        r = 2.15
        self.add(
            Circle(radius=r, color=GRAY, stroke_width=2).move_to(o),
            DashedLine(o + LEFT * r, o + RIGHT * r, color=GRAY, dash_length=0.1),
            Line(o + DOWN * r, o + UP * r, color=GRAY, stroke_width=2),
            label("|0>", 18, INK).next_to(o + UP * r, UP, buff=0.08),
            label("|1>", 18, INK).next_to(o + DOWN * r, DOWN, buff=0.08),
            label("|+>", 18, GREEN).next_to(o + RIGHT * r, RIGHT, buff=0.08),
        )
        s = ValueTracker(0.0)
        # 0-0.55 rotate theta 0 -> pi/2
        # 0.55-0.75 hold
        # 0.75-1.0 collapse: split ghost to both poles, then pick |0>

        def arrow():
            u = s.get_value()
            g = VGroup()
            if u <= 0.62:
                th = (u / 0.62) * (np.pi / 2)
                v = bloch(th)
                end = o + r * np.array([v[0], v[1], 0.0])
                g.add(Arrow(o, end, buff=0, color=BLUE, stroke_width=6, max_tip_length_to_length_ratio=0.12))
                g.add(label("叠加：还没测", 18, BLUE).move_to([3.4, 1.6, 0]))
            else:
                # 两个浅箭头 + 测量塌到 |0>
                e0 = o + r * np.array([0, 1, 0.0])
                e1 = o + r * np.array([0, -1, 0.0])
                g.add(
                    Arrow(o, e0, buff=0, color=GREEN, stroke_width=4, stroke_opacity=0.35, max_tip_length_to_length_ratio=0.12),
                    Arrow(o, e1, buff=0, color=RED, stroke_width=4, stroke_opacity=0.35, max_tip_length_to_length_ratio=0.12),
                )
                pick = e0 if u < 0.88 else e0
                g.add(Arrow(o, pick, buff=0, color=GREEN, stroke_width=6, max_tip_length_to_length_ratio=0.12))
                g.add(label("测 Z：50/50 塌到 |0> 或 |1>", 18, ORANGE).move_to([3.55, 1.6, 0]))
                g.add(label("这一次 → |0>", 18, GREEN).move_to([3.4, 1.05, 0]))
            return g

        self.add(always_redraw(arrow))
        self.wait(0.2)
        self.play(s.animate.set_value(1.0), run_time=6.8, rate_func=linear)
        self.wait(0.8)
