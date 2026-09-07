# -*- coding: utf-8 -*-
"""DH：平面 3R 齐次链，每关节一帧，连乘得到末端。"""
from __future__ import annotations

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Arrow,
    Dot,
    Line,
    Scene,
    VGroup,
    ValueTracker,
    always_redraw,
    smooth,
)

from common import BLUE, EE, GRAY, GREEN, INK, JOINT, ORANGE, RED, ground, label, xy_axes

A_LEN = [0.6, 0.5, 0.35]


def dh(a, theta):
    ct, st = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [ct, -st, 0.0, a * ct],
            [st, ct, 0.0, a * st],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def chain(thetas):
    T = np.eye(4)
    frames = [(T[:2, 3].copy(), T[:2, 0].copy(), T[:2, 1].copy())]
    for a, th in zip(A_LEN, thetas):
        T = T @ dh(a, th)
        frames.append((T[:2, 3].copy(), T[:2, 0].copy(), T[:2, 1].copy()))
    return frames


class DH3R(Scene):
    def construct(self):
        title = label("DH 链：三根平面杆，四参数连乘", 28, BLUE).to_edge(UP, buff=0.22)
        foot = label("每帧：绕 z 转 θ → 沿杆走 a。平面 α = d = 0。", 16, "#4A5568").to_edge(DOWN, buff=0.18)
        origin = LEFT * 3.2 + DOWN * 0.7
        scale = 2.55
        self.add(title, foot, ground(origin), xy_axes(origin, 1.05, 1.0))

        th = [ValueTracker(0.35), ValueTracker(-0.2), ValueTracker(0.4)]
        colors = [BLUE, GREEN, ORANGE]

        def P(p):
            return origin + scale * np.array([float(p[0]), float(p[1]), 0.0])

        def live():
            thetas = [t.get_value() for t in th]
            fr = chain(thetas)
            g = VGroup()
            for i in range(3):
                a, b = P(fr[i][0]), P(fr[i + 1][0])
                g.add(Line(a, b, color=colors[i], stroke_width=12 - 2 * i))
                g.add(Dot(a, radius=0.07, color=JOINT))
            g.add(Dot(P(fr[-1][0]), radius=0.08, color=EE))
            for i, (o, xax, yax) in enumerate(fr):
                o3 = P(o)
                xf = o3 + 0.38 * np.array([xax[0], xax[1], 0])
                yf = o3 + 0.38 * np.array([yax[0], yax[1], 0])
                g.add(Arrow(o3, xf, buff=0, stroke_width=2.5, color=RED, max_tip_length_to_length_ratio=0.28))
                g.add(Arrow(o3, yf, buff=0, stroke_width=2.5, color=GREEN, max_tip_length_to_length_ratio=0.28))
            return g

        arm = always_redraw(live)
        read = always_redraw(
            lambda: label(
                f"θ1={np.rad2deg(th[0].get_value()):.0f}°\n"
                f"θ2={np.rad2deg(th[1].get_value()):.0f}°\n"
                f"θ3={np.rad2deg(th[2].get_value()):.0f}°\n"
                "T = A1 A2 A3",
                20,
                INK,
            )
            .to_edge(RIGHT, buff=0.55)
            .shift(UP * 0.9)
        )
        note = label("红 = 各帧 x 轴\n绿 = 各帧 y 轴", 16, "#4A5568").to_edge(RIGHT, buff=0.55).shift(DOWN * 1.7)
        self.add(arm, read, note)
        self.wait(0.3)
        self.play(th[0].animate.set_value(0.9), run_time=2.0, rate_func=smooth)
        self.play(th[1].animate.set_value(-1.0), run_time=2.0, rate_func=smooth)
        self.play(th[2].animate.set_value(0.8), run_time=1.8, rate_func=smooth)
        self.wait(0.8)
