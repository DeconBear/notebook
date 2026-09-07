# -*- coding: utf-8 -*-
"""机构学：曲柄摇杆，偶联点扫封闭曲线。"""
from __future__ import annotations

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Dot,
    Line,
    Scene,
    TracedPath,
    VGroup,
    ValueTracker,
    always_redraw,
    linear,
)

from common import BLUE, EE, GRAY, GREEN, INK, JOINT, ORANGE, RED, ground, label

L0, L1, L2, L3 = 1.4, 0.45, 1.1, 0.9


def coupler(theta):
    A = np.array([L1 * np.cos(theta), L1 * np.sin(theta)])
    D = np.array([L0, 0.0])
    d = D - A
    dist = np.linalg.norm(d)
    if dist < 1e-9 or dist > L2 + L3 or dist < abs(L2 - L3):
        return None
    a = (L2**2 - L3**2 + dist**2) / (2 * dist)
    h = np.sqrt(max(L2**2 - a * a, 0.0))
    mid = A + a * d / dist
    n = np.array([-d[1], d[0]]) / dist
    B = mid + h * n
    P = 0.5 * (A + B)
    return A, B, D, P


class FourBar(Scene):
    def construct(self):
        title = label("铰链四杆：曲柄转一圈，偶联点画封闭曲线", 26, BLUE).to_edge(UP, buff=0.2)
        foot = label("自由度 1。蓝曲柄整周，紫摇杆只摆。橙点 = 连杆中点。", 15, "#4A5568").to_edge(DOWN, buff=0.16)
        origin = LEFT * 3.6 + DOWN * 1.35
        scale = 2.15

        def P(p):
            return origin + scale * np.array([float(p[0]), float(p[1]), 0.0])

        self.add(title, foot, ground(origin, 0.7))
        self.add(Dot(P([0, 0]), radius=0.07, color=JOINT), Dot(P([L0, 0]), radius=0.07, color=JOINT))
        self.add(Line(P([0, 0]), P([L0, 0]), color=GRAY, stroke_width=4))

        th = ValueTracker(0.3)
        p_dot = Dot(radius=0.08, color=ORANGE)

        def mech():
            sol = coupler(th.get_value())
            if sol is None:
                sol = coupler(0.3)
            A, B, D, Pt = sol
            g = VGroup(
                Line(P([0, 0]), P(A), color=BLUE, stroke_width=8),
                Line(P(A), P(B), color=GREEN, stroke_width=8),
                Line(P(B), P(D), color="#6B46C1", stroke_width=8),
                Dot(P(A), radius=0.06, color=JOINT),
                Dot(P(B), radius=0.06, color=JOINT),
            )
            p_dot.move_to(P(Pt))
            return g

        arm = always_redraw(mech)
        trail = TracedPath(p_dot.get_center, stroke_color=ORANGE, stroke_width=3)
        note = label("M = 3(N−1)−2J₁ = 1", 18, INK).to_edge(RIGHT, buff=0.45).shift(UP * 1.6)
        self.add(trail, arm, p_dot, note)
        self.wait(0.2)
        self.play(th.animate.set_value(0.3 + 2 * np.pi), run_time=7.5, rate_func=linear)
        self.wait(0.6)
