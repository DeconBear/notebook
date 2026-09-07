# -*- coding: utf-8 -*-
"""旋量：平面刚体绕固定瞬心拧过去（SE(2) 指数映射）。"""
from __future__ import annotations

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    UR,
    Dot,
    Polygon,
    Scene,
    VGroup,
    ValueTracker,
    always_redraw,
    linear,
)

from common import BLUE, GRAY, INK, ORANGE, RED, label, xy_axes


def se2_exp(omega, vx, vy, t=1.0):
    if abs(omega) < 1e-10:
        T = np.eye(3)
        T[0, 2] = vx * t
        T[1, 2] = vy * t
        return T
    w = omega * t
    c, s = np.cos(w), np.sin(w)
    trans = (1.0 / omega) * np.array([[s, -(1 - c)], [1 - c, s]]) @ np.array([vx, vy])
    T = np.eye(3)
    T[0, 0], T[0, 1], T[0, 2] = c, -s, trans[0]
    T[1, 0], T[1, 1], T[1, 2] = s, c, trans[1]
    return T


def apply(T, pts):
    h = np.c_[pts, np.ones(len(pts))]
    return (T @ h.T).T[:, :2]


class SE2Screw(Scene):
    def construct(self):
        title = label("平面旋量：绕固定瞬心「拧」过去", 28, BLUE).to_edge(UP, buff=0.22)
        foot = label("Chasles：有限位移 = 绕轴转 + 沿轴移。平面 demo 取纯转动，h = 0。", 15, "#4A5568").to_edge(
            DOWN, buff=0.16
        )
        origin = LEFT * 0.2 + DOWN * 0.4
        scale = 2.8

        def P(p):
            return origin + scale * np.array([float(p[0]), float(p[1]), 0.0])

        omega = 1.2
        q = np.array([0.8, 0.2])
        v = omega * np.array([q[1], -q[0]])
        square = np.array([[0.15, 0.1], [0.35, 0.1], [0.35, 0.28], [0.15, 0.28]])

        self.add(title, foot, xy_axes(origin, 1.4, 1.25))
        self.add(Dot(P(q), radius=0.09, color=RED), label("瞬心", 18, RED).next_to(P(q), UR, buff=0.08))

        t = ValueTracker(0.0)
        ghosts = VGroup()

        def body():
            T = se2_exp(omega, v[0], v[1], t.get_value())
            pts = apply(T, square)
            poly = Polygon(*[P(p) for p in pts], color=BLUE, fill_color="#BEE3F8", fill_opacity=0.7, stroke_width=3)
            return poly

        rect = always_redraw(body)
        note = label("ξ = (ω, vx, vy)\nT(t) = exp(t ξ̂)", 18, INK).to_edge(RIGHT, buff=0.5).shift(UP * 1.3)
        self.add(rect, note)
        self.wait(0.2)
        # 留下几帧残影
        for snap in (0.0, 0.4, 0.8, 1.2):
            t.set_value(snap)
            T = se2_exp(omega, v[0], v[1], snap)
            pts = apply(T, square)
            ghosts.add(
                Polygon(
                    *[P(p) for p in pts],
                    color=GRAY,
                    stroke_width=1.5,
                    fill_opacity=0.0,
                )
            )
        t.set_value(0.0)
        self.add(ghosts)
        self.play(t.animate.set_value(1.6), run_time=6.5, rate_func=linear)
        self.wait(0.7)
