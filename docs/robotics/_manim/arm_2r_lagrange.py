# -*- coding: utf-8 -*-
"""
平面 2R 拉格朗日：无阻尼 vs 有阻尼（与 dynamics/code/demo.py 同一套 M,H）。

渲染（在本目录）:
  python -m manim -qm -p --fps 30 arm_2r_lagrange.py TwoRLagrange

-p 渲完用系统播放器打开。不需要 LaTeX。
"""
from __future__ import annotations

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    WHITE,
    Circle,
    Dot,
    Line,
    Rectangle,
    Scene,
    Text,
    TracedPath,
    VGroup,
    ValueTracker,
    config,
    linear,
)

config.background_color = WHITE
config.pixel_width = 1280
config.pixel_height = 720
config.frame_rate = 30

# 与 demo.py 一致
L1 = L2 = 0.8
M1 = M2 = 1.0
G = 9.81
DT = 0.002
STEPS = 2500
DAMP = 1.6
Q0 = np.array([0.3, 0.9])

ARM_SCALE = 2.15
FONT = "Microsoft YaHei"
BLUE = "#2B6CB0"
RED = "#C53030"
GREEN = "#276749"
INK = "#1A202C"
LINK = "#4A90A4"
JOINT = "#2D3748"
EE = "#38A169"
GROUND = "#4A5568"


def mass_matrix(th2: float) -> np.ndarray:
    c2 = np.cos(th2)
    m11 = (M1 + M2) * L1**2 + M2 * L2**2 + 2 * M2 * L1 * L2 * c2
    m12 = M2 * L2**2 + M2 * L1 * L2 * c2
    m22 = M2 * L2**2
    return np.array([[m11, m12], [m12, m22]])


def h_vector(th1: float, th2: float, w1: float, w2: float) -> np.ndarray:
    s2 = np.sin(th2)
    cor = -M2 * L1 * L2 * s2
    g1 = (M1 + M2) * G * L1 * np.cos(th1) + M2 * G * L2 * np.cos(th1 + th2)
    g2 = M2 * G * L2 * np.cos(th1 + th2)
    return np.array([cor * (2 * w1 * w2 + w2**2) + g1, cor * (-(w1**2)) + g2])


def simulate(damp: float) -> np.ndarray:
    q = Q0.copy()
    w = np.zeros(2)
    qs = [q.copy()]
    for _ in range(STEPS):
        m = mass_matrix(q[1])
        h = h_vector(q[0], q[1], w[0], w[1])
        acc = np.linalg.solve(m, -h - damp * w)
        w = w + DT * acc
        q = q + DT * w
        qs.append(q.copy())
    return np.array(qs)


def fk_points(th1: float, th2: float):
    p0 = np.array([0.0, 0.0])
    p1 = np.array([L1 * np.cos(th1), L1 * np.sin(th1)])
    p2 = p1 + np.array([L2 * np.cos(th1 + th2), L2 * np.sin(th1 + th2)])
    return p0, p1, p2


def to_scene(p: np.ndarray, origin) -> np.ndarray:
    return origin + ARM_SCALE * np.array([p[0], p[1], 0.0])


def label(text: str, size: int = 22, color=INK, **kwargs) -> Text:
    return Text(text, font=FONT, font_size=size, color=color, **kwargs)


class TwoRLagrange(Scene):
    def construct(self):
        qs0 = simulate(0.0)
        qs1 = simulate(DAMP)
        n = len(qs0) - 1

        title = label("平面 2R：拉格朗日自由落体　τ = 0", 32, BLUE)
        title.to_edge(UP, buff=0.28)

        left_origin = LEFT * 3.55 + DOWN * 0.55
        right_origin = RIGHT * 3.55 + DOWN * 0.55

        left_box = Rectangle(width=6.6, height=5.6, color="#CBD5E0", stroke_width=1.5)
        left_box.move_to(LEFT * 3.55 + DOWN * 0.15)
        right_box = left_box.copy().move_to(RIGHT * 3.55 + DOWN * 0.15)

        cap_l = label("无阻尼：永远在摆", 22, RED).next_to(left_box, UP, buff=0.12)
        cap_r = label(f"阻尼 c = {DAMP}：螺旋停下", 22, BLUE).next_to(right_box, UP, buff=0.12)

        foot = label(
            "质量在杆端；θ = 0 为水平。左：能量在动能/势能间倒腾；右：摩擦把机械能耗掉。",
            16,
            "#4A5568",
        )
        foot.to_edge(DOWN, buff=0.22)

        def ground(origin):
            w = 1.15
            base = Line(
                origin + LEFT * w,
                origin + RIGHT * w,
                color=GROUND,
                stroke_width=3,
            )
            ticks = VGroup(
                *[
                    Line(
                        origin + np.array([x, 0, 0]),
                        origin + np.array([x - 0.08, -0.14, 0]),
                        color=GROUND,
                        stroke_width=2,
                    )
                    for x in np.linspace(-w, w, 9)
                ]
            )
            return VGroup(base, ticks)

        idx = ValueTracker(0.0)

        def arm_group(qs, origin, trail_color):
            ee = Dot(radius=0.07, color=EE).move_to(to_scene(fk_points(*qs[0])[2], origin))

            def redraw():
                i = int(np.clip(round(idx.get_value()), 0, n))
                p0, p1, p2 = fk_points(*qs[i])
                a, b, c = to_scene(p0, origin), to_scene(p1, origin), to_scene(p2, origin)
                link1 = Line(a, b, color=LINK, stroke_width=12)
                link2 = Line(b, c, color=LINK, stroke_width=10)
                j0 = Circle(radius=0.08, color=JOINT, fill_opacity=1, fill_color=JOINT).move_to(a)
                j1 = Circle(radius=0.07, color=JOINT, fill_opacity=1, fill_color=JOINT).move_to(b)
                j2 = Circle(radius=0.075, color=EE, fill_opacity=1, fill_color=EE, stroke_width=0).move_to(c)
                return VGroup(link1, link2, j0, j1, j2)

            arm = redraw()
            arm.add_updater(lambda m: m.become(redraw()))
            ee.add_updater(
                lambda m: m.move_to(
                    to_scene(fk_points(*qs[int(np.clip(round(idx.get_value()), 0, n))])[2], origin)
                )
            )
            trail = TracedPath(ee.get_center, stroke_color=trail_color, stroke_width=2.5, stroke_opacity=0.85)
            return VGroup(trail, arm, ee)

        left_arm = arm_group(qs0, left_origin, RED)
        right_arm = arm_group(qs1, right_origin, BLUE)

        self.add(
            title,
            left_box,
            right_box,
            cap_l,
            cap_r,
            foot,
            ground(left_origin),
            ground(right_origin),
            left_arm,
            right_arm,
        )
        self.wait(0.4)
        self.play(idx.animate.set_value(float(n)), run_time=6.5, rate_func=linear)
        self.wait(0.8)
