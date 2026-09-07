# -*- coding: utf-8 -*-
"""李群：旋转乘法不交换。用二维轴测投影画两套三轴，避免 3D 渲染过慢。"""
from __future__ import annotations

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Arrow,
    Scene,
    VGroup,
    ValueTracker,
    always_redraw,
    linear,
)

from common import BLUE, FONT, GREEN, INK, RED, label


def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def project(p):
    x, y, z = p
    return np.array([x - 0.42 * z, y + 0.28 * z, 0.0])


def triad_at(origin, R, length=1.45):
    axes = np.eye(3)
    colors = [RED, GREEN, BLUE]
    g = VGroup()
    for i in range(3):
        end = origin + length * project(R @ axes[i])
        g.add(Arrow(origin, end, buff=0, stroke_width=4, color=colors[i], max_tip_length_to_length_ratio=0.18))
    labs = ["x", "y", "z"]
    for i, lab in enumerate(labs):
        end = origin + (length + 0.18) * project(R @ axes[i])
        g.add(label(lab, 16, colors[i]).move_to(end))
    return g


class RotationOrder(Scene):
    def construct(self):
        title = label("旋转不能当向量加：乘法不交换", 28, "#2B6CB0").to_edge(UP, buff=0.22)
        foot = label("左：先 Rz 再 Rx。右：先 Rx 再 Rz。红 x / 绿 y / 蓝 z。", 16, "#4A5568").to_edge(
            DOWN, buff=0.18
        )
        self.add(title, foot)
        self.add(
            label("先绕 z 90°，再绕 x 90°", 18, RED).move_to(LEFT * 3.5 + UP * 2.45),
            label("先绕 x 90°，再绕 z 90°", 18, GREEN).move_to(RIGHT * 3.5 + UP * 2.45),
        )

        o1 = LEFT * 3.5 + DOWN * 0.35
        o2 = RIGHT * 3.5 + DOWN * 0.35
        a = ValueTracker(0.0)

        def left():
            t = a.get_value()
            if t <= 1:
                R = Rz(t * np.pi / 2)
            else:
                R = Rx((t - 1) * np.pi / 2) @ Rz(np.pi / 2)
            return triad_at(o1, R)

        def right():
            t = a.get_value()
            if t <= 1:
                R = Rx(t * np.pi / 2)
            else:
                R = Rz((t - 1) * np.pi / 2) @ Rx(np.pi / 2)
            return triad_at(o2, R)

        self.add(always_redraw(left), always_redraw(right))
        self.wait(0.25)
        self.play(a.animate.set_value(1.0), run_time=2.2, rate_func=linear)
        self.wait(0.25)
        self.play(a.animate.set_value(2.0), run_time=2.2, rate_func=linear)
        done = label("终点姿态不同，所以欧拉角不能当 R³ 加减。", 20, GREEN)
        done.next_to(foot, UP, buff=0.12)
        self.play(done.animate.set_opacity(1), run_time=0.01)
        self.add(done)
        self.wait(1.0)
