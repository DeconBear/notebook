# -*- coding: utf-8 -*-
"""导论：构型空间直线映成工作空间香蕉弯。"""
from __future__ import annotations

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Circle,
    DashedVMobject,
    Dot,
    Line,
    Scene,
    TracedPath,
    VGroup,
    ValueTracker,
    always_redraw,
    linear,
)

from common import BLUE, GRAY, INK, ORANGE, RED, arm_2r, ground, label, xy_axes

L1, L2 = 1.0, 0.7
TH_A = np.array([-0.4, 1.2])
TH_B = np.array([1.1, -0.6])


class CspaceWorkspace(Scene):
    def construct(self):
        title = label("工作空间 ≠ 构型空间", 30, BLUE).to_edge(UP, buff=0.22)
        foot = label("θ 空间走直线，手在圆环里走出香蕉弯。", 16, "#4A5568").to_edge(DOWN, buff=0.18)
        self.add(title, foot)

        left_o = LEFT * 3.5 + DOWN * 0.35
        scale = 2.05
        ws_box_title = label("工作空间（手在哪）", 20, BLUE).move_to(LEFT * 3.5 + UP * 2.55)
        cs_box_title = label("构型空间（关节角）", 20, ORANGE).move_to(RIGHT * 3.4 + UP * 2.55)
        self.add(ws_box_title, cs_box_title, ground(left_o), xy_axes(left_o, 1.0, 0.95))

        ring_out = DashedVMobject(
            Circle(radius=scale * (L1 + L2), color=GRAY, stroke_width=1.5).move_to(left_o),
            num_dashes=40,
        )
        ring_in = DashedVMobject(
            Circle(radius=scale * abs(L1 - L2), color=GRAY, stroke_width=1.5).move_to(left_o),
            num_dashes=16,
        )
        self.add(ring_out, ring_in)

        # 构型空间坐标
        cs_o = RIGHT * 2.15 + DOWN * 1.35
        cs_sx, cs_sy = 1.55, 1.35
        ax1 = Line(cs_o, cs_o + RIGHT * 3.3, color=GRAY, stroke_width=2).add_tip(tip_length=0.12)
        ay1 = Line(cs_o, cs_o + UP * 3.0, color=GRAY, stroke_width=2).add_tip(tip_length=0.12)
        self.add(
            ax1,
            ay1,
            label("θ1", 16, GRAY).next_to(ax1, RIGHT, buff=0.05),
            label("θ2", 16, GRAY).next_to(ay1, UP, buff=0.05),
        )

        def q_of(s):
            return TH_A + s * (TH_B - TH_A)

        def cs_pt(q):
            return cs_o + np.array([(q[0] + 0.6) * cs_sx, (q[1] + 0.8) * cs_sy, 0.0])

        line_cs = Line(cs_pt(TH_A), cs_pt(TH_B), color=RED, stroke_width=3)
        self.add(line_cs, Dot(cs_pt(TH_A), color=ORANGE), Dot(cs_pt(TH_B), color=ORANGE, radius=0.08))

        s = ValueTracker(0.0)
        ee_dot = Dot(radius=0.02, color=RED, fill_opacity=0)

        def live():
            q = q_of(s.get_value())
            arm, ee = arm_2r(left_o, scale, q[0], q[1], L1, L2)
            ee_dot.move_to(ee)
            return VGroup(arm, Dot(cs_pt(q), color=RED, radius=0.09))

        grp = always_redraw(live)
        trail = TracedPath(ee_dot.get_center, stroke_color=RED, stroke_width=3)
        note = label("左：圆环带里手的路径\n右：θ1-θ2 平面上的直线", 16, "#4A5568")
        note.to_edge(RIGHT, buff=0.35).shift(DOWN * 2.55)
        self.add(trail, grp, ee_dot, note)
        self.wait(0.2)
        self.play(s.animate.set_value(1.0), run_time=6.5, rate_func=linear)
        self.wait(0.7)
