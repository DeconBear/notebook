# -*- coding: utf-8 -*-
"""
平面 2R：正运动学（一组角→一个点）与逆运动学（两圆相交→两套肘）。
参数与 docs/robotics/kinematics/code/demo.py 一致。

渲染:
  python -m manim -qm -p arm_2r_ikfk.py ForwardInverse2R
"""
from __future__ import annotations

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UL,
    UP,
    UR,
    DL,
    WHITE,
    Arc,
    Circle,
    Create,
    DashedVMobject,
    Dot,
    FadeIn,
    FadeOut,
    Line,
    Scene,
    Star,
    Text,
    VGroup,
    ValueTracker,
    always_redraw,
    config,
    linear,
    smooth,
)

config.background_color = WHITE
config.pixel_width = 1280
config.pixel_height = 720
config.frame_rate = 30

L1, L2 = 1.0, 0.7
TARGET = np.array([1.1, 0.6])
SCALE = 2.32
FONT = "Microsoft YaHei"
INK = "#1A202C"
BLUE = "#2B6CB0"
GREEN = "#276749"
ORANGE = "#C05621"
RED = "#C53030"
GRAY = "#A0AEC0"
LINK1 = "#2B6CB0"
LINK2 = "#38A169"
UP_C = "#27AE60"
DOWN_C = "#DD6B20"


def fk(th1, th2):
    return np.array(
        [
            L1 * np.cos(th1) + L2 * np.cos(th1 + th2),
            L1 * np.sin(th1) + L2 * np.sin(th1 + th2),
        ]
    )


def joints(th1, th2):
    p0 = np.array([0.0, 0.0])
    p1 = np.array([L1 * np.cos(th1), L1 * np.sin(th1)])
    return p0, p1, fk(th1, th2)


def ik(x, y, elbow="up"):
    r2 = x * x + y * y
    c2 = (r2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if abs(c2) > 1.0 + 1e-9:
        return None
    c2 = float(np.clip(c2, -1.0, 1.0))
    s2 = np.sqrt(max(0.0, 1.0 - c2 * c2))
    if elbow == "down":
        s2 = -s2
    th2 = np.arctan2(s2, c2)
    k1 = L1 + L2 * c2
    k2 = L2 * s2
    th1 = np.arctan2(y, x) - np.arctan2(k2, k1)
    return th1, th2


def label(text, size=22, color=INK, **kwargs):
    return Text(text, font=FONT, font_size=size, color=color, **kwargs)


class ForwardInverse2R(Scene):
    def construct(self):
        origin = LEFT * 3.45 + DOWN * 0.55

        def P(p):
            return origin + SCALE * np.array([float(p[0]), float(p[1]), 0.0])

        def ground():
            w = 0.85
            base = Line(origin + LEFT * w, origin + RIGHT * w, color="#4A5568", stroke_width=3)
            ticks = VGroup(
                *[
                    Line(
                        origin + np.array([x, 0, 0]),
                        origin + np.array([x - 0.07, -0.12, 0]),
                        color="#4A5568",
                        stroke_width=2,
                    )
                    for x in np.linspace(-w, w, 8)
                ]
            )
            ax = Line(origin, origin + RIGHT * 1.15, color=GRAY, stroke_width=2).add_tip(
                tip_length=0.12
            )
            ay = Line(origin, origin + UP * 1.05, color=GRAY, stroke_width=2).add_tip(
                tip_length=0.12
            )
            return VGroup(
                base,
                ticks,
                ax,
                ay,
                label("x", 18, GRAY).next_to(ax, RIGHT, buff=0.06),
                label("y", 18, GRAY).next_to(ay, UP, buff=0.06),
            )

        def make_arm(th1, th2, c1=LINK1, c2=LINK2, show_angles=False):
            p0, p1, p2 = joints(th1, th2)
            a, b, c = P(p0), P(p1), P(p2)
            g = VGroup(
                Line(a, b, color=c1, stroke_width=12),
                Line(b, c, color=c2, stroke_width=10),
                Dot(a, radius=0.07, color="#2D3748"),
                Dot(b, radius=0.065, color="#2D3748"),
                Dot(c, radius=0.075, color="#38A169"),
            )
            if show_angles and abs(th1) > 1e-3:
                g.add(Arc(radius=0.42, start_angle=0, angle=th1, color=RED, stroke_width=3).move_arc_center_to(a))
            if show_angles and abs(th2) > 1e-3:
                ext = b + 0.55 * np.array([np.cos(th1), np.sin(th1), 0.0])
                g.add(DashedVMobject(Line(b, ext, color=GRAY, stroke_width=2), num_dashes=8))
                g.add(
                    Arc(radius=0.36, start_angle=th1, angle=th2, color=ORANGE, stroke_width=3).move_arc_center_to(b)
                )
            return g

        title = label("平面 2R：正运动学与逆运动学", 30, BLUE).to_edge(UP, buff=0.22)
        badge = label("1  正运动学：给定关节角，手在哪？", 22, BLUE)
        badge.next_to(title, DOWN, buff=0.12)
        foot = label("ℓ₁ = 1.0，ℓ₂ = 0.7（与运动学 demo 相同）", 16, "#4A5568").to_edge(
            DOWN, buff=0.18
        )
        world = ground()
        self.add(title, badge, foot, world)

        th1 = ValueTracker(0.0)
        th2 = ValueTracker(0.0)

        arm = always_redraw(lambda: make_arm(th1.get_value(), th2.get_value(), show_angles=True))

        def readout():
            a, b = th1.get_value(), th2.get_value()
            p = fk(a, b)
            t = label(
                f"θ1 = {np.rad2deg(a):.0f}°\nθ2 = {np.rad2deg(b):.0f}°\n手 = ({p[0]:.2f}, {p[1]:.2f})",
                20,
                INK,
            )
            t.to_edge(RIGHT, buff=0.55).shift(UP * 1.35)
            return t

        info = always_redraw(readout)
        note_fk = label("先转杆 1，再相对杆 1 转杆 2。\n一组角只对应一个点。", 18, "#4A5568")
        note_fk.to_edge(RIGHT, buff=0.45).shift(DOWN * 1.55)

        self.add(arm, info, note_fk)
        self.wait(0.25)
        self.play(th1.animate.set_value(np.deg2rad(48)), run_time=2.4, rate_func=smooth)
        self.wait(0.25)
        self.play(th2.animate.set_value(np.deg2rad(62)), run_time=2.4, rate_func=smooth)
        uniq = label("正运动学永远唯一", 20, GREEN)
        uniq.next_to(note_fk, UP, buff=0.2)
        self.play(FadeIn(uniq), run_time=0.4)
        self.wait(0.7)

        self.play(
            FadeOut(arm),
            FadeOut(info),
            FadeOut(note_fk),
            FadeOut(uniq),
            run_time=0.45,
        )
        th1.set_value(0)
        th2.set_value(0)

        badge2 = label("2  逆运动学：给定目标点，肘怎么弯？", 22, ORANGE)
        badge2.move_to(badge)
        self.play(FadeOut(badge), FadeIn(badge2), run_time=0.4)

        tgt = P(TARGET)
        star = Star(n=5, outer_radius=0.13, inner_radius=0.055, color=RED, fill_opacity=1).move_to(tgt)
        tgt_txt = label("目标 (1.1, 0.6)", 18, RED).next_to(star, UR, buff=0.08)
        diag = DashedVMobject(Line(origin, tgt, color=GRAY, stroke_width=2), num_dashes=16)

        c_inner = DashedVMobject(
            Circle(radius=SCALE * L1, color=BLUE, stroke_width=2).move_to(origin),
            num_dashes=36,
        )
        c_hand = DashedVMobject(
            Circle(radius=SCALE * L2, color=ORANGE, stroke_width=2).move_to(tgt),
            num_dashes=28,
        )
        cap_c1 = label("肘在以基座为心、ℓ₁ 为半径的圆上", 16, BLUE)
        cap_c2 = label("肘也在以目标为心、ℓ₂ 为半径的圆上", 16, ORANGE)
        cap_c1.to_edge(RIGHT, buff=0.4).shift(UP * 1.55)
        cap_c2.next_to(cap_c1, DOWN, aligned_edge=LEFT, buff=0.12)

        self.play(FadeIn(star), FadeIn(tgt_txt), Create(diag), run_time=0.6)
        self.play(Create(c_inner), FadeIn(cap_c1), run_time=0.9)
        self.play(Create(c_hand), FadeIn(cap_c2), run_time=0.9)

        sol_u = ik(*TARGET, "up")
        sol_d = ik(*TARGET, "down")
        eu = Dot(P(joints(*sol_u)[1]), radius=0.08, color=UP_C)
        ed = Dot(P(joints(*sol_d)[1]), radius=0.08, color=DOWN_C)
        lu = label("肘上", 18, UP_C).next_to(eu, UL, buff=0.08)
        ld = label("肘下", 18, DOWN_C).next_to(ed, DL, buff=0.08)
        cross = label("两圆相交 = 两套逆解", 18, INK)
        cross.next_to(cap_c2, DOWN, aligned_edge=LEFT, buff=0.18)

        self.play(FadeIn(eu), FadeIn(ed), FadeIn(lu), FadeIn(ld), FadeIn(cross), run_time=0.7)

        arm_u = make_arm(*sol_u, c1=UP_C, c2=UP_C)
        arm_d = make_arm(*sol_d, c1=DOWN_C, c2=DOWN_C)
        arm_d.set_opacity(0.45)
        self.play(FadeIn(arm_u), run_time=0.55)
        self.play(FadeIn(arm_d), run_time=0.55)

        same = label("手都在同一个点上，角却不同。", 18, GREEN)
        same.next_to(cross, DOWN, aligned_edge=LEFT, buff=0.16)
        self.play(FadeIn(same), run_time=0.4)
        self.wait(0.55)

        self.play(
            FadeOut(c_inner),
            FadeOut(c_hand),
            FadeOut(cap_c1),
            FadeOut(cap_c2),
            FadeOut(cross),
            FadeOut(diag),
            FadeOut(eu),
            FadeOut(ed),
            FadeOut(lu),
            FadeOut(ld),
            FadeOut(arm_u),
            FadeOut(arm_d),
            FadeOut(same),
            FadeOut(tgt_txt),
            run_time=0.45,
        )

        follow = label("目标移动时，两支逆解跟着走；出圆环则无解。", 18, "#4A5568")
        follow.to_edge(RIGHT, buff=0.4).shift(UP * 1.4)
        ring_out = DashedVMobject(
            Circle(radius=SCALE * (L1 + L2), color=GRAY, stroke_width=1.5).move_to(origin),
            num_dashes=48,
        )
        ring_in = DashedVMobject(
            Circle(radius=SCALE * abs(L1 - L2), color=GRAY, stroke_width=1.5).move_to(origin),
            num_dashes=20,
        )
        ring_lab = label("工作空间：|ℓ₁−ℓ₂| ≤ r ≤ ℓ₁+ℓ₂", 16, GRAY)
        ring_lab.next_to(follow, DOWN, aligned_edge=LEFT, buff=0.14)

        tx = ValueTracker(float(TARGET[0]))
        ty = ValueTracker(float(TARGET[1]))

        def live():
            x, y = tx.get_value(), ty.get_value()
            g = VGroup()
            su, sd = ik(x, y, "up"), ik(x, y, "down")
            if su is not None:
                g.add(make_arm(*su, c1=UP_C, c2=UP_C))
            if sd is not None:
                ad = make_arm(*sd, c1=DOWN_C, c2=DOWN_C)
                ad.set_opacity(0.5)
                g.add(ad)
            g.add(Star(n=5, outer_radius=0.13, inner_radius=0.055, color=RED, fill_opacity=1).move_to(P([x, y])))
            r = np.hypot(x, y)
            status = (
                label(f"r = {r:.2f}  两解", 18, GREEN)
                if su is not None
                else label(f"r = {r:.2f}  无解（圆环外）", 18, RED)
            )
            status.to_edge(RIGHT, buff=0.4).shift(DOWN * 0.4)
            g.add(status)
            return g

        live_grp = always_redraw(live)
        self.add(ring_out, ring_in, follow, ring_lab, live_grp)
        self.remove(star)
        self.wait(0.2)
        self.play(tx.animate.set_value(1.45), ty.animate.set_value(0.35), run_time=2.4, rate_func=linear)
        self.play(tx.animate.set_value(1.85), ty.animate.set_value(0.12), run_time=2.2, rate_func=linear)
        self.wait(1.1)
