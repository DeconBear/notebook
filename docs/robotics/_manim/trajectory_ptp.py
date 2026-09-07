# -*- coding: utf-8 -*-
"""轨迹：同一起终点，关节线性 vs 三次。路径相同，快慢不同。"""
from __future__ import annotations

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    DashedLine,
    DashedVMobject,
    Dot,
    Line,
    Rectangle,
    Scene,
    VGroup,
    VMobject,
    ValueTracker,
    always_redraw,
    linear,
)

from common import BLUE, GRAY, INK, ORANGE, RED, arm_2r, ground, label

L1, L2 = 1.0, 0.7
Q0 = np.array([-0.5, 1.4])
Q1 = np.array([1.2, -0.4])
T = 1.0
DQ = Q1 - Q0
DQ_NORM = float(np.linalg.norm(DQ))


def smoothstep(s):
    return s * s * (3.0 - 2.0 * s)


def joint(u):
    return Q0 + u * DQ


def qdot_lin(_s=None):
    return DQ / T


def qdot_cub(s):
    return (6.0 * s * (1.0 - s) / T) * DQ


def jac(th1, th2):
    s1, c1 = np.sin(th1), np.cos(th1)
    s12, c12 = np.sin(th1 + th2), np.cos(th1 + th2)
    return np.array(
        [
            [-L1 * s1 - L2 * s12, -L2 * s12],
            [L1 * c1 + L2 * c12, L2 * c12],
        ]
    )


def speed_q(kind, s):
    return DQ_NORM / T if kind == "lin" else abs(6.0 * s * (1.0 - s) / T) * DQ_NORM


def speed_ee(kind, s):
    u = s if kind == "lin" else smoothstep(s)
    q = joint(u)
    qd = qdot_lin() if kind == "lin" else qdot_cub(s)
    return float(np.linalg.norm(jac(q[0], q[1]) @ qd))


def polyline(points, color, width, opacity=1.0):
    m = VMobject()
    m.set_points_as_corners(points)
    m.set_stroke(color, width=width, opacity=opacity)
    return m


def prefix(points, frac, color, width=3.5):
    frac = float(np.clip(frac, 0.0, 1.0))
    if frac <= 1e-4:
        return VGroup()
    n = len(points) - 1
    u = frac * n
    i = min(int(np.floor(u)), n - 1)
    a = u - i
    last = (1.0 - a) * points[i] + a * points[i + 1]
    corners = list(points[: i + 1]) + [last]
    if len(corners) < 2:
        corners = [points[0], last]
    return polyline(corners, color, width)


def fk_pts(origin, scale, n=120):
    pts = []
    for u in np.linspace(0.0, 1.0, n):
        q = joint(u)
        _, ee = arm_2r(origin, scale, q[0], q[1], L1, L2)
        pts.append(np.array(ee, dtype=float))
    return pts


def curve_pts(fn, bl, w, h, ymax, n=80):
    pts = []
    for s in np.linspace(0.0, 1.0, n):
        pts.append(bl + np.array([s * w, fn(s) / ymax * h, 0.0]))
    return pts


class JointTrajectory(Scene):
    def construct(self):
        title = label("关节空间 PTP：同一条弯路，两种时刻表", 26, BLUE).to_edge(UP, buff=0.16)
        self.add(title)
        self.add(
            label("线性：θ 匀速", 18, RED).move_to(LEFT * 3.55 + UP * 2.62),
            label("三次：起停速度 0", 18, BLUE).move_to(RIGHT * 3.55 + UP * 2.62),
        )

        o_l = LEFT * 3.55 + UP * 0.22
        o_r = RIGHT * 3.55 + UP * 0.22
        scale = 1.48
        self.add(ground(o_l, 0.72), ground(o_r, 0.72))

        path_l = fk_pts(o_l, scale)
        path_r = fk_pts(o_r, scale)
        ghost_l = DashedVMobject(polyline(path_l, GRAY, 2.0, 0.55), num_dashes=28)
        ghost_r = DashedVMobject(polyline(path_r, GRAY, 2.0, 0.55), num_dashes=28)
        self.add(ghost_l, ghost_r)

        # 底部双行速度图：浅色是全程，深色跟着时间画出来
        ss = np.linspace(0.0, 1.0, 80)
        ymax_q = 4.0
        ymax_v = 1.15 * max(max(speed_ee("lin", s) for s in ss), max(speed_ee("cub", s) for s in ss), 0.4)

        panel = Rectangle(
            width=11.4,
            height=2.28,
            fill_color="#F7FAFC",
            fill_opacity=1,
            stroke_color="#E2E8F0",
            stroke_width=1.5,
        ).move_to(DOWN * 2.45)
        self.add(panel)
        self.add(label("速度随时间（曲线跟着手臂一起画）", 16, INK).next_to(panel, UP, buff=0.05))

        w, h = 7.2, 0.70
        bl_q = np.array([-3.85, -2.08, 0.0])
        bl_v = np.array([-3.85, -3.10, 0.0])
        pts_q_cub = curve_pts(lambda u: speed_q("cub", u), bl_q, w, h, ymax_q)
        pts_v_lin = curve_pts(lambda u: speed_ee("lin", u), bl_v, w, h, ymax_v)
        pts_v_cub = curve_pts(lambda u: speed_ee("cub", u), bl_v, w, h, ymax_v)

        def axes(bl, ymax_label):
            return VGroup(
                Line(bl, bl + RIGHT * w, color=GRAY, stroke_width=2),
                Line(bl, bl + UP * h, color=GRAY, stroke_width=2),
                label(ymax_label, 13, GRAY).next_to(bl + UP * h, LEFT, buff=0.08),
            )

        y_lin_q = speed_q("lin", 0) / ymax_q * h
        self.add(
            axes(bl_q, "关节 |q'|"),
            axes(bl_v, "末端 |v|"),
            label("0", 13, GRAY).next_to(bl_v, DOWN, buff=0.04),
            label("t = T", 13, GRAY).next_to(bl_v + RIGHT * w, DOWN, buff=0.04),
            DashedLine(
                bl_q + UP * y_lin_q,
                bl_q + RIGHT * w + UP * y_lin_q,
                color=RED,
                stroke_width=1.5,
                dash_length=0.10,
                stroke_opacity=0.28,
            ),
            polyline(pts_q_cub, BLUE, 1.8, 0.25),
            polyline(pts_v_lin, RED, 1.8, 0.25),
            polyline(pts_v_cub, BLUE, 1.8, 0.25),
            label("红=线性", 14, RED).move_to(RIGHT * 4.55 + DOWN * 1.78),
            label("蓝=三次", 14, BLUE).move_to(RIGHT * 4.55 + DOWN * 2.12),
        )

        s = ValueTracker(0.0)

        def grown_lin_q():
            t = s.get_value()
            if t <= 1e-4:
                return VGroup()
            return DashedLine(
                bl_q + UP * y_lin_q,
                bl_q + RIGHT * (t * w) + UP * y_lin_q,
                color=RED,
                stroke_width=3.0,
                dash_length=0.12,
            )

        self.add(
            always_redraw(grown_lin_q),
            always_redraw(lambda: prefix(pts_q_cub, s.get_value(), BLUE, 3.2)),
            always_redraw(lambda: prefix(pts_v_lin, s.get_value(), RED, 3.0)),
            always_redraw(lambda: prefix(pts_v_cub, s.get_value(), BLUE, 3.2)),
        )

        def panel_arm(kind, origin, path, color):
            def pose():
                u = s.get_value() if kind == "lin" else smoothstep(s.get_value())
                q = joint(u)
                return arm_2r(origin, scale, q[0], q[1], L1, L2, c1=color, c2=color)

            arm = always_redraw(lambda: pose()[0])
            trail = always_redraw(
                lambda: prefix(
                    path,
                    s.get_value() if kind == "lin" else smoothstep(s.get_value()),
                    color,
                )
            )
            return VGroup(trail, arm)

        def cursors():
            t = s.get_value()
            x = t * w
            vline = Line(
                bl_q + RIGHT * x + UP * h,
                bl_v + RIGHT * x,
                color=ORANGE,
                stroke_width=2,
            )
            dots = VGroup(
                Dot(bl_q + np.array([x, speed_q("lin", t) / ymax_q * h, 0.0]), radius=0.055, color=RED),
                Dot(bl_q + np.array([x, speed_q("cub", t) / ymax_q * h, 0.0]), radius=0.055, color=BLUE),
                Dot(bl_v + np.array([x, speed_ee("lin", t) / ymax_v * h, 0.0]), radius=0.05, color=RED),
                Dot(bl_v + np.array([x, speed_ee("cub", t) / ymax_v * h, 0.0]), radius=0.05, color=BLUE),
            )
            return VGroup(vline, dots)

        self.add(panel_arm("lin", o_l, path_l, RED), panel_arm("cub", o_r, path_r, BLUE))
        self.add(always_redraw(cursors))
        self.wait(0.2)
        self.play(s.animate.set_value(1.0), run_time=7.0, rate_func=linear)
        self.wait(0.9)
