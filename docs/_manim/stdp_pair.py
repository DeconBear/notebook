# -*- coding: utf-8 -*-
"""STDP：先 pre 后 post 加强；反过来减弱。"""
from __future__ import annotations

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, Dot, Line, Rectangle, Scene, VGroup, ValueTracker, always_redraw, linear

from common import BLUE, GRAY, GREEN, INK, ORANGE, RED, label, polyline

A_PLUS, A_MINUS = 0.01, 0.012
TAU = 20.0


def dw(dt):
    if dt > 0:
        return A_PLUS * np.exp(-dt / TAU)
    if dt < 0:
        return -A_MINUS * np.exp(dt / TAU)
    return 0.0


class STDPPair(Scene):
    def construct(self):
        self.add(
            label("谁先放，权重往哪边走", 26, BLUE).to_edge(UP, buff=0.16),
            label("Δt = t_post − t_pre。左：pre 先 → LTP。右：post 先 → LTD。", 15, "#4A5568").to_edge(DOWN, buff=0.12),
        )
        # 学习窗
        dts = np.linspace(-80, 80, 200)
        bl = np.array([-5.6, -2.55, 0.0])
        w, h = 4.6, 2.15
        pts = []
        for d in dts:
            x = (d + 80) / 160 * w
            y = (dw(d) / 0.013 + 1) / 2 * h
            pts.append(bl + np.array([x, y, 0]))
        self.add(
            label("STDP 窗", 16, INK).move_to([-3.3, 0.05, 0]),
            polyline([bl, bl + [w, 0, 0], bl + [w, h, 0], bl + [0, h, 0], bl], GRAY, 1.4),
            polyline(pts, BLUE, 2.5),
            Line(bl + [w / 2, 0, 0], bl + [w / 2, h, 0], color=GRAY, stroke_width=1.5),
            Line(bl + [0, h / 2, 0], bl + [w, h / 2, 0], color=GRAY, stroke_width=1.5),
        )

        def neuron_pair(cx, title, color):
            self.add(label(title, 18, color).move_to([cx, 2.55, 0]))
            pre = np.array([cx - 0.9, 1.35, 0.0])
            post = np.array([cx + 0.9, 1.35, 0.0])
            self.add(
                Dot(pre, radius=0.16, color=BLUE),
                Dot(post, radius=0.16, color=ORANGE),
                label("pre", 14, BLUE).next_to(pre, DOWN, buff=0.12),
                label("post", 14, ORANGE).next_to(post, DOWN, buff=0.12),
                Line(pre, post, color=GRAY, stroke_width=6),
            )
            bar_bl = np.array([cx - 1.15, -0.15, 0.0])
            self.add(polyline([bar_bl, bar_bl + [2.3, 0, 0], bar_bl + [2.3, 0.28, 0], bar_bl + [0, 0.28, 0], bar_bl], GRAY, 1.2))
            return pre, post, bar_bl, cx

        left = neuron_pair(1.55, "先 pre 后 post  +10ms", GREEN)
        right = neuron_pair(4.85, "先 post 后 pre  −10ms", RED)

        s = ValueTracker(0.0)

        def draw():
            t = s.get_value()
            g = VGroup()
            # 左：pre at 0.15, post at 0.40
            def flashes(pre, post, t_pre, t_post, bar_bl, sign):
                gg = VGroup()
                if abs(t - t_pre) < 0.08:
                    gg.add(Dot(pre, radius=0.22, color=BLUE, fill_opacity=0.35))
                if abs(t - t_post) < 0.08:
                    gg.add(Dot(post, radius=0.22, color=ORANGE, fill_opacity=0.35))
                # 权重从 0.4 走
                prog = float(np.clip((t - max(t_pre, t_post)) / 0.55, 0, 1))
                w0, w1 = 0.4, (0.85 if sign > 0 else 0.12)
                ww = w0 + prog * (w1 - w0)
                gg.add(
                    Rectangle(
                        width=2.25 * ww,
                        height=0.22,
                        color=GREEN if sign > 0 else RED,
                        fill_color=GREEN if sign > 0 else RED,
                        fill_opacity=0.9,
                        stroke_width=0,
                    ).move_to(bar_bl + np.array([1.15 * ww, 0.14, 0]))
                )
                return gg

            g.add(flashes(left[0], left[1], 0.18, 0.38, left[2], +1))
            g.add(flashes(right[0], right[1], 0.38, 0.18, right[2], -1))
            # 窗上的 Δt 标记
            dlt = 10 if t < 0.55 else -10
            # 两个标记点
            def mark(dtms, c):
                x = (dtms + 80) / 160 * 4.6
                y = (dw(dtms) / 0.013 + 1) / 2 * 2.15
                return Dot(bl + np.array([x, y, 0]), radius=0.07, color=c)

            g.add(mark(10, GREEN), mark(-10, RED))
            return g

        self.add(always_redraw(draw))
        self.wait(0.2)
        self.play(s.animate.set_value(1.0), run_time=6.5, rate_func=linear)
        self.wait(0.7)
