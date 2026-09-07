# -*- coding: utf-8 -*-
"""频域：正弦进植物，输出落后；Bode 上同步走 ω。"""
from __future__ import annotations

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, DashedLine, Dot, Line, Scene, VGroup, ValueTracker, always_redraw, linear

from common import BLUE, GRAY, INK, ORANGE, PURPLE, RED, label, polyline

WN, ZETA, KGAIN = 4.0, 0.3, 8.0
W_GRID = np.logspace(-1, 2, 240)


def G_jw(omega):
    s = 1j * omega
    return KGAIN * (WN ** 2) / (s ** 2 + 2.0 * ZETA * WN * s + WN ** 2)


MAG = 20.0 * np.log10(np.abs(G_jw(W_GRID)))
PH = np.angle(G_jw(W_GRID), deg=True)
WC = float(W_GRID[np.where((MAG[:-1] > 0) & (MAG[1:] <= 0))[0][0]])


class BodeSine(Scene):
    def construct(self):
        self.add(
            label("正弦扫植物：幅值掉下来，相位落后", 26, BLUE).to_edge(UP, buff=0.16),
            label(f"KG(s)，K={KGAIN:g}，ωn={WN:g}，ζ={ZETA}。橙线 = 0 dB 穿越。", 15, "#4A5568").to_edge(DOWN, buff=0.12),
        )
        # 左：时间正弦
        left = np.array([-5.35, -0.2, 0.0])
        self.add(label("时间域：进 / 出", 16, INK).move_to([-3.6, 2.45, 0]))
        # 右：Bode 幅值（横轴 log ω 映射到线性像素）
        br = np.array([0.35, -0.9, 0.0])
        bw, bh = 5.6, 3.4
        self.add(label("幅频 (dB)", 16, INK).move_to([3.2, 2.45, 0]))
        self.add(
            Line(br, br + [bw, 0, 0], color=GRAY, stroke_width=2),
            Line(br, br + [0, bh, 0], color=GRAY, stroke_width=2),
        )
        db0 = 0.0
        db_min, db_max = -25.0, 25.0

        def x_of_w(w):
            lw = np.log10(w)
            return (lw - (-1.0)) / (2.0 - (-1.0)) * bw

        def y_of_db(db):
            return (db - db_min) / (db_max - db_min) * bh

        mag_pts = [br + np.array([x_of_w(w), y_of_db(db), 0.0]) for w, db in zip(W_GRID, MAG)]
        self.add(polyline(mag_pts, PURPLE, 2.5))
        y0 = br + np.array([0, y_of_db(db0), 0])
        self.add(DashedLine(y0, y0 + [bw, 0, 0], color=GRAY, dash_length=0.1))
        xc = br + np.array([x_of_w(WC), 0, 0])
        self.add(DashedLine(xc, xc + [0, bh, 0], color=ORANGE, dash_length=0.1))
        self.add(label("0 dB", 13, GRAY).next_to(y0 + [bw, 0, 0], RIGHT, buff=0.05))
        self.add(label("ωc", 14, ORANGE).next_to(xc + [0, bh, 0], UP, buff=0.04))

        s = ValueTracker(0.0)
        # ω from 0.4 to 25 log
        w0, w1 = 0.5, 25.0

        def panel():
            u = s.get_value()
            w = w0 * (w1 / w0) ** u
            g = G_jw(w)
            mag, ph = np.abs(g), np.angle(g)
            gvg = VGroup()
            # 两周期正弦
            n = 120
            t = np.linspace(0, 4 * np.pi, n)
            xin = np.sin(t)
            yout = mag * np.sin(t + ph)
            def to_left(xs, ys, yoff):
                pts = []
                for a, b in zip(xs, ys):
                    pts.append(left + np.array([(a / (4 * np.pi)) * 3.6, yoff + 0.35 * b, 0.0]))
                return pts
            tin = np.linspace(0, 1, n)
            pin = [left + np.array([ti * 3.7, 1.15 + 0.55 * np.sin(4 * np.pi * ti), 0]) for ti in tin]
            pout = [
                left + np.array([ti * 3.7, -0.55 + 0.55 * np.clip(mag / 8.0, 0.15, 1.0) * np.sin(4 * np.pi * ti + ph), 0])
                for ti in tin
            ]
            gvg.add(polyline(pin, BLUE, 3.0), polyline(pout, ORANGE, 3.0))
            gvg.add(label("输入", 14, BLUE).move_to(left + np.array([4.2, 1.15, 0])))
            gvg.add(label("输出", 14, ORANGE).move_to(left + np.array([4.2, -0.55, 0])))
            db = 20 * np.log10(mag + 1e-12)
            gvg.add(Dot(br + np.array([x_of_w(w), y_of_db(db), 0]), radius=0.08, color=ORANGE))
            gvg.add(label(f"ω={w:.1f}  相位 {np.degrees(ph):.0f}°", 16, INK).move_to([3.2, -2.55, 0]))
            return gvg

        self.add(always_redraw(panel))
        self.wait(0.15)
        self.play(s.animate.set_value(1.0), run_time=7.5, rate_func=linear)
        self.wait(0.6)
