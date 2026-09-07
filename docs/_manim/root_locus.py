# -*- coding: utf-8 -*-
"""根轨迹：K 增大，极点走路，穿过虚轴就不稳。"""
from __future__ import annotations

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, DashedLine, Dot, Line, Scene, VGroup, ValueTracker, always_redraw, linear

from common import BLUE, GRAY, GREEN, INK, ORANGE, PURPLE, RED, label, polyline, prefix

KS = np.linspace(0.0, 40.0, 81)


def roots_of(k):
    return np.roots([1.0, 4.0, 3.0, k])


def track_poles():
    prev = list(roots_of(KS[0]))
    trails = [[r] for r in prev]
    for k in KS[1:]:
        rts = list(roots_of(k))
        used = [False] * 3
        nxt = []
        for p in prev:
            dists = [abs(p - r) if not used[i] else 1e9 for i, r in enumerate(rts)]
            j = int(np.argmin(dists))
            used[j] = True
            nxt.append(rts[j])
        prev = nxt
        for i, r in enumerate(prev):
            trails[i].append(r)
    return trails


class RootLocusWalk(Scene):
    def construct(self):
        self.add(
            label("根轨迹：增益 K 把极点推过虚轴", 26, BLUE).to_edge(UP, buff=0.18),
            label("1 + K / (s(s+1)(s+3)) = 0。叉号 = 开环极点。", 15, "#4A5568").to_edge(DOWN, buff=0.14),
        )
        origin = np.array([-0.4, -0.15, 0.0])
        sx, sy = 1.15, 0.95

        def P(re, im):
            return origin + np.array([(re + 2.0) * sx, im * sy, 0.0])

        self.add(
            Line(P(-4.2, 0), P(1.3, 0), color=GRAY, stroke_width=2),
            Line(P(0, -3.6), P(0, 3.6), color=GRAY, stroke_width=2),
            label("Re", 14, GRAY).next_to(P(1.3, 0), RIGHT, buff=0.06),
            label("Im", 14, GRAY).next_to(P(0, 3.6), UP, buff=0.06),
            label("虚轴", 16, ORANGE).move_to(P(0.85, 3.15)),
        )
        for re, name in ((0.0, "0"), (-1.0, "-1"), (-3.0, "-3")):
            self.add(Dot(P(re, 0), radius=0.07, color=RED), label(name, 14, RED).next_to(P(re, 0), DOWN, buff=0.08))

        raw = track_poles()
        trails = [[P(float(np.real(z)), float(np.imag(z))) for z in tr] for tr in raw]
        colors = [BLUE, GREEN, PURPLE]
        for tr, c in zip(trails, colors):
            self.add(polyline(tr, c, 1.6, 0.22))

        s = ValueTracker(0.0)

        def overlay():
            idx = int(np.clip(s.get_value() * (len(KS) - 1), 0, len(KS) - 1))
            k = float(KS[idx])
            g = VGroup(label(f"K = {k:.0f}", 22, INK).to_edge(RIGHT, buff=0.5).shift(UP * 2.15))
            for tr, c in zip(trails, colors):
                g.add(prefix(tr, s.get_value(), c, 3.2), Dot(tr[idx], radius=0.08, color=c))
            rts = roots_of(k)
            if np.max(np.real(rts)) >= -1e-3:
                g.add(label("穿过虚轴：振荡发散", 20, RED).to_edge(RIGHT, buff=0.35).shift(DOWN * 0.15))
            else:
                g.add(label("全在左半平面：稳定", 20, GREEN).to_edge(RIGHT, buff=0.35).shift(DOWN * 0.15))
            return g

        self.add(always_redraw(overlay))
        self.wait(0.15)
        self.play(s.animate.set_value(1.0), run_time=7.5, rate_func=linear)
        self.wait(0.7)
