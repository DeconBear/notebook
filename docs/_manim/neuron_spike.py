# -*- coding: utf-8 -*-
"""LIF 积分到阈值开火；HH 阶跃电流出一个尖峰。"""
from __future__ import annotations

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, DashedLine, Dot, Scene, VGroup, ValueTracker, always_redraw, linear

from common import BLUE, GRAY, GREEN, INK, ORANGE, RED, label, polyline, prefix

# ---- 精简 HH / LIF（与 demo 同参数）----
def _alpha_n(V):
    x = 10.0 - (V + 65.0)
    return 0.1 if abs(x) < 1e-6 else 0.01 * x / (np.exp(x / 10.0) - 1.0)


def _beta_n(V):
    return 0.125 * np.exp(-(V + 65.0) / 80.0)


def _alpha_m(V):
    x = 25.0 - (V + 65.0)
    return 1.0 if abs(x) < 1e-6 else 0.1 * x / (np.exp(x / 10.0) - 1.0)


def _beta_m(V):
    return 4.0 * np.exp(-(V + 65.0) / 18.0)


def _alpha_h(V):
    return 0.07 * np.exp(-(V + 65.0) / 20.0)


def _beta_h(V):
    return 1.0 / (np.exp((30.0 - (V + 65.0)) / 10.0) + 1.0)


def sim_hh():
    dt, n = 0.02, int(50 / 0.02)
    I = np.zeros(n)
    I[int(10 / dt) : int(40 / dt)] = 10.0
    V = -65.0
    an, bn = _alpha_n(V), _beta_n(V)
    am, bm = _alpha_m(V), _beta_m(V)
    ah, bh = _alpha_h(V), _beta_h(V)
    nG, m, h = an / (an + bn), am / (am + bm), ah / (ah + bh)
    vs = np.empty(n)
    C_m, g_Na, g_K, g_L = 1.0, 120.0, 36.0, 0.3
    E_Na, E_K, E_L = 50.0, -77.0, -54.387
    for i in range(n):
        I_Na = g_Na * (m ** 3) * h * (V - E_Na)
        I_K = g_K * (nG ** 4) * (V - E_K)
        I_L = g_L * (V - E_L)
        V = V + dt * (I[i] - I_Na - I_K - I_L) / C_m
        nG = nG + dt * (_alpha_n(V) * (1 - nG) - _beta_n(V) * nG)
        m = m + dt * (_alpha_m(V) * (1 - m) - _beta_m(V) * m)
        h = h + dt * (_alpha_h(V) * (1 - h) - _beta_h(V) * h)
        vs[i] = V
    return vs


def sim_lif():
    dt, tau, Rm = 0.1, 20.0, 20.0
    V_rest, V_th, t_ref = -70.0, -50.0, 2.0
    n = int(200 / dt)
    I = np.zeros(n)
    I[int(20 / dt) :] = 1.5
    V, ref, vs = V_rest, 0.0, np.empty(n)
    for i in range(n):
        if ref > 0:
            ref -= dt
            V = V_rest
        else:
            V = V + dt * (-(V - V_rest) + Rm * I[i]) / tau
            if V >= V_th:
                V = V_rest
                ref = t_ref
        vs[i] = V
    return vs


def pack(vs, bl, w, h, vmin, vmax):
    return [
        bl + np.array([(i / (len(vs) - 1)) * w, (float(vs[i]) - vmin) / (vmax - vmin) * h, 0.0])
        for i in range(len(vs))
    ]


class NeuronSpike(Scene):
    def construct(self):
        self.add(
            label("LIF 积分到阈值；HH 有尖峰形状", 26, BLUE).to_edge(UP, buff=0.16),
            label("左：恒流 1.5 nA。右：10–40 ms 注入 10 μA/cm²。", 15, "#4A5568").to_edge(DOWN, buff=0.12),
        )
        lif, hh = sim_lif(), sim_hh()
        panels = [
            ("LIF", lif, GREEN, np.array([-5.7, -1.7, 0.0]), -80.0, -40.0, -50.0),
            ("HH", hh, BLUE, np.array([0.45, -1.7, 0.0]), -80.0, 50.0, None),
        ]
        series = []
        for name, vs, color, bl, vmin, vmax, th in panels:
            w, h = 5.15, 3.55
            self.add(label(name, 20, color).move_to(bl + np.array([2.4, h + 0.55, 0])))
            self.add(polyline([bl, bl + [w, 0, 0], bl + [w, h, 0], bl + [0, h, 0], bl], GRAY, 1.5))
            if th is not None:
                y = (th - vmin) / (vmax - vmin) * h
                self.add(DashedLine(bl + [0, y, 0], bl + [w, y, 0], color=ORANGE, dash_length=0.1))
                self.add(label("阈值", 13, ORANGE).next_to(bl + [w, y, 0], RIGHT, buff=0.05))
            pts = pack(vs, bl, w, h, vmin, vmax)
            self.add(polyline(pts, color, 1.3, 0.2))
            series.append((pts, color))

        s = ValueTracker(0.0)

        def draw():
            g = VGroup()
            for pts, c in series:
                g.add(prefix(pts, s.get_value(), c, 3.2))
                i = min(int(s.get_value() * (len(pts) - 1)), len(pts) - 1)
                g.add(Dot(pts[i], radius=0.06, color=c))
            return g

        self.add(always_redraw(draw))
        self.wait(0.12)
        self.play(s.animate.set_value(1.0), run_time=7.2, rate_func=linear)
        self.wait(0.6)
