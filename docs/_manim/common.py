# -*- coding: utf-8 -*-
"""跨领域 Manim 短片共用样式。目录名 _manim，不进 VitePress。"""
from manim import WHITE, Dot, Line, Text, VGroup, VMobject, config
import numpy as np

config.background_color = WHITE
config.pixel_width = 1280
config.pixel_height = 720
config.frame_rate = 30

FONT = "Microsoft YaHei"
INK = "#1A202C"
BLUE = "#2B6CB0"
GREEN = "#276749"
ORANGE = "#C05621"
RED = "#C53030"
GRAY = "#A0AEC0"
PURPLE = "#6B46C1"


def label(text, size=22, color=INK, **kwargs):
    return Text(text, font=FONT, font_size=size, color=color, **kwargs)


def polyline(points, color, width, opacity=1.0):
    m = VMobject()
    m.set_points_as_corners([np.array(p, dtype=float) for p in points])
    m.set_stroke(color, width=width, opacity=opacity)
    return m


def prefix(points, frac, color, width=3.0):
    frac = float(np.clip(frac, 0.0, 1.0))
    if frac <= 1e-4 or len(points) < 2:
        return VGroup()
    n = len(points) - 1
    u = frac * n
    i = min(int(np.floor(u)), n - 1)
    a = u - i
    last = (1.0 - a) * np.array(points[i]) + a * np.array(points[i + 1])
    corners = list(points[: i + 1]) + [last]
    if len(corners) < 2:
        corners = [points[0], last]
    return polyline(corners, color, width)


def axes_xy(origin, xlen, ylen, xlabel="x", ylabel="y"):
    from manim import RIGHT, UP

    ax = Line(origin, origin + RIGHT * xlen, color=GRAY, stroke_width=2).add_tip(tip_length=0.1)
    ay = Line(origin, origin + UP * ylen, color=GRAY, stroke_width=2).add_tip(tip_length=0.1)
    return VGroup(
        ax,
        ay,
        label(xlabel, 14, GRAY).next_to(ax, RIGHT, buff=0.05),
        label(ylabel, 14, GRAY).next_to(ay, UP, buff=0.05),
    )
