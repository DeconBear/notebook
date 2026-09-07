# -*- coding: utf-8 -*-
"""Manim 机器人动画共用样式。不进 VitePress，不进 sync-code。"""
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    WHITE,
    Dot,
    Line,
    Text,
    VGroup,
    config,
)
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
TEAL = "#2B6CB0"
LINK = "#4A90A4"
JOINT = "#2D3748"
EE = "#38A169"


def label(text, size=22, color=INK, **kwargs):
    return Text(text, font=FONT, font_size=size, color=color, **kwargs)


def ground(origin, width=0.9):
    base = Line(origin + LEFT * width, origin + RIGHT * width, color="#4A5568", stroke_width=3)
    ticks = VGroup(
        *[
            Line(
                origin + np.array([x, 0, 0]),
                origin + np.array([x - 0.07, -0.12, 0]),
                color="#4A5568",
                stroke_width=2,
            )
            for x in np.linspace(-width, width, 8)
        ]
    )
    return VGroup(base, ticks)


def xy_axes(origin, xlen=1.1, ylen=1.0):
    ax = Line(origin, origin + RIGHT * xlen, color=GRAY, stroke_width=2).add_tip(tip_length=0.12)
    ay = Line(origin, origin + UP * ylen, color=GRAY, stroke_width=2).add_tip(tip_length=0.12)
    return VGroup(
        ax,
        ay,
        label("x", 16, GRAY).next_to(ax, RIGHT, buff=0.05),
        label("y", 16, GRAY).next_to(ay, UP, buff=0.05),
    )


def arm_2r(origin, scale, th1, th2, l1, l2, c1=LINK, c2=LINK):
    p0 = np.array([0.0, 0.0])
    p1 = np.array([l1 * np.cos(th1), l1 * np.sin(th1)])
    p2 = p1 + np.array([l2 * np.cos(th1 + th2), l2 * np.sin(th1 + th2)])

    def P(p):
        return origin + scale * np.array([float(p[0]), float(p[1]), 0.0])

    a, b, c = P(p0), P(p1), P(p2)
    return VGroup(
        Line(a, b, color=c1, stroke_width=11),
        Line(b, c, color=c2, stroke_width=9),
        Dot(a, radius=0.065, color=JOINT),
        Dot(b, radius=0.06, color=JOINT),
        Dot(c, radius=0.07, color=EE),
    ), P(p2)
