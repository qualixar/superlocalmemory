# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — Track C.2 (LLD-14 §6)

"""Stdlib-only SVG chart export for the evo-memory benchmark.

Every byte of the output is deterministic given deterministic inputs:

- Integer coordinates, fixed ``800x400`` viewBox.
- Child element ordering is explicit, never hash-randomised.
- No timestamps, no embedded fonts (``font-family="monospace"``
  inherits per-viewer metrics — but the XML is identical).
- Uses only ``xml.etree.ElementTree`` — no matplotlib / plotly / any
  external dependency.

LLD-14 §6 locks matplotlib out of the build because its output varies
across versions (font metrics, tick placement) which breaks
bit-identical reproducibility. The chart is intentionally spartan: an
axis frame, labelled ticks, a polyline for the series, and an optional
horizontal gate reference line.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Sequence

# Viewbox constants — fixed forever so stored SVGs diff cleanly across
# runs. Changing any of these is a behaviour change and must bump the
# benchmark schema version (LLD-14 §5.3 ``schema_version``).
_VIEW_W = 800
_VIEW_H = 400
_MARGIN_L = 60
_MARGIN_R = 20
_MARGIN_T = 40
_MARGIN_B = 50
_PLOT_W = _VIEW_W - _MARGIN_L - _MARGIN_R       # 720
_PLOT_H = _VIEW_H - _MARGIN_T - _MARGIN_B       # 310


def _axis_bounds(
    points: Sequence[tuple[int, float]],
    *,
    y_floor: float = 0.0,
    y_ceiling: float = 1.0,
) -> tuple[int, int, float, float]:
    """Return (x_min, x_max, y_min, y_max) snapped to integer x-axis and
    a sane y-range that always contains 0..1 for MRR/Recall style data.

    Deterministic: relies purely on input values, no randomness.
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min = min(xs) if xs else 0
    x_max = max(xs) if xs else 1
    if x_min == x_max:
        x_max = x_min + 1
    y_min = min(y_floor, min(ys) if ys else y_floor)
    y_max = max(y_ceiling, max(ys) if ys else y_ceiling)
    if y_max <= y_min:
        y_max = y_min + 1.0
    return x_min, x_max, y_min, y_max


def _to_pixel(
    x: float, y: float,
    *, x_min: int, x_max: int, y_min: float, y_max: float,
) -> tuple[int, int]:
    """Map data coords → SVG pixel coords (int-snapped, y inverted)."""
    px = _MARGIN_L + int(round((x - x_min) / (x_max - x_min) * _PLOT_W))
    # y grows downward in SVG.
    py = _MARGIN_T + int(
        round((y_max - y) / (y_max - y_min) * _PLOT_H)
    )
    return px, py


def _fmt_num(v: float) -> str:
    """Fixed-precision formatter — never uses scientific notation so the
    string is byte-stable across platforms / Python versions."""
    if abs(v) >= 100.0:
        return f"{v:.0f}"
    if abs(v) >= 10.0:
        return f"{v:.1f}"
    return f"{v:.3f}"


def line_chart_svg(
    points: Sequence[tuple[int, float]],
    *,
    title: str,
    y_label: str,
    gate_line: float | None = None,
) -> str:
    """Return a deterministic SVG string for a single-series line chart.

    Parameters
    ----------
    points:
        Iterable of ``(x_day, y_value)`` pairs, already sorted by day.
    title:
        Chart title, rendered at the top-centre.
    y_label:
        Y-axis label, rendered rotated on the left margin.
    gate_line:
        Optional horizontal reference (e.g. ``day_1 * 1.10`` for the
        10 % publish gate). ``None`` to omit.
    """
    x_min, x_max, y_min, y_max = _axis_bounds(points)

    svg = ET.Element(
        "svg",
        {
            "xmlns": "http://www.w3.org/2000/svg",
            "viewBox": f"0 0 {_VIEW_W} {_VIEW_H}",
            "width": str(_VIEW_W),
            "height": str(_VIEW_H),
            "font-family": "monospace",
            "font-size": "12",
        },
    )

    # Background + frame (stable element order).
    ET.SubElement(svg, "rect", {
        "x": "0", "y": "0",
        "width": str(_VIEW_W), "height": str(_VIEW_H),
        "fill": "white",
    })
    ET.SubElement(svg, "rect", {
        "x": str(_MARGIN_L), "y": str(_MARGIN_T),
        "width": str(_PLOT_W), "height": str(_PLOT_H),
        "fill": "none", "stroke": "#333", "stroke-width": "1",
    })

    # Title
    title_el = ET.SubElement(svg, "text", {
        "x": str(_VIEW_W // 2),
        "y": str(_MARGIN_T - 15),
        "text-anchor": "middle",
        "font-size": "14",
        "font-weight": "bold",
    })
    title_el.text = title

    # Y label (rotated)
    ylabel_el = ET.SubElement(svg, "text", {
        "x": "15",
        "y": str(_MARGIN_T + _PLOT_H // 2),
        "transform": (
            f"rotate(-90 15 {_MARGIN_T + _PLOT_H // 2})"
        ),
        "text-anchor": "middle",
    })
    ylabel_el.text = y_label

    # Y-axis ticks (5 uniform)
    for i in range(6):
        yv = y_min + (y_max - y_min) * i / 5.0
        _, py = _to_pixel(x_min, yv, x_min=x_min, x_max=x_max,
                          y_min=y_min, y_max=y_max)
        ET.SubElement(svg, "line", {
            "x1": str(_MARGIN_L), "y1": str(py),
            "x2": str(_MARGIN_L - 5), "y2": str(py),
            "stroke": "#333", "stroke-width": "1",
        })
        tick_el = ET.SubElement(svg, "text", {
            "x": str(_MARGIN_L - 8),
            "y": str(py + 4),
            "text-anchor": "end",
        })
        tick_el.text = _fmt_num(yv)

    # X-axis ticks at each data point
    for (xv, _) in points:
        px, _ = _to_pixel(xv, y_min, x_min=x_min, x_max=x_max,
                          y_min=y_min, y_max=y_max)
        ET.SubElement(svg, "line", {
            "x1": str(px), "y1": str(_MARGIN_T + _PLOT_H),
            "x2": str(px), "y2": str(_MARGIN_T + _PLOT_H + 5),
            "stroke": "#333", "stroke-width": "1",
        })
        lbl_el = ET.SubElement(svg, "text", {
            "x": str(px),
            "y": str(_MARGIN_T + _PLOT_H + 18),
            "text-anchor": "middle",
        })
        lbl_el.text = f"day{xv}"

    # Gate reference line (dashed)
    if gate_line is not None:
        _, py = _to_pixel(x_min, gate_line, x_min=x_min, x_max=x_max,
                          y_min=y_min, y_max=y_max)
        ET.SubElement(svg, "line", {
            "x1": str(_MARGIN_L),
            "y1": str(py),
            "x2": str(_MARGIN_L + _PLOT_W),
            "y2": str(py),
            "stroke": "#c33", "stroke-width": "1",
            "stroke-dasharray": "6,4",
        })

    # Series polyline
    pixels = [
        _to_pixel(x, y, x_min=x_min, x_max=x_max,
                  y_min=y_min, y_max=y_max)
        for (x, y) in points
    ]
    if pixels:
        ET.SubElement(svg, "polyline", {
            "points": " ".join(f"{p[0]},{p[1]}" for p in pixels),
            "fill": "none", "stroke": "#06c", "stroke-width": "2",
        })
        for (px, py) in pixels:
            ET.SubElement(svg, "circle", {
                "cx": str(px), "cy": str(py),
                "r": "3",
                "fill": "#06c",
            })

    # Deterministic XML serialisation — no declaration, stable attr order
    # comes from ElementTree >=3.8.
    return ET.tostring(svg, encoding="unicode", short_empty_elements=True)


__all__ = ("line_chart_svg",)
