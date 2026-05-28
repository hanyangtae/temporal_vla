"""Plotly 3D trace builders + self-contained HTML writer (offline plotly.js)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .geometry import ellipsoid_3d

VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "plotly.min.js"


def scatter3d(xyz, color, name, *, size=3, opacity=0.8, visible=False) -> dict:
    return {"type": "scatter3d", "x": xyz[:, 0].tolist(), "y": xyz[:, 1].tolist(),
            "z": xyz[:, 2].tolist(), "mode": "markers",
            "marker": {"size": size, "color": color, "opacity": opacity},
            "name": name, "visible": visible}


def ellipsoid_surface(pts3, color, name, *, opacity=0.16, visible=False) -> dict:
    ex, ey, ez = ellipsoid_3d(np.asarray(pts3))
    return {"type": "surface", "x": ex.tolist(), "y": ey.tolist(), "z": ez.tolist(),
            "showscale": False, "opacity": opacity, "colorscale": [[0, color], [1, color]],
            "name": name, "visible": visible, "hoverinfo": "skip"}


def plotly_script_tag(online: bool = False) -> str:
    if online or not VENDOR.exists():
        return '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'
    return "<script>" + VENDOR.read_text() + "</script>"


def write_html(out_path: Path, traces, layout, *, online: bool = False, intro: str = "") -> Path:
    tag = plotly_script_tag(online)
    html = (f"<!DOCTYPE html><html><head><meta charset=\"utf-8\"/>\n{tag}\n</head><body>\n"
            f"{intro}\n<div id=\"g\" style=\"width:100%;height:800px;\"></div>\n<script>\n"
            f"var data={json.dumps(traces)};\nvar layout={json.dumps(layout)};\n"
            f"Plotly.newPlot('g',data,layout,{{responsive:true}});\n</script></body></html>")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return out_path
