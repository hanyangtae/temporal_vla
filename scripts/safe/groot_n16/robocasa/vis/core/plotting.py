"""Plotting helpers: colormaps, labeled scatter, a-vs-b plot, axis bounds, video."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


def categorical_colormap(unique_labels: np.ndarray, fail_label: int | None = None) -> dict[int, tuple]:
    """tab10-based color dict. If fail_label given, that label is dark grey + square marker."""
    cmap = plt.get_cmap("tab10")
    colors: dict[int, tuple] = {}
    for i, c in enumerate(unique_labels):
        c = int(c)
        if fail_label is not None and c == fail_label:
            colors[c] = (0.2, 0.2, 0.2, 1.0)
        else:
            colors[c] = cmap(i % 10)
    return colors


def marker_for(label: int, fail_label: int | None = None) -> str:
    return "s" if (fail_label is not None and label == fail_label) else "o"


def alpha_for(label: int, fail_label: int | None = None, base: float = 0.6, fail_alpha: float = 0.30) -> float:
    return fail_alpha if (fail_label is not None and label == fail_label) else base


def scatter_with_centroids(
    ax,
    P2: np.ndarray,
    labels: np.ndarray,
    colors: dict[int, tuple],
    names: dict[int, str],
    *,
    title: str = "",
    fail_label: int | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    show_ellipse: bool = True,
    point_size: int = 22,
) -> None:
    for c in sorted(np.unique(labels)):
        ci = int(c)
        m = labels == c
        ax.scatter(
            P2[m, 0], P2[m, 1],
            s=point_size, c=[colors[ci]],
            alpha=alpha_for(ci, fail_label),
            marker=marker_for(ci, fail_label),
            label=names.get(ci, str(ci)), linewidths=0,
        )
    for c in sorted(np.unique(labels)):
        ci = int(c)
        m = labels == c
        cx, cy = P2[m, 0].mean(), P2[m, 1].mean()
        if show_ellipse:
            sx, sy = P2[m, 0].std(), P2[m, 1].std()
            ax.add_patch(Ellipse(
                (cx, cy), width=2 * sx, height=2 * sy,
                edgecolor=colors[ci], facecolor="none", lw=1.3, alpha=0.85,
            ))
        ax.scatter(
            [cx], [cy], s=200, marker="X", c=[colors[ci]],
            edgecolors="black", linewidths=1.3, zorder=5,
        )
    if title:
        ax.set_title(title, fontsize=12)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=7, markerscale=2, framealpha=0.85)


def plot_ab(
    ax,
    own: np.ndarray,
    other: np.ndarray,
    labels: np.ndarray,
    colors: dict[int, tuple],
    names: dict[int, str],
    *,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    metric_name: str = "euclidean",
    fail_label: int | None = None,
    point_size: int = 22,
    title_prefix: str = "per-point a vs b",
) -> None:
    for c in np.sort(np.unique(labels)):
        ci = int(c)
        m = labels == c
        ax.scatter(
            own[m], other[m],
            s=point_size, c=[colors[ci]],
            alpha=alpha_for(ci, fail_label),
            marker=marker_for(ci, fail_label),
            label=names.get(ci, str(ci)), linewidths=0,
        )
    hi = float(max(
        (xlim[1] if xlim else float(own.max())),
        (ylim[1] if ylim else float(other.max())),
    ))
    ax.plot([0, hi], [0, hi], "k--", lw=1, label="y = x")
    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(0, hi)
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(0, hi)
    ax.set_xlabel(f"a: {metric_name} dist to own centroid")
    ax.set_ylabel(f"b: {metric_name} dist to nearest other centroid")
    sil_proxy = float(np.mean((other - own) / np.maximum(own, other)))
    ax.set_title(f"{title_prefix} ({metric_name})\nmean (b−a)/max(a,b) = {sil_proxy:+.4f}", fontsize=12)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=7, markerscale=2, framealpha=0.85)


def global_bounds_2d(P2_list: list[np.ndarray], pad_ratio: float = 0.05):
    arr = np.concatenate(P2_list, axis=0)
    lo = arr.min(axis=0); hi = arr.max(axis=0)
    pad = (hi - lo) * pad_ratio
    return (float(lo[0] - pad[0]), float(hi[0] + pad[0])), (float(lo[1] - pad[1]), float(hi[1] + pad[1]))


def global_ab_bounds(own_list: list[np.ndarray], other_list: list[np.ndarray], pad_ratio: float = 0.05):
    hi = max(
        max(o.max() for o in own_list),
        max(o.max() for o in other_list),
    )
    bound = (0.0, float(hi * (1 + pad_ratio)))
    return bound, bound


def make_video(image_paths: list[Path], out_path: Path, fps: float) -> None:
    """Write an mp4 video from a list of PNG frames using OpenCV."""
    import cv2  # imported here so non-video scripts don't need cv2
    first = cv2.imread(str(image_paths[0]))
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    try:
        for p in image_paths:
            img = cv2.imread(str(p))
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(img)
    finally:
        writer.release()
