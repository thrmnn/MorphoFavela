"""Preset-switchable matplotlib styling for paper vs. slide-deck figures.

Two presets:

* ``paper`` — matplotlib defaults (no-op). Lets existing per-figure scripts
  keep whatever local sizing/spacing they already define.
* ``presentation`` — bumps every text element to projection-readable sizes
  (axis labels ≥ 14 pt, ticks ≥ 13 pt, legend ≥ 13 pt, colorbar ticks ≥ 12 pt)
  and adopts a slightly heavier line/spine weight that survives a projector
  with mediocre contrast.

Map-axis helpers (``apply_to_map_axes``, ``add_scale_bar``) are tiny enough
to inline without pulling matplotlib-scalebar or contextily. They are
preset-agnostic — call them only when ``--preset presentation`` is active.

Usage::

    from src.viz import presentation_style as ps
    ps.apply(args.preset)
    # ...
    if args.preset == "presentation":
        ps.apply_to_map_axes(ax)
        ps.add_scale_bar(ax, 200)
        cbar.set_label(ps.short_label("lambda_f_mean"))
"""

from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

__all__ = [
    "add_scale_bar",
    "apply",
    "apply_to_map_axes",
    "short_label",
]

Preset = Literal["paper", "presentation"]

# Conservative pass-through map. Keys: long-form column names that show up in
# raw GeoDataFrames; values: short label suitable for a colorbar at projection
# scale. Unknown names pass through untouched (the caller's intent wins).
_SHORT_LABELS: dict[str, str] = {
    "lambda_f_mean": r"$\lambda_f$",
    "lambda_f": r"$\lambda_f$",
    "lambda_p": r"$\lambda_p$",
    "lambda_p_mean": r"$\lambda_p$",
    "sky_view_factor": "SVF",
    "svf": "SVF",
    "svf_mean": "SVF",
    "slope_deg": "slope (°)",
    "slope": "slope (°)",
    "aspect_deg": "aspect (°)",
    "aspect": "aspect (°)",
    "southness": "southness",
    "northness": "northness",
    "eastness": "eastness",
    "sigma_h": r"$\sigma_H$",
    "sigma_h_mean": r"$\sigma_H$",
    "solar_hours_winter": "winter sun (h)",
    "solar_hours_summer": "summer sun (h)",
}

_PRESENTATION_RC: dict[str, object] = {
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "legend.title_fontsize": 14,
    "figure.titlesize": 18,
    "axes.linewidth": 0.9,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "lines.linewidth": 1.6,
    "patch.linewidth": 0.7,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "mathtext.default": "regular",
}


def apply(preset: Preset = "paper") -> None:
    """Configure ``plt.rcParams`` for the chosen preset.

    ``paper`` is a no-op: existing scripts already set whatever sizing they
    need locally and overriding here would silently change paper output.
    ``presentation`` applies the projection-readable bundle defined above.
    Any subsequent ``apply_style()`` call from the per-paper style module
    will *override* these settings — so call ``apply`` AFTER any legacy
    style helpers when running in presentation mode.
    """
    if preset == "paper":
        return
    if preset != "presentation":
        raise ValueError(f"unknown preset: {preset!r}")
    plt.rcParams.update(_PRESENTATION_RC)


def apply_to_map_axes(ax: Axes) -> None:
    """Strip ticks/spines from a map axes. Idempotent."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, right=False, top=False, bottom=False,
                   labelleft=False, labelright=False, labeltop=False,
                   labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def short_label(name: str) -> str:
    """Map a long-form metric name to a short colorbar/legend label.

    Conservative pass-through: unknown names are returned unchanged so the
    helper never silently rewrites a custom label the caller cares about.
    """
    return _SHORT_LABELS.get(name, name)


def add_scale_bar(
    ax: Axes,
    length_m: float,
    label: str | None = None,
    *,
    loc: str = "lower right",
    color: str = "#111111",
    fontsize: int = 14,
) -> None:
    """Draw a scale bar + N arrow on a map axes.

    Assumes ``ax`` is a metric-projected map (length_m in axis units).
    Places the bar in the named corner with a small data-fraction inset;
    the N arrow sits in the upper-right of the same axes regardless of
    ``loc`` (north convention; doesn't fight the bar for the bottom-right).
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ex = xlim[1] - xlim[0]
    ey = ylim[1] - ylim[0]

    inset_x = ex * 0.04
    inset_y = ey * 0.05
    bar_h = ey * 0.012

    if "right" in loc:
        x0 = xlim[1] - inset_x - length_m
    else:
        x0 = xlim[0] + inset_x
    if "upper" in loc:
        y0 = ylim[1] - inset_y - bar_h
    else:
        y0 = ylim[0] + inset_y

    ax.add_patch(plt.Rectangle(
        (x0, y0), length_m, bar_h,
        facecolor=color, edgecolor=color, linewidth=0.6, zorder=20,
        clip_on=False,
    ))
    if label is None:
        label = f"{int(length_m)} m" if length_m < 1000 else f"{length_m / 1000:.0f} km"
    ax.text(
        x0 + length_m / 2, y0 + bar_h + ey * 0.012,
        label,
        ha="center", va="bottom",
        fontsize=fontsize, fontweight="bold", color=color, zorder=20,
        clip_on=False,
    )

    # North arrow — upper right, sized relative to axis height.
    nx = xlim[1] - ex * 0.05
    arrow_h = ey * 0.10
    ny_top = ylim[1] - ey * 0.06
    ny_base = ny_top - arrow_h
    ax.annotate(
        "",
        xy=(nx, ny_top), xytext=(nx, ny_base),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8,
                        mutation_scale=18),
        zorder=20,
    )
    ax.text(
        nx, ny_top + ey * 0.012, "N",
        ha="center", va="bottom", fontsize=fontsize, fontweight="bold",
        color=color, zorder=20,
    )
