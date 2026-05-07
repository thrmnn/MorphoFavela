"""Figure 7 — Vidigal street-level solar envelope (winter / annual / summer).

Three panels on the same point set, same colour scale [0, 12] hours,
same buildings backdrop:

  (a) winter solstice    — worst case (sun stays north of zenith)
  (b) annual proxy       — mean of 4 reference dates
  (c) summer solstice    — best case (sun crosses south of zenith)

The three colormaps are identical so a quick mental scrub left→right
reveals the seasonal swing per street: a S-facing alley that is dark
in (a) often turns bright in (c). The point of the figure is to make
the seasonal envelope visible at a glance, before the §5.4 narrative
quantifies it.

Source: ``outputs/vidigal/morphometrics/svf/svf_streets_solar.gpkg``,
produced by ``scripts/run_street_solar.py`` (30-min sampling, 4
reference dates, Hottel + Liu-Jordan irradiance, ray-cast against
``scene.stl`` = DTM + extruded LiDAR-height footprints).

Per panel a small inline stats card gives mean / median / share-of-
zero-hour cells so the maps are not the only quantitative anchor.
A combined version is saved as ``fig07_solar_envelope_vidigal.png``;
each panel is also saved standalone as
``fig07a_solar_winter.png``, ``fig07b_solar_annual.png``,
``fig07c_solar_summer.png`` for re-use in slides.
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import (  # noqa: E402
    PROJECT_ROOT,
    WIDTH_DOUBLE,
    add_north_arrow,
    add_scalebar,
    apply_style,
    clean_map_axes,
    load_buildings,
    save_fig,
)


VMAX_HOURS = 12.0  # shared colour ceiling so panels are directly comparable
CMAP = "YlOrRd_r"  # legacy palette (reversed): deep red (0 h) → orange → pale yellow (12 h)


def _load() -> gpd.GeoDataFrame:
    path = (
        PROJECT_ROOT
        / "outputs"
        / "vidigal"
        / "morphometrics"
        / "svf"
        / "svf_streets_solar.gpkg"
    )
    return gpd.read_file(path)


def _draw_panel(
    ax: plt.Axes,
    pts: gpd.GeoDataFrame,
    column: str,
    title: str,
    buildings: gpd.GeoDataFrame,
    bbox: tuple[float, float, float, float],
) -> object:
    xmin, ymin, xmax, ymax = bbox
    buildings.plot(ax=ax, facecolor="0.86", edgecolor="0.62", linewidth=0.18, zorder=1)
    sc = ax.scatter(
        pts.geometry.x,
        pts.geometry.y,
        c=pts[column],
        cmap=CMAP,
        vmin=0,
        vmax=VMAX_HOURS,
        s=4,
        alpha=0.95,
        linewidths=0,
        zorder=2,
    )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    clean_map_axes(ax)
    ax.set_title(title, fontsize=8, loc="left", pad=2)

    # Inline stats card.
    mean_h = float(pts[column].mean())
    med_h = float(pts[column].median())
    zero_pct = 100.0 * float((pts[column] <= 0.01).mean())
    ax.text(
        0.02,
        0.98,
        f"mean {mean_h:4.2f} h\nmedian {med_h:4.2f} h\nzero-share {zero_pct:4.1f}%",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.6,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.7", alpha=0.92),
        zorder=4,
    )
    return sc


def _save_standalone_panel(
    pts: gpd.GeoDataFrame,
    column: str,
    title: str,
    buildings: gpd.GeoDataFrame,
    bbox: tuple[float, float, float, float],
    name: str,
) -> None:
    fig, ax = plt.subplots(figsize=(WIDTH_DOUBLE * 0.6, WIDTH_DOUBLE * 0.55))
    sc = _draw_panel(ax, pts, column=column, title=title, buildings=buildings, bbox=bbox)
    cbar = fig.colorbar(sc, ax=ax, orientation="horizontal", shrink=0.7, pad=0.04, aspect=24)
    cbar.set_label("hours of direct sun", fontsize=6)
    cbar.ax.tick_params(labelsize=5)
    add_scalebar(ax, length_m=200)
    add_north_arrow(ax)
    fig.tight_layout()
    save_fig(fig, name)


def main() -> None:
    apply_style()
    pts = _load()
    buildings = load_buildings("vidigal", extended=False)
    if buildings.crs != pts.crs:
        buildings = buildings.to_crs(pts.crs)
    bbox_arr = pts.total_bounds
    pad = 30.0
    bbox = (
        bbox_arr[0] - pad,
        bbox_arr[1] - pad,
        bbox_arr[2] + pad,
        bbox_arr[3] + pad,
    )

    panels = [
        ("solar_hours_winter", "(a) winter solstice — 21 Jun (worst case)", "fig07a_solar_winter"),
        ("solar_hours_annual", "(b) annual proxy — mean of 4 ref. dates", "fig07b_solar_annual"),
        ("solar_hours_summer", "(c) summer solstice — 21 Dec (best case)", "fig07c_solar_summer"),
    ]

    # Standalone PNGs (re-use in slides / SI).
    for column, title, name in panels:
        _save_standalone_panel(pts, column=column, title=title, buildings=buildings, bbox=bbox, name=name)

    # Combined 1×3 figure.
    fig = plt.figure(figsize=(WIDTH_DOUBLE, WIDTH_DOUBLE * 0.55))
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.0, 0.06],
        hspace=0.12,
        wspace=0.04,
        left=0.02,
        right=0.985,
        top=0.84,
        bottom=0.10,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[1, :])

    last_sc = None
    for ax, (column, title, _) in zip(axes, panels):
        last_sc = _draw_panel(ax, pts, column=column, title=title, buildings=buildings, bbox=bbox)

    add_scalebar(axes[0], length_m=200)
    add_north_arrow(axes[2])

    cbar = fig.colorbar(last_sc, cax=cax, orientation="horizontal")
    cbar.set_label("hours of direct sun  (shared scale 0 – 12 h)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    fig.suptitle(
        "Vidigal — street-level solar envelope.  Same points, same scale, three reference cases.\n"
        "(a) §5.3 winter dissociation panel; (b) the figure year-round comfort & PV studies should use; "
        "(c) S-facing alleys recover in summer.",
        fontsize=8.5,
        y=0.97,
    )

    save_fig(fig, "fig07_solar_envelope_vidigal")


if __name__ == "__main__":
    main()
