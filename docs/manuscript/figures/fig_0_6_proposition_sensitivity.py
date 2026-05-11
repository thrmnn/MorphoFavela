#!/usr/bin/env python3
"""Figure 0.6 (proposition B) — Sensitivity to threshold choice.

The 4-state diagnosis from Fig 0.4 depends on two threshold choices
(Lawson 1.0 m/s for ventilation, WHO 2 h for sunlight). Both have
defensible alternatives. This figure surfaces how the diagnosis
moves when those thresholds are varied, identifying the cells whose
state is stable across the dial (the high-confidence diagnostic
core).

Panels
------
A   Threshold dial: 3×3 stacked-bar small multiples per typology,
    U_vent ∈ {0.8, 1.0, 1.2} m/s × sun_h ∈ {1, 2, 3} h.
B   Cell-level stability map per site (5 panels). Each cell is
    coloured by % of (U_vent × sun_h) scenarios in which it is
    classified as failing.
C   Threshold-sensitivity heatmap per typology. x = U_vent,
    y = sun_h, color = compound %. Highlights which threshold
    moves the headline most.

Run:
    python docs/manuscript/figures/fig_0_6_proposition_sensitivity.py
"""

from __future__ import annotations

import sys
import warnings
from itertools import product
from pathlib import Path

import geopandas as gpd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "outputs" / "paper_figures"))
sys.path.insert(0, str(PROJECT_ROOT / "docs" / "manuscript" / "figures"))

from fig_style import (  # noqa: E402
    SITE_LABELS,
    SITE_ORDER,
    WIDTH_DOUBLE,
    add_north_arrow,
    add_scalebar,
    apply_style,
    clean_map_axes,
    load_boundary,
    load_buildings,
)
from fig_0_4_diagnostic import (  # noqa: E402
    BUILDING_COLOR,
    NOT_ASSESSED_COLOR,
    STATE_COLORS,
    STATE_KEYS,
    STATE_LABELS,
    TYPOLOGY_AGGREGATE_ORDER,
    TYPOLOGY_OF,
    _aggregate_solar_to_cells,
)

EXPORTS_DIR = Path(__file__).resolve().parent / "exports"
EXPORTS_DIR.mkdir(exist_ok=True)

U_THRESHOLDS = [0.8, 1.0, 1.2]
SUN_THRESHOLDS = [1.0, 2.0, 3.0]
ALL_COMBOS = list(product(U_THRESHOLDS, SUN_THRESHOLDS))

# Finer sensitivity grid for panel C heatmap.
U_GRID = np.linspace(0.5, 1.5, 11)
SUN_GRID = np.linspace(0.5, 3.5, 11)


def classify(u: pd.Series, sun: pd.Series,
             u_thr: float, sun_thr: float) -> pd.Series:
    classified = u.notna() & sun.notna()
    vent_ok = u >= u_thr
    sun_ok = sun >= sun_thr
    state = pd.Series(index=u.index, dtype=object)
    state.loc[classified & vent_ok & sun_ok] = "adequate"
    state.loc[classified & vent_ok & ~sun_ok] = "sun"
    state.loc[classified & ~vent_ok & sun_ok] = "vent"
    state.loc[classified & ~vent_ok & ~sun_ok] = "compound"
    return state


def state_fractions(states: pd.Series) -> dict[str, float]:
    classified = states.dropna()
    total = max(len(classified), 1)
    return {k: 100.0 * (classified == k).sum() / total for k in STATE_KEYS}


def build_site_data(site: str) -> gpd.GeoDataFrame:
    g = gpd.read_file(
        PROJECT_ROOT / "outputs" / site / "cfd_analysis" / "grid_with_cfd.gpkg"
    )
    g = _aggregate_solar_to_cells(g, site)
    return g


# ---------------------------------------------------------------------------
# Panel A: threshold-dial small multiples per typology
# ---------------------------------------------------------------------------
def aggregate_per_typology(grids: dict[str, gpd.GeoDataFrame],
                           u_thr: float, sun_thr: float) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for typ in TYPOLOGY_AGGREGATE_ORDER:
        members = [s for s in SITE_ORDER if TYPOLOGY_OF[s] == typ]
        rows = []
        for s in members:
            g = grids[s]
            st = classify(g["annual_cfd_U_mean"], g["solar_hours_winter"],
                          u_thr, sun_thr)
            cls = st.dropna()
            for k in STATE_KEYS:
                rows.append((s, k, int((cls == k).sum())))
        df = pd.DataFrame(rows, columns=["site", "state", "n"])
        totals = df.groupby("state")["n"].sum()
        total = totals.sum()
        if total == 0:
            out[typ] = {k: 0.0 for k in STATE_KEYS}
        else:
            out[typ] = {k: 100.0 * float(totals.get(k, 0)) / total for k in STATE_KEYS}
    return out


def draw_panel_a(fig, outer_slot, grids: dict[str, gpd.GeoDataFrame]) -> None:
    inner = outer_slot.subgridspec(
        len(SUN_THRESHOLDS), len(U_THRESHOLDS),
        wspace=0.12, hspace=0.22,
    )
    for i, sun_thr in enumerate(SUN_THRESHOLDS[::-1]):  # top = highest sun threshold
        for j, u_thr in enumerate(U_THRESHOLDS):
            ax = fig.add_subplot(inner[i, j])
            agg = aggregate_per_typology(grids, u_thr, sun_thr)
            y_pos = np.arange(len(TYPOLOGY_AGGREGATE_ORDER))[::-1]
            for k_t, typ in enumerate(TYPOLOGY_AGGREGATE_ORDER):
                left = 0.0
                for k in STATE_KEYS:
                    v = agg[typ][k]
                    ax.barh(y_pos[k_t], v, left=left, height=0.7,
                            color=STATE_COLORS[k], edgecolor="white",
                            linewidth=0.3, zorder=2)
                    left += v
            ax.set_xlim(0, 100)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(TYPOLOGY_AGGREGATE_ORDER, fontsize=5.5)
            ax.set_xticks([])
            ax.tick_params(axis="y", length=0, pad=1)
            for s in ("top", "right", "left", "bottom"):
                ax.spines[s].set_visible(False)
            # Cell label: U / sun thresholds. Bold the canonical Fig 0.4 pair.
            is_default = (u_thr == 1.0 and sun_thr == 2.0)
            text = f"U≥{u_thr}, sun≥{sun_thr:.0f}h"
            ax.text(0.99, 1.02, text, transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=5.5,
                    color="#222" if is_default else "#555",
                    fontweight="bold" if is_default else "normal")


# ---------------------------------------------------------------------------
# Panel B: cell-level stability map
# ---------------------------------------------------------------------------
def compute_stability(grid: gpd.GeoDataFrame) -> pd.Series:
    """Fraction of (U_vent, sun_h) combos in which the cell is failing
    (not 'adequate'). NaN for cells without inputs."""
    counts = pd.Series(0, index=grid.index, dtype=float)
    valid = pd.Series(0, index=grid.index, dtype=int)
    for u_thr, sun_thr in ALL_COMBOS:
        st = classify(grid["annual_cfd_U_mean"], grid["solar_hours_winter"],
                      u_thr, sun_thr)
        mask = st.notna()
        valid += mask.astype(int)
        counts += (st != "adequate").astype(float) * mask.astype(float)
    n_combos = len(ALL_COMBOS)
    frac = counts / n_combos
    frac.loc[valid == 0] = np.nan
    return frac


def draw_stability_map(ax, site: str, grid: gpd.GeoDataFrame,
                       buildings: gpd.GeoDataFrame,
                       boundary: gpd.GeoDataFrame) -> None:
    minx, miny, maxx, maxy = boundary.total_bounds
    pad = 0.02 * max(maxx - minx, maxy - miny)
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_aspect("equal")
    buildings.plot(ax=ax, facecolor=BUILDING_COLOR, edgecolor="none",
                   linewidth=0.0, zorder=1)
    stab = compute_stability(grid)
    grid = grid.assign(stab=stab)
    unc = grid[grid["stab"].isna()]
    if len(unc):
        unc.plot(ax=ax, facecolor=NOT_ASSESSED_COLOR, edgecolor="none",
                 alpha=0.5, zorder=2)
    valid = grid[grid["stab"].notna()]
    if len(valid):
        valid.plot(ax=ax, column="stab", cmap="magma_r", vmin=0.0, vmax=1.0,
                   edgecolor="none", alpha=0.92, zorder=3)
    boundary.boundary.plot(ax=ax, color="black", linewidth=0.4,
                           linestyle=(0, (3, 2)), zorder=4, alpha=0.65)
    clean_map_axes(ax)
    ew = maxx - minx
    bar = 100 if ew < 1500 else (200 if ew < 2500 else 500)
    add_scalebar(ax, length_m=bar, loc="lower left")
    add_north_arrow(ax, loc="upper right", size=0.05)
    ax.set_title(SITE_LABELS[site], fontsize=7.5, pad=4, fontweight="bold",
                 color="#222")
    cls = stab.dropna()
    if len(cls):
        always_fail = 100.0 * (cls >= 0.999).sum() / len(cls)
        ax.text(0.5, -0.04, f"always failing {always_fail:.0f}%",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=6, color="#222", fontweight="bold")


# ---------------------------------------------------------------------------
# Panel C: threshold heatmap per typology
# ---------------------------------------------------------------------------
def compound_grid_per_typology(grids: dict[str, gpd.GeoDataFrame],
                               typ: str) -> np.ndarray:
    members = [s for s in SITE_ORDER if TYPOLOGY_OF[s] == typ]
    Z = np.zeros((len(SUN_GRID), len(U_GRID)))
    for j, u_thr in enumerate(U_GRID):
        for i, sun_thr in enumerate(SUN_GRID[::-1]):  # invert so sun_h increases upward
            n_comp = 0
            n_cls = 0
            for s in members:
                g = grids[s]
                st = classify(g["annual_cfd_U_mean"], g["solar_hours_winter"],
                              float(u_thr), float(sun_thr))
                cls = st.dropna()
                n_cls += len(cls)
                n_comp += int((cls == "compound").sum())
            Z[i, j] = 100.0 * n_comp / max(n_cls, 1)
    return Z


def draw_panel_c(fig, outer_slot,
                 grids: dict[str, gpd.GeoDataFrame]) -> None:
    inner = outer_slot.subgridspec(1, len(TYPOLOGY_AGGREGATE_ORDER) + 1,
                                   width_ratios=[1.0] * len(TYPOLOGY_AGGREGATE_ORDER) + [0.06],
                                   wspace=0.15)
    Zs = {t: compound_grid_per_typology(grids, t) for t in TYPOLOGY_AGGREGATE_ORDER}
    vmax = max(np.nanmax(z) for z in Zs.values())
    for j, typ in enumerate(TYPOLOGY_AGGREGATE_ORDER):
        ax = fig.add_subplot(inner[0, j])
        Z = Zs[typ]
        im = ax.imshow(Z, cmap="magma_r", vmin=0, vmax=vmax,
                       extent=[U_GRID[0], U_GRID[-1], SUN_GRID[0], SUN_GRID[-1]],
                       origin="lower", aspect="auto")
        ax.set_xlabel("U_vent threshold (m/s)", fontsize=6.5, labelpad=2)
        if j == 0:
            ax.set_ylabel("Sun threshold (h)", fontsize=6.5, labelpad=2)
        ax.set_title(typ, fontsize=7, fontweight="bold", color="#222", pad=3)
        ax.tick_params(labelsize=5.5, length=2, width=0.4, pad=2)
        # Canonical reference point.
        ax.scatter([1.0], [2.0], marker="o", s=18,
                   edgecolor="white", facecolor="#1a1a1a",
                   linewidths=0.6, zorder=4)
        ax.text(1.04, 2.06, "Fig 0.4", fontsize=5.5,
                color="#1a1a1a", fontweight="bold")
        for s in ax.spines.values():
            s.set_linewidth(0.4)

    cax = fig.add_subplot(inner[0, -1])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Compound failure (%)", fontsize=6, labelpad=2)
    cb.ax.tick_params(labelsize=5.5, length=2, width=0.4, pad=1)
    cb.outline.set_linewidth(0.4)


def main() -> None:
    apply_style()
    warnings.filterwarnings("ignore", category=UserWarning, module="geopandas")
    warnings.filterwarnings("ignore", category=FutureWarning)

    print("Loading site grids ...")
    grids = {site: build_site_data(site) for site in SITE_ORDER}
    for site in SITE_ORDER:
        n_cls = (grids[site]["annual_cfd_U_mean"].notna()
                 & grids[site]["solar_hours_winter"].notna()).sum()
        print(f"  {site:<22} classifiable={int(n_cls):4}")

    boundaries = {s: load_boundary(s) for s in SITE_ORDER}
    buildings = {s: load_buildings(s, extended=False) for s in SITE_ORDER}

    fig = plt.figure(figsize=(WIDTH_DOUBLE, WIDTH_DOUBLE * 1.05))
    outer = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[2.2, 2.2, 1.4],
        hspace=0.32,
        left=0.05, right=0.96, top=0.93, bottom=0.06,
    )

    # Row 0: panel A grid (left), legend (right).
    print("Drawing panel A (threshold dial) ...")
    a_gs = outer[0].subgridspec(1, 2, width_ratios=[0.78, 0.22], wspace=0.05)
    draw_panel_a(fig, a_gs[0, 0], grids)

    ax_legend = fig.add_subplot(a_gs[0, 1])
    ax_legend.set_xticks([])
    ax_legend.set_yticks([])
    for sp in ax_legend.spines.values():
        sp.set_visible(False)
    handles = [Patch(facecolor=STATE_COLORS[k], edgecolor="none",
                     label=STATE_LABELS[k]) for k in STATE_KEYS]
    leg = ax_legend.legend(handles=handles, loc="upper left",
                           bbox_to_anchor=(0.0, 0.98),
                           frameon=False, fontsize=6.5,
                           handlelength=1.4, labelspacing=0.45,
                           title="State",
                           title_fontsize=6.5, borderpad=0.0)
    leg.get_title().set_fontweight("bold")

    caption_A = (
        "Threshold dial: 3 U-thresholds × 3\n"
        "sun-thresholds = 9 small multiples.\n"
        "Bars per typology. Bold tile = the\n"
        "Fig 0.4 default. Hillside/Flatland\n"
        "ranking is stable across the dial."
    )
    ax_legend.text(0.0, 0.45, caption_A, transform=ax_legend.transAxes,
                   ha="left", va="top", fontsize=5.8, color="#333",
                   linespacing=1.4)

    # Row 1: stability maps.
    print("Drawing panel B (stability maps) ...")
    maps_gs = outer[1].subgridspec(1, 5, wspace=0.02)
    for i, site in enumerate(SITE_ORDER):
        ax = fig.add_subplot(maps_gs[0, i])
        draw_stability_map(ax, site, grids[site], buildings[site], boundaries[site])

    # Row 2: heatmaps.
    print("Drawing panel C (sensitivity heatmap) ...")
    draw_panel_c(fig, outer[2], grids)

    # Panel labels.
    a_axes = [a for a in fig.axes if a.get_subplotspec().get_gridspec()
              is a_gs[0, 0].get_gridspec()]
    if a_axes:
        ax_a_first = a_axes[0]
        pos_a = ax_a_first.get_position()
        fig.text(pos_a.x0 - 0.025, pos_a.y1 + 0.020, "a",
                 fontsize=9, fontweight="bold", va="bottom", ha="left")
    first_map = fig.axes[len(U_THRESHOLDS) * len(SUN_THRESHOLDS) + 1]  # after dial + legend
    pos_m = first_map.get_position()
    fig.text(pos_m.x0 - 0.025, pos_m.y1 + 0.015, "b",
             fontsize=9, fontweight="bold", va="bottom", ha="left")
    # Panel C label: first axis of the heatmap row.
    # The heatmap subgridspec is inside outer[2] — pick its leftmost axis.
    heat_axes = sorted(
        [a for a in fig.axes if a.get_subplotspec().get_gridspec().get_geometry()
         == (1, len(TYPOLOGY_AGGREGATE_ORDER) + 1)],
        key=lambda a: a.get_position().x0,
    )
    if heat_axes:
        pos_c = heat_axes[0].get_position()
        fig.text(pos_c.x0 - 0.025, pos_c.y1 + 0.015, "c",
                 fontsize=9, fontweight="bold", va="bottom", ha="left")

    fig.text(0.5, 0.97,
             "Sensitivity: the diagnostic typology ranking holds across reasonable threshold variation",
             ha="center", va="top", fontsize=8.5, fontweight="bold",
             color="#1a1a1a")
    fig.text(0.5, 0.95,
             "PROPOSITION B — robustness audit on synthetic CFD; thresholds sweep, ranking conserved.",
             ha="center", va="top", fontsize=5.5, style="italic", color="#a0522d")

    out_png = EXPORTS_DIR / "fig_0_6_proposition_sensitivity.png"
    out_svg = EXPORTS_DIR / "fig_0_6_proposition_sensitivity.svg"
    print(f"Saving {out_png.name} + {out_svg.name} ...")
    fig.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.05,
                facecolor="white")
    fig.savefig(out_svg, format="svg", bbox_inches="tight", pad_inches=0.05,
                facecolor="white")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
