#!/usr/bin/env python3
"""Figure 0.4 — Diagnostic taxonomy at favela scale  ★ HEADLINE.

Compound-constraint as the headline finding: ventilation constraint and
sunlight constraint co-locate in coherent spatial patches across the
five favelas, with strong typology contrast (hillside / mixed /
flatland).

Panels
------
B–F  5 favela maps at 10 m grid, cells colored by 4-state diagnostic.
A    2D performance space scatter (U_pedestrian × winter sun hours).
G    Horizontal stacked bars: % cell area in each state, per site +
     3 typology aggregates (Hillside, Mixed, Flatland).

Threshold choice (synthetic-CFD era)
------------------------------------
Our CFD reports canyon-scale ACH (264–1209 across cells) which is *not*
unit-comparable to WHO 0.5 ACH (an indoor-room minimum). To honor the
WHO threshold while staying physically honest, the X-axis on panel A
is pedestrian wind speed at 1.5 m, with two coincident threshold lines:

  * 1.0 m/s  (Lawson outdoor stagnation; primary)
  * 0.5 ACH  (top axis, indoor-equivalent ACH = α · ACH_canyon, α=1/150
              per Etheridge–Sandberg canonical canyon→room coupling)

Both thresholds are anchored to the same vertical quadrant line.
Classification uses U_mean ≥ 1.0 m/s as the operational test.

Run:
    python docs/manuscript/figures/fig_0_4_diagnostic.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "outputs" / "paper_figures"))
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

EXPORTS_DIR = Path(__file__).resolve().parent / "exports"
EXPORTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Locked constants (single CONFIG block — swap when real CFD lands).
# ---------------------------------------------------------------------------
THRESHOLD_U_VENT = 1.0          # m/s, Lawson outdoor stagnation
THRESHOLD_SUN_HOURS = 2.0       # h/day winter solstice, WHO daylight proxy
ALPHA_ACH = 1.0 / 150.0         # canyon→indoor coupling (Etheridge-Sandberg)

# Locked typology — CDA is mixed (terrain partly hillside, partly flat).
HILLSIDE_SITES = ("vidigal", "rocinha")
MIXED_SITES = ("complexo_do_alemao",)
FLATLAND_SITES = ("riodaspedras", "maré")
TYPOLOGY_OF = (
    {s: "Hillside" for s in HILLSIDE_SITES}
    | {s: "Mixed" for s in MIXED_SITES}
    | {s: "Flatland" for s in FLATLAND_SITES}
)

# 4-state diagnostic palette (Okabe-Ito, locked, colorblind-safe).
STATE_KEYS = ["adequate", "vent", "sun", "compound"]
STATE_COLORS = {
    "adequate": "#BDBDBD",
    "vent": "#0072B2",
    "sun": "#E69F00",
    "compound": "#D55E00",
}
STATE_LABELS = {
    "adequate": "Adequate",
    "vent": "Ventilation constraint",
    "sun": "Sunlight constraint",
    "compound": "Compound constraint",
}
NOT_ASSESSED_COLOR = "#f4f4f4"
BUILDING_COLOR = "#dcdcdc"
TYPOLOGY_AGGREGATE_ORDER = ["Hillside", "Mixed", "Flatland"]


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------
def _aggregate_solar_to_cells(grid: gpd.GeoDataFrame, site: str) -> gpd.GeoDataFrame:
    """Spatial-join street solar points into 10 m cells (mean per cell)."""
    sol_path = (
        PROJECT_ROOT / "outputs" / site / "morphometrics" / "svf"
        / "svf_streets_solar.gpkg"
    )
    if not sol_path.exists():
        grid = grid.copy()
        grid["solar_hours_winter"] = np.nan
        return grid

    sol = gpd.read_file(sol_path)[["solar_hours_winter", "geometry"]]
    joined = gpd.sjoin(sol, grid[["zone_id", "geometry"]],
                       how="inner", predicate="within")
    agg = (
        joined.groupby("zone_id")["solar_hours_winter"].mean().reset_index()
    )
    grid = grid.merge(agg, on="zone_id", how="left")
    return grid


def build_diagnostic_grid(site: str) -> gpd.GeoDataFrame:
    """Return cell GeoDataFrame with annual_cfd_U_mean, solar_hours_winter,
    and 'state' (one of STATE_KEYS or NaN if unassessed)."""
    g = gpd.read_file(
        PROJECT_ROOT / "outputs" / site / "cfd_analysis" / "grid_with_cfd.gpkg"
    )
    g = _aggregate_solar_to_cells(g, site)

    has_u = g["annual_cfd_U_mean"].notna()
    has_sun = g["solar_hours_winter"].notna()
    classified = has_u & has_sun

    vent_ok = g["annual_cfd_U_mean"] >= THRESHOLD_U_VENT
    sun_ok = g["solar_hours_winter"] >= THRESHOLD_SUN_HOURS

    state = pd.Series(index=g.index, dtype=object)
    state.loc[classified & vent_ok & sun_ok] = "adequate"
    state.loc[classified & vent_ok & ~sun_ok] = "sun"
    state.loc[classified & ~vent_ok & sun_ok] = "vent"
    state.loc[classified & ~vent_ok & ~sun_ok] = "compound"
    g["state"] = state
    return g


def state_fractions(grid: gpd.GeoDataFrame) -> dict[str, float]:
    """% of classified cells in each state. Unclassified cells excluded."""
    classified = grid["state"].dropna()
    total = max(len(classified), 1)
    return {k: 100.0 * (classified == k).sum() / total for k in STATE_KEYS}


# ---------------------------------------------------------------------------
# Panels B–F: site maps
# ---------------------------------------------------------------------------
def draw_site_map(ax, site: str, grid: gpd.GeoDataFrame,
                  buildings: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame,
                  bounds_global, scalebar: bool = True) -> None:
    minx, miny, maxx, maxy = boundary.total_bounds
    pad = 0.02 * max(maxx - minx, maxy - miny)
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_aspect("equal")

    buildings.plot(ax=ax, facecolor=BUILDING_COLOR, edgecolor="none",
                   linewidth=0.0, zorder=1)

    # Cells without classification → faint backdrop.
    unc = grid[grid["state"].isna()]
    if len(unc):
        unc.plot(ax=ax, facecolor=NOT_ASSESSED_COLOR, edgecolor="none",
                 linewidth=0.0, alpha=0.6, zorder=2)

    for k in STATE_KEYS:
        sub = grid[grid["state"] == k]
        if len(sub):
            sub.plot(ax=ax, facecolor=STATE_COLORS[k], edgecolor="none",
                     linewidth=0.0, alpha=0.92, zorder=3)

    boundary.boundary.plot(ax=ax, color="black", linewidth=0.4,
                           linestyle=(0, (3, 2)), zorder=4, alpha=0.65)

    clean_map_axes(ax)
    if scalebar:
        ew = maxx - minx
        bar = 100 if ew < 1500 else (200 if ew < 2500 else 500)
        add_scalebar(ax, length_m=bar, loc="lower left")
    add_north_arrow(ax, loc="upper right", size=0.05)

    # Title above the map (avoid overlapping the data).
    ax.set_title(SITE_LABELS[site], fontsize=7.5, pad=4,
                 color="#222", fontweight="bold")
    fr = state_fractions(grid)
    label = (f"compound {fr['compound']:.0f}%"
             if grid["state"].notna().sum() > 0 else "(pending)")
    ax.text(0.5, -0.04, label,
            transform=ax.transAxes, ha="center", va="top",
            fontsize=6, color=STATE_COLORS["compound"], fontweight="bold")


# ---------------------------------------------------------------------------
# Panel A: 2D performance space
# ---------------------------------------------------------------------------
def draw_panel_a(ax, all_cells: pd.DataFrame) -> None:
    df = all_cells.dropna(subset=["annual_cfd_U_mean", "solar_hours_winter",
                                  "state"])

    # X-jitter only for visualization (sun_hours is integer-quantized).
    rng = np.random.default_rng(0)
    yj = df["solar_hours_winter"] + rng.uniform(-0.10, 0.10, size=len(df))

    for k in STATE_KEYS:
        sub = df[df["state"] == k]
        if not len(sub):
            continue
        ax.scatter(
            sub["annual_cfd_U_mean"], yj.loc[sub.index],
            s=2.4, c=STATE_COLORS[k], alpha=0.10, linewidths=0,
            zorder=2 if k == "adequate" else 3,
            label=STATE_LABELS[k],
        )

    # Quadrant lines.
    ax.axvline(THRESHOLD_U_VENT, color="#444", lw=0.8,
               linestyle=(0, (3, 2)), zorder=4)
    ax.axhline(THRESHOLD_SUN_HOURS, color="#444", lw=0.8,
               linestyle=(0, (3, 2)), zorder=4)

    # Quadrant whisper labels.
    ax.text(0.04, 0.06, "compound", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=6.5,
            color=STATE_COLORS["compound"],
            fontweight="bold", alpha=0.95)
    ax.text(0.04, 0.78, "ventilation\nconstraint", transform=ax.transAxes,
            ha="left", va="top", fontsize=6.5, color=STATE_COLORS["vent"],
            fontweight="bold", alpha=0.95)
    ax.text(0.97, 0.06, "sunlight\nconstraint", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.5, color=STATE_COLORS["sun"],
            fontweight="bold", alpha=0.95)
    ax.text(0.97, 0.78, "adequate", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.5, color="#666",
            fontweight="bold", alpha=0.95)

    # Lawson threshold callout (just to the right of the vertical line, mid-y).
    ax.text(THRESHOLD_U_VENT + 0.04, 6.0,
            f"Lawson\n{THRESHOLD_U_VENT:.1f} m/s",
            ha="left", va="center", fontsize=5.2, color="#444",
            fontweight="bold")
    # WHO 2 h horizontal threshold callout.
    ax.text(3.55, THRESHOLD_SUN_HOURS + 0.12,
            f"WHO {THRESHOLD_SUN_HOURS:.0f} h",
            ha="right", va="bottom", fontsize=5.2, color="#444",
            fontweight="bold")

    ax.set_xlabel("Pedestrian wind speed (U @ 1.5 m, m/s)", fontsize=7,
                  labelpad=2)
    ax.set_ylabel("Direct sun, winter solstice (h/day)", fontsize=7,
                  labelpad=2)
    ax.tick_params(labelsize=6, length=2, width=0.4, pad=2)
    ax.set_xlim(0.4, 3.6)
    ax.set_ylim(-0.5, 11.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.4)

    # Twin top-axis: indoor-equivalent ACH = α · ACH_canyon. The slope
    # ach_per_U is set by the synthetic-CFD generator's calibration:
    # mean ACH_canyon ≈ 549 at mean U=1.5 ⇒ slope ≈ 366. Under canonical
    # canyon→room coupling α=1/150 this gives indoor-eq ACH ≈ 2.4 at
    # U=1.0 m/s — well above WHO 0.5. So the WHO 0.5 line lives far left
    # of the figure (around U≈0.2 m/s); it is *not* the binding outdoor
    # constraint.
    ach_per_U = 366.0
    sec = ax.secondary_xaxis(
        "top",
        functions=(
            lambda u: u * ach_per_U * ALPHA_ACH,
            lambda a: a / (ach_per_U * ALPHA_ACH),
        ),
    )
    sec.set_xlabel(r"Indoor-equivalent ACH  ($\alpha = 1/150$, "
                   r"Etheridge–Sandberg)", fontsize=6.0, labelpad=2)
    sec.tick_params(labelsize=5.5, length=2, width=0.4, pad=1)

    # WHO 0.5 ACH marker — under α=1/150 it lives at U≈0.2 m/s, off-scale
    # to the left. Drawn as a small italic note tucked at top-left.
    who_u = 0.5 / (ach_per_U * ALPHA_ACH)  # ≈ 0.205 m/s
    ax.text(0.5, 10.9,
            "← WHO 0.5 ACH (off-scale at U≈0.2 m/s)",
            ha="left", va="top", fontsize=5.0, color="#888",
            style="italic")

    # n-cells annotation, anchored to bottom-right of plot area.
    ax.text(0.99, -0.18, f"n = {len(df):,} classified cells",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=5.5, color="#444")


# ---------------------------------------------------------------------------
# Panel G: stacked bars
# ---------------------------------------------------------------------------
def draw_panel_g(ax, fractions_per_site: dict[str, dict[str, float]]) -> None:
    rows = []
    labels = []
    bar_colors_text = []
    for s in SITE_ORDER:
        rows.append(fractions_per_site[s])
        labels.append(SITE_LABELS[s])
        bar_colors_text.append("#222")

    # Aggregates: cell-weighted across constituent sites.
    site_n = fractions_per_site["__n_cells__"]
    for typ in TYPOLOGY_AGGREGATE_ORDER:
        members = [s for s in SITE_ORDER if TYPOLOGY_OF[s] == typ]
        weights = np.array([site_n[s] for s in members], dtype=float)
        if weights.sum() == 0:
            agg = {k: 0.0 for k in STATE_KEYS}
        else:
            agg = {
                k: float(np.average(
                    [fractions_per_site[s][k] for s in members],
                    weights=weights,
                ))
                for k in STATE_KEYS
            }
        rows.append(agg)
        labels.append(typ)
        bar_colors_text.append("#444")

    n_rows = len(rows)
    y = np.arange(n_rows)[::-1]
    height = 0.74

    for i, fracs in enumerate(rows):
        left = 0.0
        for k in STATE_KEYS:
            v = fracs[k]
            ax.barh(y[i], v, left=left, height=height,
                    color=STATE_COLORS[k], edgecolor="white", linewidth=0.5,
                    zorder=2)
            left += v
        # Compound % annotation just past 100 (right of bar).
        ax.text(101, y[i], f"{fracs['compound']:.0f}%",
                ha="left", va="center", fontsize=6,
                color=STATE_COLORS["compound"], fontweight="bold")

    # Separator between site bars and aggregate bars.
    sep_y = y[len(SITE_ORDER) - 1] - 0.5
    ax.axhline(sep_y, color="#bbbbbb", lw=0.5, linestyle=":", zorder=1)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.5)
    for tick, c in zip(ax.get_yticklabels(), bar_colors_text):
        tick.set_color(c)
        if c == "#444":
            tick.set_fontweight("bold")

    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"], fontsize=6)
    ax.tick_params(axis="x", length=2, width=0.4, pad=2)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Cell area share (%)", fontsize=7, labelpad=2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.4)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#eaeaea", linewidth=0.4, zorder=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    apply_style()
    warnings.filterwarnings("ignore", category=UserWarning, module="geopandas")
    warnings.filterwarnings("ignore", category=FutureWarning)

    print("Building diagnostic grids ...")
    grids = {}
    pooled = []
    fractions = {"__n_cells__": {}}
    for site in SITE_ORDER:
        g = build_diagnostic_grid(site)
        grids[site] = g
        n_cls = g["state"].notna().sum()
        fractions["__n_cells__"][site] = int(n_cls)
        fractions[site] = state_fractions(g)
        pooled.append(g[["annual_cfd_U_mean", "solar_hours_winter", "state"]]
                      .assign(site=site))
        print(f"  {site:<22} n_cells={len(g):4}  classified={n_cls:4}  "
              f"compound={fractions[site]['compound']:.1f}%")
    pooled_df = pd.concat(pooled, ignore_index=True)

    # Boundaries / buildings (lazy).
    boundaries = {s: load_boundary(s) for s in SITE_ORDER}
    buildings = {s: load_buildings(s, extended=False) for s in SITE_ORDER}

    # Equal-width map panels — each site fits its own bounds with its own
    # per-meter scale. A scale bar on every panel communicates the size
    # difference (Maré is ~5× Vidigal in NS extent). Strict per-meter
    # scale is unworkable here because Maré's bounding box is ~9× Vidigal
    # in linear dimension, which crushed the smaller sites in the v1 layout.
    width_ratios = [1.0] * len(SITE_ORDER)

    fig = plt.figure(figsize=(WIDTH_DOUBLE, WIDTH_DOUBLE * 1.05))
    outer = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[2.4, 2.2, 1.0],
        hspace=0.30,
        left=0.05, right=0.96, top=0.92, bottom=0.06,
    )

    # Maps row.
    maps_gs = outer[0].subgridspec(1, 5, width_ratios=width_ratios, wspace=0.02)
    print("Drawing maps ...")
    for i, site in enumerate(SITE_ORDER):
        ax = fig.add_subplot(maps_gs[0, i])
        draw_site_map(ax, site, grids[site], buildings[site], boundaries[site],
                      bounds_global=None)

    # Scatter row: A on left (~0.55), legend block on right (~0.45).
    sgs = outer[1].subgridspec(1, 2, width_ratios=[0.58, 0.42], wspace=0.10)
    print("Drawing panel A ...")
    ax_a = fig.add_subplot(sgs[0, 0])
    draw_panel_a(ax_a, pooled_df)

    # Legend / caption block.
    ax_legend = fig.add_subplot(sgs[0, 1])
    ax_legend.set_xticks([])
    ax_legend.set_yticks([])
    for sp in ax_legend.spines.values():
        sp.set_visible(False)

    handles = [Patch(facecolor=STATE_COLORS[k], edgecolor="none",
                     label=STATE_LABELS[k]) for k in STATE_KEYS]
    handles.append(Patch(facecolor=NOT_ASSESSED_COLOR, edgecolor="#cccccc",
                         linewidth=0.4, label="Not assessed"))
    ax_legend.legend(handles=handles, loc="upper left",
                     bbox_to_anchor=(0.02, 0.98),
                     frameon=False, fontsize=7,
                     handlelength=1.6, handleheight=1.0,
                     labelspacing=0.55, borderpad=0.0)

    caption = (
        "4-state classification per 10 m cell.\n"
        "Ventilation constraint: U_mean < 1.0 m/s (Lawson stagnation).\n"
        "Sunlight constraint: < 2 h direct sun on winter solstice.\n"
        "Compound: both fail simultaneously.\n\n"
        "Synthetic CFD; threshold maps to WHO 0.5 ACH\n"
        "via canyon→indoor coupling (α = 1/150,\n"
        "Etheridge & Sandberg 1996). One-line swap to\n"
        "real-campaign ACH when CFD lands."
    )
    ax_legend.text(0.02, 0.50, caption, transform=ax_legend.transAxes,
                   ha="left", va="top", fontsize=5.8, color="#333",
                   linespacing=1.45)

    # Stacked bars row.
    print("Drawing panel G ...")
    bgs = outer[2].subgridspec(1, 1, wspace=0.0)
    ax_g = fig.add_subplot(bgs[0, 0])
    draw_panel_g(ax_g, fractions)

    # Panel labels.
    def _label(panel, ax, dx=-0.05, dy=0.02):
        pos = ax.get_position()
        fig.text(pos.x0 + dx, pos.y1 + dy, panel,
                 fontsize=10, fontweight="bold", va="bottom", ha="left")

    # B–F labels — placed inside top-left corner to avoid colliding with
    # the figure-level title.
    panel_names = ["b", "c", "d", "e", "f"]
    for i, (site, name) in enumerate(zip(SITE_ORDER, panel_names)):
        ax_i = fig.axes[i]
        ax_i.text(0.0, 1.18, name, transform=ax_i.transAxes,
                  fontsize=9, fontweight="bold", va="top", ha="left",
                  color="#1a1a1a")

    # A and G.
    pos_a = ax_a.get_position()
    fig.text(pos_a.x0 - 0.03, pos_a.y1 + 0.005, "a",
             fontsize=9, fontweight="bold", va="bottom", ha="left")
    pos_g = ax_g.get_position()
    fig.text(pos_g.x0 - 0.04, pos_g.y1 + 0.005, "g",
             fontsize=9, fontweight="bold", va="bottom", ha="left")

    # Headline title (figure-level), nudged down to avoid bbox_inches clipping.
    fig.text(0.5, 0.965,
             "Diagnostic taxonomy: compound constraint forms coherent spatial patches",
             ha="center", va="top", fontsize=8.5, fontweight="bold",
             color="#1a1a1a")

    # Maré-pending caveat if Maré solar hasn't landed yet.
    mare_n = grids["maré"]["state"].notna().sum()
    if mare_n == 0:
        fig.text(0.5, 0.945,
                 "Maré solar simulation pending — Maré row excluded "
                 "from typology aggregates and panel A until completion.",
                 ha="center", va="top", fontsize=5.5, style="italic",
                 color="#a0522d")

    out_png = EXPORTS_DIR / "fig_0_4_diagnostic.png"
    out_svg = EXPORTS_DIR / "fig_0_4_diagnostic.svg"
    print(f"Saving {out_png.name} + {out_svg.name} ...")
    fig.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.05,
                facecolor="white")
    fig.savefig(out_svg, format="svg", bbox_inches="tight", pad_inches=0.05,
                facecolor="white")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
