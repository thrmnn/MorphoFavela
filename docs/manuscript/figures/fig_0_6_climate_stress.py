#!/usr/bin/env python3
"""Figure 0.6 — Climate stress test: wind stilling shifts the 4-state
distribution non-uniformly across typologies.

Under a documented tropical wind-stilling trend (−10 to −15 % mean U
projected for SE Brazil mid-century; IPCC AR6 WGI Atlas regional
summary), the 4-state diagnosis from Fig 0.4 tilts away from adequate
toward compound constraint, but not uniformly across typologies.

This is a stylized scenario — wind speeds are uniformly scaled by a
factor; no CMIP6 ensemble or thermal coupling. Sunlight is unaffected
by the wind scaling so only the ventilation half of the classification
moves. The figure surfaces the marginal recruits that the climate
signal pulls below the Lawson stagnation threshold.

Panels
------
A   Stacked bars per site under {U×1.00, U×0.85, U×0.70}.
B   Cell-level state-flip map per site under U×0.85.
C   Typology vulnerability ladder: % of cells flipping to compound
    constraint under U×0.85, ordered.

Run:
    python docs/manuscript/figures/fig_0_6_climate_stress.py
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
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "outputs" / "paper_figures"))
sys.path.insert(0, str(PROJECT_ROOT / "docs" / "manuscript" / "figures"))

from fig_0_4_diagnostic import (
    BUILDING_COLOR,
    NOT_ASSESSED_COLOR,
    STATE_COLORS,
    STATE_KEYS,
    STATE_LABELS,
    THRESHOLD_SUN_HOURS,
    THRESHOLD_U_VENT,
    TYPOLOGY_AGGREGATE_ORDER,
    TYPOLOGY_OF,
    _aggregate_solar_to_cells,
)
from fig_style import (
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

SCENARIOS = [
    ("baseline", 1.00),
    ("−15 %", 0.85),
    ("−30 %", 0.70),
]
FOCAL_SCENARIO = ("−15 %", 0.85)

FLIP_COLORS = {
    "stable_adequate": "#BDBDBD",
    "stable_fail": "#666666",
    "flip_to_vent": "#0072B2",
    "flip_to_compound": "#D55E00",
}
FLIP_LABELS = {
    "stable_adequate": "Stays adequate",
    "stable_fail": "Stays constrained",
    "flip_to_vent": "Flips → ventilation constraint",
    "flip_to_compound": "Flips → compound constraint",
}


def classify(u: pd.Series, sun: pd.Series) -> pd.Series:
    classified = u.notna() & sun.notna()
    vent_ok = u >= THRESHOLD_U_VENT
    sun_ok = sun >= THRESHOLD_SUN_HOURS
    state = pd.Series(index=u.index, dtype=object)
    state.loc[classified & vent_ok & sun_ok] = "adequate"
    state.loc[classified & vent_ok & ~sun_ok] = "sun"
    state.loc[classified & ~vent_ok & sun_ok] = "vent"
    state.loc[classified & ~vent_ok & ~sun_ok] = "compound"
    return state


def build_site_data(site: str) -> gpd.GeoDataFrame:
    g = gpd.read_file(
        PROJECT_ROOT / "outputs" / site / "cfd_analysis" / "grid_with_cfd.gpkg"
    )
    g = _aggregate_solar_to_cells(g, site)
    g["state_base"] = classify(g["annual_cfd_U_mean"], g["solar_hours_winter"])
    for label, k in SCENARIOS:
        if label == "baseline":
            g["state_baseline"] = g["state_base"]
            continue
        u_scaled = g["annual_cfd_U_mean"] * k
        g[f"state_{label}"] = classify(u_scaled, g["solar_hours_winter"])
    return g


def state_fractions(states: pd.Series) -> dict[str, float]:
    classified = states.dropna()
    total = max(len(classified), 1)
    return {k: 100.0 * (classified == k).sum() / total for k in STATE_KEYS}


def flip_label(before: str, after: str) -> str | None:
    if pd.isna(before) or pd.isna(after):
        return None
    if before == "adequate" and after == "adequate":
        return "stable_adequate"
    if before == "adequate" and after == "vent":
        return "flip_to_vent"
    if before == "adequate" and after == "compound":
        return "flip_to_compound"
    if before in ("sun",) and after == "compound":
        return "flip_to_compound"
    return "stable_fail"


# ---------------------------------------------------------------------------
# Panel A: stacked bars per site × scenario
# ---------------------------------------------------------------------------
def draw_panel_a(ax, per_site_per_scenario: dict[str, dict[str, dict[str, float]]],
                 site_n: dict[str, int]) -> None:
    rows = []
    labels = []
    label_colors = []
    label_weights = []
    # Per-site x 3 scenarios.
    for s in SITE_ORDER:
        if site_n[s] == 0:
            continue
        for sc_label, _ in SCENARIOS:
            rows.append(per_site_per_scenario[s][sc_label])
            labels.append(f"{SITE_LABELS[s]}  {sc_label}")
            label_colors.append("#222" if sc_label == "baseline" else "#555")
            label_weights.append("bold" if sc_label == "baseline" else "normal")

    n_rows = len(rows)
    y = np.arange(n_rows)[::-1]
    height = 0.78
    for i, fracs in enumerate(rows):
        left = 0.0
        for k in STATE_KEYS:
            v = fracs[k]
            ax.barh(y[i], v, left=left, height=height,
                    color=STATE_COLORS[k], edgecolor="white", linewidth=0.4,
                    zorder=2)
            left += v
        ax.text(101, y[i], f"{fracs['compound']:.0f}%",
                ha="left", va="center", fontsize=5.5,
                color=STATE_COLORS["compound"], fontweight="bold")

    # Group separators (between sites).
    for i in range(3, n_rows, 3):
        ax.axhline(y[i] + 0.5, color="#dddddd", lw=0.4, linestyle=":")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6)
    for tick, c, w in zip(ax.get_yticklabels(), label_colors, label_weights):
        tick.set_color(c)
        if w == "bold":
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
# Panel B: state-flip map per site (under focal scenario)
# ---------------------------------------------------------------------------
def draw_flip_map(ax, site: str, grid: gpd.GeoDataFrame,
                  buildings: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame) -> None:
    minx, miny, maxx, maxy = boundary.total_bounds
    pad = 0.02 * max(maxx - minx, maxy - miny)
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_aspect("equal")
    buildings.plot(ax=ax, facecolor=BUILDING_COLOR, edgecolor="none",
                   linewidth=0.0, zorder=1)

    focal_label = FOCAL_SCENARIO[0]
    state_focal_col = f"state_{focal_label}"
    if state_focal_col not in grid.columns:
        clean_map_axes(ax)
        ax.text(0.5, 0.5, "(pending)", transform=ax.transAxes,
                ha="center", va="center", fontsize=7, color="#888")
        return

    grid = grid.copy()
    grid["flip"] = grid.apply(
        lambda r: flip_label(r["state_baseline"], r[state_focal_col]),
        axis=1,
    )
    unc = grid[grid["flip"].isna()]
    if len(unc):
        unc.plot(ax=ax, facecolor=NOT_ASSESSED_COLOR, edgecolor="none",
                 alpha=0.5, zorder=2)
    for k, c in FLIP_COLORS.items():
        sub = grid[grid["flip"] == k]
        if len(sub):
            sub.plot(ax=ax, facecolor=c, edgecolor="none", alpha=0.92, zorder=3)
    boundary.boundary.plot(ax=ax, color="black", linewidth=0.4,
                           linestyle=(0, (3, 2)), zorder=4, alpha=0.65)
    clean_map_axes(ax)
    ew = maxx - minx
    bar = 100 if ew < 1500 else (200 if ew < 2500 else 500)
    add_scalebar(ax, length_m=bar, loc="lower left")
    add_north_arrow(ax, loc="upper right", size=0.05)
    ax.set_title(SITE_LABELS[site], fontsize=7.5, pad=4, fontweight="bold",
                 color="#222")

    # Quick stat: % flipping to compound.
    cls = grid["flip"].dropna()
    if len(cls):
        p_comp = 100.0 * (cls == "flip_to_compound").sum() / len(cls)
        ax.text(0.5, -0.04, f"flip → compound {p_comp:.1f}%",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=6, color=FLIP_COLORS["flip_to_compound"],
                fontweight="bold")


# ---------------------------------------------------------------------------
# Panel C: typology vulnerability ladder
# ---------------------------------------------------------------------------
def draw_panel_c(ax, vulnerability: dict[str, float]) -> None:
    typs = [t for t in TYPOLOGY_AGGREGATE_ORDER if t in vulnerability]
    vals = [vulnerability[t] for t in typs]
    order = np.argsort(vals)[::-1]
    typs = [typs[i] for i in order]
    vals = [vals[i] for i in order]
    y = np.arange(len(typs))[::-1]
    bars = ax.barh(y, vals, height=0.62,
                   color=FLIP_COLORS["flip_to_compound"],
                   edgecolor="white", linewidth=0.6, zorder=2)
    for i, v in enumerate(vals):
        ax.text(v + 0.4, y[i], f"{v:.1f}%", va="center", ha="left",
                fontsize=6.5, color="#333", fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(typs, fontsize=7, fontweight="bold")
    ax.set_xlabel("% cells flipping → compound constraint under −15 % U", fontsize=6.5,
                  labelpad=2)
    ax.tick_params(axis="x", labelsize=6, length=2, width=0.4, pad=2)
    ax.tick_params(axis="y", length=0)
    max_x = max(vals) if vals else 10
    ax.set_xlim(0, max_x * 1.20 + 0.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.4)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#eaeaea", linewidth=0.4, zorder=0)


def main() -> None:
    apply_style()
    warnings.filterwarnings("ignore", category=UserWarning, module="geopandas")
    warnings.filterwarnings("ignore", category=FutureWarning)

    print("Building scenario grids ...")
    grids: dict[str, gpd.GeoDataFrame] = {}
    site_n: dict[str, int] = {}
    per_site_scenario: dict[str, dict[str, dict[str, float]]] = {}
    for site in SITE_ORDER:
        g = build_site_data(site)
        grids[site] = g
        site_n[site] = int(g["state_baseline"].notna().sum())
        per_site_scenario[site] = {}
        for label, _ in SCENARIOS:
            col = "state_baseline" if label == "baseline" else f"state_{label}"
            per_site_scenario[site][label] = state_fractions(g[col])
        print(f"  {site:<22} classified={site_n[site]:4}  "
              f"base_compound={per_site_scenario[site]['baseline']['compound']:.1f}%  "
              f"focal_compound={per_site_scenario[site][FOCAL_SCENARIO[0]]['compound']:.1f}%")

    # Typology vulnerability: % of (currently-adequate or sun-only) cells that
    # flip into a vent-fail or compound state under FOCAL_SCENARIO.
    vulnerability: dict[str, float] = {}
    for typ in TYPOLOGY_AGGREGATE_ORDER:
        members = [s for s in SITE_ORDER if TYPOLOGY_OF[s] == typ and site_n[s] > 0]
        if not members:
            continue
        total_flip_compound = 0
        total_classified = 0
        for s in members:
            g = grids[s]
            focal_col = f"state_{FOCAL_SCENARIO[0]}"
            flips = g.apply(
                lambda r, focal_col=focal_col: flip_label(r["state_baseline"], r[focal_col]),
                axis=1,
            )
            cls = flips.dropna()
            total_classified += len(cls)
            total_flip_compound += int((cls == "flip_to_compound").sum())
        if total_classified:
            vulnerability[typ] = 100.0 * total_flip_compound / total_classified

    # Boundaries / buildings.
    boundaries = {s: load_boundary(s) for s in SITE_ORDER}
    buildings = {s: load_buildings(s, extended=False) for s in SITE_ORDER}

    fig = plt.figure(figsize=(WIDTH_DOUBLE, WIDTH_DOUBLE * 1.05))
    outer = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[2.0, 2.2, 0.95],
        hspace=0.32,
        left=0.05, right=0.96, top=0.93, bottom=0.06,
    )

    # Row 0: panel A on left (wide stacked bars), legend on right.
    a_gs = outer[0].subgridspec(1, 2, width_ratios=[0.62, 0.38], wspace=0.10)
    ax_a = fig.add_subplot(a_gs[0, 0])
    draw_panel_a(ax_a, per_site_scenario, site_n)

    ax_legend = fig.add_subplot(a_gs[0, 1])
    ax_legend.set_xticks([])
    ax_legend.set_yticks([])
    for sp in ax_legend.spines.values():
        sp.set_visible(False)
    handles_a = [Patch(facecolor=STATE_COLORS[k], edgecolor="none",
                       label=STATE_LABELS[k]) for k in STATE_KEYS]
    leg1 = ax_legend.legend(handles=handles_a, loc="upper left",
                            bbox_to_anchor=(0.02, 0.98),
                            frameon=False, fontsize=6.5,
                            handlelength=1.4, labelspacing=0.45,
                            title="State (panel A)",
                            title_fontsize=6.5, borderpad=0.0)
    leg1.get_title().set_fontweight("bold")
    leg1.get_title().set_color("#222")
    ax_legend.add_artist(leg1)

    handles_b = [Patch(facecolor=FLIP_COLORS[k], edgecolor="none",
                       label=FLIP_LABELS[k]) for k in
                 ("stable_adequate", "stable_fail", "flip_to_vent", "flip_to_compound")]
    leg2 = ax_legend.legend(handles=handles_b, loc="lower left",
                            bbox_to_anchor=(0.02, 0.02),
                            frameon=False, fontsize=6.5,
                            handlelength=1.4, labelspacing=0.45,
                            title=f"Transition (panels B, U×{FOCAL_SCENARIO[1]})",
                            title_fontsize=6.5, borderpad=0.0)
    leg2.get_title().set_fontweight("bold")
    leg2.get_title().set_color("#222")

    # Row 1: 5-site flip maps (focal scenario).
    maps_gs = outer[1].subgridspec(1, 5, wspace=0.02)
    for i, site in enumerate(SITE_ORDER):
        ax = fig.add_subplot(maps_gs[0, i])
        draw_flip_map(ax, site, grids[site], buildings[site], boundaries[site])

    # Row 2: panel C ladder (left), caption (right).
    c_gs = outer[2].subgridspec(1, 2, width_ratios=[0.55, 0.45], wspace=0.10)
    ax_c = fig.add_subplot(c_gs[0, 0])
    draw_panel_c(ax_c, vulnerability)

    ax_cap = fig.add_subplot(c_gs[0, 1])
    ax_cap.set_xticks([])
    ax_cap.set_yticks([])
    for sp in ax_cap.spines.values():
        sp.set_visible(False)
    caption = (
        "Stylized wind-stilling scenario: U at every cell scaled by\n"
        "{0.85, 0.70}. Sunlight inputs unchanged. The −15 % column is\n"
        "consistent with IPCC AR6 mid-century projections for SE Brazil;\n"
        "−30 % is a stress-test upper bound. Thermal coupling is not\n"
        "modelled — only the ventilation half of the diagnosis moves.\n"
        "Cells already constrained stay there; the figure surfaces the\n"
        "marginal recruits the climate signal pulls below threshold."
    )
    ax_cap.text(0.0, 0.95, caption, transform=ax_cap.transAxes,
                ha="left", va="top", fontsize=5.8, color="#333",
                linespacing=1.45)

    # Panel labels.
    pos_a = ax_a.get_position()
    fig.text(pos_a.x0 - 0.025, pos_a.y1 + 0.005, "a",
             fontsize=9, fontweight="bold", va="bottom", ha="left")
    for i, _ in enumerate(SITE_ORDER):
        ax_i = fig.axes[i + 2]  # axes 0=a, 1=legend, 2..6=maps, 7=c, 8=caption
    # The panel label for the maps row goes above the first map.
    first_map = fig.axes[2]
    pos_m = first_map.get_position()
    fig.text(pos_m.x0 - 0.025, pos_m.y1 + 0.015, "b",
             fontsize=9, fontweight="bold", va="bottom", ha="left")
    pos_c = ax_c.get_position()
    fig.text(pos_c.x0 - 0.04, pos_c.y1 + 0.005, "c",
             fontsize=9, fontweight="bold", va="bottom", ha="left")

    # Figure-level title.
    fig.text(0.5, 0.985,
             "Climate stress test: wind stilling shifts the 4-state distribution non-uniformly across typologies",
             ha="center", va="top", fontsize=8.5, fontweight="bold",
             color="#1a1a1a")
    fig.text(0.5, 0.968,
             "Synthetic CFD; U scaled uniformly to mimic the IPCC AR6 wind-stilling signal for SE Brazil. "
             "Thermal coupling not modelled — only ventilation moves.",
             ha="center", va="top", fontsize=5.5, style="italic", color="#666")

    out_png = EXPORTS_DIR / "fig_0_6_climate_stress.png"
    out_svg = EXPORTS_DIR / "fig_0_6_climate_stress.svg"
    print(f"Saving {out_png.name} + {out_svg.name} ...")
    fig.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.05,
                facecolor="white")
    fig.savefig(out_svg, format="svg", bbox_inches="tight", pad_inches=0.05,
                facecolor="white")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
