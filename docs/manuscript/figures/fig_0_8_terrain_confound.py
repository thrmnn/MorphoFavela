#!/usr/bin/env python3
"""Figure 0.8 — Terrain confound: does typology survive when slope is controlled?

The 4-state diagnostic in 0.4 and the climate-stress vulnerability ladder
in 0.6 both implicate "hillside" as the most compound-constraint-prone
typology. But the hillside label is partially proxying terrain slope.
This figure tests whether the typology effect survives slope control.

Panels
------
A   Per-site horizontal stacked bars of the 4-state shares, stratified
    by slope bin (0–5°, 5–15°, 15–25°, ≥25°). Bin-conditional state
    composition shifts; rising compound share at steeper bins quantifies
    the raw terrain confound.
B   5-site grid maps colored by slope bin, with cells in compound
    failure outlined. Spatial expression: where do compound-constraint
    cells co-locate with steep ground?
C   Typology comparison stratified by slope bin: % compound-constraint
    cells per typology, for each slope bin. If the hillside–flatland
    gap persists across slope bins, typology beats slope. If the gap
    collapses at low slope, typology was slope.

Slope is merged from `grid_metrics.gpkg` (zone_id) — about ~90 % of
cfd-covered cells have a morphometric match; cells without a slope
match are dropped from panel A/C (still shown on the map in panel B's
"unbinned" colour).

Run:
    python docs/manuscript/figures/fig_0_8_terrain_confound.py
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

SLOPE_BINS = [(0.0, 5.0), (5.0, 15.0), (15.0, 25.0), (25.0, 90.0)]
SLOPE_LABELS = ["0–5°", "5–15°", "15–25°", "≥25°"]
SLOPE_PALETTE = ["#f1eef6", "#bdc9e1", "#74a9cf", "#0570b0"]

TYPOLOGY_COLORS = {
    "Hillside": "#7a3232",
    "Mixed": "#7a5a2f",
    "Flatland": "#2f5a7a",
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


def slope_bin_index(slope: float) -> int | None:
    if not np.isfinite(slope):
        return None
    for i, (lo, hi) in enumerate(SLOPE_BINS):
        if lo <= slope < hi:
            return i
    return None


def build_site_data(site: str) -> gpd.GeoDataFrame:
    g = gpd.read_file(
        PROJECT_ROOT / "outputs" / site / "cfd_analysis" / "grid_with_cfd.gpkg"
    )
    g = _aggregate_solar_to_cells(g, site)
    g["state"] = classify(g["annual_cfd_U_mean"], g["solar_hours_winter"])

    morph = gpd.read_file(
        PROJECT_ROOT / "outputs" / site / "morphometrics" / "grid"
        / "grid_metrics.gpkg"
    )[["zone_id", "slope_deg"]]
    g = g.merge(morph, on="zone_id", how="left")
    g["slope_bin"] = g["slope_deg"].apply(slope_bin_index)
    return g


def state_fractions_by_bin(
    g: gpd.GeoDataFrame,
) -> list[dict[str, float] | None]:
    """One dict per slope bin, or None if the bin is empty for this site."""
    out: list[dict[str, float] | None] = []
    for i in range(len(SLOPE_BINS)):
        sub = g[(g["slope_bin"] == i) & g["state"].notna()]
        if len(sub) == 0:
            out.append(None)
            continue
        total = len(sub)
        out.append({k: 100.0 * (sub["state"] == k).sum() / total for k in STATE_KEYS})
    return out


# ---------------------------------------------------------------------------
# Panel A: per-site horizontal stacked bars × slope bin
# ---------------------------------------------------------------------------
def draw_panel_a(ax, fractions: dict[str, list[dict[str, float] | None]],
                 site_ok: dict[str, bool], counts: dict[str, list[int]]) -> None:
    rows = []
    labels = []
    label_colors = []
    label_weights = []
    row_n: list[int] = []

    for s in SITE_ORDER:
        if not site_ok[s]:
            continue
        for i, lab in enumerate(SLOPE_LABELS):
            row = fractions[s][i]
            if row is None or counts[s][i] < 5:
                continue
            rows.append(row)
            labels.append(f"{SITE_LABELS[s]}  {lab}")
            label_colors.append("#222" if i == 0 else "#555")
            label_weights.append("bold" if i == 0 else "normal")
            row_n.append(counts[s][i])

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
        ax.text(-2, y[i], f"n={row_n[i]}",
                ha="right", va="center", fontsize=5.0, color="#888")

    # Group separators (between sites).
    for i in range(len(SLOPE_BINS), n_rows, len(SLOPE_BINS)):
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
# Panel B: slope-binned site maps with compound-constraint cells outlined
# ---------------------------------------------------------------------------
def draw_slope_map(ax, site: str, grid: gpd.GeoDataFrame,
                   buildings: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame,
                   site_has_solar: bool) -> None:
    minx, miny, maxx, maxy = boundary.total_bounds
    pad = 0.02 * max(maxx - minx, maxy - miny)
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_aspect("equal")
    buildings.plot(ax=ax, facecolor=BUILDING_COLOR, edgecolor="none",
                   linewidth=0.0, zorder=1)

    # Cells without a slope_bin (edge/no-morph match): faint background
    unbinned = grid[grid["slope_bin"].isna()]
    if len(unbinned):
        unbinned.plot(ax=ax, facecolor=NOT_ASSESSED_COLOR, edgecolor="none",
                      alpha=0.4, zorder=2)

    # Slope-bin fills
    for i, color in enumerate(SLOPE_PALETTE):
        sub = grid[grid["slope_bin"] == i]
        if len(sub):
            sub.plot(ax=ax, facecolor=color, edgecolor="none", alpha=0.88,
                     zorder=3)

    # Compound-constraint cells outlined in red
    if site_has_solar:
        comp = grid[grid["state"] == "compound"]
        if len(comp):
            comp.plot(ax=ax, facecolor="none",
                      edgecolor=STATE_COLORS["compound"], linewidth=0.55,
                      zorder=4)

    boundary.boundary.plot(ax=ax, color="black", linewidth=0.4,
                           linestyle=(0, (3, 2)), zorder=5, alpha=0.65)
    clean_map_axes(ax)
    ew = maxx - minx
    bar = 100 if ew < 1500 else (200 if ew < 2500 else 500)
    add_scalebar(ax, length_m=bar, loc="lower left")
    add_north_arrow(ax, loc="upper right", size=0.05)
    ax.set_title(SITE_LABELS[site], fontsize=7.5, pad=4, fontweight="bold",
                 color="#222")

    # Subtitle: % cells with slope ≥ 15° and % compound
    cls = grid["slope_bin"].dropna()
    pct_steep = 100.0 * (cls >= 2).sum() / max(len(cls), 1)
    if site_has_solar:
        states = grid["state"].dropna()
        pct_comp = 100.0 * (states == "compound").sum() / max(len(states), 1)
        ax.text(0.5, -0.04,
                f"≥15° {pct_steep:.0f}%  •  compound {pct_comp:.0f}%",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=6, color="#444")
    else:
        ax.text(0.5, -0.04, f"≥15° {pct_steep:.0f}%  •  compound (pending)",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=6, color="#888", style="italic")


# ---------------------------------------------------------------------------
# Panel C: typology × slope-bin compound constraint share
# ---------------------------------------------------------------------------
def draw_panel_c(ax,
                 typology_by_bin: dict[str, list[dict[str, float] | None]],
                 typology_counts: dict[str, list[int]]) -> None:
    bins = SLOPE_LABELS
    n_bins = len(bins)
    typs = [t for t in TYPOLOGY_AGGREGATE_ORDER if t in typology_by_bin]
    n_typs = len(typs)
    bar_w = 0.78 / n_typs
    x = np.arange(n_bins)

    for ti, typ in enumerate(typs):
        comp = []
        ns = []
        for i in range(n_bins):
            entry = typology_by_bin[typ][i]
            n = typology_counts[typ][i]
            if entry is None or n < 10:
                comp.append(np.nan)
                ns.append(0)
            else:
                comp.append(entry["compound"])
                ns.append(n)
        xs = x + (ti - (n_typs - 1) / 2) * bar_w
        bars = ax.bar(xs, comp, width=bar_w * 0.92,
                      color=TYPOLOGY_COLORS[typ], edgecolor="white",
                      linewidth=0.5, label=typ, zorder=2)
        for xi, v, n in zip(xs, comp, ns):
            if np.isfinite(v):
                ax.text(xi, v + 1.0, f"{v:.0f}",
                        ha="center", va="bottom", fontsize=5.5,
                        color=TYPOLOGY_COLORS[typ], fontweight="bold")
                ax.text(xi, -2.4, f"n={n}",
                        ha="center", va="top", fontsize=4.8, color="#888")

    ax.set_xticks(x)
    ax.set_xticklabels(bins, fontsize=6.5)
    ax.set_xlabel("slope bin", fontsize=7, labelpad=2)
    ax.set_ylabel("% compound-constraint cells", fontsize=7, labelpad=2)
    ax.tick_params(axis="x", length=2, width=0.4, pad=2)
    ax.tick_params(axis="y", labelsize=6, length=2, width=0.4, pad=2)
    ax.set_ylim(0, 100)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.4)
    ax.spines["left"].set_linewidth(0.4)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#eaeaea", linewidth=0.4, zorder=0)
    ax.legend(loc="upper left", fontsize=6.5, frameon=False,
              title="typology", title_fontsize=6.5, handlelength=1.2,
              labelspacing=0.4)


def aggregate_typology(
    grids: dict[str, gpd.GeoDataFrame],
) -> tuple[dict[str, list[dict[str, float] | None]], dict[str, list[int]]]:
    by_bin: dict[str, list[dict[str, float] | None]] = {}
    counts: dict[str, list[int]] = {}
    for typ in TYPOLOGY_AGGREGATE_ORDER:
        members = [s for s in SITE_ORDER if TYPOLOGY_OF[s] == typ
                   and grids[s]["state"].notna().any()]
        by_bin[typ] = []
        counts[typ] = []
        for i in range(len(SLOPE_BINS)):
            total = 0
            comp = 0
            for s in members:
                sub = grids[s][(grids[s]["slope_bin"] == i)
                               & grids[s]["state"].notna()]
                total += len(sub)
                comp += int((sub["state"] == "compound").sum())
            if total == 0:
                by_bin[typ].append(None)
                counts[typ].append(0)
            else:
                by_bin[typ].append({"compound": 100.0 * comp / total})
                counts[typ].append(total)
    return by_bin, counts


def main() -> None:
    apply_style()
    warnings.filterwarnings("ignore", category=UserWarning, module="geopandas")
    warnings.filterwarnings("ignore", category=FutureWarning)

    print("Building slope-stratified grids ...")
    grids: dict[str, gpd.GeoDataFrame] = {}
    fractions: dict[str, list[dict[str, float] | None]] = {}
    counts: dict[str, list[int]] = {}
    site_has_solar: dict[str, bool] = {}
    site_ok: dict[str, bool] = {}
    for site in SITE_ORDER:
        g = build_site_data(site)
        grids[site] = g
        site_has_solar[site] = g["state"].notna().any()
        site_ok[site] = g["state"].notna().sum() > 0
        fractions[site] = state_fractions_by_bin(g)
        counts[site] = []
        for i in range(len(SLOPE_BINS)):
            counts[site].append(int(
                ((g["slope_bin"] == i) & g["state"].notna()).sum()
            ))
        report_bin = [
            f"{SLOPE_LABELS[i]}:{c}" for i, c in enumerate(counts[site])
        ]
        if site_has_solar[site]:
            steep_compound = (
                (g["slope_bin"] >= 2) & (g["state"] == "compound")
            ).sum()
            flat_compound = (
                (g["slope_bin"] == 0) & (g["state"] == "compound")
            ).sum()
            print(f"  {site:<22} {report_bin}  "
                  f"steep_compound={steep_compound} flat_compound={flat_compound}")
        else:
            print(f"  {site:<22} {report_bin}  (no solar — Maré pending)")

    typology_by_bin, typology_counts = aggregate_typology(grids)
    print("Typology compound% by slope bin:")
    for typ in TYPOLOGY_AGGREGATE_ORDER:
        row = []
        for i, lab in enumerate(SLOPE_LABELS):
            e = typology_by_bin[typ][i]
            n = typology_counts[typ][i]
            row.append(f"{lab}:{e['compound']:.0f}%(n={n})" if e else f"{lab}:—")
        print(f"  {typ:<10} {' '.join(row)}")

    boundaries = {s: load_boundary(s) for s in SITE_ORDER}
    buildings = {s: load_buildings(s, extended=False) for s in SITE_ORDER}

    fig = plt.figure(figsize=(WIDTH_DOUBLE, WIDTH_DOUBLE * 1.10))
    outer = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[2.0, 2.2, 1.20],
        hspace=0.32,
        left=0.05, right=0.96, top=0.93, bottom=0.06,
    )

    # Row 0: panel A on left (wide stacked bars), legends on right.
    a_gs = outer[0].subgridspec(1, 2, width_ratios=[0.62, 0.38], wspace=0.10)
    ax_a = fig.add_subplot(a_gs[0, 0])
    draw_panel_a(ax_a, fractions, site_ok, counts)

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

    handles_b = [
        Patch(facecolor=SLOPE_PALETTE[i], edgecolor="none", label=SLOPE_LABELS[i])
        for i in range(len(SLOPE_LABELS))
    ]
    handles_b.append(
        Patch(facecolor="none", edgecolor=STATE_COLORS["compound"],
              linewidth=0.8, label="compound-constraint outline")
    )
    leg2 = ax_legend.legend(handles=handles_b, loc="lower left",
                            bbox_to_anchor=(0.02, 0.02),
                            frameon=False, fontsize=6.5,
                            handlelength=1.4, labelspacing=0.45,
                            title="Slope bin (panel B)",
                            title_fontsize=6.5, borderpad=0.0)
    leg2.get_title().set_fontweight("bold")
    leg2.get_title().set_color("#222")

    # Row 1: 5-site slope-bin maps.
    maps_gs = outer[1].subgridspec(1, 5, wspace=0.02)
    for i, site in enumerate(SITE_ORDER):
        ax = fig.add_subplot(maps_gs[0, i])
        draw_slope_map(ax, site, grids[site], buildings[site],
                       boundaries[site], site_has_solar[site])

    # Row 2: panel C (typology x slope bin) left, caption right.
    c_gs = outer[2].subgridspec(1, 2, width_ratios=[0.62, 0.38], wspace=0.10)
    ax_c = fig.add_subplot(c_gs[0, 0])
    draw_panel_c(ax_c, typology_by_bin, typology_counts)

    ax_cap = fig.add_subplot(c_gs[0, 1])
    ax_cap.set_xticks([])
    ax_cap.set_yticks([])
    for sp in ax_cap.spines.values():
        sp.set_visible(False)
    # Concise reading line: gap at low slope vs high slope.
    hill = typology_by_bin.get("Hillside", [None] * 4)
    flat = typology_by_bin.get("Flatland", [None] * 4)
    h0 = hill[0]["compound"] if hill[0] else np.nan
    f0 = flat[0]["compound"] if flat[0] else np.nan
    h_steep_entry = hill[3] if hill[3] else hill[2]
    f_steep_entry = flat[3] if flat[3] else flat[2]
    h_steep = h_steep_entry["compound"] if h_steep_entry else np.nan
    f_steep = f_steep_entry["compound"] if f_steep_entry else np.nan
    gap_low = h0 - f0 if np.isfinite(h0) and np.isfinite(f0) else np.nan
    gap_steep = (h_steep - f_steep) if np.isfinite(h_steep) and np.isfinite(f_steep) else np.nan
    reading = (
        f"At slope < 5°  hillside − flatland = {gap_low:+.0f} pp\n"
        f"At steepest slope bin  hillside − flatland = {gap_steep:+.0f} pp\n"
        "\n"
        "Interpretation: if the typology gap holds across slope bins, the\n"
        "hillside label encodes morphology that is not reducible to slope\n"
        "alone (canyon depth, façade alignment, packing density). If the\n"
        "gap shrinks toward zero at low slope, typology is largely slope\n"
        "in disguise. Maré pending — flatland row uses Rio das Pedras only\n"
        "until the Maré street-solar pipeline lands."
    )
    ax_cap.text(0.0, 0.95, reading, transform=ax_cap.transAxes,
                ha="left", va="top", fontsize=5.8, color="#333",
                linespacing=1.45)

    # Panel labels.
    pos_a = ax_a.get_position()
    fig.text(pos_a.x0 - 0.025, pos_a.y1 + 0.005, "a",
             fontsize=9, fontweight="bold", va="bottom", ha="left")
    first_map = fig.axes[2]
    pos_m = first_map.get_position()
    fig.text(pos_m.x0 - 0.025, pos_m.y1 + 0.015, "b",
             fontsize=9, fontweight="bold", va="bottom", ha="left")
    pos_c = ax_c.get_position()
    fig.text(pos_c.x0 - 0.04, pos_c.y1 + 0.005, "c",
             fontsize=9, fontweight="bold", va="bottom", ha="left")

    fig.text(0.5, 0.985,
             "Terrain confound: does typology survive slope control?",
             ha="center", va="top", fontsize=8.5, fontweight="bold",
             color="#1a1a1a")
    fig.text(0.5, 0.968,
             "4-state shares stratified by terrain slope. If the hillside–flatland gap "
             "in compound constraint persists at the lowest slope bin, typology encodes "
             "morphology beyond slope.",
             ha="center", va="top", fontsize=5.5, style="italic", color="#666")

    out_png = EXPORTS_DIR / "fig_0_8_terrain_confound.png"
    out_svg = EXPORTS_DIR / "fig_0_8_terrain_confound.svg"
    print(f"Saving {out_png.name} + {out_svg.name} ...")
    fig.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.05,
                facecolor="white")
    fig.savefig(out_svg, format="svg", bbox_inches="tight", pad_inches=0.05,
                facecolor="white")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
