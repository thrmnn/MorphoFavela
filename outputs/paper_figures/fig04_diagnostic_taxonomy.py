#!/usr/bin/env python3
"""Fig 04 — Cross-site diagnostic taxonomy (Nature Cities sizing).

Six-panel composite for the BRISA paper:
  A–E  Per-site 4-state diagnostic maps (adequate / sunlight constraint /
       ventilation constraint / compound constraint).
  F    Typology stacked bar (hillside vs flatland aggregate).

Maps are rendered by classifying each site's grid on the fly. Per-panel
share annotations and the typology aggregates are loaded from the canonical
interim taxonomy JSON (outputs/brisa_ventilation_fix/
taxonomy_interim_lambda_f.json) so the printed numbers match the manuscript
exactly. Threshold is interim/relative: λf > 2.75 (pooled p75 of corrected
λf) ∧ winter direct sun < 2 h.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches as mpatches
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import (
    PROJECT_ROOT,
    SITE_LABELS,
    SIZE_DOUBLE_TALL,
    apply_style,
    save_fig,
)

SITES = ["vidigal", "rocinha", "complexo_do_alemao", "maré", "riodaspedras"]
HILLSIDE = {"vidigal", "rocinha", "complexo_do_alemao"}

THRESHOLD_SUN_HRS = 2.0
THRESHOLD_LAMBDA_F = 2.75  # interim relative pre-screen (pooled p75 of corrected λf)

TAXONOMY_JSON = (
    PROJECT_ROOT
    / "outputs"
    / "brisa_ventilation_fix"
    / "taxonomy_interim_lambda_f.json"
)


def load_taxonomy() -> dict:
    with open(TAXONOMY_JSON) as f:
        return json.load(f)

STATE_NAMES = ("adequate", "sunlight_constraint", "ventilation_constraint", "compound_constraint")
STATE_INT = {
    "adequate": 0,
    "sunlight_constraint": 1,
    "ventilation_constraint": 2,
    "compound_constraint": 3,
    "nodata": 4,
}
STATE_COLORS = ["#FFFFFF", "#D9D9D9", "#7F7F7F", "#111111", "#F4F0E8"]
STATE_LEGEND = [
    ("adequate", "adequate (both pass)"),
    ("sunlight_constraint", f"sunlight constraint (<{THRESHOLD_SUN_HRS:.0f} h winter sun)"),
    ("ventilation_constraint", f"ventilation constraint (λ$_f$ > {THRESHOLD_LAMBDA_F:.2f})"),
    ("compound_constraint", "compound constraint (both)"),
]


def aggregate_solar(
    grid: gpd.GeoDataFrame, solar_path: Path, max_radius: float = 25.0, k: int = 3
) -> gpd.GeoDataFrame:
    sol = gpd.read_file(solar_path)[["solar_hours_winter", "geometry"]]
    j = gpd.sjoin(sol, grid[["zone_id", "geometry"]], how="inner", predicate="within")
    primary = j.groupby("zone_id")["solar_hours_winter"].median().reset_index()
    grid = grid.merge(primary, on="zone_id", how="left")
    missing = grid["solar_hours_winter"].isna()
    if missing.any():
        sol_pts = np.array([(g.x, g.y) for g in sol.geometry])
        sol_vals = sol["solar_hours_winter"].to_numpy()
        tree = cKDTree(sol_pts)
        cents = grid.loc[missing, ["centroid_x", "centroid_y"]].to_numpy()
        dists, idxs = tree.query(cents, k=k, distance_upper_bound=max_radius)
        with np.errstate(invalid="ignore"):
            valid = np.isfinite(dists)
        idxs = np.where(valid, idxs, 0)
        vals = sol_vals[idxs]
        vals = np.where(valid, vals, np.nan)
        with np.errstate(invalid="ignore"):
            cell_vals = np.nanmedian(vals, axis=1)
        grid.loc[missing, "solar_hours_winter"] = cell_vals
    return grid


def classify_int(grid: gpd.GeoDataFrame) -> np.ndarray:
    sun = grid["solar_hours_winter"]
    vent = grid["lambda_f_mean"]
    state = np.full(len(grid), STATE_INT["nodata"], dtype=int)
    built = (grid["lambda_p"].fillna(0) > 0.01) | (grid["building_count"] > 0)
    sun_known = sun.notna()
    vent_known = vent.notna()
    both = built & sun_known & vent_known
    sun_fail = both & (sun < THRESHOLD_SUN_HRS)
    vent_fail = both & (vent > THRESHOLD_LAMBDA_F)
    state[both & ~sun_fail & ~vent_fail] = STATE_INT["adequate"]
    state[both & sun_fail & ~vent_fail] = STATE_INT["sunlight_constraint"]
    state[both & ~sun_fail & vent_fail] = STATE_INT["ventilation_constraint"]
    state[both & sun_fail & vent_fail] = STATE_INT["compound_constraint"]
    return state


def load_and_classify(site: str) -> tuple[gpd.GeoDataFrame, np.ndarray]:
    grid = gpd.read_file(
        PROJECT_ROOT / "outputs" / site / "morphometrics" / "grid" / "grid_metrics.gpkg"
    )
    sol_path = PROJECT_ROOT / "outputs" / site / "morphometrics" / "svf" / "svf_streets_solar.gpkg"
    grid = aggregate_solar(grid, sol_path)
    return grid, classify_int(grid)


def draw_site_panel(
    ax,
    site: str,
    grid: gpd.GeoDataFrame,
    state: np.ndarray,
    panel: str,
    canonical_shares: dict,
    canonical_n: int,
) -> dict:
    """Render per-site map; annotation text uses canonical shares from the
    taxonomy JSON, NOT the on-the-fly classification, so printed numbers
    match the manuscript exactly."""
    ax.set_facecolor("white")
    cmap = ListedColormap(STATE_COLORS)
    norm = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], ncolors=5)
    grid_to_plot = grid.copy()
    grid_to_plot["state"] = state
    classified = grid_to_plot[grid_to_plot["state"] != STATE_INT["nodata"]]
    classified.plot(
        ax=ax, column="state", cmap=cmap, norm=norm, edgecolor="#FFFFFF", linewidth=0.05
    )
    bldg_path = PROJECT_ROOT / "data" / site / "buildings_extended_300m.gpkg"
    if bldg_path.exists():
        bldg = gpd.read_file(bldg_path)
        if bldg.crs != grid.crs:
            bldg = bldg.to_crs(grid.crs)
        bldg.boundary.plot(ax=ax, color="#000000", linewidth=0.10, alpha=0.4)
    bbox = classified.total_bounds
    pad = 15.0
    ax.set_xlim(bbox[0] - pad, bbox[2] + pad)
    ax.set_ylim(bbox[1] - pad, bbox[3] + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    label = SITE_LABELS[site]
    typology = "hillside" if site in HILLSIDE else "flatland"
    ax.set_title(
        f"({panel}) {label}  ·  {typology}  ·  n={canonical_n:,}",
        loc="left",
        fontsize=7.5,
        color="#222222",
        pad=2.5,
    )
    s = canonical_shares
    ax.text(
        0.02,
        0.04,
        f"compound {s['compound_constraint'] * 100:.1f}% · "
        f"vent. {s['ventilation_constraint'] * 100:.1f}% · "
        f"sun. {s['sunlight_constraint'] * 100:.1f}% · "
        f"adeq. {s['adequate'] * 100:.1f}%",
        transform=ax.transAxes,
        fontsize=5.0,
        color="#444444",
        va="bottom",
    )
    return s


def draw_typology_panel(ax, taxonomy: dict) -> None:
    """Pull hillside / flatland / all aggregates directly from the canonical
    taxonomy JSON so the bar shares match the manuscript verbatim."""
    aggregates = taxonomy["aggregates"]
    rows = []
    for label in ("hillside", "flatland", "all"):
        agg = aggregates[label]
        shares = {k: agg["shares"][k] for k in STATE_NAMES}
        total = agg["n_classified_cells"]
        rows.append((label, shares, total))

    labels = [f"{lab}\n(n={t:,})" for lab, _, t in rows]
    bottoms = np.zeros(len(rows))
    colors_for_states = {
        "adequate": "#FFFFFF",
        "sunlight_constraint": "#D9D9D9",
        "ventilation_constraint": "#7F7F7F",
        "compound_constraint": "#111111",
    }
    edge = "#444444"
    for k, _ in STATE_LEGEND:
        vals = np.array([r[1][k] for r in rows]) * 100.0
        ax.barh(
            labels,
            vals,
            left=bottoms,
            color=colors_for_states[k],
            edgecolor=edge,
            linewidth=0.5,
            height=0.62,
        )
        for i, v in enumerate(vals):
            if v >= 4:
                text_color = "#FFFFFF" if k == "compound_constraint" else "#222222"
                ax.text(
                    bottoms[i] + v / 2,
                    i,
                    f"{v:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=6.0,
                    color=text_color,
                )
        bottoms += vals
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of built cells (%)", fontsize=6.5)
    ax.tick_params(axis="x", labelsize=6.0)
    ax.tick_params(axis="y", labelsize=7.0)
    ax.set_title("(F) Typology aggregate", loc="left", fontsize=7.5, color="#222222", pad=4)
    ax.invert_yaxis()
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.4)
    ax.spines["left"].set_linewidth(0.4)


def draw_shared_legend(fig) -> None:
    handles = []
    for key, label in STATE_LEGEND:
        color = {
            "adequate": "#FFFFFF",
            "sunlight_constraint": "#D9D9D9",
            "ventilation_constraint": "#7F7F7F",
            "compound_constraint": "#111111",
        }[key]
        handles.append(
            mpatches.Patch(facecolor=color, edgecolor="#444444", linewidth=0.6, label=label)
        )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.012),
        ncol=4,
        frameon=False,
        fontsize=6.5,
        handlelength=1.8,
        handleheight=1.0,
        columnspacing=2.0,
    )


def main() -> None:
    apply_style()
    taxonomy = load_taxonomy()
    fig = plt.figure(figsize=SIZE_DOUBLE_TALL)
    gs = fig.add_gridspec(
        nrows=4,
        ncols=4,
        height_ratios=[1.0, 1.0, 1.0, 0.65],
        hspace=0.18,
        wspace=0.10,
    )
    # Panels A–E layout (3 hillside row + 2 flatland row → use sites in narrative order)
    axes_layout = {
        "A": (gs[0, 0:2], "vidigal"),
        "B": (gs[0, 2:4], "rocinha"),
        "C": (gs[1, 0:2], "complexo_do_alemao"),
        "D": (gs[1, 2:4], "riodaspedras"),
        "E": (gs[2, :], "maré"),  # widest because maré is geographically largest
    }
    for panel, (gsspec, site) in axes_layout.items():
        ax = fig.add_subplot(gsspec)
        grid, state = load_and_classify(site)
        canonical = taxonomy["per_site"][site]
        draw_site_panel(
            ax,
            site,
            grid,
            state,
            panel,
            canonical_shares=canonical["shares"],
            canonical_n=canonical["n_classified"],
        )

    ax_f = fig.add_subplot(gs[3, :])
    draw_typology_panel(ax_f, taxonomy)
    draw_shared_legend(fig)
    save_fig(fig, "fig04_diagnostic_taxonomy")


if __name__ == "__main__":
    main()
