#!/usr/bin/env python3
"""Fig 04 — Cross-site diagnostic taxonomy (Nature Cities sizing).

Layout:
  A–E  Per-site 4-state diagnostic maps (adequate / sunlight constraint /
       ventilation constraint / compound constraint) in a colourblind-safe
       cividis categorical palette (deficit toward dark blue, never alarm red),
       with the two geometric flow-regime axis states hatched (ordinal, not a
       per-cell calibrated field).
  F    Typology stacked bar (hillside vs flatland aggregate).
  G    Within-fabric clustering (not a ranking): per-site Moran's I of the
       compound state, sites in fixed typology order.
  H    Side table: per-site state shares (moved off the maps) + the
       compound-state clustering statistics (Moran's I, black–black join-count
       z, share of compound cells in patches ≥ 5 cells / 500 m²), sites in
       fixed typology order. No parcel-scale patch-size column (clearance read).

A full-width in-pixel anti-misuse banner and a per-panel within-fabric stamp
carry the framing inside the frame so it survives a crop of any single panel.

Maps are rendered by classifying each site's grid on the fly. Per-panel
share annotations and the typology aggregates are loaded from the canonical
regime-stratified taxonomy JSON (outputs/brisa_ventilation_fix/
taxonomy_regime.json) so the printed shares match the manuscript exactly.
The spatial-clustering statistics are computed here from the same on-the-fly
classification the maps render, and written to
outputs/brisa_ventilation_fix/compound_spatial_clustering.json so the
manuscript numbers are traceable.

Ventilation axis is the REGIME boundary (PI-adopted 2026-06-25): dissolved
λf > 0.65 (skimming flow, Grimmond & Oke 1999) ∧ winter direct sun < 2 h.
This replaces the deprecated λf > 2.75 relative pre-screen. Geometry
classifies the flow *regime*, delivered and read ordinally; CFD validates the
regime ordering and adds hyper-local detail, and does not gate this result. A
calibrated per-cell age-of-air field is future work.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import esda
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from libpysal.weights import W
from matplotlib import patches as mpatches
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.spatial import cKDTree
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import (
    PROJECT_ROOT,
    SITE_LABELS,
    SITE_TYPES,
    SIZE_DOUBLE_TALL,
    apply_style,
    save_fig,
)

# Fixed typology order (hillside Vidigal/Rocinha/Alemão morro, then flatland
# Rio das Pedras/Maré) — used for panels G and H so neither reads as a severity
# league table. Never sort by deficit or Moran's I.
SITES = ["vidigal", "rocinha", "complexo_do_alemao", "riodaspedras", "maré"]

THRESHOLD_SUN_HRS = 2.0
THRESHOLD_LAMBDA_F = 0.65  # skimming-flow boundary (Grimmond & Oke 1999), dissolved λf

GRID_SPACING_M = 10.0  # regular analysis grid; one cell = 100 m²
PATCH_MIN_CELLS = 5  # ≥ 5 contiguous cells = ≥ 500 m² compound cluster

TAXONOMY_JSON = (
    PROJECT_ROOT
    / "outputs"
    / "brisa_ventilation_fix"
    / "taxonomy_regime.json"
)
CLUSTERING_JSON = (
    PROJECT_ROOT
    / "outputs"
    / "brisa_ventilation_fix"
    / "compound_spatial_clustering.json"
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

# Colourblind-safe CATEGORICAL palette sampled from cividis so the deficit axis
# runs toward DARK BLUE (mirroring fig_solar_deficit), never an alarm-red danger
# ramp. Adequate = light (cividis high end); the compound deficit reads darkest.
# Lightness is monotonic so the four states stay separable in greyscale and for
# deuteranopes; redundant per-state hatching (below) reinforces that.
_CIVIDIS = plt.get_cmap("cividis")
STATE_FILL = {
    "adequate": mcolors.to_hex(_CIVIDIS(0.93)),
    "sunlight_constraint": mcolors.to_hex(_CIVIDIS(0.60)),
    "ventilation_constraint": mcolors.to_hex(_CIVIDIS(0.32)),
    "compound_constraint": mcolors.to_hex(_CIVIDIS(0.05)),
    "nodata": "#EEEAE1",
}
# Hatch marks the two states that carry the geometric flow-regime axis (ordinal,
# not a per-cell calibrated field) so they read as texture-distinct from the
# solar-only states, a cue redundant with colour. Single-direction and coarse;
# empty = solid (adequate + sunlight rest on the delivered solar axis).
STATE_HATCH = {
    "adequate": "",
    "sunlight_constraint": "",
    "ventilation_constraint": "//",
    "compound_constraint": "//",
}
FLOW_REGIME_STATES = ("ventilation_constraint", "compound_constraint")


def _luminance(c) -> float:
    r, g, b = mcolors.to_rgb(c)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _tint(c, keep: float = 0.20) -> str:
    """Light column-tint of a state fill (blend toward white)."""
    r, g, b = mcolors.to_rgb(c)
    return mcolors.to_hex((r * keep + (1 - keep), g * keep + (1 - keep), b * keep + (1 - keep)))


# White overlaid text/hatch where the fill is dark.
STATE_DARK = {k: _luminance(v) < 0.5 for k, v in STATE_FILL.items() if k != "nodata"}
# Hatch stroke colour: a light tint of each state's own fill (off pure white) so
# the texture reads as belonging to the state, not as a white grid overlay.
STATE_HATCH_COLOR = {k: _tint(STATE_FILL[k], keep=0.45) for k in FLOW_REGIME_STATES}
STATE_COLORS = [STATE_FILL[s] for s in (*STATE_NAMES, "nodata")]
STATE_LEGEND = [
    ("adequate", "adequate (both pass)"),
    ("sunlight_constraint", f"sunlight constraint (<{THRESHOLD_SUN_HRS:.0f} h winter sun)"),
    ("ventilation_constraint", f"ventilation constraint (skimming: λ$_f$ > {THRESHOLD_LAMBDA_F:.2f})"),
    ("compound_constraint", "compound constraint (both)"),
]
COMPOUND_INK = STATE_FILL["compound_constraint"]


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


# ── Spatial clustering of the compound state ─────────────────────────────────
def _lattice_neighbors(g: gpd.GeoDataFrame) -> dict[int, list[int]]:
    """Rook (edge-sharing) contiguity among classified cells, built directly
    from the regular grid lattice — robust and far faster than polygon weights."""
    x0 = g["centroid_x"].min()
    y0 = g["centroid_y"].min()
    ix = np.round((g["centroid_x"].to_numpy() - x0) / GRID_SPACING_M).astype(int)
    iy = np.round((g["centroid_y"].to_numpy() - y0) / GRID_SPACING_M).astype(int)
    pos = {(a, b): k for k, (a, b) in enumerate(zip(ix, iy))}
    neighbors: dict[int, list[int]] = {}
    for k, (a, b) in enumerate(zip(ix, iy)):
        nb = []
        for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = pos.get((a + da, b + db))
            if j is not None:
                nb.append(j)
        neighbors[k] = nb
    return neighbors


def _patch_sizes(neighbors: dict[int, list[int]], mask: np.ndarray) -> list[int]:
    """Connected-component sizes of the True cells in `mask` over the contiguity
    graph (the contiguous-patch-size distribution of the compound state)."""
    members = set(np.where(mask)[0])
    seen: set[int] = set()
    sizes: list[int] = []
    for s in members:
        if s in seen:
            continue
        stack = [s]
        seen.add(s)
        size = 0
        while stack:
            c = stack.pop()
            size += 1
            for nb in neighbors[c]:
                if nb in members and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def compound_clustering(g: gpd.GeoDataFrame, state: np.ndarray) -> dict:
    """Moran's I, black–black join-count z, and patch-size distribution for the
    compound state, restricted to classified cells. Permutation-based (999)."""
    classified = state != STATE_INT["nodata"]
    gc = g[classified].reset_index(drop=True)
    st = state[classified]
    y = (st == STATE_INT["compound_constraint"]).astype(int)
    neighbors = _lattice_neighbors(gc)
    w = W(neighbors, silence_warnings=True)

    w.transform = "r"
    moran = esda.Moran(y, w, permutations=999)
    w.transform = "b"
    jc = esda.Join_Counts(y, w, permutations=999)
    bb_z = float((jc.bb - jc.sim_bb.mean()) / jc.sim_bb.std())

    sizes = _patch_sizes(neighbors, y.astype(bool))
    n_comp = int(y.sum())
    big = [s for s in sizes if s >= PATCH_MIN_CELLS]
    frac_big = float(sum(big) / n_comp) if n_comp else 0.0
    return {
        "n_classified": int(len(gc)),
        "n_compound": n_comp,
        "morans_i": float(moran.I),
        "morans_p_sim": float(moran.p_sim),
        "bb_joins_obs": float(jc.bb),
        "bb_joins_exp": float(jc.sim_bb.mean()),
        "bb_join_z": bb_z,
        "bb_p_sim": float(jc.p_sim_bb),
        "n_patches": len(sizes),
        "largest_patch_cells": int(sizes[0]) if sizes else 0,
        "largest_patch_m2": int((sizes[0] if sizes else 0) * GRID_SPACING_M**2),
        "median_patch_cells": int(np.median(sizes)) if sizes else 0,
        "frac_compound_in_patches_ge5": frac_big,
        "patch_sizes": sizes,
    }


# ── Map panels ───────────────────────────────────────────────────────────────
def draw_site_panel(
    ax,
    site: str,
    grid: gpd.GeoDataFrame,
    state: np.ndarray,
    panel: str,
    canonical_n: int,
) -> None:
    """Render the per-site diagnostic map. The numeric shares now live in the
    side table (panel H), keeping the maps clean per PI feedback."""
    ax.set_facecolor("white")
    cmap = ListedColormap(STATE_COLORS)
    norm = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], ncolors=5)
    grid_to_plot = grid.copy()
    grid_to_plot["state"] = state
    classified = grid_to_plot[grid_to_plot["state"] != STATE_INT["nodata"]]
    classified.plot(
        ax=ax, column="state", cmap=cmap, norm=norm, edgecolor="#FFFFFF", linewidth=0.05
    )
    # Hatch overlay for the two flow-regime-axis states: dissolve each state's
    # cells into a region and stroke a coarse single-direction hatch on top so the
    # states read as texture-distinct at region scale (per-cell hatching on 10 m
    # squares would not resolve). The stroke is a light tint of the state's own
    # fill, so it reads as the state's texture rather than a white grid.
    for key in FLOW_REGIME_STATES:
        sub = classified[classified["state"] == STATE_INT[key]]
        if sub.empty:
            continue
        region = unary_union(sub.geometry.to_list())
        gpd.GeoSeries([region], crs=grid.crs).plot(
            ax=ax, facecolor="none", edgecolor=STATE_HATCH_COLOR[key],
            hatch=STATE_HATCH[key], linewidth=0.2, zorder=5,
        )
    bldg_path = PROJECT_ROOT / "data" / site / "buildings_extended_300m.gpkg"
    if bldg_path.exists():
        bldg = gpd.read_file(bldg_path)
        if bldg.crs != grid.crs:
            bldg = bldg.to_crs(grid.crs)
        bldg.boundary.plot(ax=ax, color="#555555", linewidth=0.08, alpha=0.18)
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
    typology = SITE_TYPES[site]  # canonical descriptor (fig_style); Alemão = "mixed"
    ax.set_title(
        f"({panel}) {label}  ·  {typology}  ·  n={canonical_n:,}",
        loc="left",
        fontsize=7.5,
        color="#222222",
        pad=2.5,
    )
    # Per-panel anti-misuse stamp so the within-fabric framing survives a crop of
    # any single site map (the banner alone would be lost on a one-panel screenshot).
    ax.text(
        0.015, 0.02, "within-fabric · not a ranking",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=4.3, color="#1A1A1A",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.72, edgecolor="none"),
        zorder=30,
    )


def draw_typology_panel(ax, taxonomy: dict) -> None:
    """Hillside / flatland / all aggregates pulled from the canonical taxonomy
    JSON so the bar shares match the manuscript verbatim."""
    aggregates = taxonomy["aggregates_lambda_f"]
    rows = []
    for label in ("hillside", "flatland", "all"):
        agg = aggregates[label]
        shares = {k: agg["shares"][k] for k in STATE_NAMES}
        total = agg["n_classified"]
        rows.append((label, shares, total))

    labels = [f"{lab}\n(n={t:,})" for lab, _, t in rows]
    bottoms = np.zeros(len(rows))
    for state_key, _ in STATE_LEGEND:
        vals = np.array([r[1][state_key] for r in rows]) * 100.0
        ax.barh(
            labels,
            vals,
            left=bottoms,
            color=STATE_FILL[state_key],
            edgecolor="#444444",
            linewidth=0.5,
            height=0.62,
        )
        for i, v in enumerate(vals):
            if v >= 4:
                text_color = "#FFFFFF" if STATE_DARK[state_key] else "#222222"
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
    ax.text(
        0.0,
        -0.34,
        "binary terrain split; Alemão (mixed) grouped with its hillside morro. "
        "Hillside > flatland compound; λf-independent H/W cross-check agrees",
        transform=ax.transAxes,
        fontsize=4.8,
        color="#666666",
        va="top",
    )
    ax.invert_yaxis()
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.4)
    ax.spines["left"].set_linewidth(0.4)


def draw_moran_panel(ax, clustering: dict[str, dict]) -> None:
    """Per-site Moran's I of the compound state — the at-a-glance evidence that
    compound cells form contiguous clusters rather than scattering."""
    sites = SITES
    vals = [clustering[s]["morans_i"] for s in sites]
    y_pos = np.arange(len(sites))
    ax.barh(y_pos, vals, color=COMPOUND_INK, height=0.6, edgecolor="none", zorder=3)
    for i, s in enumerate(sites):
        c = clustering[s]
        star = "*" if c["morans_p_sim"] < 0.05 else ""
        ax.text(
            c["morans_i"] + 0.015,
            i,
            f"{c['morans_i']:.2f}{star}",
            va="center",
            ha="left",
            fontsize=6.0,
            color="#222222",
            zorder=4,
        )
    ax.set_yticks(y_pos)
    ax.set_yticklabels([SITE_LABELS[s] for s in sites], fontsize=6.5)
    ax.yaxis.tick_right()  # keep site labels inside panel G, off panel F
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.35)
    ax.set_xlabel(
        "Moran's I, compound state (rook, 999 perm.; all p = 0.001)", fontsize=6.0
    )
    ax.tick_params(axis="x", labelsize=5.8)
    ax.tick_params(axis="y", length=0)
    ax.set_title(
        "(G) Within-fabric clustering (not a ranking)",
        loc="left", fontsize=7.5, color="#222222", pad=4,
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.4)
    ax.spines["left"].set_linewidth(0.4)
    ax.xaxis.grid(True, color="#E6E6E6", lw=0.4, zorder=0)
    ax.set_axisbelow(True)


def draw_table_panel(ax, taxonomy: dict, clustering: dict[str, dict]) -> None:
    """Side table: per-site state shares (off the maps) + compound-state
    clustering statistics. Replaces the inline map annotations."""
    ax.axis("off")
    # "max upgrade unit (ha)" column dropped: a parcel-scale patch size invites a
    # clearance/site-works reading. Sites stay in fixed typology order (SITES).
    col_labels = [
        "site", "n", "adeq", "sun", "vent", "comp", "Moran I", "z$_{BB}$", "% in ≥500 m²",
    ]
    col_widths = [0.22, 0.10, 0.09, 0.09, 0.09, 0.09, 0.11, 0.09, 0.12]
    tint = {k: _tint(STATE_FILL[k]) for k in STATE_NAMES}
    cell_text = []
    cell_colors = []
    for site in SITES:
        s = taxonomy["per_site"][site]["lambda_f_axis"]["shares"]
        n = taxonomy["per_site"][site]["lambda_f_axis"]["n_classified"]
        c = clustering[site]
        star = "*" if c["morans_p_sim"] < 0.05 else ""
        row = [
            SITE_LABELS[site],
            f"{n:,}",
            f"{s['adequate'] * 100:.0f}%",
            f"{s['sunlight_constraint'] * 100:.0f}%",
            f"{s['ventilation_constraint'] * 100:.0f}%",
            f"{s['compound_constraint'] * 100:.0f}%",
            f"{c['morans_i']:.2f}{star}",
            f"{c['bb_join_z']:.0f}",
            f"{c['frac_compound_in_patches_ge5'] * 100:.0f}%",
        ]
        cell_text.append(row)
        rc = ["white"] * len(col_labels)
        # Tint the four state-share columns with their (light) state colour.
        rc[2] = tint["adequate"]
        rc[3] = tint["sunlight_constraint"]
        rc[4] = tint["ventilation_constraint"]
        rc[5] = tint["compound_constraint"]
        cell_colors.append(rc)

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        cellLoc="center",
        colLoc="center",
        loc="center",
        bbox=[0.0, 0.0, 1.0, 0.86],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6.0)
    for (r, col), cell in tbl.get_celld().items():
        cell.set_width(col_widths[col])
        cell.set_linewidth(0.3)
        cell.set_edgecolor("#CCCCCC")
        if r == 0:
            cell.set_facecolor("#33373B")
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_fontsize(5.8)
        elif col == 0:
            cell.set_text_props(ha="left")
    ax.set_title(
        "(H) Per-site state shares and within-fabric clustering statistics",
        loc="left",
        fontsize=7.5,
        color="#222222",
        pad=2,
    )
    ax.text(
        0.0,
        -0.06,
        "shares from canonical taxonomy JSON · clustering on classified cells "
        "(rook contiguity, 999 perm.) · * Moran p = 0.001\n"
        "vent/comp columns from the geometric flow-regime axis (ordinal) · investment "
        "units, never dwellings; sites in fixed typology order, not ranked; not a clearance basis",
        transform=ax.transAxes,
        fontsize=4.8,
        color="#666666",
        va="top",
    )


def draw_anti_misuse_banner(fig) -> None:
    """High-contrast in-pixel banner spanning the figure width. Load-bearing
    anti-weaponization gate: the within-fabric / not-a-ranking / not-a-clearance
    framing must be legible inside the frame, not only in the table footnote."""
    fig.add_artist(
        mpatches.Rectangle(
            (0.015, 0.957), 0.97, 0.039,
            transform=fig.transFigure, facecolor="#002b62",
            edgecolor="none", zorder=1000, clip_on=False,
        )
    )
    fig.text(
        0.5, 0.983,
        "Within-fabric adequacy deficit, not a cross-site ranking or clearance basis.",
        ha="center", va="center", fontsize=8.0, fontweight="bold",
        color="#FFFFFF", zorder=1001,
    )
    fig.text(
        0.5, 0.966,
        "Investment units, not dwellings. Input data not open.",
        ha="center", va="center", fontsize=6.8, color="#E6ECF5", zorder=1001,
    )


def draw_shared_legend(fig) -> None:
    handles = []
    for key, label in STATE_LEGEND:
        hatch = STATE_HATCH[key] or None
        # Hatched swatches stroke a light tint of the state's own fill (off pure
        # white) to match the map; solid states keep a dark border.
        edge = STATE_HATCH_COLOR[key] if hatch else "#444444"
        handles.append(
            mpatches.Patch(
                facecolor=STATE_FILL[key], edgecolor=edge, hatch=hatch,
                linewidth=0.6, label=label,
            )
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
        title="hatched = geometric flow-regime axis (ordinal), not a per-cell calibrated field; deficit runs toward dark blue",
        title_fontsize=6.0,
    )


def main() -> None:
    apply_style()
    plt.rcParams["hatch.linewidth"] = 0.35  # thin hatch so the flow-regime texture stays crisp
    taxonomy = load_taxonomy()
    fig = plt.figure(figsize=SIZE_DOUBLE_TALL)
    gs = fig.add_gridspec(
        nrows=5,
        ncols=4,
        height_ratios=[1.0, 1.0, 1.0, 0.85, 0.95],
        hspace=0.5,
        wspace=0.28,
    )
    axes_layout = {
        "A": (gs[0, 0:2], "vidigal"),
        "B": (gs[0, 2:4], "rocinha"),
        "C": (gs[1, 0:2], "complexo_do_alemao"),
        "D": (gs[1, 2:4], "riodaspedras"),
        "E": (gs[2, :], "maré"),  # widest because maré is geographically largest
    }
    clustering: dict[str, dict] = {}
    for panel, (gsspec, site) in axes_layout.items():
        ax = fig.add_subplot(gsspec)
        grid, state = load_and_classify(site)
        clustering[site] = compound_clustering(grid, state)
        draw_site_panel(
            ax, site, grid, state, panel,
            canonical_n=taxonomy["per_site"][site]["lambda_f_axis"]["n_classified"],
        )

    write_clustering_json(clustering)

    draw_typology_panel(fig.add_subplot(gs[3, 0:2]), taxonomy)
    draw_moran_panel(fig.add_subplot(gs[3, 2:4]), clustering)
    draw_table_panel(fig.add_subplot(gs[4, :]), taxonomy, clustering)
    draw_shared_legend(fig)
    draw_anti_misuse_banner(fig)
    save_fig(fig, "fig04_diagnostic_taxonomy", gate=True)


def write_clustering_json(clustering: dict[str, dict]) -> None:
    pooled_sizes: list[int] = []
    for c in clustering.values():
        pooled_sizes.extend(c["patch_sizes"])
    pooled_sizes.sort(reverse=True)
    morans = [c["morans_i"] for c in clustering.values()]
    out = {
        "method": (
            "rook contiguity on the regular 10 m analysis grid (classified cells "
            "only); Moran's I and black–black join counts with 999 permutations; "
            "patch = connected component of compound cells."
        ),
        "grid_spacing_m": GRID_SPACING_M,
        "patch_min_cells": PATCH_MIN_CELLS,
        "per_site": {
            s: {k: v for k, v in c.items() if k != "patch_sizes"}
            for s, c in clustering.items()
        },
        "pooled": {
            "morans_i_min": float(min(morans)),
            "morans_i_max": float(max(morans)),
            "largest_patch_cells": int(pooled_sizes[0]) if pooled_sizes else 0,
            "n_patches": len(pooled_sizes),
            "patch_size_histogram": pooled_sizes,
        },
    }
    CLUSTERING_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(CLUSTERING_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {CLUSTERING_JSON.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
