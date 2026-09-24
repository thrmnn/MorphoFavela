"""
Folha de Rua — per-site grid-based morphology dashboard (v3).

A3-portrait static PNG + atomic per-panel exports + low-fidelity PDF and a
web-1200 thumbnail. One parameterised entry point handles every site via
--site. Reads only existing pipeline outputs; never regenerates SVF, solar,
or the 10 m geometry grid.

v3 restructure (docs/folha_v3_spec.md, PI brief 2026-09-16): the sheet's
spine is a 10 m grid row — terrain, density, SVF, sunlight, in the PI's own
causal order — with two code-selected zoom inlets on the extreme cells the
analysis itself flags. The per-site cross-comparison strip is gone (PI: "no
need to show the other sites"); the single-site street SVF hero map and the
per-class ridgelines are superseded by the grid row (see build_dashboard's
docstring for the reasoning).

Run:
    python scripts/build_site_dashboard.py --site rocinha
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy import ndimage
from scipy.stats import pearsonr

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path("/home/theo/SCL/SCR/MorphoFavela")
# sys.path uses the RUNNING checkout's own root (may be a worktree ahead of
# ROOT, e.g. carrying a module not yet merged to the main checkout) —
# ROOT itself stays hardcoded for data/outputs paths regardless.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.svf_v2.paths import AREA_FILES
from src.viz.folha.sites import SHEET_NUMBER
from src.brisa_solar import mare_study_area as msa
from src.brisa_solar.wp07_figures import BOUNDARY_STROKE_PX

TYPOLOGY = {
    "vidigal": ("hillside canyon", "#2C5F8D"),
    "rocinha": ("hillside canyon", "#2C5F8D"),
    "complexo_do_alemao": ("hillside canyon", "#2C5F8D"),
    "maré": ("dense low-rise grid", "#6B7280"),
    "mare": ("dense low-rise grid", "#6B7280"),
    "riodaspedras": ("flat-dense alleys", "#5B7C5B"),
}

SITE_DISPLAY = {
    "vidigal": "Vidigal",
    "rocinha": "Rocinha",
    "complexo_do_alemao": "Complexo do Alemão",
    "maré": "Maré",
    "mare": "Maré",
    "riodaspedras": "Rio das Pedras",
}

STRIP_ORDER = ["vidigal", "rocinha", "complexo_do_alemao", "riodaspedras", "maré"]

# 10 m analysis grid, one row per site: outputs/<site>/geometry_indicators/
# per_patch_geometry.csv (src/brisa_solar/wp06_geometry.py). The sheet's new
# spine — terrain -> density -> SVF -> sunlight, the PI's own order (docs/
# folha_v3_spec.md), each panel a sequential colormap over the same 10 m
# lattice, same extent, same boundary outline. svf_c_p50/kwh_m2_p50 carry
# real NaNs where a cell has no nearby street observer (has_street_support
# is False upstream) — never interpolated, so the panel shows the gap
# honestly (buildings visible through it) rather than fabricating a value.
GRID_CELL_M = 10.0
GRID_LAYERS = [
    dict(key="terrain", col="slope_deg", title="Terrain", unit="slope (°)", cmap="Greys"),
    dict(key="density", col="lambda_p", title="Density", unit="λp (–)", cmap="Purples"),
    dict(key="svf", col="svf_c_p50", title="SVF", unit="SVF (–)", cmap="YlGnBu_r"),
    dict(key="sunlight", col="kwh_m2_p50", title="Sunlight", unit="kWh/m²·yr", cmap="YlOrRd"),
]
# Fraction-bounded columns get a fixed [0, 1] scale (their own definition,
# not a value read from a file); everything else is normalised per-site from
# its own 2nd/98th percentile, computed by code — never a constant tuned to
# one site (folha_v3_spec.md's standing hexbin residual applies here too).
GRID_FIXED_01 = {"density", "svf"}
GRID_LAYERS_BY_KEY = {layer["key"]: layer for layer in GRID_LAYERS}

# Zoom inlets: A = densest n_constraints==3 cluster, shown as density+SVF
# (the two spine layers that drive a constraint score); B = densest
# bottom-decile-SVF cluster, shown as SVF+sunlight (the direct causal pair
# the PI's brief and the hexbin below are both about).
ZOOM_LABELS = {"A": "constraint cluster", "B": "low-SVF cluster"}
ZOOM_INLET_LAYER_KEYS = {"A": ["density", "svf"], "B": ["svf", "sunlight"]}
ZOOM_COLORS = {"A": "#B91C1C", "B": "#C026D3"}
ZOOM_MIN_CLUSTER_CELLS = 3
# Below this the crop is ~4x4 cells: blocky, and it tells the reader nothing the
# locator rectangle on the grid row did not already (critic round 1, finding 3).
# Dropping it and saying so beats drawing a decoration.
ZOOM_MIN_USEFUL_CELLS = 25
ZOOM_SVF_DECILE = 0.10
ZOOM_PAD_CELLS = 3.0

INK = "#1A1A1A"
PAPER = "#FAFAF7"
ACCENT = "#D97706"
SVF_FILL = "#2C5F8D"
SOLAR_FILL = "#D97706"
MAGENTA = "#C026D3"
MUTED = "#6B7280"
GREEN = "#5B7C5B"
RED = "#B91C1C"
# Maré study-area community outlines: thin (BOUNDARY_STROKE_PX-derived, like
# src/brisa_solar/wp07_figures.py), a distinct colour from INK (the site
# boundary / data extent outline) so the two boundaries in play never read
# as one line, but muted — this is an administrative-unit outline for
# geographic orientation, not a value encoding.
MARE_COMMUNITY_STROKE = "#7C3AED"

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "savefig.facecolor": PAPER,
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "pdf.fonttype": 42,
})


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=7", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "0000000"


def site_paths(site: str) -> dict:
    files = AREA_FILES[site]
    raw = ROOT / "data" / site / "raw"
    out = ROOT / "outputs" / site
    return {
        "boundary": raw / files["boundary"],
        "buildings": raw / files["footprints"],
        "svf": out / "morphometrics" / "svf" / "svf_streets.gpkg",
        "solar": out / "morphometrics" / "svf" / "svf_streets_solar.gpkg",
        "segments": out / "morphometrics" / "svf" / "svf_streets_segments.gpkg",
        "observers": out / "sampling_streets" / "observers.gpkg",
        "manifest": out / "sampling_streets" / "manifest.json",
        "grid": out / "geometry_indicators" / "per_patch_geometry.csv",
    }


def load_grid_table(site: str) -> pd.DataFrame:
    """The 10 m analysis grid (src/brisa_solar/wp06_geometry.py output) —
    single source for the grid row, the zoom-inlet selection, and the
    identity card's cell count. Verified header (2026-09-16): patch_id,
    center_x, center_y, svf, lambda_p, slope_deg, ..., n_constraints — 35
    columns, one row per built 10 m cell, same lattice for all five sites."""
    return pd.read_csv(site_paths(site)["grid"])


def load_site(site: str, issues: list) -> dict:
    p = site_paths(site)
    svf = gpd.read_file(p["svf"])
    try:
        solar = gpd.read_file(p["solar"])
    except Exception as e:
        issues.append(f"{site}: solar gpkg missing or unreadable ({e})")
        solar = None
    try:
        seg = gpd.read_file(p["segments"])
    except Exception as e:
        issues.append(f"{site}: segments gpkg missing ({e})")
        seg = None
    boundary = gpd.read_file(p["boundary"])
    if boundary.crs is None or boundary.crs.to_epsg() != 31983:
        boundary = boundary.to_crs(31983)
    try:
        buildings = gpd.read_file(p["buildings"])
        if buildings.crs is None or buildings.crs.to_epsg() != 31983:
            buildings = buildings.to_crs(31983)
    except Exception as e:
        issues.append(f"{site}: buildings unreadable ({e})")
        buildings = None
    with open(p["manifest"]) as f:
        manifest = json.load(f)
    grid = load_grid_table(site)

    # Maré only: the study area (union of the 16 Redes da Maré communities ∩
    # the site data extent) governs which grid cells/observers count in
    # Maré's statistics and gets outlined on the sheet (src/brisa_solar/
    # mare_study_area.py). `boundary` above stays the DATA EXTENT (the
    # bairro) unconditionally — near_boundary/edge-halo logic in
    # compute_stats() must keep measuring against it, never the study area.
    communities = None
    excluded_communities = None
    study_area_geom = None
    study_area_mask = None
    if site in ("maré", "mare"):
        sa = msa.load_study_area()
        communities = sa["included"]
        excluded_communities = sa["excluded"]
        study_area_geom = sa["study_area"]
        study_area_mask = msa.within_mask(
            grid["center_x"].to_numpy(), grid["center_y"].to_numpy(), study_area_geom)

    return dict(
        svf=svf, solar=solar, seg=seg, boundary=boundary,
        buildings=buildings, manifest=manifest, grid=grid,
        communities=communities, excluded_communities=excluded_communities,
        study_area_geom=study_area_geom, study_area_mask=study_area_mask,
    )


def compute_stats(d: dict) -> dict:
    svf = d["svf"]
    solar = d["solar"]
    seg = d["seg"]
    boundary = d["boundary"]  # DATA EXTENT — near_boundary/edge-halo below must keep using this, never the study area
    manifest = d["manifest"]
    grid = d["grid"]
    study_area_geom = d.get("study_area_geom")
    study_area_mask = d.get("study_area_mask")

    # Maré only: "which cells/observers/buildings count in statistics" is
    # the study area, not the whole data extent (MAREBOUND). svf_stats/
    # seg_stats are the study-area-restricted observer/segment tables used
    # for every statistic BELOW this point (edge share, mean SVF, the
    # SVF x solar hexbin/Pearson r); `boundary`/`inner_union` themselves
    # stay the unfiltered DATA EXTENT throughout — see the near_boundary
    # comment below. Every other site keeps its existing whole-boundary
    # tables and counts (study_area_geom is None for them).
    if study_area_geom is not None:
        n_grid_cells = int(study_area_mask.sum()) if study_area_mask is not None else len(grid)
        area_km2 = float(study_area_geom.area / 1e6)
        obs_in_area = msa.within_mask(svf.geometry.x.to_numpy(), svf.geometry.y.to_numpy(), study_area_geom)
        svf_stats = svf.loc[obs_in_area].reset_index(drop=True)
        n_obs = int(len(svf_stats))
        if seg is not None:
            seg_c = seg.geometry.centroid
            seg_in_area = msa.within_mask(seg_c.x.to_numpy(), seg_c.y.to_numpy(), study_area_geom)
            seg_stats = seg.loc[seg_in_area].reset_index(drop=True)
        else:
            seg_stats = None
    else:
        n_grid_cells = len(grid)
        area_km2 = float(boundary.geometry.area.sum() / 1e6)
        svf_stats = svf
        n_obs = len(svf)
        seg_stats = seg
    road_km = float(seg_stats.geometry.length.sum() / 1000.0) if seg_stats is not None else float("nan")
    obs_per_km2 = n_obs / area_km2 if area_km2 > 0 else float("nan")

    # near_boundary: 15 m inward buffer of the DATA EXTENT boundary — kept
    # unconditionally on `boundary` (the bairro), never the study area.
    # Buildings actually stop at the bairro edge, not at an interior
    # community-to-community line; using the study-area edge here would
    # flag interior community borders as "edge halo", which is the bug
    # MAREBOUND's split (data extent vs. study area) exists to prevent.
    # Only the POPULATION being tested (svf_stats) narrows to the study
    # area; the boundary/buffer geometry it is tested against does not.
    inner = boundary.buffer(-15.0)
    inner_union = inner.union_all() if hasattr(inner, "union_all") else inner.unary_union
    near = ~svf_stats.geometry.within(inner_union)
    edge_share = float(near.mean())

    # length-weighted mean SVF, segment-level
    if seg_stats is not None and "svf_mean" in seg_stats.columns:
        w = seg_stats.geometry.length
        valid = seg_stats["svf_mean"].notna() & (w > 0)
        if valid.any():
            mean_svf = float(np.average(seg_stats.loc[valid, "svf_mean"], weights=w[valid]))
        else:
            mean_svf = float("nan")
    else:
        mean_svf = float(svf_stats["svf"].mean())

    # Pearson — observers not near boundary, with valid solar
    if solar is not None and "solar_hours_annual" in solar.columns:
        m = solar.set_index(["street_id", "distance_along"])
        s = svf_stats.set_index(["street_id", "distance_along"])
        merged = s[["svf", "geometry", "offset_distance"]].join(
            m[["solar_hours_annual", "sunshine_ratio_mean"]], how="inner",
        ).reset_index()
        merged_near = ~gpd.GeoSeries(merged["geometry"], crs=svf.crs).within(inner_union)
        merged["near_boundary"] = merged_near.values
        keep = ~merged["near_boundary"] & merged["svf"].notna() & merged["solar_hours_annual"].notna()
        if keep.sum() > 10:
            r, _ = pearsonr(merged.loc[keep, "svf"], merged.loc[keep, "solar_hours_annual"])
        else:
            r = float("nan")
    else:
        merged = None
        r = float("nan")

    offset_frac = manifest.get("qa", {}).get("offset_fraction", float("nan"))

    excluded = d.get("excluded_communities")
    excluded_names = ", ".join(sorted(excluded["community"])) if excluded is not None and len(excluded) else None
    communities = d.get("communities")
    n_communities = int(len(communities)) if communities is not None else None

    return dict(
        n_obs=n_obs, n_grid_cells=n_grid_cells, road_km=road_km, area_km2=area_km2,
        obs_per_km2=obs_per_km2, edge_share=edge_share,
        mean_svf=mean_svf, pearson=r, offset_frac=offset_frac,
        merged=merged, inner_union=inner_union, near=near,
        study_area_excluded_names=excluded_names, study_area_n_communities=n_communities,
    )


# --- panel renderers ---------------------------------------------------------


def draw_masthead(ax, site: str, stats: dict, sha: str, build_date: str,
                   folha_nn: str, provenance: str = "") -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # ruled top and bottom
    ax.axhline(1.0, color=INK, lw=0.4)
    ax.axhline(0.0, color=INK, lw=0.4)

    ax.text(0.005, 0.66, "MORPHOFAVELA", fontsize=14, fontweight="bold",
            family="DejaVu Sans", color=INK, va="center", ha="left")

    ax.text(0.5, 0.60, SITE_DISPLAY.get(site, site.title()),
            fontsize=30, family="DejaVu Serif", color=INK,
            ha="center", va="center")

    typ_label, typ_color = TYPOLOGY[site]
    right_lines = [
        f"Folha {folha_nn}/05 · EPSG:31983",
        f"build {build_date} · {sha}",
    ]
    ax.text(0.995, 0.84, right_lines[0], fontsize=8, family="DejaVu Sans Mono",
            ha="right", va="center", color=INK)
    ax.text(0.995, 0.64, right_lines[1], fontsize=8, family="DejaVu Sans Mono",
            ha="right", va="center", color=INK)
    ax.text(0.995, 0.42, typ_label.upper(), fontsize=9, family="DejaVu Sans",
            fontweight="bold", ha="right", va="center", color=typ_color)
    # Zoom-inlet selection rule (docs/folha_v3_spec.md: "the selection rule
    # printed in the sheet's provenance line") — states the rule AND this
    # build's actual outcome per site, not just the method.
    if provenance:
        ax.text(0.005, 0.10, provenance, fontsize=5.8, style="italic",
                family="DejaVu Sans Mono", color=MUTED, ha="left", va="center")


def draw_identity_card(ax, site: str, stats: dict) -> None:
    """Orients a reader who has never seen the site — nothing more
    (docs/folha_v3_spec.md §2): extent, grid cell count, observer count.
    Every other number that used to live here (edge share, mean SVF, the
    SVF~solar r) is read directly off the grid row / graph below instead of
    being repeated as text.

    Honesty carry-forward: n_obs is the TRUE total (len of the full street
    observer table, no decimation) — this script never draws a decimated
    map sample, so there is no separate "display sample" figure to show
    here (that distinction lives in build_html_dashboard.py's map layer,
    out of scope this round)."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    cells = [
        ("extent", f"{stats['area_km2']:.2f} km²"),
        ("grid cells (10 m)", f"{stats['n_grid_cells']:,}"),
        ("observers", f"{stats['n_obs']:,}"),
    ]
    n = len(cells)
    for i, (lab, val) in enumerate(cells):
        x0 = i / n
        x1 = (i + 1) / n
        cx = (x0 + x1) / 2
        if i > 0:
            ax.plot([x0, x0], [0.05, 0.95], color=INK, lw=0.3)
        ax.text(cx, 0.58, val, fontsize=14, family="DejaVu Sans Mono",
                ha="center", va="center", color=INK)
        ax.text(cx, 0.20, lab.upper(), fontsize=7.5, family="DejaVu Sans",
                fontweight="bold", ha="center", va="center", color=MUTED)


# --- grid row + zoom inlets --------------------------------------------------


def _grid_lattice(grid: pd.DataFrame, cell: float = GRID_CELL_M) -> dict:
    """Index every grid row onto its (ix, iy) cell in a shared, gap-free
    lattice, floored against the grid's own minimum center — the same
    convention src.brisa_solar.wp06_geometry.bin_ground_to_cells uses, so
    this never depends on a hand-picked origin."""
    x0 = float(grid["center_x"].min())
    y0 = float(grid["center_y"].min())
    ix = np.round((grid["center_x"].to_numpy() - x0) / cell).astype(np.int64)
    iy = np.round((grid["center_y"].to_numpy() - y0) / cell).astype(np.int64)
    return dict(ix=ix, iy=iy, x0=x0, y0=y0, cell=cell,
                nx=int(ix.max()) + 1, ny=int(iy.max()) + 1)


def _grid_to_2d(lat: dict, values: np.ndarray) -> np.ndarray:
    """Scatter a per-row array onto the (ny, nx) lattice. A cell with no row
    (not built, or this layer's own coverage gap) stays NaN — never
    interpolated, so a coverage gap renders as a stated gap, not a guess."""
    arr = np.full((lat["ny"], lat["nx"]), np.nan)
    arr[lat["iy"], lat["ix"]] = values
    return arr


def _grid_coords(lat: dict) -> tuple:
    """Cell-center X/Y meshgrid, shape (ny, nx), for pcolormesh(shading='nearest')."""
    xs = lat["x0"] + np.arange(lat["nx"]) * lat["cell"]
    ys = lat["y0"] + np.arange(lat["ny"]) * lat["cell"]
    return np.meshgrid(xs, ys)


def _panel_norm(key: str, values: np.ndarray) -> tuple:
    """[0, 1] for the two fraction-bounded layers (their own definition, not
    a number read from a file); every other layer is normalised from its
    own 2nd/98th percentile, computed here from the data — never a constant
    tuned to fit one site (the hexbin's standing residual, applied
    up-front to the grid row too)."""
    if key in GRID_FIXED_01:
        return 0.0, 1.0
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.nanpercentile(finite, [2.0, 98.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
        if hi <= lo:
            hi = lo + 1.0
    return float(lo), float(hi)


def _largest_cluster(lat: dict, mask: np.ndarray, min_cells: int = ZOOM_MIN_CLUSTER_CELLS):
    """8-connected components of `mask` over the site's own 10 m lattice;
    returns the biggest cluster's cell-index bounding box, or None if no
    cluster reaches `min_cells`. This — not a human picking a spot on the
    map — is the zoom-window selection rule (docs/folha_v3_spec.md: 'never
    pick a zoom window by eye; the rule must be in the code')."""
    field = np.zeros((lat["ny"], lat["nx"]), dtype=bool)
    field[lat["iy"][mask], lat["ix"][mask]] = True
    if not field.any():
        return None
    labels, n = ndimage.label(field, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return None
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    best = int(np.argmax(counts))
    n_cells = int(counts[best])
    if n_cells < min_cells:
        return None
    iy_idx, ix_idx = np.where(labels == best)
    return dict(
        n_cells=n_cells,
        ix_min=int(ix_idx.min()), ix_max=int(ix_idx.max()),
        iy_min=int(iy_idx.min()), iy_max=int(iy_idx.max()),
    )


def _cluster_bounds_xy(lat: dict, cluster: dict, pad_cells: float = ZOOM_PAD_CELLS) -> tuple:
    """Cluster's cell-index bbox -> real-world (xmin, xmax, ymin, ymax),
    padded by a fixed cell count for visual context — a formula, not an
    eyeballed crop."""
    cell = lat["cell"]
    xmin = lat["x0"] + (cluster["ix_min"] - 0.5 - pad_cells) * cell
    xmax = lat["x0"] + (cluster["ix_max"] + 0.5 + pad_cells) * cell
    ymin = lat["y0"] + (cluster["iy_min"] - 0.5 - pad_cells) * cell
    ymax = lat["y0"] + (cluster["iy_max"] + 0.5 + pad_cells) * cell
    return xmin, xmax, ymin, ymax


def select_zoom_windows(grid: pd.DataFrame, lat: dict) -> dict:
    """The two zoom-inlet selection rules, verbatim from docs/folha_v3_spec.md
    §4: A = the densest cluster of n_constraints==3 cells; B = the densest
    cluster of cells at/below svf_c_p50's bottom decile — the places the
    analysis itself flags as extreme. Either can come back not-ok ('no
    qualifying cluster') — that drops the inlet; it is never fabricated."""
    out = {}

    mask_a = (grid["n_constraints"] == 3).to_numpy()
    cluster_a = _largest_cluster(lat, mask_a)
    if cluster_a is None:
        out["A"] = dict(ok=False, reason=(
            f"no cluster of >= {ZOOM_MIN_CLUSTER_CELLS} connected n_constraints=3 "
            f"cells ({int(mask_a.sum())} such cell(s) total)"))
    elif cluster_a["n_cells"] < ZOOM_MIN_USEFUL_CELLS:
        out["A"] = dict(ok=False, reason=(
            f"largest n_constraints=3 cluster is {cluster_a["n_cells"]} cells, below the "
            f"{ZOOM_MIN_USEFUL_CELLS}-cell floor at which a crop shows more than its locator box"))
    else:
        out["A"] = dict(ok=True, bounds=_cluster_bounds_xy(lat, cluster_a), **cluster_a)

    svf_valid = grid["svf_c_p50"].notna()
    if svf_valid.sum() < 10:
        out["B"] = dict(ok=False, reason="fewer than 10 cells with svf_c_p50 — decile undefined")
    else:
        p10 = float(grid.loc[svf_valid, "svf_c_p50"].quantile(ZOOM_SVF_DECILE))
        mask_b = (grid["svf_c_p50"] <= p10).to_numpy() & svf_valid.to_numpy()
        cluster_b = _largest_cluster(lat, mask_b)
        if cluster_b is None:
            out["B"] = dict(ok=False, reason=(
                f"no cluster of >= {ZOOM_MIN_CLUSTER_CELLS} connected cells at/below "
                f"svf_c_p50's {ZOOM_SVF_DECILE:.0%} decile (p10={p10:.2f})"))
        elif cluster_b["n_cells"] < ZOOM_MIN_USEFUL_CELLS:
            out["B"] = dict(ok=False, reason=(
                f"largest bottom-decile cluster is {cluster_b["n_cells"]} cells, below the "
                f"{ZOOM_MIN_USEFUL_CELLS}-cell floor"))
        else:
            out["B"] = dict(ok=True, bounds=_cluster_bounds_xy(lat, cluster_b), p10=p10, **cluster_b)

    return out


def provenance_line(zooms: dict) -> str:
    """One line printed on the masthead (docs/folha_v3_spec.md: 'the
    selection rule printed in the sheet's provenance line'): the rule AND
    this build's actual outcome per site — not just the method."""
    parts = []
    for key in ("A", "B"):
        z = zooms.get(key, {})
        parts.append(f"{key}: n={z['n_cells']} cells" if z.get("ok")
                     else f"{key}: none ({z.get('reason', 'n/a')})")
    return ("zoom by code, 10 m lattice, 8-connected: A=largest cluster of "
            f"n_constraints==3, B=largest cluster <= svf_c_p50 p{int(ZOOM_SVF_DECILE*100)} — "
            + " · ".join(parts))


# Fixed inches, not fractions of a cell: title/colorbar strips need the
# same physical size everywhere, and the alternative — matplotlib's own
# set_aspect('equal') auto-shrink, anchored center by default — leaves a
# gap-sized-by-aspect-ratio blank band and strands a pre-shrink-anchored
# colorbar/badge far from the now-smaller map (round-1's original hero_map
# bug, in a form that would otherwise repeat for every one of the four
# narrower grid-row columns and each site's own aspect ratio).
_TITLE_RESERVE_IN = 0.20
_CBAR_RESERVE_IN = 0.34
_CBAR_GAP_IN = 0.05


def _fit_square_axes(ax) -> None:
    """Resize+reposition `ax` in place so its box has exactly the physical
    (inches) aspect ratio of its current data limits, anchored to the TOP of
    its originally-allocated cell — the map sits directly under its title
    with deterministic, computable room left below for a colorbar, instead
    of matplotlib centring an auto-shrunk box and leaving blank margin on
    both sides. Call after set_xlim/set_ylim, before drawing the colorbar."""
    fig = ax.figure
    cell = ax.get_position()
    fig_w_in, fig_h_in = fig.get_size_inches()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    data_w, data_h = xmax - xmin, ymax - ymin

    cell_w_in = cell.width * fig_w_in
    cell_h_in = max(cell.height * fig_h_in - _TITLE_RESERVE_IN - _CBAR_RESERVE_IN, 0.05)

    if data_h / data_w > cell_h_in / cell_w_in:
        box_h_in = cell_h_in
        box_w_in = box_h_in * data_w / data_h
    else:
        box_w_in = cell_w_in
        box_h_in = box_w_in * data_h / data_w

    box_w_frac = box_w_in / fig_w_in
    box_h_frac = box_h_in / fig_h_in
    box_x0 = cell.x0 + (cell.width - box_w_frac) / 2
    box_y0 = cell.y1 - _TITLE_RESERVE_IN / fig_h_in - box_h_frac
    ax.set_position([box_x0, box_y0, box_w_frac, box_h_frac])
    ax.set_aspect("equal", adjustable="box")


def draw_grid_panel(ax, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, boundary,
                     buildings, cmap: str, vmin: float, vmax: float, title: str,
                     unit: str, scalebar: bool = False, zoom_bounds: dict | None = None,
                     window: tuple | None = None, coverage_pct: float | None = None,
                     communities=None, label_communities: bool = False) -> None:
    """One grid-row map, or the same map re-rendered at a zoom `window`
    inside an inlet: buildings for context, one shared boundary outline, the
    metric as a masked pcolormesh (NaN cells stay transparent — a stated
    gap, never fabricated), a compact horizontal colorbar whose label is the
    whole caption. Title budget: 2-3 words + a unit, nothing else
    (docs/folha_v3_spec.md's text budget).

    `communities` (Maré only): the 16-community study-area outline, drawn
    THIN — BOUNDARY_STROKE_PX-derived, same technique as
    src/brisa_solar/wp07_figures.py (PI standing complaint: a stroke given
    in points reads as a hairline at one dpi and a masking slab at
    another). `label_communities=True` additionally prints each community's
    name (from the gpkg, never typed) at print-legible size — the caller
    sets this True on exactly one panel, never on every panel."""
    ax.set_facecolor(PAPER)
    if buildings is not None:
        try:
            buildings.plot(ax=ax, color="#E4E4E0", edgecolor="none", linewidth=0, zorder=1)
        except Exception:
            pass
    try:
        boundary.boundary.plot(ax=ax, color=INK, linewidth=0.5, zorder=2)
    except Exception:
        pass
    if communities is not None and len(communities):
        try:
            community_lw = BOUNDARY_STROKE_PX / ax.figure.dpi * 72.0
            communities.boundary.plot(ax=ax, color=MARE_COMMUNITY_STROKE,
                                       linewidth=community_lw, zorder=2.5)
            if label_communities:
                import matplotlib.patheffects as pe
                for row in communities.itertuples():
                    c = row.geometry.centroid
                    ax.text(c.x, c.y, row.community, fontsize=3.6,
                            color=MARE_COMMUNITY_STROKE, ha="center", va="center",
                            zorder=6, fontweight="bold",
                            path_effects=[pe.withStroke(linewidth=1.2, foreground="white")])
        except Exception:
            pass

    Zm = np.ma.masked_invalid(Z)
    pc = ax.pcolormesh(X, Y, Zm, cmap=cmap, vmin=vmin, vmax=vmax,
                        shading="nearest", zorder=3)

    if window is not None:
        xmin, xmax, ymin, ymax = window
    else:
        xmin, ymin, xmax, ymax = boundary.total_bounds
        pad_x = 0.04 * (xmax - xmin)
        pad_y = 0.04 * (ymax - ymin)
        xmin, xmax = xmin - pad_x, xmax + pad_x
        ymin, ymax = ymin - pad_y, ymax + pad_y
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    _fit_square_axes(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=9.5, color=INK, pad=3, loc="left")

    if zoom_bounds:
        for key, b in zoom_bounds.items():
            if not b.get("ok"):
                continue
            bxmin, bxmax, bymin, bymax = b["bounds"]
            ax.add_patch(Rectangle((bxmin, bymin), bxmax - bxmin, bymax - bymin,
                                   facecolor="none", edgecolor=ZOOM_COLORS[key],
                                   linewidth=1.1, zorder=5))
            ax.text(bxmax, bymax, key, fontsize=6.5, fontweight="bold",
                    color=ZOOM_COLORS[key], ha="left", va="bottom", zorder=6)

    if scalebar:
        bar_y = ymin + (ymax - ymin) * 0.02
        bar_x0 = xmin + (xmax - xmin) * 0.04
        bar_x1 = bar_x0 + 200.0
        ax.plot([bar_x0, bar_x1], [bar_y, bar_y], color=INK, lw=1.2, zorder=7)
        ax.text((bar_x0 + bar_x1) / 2, bar_y + (ymax - ymin) * 0.015,
                "200 m", fontsize=6.5, family="DejaVu Sans Mono",
                ha="center", va="bottom", color=INK, zorder=7)

    # Coverage is of BUILT cells (len(grid) — this layer's own denominator,
    # passed in by the caller), never of the lattice's full rectangular
    # envelope: most of that envelope is simply not built, which is not a
    # data gap. Conflating the two would print e.g. "cov 26%" on Maré's
    # Terrain panel — implying slope_deg is missing for 3/4 of the site —
    # when slope_deg is in fact ~100% complete for every built cell; the
    # real, load-bearing gap is svf_c_p50/kwh_m2_p50's missing street
    # support, which is what this badge is for.
    if coverage_pct is not None and coverage_pct < 99.5:
        ax.text(0.02, 0.98, f"cov {coverage_pct:.0f}%", fontsize=6, color=MUTED,
                family="DejaVu Sans Mono", ha="left", va="top",
                transform=ax.transAxes, zorder=8)

    # Colorbar directly below the now-fitted map box — its own physical
    # strip (_CBAR_RESERVE_IN), not a fraction of the original tall cell,
    # so it never floats away from a map that _fit_square_axes shrank.
    panel_box = ax.get_position()
    fig_w_in, fig_h_in = ax.figure.get_size_inches()
    cbar_w = panel_box.width
    cbar_h = 0.055 / fig_h_in
    cbar_x0 = panel_box.x0
    cbar_y0 = panel_box.y0 - _CBAR_GAP_IN / fig_h_in - cbar_h
    cbar_ax = ax.figure.add_axes([cbar_x0, cbar_y0, cbar_w, cbar_h])
    cb = plt.colorbar(pc, cax=cbar_ax, orientation="horizontal")
    cb.outline.set_linewidth(0.3)
    cb.ax.tick_params(labelsize=6, length=2, pad=1)
    cb.set_label(unit, fontsize=7, color=INK, labelpad=2)


def draw_grid_row(fig, gs_cell, d: dict, lat: dict, X: np.ndarray, Y: np.ndarray,
                   layer_arrays: dict, panel_norms: dict, windows: dict) -> list:
    """The sheet's new spine: terrain, density, SVF, sunlight — same 10 m
    lattice, same extent, aligned axes, one shared boundary outline, in the
    PI's own causal order (docs/folha_v3_spec.md)."""
    sub = gs_cell.subgridspec(1, len(GRID_LAYERS), wspace=0.12)
    # Coverage badge denominator (Maré only): the study area's own built-cell
    # count, not the whole data extent's (MAREBOUND — "re-derive the badge
    # against the new study area"). study_area_mask is over d["grid"]'s own
    # row order, so boolean-AND with notna() stays aligned.
    study_mask = d.get("study_area_mask")
    if study_mask is not None:
        n_built = int(study_mask.sum())
    else:
        n_built = len(d["grid"])
    communities = d.get("communities")
    axes = []
    for i, layer in enumerate(GRID_LAYERS):
        ax = fig.add_subplot(sub[0, i])
        notna = d["grid"][layer["col"]].notna().to_numpy()
        if study_mask is not None:
            notna = notna & study_mask
        cov = 100.0 * notna.sum() / n_built if n_built else None
        draw_grid_panel(
            ax, X, Y, layer_arrays[layer["key"]], d["boundary"], d["buildings"],
            layer["cmap"], *panel_norms[layer["key"]], layer["title"], layer["unit"],
            scalebar=(i == 0), zoom_bounds=windows, coverage_pct=cov,
            communities=communities, label_communities=(i == 0),
        )
        axes.append(ax)
    return axes


def draw_zoom_row(fig, gs_cell, d: dict, X: np.ndarray, Y: np.ndarray,
                   layer_arrays: dict, panel_norms: dict, windows: dict) -> None:
    """Two (at most three) code-selected zoom inlets, each re-rendering the
    two grid layers most relevant to why that cluster was flagged, at the
    same colour scale as the grid row above so the two read as one
    analysis. A dropped inlet states the gap in one line — never a
    fabricated window (docs/folha_v3_spec.md §4)."""
    outer = fig.add_subplot(gs_cell)
    outer.axis("off")
    bbox = outer.get_position()
    half_w = bbox.width / 2.0

    for j, key in enumerate(("A", "B")):
        h_x0 = bbox.x0 + j * half_w
        fig.text(h_x0 + half_w * 0.02, bbox.y0 + bbox.height * 0.97,
                 f"Zoom {key} · {ZOOM_LABELS[key]}", fontsize=8.5,
                 fontweight="bold", color=ZOOM_COLORS[key], ha="left", va="top")

        z = windows.get(key, {})
        if not z.get("ok"):
            fig.text(h_x0 + half_w / 2, bbox.y0 + bbox.height * 0.45,
                     f"no qualifying cluster —\n{z.get('reason', 'n/a')}",
                     fontsize=7.5, style="italic", color=MUTED,
                     ha="center", va="center", wrap=True)
            continue

        layer_keys = ZOOM_INLET_LAYER_KEYS[key]
        sub_w = half_w * 0.94 / len(layer_keys)
        for k, lkey in enumerate(layer_keys):
            layer = GRID_LAYERS_BY_KEY[lkey]
            sub_x0 = h_x0 + half_w * 0.02 + k * sub_w
            sub_y0 = bbox.y0 + bbox.height * 0.04
            sub_h = bbox.height * 0.80
            ax = fig.add_axes([sub_x0, sub_y0, sub_w * 0.94, sub_h])
            draw_grid_panel(
                ax, X, Y, layer_arrays[lkey], d["boundary"], d["buildings"],
                layer["cmap"], *panel_norms[lkey], layer["title"], layer["unit"],
                window=z["bounds"], communities=d.get("communities"),
            )


def _uses_quadrant_fallback(d: dict) -> bool:
    """True when a per-street-class panel would have to fall back to compass
    quadrants because no usable street-class column is present (the Maré
    case: no tipo_logra column). round-1 finding 3 was a panel titled
    "street class" while it actually plotted NE/SE/SW/NW — this predicate
    is what made that title conditional and honest.

    The v3 restructure (docs/folha_v3_spec.md) drops the ridgeline panels
    that used to call this (grid_row + the hexbin cover the same ground
    without a text-heavy per-class breakdown — see build_dashboard's
    docstring for the full reasoning). Kept and tested regardless: it is a
    "do not regress" honesty fix per the v3 spec's non-negotiables, and
    stays correct/available if a future cycle reinstates a per-class
    panel."""
    seg = d.get("seg")
    return not (seg is not None and "tipo_logra" in seg.columns)


def _dynamic_hexbin_gridsize(x: np.ndarray, y: np.ndarray, x_range: tuple,
                              y_range: tuple, lo: int = 12, hi: int = 45) -> tuple:
    """Bin count from the data (n, IQR), not a constant tuned to one site
    (docs/folha_v3_spec.md's standing residual: gridsize=30 read as
    near-blank on Rocinha/Alemão's lower N and tighter SVF range). One
    Freedman-Diaconis bin width per axis: bin_w = 2*IQR*n^(-1/3), gridsize =
    axis span / bin_w, clamped to [lo, hi] so a very small or very
    degenerate sample still renders a legible hexbin rather than one giant
    or thousands of empty cells."""
    def fd_bins(v: np.ndarray, span: float) -> int:
        v = v[np.isfinite(v)]
        if len(v) < 5 or span <= 0:
            return lo
        iqr = float(np.subtract(*np.percentile(v, [75, 25])))
        if iqr <= 0:
            return lo
        bin_w = 2.0 * iqr * len(v) ** (-1.0 / 3.0)
        if bin_w <= 0:
            return lo
        return int(np.clip(round(span / bin_w), lo, hi))

    nx = fd_bins(np.asarray(x, dtype=float), x_range[1] - x_range[0])
    ny = fd_bins(np.asarray(y, dtype=float), y_range[1] - y_range[0])
    return (nx, ny)


def draw_hexbin(ax, d: dict, stats: dict) -> None:
    merged = stats.get("merged")
    if merged is None:
        ax.text(0.5, 0.5, "no joinable solar data",
                ha="center", va="center", color=MUTED)
        ax.axis("off")
        return
    keep = ~merged["near_boundary"] & merged["svf"].notna() \
        & merged["solar_hours_annual"].notna()
    n_excl = (~keep).sum()
    pct_excl = n_excl / len(merged) * 100
    sub = merged.loc[keep]

    y_top = float(sub["solar_hours_annual"].max()) * 1.05 if len(sub) else 12.0
    y_range = (0.0, min(12.5, y_top))
    gridsize = _dynamic_hexbin_gridsize(
        sub["svf"].to_numpy(), sub["solar_hours_annual"].to_numpy(),
        (0.0, 1.0), y_range,
    )

    # PowerNorm (gamma<1) lifts low-count bins' visual weight instead of
    # leaving them near-white against the paper background — a linear norm
    # made Vidigal (smallest N) and especially Rocinha (lowest mean SVF, so
    # its cloud sits tightly near the origin) read as near-blank even
    # though both panels were rendering real, non-broken data (round-2
    # finding D). gridsize is now data-driven (see _dynamic_hexbin_gridsize)
    # rather than the flat 30 that under-resolved Rocinha/Alemão.
    hb = ax.hexbin(sub["svf"], sub["solar_hours_annual"], gridsize=gridsize,
                   cmap="Greys", mincnt=1, linewidths=0.0)

    # The gridsize fix alone still left Rocinha washed out: its single
    # densest bin (~4,400 points, a sharp canyon-floor peak) is >4x any
    # other site's peak while its OWN median bin sits at 6 — a PowerNorm
    # scaled to the true max compresses every other bin toward zero. Fixed
    # from this panel's own dynamic range: cap the norm at the 98th
    # percentile of its OWN nonzero bin counts (never a constant that
    # happens to suit one site) and let the colorbar's ">" arrow disclose
    # that the densest bins are clipped, rather than let them define the
    # scale for everyone else.
    counts = hb.get_array()
    vmax = float(np.percentile(counts, 98)) if len(counts) else 1.0
    vmax = max(vmax, 1.0)
    hexbin_norm = mpl.colors.PowerNorm(gamma=0.45, vmin=1.0, vmax=vmax)
    hb.set_norm(hexbin_norm)
    hb.set_clim(1.0, vmax)
    cbar_extend = "max" if len(counts) and counts.max() > vmax else "neither"

    ax.set_xlim(0, 1)
    ax.set_ylim(*y_range)
    ax.set_xlabel("SVF", fontsize=9)
    ax.set_ylabel("solar hours / day (annual mean)", fontsize=9)
    ax.tick_params(labelsize=7)

    # quantile regression q=0.5 fallback: simple linear fit on medians per bin
    try:
        from scipy.optimize import minimize
        x = sub["svf"].values
        y = sub["solar_hours_annual"].values

        def q_loss(beta):
            a, b = beta
            resid = y - (a * x + b)
            return np.sum(np.where(resid >= 0, 0.5 * resid, -0.5 * resid))
        res = minimize(q_loss, x0=[8.0, 1.5], method="Nelder-Mead")
        a, b = res.x
        xs = np.linspace(0, 1, 100)
        ax.plot(xs, a * xs + b, color=INK, lw=0.8)
    except Exception:
        pass

    if len(sub) > 10:
        r, _ = pearsonr(sub["svf"], sub["solar_hours_annual"])
        ax.text(0.98, 0.97, f"r = {r:.2f}", transform=ax.transAxes,
                fontsize=10, family="DejaVu Sans Mono", ha="right", va="top")
    # Fixed point offset, not an axes-fraction one: the hexbin panel's own
    # height now varies a lot by context (full-width sheet row vs a small
    # atom export), and a fraction like the old -0.18 scales with it —
    # on the v3 sheet's taller hexbin row that pushed this caption down
    # into the caveat strip below.
    ax.annotate(f"edge n={int(n_excl)} excluded ({pct_excl:.1f}% of network)",
                xy=(0.0, 0.0), xycoords="axes fraction",
                xytext=(0, -22), textcoords="offset points",
                fontsize=7, color=MUTED, ha="left", va="top")

    # GMM bimodality check — skip when SVF×solar is strongly linear
    # (|r| > 0.9 implies a single elongated cluster, not two modes)
    try:
        r_check, _ = pearsonr(sub["svf"], sub["solar_hours_annual"])
    except Exception:
        r_check = 0.0
    try:
        from sklearn.mixture import GaussianMixture
        X = np.column_stack([
            (sub["svf"] - sub["svf"].mean()) / sub["svf"].std(),
            (sub["solar_hours_annual"] - sub["solar_hours_annual"].mean())
            / sub["solar_hours_annual"].std(),
        ])
        if len(X) > 50 and abs(r_check) < 0.9:
            g1 = GaussianMixture(n_components=1, random_state=0).fit(X)
            g2 = GaussianMixture(n_components=2, random_state=0).fit(X)
            if g2.bic(X) < g1.bic(X) - 10:
                centroids = g2.means_
                centroids_real = np.column_stack([
                    centroids[:, 0] * sub["svf"].std() + sub["svf"].mean(),
                    centroids[:, 1] * sub["solar_hours_annual"].std()
                    + sub["solar_hours_annual"].mean(),
                ])
                ax.scatter(centroids_real[:, 0], centroids_real[:, 1],
                           marker="+", c=MAGENTA, s=80, linewidths=1.5,
                           zorder=10)
                dbic = g1.bic(X) - g2.bic(X)
                ax.text(0.02, 0.97,
                        "+ 2-component Gaussian mixture (\u0394BIC "
                        f"{dbic:.0f} vs 1) \u2014 canyon / open",
                        transform=ax.transAxes, fontsize=7.5, style="italic",
                        color=MAGENTA, va="top", ha="left")
    except Exception:
        pass

    cb_ax = ax.inset_axes([1.02, 0.0, 0.025, 1.0])
    cb = plt.colorbar(hb, cax=cb_ax, extend=cbar_extend)
    cb.ax.tick_params(labelsize=6)
    cb.set_label("count", fontsize=7)


def draw_caveats(ax, site: str, stats: dict) -> None:
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axhline(1.0, color=INK, lw=0.4)

    edge_pct = stats["edge_share"] * 100
    obs_per_km2 = stats["obs_per_km2"]
    road_km = stats["road_km"]
    n_obs = stats["n_obs"]
    offset_pct = stats["offset_frac"] * 100

    caveats = [
        ("[H1] EDGE HALO",
         f"{edge_pct:.1f}% of observers fall within 15 m of the boundary; "
         "external building footprints beyond the site are not modelled, "
         "so perimeter SVF and solar are biased high. Affected zone is "
         "hatched on the hero map."),
        ("[H2] OBSERVER DENSITY",
         f"{obs_per_km2:.0f} obs/km² ({road_km:.1f} km of network, "
         f"n={n_obs:,}). Cross-site density spreads up to 2.3×; raw counts "
         "are not directly comparable across sites."),
        ("[M2] SOLAR UNITS",
         "solar_hours_annual and sunshine_ratio_mean are canonical; the "
         "legacy solar_hours column is deprecated and not used. r̄ in "
         "[0.45, 0.85] is consistent with Rio (~22.9°S) latitude."),
        ("[M3] MEAN SVF · [L1] OFFSET",
         f"Mean SVF reported length-weighted at segment level; pre-1bffc3e "
         f"runs were observer-weighted and over-counted short segments. "
         f"{offset_pct:.1f}% of observers were repositioned > 2.5 m from "
         "the original sample location (low-confidence glyphs on map)."),
    ]
    n = len(caveats)
    for i, (head, body) in enumerate(caveats):
        x0 = i / n + 0.005
        ax.text(x0, 0.88, head, fontsize=7, fontweight="bold",
                family="DejaVu Sans Mono", color=MUTED,
                ha="left", va="top")
        ax.text(x0, 0.72, body, fontsize=7.5, color=INK,
                ha="left", va="top", wrap=True,
                family="DejaVu Sans",
                bbox=None)
        # text wrap by manual width
    ax.text(0.995, 0.05, "See technical_report §4.2, §6",
            fontsize=6.5, style="italic", color=MUTED,
            ha="right", va="bottom")


def _wrap_text(s: str, width: int) -> str:
    import textwrap
    return "\n".join(textwrap.wrap(s, width=width))


def draw_caveats_v2(ax, site: str, stats: dict) -> None:
    """Cleaner caveat strip with proper wrapping."""
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axhline(1.0, color=INK, lw=0.4)

    edge_pct = stats["edge_share"] * 100
    obs_per_km2 = stats["obs_per_km2"]
    road_km = stats["road_km"]
    n_obs = stats["n_obs"]
    offset_pct = stats["offset_frac"] * 100

    caveats = [
        ("[H1] EDGE HALO",
         f"{edge_pct:.1f}% of observers fall within 15 m of the "
         "boundary; external footprints beyond the site are not "
         "modelled, so perimeter SVF and solar are biased high."),
        ("[H2] OBSERVER DENSITY",
         f"{obs_per_km2:.0f} obs/km² ({road_km:.1f} km of "
         f"network, n={n_obs:,}). Cross-site density spreads up "
         "to 2.3×; raw counts are not directly comparable."),
        ("[M2] SOLAR UNITS",
         "solar_hours_annual and sunshine_ratio_mean are canonical; "
         "legacy solar_hours is deprecated. Sunshine ratio in "
         "[0.45, 0.85] is consistent with Rio's ~22.9°S latitude."),
        ("[M3/L1] MEAN SVF · OFFSET",
         "Mean SVF is length-weighted segment-level; pre-1bffc3e "
         f"runs were observer-weighted. {offset_pct:.1f}% of "
         "observers were repositioned > 2.5 m from their original "
         "sample location (low-confidence)."),
    ]
    if site in ("maré", "mare"):
        excluded = stats.get("study_area_excluded_names") or "none"
        n_included = stats.get("study_area_n_communities")
        caveats.append((
            "[H3] STUDY AREA",
            f"Cells/observers here: {n_included} Redes da Maré "
            f"communities, clipped to the data extent — not the whole "
            f"bairro. {excluded} lies outside the extent, excluded. "
            "Edge-halo still uses the bairro, not this outline."
        ))
    n = len(caveats)
    col_w = 1.0 / n
    wrap_width = 42 if n <= 4 else int(42 * 4 / n)
    for i, (head, body) in enumerate(caveats):
        cx = i * col_w + 0.01
        ax.text(cx, 0.92, head, fontsize=7.5, fontweight="bold",
                family="DejaVu Sans Mono", color=MUTED,
                ha="left", va="top")
        body_wrapped = _wrap_text(body, wrap_width)
        ax.text(cx, 0.78, body_wrapped, fontsize=7, color=INK,
                ha="left", va="top", family="DejaVu Sans",
                linespacing=1.35)
    ax.text(0.995, 0.04, "See technical_report §4.2, §6",
            fontsize=6.5, style="italic", color=MUTED,
            ha="right", va="bottom")


# --- atomic exports ---------------------------------------------------------


def _save_atom(out_dir: Path, name: str, renderer, figsize, **kwargs):
    fig = plt.figure(figsize=figsize, dpi=200, facecolor=PAPER)
    ax = fig.add_subplot(111)
    renderer(ax, **kwargs)
    p = out_dir / f"{name}.png"
    fig.savefig(p, dpi=200, facecolor=PAPER, bbox_inches="tight")
    plt.close(fig)
    return p


def build_dashboard(site: str) -> dict:
    """v3 layout (docs/folha_v3_spec.md): masthead, identity card, the grid
    row (terrain/density/SVF/sunlight, PI's own order), two code-selected
    zoom inlets, the SVF×solar graph, the caveat strip.

    Two panels from the pre-v3 sheet are gone, both decided this cycle:

    - The street-level SVF "hero" map + its legend are superseded by the
      grid row's own SVF panel. Keeping both would draw SVF twice at full
      sheet width for no new information, exactly the "unnecessary" the PI
      asked to cut; the grid row is also the more current column
      (svf_c_p50, WP-04's C' engine) and it now carries the north
      arrow/scalebar/200 m bar the hero map used to.
    - The two ridgeline panels (SVF and solar by street class / orientation
      quadrant) are DROPPED. They were the standing "text-heavy panels"
      critic residual (per-class n=, median ticks, sunshine-ratio badges —
      exactly the prose the v3 text budget forbids), and once the grid row
      exists they are largely redundant with it: the grid's SVF panel + the
      hexbin below already show the SVF/solar distribution and their
      relationship, just without an artificial street-class partition. The
      underlying functions were NOT deleted — only unreferenced here — and
      _uses_quadrant_fallback (the honesty helper their titles depended on)
      is kept intact and tested, in case a future cycle reinstates them.
    """
    issues: list = []
    panels: list = []

    out_dir = ROOT / "outputs" / "_distribution" / "site_dashboards" / site
    atoms = out_dir / "atoms"
    out_dir.mkdir(parents=True, exist_ok=True)
    atoms.mkdir(parents=True, exist_ok=True)
    # Clear atoms of panels the v3 restructure removed (hero map/legend,
    # both ridgelines, the cross-site strip) — a stale prior-cycle PNG left
    # sitting next to this run's fresh ones would be the same "fabricated/
    # placeholder" trap the spec warns against, just one directory listing
    # away instead of on the sheet itself.
    for stale in ("hero_map.png", "hero_legend.png", "svf_ridgeline.png",
                  "solar_ridgeline.png", "small_multiples_strip.png"):
        (atoms / stale).unlink(missing_ok=True)

    d = load_site(site, issues)
    stats = compute_stats(d)
    sha = git_sha()
    build_date = datetime.date.today().isoformat()
    folha_nn = SHEET_NUMBER.get(site, "00")

    grid = d["grid"]
    lat = _grid_lattice(grid)
    X, Y = _grid_coords(lat)
    layer_arrays = {
        layer["key"]: _grid_to_2d(lat, grid[layer["col"]].to_numpy())
        for layer in GRID_LAYERS
    }
    panel_norms = {
        key: _panel_norm(key, arr) for key, arr in layer_arrays.items()
    }
    windows = select_zoom_windows(grid, lat)
    for key, z in windows.items():
        if not z.get("ok"):
            issues.append(f"{site}: zoom inlet {key} dropped — {z['reason']}")
    provenance = provenance_line(windows)

    # A3 portrait: 297 x 420 mm → 11.69 x 16.54 in
    fig = plt.figure(figsize=(11.69, 16.54), dpi=200, facecolor=PAPER)

    # Aspect-aware row heights. A fixed near-square frame per grid panel wasted
    # most of its box on the two shape extremes: Vidigal (bbox ~0.45 tall/wide)
    # rendered as a horizontal wisp a few pixels high, Maré (~2.06) as a narrow
    # ribbon with white margins on both sides — the requested spine came out the
    # least legible thing on the sheet (critic round 1, findings 1 and 2). Each
    # panel is one quarter of the row's width, so the height that actually fits
    # the footprint is that width times the site's own bbox aspect. The scatter
    # shrinks at the same time: it had been taking as much height as the spine
    # and both zoom rows together, for a secondary consistency check.
    bx0, by0, bx1, by1 = d["boundary"].total_bounds
    site_aspect = (by1 - by0) / (bx1 - bx0) if (bx1 - bx0) else 1.0
    panel_w_frac = (0.97 - 0.04) / len(GRID_LAYERS)
    grid_row = panel_w_frac * site_aspect * (11.69 / 16.54) * 16.54
    grid_row = float(np.clip(grid_row, 1.6, 4.6))   # keep the sheet balanced at the extremes
    zoom_row = float(np.clip(grid_row * 0.85, 1.5, 3.4))
    gs = fig.add_gridspec(
        nrows=6, ncols=1,
        height_ratios=[1.3, 0.75, grid_row, zoom_row, 4.0, 1.7],
        left=0.04, right=0.97, top=0.985, bottom=0.015,
        hspace=0.30,
    )

    ax_mast = fig.add_subplot(gs[0, 0])
    draw_masthead(ax_mast, site, stats, sha, build_date, folha_nn, provenance)
    panels.append("masthead")

    ax_id = fig.add_subplot(gs[1, 0])
    draw_identity_card(ax_id, site, stats)
    panels.append("identity_card")

    draw_grid_row(fig, gs[2, 0], d, lat, X, Y, layer_arrays, panel_norms, windows)
    panels.append("grid_row")

    draw_zoom_row(fig, gs[3, 0], d, X, Y, layer_arrays, panel_norms, windows)
    panels.append("zoom_inlets")

    ax_hex = fig.add_subplot(gs[4, 0])
    draw_hexbin(ax_hex, d, stats)
    ax_hex.set_title("SVF × solar — physical consistency check",
                     fontsize=10, color=INK, pad=4, loc="left")
    panels.append("svf_solar_hexbin")

    ax_cav = fig.add_subplot(gs[5, 0])
    draw_caveats_v2(ax_cav, site, stats)
    panels.append("caveat_strip")

    a3_path = out_dir / f"folha_{site}_A3.png"
    fig.savefig(a3_path, dpi=220, facecolor=PAPER, bbox_inches=None)

    pdf_path = out_dir / f"folha_{site}.pdf"
    fig.savefig(pdf_path, facecolor=PAPER)

    # Web 1200 thumb
    web_path = out_dir / f"folha_{site}_web1200.png"
    fig.savefig(web_path, dpi=85, facecolor=PAPER)

    plt.close(fig)

    # Atomic exports — render each panel into its own figure
    try:
        _save_atom(atoms, "masthead", lambda ax: draw_masthead(
            ax, site, stats, sha, build_date, folha_nn, provenance),
                  figsize=(11.69, 1.3))
    except Exception as e:
        issues.append(f"atom masthead: {e}")

    try:
        _save_atom(atoms, "identity_card", lambda ax: draw_identity_card(
            ax, site, stats), figsize=(11.69, 0.9))
    except Exception as e:
        issues.append(f"atom identity_card: {e}")

    try:
        fig2 = plt.figure(figsize=(11.69, 4.5), dpi=200, facecolor=PAPER)
        gs2 = fig2.add_gridspec(1, 1, left=0.02, right=0.98, top=0.90, bottom=0.05)
        draw_grid_row(fig2, gs2[0, 0], d, lat, X, Y, layer_arrays, panel_norms, windows)
        fig2.savefig(atoms / "grid_row.png", dpi=200, facecolor=PAPER)
        plt.close(fig2)
    except Exception as e:
        issues.append(f"atom grid_row: {e}")

    try:
        fig2 = plt.figure(figsize=(11.69, 3.2), dpi=200, facecolor=PAPER)
        gs2 = fig2.add_gridspec(1, 1, left=0.02, right=0.98, top=0.92, bottom=0.05)
        draw_zoom_row(fig2, gs2[0, 0], d, X, Y, layer_arrays, panel_norms, windows)
        fig2.savefig(atoms / "zoom_inlets.png", dpi=200, facecolor=PAPER)
        plt.close(fig2)
    except Exception as e:
        issues.append(f"atom zoom_inlets: {e}")

    try:
        fig2 = plt.figure(figsize=(10.0, 5.0), dpi=200, facecolor=PAPER)
        ax2 = fig2.add_subplot(111)
        draw_hexbin(ax2, d, stats)
        fig2.savefig(atoms / "svf_solar_hexbin.png", dpi=200,
                     facecolor=PAPER, bbox_inches="tight")
        plt.close(fig2)
    except Exception as e:
        issues.append(f"atom svf_solar_hexbin: {e}")

    try:
        _save_atom(atoms, "caveat_strip",
                   lambda ax: draw_caveats_v2(ax, site, stats),
                   figsize=(11.69, 1.6))
    except Exception as e:
        issues.append(f"atom caveat: {e}")

    # metadata
    meta = {
        "site": site,
        "site_display": SITE_DISPLAY.get(site, site),
        "build_date": build_date,
        "git_sha7": sha,
        "panels_built": panels,
        "zoom_windows_provenance": provenance,
        "zoom_windows": {
            key: {k: v for k, v in z.items() if k != "bounds"} | (
                {"bounds": list(z["bounds"])} if z.get("ok") else {}
            )
            for key, z in windows.items()
        },
        "stats": {
            "n_obs": int(stats["n_obs"]),
            "n_grid_cells": int(stats["n_grid_cells"]),
            "road_km": float(stats["road_km"]),
            "area_km2": float(stats["area_km2"]),
            "obs_per_km2": float(stats["obs_per_km2"]),
            "edge_share": float(stats["edge_share"]),
            "mean_svf_length_weighted": float(stats["mean_svf"]),
            "pearson_svf_solar": None if np.isnan(stats["pearson"])
                                 else float(stats["pearson"]),
            "offset_fraction": float(stats["offset_frac"]),
        },
        "issues": issues,
        "outputs": {
            "A3_png": str(a3_path),
            "pdf": str(pdf_path),
            "web1200": str(web_path),
            "atoms_dir": str(atoms),
        },
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    return dict(
        a3_path=a3_path, pdf=pdf_path, web=web_path,
        metadata=out_dir / "metadata.json",
        stats=meta["stats"], issues=issues, panels=panels,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--site")
    p.add_argument("--all", action="store_true", help="build every site")
    args = p.parse_args()
    if not args.site and not args.all:
        p.error("must pass --site <name> or --all")

    sites = STRIP_ORDER if args.all else [args.site]
    failed = []
    for site in sites:
        try:
            result = build_dashboard(site)
        except Exception as e:
            print(f"[{site}] FAILED: {e}")
            failed.append(site)
            continue
        n_issues = len(result["issues"])
        print(f"[{site}] OK — {result['web']} ({n_issues} issue(s))")
        if not args.all:
            print(f"A3 PNG: {result['a3_path']}")
            print(f"PDF:    {result['pdf']}")
            print(f"web1200: {result['web']}")
            print(f"metadata: {result['metadata']}")
        if result["issues"]:
            print("ISSUES:")
            for i in result["issues"]:
                print(f"  - {i}")

    if args.all:
        print(f"\n{len(sites) - len(failed)}/{len(sites)} sites built")
        if failed:
            print(f"FAILED: {', '.join(failed)}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
