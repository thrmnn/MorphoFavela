#!/usr/bin/env python3
"""Figure 0.3 — Environmental performance from CFD and solar campaigns.

Four panels:

  A. Wind speed at 1.5 m for 4 representative campaign patches spanning
     a 2x2 of typology (hillside / flatland) x density (open / dense).
     Each patch heatmap is climatology-weighted from 8 wind directions
     using the site's INMET wind rose. A small wind rose inset sits in
     the corner of each patch.
  B. Winter-solstice direct-sun-hours for the SAME 4 patches.
  C. Cell-level ACH distribution across all ~7,800 CFD-covered cells
     (5 sites). Sub-threshold bars shaded in alarm red.
  D. Per-point winter sun-hour distribution across all street-level
     solar samples (5 sites once Maré finishes). Sub-threshold bars
     shaded in alarm red.

Note. CFD inputs are SYNTHETIC pending the real campaign. The script
will swap to real outputs once they land in `data/{site}/cfd_results/`
without code changes.

Run:
    python docs/manuscript/figures/fig_0_3_performance.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic_2d

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "outputs" / "paper_figures"))
from fig_style import (
    SITE_ORDER,
    WIDTH_DOUBLE,
    add_north_arrow,
    add_scalebar,
    apply_style,
    clean_map_axes,
)

EXPORTS_DIR = Path(__file__).resolve().parent / "exports"
EXPORTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Patch selection — 2x2 of typology x density
# ---------------------------------------------------------------------------
# Selected by closest-SVF match within each (typology, density) bin.
# Re-run scripts/run_campaign_sampling.py and these may shift; the picker
# below recomputes them from per_patch_indicators.csv at script start.
HILLSIDE_SITES = {"vidigal", "rocinha"}
MIXED_SITES = {"complexo_do_alemao"}
FLATLAND_SITES = {"riodaspedras", "maré"}

# Targets for 2x2 patch selection: (typology, density) -> target SVF.
TARGETS = {
    ("hillside", "dense"): 0.15,
    ("hillside", "open"): 0.55,
    ("flatland", "dense"): 0.20,
    ("flatland", "open"): 0.60,
}
PATCH_GRID = [
    ("hillside", "dense"),
    ("hillside", "open"),
    ("flatland", "dense"),
    ("flatland", "open"),
]

# WHO thresholds for the diagnostic anchors.
WHO_ACH = 0.5         # air changes per hour
WHO_SUN_HOURS = 2.0   # hours/day on winter solstice

# Diagnostic colours (consistent with Fig 0.1 taxonomy).
ALARM_RED = "#c0392b"     # WHO violation
NEUTRAL_BAR = "#cfcfcf"   # above-threshold histogram bars
BUILDING_GREY = "#3a3a3a"
BLDG_OUTLINE = "#1a1a1a"

# Patch heatmap window (square half-side, in metres).
PATCH_WIN_M = 60.0
GRID_N = 48  # cells per side; coarser bin = fewer NaN gaps, smoother heatmap

WIND_CMAP = "viridis"
SUN_CMAP = "inferno_r"
WIND_VMIN, WIND_VMAX = 0.0, 4.0     # m/s
SUN_VMIN, SUN_VMAX = 0.0, 10.0      # hours


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def load_wind_rose(site: str) -> dict:
    with open(PROJECT_ROOT / "data" / site / "wind_rose.json") as f:
        return json.load(f)


def load_buildings(site: str) -> gpd.GeoDataFrame:
    path = PROJECT_ROOT / "data" / site / "buildings_extended_300m.gpkg"
    return gpd.read_file(path)


def load_solar_points(site: str) -> gpd.GeoDataFrame | None:
    path = PROJECT_ROOT / "outputs" / site / "morphometrics" / "svf" / "svf_streets_solar.gpkg"
    if not path.exists():
        return None
    return gpd.read_file(path)


def select_patches() -> list[dict]:
    """Pick the 2x2 of patches across all 5 sites, closest-SVF to TARGETS."""
    dfs = []
    for s in SITE_ORDER:
        df = pd.read_csv(PROJECT_ROOT / "outputs" / s / "cfd_analysis" / "per_patch_indicators.csv")
        df["site"] = s
        dfs.append(df)
    all_p = pd.concat(dfs, ignore_index=True)
    def _typ(s: str) -> str:
        if s in HILLSIDE_SITES:
            return "hillside"
        if s in MIXED_SITES:
            return "mixed"
        return "flatland"
    all_p["typology"] = all_p["site"].apply(_typ)
    all_p["density"] = pd.cut(all_p["lambda_p"], [0, 0.5, 1.01],
                              labels=["open", "dense"])
    picks = []
    for typ, dens in PATCH_GRID:
        target = TARGETS[(typ, dens)]
        pool = all_p[(all_p.typology == typ) & (all_p.density == dens)].copy()
        pool["dist"] = (pool["svf"] - target).abs()
        pick = pool.sort_values("dist").iloc[0]
        picks.append(dict(
            patch_id=pick.patch_id, site=pick.site,
            typology=typ, density=dens,
            cx=float(pick.center_x), cy=float(pick.center_y),
            svf=float(pick.svf), lambda_p=float(pick.lambda_p),
            U_mean=float(pick.annual_U_mean),
            stag=float(pick.annual_stagnation_frac),
        ))
    return picks


# ---------------------------------------------------------------------------
# Climatology-weighted CFD grid
# ---------------------------------------------------------------------------
DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def climatology_grid(patch_id: str, site: str, cx: float, cy: float,
                     half_win: float = PATCH_WIN_M, n: int = GRID_N) -> tuple:
    """Bin each direction's sample points to a common grid and frequency-weight."""
    rose = load_wind_rose(site)
    freqs = rose["frequencies"]

    x_edges = np.linspace(cx - half_win, cx + half_win, n + 1)
    y_edges = np.linspace(cy - half_win, cy + half_win, n + 1)

    weighted = np.zeros((n, n))
    coverage = np.zeros((n, n))  # how much frequency mass landed per cell
    cfd_root = PROJECT_ROOT / "data" / site / "cfd_results" / patch_id
    for d in DIRECTIONS:
        sp_path = cfd_root / d / "sample_points.csv"
        if not sp_path.exists():
            continue
        df = pd.read_csv(sp_path)
        # Restrict to the analysis window for speed.
        m = ((df.x >= x_edges[0]) & (df.x <= x_edges[-1])
             & (df.y >= y_edges[0]) & (df.y <= y_edges[-1]))
        df = df[m]
        if df.empty:
            continue
        stat, _, _, _ = binned_statistic_2d(
            df.x.to_numpy(), df.y.to_numpy(), df.U_mag.to_numpy(),
            statistic="mean", bins=[x_edges, y_edges]
        )
        f = freqs.get(d, 0.0)
        # binned_statistic_2d uses (x, y) -> shape (nx, ny); transpose so y is row.
        stat = stat.T
        valid = ~np.isnan(stat)
        weighted[valid] += f * stat[valid]
        coverage[valid] += f

    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(coverage > 0, weighted / coverage, np.nan)
    return result, x_edges, y_edges


# ---------------------------------------------------------------------------
# Panel A: wind speed maps
# ---------------------------------------------------------------------------
def draw_wind_rose_inset(parent_ax, rose: dict, anchor=(0.86, 0.18),
                         size: float = 0.24) -> None:
    """Compact wind rose inset; bars highlight prevailing direction."""
    fig = parent_ax.figure
    bb = parent_ax.get_position()
    w = bb.width * size
    h = bb.height * size
    x = bb.x0 + bb.width * anchor[0] - w / 2
    y = bb.y0 + bb.height * anchor[1] - h / 2

    # White-disc backing so the rose sits readably on the heatmap.
    backing = fig.add_axes([x - w * 0.08, y - h * 0.08,
                            w * 1.16, h * 1.16])
    backing.set_xticks([])
    backing.set_yticks([])
    for s in backing.spines.values():
        s.set_visible(False)
    backing.add_patch(plt.Circle((0.5, 0.5), 0.5, transform=backing.transAxes,
                                 facecolor="white", edgecolor="none",
                                 alpha=0.85, zorder=0))
    backing.set_xlim(0, 1)
    backing.set_ylim(0, 1)
    backing.set_aspect("equal")

    rose_ax = fig.add_axes([x, y, w, h], projection="polar")
    rose_ax.set_facecolor("none")

    angles_deg = {"N": 90, "NE": 45, "E": 0, "SE": -45,
                  "S": -90, "SW": -135, "W": 180, "NW": 135}
    freqs = rose["frequencies"]
    fmax = max(freqs.values())
    prevailing = max(freqs, key=freqs.get)
    width = np.deg2rad(38)
    for d, freq in freqs.items():
        theta = np.deg2rad(angles_deg[d])
        if d == prevailing:
            color = "#c0392b"  # alarm red highlight on prevailing direction
        else:
            shade = 0.75 - 0.45 * (freq / fmax)
            color = str(shade)
        rose_ax.bar(theta, freq, width=width, bottom=0.0,
                    color=color, edgecolor="white",
                    linewidth=0.3, alpha=0.95)
    rose_ax.set_yticks([])
    rose_ax.set_xticks([])
    rose_ax.set_theta_zero_location("E")
    rose_ax.set_theta_direction(1)
    rose_ax.spines["polar"].set_visible(False)
    rose_ax.grid(False)
    rose_ax.text(np.deg2rad(90), fmax * 1.45, "N", ha="center", va="center",
                 fontsize=4, fontweight="bold", color="#222",
                 clip_on=False)


def draw_patch_wind_cell(ax, patch: dict, buildings: gpd.GeoDataFrame,
                         rose: dict, show_wind_rose: bool = True) -> object:
    grid, xe, ye = climatology_grid(
        patch["patch_id"], patch["site"], patch["cx"], patch["cy"]
    )
    extent = (xe[0], xe[-1], ye[0], ye[-1])
    cmap = plt.get_cmap(WIND_CMAP).copy()
    cmap.set_bad(alpha=0.0)  # NaN cells render fully transparent
    # Fill small NaN gaps via nearest-neighbour to avoid speckle. Cells
    # that remain NaN after a few iterations stay transparent (they're
    # genuinely outside the CFD coverage, e.g. interior of a building).
    filled = grid.copy()
    for _ in range(3):
        nan_mask = ~np.isfinite(filled)
        if not nan_mask.any():
            break
        # Shifted neighbour fill
        for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            shifted = np.roll(filled, (dy, dx), axis=(0, 1))
            fillable = nan_mask & np.isfinite(shifted)
            filled[fillable] = shifted[fillable]
            nan_mask = ~np.isfinite(filled)
    masked = np.ma.array(filled, mask=~np.isfinite(filled))
    im = ax.imshow(
        masked, extent=extent, origin="lower",
        cmap=cmap, vmin=WIND_VMIN, vmax=WIND_VMAX,
        interpolation="bilinear", zorder=1,
    )

    # Building footprints, intersected with the window.
    win = buildings.cx[xe[0]:xe[-1], ye[0]:ye[-1]]
    if not win.empty:
        win.plot(ax=ax, facecolor=BUILDING_GREY,
                 edgecolor=BLDG_OUTLINE, linewidth=0.15, zorder=3)

    ax.set_xlim(xe[0], xe[-1])
    ax.set_ylim(ye[0], ye[-1])
    ax.set_aspect("equal")
    clean_map_axes(ax)

    # Patch label — small box inside the cell (top-left).
    ax.text(0.03, 0.97,
            f"{patch['patch_id']}\n{patch['typology']} · {patch['density']}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=5.5, fontweight="bold", color="#111",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                      edgecolor="none", alpha=0.85))

    # Numeric tag for SVF / lambda_p (bottom-left).
    ax.text(0.03, 0.03,
            f"SVF {patch['svf']:.2f}  λp {patch['lambda_p']:.2f}",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=5, color="white",
            bbox=dict(boxstyle="round,pad=0.16", facecolor="black",
                      edgecolor="none", alpha=0.55))

    if show_wind_rose:
        draw_wind_rose_inset(ax, rose, anchor=(0.82, 0.82), size=0.30)
    return im


# ---------------------------------------------------------------------------
# Panel B: winter sun-hours maps
# ---------------------------------------------------------------------------
def solar_window_for_patch(patch: dict, half_win: float = PATCH_WIN_M):
    pts = load_solar_points(patch["site"])
    if pts is None:
        return None
    x = pts.geometry.x
    y = pts.geometry.y
    m = ((x >= patch["cx"] - half_win) & (x <= patch["cx"] + half_win)
         & (y >= patch["cy"] - half_win) & (y <= patch["cy"] + half_win))
    return pts[m]


def draw_patch_solar_cell(ax, patch: dict, buildings: gpd.GeoDataFrame) -> object:
    pts = solar_window_for_patch(patch)
    half = PATCH_WIN_M
    cx, cy = patch["cx"], patch["cy"]
    extent = (cx - half, cx + half, cy - half, cy + half)

    if pts is not None and len(pts) > 0:
        sc = ax.scatter(
            pts.geometry.x, pts.geometry.y,
            c=pts["solar_hours_winter"],
            cmap=SUN_CMAP, vmin=SUN_VMIN, vmax=SUN_VMAX,
            s=2.5, alpha=0.95, linewidths=0, zorder=2,
        )
    else:
        # Maré pending — draw a placeholder text.
        ax.text(0.5, 0.5, "Maré solar\npending",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=7, style="italic", color="#888")
        sc = None

    win = buildings.cx[extent[0]:extent[1], extent[2]:extent[3]]
    if not win.empty:
        win.plot(ax=ax, facecolor=BUILDING_GREY,
                 edgecolor=BLDG_OUTLINE, linewidth=0.15, zorder=3)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    clean_map_axes(ax)

    ax.text(0.03, 0.97,
            f"{patch['patch_id']}\n{patch['typology']} · {patch['density']}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=5.5, fontweight="bold", color="#111",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                      edgecolor="none", alpha=0.85))
    return sc


# ---------------------------------------------------------------------------
# Panels C / D: histograms with WHO threshold
# ---------------------------------------------------------------------------
def draw_threshold_histogram(ax, values: np.ndarray, threshold: float,
                             xlabel: str, who_label: str,
                             above_color: str = NEUTRAL_BAR,
                             below_color: str = ALARM_RED,
                             nbins: int = 35,
                             use_log: bool = False) -> None:
    v = values[np.isfinite(values)]
    if len(v) == 0:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", color="#888")
        return

    if use_log:
        v_pos = v[v > 0]
        bins = np.logspace(np.log10(v_pos.min()), np.log10(v_pos.max()), nbins)
        ax.set_xscale("log")
    else:
        bins = np.linspace(v.min(), v.max(), nbins)

    counts, edges = np.histogram(v, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    colors = [below_color if c < threshold else above_color for c in centers]
    ax.bar(centers, counts, width=widths, color=colors,
           edgecolor="white", linewidth=0.25, align="center", zorder=2)

    # Threshold line + label.
    ax.axvline(threshold, color=ALARM_RED, linestyle="--", linewidth=0.9,
               zorder=3)
    ax.text(threshold, ax.get_ylim()[1] * 0.95, f" {who_label}",
            color=ALARM_RED, fontsize=6, fontweight="bold",
            ha="left", va="top", clip_on=False)

    # Median + IQR markers.
    q25, med, q75 = np.percentile(v, [25, 50, 75])
    ax.axvline(med, color="#222222", linestyle="-", linewidth=0.7, zorder=4)
    ax.axvspan(q25, q75, color="#222222", alpha=0.07, zorder=1)

    # Sub-threshold count annotation.
    n_below = int((v < threshold).sum())
    pct_below = 100 * n_below / len(v)
    ax.text(0.98, 0.98,
            f"n = {len(v):,}\n{n_below:,} below WHO ({pct_below:.1f}%)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=5.5, color="#333",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                      edgecolor="#bbbbbb", linewidth=0.3, alpha=0.92))

    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_ylabel("count", fontsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=6)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    apply_style()
    warnings.filterwarnings("ignore", category=UserWarning)

    print("Selecting representative patches ...")
    patches = select_patches()
    for p in patches:
        print(f"  {p['patch_id']:8s} {p['site']:20s} "
              f"{p['typology']}/{p['density']:5s} "
              f"SVF={p['svf']:.2f} λp={p['lambda_p']:.2f}")

    # Pre-cache buildings + roses per site (4 patches → up to 4 sites).
    print("Loading site assets ...")
    sites_used = sorted({p["site"] for p in patches})
    buildings_per_site = {s: load_buildings(s) for s in sites_used}
    rose_per_site = {s: load_wind_rose(s) for s in sites_used}

    # ---- Figure layout ----
    fig = plt.figure(figsize=(WIDTH_DOUBLE, WIDTH_DOUBLE * 1.10))
    outer = gridspec.GridSpec(
        2, 2, figure=fig,
        height_ratios=[0.62, 0.38],
        hspace=0.20, wspace=0.18,
        left=0.06, right=0.96, top=0.94, bottom=0.07,
    )

    # ── Panel A: 2x2 wind speed patch grid (top-left) ─────────────────
    print("Drawing Panel A (wind) ...")
    gs_a = outer[0, 0].subgridspec(2, 2, hspace=0.10, wspace=0.10)
    a_axes = []
    a_im = None
    for idx, patch in enumerate(patches):
        r, c = divmod(idx, 2)
        ax = fig.add_subplot(gs_a[r, c])
        a_axes.append(ax)
        im = draw_patch_wind_cell(
            ax, patch,
            buildings_per_site[patch["site"]],
            rose_per_site[patch["site"]],
        )
        if im is not None:
            a_im = im
        # Scale-bar + north arrow on the top-left patch only.
        if idx == 0:
            add_scalebar(ax, length_m=20)
            add_north_arrow(ax, size=0.06)

    # Shared horizontal colorbar below panel A.
    pos = a_axes[2].get_position()
    cax_a = fig.add_axes([pos.x0, pos.y0 - 0.030,
                          a_axes[3].get_position().x1 - pos.x0, 0.012])
    cb_a = fig.colorbar(a_im, cax=cax_a, orientation="horizontal")
    cb_a.set_label("|U| at 1.5 m, climatology-weighted (m/s)", fontsize=6.5)
    cb_a.ax.tick_params(labelsize=5.5, length=2, width=0.3)
    cb_a.outline.set_visible(False)

    # Panel A title.
    a_pos = a_axes[0].get_position()
    fig.text(a_pos.x0 - 0.022, a_pos.y1 + 0.012,
             "a", fontsize=10, fontweight="bold", va="bottom", ha="left")
    fig.text(a_pos.x0 + 0.005, a_pos.y1 + 0.012,
             "Wind speed at pedestrian level (4 representative patches)",
             fontsize=7.5, va="bottom", ha="left")

    # ── Panel B: 2x2 winter sun-hours patch grid (top-right) ──────────
    print("Drawing Panel B (sun) ...")
    gs_b = outer[0, 1].subgridspec(2, 2, hspace=0.10, wspace=0.10)
    b_axes = []
    b_sc = None
    for idx, patch in enumerate(patches):
        r, c = divmod(idx, 2)
        ax = fig.add_subplot(gs_b[r, c])
        b_axes.append(ax)
        sc = draw_patch_solar_cell(ax, patch, buildings_per_site[patch["site"]])
        if sc is not None:
            b_sc = sc

    if b_sc is not None:
        pos = b_axes[2].get_position()
        cax_b = fig.add_axes([pos.x0, pos.y0 - 0.030,
                              b_axes[3].get_position().x1 - pos.x0, 0.012])
        cb_b = fig.colorbar(b_sc, cax=cax_b, orientation="horizontal")
        cb_b.set_label("Direct sunlight on winter solstice (h)", fontsize=6.5)
        cb_b.ax.tick_params(labelsize=5.5, length=2, width=0.3)
        cb_b.outline.set_visible(False)

    b_pos = b_axes[0].get_position()
    fig.text(b_pos.x0 - 0.022, b_pos.y1 + 0.012,
             "b", fontsize=10, fontweight="bold", va="bottom", ha="left")
    fig.text(b_pos.x0 + 0.005, b_pos.y1 + 0.012,
             "Direct sun hours at street level (same patches, winter solstice)",
             fontsize=7.5, va="bottom", ha="left")

    # ── Panel C: cell-level ACH histogram (bottom-left) ───────────────
    print("Drawing Panel C (ACH) ...")
    ach_dfs = []
    for s in SITE_ORDER:
        g = gpd.read_file(PROJECT_ROOT / "outputs" / s / "cfd_analysis" / "grid_with_cfd.gpkg")
        ach_dfs.append(g["annual_cfd_ach"].dropna().to_numpy())
    ach_all = np.concatenate(ach_dfs)
    ax_c = fig.add_subplot(outer[1, 0])
    draw_threshold_histogram(
        ax_c, ach_all, threshold=WHO_ACH,
        xlabel="cell-level ACH (h⁻¹)",
        who_label=f"WHO {WHO_ACH:g} ACH",
    )
    c_pos = ax_c.get_position()
    fig.text(c_pos.x0 - 0.040, c_pos.y1 + 0.018,
             "c", fontsize=10, fontweight="bold", va="bottom", ha="left")
    fig.text(c_pos.x0 - 0.018, c_pos.y1 + 0.018,
             "ACH distribution across all CFD-covered cells",
             fontsize=7.5, va="bottom", ha="left")
    # Synthetic-data caveat — placed below x-label, away from histogram.
    fig.text(c_pos.x0 + c_pos.width * 0.5, c_pos.y0 - 0.045,
             "Synthetic CFD; threshold sits off-scale until real-campaign data lands.",
             fontsize=5, style="italic", color="#888",
             ha="center", va="top")

    # ── Panel D: per-point winter sun-hours histogram (bottom-right) ──
    print("Drawing Panel D (sun) ...")
    sun_arrs = []
    sites_with_solar = []
    for s in SITE_ORDER:
        pts = load_solar_points(s)
        if pts is None:
            print(f"  ! {s}: no solar yet (Maré solar still running)")
            continue
        sun_arrs.append(pts["solar_hours_winter"].dropna().to_numpy())
        sites_with_solar.append(s)
    sun_all = np.concatenate(sun_arrs) if sun_arrs else np.array([])

    ax_d = fig.add_subplot(outer[1, 1])
    draw_threshold_histogram(
        ax_d, sun_all, threshold=WHO_SUN_HOURS,
        xlabel="winter direct-sun hours/day",
        who_label=f"WHO {WHO_SUN_HOURS:g} h",
    )
    d_pos = ax_d.get_position()
    fig.text(d_pos.x0 - 0.040, d_pos.y1 + 0.018,
             "d", fontsize=10, fontweight="bold", va="bottom", ha="left")
    fig.text(d_pos.x0 - 0.018, d_pos.y1 + 0.018,
             "Sun-hours distribution across street-level points",
             fontsize=7.5, va="bottom", ha="left")
    if "maré" not in sites_with_solar:
        fig.text(d_pos.x0 + d_pos.width * 0.5, d_pos.y0 - 0.045,
                 "Maré solar pending — n excludes Maré (will be added on rebuild).",
                 fontsize=5, style="italic", color="#888",
                 ha="center", va="top")

    # ---- Save ----
    out_png = EXPORTS_DIR / "fig_0_3_performance.png"
    out_svg = EXPORTS_DIR / "fig_0_3_performance.svg"
    print(f"Saving {out_png.name} + {out_svg.name} ...")
    fig.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.05,
                facecolor="white")
    fig.savefig(out_svg, format="svg", bbox_inches="tight", pad_inches=0.05,
                facecolor="white")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
