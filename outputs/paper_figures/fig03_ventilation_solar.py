#!/usr/bin/env python3
"""Fig 03 — Frontal-area density and winter direct-sun across five favelas.

Four panels in a 4-row layout (rows 1+2 are choropleths, rows 3+4 are ridges):
  Row 1 — (A) Per-site dissolved-λf choropleths (party-wall corrected),
          5 panels + shared colorbar; flow-regime band markers on the bar.
  Row 2 — (B) Per-site winter-sun choropleths, 5 panels + shared colorbar.
  Row 3 — (C) λf distribution as a ridge plot, with the Oke/Grimmond-&-Oke
          flow-regime bands (isolated < 0.15 ≤ wake < 0.65 ≤ skimming) shaded.
  Row 4 — (D) Winter-sun-hour distribution as a ridge plot, WHO 2 h marker.

DELIVERED vs PROVISIONAL asymmetry (the framing of this figure):
  · The SUN axis (rows B, D) is DELIVERED — measured winter direct-sun hours,
    a per-cell outcome that stands on its own (WHO 2 h floor).
  · The λf axis (rows A, C) is PROVISIONAL — geometry classifies the flow
    *regime* (isolated / wake / skimming, Grimmond & Oke 1999); it does NOT
    deliver a per-cell ventilation adequacy. Adequacy (age-of-air τ) is
    CFD-gated and deliberately unquantified here. The skimming onset
    (dissolved λf ≥ 0.65) is the regime boundary, not a failure threshold.
λf is dissolved (touching footprints unioned into blocks before projecting);
canonical source outputs/brisa_ventilation_fix/lambda_f_canonical.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import (
    PROJECT_ROOT,
    SITE_COLORS,
    SITE_LABELS,
    WIDTH_DOUBLE,
    apply_style,
    save_fig,
)

SITES = ["vidigal", "rocinha", "complexo_do_alemao", "riodaspedras", "maré"]

THRESHOLD_SUN_HRS = 2.0

# Oke (1988) / Grimmond & Oke (1999) flow-regime bands on dissolved λf.
# These are REGIME boundaries, not failure thresholds: geometry says which
# regime, not whether a cell ventilates (that is CFD-gated, see docstring).
ISOLATED_MAX = 0.15
SKIMMING_MIN = 0.65
REGIME_FILL = {"isolated": "#FDD9A8", "wake": "#F08A3C", "skimming": "#B2182B"}

LAMBDA_F_CMAP = "Greys"  # sequential, deliberately not a red=alarm map
SUN_CMAP = "cividis"

# dissolved λf: pooled p95 ≈ 1.79, p99 ≈ 2.34 — cap at 2.0 so the 0.65 skimming
# onset sits at a readable ~1/3 of the range and the body is not compressed.
LAMBDA_VMIN, LAMBDA_VMAX = 0.0, 2.0
SUN_VMIN, SUN_VMAX = 0.0, 10.5


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


def load_grid(site: str) -> gpd.GeoDataFrame:
    grid = gpd.read_file(
        PROJECT_ROOT / "outputs" / site / "morphometrics" / "grid" / "grid_metrics.gpkg"
    )
    grid = aggregate_solar(
        grid, PROJECT_ROOT / "outputs" / site / "morphometrics" / "svf" / "svf_streets_solar.gpkg"
    )
    built = (grid["lambda_p"].fillna(0) > 0.01) | (grid["building_count"] > 0)
    return grid[built].copy()


def choropleth_panel(
    ax,
    gdf: gpd.GeoDataFrame,
    column: str,
    cmap: str,
    vmin: float,
    vmax: float,
    title: str | None = None,
) -> None:
    valid = gdf[column].notna() & (gdf[column] > -np.inf)
    plot_g = gdf[valid].copy()
    plot_g[column] = plot_g[column].clip(upper=vmax)
    plot_g.plot(
        ax=ax, column=column, cmap=cmap, vmin=vmin, vmax=vmax, edgecolor="none", linewidth=0.0
    )
    bbox = plot_g.total_bounds
    pad = 15.0
    ax.set_xlim(bbox[0] - pad, bbox[2] + pad)
    ax.set_ylim(bbox[1] - pad, bbox[3] + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=6.5, color="#222222", pad=2)


def add_vertical_colorbar(
    ax_target,
    cmap: str,
    vmin: float,
    vmax: float,
    markers: list[tuple[float, str]],
    label: str,
) -> None:
    """Attach a tall thin colorbar to the right of ax_target.

    ``markers`` is a list of (value, label) dashed lines on the bar — one for a
    single WHO floor, two for the isolated/wake/skimming regime boundaries.
    """
    fig = ax_target.figure
    pos = ax_target.get_position()
    cb_x = pos.x1 + 0.012
    cb_w = 0.012
    cb_y = pos.y0 + pos.height * 0.04
    cb_h = pos.height * 0.92
    cax = fig.add_axes([cb_x, cb_y, cb_w, cb_h])
    sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="vertical")
    cb.outline.set_linewidth(0.4)
    cb.set_label(label, fontsize=5.8, color="#222222", labelpad=2.0)
    cb.ax.tick_params(labelsize=5.2, color="#444444", length=2.0)
    for value, mlabel in markers:
        cax.axhline(value, color="#000000", lw=1.0, ls="--")
        cax.text(
            2.2,
            value,
            mlabel,
            fontsize=5.5,
            ha="left",
            va="center",
            color="#222222",
            transform=cax.get_yaxis_transform(),
        )


def _darken(hex_color: str, factor: float = 0.55) -> tuple[float, float, float]:
    rgb = tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (1, 3, 5))
    return tuple(c * factor for c in rgb)


def _row_tag(fig, axes, text: str, color: str) -> None:
    """Short bold tag in the clear left margin of a choropleth row, rotated to
    run up the row (DELIVERED/PROVISIONAL). Kept out of the per-panel titles."""
    pos = axes[0].get_position()
    fig.text(pos.x0 - 0.052, pos.y0 + pos.height / 2, text, fontsize=6.0,
             fontweight="bold", ha="center", va="center", rotation=90, color=color)


def ridge_panel(
    ax,
    data: dict,
    threshold: float,
    xlabel: str,
    xlim: tuple[float, float],
    threshold_label: str,
    title: str,
    fail_side: str,
    bw_method: float = 0.10,
    regime_bands: list[tuple[float, float, str]] | None = None,
    hatch_fail: bool = True,
) -> None:
    n_sites = len(SITES)
    row_step = 1.6
    offsets = np.arange(n_sites) * row_step
    height_scale = 0.95
    # Regime bands sit behind the ridges: a PROVISIONAL backdrop (which flow
    # regime), not a failure shading. Drawn first so ridges read on top.
    if regime_bands:
        for lo, hi, color in regime_bands:
            ax.axvspan(lo, hi, color=color, alpha=0.16, linewidth=0, zorder=0)
    for i, site in enumerate(SITES):
        vals = data[site]
        vals = vals[(vals >= xlim[0]) & (vals <= xlim[1])]
        if len(vals) < 30:
            continue
        try:
            kde = gaussian_kde(vals, bw_method=bw_method)
            xs = np.linspace(xlim[0], xlim[1], 400)
            density = kde(xs)
            # A delta-like atom at zero (sun) would saturate the per-ridge
            # max and flatten the body; normalise to a robust quantile and
            # clip so the spike reads as "tall" without erasing structure.
            norm = np.quantile(density, 0.985)
            density = np.clip(density / norm, 0, 1.5) * height_scale
            base = offsets[i]
            top = base + density
            fill = SITE_COLORS[site]
            edge = _darken(fill)
            ax.fill_between(xs, base, top, color=fill, alpha=0.65, linewidth=0)
            if hatch_fail:
                fail = xs >= threshold if fail_side == "right" else xs <= threshold
                ax.fill_between(
                    xs,
                    base,
                    top,
                    where=fail,
                    color=edge,
                    alpha=0.9,
                    hatch="////",
                    linewidth=0,
                )
            ax.plot(xs, top, color=edge, lw=0.6)
        except Exception:
            pass
    ax.axvline(threshold, color="#222222", lw=0.8, ls="--", alpha=0.9)
    ax.text(
        threshold,
        offsets[-1] + row_step * 0.55,
        f"{threshold_label}",
        fontsize=5.5,
        ha="center",
        va="bottom",
        color="#222222",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#888888", linewidth=0.4),
    )
    ax.set_yticks(offsets)
    ax.set_yticklabels([SITE_LABELS[s] for s in SITES], fontsize=6.0)
    ax.set_ylim(offsets[0] - row_step * 0.35, offsets[-1] + row_step * 1.05)
    ax.invert_yaxis()
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel, fontsize=6.5)
    ax.tick_params(axis="x", labelsize=6)
    ax.set_title(title, loc="left", fontsize=7.5, pad=3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.4)
    ax.spines["left"].set_linewidth(0.4)


def main() -> None:
    apply_style()
    grids = {s: load_grid(s) for s in SITES}

    fig = plt.figure(figsize=(WIDTH_DOUBLE, WIDTH_DOUBLE * 1.15))
    # Leave column 6 reserved for the colorbars next to the last choropleth.
    gs = fig.add_gridspec(
        nrows=4,
        ncols=5,
        height_ratios=[0.9, 0.9, 0.50, 0.50],
        hspace=0.55,
        wspace=0.08,
    )

    # Row A — λf choropleths
    row_a_axes = []
    for i, site in enumerate(SITES):
        ax = fig.add_subplot(gs[0, i])
        prefix = "(A) " if i == 0 else ""
        choropleth_panel(
            ax,
            grids[site],
            "lambda_f_mean",
            LAMBDA_F_CMAP,
            LAMBDA_VMIN,
            LAMBDA_VMAX,
            title=f"{prefix}{SITE_LABELS[site]}",
        )
        row_a_axes.append(ax)

    # Row B — winter-sun choropleths
    row_b_axes = []
    for i, site in enumerate(SITES):
        ax = fig.add_subplot(gs[1, i])
        prefix = "(B) " if i == 0 else ""
        choropleth_panel(
            ax,
            grids[site],
            "solar_hours_winter",
            SUN_CMAP,
            SUN_VMIN,
            SUN_VMAX,
            title=f"{prefix}{SITE_LABELS[site]}",
        )
        row_b_axes.append(ax)

    # Apply layout adjustment before colorbar placement so positions are stable.
    fig.canvas.draw()

    add_vertical_colorbar(
        row_a_axes[-1],
        LAMBDA_F_CMAP,
        LAMBDA_VMIN,
        LAMBDA_VMAX,
        [(ISOLATED_MAX, "0.15"), (SKIMMING_MIN, "0.65")],
        r"dissolved $\lambda_f$",
    )
    add_vertical_colorbar(
        row_b_axes[-1],
        SUN_CMAP,
        SUN_VMIN,
        SUN_VMAX,
        [(THRESHOLD_SUN_HRS, "2 h")],
        "winter direct-sun h",
    )

    # DELIVERED vs PROVISIONAL row tags — the figure's framing.
    _row_tag(fig, row_a_axes, r"$\lambda_f$ · PROVISIONAL", "#8C5A0F")
    _row_tag(fig, row_b_axes, "sun · DELIVERED", "#2E5E1F")

    # Row C — λf ridge with Oke/GO flow-regime bands (PROVISIONAL: regime, not adequacy)
    ax_c = fig.add_subplot(gs[2, :])
    lambda_data = {s: grids[s]["lambda_f_mean"].dropna().values for s in SITES}
    ridge_panel(
        ax_c,
        lambda_data,
        SKIMMING_MIN,
        r"dissolved $\lambda_f$ (frontal-area density)",
        xlim=(0.0, 2.5),
        threshold_label=r"skimming onset, $\lambda_f = 0.65$",
        title=r"(C) $\lambda_f$ per favela — flow-regime bands "
        r"(isolated $<$ 0.15 $\leq$ wake $<$ 0.65 $\leq$ skimming); regime tendency, not adequacy",
        fail_side="right",
        regime_bands=[
            (0.0, ISOLATED_MAX, REGIME_FILL["isolated"]),
            (ISOLATED_MAX, SKIMMING_MIN, REGIME_FILL["wake"]),
            (SKIMMING_MIN, 2.5, REGIME_FILL["skimming"]),
        ],
        hatch_fail=False,
    )

    # Row D — winter-sun ridge (DELIVERED: real per-cell failure below WHO 2 h, hatched)
    ax_d = fig.add_subplot(gs[3, :])
    sun_data = {s: grids[s]["solar_hours_winter"].dropna().values for s in SITES}
    ridge_panel(
        ax_d,
        sun_data,
        THRESHOLD_SUN_HRS,
        "winter-solstice direct-sun hours",
        xlim=(0.0, 10.5),
        threshold_label="WHO 2 h",
        title="(D) Winter-solstice direct-sun hours per favela — DELIVERED (hatched mass < WHO 2 h)",
        fail_side="left",
    )

    save_fig(fig, "fig03_ventilation_solar", gate=True)


if __name__ == "__main__":
    main()
