"""Figure 8 — cross-site street-level solar by aspect quadrant.

Tests the §5.4 hypothesis directly: hillside sites (Vidigal, Rocinha)
should show strong aspect dissociation in winter (S quadrant >> N
quadrant for hours of direct sun), while flatland sites (Rio das
Pedras, Maré) should show all four quadrants at roughly the same
height (no aspect dependence — the sun's elevation pattern dominates,
not local slope orientation). Complexo do Alemão (mixed) sits between.

Layout: 1×4 horizontal panels — winter / Mar-eq / Sep-eq / summer.
Each panel: x = aspect quadrant (N, E, S, W); y = mean street-level
hours of direct sun; one line per site, styled by typology.

Inputs per site (aspect computed at run time from the morphometric
grid via point-in-cell sjoin, matching scripts/run_aspect_analysis.py):

* ``outputs/{site}/morphometrics/grid/grid_metrics.gpkg`` —
  per-cell aspect_deg + slope_deg.
* ``outputs/{site}/morphometrics/svf/svf_streets_solar.gpkg`` —
  per-street-point seasonal solar hours.

Outputs (under ``outputs/paper_figures/exports/``):

* ``fig08_solar_cross_site.png`` / ``.svg``

Usage::

    python outputs/paper_figures/fig08_solar_cross_site.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from fig_style import (  # noqa: E402
    PROJECT_ROOT,
    SITE_COLORS,
    SITE_LABELS,
    SITE_TYPES,
    WIDTH_DOUBLE,
    apply_style,
    save_fig,
)
from src.morphometry.aspect import aspect_quadrant  # noqa: E402

SITES = ["vidigal", "rocinha", "complexo_do_alemao", "riodaspedras", "maré"]
QUADRANTS = ("N", "E", "S", "W")
SEASONS = [
    ("solar_hours_winter", "(a) winter solstice — 21 Jun"),
    ("solar_hours_equinox_mar", "(b) March equinox — 20 Mar"),
    ("solar_hours_equinox_sep", "(c) September equinox — 22 Sep"),
    ("solar_hours_summer", "(d) summer solstice — 21 Dec"),
]
# §5.4 framing: cross-site comparison of *all* streets by aspect quadrant.
# We deliberately do *not* impose a slope-threshold here (Figure 6 / §5.3
# already does the slope-restricted aspect analysis). Including flat cells
# is the whole point of the cross-site narrative — flatland sites have
# most of their streets on near-zero slope where aspect_deg is geometrically
# defined but solar-irrelevant; averaging over those cells dilutes the
# aspect signal towards a uniform per-quadrant mean. Hillside sites' cells
# are mostly on real slopes, so the aspect signal survives.

# Visual encoding: typology → linestyle, site → colour. The §5.4 narrative
# rests on hillside-vs-flatland contrast, so style this via line-shape.
TYPOLOGY_LINESTYLE = {"hillside": "-", "mixed": "-.", "flatland": "--"}


def _load_site_solar_quadrants(site: str) -> pd.DataFrame | None:
    """Per-quadrant means of all four seasonal solar columns for one site.

    Reads grid_metrics for aspect/slope and the seasonal solar gpkg for
    hours; sjoin on point-in-cell (matches run_aspect_analysis.py),
    drops near-flat cells, and returns a long-format DataFrame.
    """
    grid_path = PROJECT_ROOT / "outputs" / site / "morphometrics" / "grid" / "grid_metrics.gpkg"
    solar_path = PROJECT_ROOT / "outputs" / site / "morphometrics" / "svf" / "svf_streets_solar.gpkg"
    if not grid_path.exists() or not solar_path.exists():
        return None

    grid = gpd.read_file(grid_path)
    solar = gpd.read_file(solar_path)

    if "aspect_deg" not in grid.columns or "slope_deg" not in grid.columns:
        return None
    keep = grid[["geometry", "aspect_deg", "slope_deg"]].copy()
    if solar.crs != keep.crs:
        solar = solar.to_crs(keep.crs)

    sol = gpd.sjoin(solar, keep, how="left", predicate="within").drop(columns="index_right")
    sol = sol.dropna(subset=["aspect_deg"])  # drop points outside the grid
    if sol.empty:
        return None
    sol["quadrant"] = aspect_quadrant(sol["aspect_deg"].values)

    rows = []
    for season_col, _ in SEASONS:
        if season_col not in sol.columns:
            continue
        for q in QUADRANTS:
            sub = sol[sol["quadrant"] == q]
            rows.append(
                {
                    "site": site,
                    "season": season_col,
                    "quadrant": q,
                    "n": int(len(sub)),
                    "mean_h": float(sub[season_col].mean()) if len(sub) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _draw_panel(
    ax: plt.Axes, df: pd.DataFrame, season_col: str, title: str, ymax: float
) -> None:
    ax.set_title(title, fontsize=8, loc="left", pad=2)
    sub = df[df["season"] == season_col]
    for site in SITES:
        srow = sub[sub["site"] == site]
        if srow.empty:
            continue
        ys = [
            float(srow.loc[srow["quadrant"] == q, "mean_h"].iloc[0])
            if not srow.loc[srow["quadrant"] == q].empty
            else np.nan
            for q in QUADRANTS
        ]
        ax.plot(
            QUADRANTS,
            ys,
            color=SITE_COLORS[site],
            linestyle=TYPOLOGY_LINESTYLE[SITE_TYPES[site]],
            marker="o",
            markersize=4,
            linewidth=1.6,
            label=SITE_LABELS[site],
            zorder=3,
        )
    ax.set_xlabel("aspect quadrant", fontsize=7)
    ax.set_ylabel("hours of direct sun", fontsize=7)
    ax.set_ylim(0, ymax)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.25, linewidth=0.5, zorder=1)
    ax.set_axisbelow(True)


def main() -> int:
    apply_style()

    parts = []
    for site in SITES:
        df = _load_site_solar_quadrants(site)
        if df is None:
            print(f"  {site}: SKIP (gpkg missing or aspect missing)")
            continue
        parts.append(df)
        print(f"  {site}: {len(df)} quadrant×season rows")

    if not parts:
        print("ERROR: no sites loaded.", file=sys.stderr)
        return 1

    df = pd.concat(parts, ignore_index=True)

    ymax = float(df["mean_h"].max()) * 1.10
    ymax = max(ymax, 1.0)  # guard for empty data

    fig, axes = plt.subplots(
        1, 4, figsize=(WIDTH_DOUBLE, WIDTH_DOUBLE * 0.32), sharey=True
    )
    for ax, (col, title) in zip(axes, SEASONS):
        _draw_panel(ax, df, col, title, ymax)
        # Only the first panel shows the y-axis label (others share).
        if ax is not axes[0]:
            ax.set_ylabel("")

    # Single shared legend below all panels — typology + site in one row.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(SITES),
        fontsize=7,
        frameon=False,
    )
    fig.suptitle(
        "Cross-site solar by aspect quadrant.  Hillside sites (Vidigal, Rocinha) show winter aspect dissociation;\n"
        "flatland sites (Rio das Pedras, Maré) collapse to a single curve. Complexo do Alemão (mixed) sits between.",
        fontsize=8.5,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))

    save_fig(fig, "fig08_solar_cross_site")

    # Print a compact text summary so the §5.4 narrative is grounded.
    print()
    print("Quadrant means (winter, hours of direct sun):")
    print("  site                  N      E      S      W   N-S  contrast")
    for site in SITES:
        sub = df[(df["site"] == site) & (df["season"] == "solar_hours_winter")]
        if sub.empty:
            continue
        v = {r["quadrant"]: r["mean_h"] for _, r in sub.iterrows()}
        contrast = (v.get("N", np.nan) - v.get("S", np.nan))
        print(
            f"  {SITE_LABELS[site]:24s}  "
            f"{v.get('N', np.nan):5.2f}  {v.get('E', np.nan):5.2f}  "
            f"{v.get('S', np.nan):5.2f}  {v.get('W', np.nan):5.2f}  "
            f"{contrast:+5.2f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
