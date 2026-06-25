#!/usr/bin/env python3
"""PROTOTYPE — dissolved (party-wall-corrected) frontal-area density λf.

The canonical cell λf sums each *cadastral* building's frontal area
(``compute_frontal_area_ratio``). In fused favela fabric (party walls
everywhere) that counts internal shared walls, inflating λf ~1.7× (median;
Rocinha). NOTE the inflation does NOT explain the ventilation-axis saturation:
even the corrected dissolved λf leaves ~90% of the fabric past the skimming-flow
onset, so the round-3 "knife-edge" saturation is a real fabric property, not an
artifact — the dissolve fixes magnitude (matters for roughness z0/zd), the
multi-variable index is still needed for axis discrimination.

This prototype recomputes λf on the DISSOLVED footprints: within each cell the
clipped footprints are unioned into physical blocks (touching parcels merge, so
internal party walls vanish), and each block contributes its silhouette width ×
an area-weighted representative height. Distinct (non-touching) blocks are still
summed — this removes party-wall over-count without erasing genuine separate
roughness elements or along-wind sheltering between blocks.

It is NON-DESTRUCTIVE: writes a side-by-side comparison gpkg + review figure for
ONE site; the canonical grid and fig03/fig04 are untouched. If the review
approves, the dissolved λf is rolled into the pipeline (see the plan printed at
the end).

    python scripts/prototype_lambda_f_dissolve.py --site rocinha

Outputs:
  outputs/{site}/morphometrics/grid/lambda_f_dissolve_compare.gpkg
  outputs/paper_figures/exports/lambda_f_dissolve_review_{site}.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.ops import unary_union

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "outputs" / "paper_figures"))
from fig_style import apply_style  # noqa: E402

from src.urban_morphology import _projected_width  # noqa: E402

# The 8 wind directions collapse to 4 distinct cross-wind axes (N≡S, E≡W,
# NE≡SW, SE≡NW); averaging these four equals the canonical 8-direction mean.
AXES_DEG = [0.0, 45.0, 90.0, 135.0]
SKIMMING_ONSET = 0.35  # Grimmond–Oke skimming-flow onset (neighbourhood scale)
HEIGHT_COL_CANDIDATES = ("height", "altura")


def _height_col(bld: gpd.GeoDataFrame) -> str:
    for c in HEIGHT_COL_CANDIDATES:
        if c in bld.columns:
            return c
    raise KeyError("no height column found in buildings")


def dissolved_lambda_f(cell_geom, zone_area, parcels, heights) -> float:
    """Mean over the 4 cross-wind axes of Σ_block (silhouette width × rep height)
    / zone_area, on footprints unioned (party walls removed) within the cell."""
    clipped, hs = [], []
    for geom, h in zip(parcels, heights):
        g = geom.intersection(cell_geom)
        if g.is_empty or g.area <= 0:
            continue
        clipped.append(g)
        hs.append(float(h))
    if not clipped:
        return 0.0
    merged = unary_union(clipped)
    blocks = list(merged.geoms) if merged.geom_type.startswith("Multi") else [merged]
    clipped_arr = np.array(clipped, dtype=object)
    h_arr = np.array(hs)
    areas = np.array([c.area for c in clipped])

    # Per-block representative (area-weighted) height — computed once.
    block_info = []
    for block in blocks:
        sel = np.array([c.intersects(block) for c in clipped_arr])
        if not sel.any():
            continue
        w = areas[sel]
        h_rep = float(np.average(h_arr[sel], weights=w)) if w.sum() > 0 else float(h_arr[sel].mean())
        block_info.append((block, h_rep))

    per_axis = []
    for deg in AXES_DEG:
        wr = np.deg2rad(deg)
        total = sum(_projected_width(b, wr) * h for b, h in block_info)
        per_axis.append(total / zone_area if zone_area > 0 else 0.0)
    return float(np.mean(per_axis))


def compute_for_site(site: str) -> gpd.GeoDataFrame:
    grid = gpd.read_file(
        PROJECT_ROOT / "outputs" / site / "morphometrics" / "grid" / "grid_metrics.gpkg"
    )
    bld = gpd.read_file(PROJECT_ROOT / "data" / site / "buildings_extended_300m.gpkg")
    if bld.crs != grid.crs:
        bld = bld.to_crs(grid.crs)
    hcol = _height_col(bld)
    built = grid[(grid["building_count"] > 0)].copy()

    joined = gpd.sjoin(
        bld[[hcol, "geometry"]], built[["zone_id", "geometry"]], how="inner", predicate="intersects"
    )
    diss = {}
    cells = built.set_index("zone_id")
    for zid, grp in joined.groupby("zone_id"):
        cell = cells.loc[zid, "geometry"]
        diss[zid] = dissolved_lambda_f(cell, cells.loc[zid, "zone_area"],
                                       list(grp.geometry), list(grp[hcol]))
    built["lambda_f_mean_dissolved"] = built["zone_id"].map(diss).fillna(0.0)
    built["over_count_ratio"] = built["lambda_f_mean"] / built["lambda_f_mean_dissolved"].replace(0, np.nan)
    return built


def review_figure(g: gpd.GeoDataFrame, site: str) -> Path:
    apply_style()
    summed = g["lambda_f_mean"].to_numpy()
    diss = g["lambda_f_mean_dissolved"].to_numpy()
    fs = float((summed >= SKIMMING_ONSET).mean())
    fd = float((diss >= SKIMMING_ONSET).mean())
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.4))

    # (A) distributions — the magnitude correction
    ax = axes[0, 0]
    bins = np.linspace(0, 5, 60)
    ax.hist(summed, bins=bins, color="#999999", alpha=0.75,
            label=f"summed (median {np.median(summed):.2f})")
    ax.hist(diss, bins=bins, color="#2A7DB5", alpha=0.75,
            label=f"dissolved (median {np.median(diss):.2f})")
    ax.axvline(SKIMMING_ONSET, color="#B2182B", ls="--", lw=1.2)
    ax.text(SKIMMING_ONSET + 0.05, ax.get_ylim()[1] * 0.9, "skimming 0.35",
            color="#B2182B", fontsize=7)
    ax.set_xlabel(r"cell $\lambda_f$")
    ax.set_ylabel("cells")
    ax.set_title("(A) magnitude: dissolve removes the party-wall over-count", loc="left", fontsize=8.5)
    ax.legend(fontsize=7)

    # (B) scatter
    ax = axes[0, 1]
    ax.scatter(diss, summed, s=3, alpha=0.25, color="#444444", linewidths=0)
    lim = float(np.nanpercentile(summed, 99))
    ax.plot([0, lim], [0, lim], color="#B2182B", lw=1, ls=":")
    r = g["over_count_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
    ax.set_xlim(0, float(np.nanpercentile(diss, 99)))
    ax.set_ylim(0, lim)
    ax.set_xlabel("dissolved λf")
    ax.set_ylabel("summed λf")
    ax.set_title(f"(B) over-count ratio median {r.median():.2f} (p90 {r.quantile(.9):.2f})",
                 loc="left", fontsize=8.5)

    # (C) fraction past the skimming onset — the HONEST result: dissolve does NOT
    # un-saturate; the fabric is genuinely skimming-flow even with correct λf.
    ax = axes[1, 0]
    ax.bar(["summed", "dissolved"], [fs * 100, fd * 100], color=["#999999", "#2A7DB5"])
    for i, v in enumerate([fs, fd]):
        ax.text(i, v * 100 + 1, f"{v * 100:.0f}%", ha="center", fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_ylabel("% cells ≥ 0.35 skimming onset")
    ax.set_title("(C) saturation is REAL — dissolve does not rescue the vent. axis",
                 loc="left", fontsize=8.5)

    # (D) spatial: dissolved λf map
    ax = axes[1, 1]
    vmax = float(np.nanpercentile(diss, 97))
    g.plot(ax=ax, column="lambda_f_mean_dissolved", cmap="YlGnBu", vmin=0, vmax=vmax,
           linewidth=0, legend=True, legend_kwds={"shrink": 0.6, "label": "dissolved λf"})
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("(D) dissolved λf — spatial pattern preserved", loc="left", fontsize=8.5)

    fig.suptitle(
        f"λf dissolve-fix review — {site} ({len(g):,} built cells): "
        f"magnitude {np.median(summed):.2f}→{np.median(diss):.2f}, "
        f"but {fd * 100:.0f}% still past skimming onset",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = PROJECT_ROOT / "outputs" / "paper_figures" / "exports" / f"lambda_f_dissolve_review_{site}.png"
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="rocinha")
    args = ap.parse_args()
    g = compute_for_site(args.site)

    summed, diss = g["lambda_f_mean"], g["lambda_f_mean_dissolved"]
    print(f"  {args.site}: {len(g):,} built cells")
    print(f"  summed    λf: median {summed.median():.2f}  p75 {summed.quantile(.75):.2f}  "
          f"%≥0.35 {(summed >= SKIMMING_ONSET).mean() * 100:.0f}%")
    print(f"  dissolved λf: median {diss.median():.2f}  p75 {diss.quantile(.75):.2f}  "
          f"%≥0.35 {(diss >= SKIMMING_ONSET).mean() * 100:.0f}%")
    r = g["over_count_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
    print(f"  over-count ratio: median {r.median():.2f}  p90 {r.quantile(.9):.2f}")

    gpkg = PROJECT_ROOT / "outputs" / args.site / "morphometrics" / "grid" / "lambda_f_dissolve_compare.gpkg"
    g[["zone_id", "lambda_f_mean", "lambda_f_mean_dissolved", "over_count_ratio",
       "lambda_p", "H_mean", "building_count", "geometry"]].to_file(gpkg, driver="GPKG")
    print(f"  wrote {gpkg.relative_to(PROJECT_ROOT)}")
    out = review_figure(g, args.site)
    print(f"  wrote {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
