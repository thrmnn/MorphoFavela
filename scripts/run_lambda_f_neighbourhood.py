#!/usr/bin/env python3
"""Neighbourhood-scale frontal-area density (λf) on the Grimmond–Oke regime scale.

Why
---
The canonical grid λf is computed at 10 m cell support, where a single tall
building fills a cell and drives λf to 0–~5 (pooled p75 = 2.75). That is the
right support for the cell-scale diagnostic figures (fig03/fig04), but it is NOT
the scale at which the Grimmond–Oke flow-regime thresholds (isolated < 0.15,
wake-interference 0.15–0.35, skimming > ~0.35) are defined — those are
*neighbourhood*-scale quantities.

This supplement aggregates the cell λf to a 100 m-diameter neighbourhood (the
CFD analysis-patch scale): for every built cell, the mean of cell λf over all
grid cells (built AND unbuilt → the open-area denominator is preserved) whose
centroid lies within 50 m. Because all cells share the same 100 m² plan area,
this window mean equals Σ(frontal area) / Σ(plan area) over the neighbourhood —
a proper Grimmond–Oke λf, WITHOUT touching the cell-scale figures.

Finding: neighbourhood λf is lower than the cell-scale field but still runs
~1–3 (pooled median ≈ 1.65); ~96% of the built fabric sits ABOVE the 0.35
skimming-flow onset, and the densest sites (Rio das Pedras, Rocinha) sit at
median 2.4–2.7. So the favela fabric is uniformly in the skimming regime even at
neighbourhood support.

CAVEAT (frontal-area over-count, verified 2026-06-24): the underlying cell λf
(``compute_frontal_area_ratio``) SUMS each cadastral building's frontal area, so
in fused party-wall favela fabric it counts internal shared walls. Against a
dissolved merged-envelope estimate the summed λf over-counts by ~2.5× (median;
p90 ~4.3): summed cell-λf median ≈ 1.6 vs dissolved ≈ 0.65. The aerodynamically
correct neighbourhood λf is therefore ~2–2.5× below the numbers here — still
ABOVE the 0.35 skimming onset (≈0.65 > 0.35), so the E2 regime conclusion holds,
but the magnitude is inflated. A proper fix (dissolve footprints before the
frontal projection) is a pipeline-wide change touching fig03/fig04/the predictor
/the roughness track and is deferred to a user decision; this supplement reports
the summed-λf neighbourhood field with the over-count flagged.

Outputs:
  outputs/paper_figures/lambda_f_neighbourhood.json
  outputs/paper_figures/exports/lambda_f_neighbourhood.png (+ .svg)
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "outputs" / "paper_figures"))
from fig_style import SITE_COLORS, SITE_LABELS, apply_style, save_fig

warnings.filterwarnings("ignore", category=FutureWarning)

SITES = ["vidigal", "rocinha", "complexo_do_alemao", "riodaspedras", "maré"]
RADIUS_M = 50.0  # 100 m-diameter neighbourhood = CFD analysis-patch scale
SKIMMING_ONSET = 0.35  # Grimmond–Oke skimming-flow onset
WAKE_ONSET = 0.15  # isolated → wake-interference onset


def neighbourhood_mean(
    cx: np.ndarray, cy: np.ndarray, values: np.ndarray, radius: float, eval_mask: np.ndarray
) -> np.ndarray:
    """Mean of ``values`` over all points within ``radius`` of each evaluated
    point. Pure (cKDTree) → testable. Returns NaN where a window is empty."""
    tree = cKDTree(np.column_stack([cx, cy]))
    eval_idx = np.flatnonzero(eval_mask)
    pts = np.column_stack([cx[eval_idx], cy[eval_idx]])
    neighbours = tree.query_ball_point(pts, r=radius)
    out = np.full(eval_idx.shape, np.nan)
    for i, nb in enumerate(neighbours):
        if nb:
            out[i] = float(np.mean(values[nb]))
    return out


def site_neighbourhood_lambda_f(site: str) -> np.ndarray:
    grid = gpd.read_file(
        PROJECT_ROOT / "outputs" / site / "morphometrics" / "grid" / "grid_metrics.gpkg"
    )
    built = (grid["lambda_p"].fillna(0) > 0.01) | (grid["building_count"] > 0)
    cx = grid["centroid_x"].to_numpy()
    cy = grid["centroid_y"].to_numpy()
    lf = grid["lambda_f_mean"].fillna(0.0).to_numpy()
    return neighbourhood_mean(cx, cy, lf, RADIUS_M, built.to_numpy())


def summarise(vals: np.ndarray) -> dict:
    v = vals[np.isfinite(vals)]
    return {
        "n": int(v.size),
        "median": float(np.median(v)),
        "p25": float(np.percentile(v, 25)),
        "p75": float(np.percentile(v, 75)),
        "p95": float(np.percentile(v, 95)),
        "max": float(v.max()),
        "frac_skimming_ge_0_35": float((v >= SKIMMING_ONSET).mean()),
        "frac_wake_0_15_to_0_35": float(((v >= WAKE_ONSET) & (v < SKIMMING_ONSET)).mean()),
        "frac_isolated_lt_0_15": float((v < WAKE_ONSET).mean()),
    }


def make_figure(per_site_vals: dict[str, np.ndarray]) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    xmax = 3.6
    xs = np.linspace(0, xmax, 400)
    offsets = np.arange(len(SITES)) * 1.1
    for i, site in enumerate(SITES):
        v = per_site_vals[site]
        v = v[np.isfinite(v)]
        v = v[(v >= 0) & (v <= xmax)]
        if v.size < 30:
            continue
        kde = gaussian_kde(v, bw_method=0.18)
        dens = kde(xs)
        dens = dens / dens.max() * 0.95
        base = offsets[i]
        # Shade the sub-skimming mass (λf < 0.35) distinctly — it is tiny.
        ax.fill_between(xs, base, base + dens, color=SITE_COLORS[site], alpha=0.7, lw=0)
        ax.plot(xs, base + dens, color="#333333", lw=0.5)
        med = np.median(v)
        ax.plot([med, med], [base, base + 0.28], color="#111111", lw=0.9)
        ax.text(med, base + 0.32, f"med {med:.1f}", ha="center", va="bottom",
                fontsize=5.5, color="#222222")
    ax.axvspan(0, SKIMMING_ONSET, color="#EEEEEE", zorder=0)
    ax.axvline(SKIMMING_ONSET, color="#B2182B", lw=1.1, ls="--")
    ax.text(SKIMMING_ONSET + 0.03, offsets[-1] + 0.95, "skimming onset 0.35",
            color="#B2182B", fontsize=7, va="top")
    ax.set_yticks(offsets)
    ax.set_yticklabels([SITE_LABELS[s] for s in SITES], fontsize=8)
    ax.set_xlim(0, xmax)
    ax.set_xlabel(r"neighbourhood $\lambda_f$ (100 m window, Grimmond–Oke scale)", fontsize=8)
    ax.set_title(
        r"Neighbourhood $\lambda_f$ per favela — fabric sits well past the 0.35 "
        r"skimming onset (~96% of cells)",
        loc="left", fontsize=8.5, pad=4,
    )
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    save_fig(fig, "lambda_f_neighbourhood")


def main() -> None:
    per_site_vals = {}
    per_site_summary = {}
    for site in SITES:
        vals = site_neighbourhood_lambda_f(site)
        per_site_vals[site] = vals
        per_site_summary[site] = summarise(vals)
        s = per_site_summary[site]
        print(f"  {site:20s} median={s['median']:.2f}  p75={s['p75']:.2f}  "
              f"skimming≥0.35={s['frac_skimming_ge_0_35'] * 100:.0f}%")
    pooled = np.concatenate([per_site_vals[s] for s in SITES])
    pooled_summary = summarise(pooled)
    print(f"  POOLED median={pooled_summary['median']:.2f}  "
          f"skimming≥0.35={pooled_summary['frac_skimming_ge_0_35'] * 100:.0f}%")

    out = {
        "scale": "neighbourhood (100 m-diameter window = CFD analysis-patch scale)",
        "radius_m": RADIUS_M,
        "skimming_onset": SKIMMING_ONSET,
        "wake_onset": WAKE_ONSET,
        "note": (
            "Window mean of cell λf over all grid cells (built + unbuilt) within "
            "the radius; equals Σ frontal / Σ plan area at uniform 100 m² cells. "
            "Cell-scale λf (p75 = 2.75, threshold of fig03/fig04) is a different "
            "support and is unchanged."
        ),
        "over_count_caveat": {
            "issue": (
                "Underlying cell λf sums per-cadastral-building frontal area, "
                "counting internal party walls in fused favela fabric."
            ),
            "summed_vs_dissolved_ratio_median": 2.5,
            "summed_cell_lambda_f_median": 1.6,
            "dissolved_cell_lambda_f_median": 0.65,
            "implication": (
                "Aerodynamically correct neighbourhood λf is ~2–2.5× below these "
                "values but remains > 0.35 skimming onset; E2 regime conclusion "
                "holds, magnitude inflated. Pipeline-wide dissolve fix deferred to "
                "user decision."
            ),
        },
        "per_site": per_site_summary,
        "pooled": pooled_summary,
    }
    out_json = PROJECT_ROOT / "outputs" / "paper_figures" / "lambda_f_neighbourhood.json"
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"  wrote {out_json.relative_to(PROJECT_ROOT)}")
    make_figure(per_site_vals)


if __name__ == "__main__":
    main()
