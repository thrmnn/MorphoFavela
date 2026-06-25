"""Effective wind-exposure scalar — directional λf weighted by the wind rose.

Roadmap #4. The grid carries 8-sector frontal-area density (lambda_f_N..NW) and
each site has a measured wind rose (data/{site}/wind_rose.json); they sat unused
together. The effective exposure of a cell is the frontal-area density the wind
actually meets, weighted by how often it blows from each sector:

    exposure = Σ_θ  freq(θ) · λf(θ)         (θ over the 8 compass sectors)

λf is exactly 180°-symmetric (λf_N == λf_S, verified max|diff| = 0), so this is a
FREQUENCY weighting of the 4 distinct cross-wind axes — it captures *how often the
fabric presents blockage to the prevailing wind*, never channelling or sheltering.
The ratio exposure / λf_mean (isotropic baseline) is >1 where a cell's fabric is
aligned to block the prevailing wind, <1 where the prevailing wind hits its open axis.

STATUS: geometry × climatology, pre-CFD, QUALITATIVE exposure TENDENCY — NOT a
ventilation adequacy (age-of-air τ is CFD-gated). Companion to the λf vertical
flow regime and the lateral open-edge depth.

Outputs:
  outputs/paper_figures/wind_exposure.json
  outputs/paper_figures/exports/wind_exposure.png (+ .svg)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "outputs" / "paper_figures"))
from fig_style import SITE_LABELS, apply_style, save_fig  # noqa: E402

SITES = ["vidigal", "rocinha", "complexo_do_alemao", "riodaspedras", "maré"]
SECTORS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
CMAP = "viridis"
OUT_JSON = PROJECT_ROOT / "outputs" / "paper_figures" / "wind_exposure.json"


def load_freq(site: str) -> dict:
    d = json.loads((PROJECT_ROOT / "data" / site / "wind_rose.json").read_text())
    f = d["frequencies"]
    tot = sum(f[s] for s in SECTORS)
    return {s: f[s] / tot for s in SECTORS}  # renormalise (calm already excluded)


def wind_exposure(grid: gpd.GeoDataFrame, freq: dict) -> np.ndarray:
    """Σ_θ freq(θ)·λf(θ) per cell over the 8 sectors. freq must sum to 1."""
    cols = [f"lambda_f_{s}" for s in SECTORS]
    lf = grid[cols].to_numpy(float)
    w = np.array([freq[s] for s in SECTORS])
    return lf @ w


def built(site: str) -> gpd.GeoDataFrame:
    g = gpd.read_file(PROJECT_ROOT / "outputs" / site / "morphometrics" / "grid" / "grid_metrics.gpkg")
    return g[g["building_count"] > 0].copy()


def _stats(v: np.ndarray) -> dict:
    v = v[np.isfinite(v)]
    return {"n": int(v.size), "median": float(np.median(v)), "mean": float(np.mean(v))}


def main() -> None:
    grids, per_site, exp_all, ratio_all = {}, {}, [], []
    for s in SITES:
        g = built(s)
        freq = load_freq(s)
        g["wind_exposure"] = wind_exposure(g, freq)
        g["exposure_ratio"] = g["wind_exposure"] / g["lambda_f_mean"].replace(0, np.nan)
        grids[s] = g
        e = g["wind_exposure"].to_numpy()
        r = g["exposure_ratio"].to_numpy()
        per_site[s] = {
            "exposure": _stats(e),
            "ratio_to_isotropic": _stats(r),
            "dominant_sector": max(freq, key=freq.get),
            "dominant_freq": float(max(freq.values())),
        }
        exp_all.append(e)
        ratio_all.append(r[np.isfinite(r)])

    payload = {
        "title": "Effective wind-exposure scalar — Σ freq(θ)·λf(θ) (2026-06-25)",
        "definition": (
            "Per built cell, the 8-sector frontal-area density weighted by the measured "
            "wind-rose frequency. λf is 180°-symmetric → frequency weighting of the 4 "
            "cross-wind axes; captures how often the fabric blocks the prevailing wind, "
            "not channelling/sheltering."
        ),
        "status": "geometry × climatology, pre-CFD, QUALITATIVE exposure tendency (NOT adequacy; τ CFD-gated)",
        "per_site": per_site,
        "pooled_exposure": _stats(np.concatenate(exp_all)),
        "pooled_ratio_to_isotropic": _stats(np.concatenate(ratio_all)),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Wrote {OUT_JSON.relative_to(PROJECT_ROOT)}\n")
    print(f"{'site':<22s} {'domwind':>8s} {'exp_med':>8s} {'ratio_med':>10s}")
    for s in SITES:
        p = per_site[s]
        print(f"{SITE_LABELS[s]:<22s} {p['dominant_sector']:>5s}"
              f"{p['dominant_freq']*100:>3.0f}% {p['exposure']['median']:>8.3f} "
              f"{p['ratio_to_isotropic']['median']:>10.3f}")

    apply_style()
    vmax = float(np.percentile(np.concatenate(exp_all), 98))
    fig = plt.figure(figsize=(7.09, 3.4))
    gs = fig.add_gridspec(1, len(SITES) + 1, width_ratios=[1] * len(SITES) + [0.06],
                          wspace=0.08, top=0.80, bottom=0.04, left=0.015, right=0.93)
    norm = Normalize(0, vmax)
    for i, s in enumerate(SITES):
        ax = fig.add_subplot(gs[0, i])
        grids[s].plot(ax=ax, column="wind_exposure", cmap=CMAP, vmin=0, vmax=vmax,
                      edgecolor="none", linewidth=0.0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        p = per_site[s]
        ax.set_title(f"{'(A) ' if i == 0 else ''}{SITE_LABELS[s]}\n"
                     f"{p['dominant_sector']} {p['dominant_freq']*100:.0f}% · "
                     f"med {p['exposure']['median']:.2f}", fontsize=6.2, pad=2)
    cax = fig.add_subplot(gs[0, -1])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=CMAP), cax=cax)
    cb.set_label("effective wind exposure  Σ freq(θ)·λ$_f$(θ)", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    fig.text(0.5, 0.965,
             "Effective wind-exposure tendency — directional λ$_f$ weighted by the measured "
             "wind rose\n(geometry × climatology, pre-CFD; frequency weighting only, not "
             "channelling or adequacy)", ha="center", va="top", fontsize=7.0)
    save_fig(fig, "wind_exposure", gate=True)


if __name__ == "__main__":
    main()
