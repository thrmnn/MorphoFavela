"""Lateral-connectivity scalar — distance to the nearest open edge.

A geometry-only, pre-CFD ventilation-*tendency* proxy: for each built 10 m cell,
the Euclidean distance to the nearest OPEN cell — either an interior unbuilt cell
(plaza, wide street, clearing) or the settlement perimeter (outside the favela is
open). A cell deep inside a large contiguous fabric is far from any opening, so
lateral inflow must penetrate further to reach it; a perimeter cell is one step
from open air.

This is the LATERAL companion to λf's VERTICAL flow-regime signal
(scripts/run_lambda_f_regime.py): λf says how suppressed the vertical exchange is
(isolated / wake / skimming); the open-edge distance says how deep laterally the
cell sits. Both are pre-CFD GEOMETRIC tendencies — neither delivers a per-cell
ventilation adequacy. Age-of-air τ stays CFD-gated; this scalar is qualitative and
τ-superseded, included because the council flagged it as the one independent
lateral signal available before the OpenFOAM campaign.

Computation: rasterise the built mask on the grid's 10 m lattice, pad a 1-cell
open border (the favela exterior), and take a Euclidean distance transform. The
distance at each built cell is its depth into the fabric.

Outputs:
  outputs/paper_figures/lateral_connectivity.json
  outputs/paper_figures/exports/lateral_connectivity.png (+ .svg)
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
from scipy.ndimage import distance_transform_edt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "outputs" / "paper_figures"))
from fig_style import SITE_LABELS, apply_style, save_fig  # noqa: E402

SITES = ["vidigal", "rocinha", "complexo_do_alemao", "riodaspedras", "maré"]
CELL_M = 10.0
CMAP = "magma_r"  # dark = deep interior (low lateral tendency); light = near open
OUT_JSON = PROJECT_ROOT / "outputs" / "paper_figures" / "lateral_connectivity.json"


def open_edge_distance(
    centroid_x: np.ndarray, centroid_y: np.ndarray, built: np.ndarray, cell: float = CELL_M
) -> np.ndarray:
    """Distance (m) from each cell to the nearest open cell. Built cells get their
    depth into the fabric; unbuilt cells get 0. A 1-cell open border is padded so
    the settlement perimeter always counts as an open edge."""
    ix = np.round((centroid_x - centroid_x.min()) / cell).astype(int)
    iy = np.round((centroid_y - centroid_y.min()) / cell).astype(int)
    mask = np.zeros((iy.max() + 3, ix.max() + 3), dtype=bool)  # +2 pad +1 for 0-index
    mask[iy[built] + 1, ix[built] + 1] = True
    dist = distance_transform_edt(mask) * cell
    out = dist[iy + 1, ix + 1]
    out[~built] = 0.0
    return out


def load_site(site: str) -> gpd.GeoDataFrame:
    g = gpd.read_file(
        PROJECT_ROOT / "outputs" / site / "morphometrics" / "grid" / "grid_metrics.gpkg"
    )
    built = (g["building_count"] > 0).to_numpy()
    g["open_edge_dist_m"] = open_edge_distance(
        g["centroid_x"].to_numpy(), g["centroid_y"].to_numpy(), built
    )
    g["_built"] = built
    return g


def cross_signal(grids: dict) -> dict:
    """Quantify the lateral×vertical double-constraint at the CELL level: does a
    cell that is deep in the fabric (high open-edge distance) also sit in the
    skimming λf regime? Spearman ρ(open_edge_dist, λf) + the joint fraction of
    built cells that are both deep (≥ pooled-median depth) AND skimming (λf≥0.65)."""
    from scipy.stats import spearmanr

    rows, dist_all, lf_all = {}, [], []
    for s in SITES:
        b = grids[s][grids[s]["_built"]]
        d = b["open_edge_dist_m"].to_numpy()
        lf = b["lambda_f_mean"].to_numpy()
        ok = np.isfinite(d) & np.isfinite(lf)
        d, lf = d[ok], lf[ok]
        dist_all.append(d)
        lf_all.append(lf)
        rho, p = spearmanr(d, lf)
        rows[s] = {"spearman_rho": float(rho), "p": float(p), "n": int(d.size)}
    d, lf = np.concatenate(dist_all), np.concatenate(lf_all)
    rho, p = spearmanr(d, lf)
    deep = d >= np.median(d)
    skim = lf >= 0.65
    return {
        "note": "per-built-cell ρ(open_edge_dist, λf); positive ⇒ deeper cells are "
                "also more skimming (doubly constrained at the cell level).",
        "pooled_spearman_rho": float(rho),
        "pooled_p": float(p),
        "joint_deep_and_skimming_frac": float((deep & skim).mean()),
        "per_site": rows,
    }


def _stats(v: np.ndarray) -> dict:
    return {
        "n": int(v.size),
        "median": float(np.median(v)),
        "p90": float(np.percentile(v, 90)),
        "max": float(v.max()),
        "mean": float(np.mean(v)),
    }


def make_figure(grids: dict, vmax: float) -> None:
    apply_style()
    fig = plt.figure(figsize=(7.09, 3.4))
    gs = fig.add_gridspec(1, len(SITES) + 1, width_ratios=[1] * len(SITES) + [0.06],
                          wspace=0.08, top=0.80, bottom=0.04, left=0.015, right=0.93)
    norm = Normalize(0, vmax)
    for i, site in enumerate(SITES):
        ax = fig.add_subplot(gs[0, i])
        g = grids[site]
        b = g[g["_built"]]
        b.plot(ax=ax, column="open_edge_dist_m", cmap=CMAP, vmin=0, vmax=vmax,
               edgecolor="none", linewidth=0.0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        med = b["open_edge_dist_m"].median()
        ax.set_title(f"{'(A) ' if i == 0 else ''}{SITE_LABELS[site]}\nmed {med:.0f} m",
                     fontsize=6.5, pad=2)
    cax = fig.add_subplot(gs[0, -1])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=CMAP), cax=cax)
    cb.set_label("distance to nearest open edge (m)", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    fig.text(
        0.5, 0.965,
        "Lateral-connectivity tendency — depth into contiguous fabric\n"
        "(geometry-only, pre-CFD companion to the λ$_f$ vertical regime; not an adequacy)",
        ha="center", va="top", fontsize=7.0,
    )
    save_fig(fig, "lateral_connectivity", gate=True)


def main() -> None:
    grids = {s: load_site(s) for s in SITES}
    per_site, pooled = {}, []
    for s in SITES:
        v = grids[s].loc[grids[s]["_built"], "open_edge_dist_m"].to_numpy()
        per_site[s] = _stats(v)
        pooled.append(v)
    pooled_v = np.concatenate(pooled)
    vmax = float(np.percentile(pooled_v, 98))
    cross = cross_signal(grids)

    payload = {
        "title": "Lateral-connectivity scalar — distance to nearest open edge (2026-06-25)",
        "definition": (
            "Euclidean distance (m) from each built 10 m cell to the nearest open cell "
            "(interior unbuilt cell or the padded settlement perimeter), via a distance "
            "transform on the built mask. Higher = deeper into contiguous fabric = lower "
            "lateral ventilation tendency."
        ),
        "status": (
            "GEOMETRY-ONLY, PRE-CFD, QUALITATIVE. Lateral companion to the λf vertical "
            "flow regime; neither delivers per-cell ventilation adequacy (age-of-air τ "
            "is CFD-gated and superseding)."
        ),
        "colorbar_vmax_p98": vmax,
        "per_site": per_site,
        "pooled": _stats(pooled_v),
        "lateral_vs_vertical": cross,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Wrote {OUT_JSON.relative_to(PROJECT_ROOT)}")
    print(f"\n{'site':<22s} {'n':>6s} {'med':>6s} {'p90':>6s} {'max':>6s}")
    for s in SITES:
        p = per_site[s]
        print(f"{SITE_LABELS[s]:<22s} {p['n']:>6d} {p['median']:>6.0f} "
              f"{p['p90']:>6.0f} {p['max']:>6.0f}")
    pp = payload["pooled"]
    print(f"{'POOLED':<22s} {pp['n']:>6d} {pp['median']:>6.0f} {pp['p90']:>6.0f} "
          f"{pp['max']:>6.0f}")
    print(f"\nlateral×vertical: pooled Spearman ρ(open_edge_dist, λf) = "
          f"{cross['pooled_spearman_rho']:+.3f} (p={cross['pooled_p']:.1e}); "
          f"{cross['joint_deep_and_skimming_frac']*100:.0f}% of built cells are both "
          f"deep (≥median) and skimming (λf≥0.65)")

    make_figure(grids, vmax)


if __name__ == "__main__":
    main()
