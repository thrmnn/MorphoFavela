"""Geometric multi-constraint ventilation-tendency index (E2, 2026-06-28).

Synthesises the THREE independent geometry-only ventilation signals of §5.6 into a
single *ranked* layer — WITHOUT collapsing their incommensurable continuous scales
into a weighted sum. The council was explicit (see run_ventilation_susceptibility)
that the vertical and lateral axes must never be summed into one continuous index;
that rule is respected here by counting *qualitative constraints* (a checklist),
not by averaging continuous magnitudes. This is the honest generalisation of the
existing "doubly constrained" (skimming ∩ deep) finding to a triply-constrained one.

Each built 10 m cell is flagged on three independent axes:

  VERTICAL    flag_skimming      λf_mean ≥ 0.65          (Oke/Grimmond-&-Oke skimming regime)
  LATERAL     flag_deep          open_edge_dist ≥ median (buried in contiguous fabric)
  DIRECTIONAL flag_wind_aligned  exposure_ratio ≥ 1.0    (prevailing wind meets an above-average frontal axis)

  n_constraints = flag_skimming + flag_deep + flag_wind_aligned   ∈ {0,1,2,3}

The index is the COUNT (ordinal 0–3), not a sum of magnitudes — so it imposes no
arbitrary weighting between vertical/lateral/directional and stays interpretable.

STRICTLY GEOMETRIC TENDENCY, pre-CFD — NOT an air-exchange adequacy (age-of-air τ is
CFD-gated and supersedes this). A coarse triage surface only.

Outputs:
  outputs/paper_figures/ventilation_index.json
  outputs/paper_figures/exports/ventilation_index.png (+ .svg)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "outputs" / "paper_figures"))
from fig_style import SITE_LABELS, add_provenance, apply_style, save_fig  # noqa: E402

from scripts.run_lateral_connectivity import open_edge_distance  # noqa: E402
from scripts.run_wind_exposure import load_freq, wind_exposure  # noqa: E402
from src.morphometry.invariants import built_mask as built_cell_mask  # noqa: E402

SITES = ["vidigal", "rocinha", "complexo_do_alemao", "riodaspedras", "maré"]
SKIM_MIN = 0.65
# discrete 0..3 palette (light grey → dark red); higher = more geometric constraints
CLASS_COLORS = ["#E8E8E8", "#FDD9A8", "#F08A3C", "#B2182B"]


def count_constraints(
    lambda_f_mean: np.ndarray,
    open_edge_dist: np.ndarray,
    exposure_ratio: np.ndarray,
    depth_median: float,
) -> np.ndarray:
    """Per-cell ordinal count (0–3) of triggered geometric ventilation constraints.

    Pure function over already-computed per-cell arrays so it is unit-testable
    independently of the GIS read. NaNs in any axis count as not-triggered for that
    axis (a missing signal never manufactures a constraint)."""
    flag_skimming = np.nan_to_num(lambda_f_mean, nan=0.0) >= SKIM_MIN
    flag_deep = np.nan_to_num(open_edge_dist, nan=0.0) >= depth_median
    flag_wind = np.nan_to_num(exposure_ratio, nan=0.0) >= 1.0
    return (flag_skimming.astype(int) + flag_deep.astype(int) + flag_wind.astype(int))


def build_site(site: str) -> gpd.GeoDataFrame:
    g = gpd.read_file(PROJECT_ROOT / "outputs" / site / "morphometrics" / "grid" / "grid_metrics.gpkg")
    g = g[built_cell_mask(g)].copy()
    g["open_edge_dist_m"] = open_edge_distance(
        g["centroid_x"].to_numpy(), g["centroid_y"].to_numpy(), built_cell_mask(g))
    freq = load_freq(site)
    g["wind_exposure"] = wind_exposure(g, freq)
    g["exposure_ratio"] = g["wind_exposure"] / g["lambda_f_mean"].replace(0, np.nan)
    return g


def main() -> None:
    grids = {s: build_site(s) for s in SITES}
    depth_median = float(np.median(np.concatenate(
        [grids[s]["open_edge_dist_m"].to_numpy() for s in SITES])))

    per_site, pooled_hist = {}, np.zeros(4, dtype=int)
    for s in SITES:
        g = grids[s]
        n_con = count_constraints(
            g["lambda_f_mean"].to_numpy(), g["open_edge_dist_m"].to_numpy(),
            g["exposure_ratio"].to_numpy(), depth_median)
        g["n_constraints"] = n_con
        n = len(g)
        hist = np.bincount(n_con, minlength=4)
        pooled_hist += hist
        per_site[s] = {
            "n": int(n),
            "shares": {str(k): float(hist[k]) / n for k in range(4)},
            "triply_constrained_frac": float(hist[3]) / n,
            "mean_constraints": float(n_con.mean()),
        }
    n_pool = int(pooled_hist.sum())
    pooled = {str(k): float(pooled_hist[k]) / n_pool for k in range(4)}

    payload = {
        "title": "Geometric multi-constraint ventilation-tendency index (E2, 2026-06-28)",
        "definition": (
            "Per built 10 m cell, the COUNT (0–3) of independent geometry-only "
            "ventilation constraints triggered: skimming (λf_mean ≥ 0.65), deep "
            f"(open-edge distance ≥ pooled median {depth_median:.0f} m), wind-aligned "
            "(directional exposure ratio ≥ 1.0). A checklist count, NOT a weighted sum "
            "of the incommensurable continuous axes (council no-sum rule respected)."
        ),
        "status": "STRICTLY GEOMETRIC tendency, pre-CFD; NOT air exchange / adequacy (τ CFD-gated)",
        "axes": {
            "vertical": "flag_skimming = λf_mean ≥ 0.65",
            "lateral": f"flag_deep = open_edge_dist ≥ {depth_median:.1f} m (pooled median)",
            "directional": "flag_wind_aligned = exposure_ratio ≥ 1.0",
        },
        "depth_median_m": depth_median,
        "per_site": per_site,
        "pooled_shares": pooled,
        "pooled_triply_constrained_frac": pooled["3"],
        "pooled_mean_constraints": float(np.average(range(4), weights=pooled_hist)),
    }
    OUT = PROJECT_ROOT / "outputs" / "paper_figures" / "ventilation_index.json"
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Wrote {OUT.relative_to(PROJECT_ROOT)}  (depth median {depth_median:.0f} m)\n")
    print(f"{'site':<22s} {'0':>6s} {'1':>6s} {'2':>6s} {'3':>6s}  {'mean':>5s}")
    for s in SITES:
        sh = per_site[s]["shares"]
        print(f"{SITE_LABELS[s]:<22s} "
              f"{sh['0']*100:>5.1f}% {sh['1']*100:>5.1f}% {sh['2']*100:>5.1f}% "
              f"{sh['3']*100:>5.1f}%  {per_site[s]['mean_constraints']:>5.2f}")
    print(f"{'POOLED':<22s} "
          f"{pooled['0']*100:>5.1f}% {pooled['1']*100:>5.1f}% {pooled['2']*100:>5.1f}% "
          f"{pooled['3']*100:>5.1f}%  {payload['pooled_mean_constraints']:>5.2f}")

    apply_style()
    cmap = ListedColormap(CLASS_COLORS)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    fig = plt.figure(figsize=(7.09, 3.5))
    gs = fig.add_gridspec(1, len(SITES) + 1, width_ratios=[1] * len(SITES) + [0.5],
                          wspace=0.06, top=0.78, bottom=0.04, left=0.01, right=0.99)
    for i, s in enumerate(SITES):
        ax = fig.add_subplot(gs[0, i])
        grids[s].plot(ax=ax, column="n_constraints", cmap=cmap, norm=norm,
                      edgecolor="none", linewidth=0.0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"{'(A) ' if i == 0 else ''}{SITE_LABELS[s]}\n"
                     f"3× {per_site[s]['triply_constrained_frac']*100:.0f}% · "
                     f"x̄ {per_site[s]['mean_constraints']:.2f}", fontsize=6.2, pad=2)
    lax = fig.add_subplot(gs[0, -1])
    lax.set_axis_off()
    labels = ["0 — unconstrained", "1 constraint", "2 constraints", "3 — all axes"]
    handles = [Patch(facecolor=CLASS_COLORS[k], edgecolor="#444444", linewidth=0.4,
                     label=labels[k]) for k in range(4)]
    lax.legend(handles=handles, loc="center", fontsize=5.4, frameon=False,
               title="# geometric\nconstraints", title_fontsize=6, labelspacing=0.6,
               handlelength=1.1, handleheight=1.1)
    fig.text(0.5, 0.95,
             "Geometric multi-constraint ventilation-tendency index — count of independent "
             "axes triggered\n(vertical skimming · lateral depth · directional wind-alignment; "
             "checklist count, not a weighted sum; pre-CFD, not adequacy)",
             ha="center", va="top", fontsize=7.0)
    add_provenance(fig)
    save_fig(fig, "ventilation_index", gate=True)


if __name__ == "__main__":
    main()
