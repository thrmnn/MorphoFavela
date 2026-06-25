#!/usr/bin/env python3
"""λf flow-regime classification — the "uniformly skimming" saturation finding.

Per the round-3 urban-physics council: geometry classifies the FLOW REGIME (it
cannot grade per-cell ventilation adequacy — that is CFD-τ, Tier 2). Each built
cell is binned by its (now canonical, dissolved) frontal-area density λf against
the Oke / Grimmond–Oke flow-regime thresholds:

    isolated roughness   λf < 0.15
    wake interference    0.15 ≤ λf < 0.65
    skimming flow        λf ≥ 0.65   (the most ventilation-suppressed canopy)

The result is the at-scale ventilation statement: the favela fabric sits almost
entirely in skimming flow, so frontal-area density classifies *which regime* a
cell is in but not *how ventilated* it is. No fitting, no per-cell adequacy claim.

Outputs:
  outputs/paper_figures/lambda_f_regime.json
  outputs/paper_figures/exports/lambda_f_regime.png (+ .svg)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches as mpatches
from matplotlib.colors import BoundaryNorm, ListedColormap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "outputs" / "paper_figures"))
from fig_style import SITE_LABELS, apply_style, save_fig  # noqa: E402

# Hillside → flatland order (round-2 cross-figure consistency).
SITES = ["vidigal", "rocinha", "complexo_do_alemao", "riodaspedras", "maré"]
ISOLATED_MAX = 0.15
SKIMMING_MIN = 0.65
REGIMES = ("isolated", "wake", "skimming")

# Three-class regime palette, luminance-ordered light→dark as ventilation
# suppression rises (CVD-safe YlOrRd family, ≥15% lightness separation per
# round-3 §0.9). Skimming reuses fig04's compound ink for cross-figure tie.
REGIME_FILL = {"isolated": "#FDD9A8", "wake": "#F08A3C", "skimming": "#B2182B"}
REGIME_INT = {"isolated": 0, "wake": 1, "skimming": 2}


def classify_regime(lf: np.ndarray) -> dict:
    """Shares of cells in each Oke/GO flow regime (NaNs dropped)."""
    v = np.asarray(lf, float)
    v = v[np.isfinite(v)]
    n = v.size
    if n == 0:
        return {"n": 0, **{r: float("nan") for r in REGIMES}}
    iso = float((v < ISOLATED_MAX).mean())
    sky = float((v >= SKIMMING_MIN).mean())
    wake = 1.0 - iso - sky
    return {"n": int(n), "isolated": iso, "wake": wake, "skimming": sky,
            "median": float(np.median(v))}


def load_site_grid(site: str) -> gpd.GeoDataFrame:
    """Built cells with a regime int column (0 isolated / 1 wake / 2 skimming)."""
    grid = gpd.read_file(
        PROJECT_ROOT / "outputs" / site / "morphometrics" / "grid" / "grid_metrics.gpkg"
    )
    built = grid[grid["building_count"] > 0].copy()
    lf = built["lambda_f_mean"].to_numpy()
    regime = np.full(len(built), REGIME_INT["wake"], dtype=int)
    regime[lf < ISOLATED_MAX] = REGIME_INT["isolated"]
    regime[lf >= SKIMMING_MIN] = REGIME_INT["skimming"]
    built["regime"] = regime
    built.loc[~np.isfinite(lf), "regime"] = -1  # unclassifiable, dropped from maps
    return built


def _draw_map(ax, site: str, grid: gpd.GeoDataFrame) -> None:
    cmap = ListedColormap([REGIME_FILL[r] for r in REGIMES])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], ncolors=3)
    g = grid[grid["regime"] >= 0]
    g.plot(ax=ax, column="regime", cmap=cmap, norm=norm, linewidth=0)
    b = g.total_bounds
    pad = 15.0
    ax.set_xlim(b[0] - pad, b[2] + pad)
    ax.set_ylim(b[1] - pad, b[3] + pad)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(SITE_LABELS[site], fontsize=7.5, color="#222222", pad=2)


def _draw_strip(ax, per_site: dict[str, dict], pooled: dict) -> None:
    rows = [(SITE_LABELS[s], per_site[s]) for s in SITES] + [("Pooled", pooled)]
    labels = [f"{lab}\n(n={r['n']:,})" for lab, r in rows]
    bottoms = np.zeros(len(rows))
    for reg in REGIMES:
        vals = np.array([r[reg] for _, r in rows]) * 100.0
        ax.barh(labels, vals, left=bottoms, color=REGIME_FILL[reg],
                edgecolor="#444444", linewidth=0.5, height=0.66)
        for i, v in enumerate(vals):
            if v >= 5:
                ax.text(bottoms[i] + v / 2, i, f"{v:.0f}%", ha="center", va="center",
                        fontsize=6.0, color="#FFFFFF" if reg == "skimming" else "#222222")
        bottoms += vals
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of built cells (%)", fontsize=7)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.tick_params(axis="y", labelsize=7.0)
    ax.invert_yaxis()
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.4)
    ax.spines["left"].set_linewidth(0.4)
    ax.set_title("(B) Flow-regime shares", loc="left", fontsize=8, color="#222222", pad=4)


def make_figure(grids: dict[str, gpd.GeoDataFrame],
                per_site: dict[str, dict], pooled: dict) -> None:
    apply_style()
    fig = plt.figure(figsize=(7.5, 5.4))
    gs = fig.add_gridspec(nrows=2, ncols=5, height_ratios=[1.0, 0.95],
                          hspace=0.30, wspace=0.08)
    for j, s in enumerate(SITES):
        _draw_map(fig.add_subplot(gs[0, j]), s, grids[s])
    fig.text(0.012, 0.965, "(A) Per-site flow-regime maps  ·  dissolved λ$_f$, 10 m cells",
             fontsize=8, color="#222222", va="top")
    _draw_strip(fig.add_subplot(gs[1, :3]), per_site, pooled)

    # Shared legend + framing note in the bottom-right gap.
    handles = [mpatches.Patch(facecolor=REGIME_FILL[r], edgecolor="#444444", linewidth=0.6,
               label=lab) for r, lab in [
        ("isolated", f"isolated roughness (λ$_f$ < {ISOLATED_MAX})"),
        ("wake", f"wake interference ({ISOLATED_MAX} ≤ λ$_f$ < {SKIMMING_MIN})"),
        ("skimming", f"skimming flow (λ$_f$ ≥ {SKIMMING_MIN})")]]
    lax = fig.add_subplot(gs[1, 3:])
    lax.axis("off")
    lax.legend(handles=handles, loc="upper left", frameon=False, fontsize=7,
               handlelength=1.6, bbox_to_anchor=(0.0, 0.98), title="Oke/Grimmond–Oke regime",
               title_fontsize=7.5, alignment="left")
    lax.text(0.0, 0.30,
             f"~{pooled['skimming']*100:.0f}% of built cells sit in skimming flow, "
             f"~{(1-pooled['isolated'])*100:.0f}% past the isolated-roughness regime: "
             "the fabric is uniformly low-ventilation.\n\n"
             "Geometry classifies the REGIME; per-cell ventilation adequacy is "
             "carried by CFD age-of-air (gated), not read off λf.",
             transform=lax.transAxes, fontsize=6.0, color="#555555", va="top", wrap=True)
    save_fig(fig, "lambda_f_regime", gate=True)


def main() -> None:
    grids = {s: load_site_grid(s) for s in SITES}
    per_site_vals = {s: grids[s]["lambda_f_mean"].to_numpy() for s in SITES}
    per_site = {s: classify_regime(v) for s, v in per_site_vals.items()}
    pooled = classify_regime(np.concatenate([per_site_vals[s] for s in SITES]))
    for s in SITES:
        c = per_site[s]
        print(f"  {s:20s} median {c['median']:.2f}  "
              f"skimming {c['skimming'] * 100:.0f}%  wake {c['wake'] * 100:.0f}%  "
              f"isolated {c['isolated'] * 100:.0f}%")
    print(f"  POOLED median {pooled['median']:.2f}  skimming {pooled['skimming'] * 100:.0f}%")

    out = {
        "lambda_f": "dissolved (canonical, party-wall corrected)",
        "thresholds": {"isolated_max": ISOLATED_MAX, "skimming_min": SKIMMING_MIN},
        "reference": "Oke (1988) / Grimmond & Oke (1999) flow-regime boundaries",
        "per_site": per_site,
        "pooled": pooled,
    }
    out_json = PROJECT_ROOT / "outputs" / "paper_figures" / "lambda_f_regime.json"
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"  wrote {out_json.relative_to(PROJECT_ROOT)}")
    make_figure(grids, per_site, pooled)


if __name__ == "__main__":
    main()
