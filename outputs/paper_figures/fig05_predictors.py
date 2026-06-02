#!/usr/bin/env python3
"""Fig 05 — RF predictor importance + partial-dependence (slide-optimised v3).

Two panels at ~16:9 (3600×2025 px @ 300 DPI ≈ 12.0 × 6.75 in):

  (A) LEFT — Horizontal bar chart of permutation importance for the top 4
      features (SVF, southness, slope, λf). SVF in terracotta accent;
      supporting features in muted slate. LOSO ROC-AUC range annotated.

  (B) RIGHT — Two stacked sub-panels:
        (B-top) SVF partial-dependence curve with the 0.26 inflection marked.
        (B-bot) Empirical P(sunlight failure) per aspect quadrant (N/E/S/W),
                horizontal bar chart, bars sorted descending. South in
                terracotta accent, others slate. 0.5 reference line.

Southness convention: southness = -cos(aspect), so positive values mean
south-facing slopes (which receive less winter sun in Rio's southern
hemisphere).

Aspect quadrant binning recovers aspect from (southness, eastness):
  aspect = atan2(eastness, -southness)
Quadrant edges: N=[315°,45°), E=[45°,135°), S=[135°,225°), W=[225°,315°).

Inputs:
  outputs/paper_figures/rf_predictor_stats.json
  outputs/paper_figures/rf_pd_curves.json
  outputs/paper_figures/rf_pooled_data.parquet  (for empirical quadrant rates)

Outputs (both written for redundancy):
  - artifacts/latex/figures/fig05_predictors.png   (manuscript path)
  - artifacts/slides/assets/fig05_predictors_v2.png (slide-builder path)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import PROJECT_ROOT, apply_style
sys.path.insert(0, str(PROJECT_ROOT))
from src.viz import presentation_style as ps  # noqa: E402

STATS_PATH = PROJECT_ROOT / "outputs" / "paper_figures" / "rf_predictor_stats.json"
PD_PATH = PROJECT_ROOT / "outputs" / "paper_figures" / "rf_pd_curves.json"
DATA_PATH = PROJECT_ROOT / "outputs" / "paper_figures" / "rf_pooled_data.parquet"

# Slide-friendly output paths (overwrite both)
BRISA_ROOT = Path("/home/theo/brisa_paper")
OUT_MANUSCRIPT = BRISA_ROOT / "artifacts" / "latex" / "figures" / "fig05_predictors.png"
OUT_SLIDE = BRISA_ROOT / "artifacts" / "slides" / "assets" / "fig05_predictors_v2.png"

# Presentation preset writes to a new per-site (well, paper_figures here)
# presentation_figures/ directory alongside the existing dual-write.
PRESENTATION_OUT = (
    PROJECT_ROOT / "outputs" / "paper_figures" / "presentation_figures" / "fig05_predictors.png"
)

# Top-4 features to show in panel A (ordered by current importance ranking).
TOP4 = ["svf", "southness", "slope_deg", "lambda_f_mean"]

FEATURE_LABELS = {
    "svf": "SVF",
    "slope_deg": "slope",
    "southness": "southness",
    "northness": "northness",
    "eastness": "eastness",
    "lambda_p": r"$\lambda_p$",
    "lambda_f_mean": r"$\lambda_f$",
    "sigma_h": r"$\sigma_H$",
    "street_orientation_entropy": "street entropy",
}

# Palette
ACCENT = "#E76F51"   # terracotta — matches deck
SLATE = "#5A6B7C"    # muted slate for supporting bars/curves
INK = "#1F2933"      # near-black for axes, text
GRID = "#D9D9D9"

# Font sizes (audience-readable on slides)
FS_PANEL = 18          # A, B labels (bold)
FS_AXLABEL = 15        # axis labels
FS_TICK = 13           # tick labels
FS_CAPTION = 12        # in-chart captions (e.g. inflection callout)
FS_QUAD = 16           # N/E/S/W quadrant labels (bold)
FS_BARVAL = 13         # value labels on bars
FS_SUBHEADER = 14      # right-panel sub-header
FS_FEATURE_BADGE = 15  # in-panel feature name badge


def _imp_dict(stats: dict) -> dict[str, float]:
    return {f: v for f, v in stats["pooled_rf"]["permutation_importance"]}


def panel_importance(ax, stats: dict) -> None:
    imp = _imp_dict(stats)
    vals = [imp[f] for f in TOP4]
    labels = [FEATURE_LABELS[f] for f in TOP4]
    y_pos = np.arange(len(TOP4))
    colors = [ACCENT if f == "svf" else SLATE for f in TOP4]

    ax.barh(y_pos, vals, color=colors, height=0.62, edgecolor="none", zorder=3)
    for i, v in enumerate(vals):
        ax.text(
            v + max(vals) * 0.012, i, f"{v:.3f}",
            va="center", ha="left", fontsize=FS_BARVAL, color=INK, zorder=4,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=FS_AXLABEL, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel(
        "Permutation importance  (mean accuracy drop)",
        fontsize=FS_AXLABEL, color=INK, labelpad=8,
    )
    ax.tick_params(axis="x", labelsize=FS_TICK, color=INK, length=3, width=0.6)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(vals) * 1.18)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.spines["bottom"].set_color(INK)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["left"].set_color(INK)

    ax.xaxis.grid(True, color=GRID, lw=0.5, zorder=1)
    ax.set_axisbelow(True)

    # Panel label A
    ax.text(
        -0.20, 1.02, "A", transform=ax.transAxes,
        fontsize=FS_PANEL, fontweight="bold", color=INK, va="bottom", ha="left",
    )

    # LOSO annotation
    auc_min = stats["loso"]["auc_min"]
    auc_max = stats["loso"]["auc_max"]
    ax.text(
        0.98, 0.03,
        f"LOSO ROC-AUC: {auc_min:.2f}–{auc_max:.2f}\n(5-fold, leave-one-site-out)",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=FS_CAPTION, color=INK,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor=SLATE, linewidth=0.6),
    )


def panel_svf_pd(ax, stats: dict, pd_curves: dict) -> None:
    g = np.array(pd_curves["svf"]["grid"])
    v = np.array(pd_curves["svf"]["values"])

    ax.plot(g, v, color=ACCENT, lw=2.4, zorder=3)
    ax.fill_between(g, 0, v, color=ACCENT, alpha=0.18, zorder=2)
    ax.axhline(0.5, color="#888888", lw=0.7, ls="--", zorder=1)

    ax.set_ylim(0, 1)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.tick_params(axis="x", labelsize=FS_TICK, color=INK, length=3, width=0.6)
    ax.tick_params(axis="y", labelsize=FS_TICK, color=INK, length=3, width=0.6)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.spines["bottom"].set_color(INK)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["left"].set_color(INK)

    ax.yaxis.grid(True, color=GRID, lw=0.4, zorder=0)
    ax.set_axisbelow(True)

    ax.set_ylabel("P(fail)", fontsize=FS_AXLABEL, color=INK, labelpad=4)
    ax.set_xlabel("SVF", fontsize=FS_AXLABEL, color=INK, labelpad=4)

    # In-panel feature badge
    ax.text(
        0.985, 0.92, "SVF",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=FS_FEATURE_BADGE, color=INK, fontweight="bold",
    )

    thr = stats["pooled_rf"].get("svf_threshold_pd_0_5")
    if thr is not None:
        ax.axvline(thr, color=INK, lw=1.0, ls=":", zorder=4)
        ax.text(
            thr + 0.005, 0.94,
            f"SVF inflection ≈ {thr:.2f}",
            fontsize=FS_CAPTION, color=INK,
            ha="left", va="top",
        )


def compute_quadrant_pfail(df: pd.DataFrame) -> list[tuple[str, float, int]]:
    """Compute empirical P(sunlight failure) per aspect quadrant.

    Returns list of (label, p_fail, n) tuples sorted descending by p_fail.
    """
    aspect_rad = np.arctan2(df["eastness"].values, -df["southness"].values)
    aspect_deg = np.rad2deg(aspect_rad) % 360.0

    def quad(a: float) -> str:
        if (a >= 315.0) or (a < 45.0):
            return "N"
        if a < 135.0:
            return "E"
        if a < 225.0:
            return "S"
        return "W"

    df = df.copy()
    df["__quad"] = [quad(a) for a in aspect_deg]
    rows = []
    for label in ("N", "E", "S", "W"):
        sub = df[df["__quad"] == label]
        if len(sub) == 0:
            continue
        rows.append((label, float(sub["sun_fail"].mean()), int(len(sub))))
    rows.sort(key=lambda r: -r[1])
    return rows


def panel_aspect_quadrants(ax, df: pd.DataFrame) -> None:
    rows = compute_quadrant_pfail(df)
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    ns = [r[2] for r in rows]
    y_pos = np.arange(len(rows))
    colors = [ACCENT if lab == "S" else SLATE for lab in labels]

    ax.barh(y_pos, vals, color=colors, height=0.62, edgecolor="none", zorder=3)
    for i, (v, n) in enumerate(zip(vals, ns)):
        ax.text(
            v + 0.012, i, f"{v*100:.0f}%  (n={n:,})",
            va="center", ha="left", fontsize=FS_BARVAL, color=INK, zorder=4,
        )

    # 0.5 reference line + caption above top bar
    ax.axvline(0.5, color="#888888", lw=0.9, ls="--", zorder=1)
    ax.text(
        0.505, -0.55, "P = 0.5",
        ha="left", va="bottom", fontsize=FS_CAPTION, color="#666666",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=FS_QUAD, fontweight="bold", color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.50", "0.75", "1.0"])
    ax.set_xlabel(
        "P(sunlight failure)  — empirical share per aspect quadrant",
        fontsize=FS_AXLABEL, color=INK, labelpad=6,
    )
    ax.tick_params(axis="x", labelsize=FS_TICK, color=INK, length=3, width=0.6)
    ax.tick_params(axis="y", length=0)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.spines["bottom"].set_color(INK)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["left"].set_color(INK)

    ax.xaxis.grid(True, color=GRID, lw=0.5, zorder=1)
    ax.set_axisbelow(True)


def panel_b(axes, stats: dict, pd_curves: dict, df: pd.DataFrame) -> None:
    """Panel B: SVF PD (top) + aspect-quadrant bars (bottom)."""
    panel_svf_pd(axes[0], stats, pd_curves)
    panel_aspect_quadrants(axes[1], df)

    # Panel label B on top-most axis
    axes[0].text(
        -0.14, 1.05, "B", transform=axes[0].transAxes,
        fontsize=FS_PANEL, fontweight="bold", color=INK, va="bottom", ha="left",
    )
    # Shared sub-header
    axes[0].text(
        0.5, 1.20,
        "How predictors map to sunlight-failure risk",
        transform=axes[0].transAxes,
        fontsize=FS_SUBHEADER, color=INK, ha="center", va="bottom",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--preset", choices=("paper", "presentation"), default="paper")
    args = parser.parse_args()

    apply_style()
    ps.apply(args.preset)

    stats = json.loads(STATS_PATH.read_text())
    pd_curves = json.loads(PD_PATH.read_text())
    df = pd.read_parquet(DATA_PATH)

    # Report quadrant numbers to stdout for sanity-checking.
    quads = compute_quadrant_pfail(df)
    print("Aspect quadrant P(sunlight failure):")
    for label, p, n in quads:
        print(f"  {label}: {p*100:5.1f}%   (n={n:,})")

    # 16:9 figure, 12 in × 6.75 in @ 300 DPI = 3600 × 2025 px
    fig = plt.figure(figsize=(12.0, 6.75), dpi=300)

    # Outer layout: A on the left (wider), B on the right (2 stacked panels)
    outer = fig.add_gridspec(
        nrows=1, ncols=2,
        width_ratios=[1.0, 1.05],
        wspace=0.30,
        left=0.095, right=0.965, top=0.88, bottom=0.13,
    )

    ax_a = fig.add_subplot(outer[0, 0])
    panel_importance(ax_a, stats)

    inner = outer[0, 1].subgridspec(nrows=2, ncols=1, hspace=0.55)
    axes_b = [fig.add_subplot(inner[i, 0]) for i in range(2)]
    panel_b(axes_b, stats, pd_curves, df)

    save_kw = dict(dpi=300, facecolor="white", bbox_inches=None, pad_inches=0.0)
    if args.preset == "presentation":
        PRESENTATION_OUT.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(PRESENTATION_OUT, **save_kw)
        print(f"  wrote {PRESENTATION_OUT}")
    else:
        OUT_MANUSCRIPT.parent.mkdir(parents=True, exist_ok=True)
        OUT_SLIDE.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_MANUSCRIPT, **save_kw)
        fig.savefig(OUT_SLIDE, **save_kw)
        print(f"  wrote {OUT_MANUSCRIPT}")
        print(f"  wrote {OUT_SLIDE}")
    plt.close(fig)


if __name__ == "__main__":
    main()
