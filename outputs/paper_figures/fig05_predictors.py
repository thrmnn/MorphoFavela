#!/usr/bin/env python3
"""Fig 05 — RF predictor importance, partial-dependence, and logistic interactions.

Four panels:
  (A) RF permutation importance, horizontal bar chart, sorted descending.
  (B) Partial-dependence curves for the top three features, three small panels.
  (C) Pooled logistic-regression coefficients with 95% CIs, plus LOSO fold
      lines for stability.
  (D) Deferred-to-v2 SVF–ACH changepoint slot (explicit placeholder panel —
      target variable requires the concurrent CFD campaign's per-cell ACH).

Inputs: outputs/paper_figures/rf_predictor_stats.json and rf_pd_curves.json
produced by scripts/run_predictor_analysis.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import (
    PROJECT_ROOT,
    WIDTH_DOUBLE,
    apply_style,
    save_fig,
)

STATS_PATH = PROJECT_ROOT / "outputs" / "paper_figures" / "rf_predictor_stats.json"
PD_PATH = PROJECT_ROOT / "outputs" / "paper_figures" / "rf_pd_curves.json"

FEATURE_LABELS = {
    "svf": r"SVF",
    "slope_deg": r"slope",
    "northness": r"northness",
    "eastness": r"eastness",
    "lambda_p": r"$\lambda_p$",
    "lambda_f_mean": r"$\lambda_f$",
    "sigma_h": r"$\sigma_H$",
    "street_orientation_entropy": "street entropy",
}

THRESHOLD_SUN_HRS = 2.0


def panel_importance(ax, stats: dict) -> None:
    imp = stats["pooled_rf"]["permutation_importance"]
    feats = [f for f, _ in imp]
    vals = [v for _, v in imp]
    labels = [FEATURE_LABELS.get(f, f) for f in feats]
    y_pos = np.arange(len(feats))
    bars = ax.barh(y_pos, vals, color="#222222", height=0.65, edgecolor="none")
    top3 = stats["pooled_rf"]["top3"]
    for i, f in enumerate(feats):
        if f in top3:
            bars[i].set_color("#666666")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("permutation importance\n(mean accuracy drop, balanced RF)", fontsize=6.5)
    ax.set_title("(A) Permutation importance — sunlight failure", loc="left", fontsize=7.5, pad=3)
    ax.tick_params(axis="x", labelsize=6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.4)
    ax.spines["left"].set_linewidth(0.4)
    auc_mean = stats["loso"]["auc_mean"]
    auc_min = stats["loso"]["auc_min"]
    auc_max = stats["loso"]["auc_max"]
    ax.text(
        0.98,
        0.04,
        f"LOSO ROC-AUC: {auc_mean:.2f}\n  range {auc_min:.2f}--{auc_max:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#888888", linewidth=0.4),
    )


def panel_pd(ax_grid, stats: dict, pd_curves: dict) -> None:
    top3 = stats["pooled_rf"]["top3"]
    for i, feat in enumerate(top3):
        ax = ax_grid[i]
        if feat not in pd_curves or "error" in pd_curves[feat]:
            ax.text(0.5, 0.5, "n/a", ha="center", va="center", fontsize=7)
            continue
        g = np.array(pd_curves[feat]["grid"])
        v = np.array(pd_curves[feat]["values"])
        ax.plot(g, v, color="#222222", lw=1.0)
        ax.fill_between(g, 0, v, color="#999999", alpha=0.25)
        ax.axhline(0.5, color="#888888", lw=0.4, ls="--")
        ax.set_ylim(0, 1)
        ax.set_xlabel(FEATURE_LABELS.get(feat, feat), fontsize=6.5)
        ax.tick_params(labelsize=6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.4)
        ax.spines["left"].set_linewidth(0.4)
        if feat == "svf":
            thr = stats["pooled_rf"].get("svf_threshold_pd_0_5")
            if thr is not None:
                ax.axvline(thr, color="#222222", lw=0.6, ls=":")
                ax.text(
                    thr,
                    0.96,
                    f"SVF={thr:.2f}",
                    fontsize=5.5,
                    rotation=90,
                    ha="right",
                    va="top",
                    color="#222222",
                )
        if i == 0:
            ax.set_ylabel("P(sunlight fail)", fontsize=6.5)
        else:
            ax.set_yticklabels([])
    ax_grid[1].text(
        0.5,
        1.10,
        "(B) Partial-dependence — top 3 features",
        transform=ax_grid[1].transAxes,
        fontsize=7.5,
        ha="center",
        va="bottom",
    )


def panel_coefficients(ax, stats: dict) -> None:
    if "error" in stats["logit_interactions"]:
        ax.text(0.5, 0.5, "logit fit failed", ha="center", va="center")
        return
    coefs = stats["logit_interactions"]["coefficients"]
    name_order = [
        "svf",
        "northness",
        "slope_deg",
        "slope_x_northness",
        "slope_x_svf",
        "lambda_p",
        "lambda_f_mean",
        "sigma_h",
    ]
    name_order = [n for n in name_order if n in coefs]
    pretty = {
        "svf": r"SVF",
        "northness": r"northness",
        "slope_deg": r"slope",
        "slope_x_northness": r"slope $\times$ northness",
        "slope_x_svf": r"slope $\times$ SVF",
        "lambda_p": r"$\lambda_p$",
        "lambda_f_mean": r"$\lambda_f$",
        "sigma_h": r"$\sigma_H$",
    }
    y = np.arange(len(name_order))
    est = [coefs[n]["estimate"] for n in name_order]
    lo = [coefs[n]["ci_low"] for n in name_order]
    hi = [coefs[n]["ci_high"] for n in name_order]
    ax.errorbar(
        est,
        y,
        xerr=[np.array(est) - np.array(lo), np.array(hi) - np.array(est)],
        fmt="o",
        color="#222222",
        ecolor="#444444",
        markersize=3.0,
        capsize=2.0,
        lw=0.8,
    )
    # LOSO fold lines behind pooled
    loso_folds = stats["logit_interactions"].get("loso_folds", [])
    for fold in loso_folds:
        c = fold["coefficients"]
        xs = [c.get(n, np.nan) for n in name_order]
        ax.plot(xs, y, color="#bbbbbb", lw=0.4, alpha=0.7, zorder=1)
    ax.axvline(0, color="#888888", lw=0.4, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([pretty[n] for n in name_order], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("standardized logit coefficient (95% CI)\nfaint lines = LOSO folds", fontsize=6.5)
    ax.set_title("(C) Pooled logistic regression coefficients", loc="left", fontsize=7.5, pad=3)
    ax.tick_params(axis="x", labelsize=6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.4)
    ax.spines["left"].set_linewidth(0.4)
    pr2 = stats["logit_interactions"].get("pseudo_r2")
    if pr2 is not None:
        ax.text(
            0.98,
            0.04,
            f"pseudo $R^2$ = {pr2:.2f}  ·  n = {stats['logit_interactions']['n']:,}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.5,
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="white", edgecolor="#888888", linewidth=0.4
            ),
        )


def panel_deferred(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.4)
        spine.set_linestyle("--")
        spine.set_color("#888888")
    ax.set_facecolor("#fbfbfb")
    ax.text(
        0.5,
        0.62,
        "SVF–ACH changepoint",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.0,
        color="#444444",
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.46,
        "deferred to the CFD-validation paper\n(target variable requires per-cell ACH\nfrom the concurrent OpenFOAM RANS campaign)",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.0,
        color="#666666",
    )
    ax.set_title("(D) Pre-registered for v2", loc="left", fontsize=7.5, pad=3, color="#666666")


def main() -> None:
    apply_style()
    stats = json.loads(STATS_PATH.read_text())
    pd_curves = json.loads(PD_PATH.read_text())

    fig = plt.figure(figsize=(WIDTH_DOUBLE, WIDTH_DOUBLE * 0.72))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=4,
        width_ratios=[1.0, 0.55, 0.55, 0.55],
        height_ratios=[1.0, 1.0],
        wspace=0.55,
        hspace=0.45,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    panel_importance(ax_a, stats)

    ax_b1 = fig.add_subplot(gs[0, 1])
    ax_b2 = fig.add_subplot(gs[0, 2])
    ax_b3 = fig.add_subplot(gs[0, 3])
    panel_pd([ax_b1, ax_b2, ax_b3], stats, pd_curves)

    ax_c = fig.add_subplot(gs[1, 0:2])
    panel_coefficients(ax_c, stats)

    ax_d = fig.add_subplot(gs[1, 2:4])
    panel_deferred(ax_d)

    save_fig(fig, "fig05_predictors")


if __name__ == "__main__":
    main()
