#!/usr/bin/env python3
"""Fig 05 — predictors of the winter sun-adequacy floor (4-panel A–D).

Target throughout: P(winter direct sun < WHO 2 h) — failure of the
sun-adequacy floor, NOT a ventilation outcome (which stays CFD-gated).

  (A) Random-forest permutation importance, GROUPED and colored by feature
      family (openness / terrain-aspect / density) with a small family
      legend. Bars are ordered by family (families ranked by their top
      member's importance), descending within each family. An honesty box
      reports the leave-one-site-out transfer: ROC-AUC (ranking) 0.86–0.93
      transfers across cities far better than accuracy (thresholded
      decisions) 0.73–0.86 — quoting only the AUC would overstate
      field-ready accuracy; the decision threshold is site-specific. The
      aspect feature is labelled "northness"
      (importance is sign-agnostic, so the value is identical to the stored
      "southness" entry — only the label changes, to match panels B/C and
      the manuscript caption).

  (B) Partial-dependence curves for the top three features (SVF, northness,
      slope), each colored by its family hue (SVF openness, northness/slope
      terrain — differentiated by linestyle). northness is derived from the
      stored southness PD curve by negating the grid (northness = -southness)
      and re-sorting ascending. The SVF 0.5 crossing (≈0.26) is marked dotted.

  (C) Pooled logistic-regression coefficients on standardized predictors,
      converted to the northness convention (southness-derived terms have
      their sign flipped). Markers are SIGN-COLORED on a diverging scale —
      blue = negative β (reduces P(fail)), red = positive β (raises P(fail))
      — with saturation ∝ |β| and a small sign legend. CI caps and per-fold
      LOSO dots stay neutral ink/grey. pseudo-R² and n annotated.

  (D) Greyed placeholder for the SVF–ACH changepoint, pending the OpenFOAM
      RANS campaign (per-cell ACH not yet available). No data fabricated.

Two color encodings, intentionally distinct: A and B encode feature FAMILY
by hue (importance/PD are about which family of predictors matters); C
encodes the SIGN of the coefficient by hue (the regression panel is about
direction of effect). Legends in each panel name their own encoding so the
two are unambiguous.

northness convention: northness = -southness = cos(aspect). More northness =
north-facing = more winter sun (southern hemisphere) = lower P(fail).

Inputs:
  outputs/paper_figures/rf_predictor_stats.json
  outputs/paper_figures/rf_pd_curves.json

Outputs:
  artifacts/latex/figures/fig05_predictors.png   (manuscript path)
  artifacts/slides/assets/fig05_predictors_v2.png (slide path)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import PROJECT_ROOT, apply_style
sys.path.insert(0, str(PROJECT_ROOT))
from src.viz import presentation_style as ps  # noqa: E402

STATS_PATH = PROJECT_ROOT / "outputs" / "paper_figures" / "rf_predictor_stats.json"
PD_PATH = PROJECT_ROOT / "outputs" / "paper_figures" / "rf_pd_curves.json"

# The prediction target is failure of the winter sun-adequacy floor (WHO 2 h of
# winter-solstice direct sun). Name it precisely everywhere instead of "P(fail)".
TARGET_LABEL = r"P(winter sun $<$ 2 h)"  # below the sun-adequacy floor
TARGET_SHORT = r"P($<$2 h sun)"

BRISA_ROOT = Path("/home/theo/brisa_paper")
OUT_MANUSCRIPT = BRISA_ROOT / "artifacts" / "latex" / "figures" / "fig05_predictors.png"
OUT_SLIDE = BRISA_ROOT / "artifacts" / "slides" / "assets" / "fig05_predictors_v2.png"
PRESENTATION_OUT = (
    PROJECT_ROOT / "outputs" / "paper_figures" / "presentation_figures" / "fig05_predictors.png"
)
# Local, always-writable export so the recolor is verifiable without brisaverse.
LOCAL_EXPORT = PROJECT_ROOT / "outputs" / "paper_figures" / "exports" / "fig05_predictors.png"

# Panel A: top features, with the aspect feature relabelled to "northness".
TOP_A = ["svf", "southness", "slope_deg", "lambda_f_mean"]

FEATURE_LABELS = {
    "svf": "SVF",
    "slope_deg": "slope",
    "southness": "northness",   # relabel: importance is sign-agnostic
    "northness": "northness",
    "eastness": "eastness",
    "lambda_p": r"$\lambda_p$",
    "lambda_f_mean": r"$\lambda_f$",
    "sigma_h": r"$\sigma_H$",
    "street_orientation_entropy": "street entropy",
}

# Logit coefficient plot (panel C) — northness convention. southness-derived
# terms have their sign flipped relative to the stored southness fit.
LOGIT_TERMS = ["svf", "slope_deg", "northness", "slope_x_northness", "slope_x_svf"]
LOGIT_LABELS = {
    "svf": "SVF",
    "slope_deg": "slope",
    "northness": "northness",
    "slope_x_northness": "slope × northness",
    "slope_x_svf": "slope × SVF",
}
# Map a plotting term to (stored_term, sign_flip).
TERM_SOURCE = {
    "svf": ("svf", 1.0),
    "slope_deg": ("slope_deg", 1.0),
    "northness": ("southness", -1.0),
    "slope_x_northness": ("slope_x_southness", -1.0),
    "slope_x_svf": ("slope_x_svf", 1.0),
}

# Shared feature-FAMILY palette — one hue per family, reused across A/B/C.
# Colorblind-safe trio: warm terracotta vs cool steel-blue vs violet are
# distinguishable under deuteranopia/protanopia. SVF keeps the headline
# terracotta (ACCENT) it always had.
GROUP_COLORS = {
    "openness": "#E76F51",   # terracotta — SVF / sky openness
    "terrain":  "#2A7DB5",   # steel blue — northness / slope / aspect
    "density":  "#7B5EA7",   # violet     — lambda_f / lambda_p / sigma_h
}
GROUP_LABELS = {"openness": "openness", "terrain": "terrain-aspect", "density": "density"}
FEATURE_GROUP = {
    "svf": "openness",
    "southness": "terrain",
    "northness": "terrain",
    "slope_deg": "terrain",
    "eastness": "terrain",
    "lambda_f_mean": "density",
    "lambda_p": "density",
    "sigma_h": "density",
    "street_orientation_entropy": "terrain",
    "slope_x_northness": "terrain",
    "slope_x_southness": "terrain",
    "slope_x_svf": "openness",  # interaction with SVF → openness family tint
}

# Panel C sign-diverging scale (encodes direction of effect, not family).
NEG_HUE = "#2166AC"  # blue — negative β (reduces P(fail))
POS_HUE = "#B2182B"  # red  — positive β (raises P(fail))

ACCENT = GROUP_COLORS["openness"]  # terracotta (SVF headline)
SLATE = "#5A6B7C"    # supporting slate (annotation edges)
INK = "#1F2933"
GRID = "#D9D9D9"
FAINT = "#B7BFC7"    # LOSO fold dots

FS_PANEL = 17
FS_AXLABEL = 13
FS_TICK = 11
FS_CAPTION = 11
FS_FEATURE_BADGE = 13


def _imp_dict(stats: dict) -> dict[str, float]:
    return {f: v for f, v in stats["pooled_rf"]["permutation_importance"]}


def _feat_color(feat: str) -> str:
    return GROUP_COLORS[FEATURE_GROUP[feat]]


def _desaturate(hex_color: str, sat: float) -> tuple[float, float, float]:
    """Blend a hue toward white; sat in [0,1], 1 = full color, 0 = white."""
    r, g, b = to_rgb(hex_color)
    sat = float(np.clip(sat, 0.18, 1.0))
    return (1 - sat) + sat * r, (1 - sat) + sat * g, (1 - sat) + sat * b


def _group_order(feats: list[str], imp: dict[str, float]) -> list[str]:
    """Order features by family (families ranked by their top member's
    importance), descending within each family. Keeps family members
    adjacent while preserving the descending-importance reading."""
    fam_top = {}
    for f in feats:
        fam = FEATURE_GROUP[f]
        fam_top[fam] = max(fam_top.get(fam, -np.inf), imp[f])
    fam_rank = sorted(fam_top, key=lambda g: fam_top[g], reverse=True)
    return sorted(feats, key=lambda f: (fam_rank.index(FEATURE_GROUP[f]), -imp[f]))


# ── Panel A ──────────────────────────────────────────────────────────────
def panel_importance(ax, stats: dict) -> None:
    imp = _imp_dict(stats)
    order = _group_order(TOP_A, imp)
    vals = [imp[f] for f in order]
    labels = [FEATURE_LABELS[f] for f in order]
    y_pos = np.arange(len(order))
    colors = [_feat_color(f) for f in order]

    ax.barh(y_pos, vals, color=colors, height=0.62, edgecolor="none", zorder=3)
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.015, i, f"{v:.3f}",
                va="center", ha="left", fontsize=FS_TICK, color=INK, zorder=4)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=FS_AXLABEL, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("Permutation importance (mean accuracy drop)",
                  fontsize=FS_AXLABEL, color=INK, labelpad=6)
    ax.tick_params(axis="x", labelsize=FS_TICK, color=INK, length=3, width=0.6)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(vals) * 1.22)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_linewidth(0.7)
        ax.spines[spine].set_color(INK)
    ax.xaxis.grid(True, color=GRID, lw=0.5, zorder=1)
    ax.set_axisbelow(True)

    ax.text(-0.30, 1.04, "A", transform=ax.transAxes,
            fontsize=FS_PANEL, fontweight="bold", color=INK, va="bottom", ha="left")
    ax.set_title("Random-forest feature importance", fontsize=FS_AXLABEL,
                 color=INK, pad=6, loc="left")

    # Honesty box: AUC (ranking) transfers across cities far better than
    # accuracy (thresholded decisions). Both are leave-one-site-out so neither
    # is in-sample; reporting only the AUC would overstate field-ready accuracy.
    lo = stats["loso"]
    box = (
        "cross-site (LOSO, 5 held-out favelas)\n"
        f"ROC-AUC   {lo['auc_min']:.2f}–{lo['auc_max']:.2f}   ranking\n"
        f"accuracy  {lo['acc_min']:.2f}–{lo['acc_max']:.2f}   decisions\n"
        "rank transfers; threshold is site-specific"
    )
    ax.text(0.975, 0.04, box, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=FS_CAPTION - 2, color=INK, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#FBFAF6",
                      edgecolor=SLATE, linewidth=0.7))

    fams = ("openness", "terrain", "density")
    handles = [plt.Line2D([0], [0], marker="s", ls="none", markersize=8,
                          markerfacecolor=GROUP_COLORS[g], markeredgecolor="none")
               for g in fams]
    ax.legend(handles, [GROUP_LABELS[g] for g in fams],
              loc="lower right", bbox_to_anchor=(0.99, 0.30),
              fontsize=FS_CAPTION - 1, frameon=False, handletextpad=0.4,
              title="feature family", title_fontsize=FS_CAPTION - 1,
              labelspacing=0.3)


# ── Panel B ──────────────────────────────────────────────────────────────
def _northness_pd(pd_curves: dict) -> tuple[np.ndarray, np.ndarray]:
    g = np.array(pd_curves["southness"]["grid"], dtype=float)
    v = np.array(pd_curves["southness"]["values"], dtype=float)
    ng = -g
    order = np.argsort(ng)
    return ng[order], v[order]


def panel_pd(ax, stats: dict, pd_curves: dict) -> None:
    svf_g = np.array(pd_curves["svf"]["grid"])
    svf_v = np.array(pd_curves["svf"]["values"])
    nor_g, nor_v = _northness_pd(pd_curves)
    slp_g = np.array(pd_curves["slope_deg"]["grid"])
    slp_v = np.array(pd_curves["slope_deg"]["values"])

    # Normalize each feature's grid to [0,1] so three curves share one x-axis.
    def norm(g):
        return (g - g.min()) / (g.max() - g.min())

    ax.plot(norm(svf_g), svf_v, color=_feat_color("svf"), lw=2.4, zorder=4,
            label="SVF")
    ax.plot(norm(nor_g), nor_v, color=_feat_color("northness"), lw=2.2, zorder=3,
            label="northness")
    # northness and slope share the terrain hue → differentiated by linestyle.
    ax.plot(norm(slp_g), slp_v, color=_feat_color("slope_deg"), lw=2.2,
            ls=(0, (4, 1.5)), zorder=3, label="slope")

    ax.axhline(0.5, color="#999999", lw=0.7, ls="--", zorder=1)

    # SVF 0.5 crossing (≈0.26) marked on the normalized axis.
    thr = stats["pooled_rf"].get("svf_threshold_pd_0_5")
    if thr is not None:
        thr_norm = (thr - svf_g.min()) / (svf_g.max() - svf_g.min())
        ax.axvline(thr_norm, color=_feat_color("svf"), lw=1.0, ls=":", zorder=4)
        ax.text(thr_norm + 0.015, 0.95, f"SVF crossing ≈ {thr:.2f}",
                fontsize=FS_CAPTION, color=_feat_color("svf"), ha="left", va="top")

    ax.set_ylim(0, 1)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["low", "mid", "high"])
    ax.tick_params(axis="both", labelsize=FS_TICK, color=INK, length=3, width=0.6)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_linewidth(0.7)
        ax.spines[spine].set_color(INK)
    ax.yaxis.grid(True, color=GRID, lw=0.4, zorder=0)
    ax.set_axisbelow(True)

    ax.set_ylabel(TARGET_LABEL, fontsize=FS_AXLABEL, color=INK, labelpad=4)
    ax.set_xlabel("feature value (normalized range)", fontsize=FS_AXLABEL,
                  color=INK, labelpad=4)
    ax.legend(loc="lower left", fontsize=FS_CAPTION, frameon=False,
              handlelength=1.6, bbox_to_anchor=(0.0, 0.0))

    ax.text(-0.20, 1.04, "B", transform=ax.transAxes,
            fontsize=FS_PANEL, fontweight="bold", color=INK, va="bottom", ha="left")
    ax.set_title("Partial-dependence (top 3 features)", fontsize=FS_AXLABEL,
                 color=INK, pad=6, loc="left")


# ── Panel C ──────────────────────────────────────────────────────────────
def panel_logit(ax, stats: dict) -> None:
    li = stats["logit_interactions"]
    coefs = li["coefficients"]
    folds = li["loso_folds"]

    terms = LOGIT_TERMS
    y_pos = np.arange(len(terms))[::-1]  # first term at top

    est, lo, hi = [], [], []
    for t in terms:
        src, sign = TERM_SOURCE[t]
        c = coefs[src]
        est.append(sign * c["estimate"])
        # CI bounds flip and swap when the sign flips.
        a = sign * c["ci_low"]
        b = sign * c["ci_high"]
        lo.append(min(a, b))
        hi.append(max(a, b))

    max_abs = max(abs(e) for e in est)

    ax.axvline(0, color="#777777", lw=0.8, zorder=1)

    # LOSO per-fold estimates as faint neutral dots (sign-stability cloud).
    for j, t in enumerate(terms):
        src, sign = TERM_SOURCE[t]
        fold_vals = [sign * f["coefficients"][src] for f in folds]
        ax.scatter(fold_vals, [y_pos[j]] * len(fold_vals),
                   s=14, color=FAINT, alpha=0.9, zorder=2, linewidths=0)

    # 95% CI caps in neutral ink.
    for j in range(len(terms)):
        ax.plot([lo[j], hi[j]], [y_pos[j], y_pos[j]], color=INK, lw=1.4,
                zorder=3, solid_capstyle="butt")
        for xc in (lo[j], hi[j]):
            ax.plot([xc, xc], [y_pos[j] - 0.16, y_pos[j] + 0.16], color=INK,
                    lw=1.4, zorder=3)

    # Point markers SIGN-COLORED on a diverging scale, saturation ∝ |β|.
    marker_colors = [_desaturate(NEG_HUE if e < 0 else POS_HUE, abs(e) / max_abs)
                     for e in est]
    edge_colors = [NEG_HUE if e < 0 else POS_HUE for e in est]
    ax.scatter(est, y_pos, s=64, color=marker_colors, zorder=4,
               edgecolor=edge_colors, linewidths=1.2)

    for j, t in enumerate(terms):
        ax.text(est[j], y_pos[j] + 0.30, f"{est[j]:+.2f}",
                ha="center", va="bottom", fontsize=FS_CAPTION - 1, color=INK,
                zorder=5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([LOGIT_LABELS[t] for t in terms],
                       fontsize=FS_AXLABEL, color=INK)
    ax.set_ylim(min(y_pos) - 0.6, max(y_pos) + 0.6)
    ax.set_xlabel("standardized logit coefficient (β)", fontsize=FS_AXLABEL,
                  color=INK, labelpad=4)
    ax.tick_params(axis="x", labelsize=FS_TICK, color=INK, length=3, width=0.6)
    ax.tick_params(axis="y", length=0)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_linewidth(0.7)
        ax.spines[spine].set_color(INK)
    ax.xaxis.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)

    sign_handles = [
        plt.Line2D([0], [0], marker="o", ls="none", markersize=9,
                   markerfacecolor=_desaturate(NEG_HUE, 0.85),
                   markeredgecolor=NEG_HUE, markeredgewidth=1.1),
        plt.Line2D([0], [0], marker="o", ls="none", markersize=9,
                   markerfacecolor=_desaturate(POS_HUE, 0.85),
                   markeredgecolor=POS_HUE, markeredgewidth=1.1),
    ]
    ax.legend(sign_handles,
              [f"−  reduces {TARGET_SHORT}", f"+  raises {TARGET_SHORT}"],
              loc="lower left", fontsize=FS_CAPTION - 1, frameon=False,
              handletextpad=0.4, labelspacing=0.3, bbox_to_anchor=(0.0, 0.02))

    r2 = li.get("pseudo_r2", 0.0)
    n = li.get("n", 0)
    ax.text(0.02, 0.40, f"pseudo-R² = {r2:.2f}\nn = {n:,}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=FS_CAPTION, color=INK,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                      edgecolor=SLATE, linewidth=0.6))

    ax.text(-0.30, 1.04, "C", transform=ax.transAxes,
            fontsize=FS_PANEL, fontweight="bold", color=INK, va="bottom", ha="left")
    ax.set_title("Pooled logistic regression (standardized)",
                 fontsize=FS_AXLABEL, color=INK, pad=6, loc="left")


# ── Panel D ──────────────────────────────────────────────────────────────
def panel_changepoint_placeholder(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor="#EEEEEE",
                               edgecolor="#BBBBBB", linewidth=0.8,
                               hatch="////", zorder=0))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(0.5, 0.58,
            "pending the OpenFOAM RANS campaign\n(per-cell ACH not yet available)",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=FS_CAPTION, color="#666666", style="italic",
            linespacing=1.5)

    ax.text(-0.10, 1.04, "D", transform=ax.transAxes,
            fontsize=FS_PANEL, fontweight="bold", color=INK, va="bottom", ha="left")
    ax.set_title("SVF–ACH changepoint", fontsize=FS_AXLABEL, color="#666666",
                 pad=6, loc="left")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--preset", choices=("paper", "presentation"), default="paper")
    args = parser.parse_args()

    apply_style()
    ps.apply(args.preset)

    stats = json.loads(STATS_PATH.read_text())
    pd_curves = json.loads(PD_PATH.read_text())

    fig = plt.figure(figsize=(12.0, 9.0), dpi=300)
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        width_ratios=[1.0, 1.05], height_ratios=[1.0, 1.0],
        wspace=0.32, hspace=0.42,
        left=0.085, right=0.965, top=0.93, bottom=0.075,
    )

    panel_importance(fig.add_subplot(gs[0, 0]), stats)
    panel_pd(fig.add_subplot(gs[0, 1]), stats, pd_curves)
    panel_logit(fig.add_subplot(gs[1, 0]), stats)
    panel_changepoint_placeholder(fig.add_subplot(gs[1, 1]))

    save_kw = dict(dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.1)
    if args.preset == "presentation":
        PRESENTATION_OUT.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(PRESENTATION_OUT, **save_kw)
        print(f"  wrote {PRESENTATION_OUT}")
    else:
        OUT_MANUSCRIPT.parent.mkdir(parents=True, exist_ok=True)
        OUT_SLIDE.parent.mkdir(parents=True, exist_ok=True)
        LOCAL_EXPORT.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_MANUSCRIPT, **save_kw)
        fig.savefig(OUT_SLIDE, **save_kw)
        fig.savefig(LOCAL_EXPORT, **save_kw)
        print(f"  wrote {OUT_MANUSCRIPT}")
        print(f"  wrote {OUT_SLIDE}")
        print(f"  wrote {LOCAL_EXPORT}")
    plt.close(fig)


if __name__ == "__main__":
    main()
