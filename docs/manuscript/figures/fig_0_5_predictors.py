#!/usr/bin/env python3
"""Figure 0.5 — Predictors and typology contrast.

Statistical findings figure. Four panels:

  A. Random forest permutation importance (vent vs sun, grouped bars).
  B. Partial-dependence curves (2 × 3: targets × {SVF, λf, slope}).
  C. Pooled logistic forest plot (main effects + interactions, vent
     and sun stacked sub-rows).
  D. SVF → annual U_mean changepoint regression (patch-level), with
     bootstrap-95% breakpoint CI and a twin top axis showing
     indoor-equivalent ACH (α = 1/150).

Data is read from
``outputs/comparative/diagnostic_models/`` — produced by
``scripts/run_diagnostic_models.py``. Re-run that script when CFD or
solar data refreshes.

Run::

    python docs/manuscript/figures/fig_0_5_predictors.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "outputs" / "paper_figures"))
from fig_style import WIDTH_DOUBLE, apply_style

EXPORTS_DIR = Path(__file__).resolve().parent / "exports"
EXPORTS_DIR.mkdir(exist_ok=True)
ARTIFACT_DIR = PROJECT_ROOT / "outputs" / "comparative" / "diagnostic_models"

# Colors locked to Fig 0.4 diagnostic palette.
COLOR_VENT = "#0072B2"
COLOR_SUN = "#E69F00"
COLOR_NS = "#cccccc"          # not significant marker fill
COLOR_DIVIDER = "#999999"
WHO_ACH = 0.5
THRESHOLD_U_VENT = 1.0
ALPHA_ACH = 1.0 / 150.0
ACH_PER_U = 366.0

PRETTY = {
    "svf": "SVF", "lambda_p": r"$\lambda_p$",
    "lambda_f_mean": r"$\lambda_f$",
    "sigma_h": r"$\sigma_H$",
    "street_orientation_entropy": "entropy",
    "slope_deg": "slope (°)",
    "northness": "northness", "eastness": "eastness",
}
TARGET_LABEL = {"vent_fail": "Ventilation constraint",
                "sun_fail": "Sunlight constraint"}
TARGET_COLOR = {"vent_fail": COLOR_VENT, "sun_fail": COLOR_SUN}


# ---------------------------------------------------------------------------
# Panel A — RF permutation importance
# ---------------------------------------------------------------------------
def draw_panel_a(ax, rf: pd.DataFrame) -> None:
    # Sort predictors by max(vent, sun) importance, descending.
    pivot = rf.pivot(index="predictor", columns="target",
                     values="importance_mean").fillna(0.0)
    err = rf.pivot(index="predictor", columns="target",
                   values="importance_std").fillna(0.0)
    pivot["maxv"] = pivot.max(axis=1)
    pivot = pivot.sort_values("maxv", ascending=True)  # bottom to top
    err = err.reindex(pivot.index)

    n = len(pivot)
    y = np.arange(n)
    h = 0.36

    ax.barh(y - h / 2, pivot["vent_fail"], xerr=err["vent_fail"],
            height=h, color=COLOR_VENT, edgecolor="none",
            error_kw=dict(lw=0.4, ecolor="#444"), label="Ventilation")
    ax.barh(y + h / 2, pivot["sun_fail"], xerr=err["sun_fail"],
            height=h, color=COLOR_SUN, edgecolor="none",
            error_kw=dict(lw=0.4, ecolor="#444"), label="Sunlight")

    ax.set_yticks(y)
    ax.set_yticklabels([PRETTY[p] for p in pivot.index], fontsize=6.5)
    ax.tick_params(axis="x", labelsize=6, length=2, width=0.4, pad=2)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Permutation importance Δ AUC", fontsize=7, labelpad=2)
    ax.axvline(0, color="#999", lw=0.4, zorder=0)

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.4)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#eeeeee", linewidth=0.4, zorder=0)

    # AUC annotations.
    auc_v = rf[rf.target == "vent_fail"]["auc_cv5"].iloc[0]
    auc_s = rf[rf.target == "sun_fail"]["auc_cv5"].iloc[0]
    ax.text(0.99, 0.04,
            f"5-fold AUC\nvent {auc_v:.2f} | sun {auc_s:.2f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=5.5, color="#333",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                      edgecolor="#dddddd", linewidth=0.3, alpha=0.95))

    leg_handles = [
        Patch(facecolor=COLOR_VENT, edgecolor="none", label="Ventilation"),
        Patch(facecolor=COLOR_SUN, edgecolor="none", label="Sunlight"),
    ]
    ax.legend(handles=leg_handles, loc="lower right",
              bbox_to_anchor=(0.99, 0.20),
              frameon=False, fontsize=6, handlelength=1.0,
              labelspacing=0.4, borderpad=0.0)


# ---------------------------------------------------------------------------
# Panel B — Partial dependence small-multiples
# ---------------------------------------------------------------------------
def draw_panel_b(fig, gs, pdp: pd.DataFrame) -> None:
    targets = ["vent_fail", "sun_fail"]
    preds = ["svf", "lambda_f_mean", "slope_deg"]
    sub = gs.subgridspec(2, 3, hspace=0.55, wspace=0.30)

    # Common y-limits within each row for visual comparability.
    ymax_per_target = {}
    for t in targets:
        sub_t = pdp[pdp.target == t]
        ymax_per_target[t] = max(0.05, float(sub_t["hi"].max()) * 1.10)

    for r, tgt in enumerate(targets):
        for c, pred in enumerate(preds):
            ax = fig.add_subplot(sub[r, c])
            d = pdp[(pdp.target == tgt) & (pdp.predictor == pred)] \
                .sort_values("x")
            color = TARGET_COLOR[tgt]
            ax.fill_between(d.x, d.lo, d.hi, color=color, alpha=0.22,
                            linewidth=0, zorder=2)
            ax.plot(d.x, d.y, color=color, lw=1.1, zorder=3)
            ax.set_ylim(0, ymax_per_target[tgt])
            ax.tick_params(labelsize=5.5, length=2, width=0.3, pad=1)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_linewidth(0.4)
            if r == 1:
                ax.set_xlabel(PRETTY[pred], fontsize=6.5, labelpad=2)
            if c == 0:
                ylab = ("P(vent constraint)" if tgt == "vent_fail"
                        else "P(sun deprivation)")
                ax.set_ylabel(ylab, fontsize=6.5, labelpad=2,
                              color=color)


# ---------------------------------------------------------------------------
# Panel C — logistic forest plot, vent + sun stacked sub-rows
# ---------------------------------------------------------------------------
def draw_panel_c(fig, gs, logit: pd.DataFrame) -> None:
    sub = gs.subgridspec(2, 1, hspace=0.18, height_ratios=[1, 1])

    # Order: main effects (alpha order) then interactions.
    main_terms = sorted(logit[logit.kind == "main"]["term"].unique())
    ix_terms = sorted(logit[logit.kind == "interaction"]["term"].unique())
    all_terms = main_terms + ix_terms
    n = len(all_terms)

    def _term_pretty(t: str) -> str:
        if ":" in t:
            a, b = t.split(":")
            return f"{PRETTY.get(a, a)} × {PRETTY.get(b, b)}"
        return PRETTY.get(t, t)

    for i, tgt in enumerate(["vent_fail", "sun_fail"]):
        ax = fig.add_subplot(sub[i, 0])
        ld = logit[logit.target == tgt].set_index("term")

        y = np.arange(n)[::-1]
        for j, term in enumerate(all_terms):
            if term not in ld.index:
                continue
            row = ld.loc[term]
            color = TARGET_COLOR[tgt]
            face = color if row["p"] < 0.05 else "white"
            ax.plot([row["ci_lo"], row["ci_hi"]], [y[j], y[j]],
                    color=color, lw=0.8, zorder=2)
            ax.scatter(row["beta"], y[j], s=14, facecolor=face,
                       edgecolor=color, linewidth=0.9, zorder=3)

        # Divider between main and interaction blocks.
        sep = y[len(main_terms) - 1] - 0.5
        ax.axhline(sep, color=COLOR_DIVIDER, lw=0.4, linestyle=":", zorder=1)
        ax.axvline(0, color="#666", lw=0.5, zorder=1)

        ax.set_yticks(y)
        ax.set_yticklabels([_term_pretty(t) for t in all_terms], fontsize=5.5)
        ax.tick_params(axis="x", labelsize=5.5, length=2, width=0.3, pad=1)
        ax.tick_params(axis="y", length=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_linewidth(0.4)
        ax.grid(axis="x", color="#eeeeee", linewidth=0.4, zorder=0)
        ax.set_axisbelow(True)

        ax.text(0.99, 0.96, TARGET_LABEL[tgt],
                transform=ax.transAxes, ha="right", va="top",
                fontsize=6, color=TARGET_COLOR[tgt], fontweight="bold")

        if i == 1:
            ax.set_xlabel("Standardized log-odds β  (95% CI)",
                          fontsize=6.5, labelpad=2)

    # Sig legend below the second sub-row.
    handles = [
        Line2D([0], [0], marker="o", color="white",
               markerfacecolor="#888", markeredgecolor="#888",
               markersize=4, label="p < 0.05"),
        Line2D([0], [0], marker="o", color="white",
               markerfacecolor="white", markeredgecolor="#888",
               markersize=4, label="ns"),
    ]
    fig.axes[-1].legend(handles=handles, loc="lower right",
                        bbox_to_anchor=(0.99, -0.55),
                        frameon=False, fontsize=5.5,
                        handlelength=0.6, labelspacing=0.3, ncol=2,
                        borderpad=0.0, columnspacing=0.6)


# ---------------------------------------------------------------------------
# Panel D — SVF → U_mean changepoint scatter
# ---------------------------------------------------------------------------
def draw_panel_d(ax, cp: pd.DataFrame) -> None:
    fit_meta = cp[cp.patch_id != "__fit__"].iloc[0]
    bp = float(fit_meta["bp_hat"])
    bp_lo = float(fit_meta["bp_lo"])
    bp_hi = float(fit_meta["bp_hi"])
    a0 = float(fit_meta["a0"])
    a1 = float(fit_meta["a1"])
    a2 = float(fit_meta["a2"])

    pts = cp[cp.patch_id != "__fit__"]
    ax.scatter(pts.svf, pts.annual_U_mean, s=10, c="#888888", alpha=0.7,
               edgecolor="white", linewidth=0.3, zorder=2)

    # Segmented fit line.
    xs = np.linspace(pts.svf.min(), pts.svf.max(), 200)
    ys = a0 + a1 * xs + a2 * np.maximum(0, xs - bp)
    ax.plot(xs, ys, color="#1a1a1a", lw=1.4, zorder=3)

    # Breakpoint vertical line + CI band.
    ax.axvspan(bp_lo, bp_hi, color="#dddddd", alpha=0.6, zorder=1)
    ax.axvline(bp, color="#333", linestyle=(0, (3, 2)), lw=0.8, zorder=4)

    # WHO 0.5 ACH on a twin top-axis (indoor-equivalent ACH).
    sec = ax.secondary_yaxis(
        "right",
        functions=(
            lambda u: u * ACH_PER_U * ALPHA_ACH,
            lambda a: a / (ACH_PER_U * ALPHA_ACH),
        ),
    )
    sec.set_ylabel(r"Indoor-equivalent ACH ($\alpha = 1/150$)",
                   fontsize=6, labelpad=2)
    sec.tick_params(labelsize=5.5, length=2, width=0.3, pad=1)
    # WHO 0.5 ACH reference. Under α=1/150, WHO 0.5 ACH ↔ U≈0.21 m/s,
    # which sits *below* the patch-data y-range. We therefore draw a
    # downward callout with an arrow off the bottom edge.
    u_who = WHO_ACH / (ACH_PER_U * ALPHA_ACH)  # ≈ 0.205 m/s
    ax.annotate(
        f"WHO 0.5 ACH ≡ U ≈ {u_who:.2f} m/s\n(off-scale below patch range)",
        xy=(ax.get_xlim()[1] * 0.55, ax.get_ylim()[0]),
        xytext=(ax.get_xlim()[1] * 0.55,
                ax.get_ylim()[0] + 0.18 * (ax.get_ylim()[1] - ax.get_ylim()[0])),
        fontsize=5.0, color="#888", style="italic",
        ha="center", va="bottom",
        arrowprops=dict(arrowstyle="->", color="#aaaaaa", lw=0.4,
                        shrinkA=2, shrinkB=0),
    )

    ax.set_xlabel("SVF (patch)", fontsize=7, labelpad=2)
    ax.set_ylabel("Annual mean U @ 1.5 m (m/s)", fontsize=7, labelpad=2)
    ax.tick_params(labelsize=6, length=2, width=0.4, pad=2)
    for s in ("top",):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom", "right"):
        ax.spines[s].set_linewidth(0.4)
    ax.grid(color="#eeeeee", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)

    # Breakpoint annotation — anchored to the breakpoint line itself.
    y_low = ax.get_ylim()[0]
    y_high = ax.get_ylim()[1]
    annot_y = y_low + 0.78 * (y_high - y_low)
    text_x = min(bp + 0.18, ax.get_xlim()[1] - 0.22)
    ax.annotate(
        f"breakpoint SVF = {bp:.2f}\n"
        f"95% CI [{bp_lo:.2f}, {bp_hi:.2f}]",
        xy=(bp, annot_y),
        xytext=(text_x, annot_y),
        fontsize=5.5, color="#1a1a1a", fontweight="bold",
        ha="left", va="center",
        arrowprops=dict(arrowstyle="->", color="#444", lw=0.5,
                        shrinkA=0, shrinkB=2),
    )

    ax.text(0.02, 0.96, f"n = {len(pts)} CFD patches",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=5.5, color="#444",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                      edgecolor="#dddddd", linewidth=0.3, alpha=0.95))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    apply_style()
    warnings.filterwarnings("ignore")

    rf = pd.read_csv(ARTIFACT_DIR / "rf_importance.csv")
    pdp = pd.read_csv(ARTIFACT_DIR / "pdp_curves.csv")
    logit = pd.read_csv(ARTIFACT_DIR / "logit_coefs.csv")
    cp = pd.read_csv(ARTIFACT_DIR / "changepoint_svf_ach.csv")

    fig = plt.figure(figsize=(WIDTH_DOUBLE, WIDTH_DOUBLE * 1.05))
    outer = gridspec.GridSpec(
        2, 2, figure=fig,
        height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.0],
        hspace=0.42, wspace=0.30,
        left=0.10, right=0.95, top=0.92, bottom=0.07,
    )

    print("Drawing panel A (RF importance) ...")
    ax_a = fig.add_subplot(outer[0, 0])
    draw_panel_a(ax_a, rf)

    print("Drawing panel B (PDP) ...")
    draw_panel_b(fig, outer[0, 1], pdp)

    print("Drawing panel C (logistic forest) ...")
    draw_panel_c(fig, outer[1, 0], logit)

    print("Drawing panel D (changepoint) ...")
    ax_d = fig.add_subplot(outer[1, 1])
    draw_panel_d(ax_d, cp)

    # Panel labels.
    panel_pos = {
        "a": ax_a.get_position(),
        "b": (outer[0, 1].get_position(fig)),
        "c": (outer[1, 0].get_position(fig)),
        "d": ax_d.get_position(),
    }
    for letter, pos in panel_pos.items():
        x = pos.x0 - 0.045 if letter in ("a", "c") else pos.x0 - 0.025
        y = pos.y1 + 0.012
        fig.text(x, y, letter, fontsize=10, fontweight="bold",
                 va="bottom", ha="left", color="#1a1a1a")

    # Headline title.
    fig.text(0.5, 0.97,
             "Predictors and typology contrast: which morphometrics drive constraint?",
             ha="center", va="top", fontsize=8.5, fontweight="bold",
             color="#1a1a1a")

    # Caption strip — synthetic-CFD caveat.
    fig.text(0.5, 0.026,
             "Models trained on synthetic CFD; framework rotates to "
             "real-campaign CFD with a one-line config swap.",
             ha="center", va="bottom", fontsize=5.5, style="italic",
             color="#666")

    out_png = EXPORTS_DIR / "fig_0_5_predictors.png"
    out_svg = EXPORTS_DIR / "fig_0_5_predictors.svg"
    print(f"Saving {out_png.name} + {out_svg.name} ...")
    fig.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.05,
                facecolor="white")
    fig.savefig(out_svg, format="svg", bbox_inches="tight", pad_inches=0.05,
                facecolor="white")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
