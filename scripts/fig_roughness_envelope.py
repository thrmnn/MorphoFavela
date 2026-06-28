#!/usr/bin/env python3
"""E1 — aerodynamic-roughness inlet-BC method-spread envelope (2026-06-28).

The four morphometric z0 methods (Macdonald 1998, Raupach 1994, Millward-Hopkins
2011, Kanda 2013) disagree enormously in the λp > 0.5 favela regime — none was
calibrated there. The pre-existing bar figure
(``outputs/cross_site/signature/figures_v2/roughness_methods.png``) uses a LINEAR
axis whose title rounds the disagreement to "~20×"; that hides the real structure —
the spread is 4× in the steep, height-variable hillside fabric but blows out to
148× in the flat, saturated Rio das Pedras fabric, where Macdonald's σH-blind z0
collapses toward zero while Raupach stays high.

This sibling reads the **already-computed** per-method site medians
(``outputs/cross_site/roughness/method_medians.csv``) — it recomputes nothing — and
draws, on a LOG axis, the Macdonald↔Raupach **inlet-BC uncertainty envelope** per
site with the per-site spread factor annotated. The envelope, not any single
method's z0, is the reportable pre-CFD result; the absolute value is CFD-gated.

Run:
    python scripts/fig_roughness_envelope.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "outputs" / "paper_figures"))
from fig_style import (  # noqa: E402
    SITE_LABELS,
    WIDTH_DOUBLE,
    add_provenance,
    apply_style,
    save_fig,
)

CSV = PROJECT_ROOT / "outputs" / "cross_site" / "roughness" / "method_medians.csv"
SITES = ["vidigal", "rocinha", "complexo_do_alemao", "riodaspedras", "maré"]
METHOD_STYLE = {  # marker + colour per method; Kanda primary
    "Kan": ("o", "#1f4e79", "Kanda 2013 (primary, σH-aware)"),
    "Mho": ("s", "#E69F00", "Millward-Hopkins 2011"),
    "Mac": ("v", "#888888", "Macdonald 1998 (σH-blind baseline)"),
    "Rau": ("^", "#56B4E9", "Raupach 1994"),
}


def main() -> None:
    df = pd.read_csv(CSV, index_col=0).loc[SITES]
    apply_style()
    fig, ax = plt.subplots(figsize=(WIDTH_DOUBLE, 3.4))
    x = np.arange(len(SITES))

    lo = df[["Kan", "Mho", "Mac", "Rau"]].min(axis=1).to_numpy()
    hi = df[["Kan", "Mho", "Mac", "Rau"]].max(axis=1).to_numpy()
    # the inlet-BC envelope band (min..max method) per site
    ax.vlines(x, lo, hi, color="#B2182B", lw=6, alpha=0.18, zorder=1)
    for m, (mk, col, _lab) in METHOD_STYLE.items():
        ax.scatter(x, df[m].to_numpy(), marker=mk, s=34, color=col,
                   edgecolor="white", linewidth=0.4, zorder=3, label=METHOD_STYLE[m][2])
    for xi, l, h in zip(x, lo, hi):
        ax.annotate(f"{h / l:.0f}×", (xi, h), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=6.2, color="#B2182B")

    ax.set_yscale("log")
    ax.set_ylim(3e-3, 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels([SITE_LABELS[s] for s in SITES], fontsize=7)
    ax.set_ylabel("median z$_0$ (m, log)", fontsize=7.5)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.legend(fontsize=5.8, frameon=False, loc="lower left", ncol=2,
              handletextpad=0.3, columnspacing=1.0)
    ax.set_title(
        "Inlet-BC roughness envelope — four morphometric z$_0$ methods disagree "
        "4×–148× in the λ$_p$>0.5 favela regime\n(Macdonald↔Raupach band; widest "
        "where fabric is flat-saturated and most out-of-envelope; absolute z$_0$ "
        "CFD-gated)", fontsize=7.2)
    add_provenance(fig)
    save_fig(fig, "roughness_envelope", gate=True)


if __name__ == "__main__":
    main()
