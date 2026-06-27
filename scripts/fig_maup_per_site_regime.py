#!/usr/bin/env python3
"""Per-site MAUP regime composition — Fig S3 companion to the pooled A/B.

The pooled figure (``outputs/comparative/maup/maup_regime_shares.png``, drawn by
``scripts/run_maup_sensitivity.py``) collapses all five sites into one bar group.
That hides whether the +18.4 pp skimming inflation under cell-size doubling is a
uniform artefact or driven by particular morphologies. This sibling reads the
**already-computed** A/B record (``maup_sensitivity.json``) — it does not touch
any grid, recompute anything, or alter a canonical output — and draws the
per-site isolated/wake/skimming composition at 10 m vs 20 m as paired stacked
bars, one pair per site.

Run:
    conda run -n IVF python scripts/fig_maup_per_site_regime.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "outputs" / "paper_figures"))
from fig_style import (  # noqa: E402
    SITE_ORDER,
    WIDTH_DOUBLE,
    apply_style,
    check_text_overflow,
)

MAUP_DIR = PROJECT_ROOT / "outputs" / "comparative" / "maup"
JSON_PATH = MAUP_DIR / "maup_sensitivity.json"
OUT_PNG = MAUP_DIR / "maup_per_site_regime.png"

REGIMES = ["isolated", "wake", "skimming"]
REGIME_COLORS = {
    "isolated": "#44AA99",   # teal — open, ventilated
    "wake": "#DDCC77",       # sand — transitional
    "skimming": "#CC6677",   # rose — sheltered / stagnant
}
REGIME_LABELS = {"isolated": "Isolated", "wake": "Wake", "skimming": "Skimming"}


def main() -> None:
    record = json.loads(JSON_PATH.read_text())
    ab = record["ab"]
    per_site = ab["per_site"]
    sites = [s for s in SITE_ORDER if s in per_site]

    apply_style()
    fig, ax = plt.subplots(figsize=(WIDTH_DOUBLE, WIDTH_DOUBLE * 0.50))

    bar_w = 0.36
    gap = 0.06
    centers = np.arange(len(sites))
    x10 = centers - (bar_w / 2 + gap / 2)
    x20 = centers + (bar_w / 2 + gap / 2)

    for res_key, xs in (("10m", x10), ("20m", x20)):
        bottoms = np.zeros(len(sites))
        for reg in REGIMES:
            vals = np.array(
                [per_site[s][res_key]["regime_shares"][reg] * 100 for s in sites]
            )
            ax.bar(
                xs,
                vals,
                bar_w,
                bottom=bottoms,
                color=REGIME_COLORS[reg],
                edgecolor="white",
                linewidth=0.4,
                label=REGIME_LABELS[reg] if res_key == "10m" else None,
            )
            bottoms += vals

    # Resolution tick under each bar (10 / 20), site label centred under the pair.
    sub_ticks = np.concatenate([x10, x20])
    sub_labels = ["10"] * len(sites) + ["20"] * len(sites)
    ax.set_xticks(sub_ticks)
    ax.set_xticklabels(sub_labels, fontsize=5)
    ax.tick_params(axis="x", length=0, pad=1)

    site_labels = [per_site[s]["label"] for s in sites]
    for c, lab in zip(centers, site_labels):
        ax.text(c, -9.5, lab, ha="center", va="top", fontsize=6.5, fontweight="bold")

    ax.set_ylim(0, 100)
    ax.set_xlim(centers[0] - 0.6, centers[-1] + 0.6)
    ax.set_ylabel("Built-cell share (%)")
    ax.set_title(
        "Per-site flow-regime composition under cell-size doubling "
        "(10 m vs 20 m, built cells)",
        fontsize=8,
    )
    ax.text(
        0.0, 1.005, "Cell size (m):", transform=ax.transAxes,
        ha="left", va="bottom", fontsize=5, color="#555",
    )
    ax.legend(
        ncol=3, frameon=False, fontsize=6, loc="lower center",
        bbox_to_anchor=(0.5, 1.06), handlelength=1.1, columnspacing=1.4,
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(bottom=0.16, top=0.86, left=0.08, right=0.98)

    bad = check_text_overflow(fig)
    if bad:
        raise ValueError(f"text-overflow gate failed: {bad}")

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
