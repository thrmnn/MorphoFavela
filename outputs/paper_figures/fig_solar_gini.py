#!/usr/bin/env python3
"""Extended Data — inequality of winter direct-sun access (full Lorenz + Gini).

The standalone ED companion to the compact inset on Fig 2 (fig_solar_deficit).
Uses the SAME built-cell grid solar source (matching the canonical 46% marginal
and the Fig 2 inset) so main and ED agree. Within-fabric inequality (RQ2);
panels ordered by typology, never by Gini value (a Gini-sorted layout reads as
a settlement league table).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fig03_ventilation_solar import load_grid, SITE_LABELS  # noqa: E402
from fig_style import apply_style, save_fig  # noqa: E402

ORDER = ["vidigal", "rocinha", "complexo_do_alemao", "riodaspedras", "maré"]
PAL = {"vidigal": "#4477AA", "rocinha": "#EE6677", "complexo_do_alemao": "#228833",
       "riodaspedras": "#AA3377", "maré": "#CCBB44"}


def _hours(site):
    a = load_grid(site)["solar_hours_winter"].to_numpy(float)
    return a[np.isfinite(a)]


def gini(x):
    x = np.sort(x); n = x.size
    return float((2 * np.sum(np.arange(1, n + 1) * x)) / (n * np.sum(x)) - (n + 1) / n)


def lorenz(x):
    x = np.sort(x); c = np.insert(np.cumsum(x), 0, 0)
    return np.linspace(0, 1, c.size), c / c[-1]


def main():
    apply_style()
    hrs = {s: _hours(s) for s in ORDER}
    pooled = np.concatenate([hrs[s] for s in ORDER])
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    ax.plot([0, 1], [0, 1], ls="--", color="#AAAAAA", lw=1.0, label="perfect equality")
    for s in ORDER:
        p, c = lorenz(hrs[s])
        ax.plot(p, c, color=PAL[s], lw=2.0, label=f"{SITE_LABELS[s]} (G={gini(hrs[s]):.2f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("cumulative share of built street space (poorest→richest sun)", fontsize=8)
    ax.set_ylabel("cumulative share of winter direct-sun hours", fontsize=8)
    ax.set_title(f"Inequality of winter direct-sun access within favela fabric\n"
                 f"(within-fabric; sites not ranked · pooled G={gini(pooled):.2f})", fontsize=9)
    ax.legend(loc="upper left", fontsize=7.5, frameon=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    save_fig(fig, "fig_solar_gini")
    print(f"saved fig_solar_gini  (pooled G={gini(pooled):.3f})")


if __name__ == "__main__":
    main()
