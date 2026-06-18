"""Cross-site SVF–solar decoupling.

Materialises the per-aspect winter-direct-sun-hours story across all five
campaign sites from the existing ``aspect_quadrant_summary.csv`` products
written by ``scripts/run_aspect_analysis.py``. Quadrant assignment uses
the 5° slope filter from that pipeline; ``solar_hours`` in the source
files is verified equal to ``solar_hours_winter`` in
``svf_streets_solar.gpkg``.

Deliverables
------------
- ``outputs/brisa_ventilation_fix/solar_decoupling_cross_site.csv``
  long table with one row per (site, quadrant, metric)
- ``outputs/brisa_ventilation_fix/solar_decoupling_gap.csv``
  per-site N–S winter-sun gap + SVF aspect contrast
- ``outputs/brisa_ventilation_fix/fig_solar_decoupling.png``
  six-panel cross-site figure: top row = street-SVF by quadrant,
  bottom row = winter-direct-sun-hours by quadrant. One column per site.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

SITES = [
    ("vidigal", "Vidigal", "hillside"),
    ("rocinha", "Rocinha", "hillside"),
    ("complexo_do_alemao", "Complexo do Alemão", "hillside"),
    ("riodaspedras", "Rio das Pedras", "lowland"),
    ("maré", "Maré", "lowland"),
]

QUADRANTS = ("N", "E", "S", "W")
QUAD_COLORS = {"N": "#3a6ea5", "E": "#6aa84f", "S": "#cc4125", "W": "#e69138"}

OUT_DIR = _ROOT / "outputs" / "brisa_ventilation_fix"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_site(site: str) -> pd.DataFrame:
    p = _ROOT / "outputs" / site / "morphometrics" / "aspect_quadrant_summary.csv"
    return pd.read_csv(p)


def build_long_table() -> pd.DataFrame:
    rows = []
    for site, label, terrain in SITES:
        df = load_site(site)
        for _, r in df.iterrows():
            rows.append(
                {
                    "site": site,
                    "label": label,
                    "terrain": terrain,
                    "source": r["source"],
                    "metric": r["metric"],
                    "quadrant": r["quadrant"],
                    "n_cells": int(r["n_cells"]),
                    "mean": float(r["mean"]),
                    "std": float(r["std"]),
                }
            )
    return pd.DataFrame(rows)


def gap_table(long: pd.DataFrame) -> pd.DataFrame:
    """Per-site N–S winter-sun gap (h), plus SVF aspect contrast."""
    rows = []
    for site, label, terrain in SITES:
        sol = long[(long.site == site) & (long.metric == "solar_hours")].set_index("quadrant")
        svf = long[(long.site == site) & (long.metric == "svf")].set_index("quadrant")
        if sol.empty or svf.empty:
            continue
        winter_n = float(sol.loc["N", "mean"]) if "N" in sol.index else np.nan
        winter_s = float(sol.loc["S", "mean"]) if "S" in sol.index else np.nan
        svf_n = float(svf.loc["N", "mean"]) if "N" in svf.index else np.nan
        svf_s = float(svf.loc["S", "mean"]) if "S" in svf.index else np.nan
        svf_max_q = svf["mean"].idxmax() if not svf.empty else None
        sol_max_q = sol["mean"].idxmax() if not sol.empty else None
        sol_min_q = sol["mean"].idxmin() if not sol.empty else None
        rows.append(
            {
                "site": site,
                "label": label,
                "terrain": terrain,
                "winter_sun_N_h": winter_n,
                "winter_sun_S_h": winter_s,
                "winter_sun_N_minus_S_h": winter_n - winter_s,
                "svf_N": svf_n,
                "svf_S": svf_s,
                "svf_S_minus_N": svf_s - svf_n,
                "svf_aspect_argmax": svf_max_q,
                "winter_sun_aspect_argmax": sol_max_q,
                "winter_sun_aspect_argmin": sol_min_q,
                "decoupled": (svf_max_q != sol_max_q)
                or (sol_min_q == "S" and svf_max_q in ("S", "W")),
            }
        )
    return pd.DataFrame(rows)


def plot_cross_site(long: pd.DataFrame, out_path: Path) -> None:
    n = len(SITES)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 5.6), sharey="row")
    if n == 1:
        axes = np.array(axes).reshape(2, 1)

    for col, (site, label, terrain) in enumerate(SITES):
        for row_idx, metric, ylabel, ylim in (
            (0, "svf", "Street SVF (mean)", (0, 0.75)),
            (1, "solar_hours", "Winter direct sun (h)", (0, 7.0)),
        ):
            ax = axes[row_idx, col]
            sub = long[(long.site == site) & (long.metric == metric)].set_index("quadrant")
            sub = sub.reindex(QUADRANTS)
            means = sub["mean"].values.astype(float)
            stds = sub["std"].values.astype(float)
            xs = np.arange(len(QUADRANTS))
            colors = [QUAD_COLORS[q] for q in QUADRANTS]
            ax.bar(xs, means, yerr=stds, color=colors, alpha=0.85,
                   edgecolor="black", linewidth=0.4, capsize=3.0,
                   error_kw={"ecolor": "0.4", "elinewidth": 0.7})
            ax.set_xticks(xs)
            ax.set_xticklabels(list(QUADRANTS))
            ax.set_ylim(*ylim)
            ax.grid(axis="y", color="0.92", lw=0.6)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row_idx == 0:
                ax.set_title(f"{label}\n({terrain})", fontsize=10)
            if row_idx == 1 and "N" in sub.index and "S" in sub.index:
                gap = means[0] - means[2]
                ax.text(0.97, 0.95, f"N−S = {gap:+.2f} h",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=8.5,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6", lw=0.5))

    fig.suptitle(
        "SVF–winter-sun decoupling across Rio favelas\n"
        "Street SVF (top) vs winter direct-sun hours (bottom) by terrain-aspect quadrant",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    long = build_long_table()
    long_path = OUT_DIR / "solar_decoupling_cross_site.csv"
    long.to_csv(long_path, index=False)
    print(f"Wrote {long_path}  ({len(long)} rows)")

    gaps = gap_table(long)
    gap_path = OUT_DIR / "solar_decoupling_gap.csv"
    gaps.to_csv(gap_path, index=False)
    print(f"Wrote {gap_path}")
    print(gaps.to_string(index=False))

    fig_path = OUT_DIR / "fig_solar_decoupling.png"
    plot_cross_site(long, fig_path)
    print(f"Wrote {fig_path}")

    summary = {
        "source": "outputs/<site>/morphometrics/aspect_quadrant_summary.csv "
        "(re-run via scripts/run_aspect_analysis.py --all-sites)",
        "solar_hours_column": "solar_hours == solar_hours_winter (verified in svf_streets_solar.gpkg)",
        "slope_filter_deg": 5.0,
        "sites": [
            {
                "site": r.site,
                "label": r.label,
                "terrain": r.terrain,
                "N_minus_S_winter_h": float(r.winter_sun_N_minus_S_h),
                "svf_argmax": r.svf_aspect_argmax,
                "winter_sun_argmax": r.winter_sun_aspect_argmax,
                "winter_sun_argmin": r.winter_sun_aspect_argmin,
                "decoupled": bool(r.decoupled),
            }
            for r in gaps.itertuples()
        ],
    }
    (OUT_DIR / "solar_decoupling_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
