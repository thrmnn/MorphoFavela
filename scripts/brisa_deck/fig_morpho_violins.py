#!/usr/bin/env python3
"""BRISA slide deck — fig_morpho_violins.png (violin variant of fig_morpho_boxplots).

Box plots collapse each distribution to 5 numbers and hide bimodality. Favela
morphometric distributions tend to be bimodal (SVF: low-canyon mode + high-ridge
mode in hillside favelas), so this variant uses violins to reveal shape.

Layout matches fig_morpho_boxplots.py exactly:
  * 3 panels left-to-right (λf · SVF · slope)
  * Each panel: 5 violins, hillside-first ordering
    (Vidigal, Rocinha, Alemão, Rio das Pedras, Maré)
  * Hillside violins: black outline + light grey fill
  * Flatland violins: black outline + white fill + hatch
  * MEDIAN as a bold white tick (visual anchor) — not the mean.
  * Q1/Q3 as thinner dotted ticks.
  * Typology legend top-right of figure.

Aspect ≈ 1.6. 200 DPI → 2800×1750 px (matches box-plot output exactly).

Output: /home/theo/brisa_paper/artifacts/slides/assets/fig_morpho_violins.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER_FIG_DIR = PROJECT_ROOT / "outputs" / "paper_figures"
sys.path.insert(0, str(PAPER_FIG_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
from fig_style import SITE_LABELS, apply_style  # noqa: E402
from src.viz import presentation_style as ps  # noqa: E402

PAPER_OUT_DIR = Path("/home/theo/brisa_paper/artifacts/slides/assets")
PRESENTATION_OUT_DIR = PROJECT_ROOT / "outputs" / "cross_site" / "presentation_figures"
OUT_FILENAME = "fig_morpho_violins.png"

SITES = ["vidigal", "rocinha", "complexo_do_alemao", "riodaspedras", "maré"]
HILLSIDE = {"vidigal", "rocinha", "complexo_do_alemao"}

STYLE = {
    "hillside": dict(facecolor="#E76F51", edgecolor="#7a2e1e", alpha=0.72),
    "flatland": dict(facecolor="#264653", edgecolor="#13262d", alpha=0.72),
}


def load_grid(site: str) -> gpd.GeoDataFrame:
    p = PROJECT_ROOT / "outputs" / site / "morphometrics" / "grid" / "grid_metrics.gpkg"
    grid = gpd.read_file(p)
    built = (grid["lambda_p"].fillna(0) > 0.01) | (grid["building_count"] > 0)
    return grid[built].copy()


def violin_panel(ax, data: dict, *, ylim, ylabel, fmt="{:.2f}") -> list[float]:
    positions = np.arange(len(SITES))
    arrays = []
    for s in SITES:
        v = np.asarray(data[s], dtype=float)
        v = v[(v >= ylim[0]) & (v <= ylim[1])]
        arrays.append(v)

    parts = ax.violinplot(
        arrays,
        positions=positions,
        widths=0.78,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for i, (s, body) in enumerate(zip(SITES, parts["bodies"])):
        typology = "hillside" if s in HILLSIDE else "flatland"
        st = STYLE[typology]
        body.set_facecolor(st["facecolor"])
        body.set_edgecolor(st["edgecolor"])
        body.set_linewidth(1.1)
        body.set_alpha(st["alpha"])

    medians: list[float] = []
    for i, vals in enumerate(arrays):
        if len(vals) == 0:
            medians.append(float("nan"))
            continue
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        medians.append(float(med))

        # Q1 / Q3 — subtle dotted ticks behind the median
        for q in (q1, q3):
            ax.hlines(
                q, positions[i] - 0.16, positions[i] + 0.16,
                colors="#1a1a1a", linewidth=0.9, linestyles=(0, (1.2, 1.2)),
                alpha=0.55, zorder=4,
            )

        # MEDIAN — thick white bar (visual anchor), no mean indicator
        ax.hlines(
            med, positions[i] - 0.30, positions[i] + 0.30,
            colors="#ffffff", linewidth=3.6, zorder=6,
        )

        # Median value label
        ax.text(
            positions[i], med, fmt.format(med),
            ha="center", va="bottom",
            fontsize=11, color="#111111", fontweight="semibold",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.6, alpha=0.92),
            zorder=7,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [SITE_LABELS[s] for s in SITES],
        rotation=25, ha="right", fontsize=14,
    )
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=15, labelpad=8)
    ax.tick_params(axis="y", labelsize=13)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.spines["left"].set_linewidth(0.6)
    ax.yaxis.grid(True, linestyle=":", color="#bbbbbb", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)

    return medians


def detect_bimodality(values: np.ndarray, *, value_range: tuple[float, float]) -> bool:
    """Crude bimodality check: KDE on a 200-pt grid, count local maxima above
    10% of the global peak. Returns True iff ≥2 such peaks."""
    if len(values) < 30:
        return False
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(values)
    xs = np.linspace(value_range[0], value_range[1], 200)
    ys = kde(xs)
    peak = ys.max()
    if peak <= 0:
        return False
    threshold = 0.10 * peak
    peaks = []
    for i in range(1, len(ys) - 1):
        if ys[i] > ys[i - 1] and ys[i] > ys[i + 1] and ys[i] >= threshold:
            peaks.append((xs[i], ys[i]))
    return len(peaks) >= 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--preset", choices=("paper", "presentation"), default="paper")
    args = parser.parse_args()

    apply_style()
    ps.apply(args.preset)
    grids = {s: load_grid(s) for s in SITES}

    lambdaf_data = {s: grids[s]["lambda_f_mean"].dropna().values for s in SITES}
    svf_data = {s: grids[s]["svf"].dropna().values for s in SITES}
    slope_data = {s: grids[s]["slope_deg"].dropna().values for s in SITES}

    fig_w = 14.0
    fig_h = fig_w / 1.6  # ~8.75 in
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h))

    medians_lambdaf = violin_panel(
        axes[0], lambdaf_data,
        ylim=(0.0, 8.0),
        ylabel=r"$\lambda_f$ — frontal-area density",
        fmt="{:.2f}",
    )
    medians_svf = violin_panel(
        axes[1], svf_data,
        ylim=(0.0, 1.0),
        ylabel="Sky View Factor",
        fmt="{:.2f}",
    )
    medians_slope = violin_panel(
        axes[2], slope_data,
        ylim=(0.0, 45.0),
        ylabel="Slope (degrees)",
        fmt="{:.1f}°",
    )

    handles = [
        Patch(facecolor=STYLE["hillside"]["facecolor"],
              edgecolor=STYLE["hillside"]["edgecolor"],
              alpha=STYLE["hillside"]["alpha"],
              label="Hillside"),
        Patch(facecolor=STYLE["flatland"]["facecolor"],
              edgecolor=STYLE["flatland"]["edgecolor"],
              alpha=STYLE["flatland"]["alpha"],
              label="Flatland"),
    ]
    leg = fig.legend(
        handles=handles, loc="upper right",
        bbox_to_anchor=(0.995, 0.985),
        frameon=True, fontsize=13,
        handlelength=1.4, handleheight=1.1,
        ncol=2, columnspacing=1.2,
    )
    leg.get_frame().set_linewidth(0.4)
    leg.get_frame().set_edgecolor("#888888")
    leg.get_frame().set_facecolor("white")

    fig.subplots_adjust(left=0.065, right=0.99, top=0.90, bottom=0.18, wspace=0.30)

    out_dir = PRESENTATION_OUT_DIR if args.preset == "presentation" else PAPER_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / OUT_FILENAME
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches=None)
    plt.close(fig)
    size_kb = out.stat().st_size // 1024
    print(f"  Saved {out} ({size_kb} KB)")

    # Report medians + bimodality detection
    print("\nMedians (panel order: λf, SVF, slope):")
    for s, ml, ms, msl in zip(SITES, medians_lambdaf, medians_svf, medians_slope):
        print(f"  {SITE_LABELS[s]:20s}  λf={ml:6.3f}   SVF={ms:5.3f}   slope={msl:5.2f}°")

    print("\nBimodality scan (≥2 KDE peaks above 10% of global peak):")
    for label, data, vrange in (
        ("λf", lambdaf_data, (0.0, 8.0)),
        ("SVF", svf_data, (0.0, 1.0)),
        ("slope", slope_data, (0.0, 45.0)),
    ):
        flags = []
        for s in SITES:
            vals = np.asarray(data[s], dtype=float)
            vals = vals[(vals >= vrange[0]) & (vals <= vrange[1])]
            if detect_bimodality(vals, value_range=vrange):
                flags.append(SITE_LABELS[s])
        if flags:
            print(f"  {label:6s}  bimodal at: {', '.join(flags)}")
        else:
            print(f"  {label:6s}  no clear bimodality")


if __name__ == "__main__":
    main()
