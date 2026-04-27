"""Figure S5 — wind roses for the 5 campaign sites.

Each panel is a polar bar chart of measured directional frequency
(8 cardinal bins) coloured by mean wind speed within the bin. The
text annotation reports station, n_obs, time window, and calm
fraction. Data are read from data/{site}/wind_rose.json which is
built by scripts/build_wind_rose.py from INMET BDMEP CSVs (4 sites)
or the Iowa State ASOS METAR archive for SBGL/Maré.

Usage:
    python3 outputs/paper_figures/figS5_wind_roses.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import (  # noqa: E402
    DPI,
    PROJECT_ROOT,
    SITE_LABELS,
    SITE_ORDER,
    SIZE_DOUBLE,
    apply_style,
    save_fig,
)

# 8 cardinal bin centres (deg, met. convention: 0=N, 90=E)
DIR_LABELS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
DIR_DEG = np.arange(0, 360, 45.0)


def _polar(ax, freq: dict, speeds: dict, vmax_speed: float):
    """Draw one polar bar chart on `ax`."""
    angles = np.deg2rad(DIR_DEG)
    width = np.deg2rad(45.0)
    f = np.array([freq[d] for d in DIR_LABELS]) * 100  # %
    u = np.array([speeds[d] for d in DIR_LABELS])

    norm = colors.Normalize(vmin=0, vmax=vmax_speed)
    cmap = plt.get_cmap("viridis")
    bar_colors = [cmap(norm(v)) for v in u]

    ax.bar(
        angles,
        f,
        width=width * 0.92,
        bottom=0.0,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.4,
        align="center",
        zorder=3,
    )
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)  # clockwise (compass)
    ax.set_xticks(angles)
    ax.set_xticklabels(DIR_LABELS, fontsize=6)
    ax.set_yticklabels([])
    ax.set_ylim(0, max(f.max() * 1.15, 5.0))
    ax.tick_params(axis="x", pad=-2)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    return norm, cmap


def main() -> None:
    apply_style()

    # Load all 5 roses
    roses = {}
    for site in SITE_ORDER:
        p = PROJECT_ROOT / "data" / site / "wind_rose.json"
        if not p.exists():
            print(f"  skipping {site}: no wind_rose.json")
            continue
        roses[site] = json.loads(p.read_text())

    if not roses:
        raise FileNotFoundError("No wind_rose.json files under data/{site}/")

    # Common speed scale across panels
    vmax_speed = max(max(r["mean_speeds"].values()) for r in roses.values())

    fig = plt.figure(figsize=(SIZE_DOUBLE[0], SIZE_DOUBLE[0] * 0.45))
    n = len(roses)
    axes = [fig.add_subplot(1, n, i + 1, projection="polar") for i in range(n)]

    norm = cmap = None
    for ax, (site, rose) in zip(axes, roses.items()):
        norm, cmap = _polar(ax, rose["frequencies"], rose["mean_speeds"], vmax_speed)
        title = SITE_LABELS.get(site, site)
        ax.set_title(title, fontsize=8, pad=6)
        # Footnote with provenance
        n_obs = rose.get("n_observations", 0)
        calm = rose.get("calm_fraction", 0.0) or 0.0
        tw_a = (rose.get("time_window_start") or "?")[:4]
        tw_b = (rose.get("time_window_end") or "?")[:4]
        sid = rose.get("station_id", "?")
        ax.text(
            0.5,
            -0.18,
            f"{sid}  {tw_a}–{tw_b}\nn={n_obs:,}  calm={calm:.1%}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=5.2,
            color="#444",
        )

    # Shared colorbar
    cbar_ax = fig.add_axes([0.30, 0.08, 0.40, 0.025])
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cb.set_label("Mean wind speed within direction bin (m/s)", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    cb.outline.set_linewidth(0.4)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.22, wspace=0.55)
    save_fig(fig, "figS5_wind_roses")


if __name__ == "__main__":
    main()
