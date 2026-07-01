#!/usr/bin/env python3
"""Flagship Fig 2 — the delivered sunlight-adequacy deficit (+ within-fabric inequality).

Council split of the old combined fig03: this is the SOLAR-ONLY, CFD-independent
lead result. Reuses fig03's proven, stroke-free (occlusion-safe) choropleth code.
  Top   Per-site winter-solstice direct-sun choropleths (cividis, 0-10.5 h);
        colorbar marks the >=2 h reference floor.
  BL    Per-site winter direct-sun distribution (street points) with the >=2 h
        reference and per-site share below it (pooled 46%; 35-74% by site).
  BR    Within-fabric inequality inset: Lorenz curves + per-site Gini (0.41-0.77;
        RQ2). Ordered by typology (hillside->flatland), NOT by Gini value.

The >=2 h winter-sun floor is a GENERAL DAYLIGHT/HEALTH REFERENCE (exogenous,
illustrative) -- never attributed to WHO (WHO sets no winter-sun standard; that
attribution is reserved for the ACH ventilation guidance).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fig03_ventilation_solar import (  # noqa: E402  reuse proven components
    load_grid, choropleth_panel, add_vertical_colorbar,
    SITES, SITE_LABELS, SUN_CMAP, SUN_VMIN, SUN_VMAX, THRESHOLD_SUN_HRS, PROJECT_ROOT,
)
from fig_style import apply_style, save_fig  # noqa: E402

# hillside then flatland (typology order — NOT sorted by deficit/Gini)
ORDER = ["vidigal", "rocinha", "complexo_do_alemao", "riodaspedras", "maré"]
PAL = {"vidigal": "#4477AA", "rocinha": "#EE6677", "complexo_do_alemao": "#228833",
       "riodaspedras": "#AA3377", "maré": "#CCBB44"}


def street_hours(site: str) -> np.ndarray:
    g = gpd.read_file(PROJECT_ROOT / "outputs" / site / "morphometrics" / "svf" / "svf_streets_solar.gpkg")
    h = g["solar_hours_winter"].to_numpy(float)
    return h[np.isfinite(h)]


def gini(x: np.ndarray) -> float:
    x = np.sort(x); n = x.size
    return float((2 * np.sum(np.arange(1, n + 1) * x)) / (n * np.sum(x)) - (n + 1) / n)


def lorenz(x: np.ndarray):
    x = np.sort(x); c = np.insert(np.cumsum(x), 0, 0)
    return np.linspace(0, 1, c.size), c / c[-1]


def main() -> None:
    apply_style()
    grids = {s: load_grid(s) for s in ORDER}
    # Built-cell grid solar (SAME source as the maps + the canonical 46% taxonomy
    # marginal) — not the denser street points, so figure and manuscript agree.
    hours = {}
    for s in ORDER:
        a = grids[s]["solar_hours_winter"].to_numpy(float)
        hours[s] = a[np.isfinite(a)]
    pooled = np.concatenate([hours[s] for s in ORDER])
    dep_all = (pooled < THRESHOLD_SUN_HRS).mean() * 100

    fig = plt.figure(figsize=(7.5, 5.9))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.18], hspace=0.42, wspace=0.32,
                          left=0.075, right=0.92, top=0.9, bottom=0.11)

    # ── Top: winter-sun choropleths ───────────────────────────────────────────
    row = []
    for i, s in enumerate(ORDER):
        ax = fig.add_subplot(gs[0, i])
        choropleth_panel(ax, grids[s], "solar_hours_winter", SUN_CMAP, SUN_VMIN, SUN_VMAX,
                         title=f"{'(A) ' if i == 0 else ''}{SITE_LABELS[s]}")
        row.append(ax)
    fig.canvas.draw()
    add_vertical_colorbar(row[-1], SUN_CMAP, SUN_VMIN, SUN_VMAX,
                          [(THRESHOLD_SUN_HRS, "floor")], "winter direct-sun h")

    # ── Bottom-left: per-site distribution ────────────────────────────────────
    axB = fig.add_subplot(gs[1, 0:4])
    data = [hours[s] for s in ORDER]
    parts = axB.violinplot(data, positions=np.arange(len(ORDER)), showextrema=False, widths=0.85)
    for pc, s in zip(parts["bodies"], ORDER):
        pc.set_facecolor(PAL[s]); pc.set_alpha(0.55); pc.set_edgecolor("#444444"); pc.set_linewidth(0.4)
    for i, s in enumerate(ORDER):
        med = np.median(hours[s]); dep = (hours[s] < THRESHOLD_SUN_HRS).mean() * 100
        axB.scatter(i, med, color="#222222", s=8, zorder=4)
        axB.text(i, 10.9, f"{dep:.0f}%", ha="center", va="top", fontsize=6.5, fontweight="bold", color="#7A1F1F")
    axB.axhline(THRESHOLD_SUN_HRS, ls="--", color="#000000", lw=0.9)
    axB.text(len(ORDER) - 0.5, THRESHOLD_SUN_HRS + 0.15, "≥2 h general daylight/health reference (exogenous)",
             ha="right", va="bottom", fontsize=5.8, color="#333333")
    axB.set_xticks(np.arange(len(ORDER)))
    axB.set_xticklabels([SITE_LABELS[s] for s in ORDER], fontsize=6.8)
    axB.set_ylim(0, 11.6); axB.set_ylabel("winter direct-sun (h/day)", fontsize=7)
    axB.set_title("(B)  Winter direct-sun deficit — % of street cells below the reference floor",
                  loc="left", fontsize=8, pad=4)
    axB.text(0.0, -0.16, f"pooled {dep_all:.0f}% of street cells below the ≥2 h reference floor "
             f"(35–74% by site); the delivered, CFD-independent axis",
             transform=axB.transAxes, ha="left", va="top", fontsize=6.0, color="#555555")
    for sp in ("top", "right"):
        axB.spines[sp].set_visible(False)

    # ── Bottom-right: inequality (Lorenz + Gini) ──────────────────────────────
    axC = fig.add_subplot(gs[1, 4:6])
    axC.plot([0, 1], [0, 1], ls="--", color="#AAAAAA", lw=0.8)
    for s in ORDER:
        p, c = lorenz(hours[s])
        axC.plot(p, c, color=PAL[s], lw=1.4, label=f"{SITE_LABELS[s]} {gini(hours[s]):.2f}")
    axC.set_xlim(0, 1); axC.set_ylim(0, 1); axC.set_aspect("equal")
    axC.set_xlabel("share of street space", fontsize=6.5)
    axC.set_ylabel("share of winter sun", fontsize=6.5)
    axC.tick_params(labelsize=5.6)
    axC.set_title("(C)  Within-fabric inequality (Gini)", loc="left", fontsize=8, pad=4)
    axC.legend(loc="upper left", fontsize=5.2, frameon=False, handlelength=1.1, title="Gini", title_fontsize=5.4)
    axC.text(0.0, -0.20, "inequality WITHIN each favela; sites not ranked", transform=axC.transAxes,
             ha="left", va="top", fontsize=5.6, color="#555555")
    for sp in ("top", "right"):
        axC.spines[sp].set_visible(False)

    fig.suptitle("The germicidal-sunlight adequacy deficit across five favelas", x=0.075, ha="left",
                 fontsize=10, fontweight="bold")
    fig.text(0.075, 0.02, "Buildings © IPP (cadaster); ALS heights MIT/SondoTecnica. Pipeline open; "
             "input data not redistributable.", fontsize=5.6, color="#999999")
    save_fig(fig, "fig_solar_deficit")
    print("saved fig_solar_deficit  (pooled dep %.1f%%)" % dep_all)


if __name__ == "__main__":
    main()
