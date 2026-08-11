#!/usr/bin/env python3
"""Flagship Fig 2 — the delivered sunlight-adequacy deficit (+ within-fabric inequality).

Council split of the old combined fig03: this is the SOLAR-ONLY, CFD-independent
lead result. Reuses fig03's proven, stroke-free (occlusion-safe) choropleth code.
  Top   Per-site winter-solstice direct-sun choropleths (cividis, 0-10.5 h);
        colorbar marks the >=2 h reference floor.
  BL    Per-site winter direct-sun histogram (0.5 h bins, the native 30-min
        slices), with a RED >=2 h reference line, a thin median line, and the
        per-site share below the floor (pooled 46%; 35-74% by site).
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
from matplotlib.transforms import blended_transform_factory

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

    fig = plt.figure(figsize=(7.6, 5.8))
    # Two row-groups: (0) winter-sun maps, (1) distributions (B) + Lorenz/Gini (C).
    gs = fig.add_gridspec(2, 6, height_ratios=[0.92, 1.0], hspace=0.62, wspace=0.12,
                          left=0.065, right=0.88, top=0.865, bottom=0.098)

    # ── Top: winter-sun maps — DISCRETE 4-class for legibility at 10 m cells ────
    # Continuous cividis turned the dense low-sun fabric into dark speckle at this
    # panel size; four classes (0 h / <2 h deficit / 2-5 h / >=5 h) make the
    # sub-2 h deficit read as coherent zones. Cells are 10 m polygons, stroke-free.
    import matplotlib.patches as mpatches
    CLASS_EDGES = [0.001, THRESHOLD_SUN_HRS, 5.0]        # -> 0 h | <2 h | 2-5 h | >=5 h
    CLASS_COLS = ["#0b1b3f", "#3b5488", "#b8b06a", "#f5e11d"]
    CLASS_LBL = ["0 h (no winter sun)", "< 2 h (below floor)", "2–5 h", "≥ 5 h"]
    row = []
    for i, s in enumerate(ORDER):
        ax = fig.add_subplot(gs[0, i]); row.append(ax)
        g = grids[s].copy()
        v = g["solar_hours_winter"].to_numpy(float)
        g["_col"] = np.array(CLASS_COLS)[np.digitize(v, CLASS_EDGES)]
        g.plot(ax=ax, color=g["_col"].values, edgecolor="none", linewidth=0.0)
        b = g.total_bounds; pad = 12.0
        ax.set_xlim(b[0] - pad, b[2] + pad); ax.set_ylim(b[1] - pad, b[3] + pad)
        ax.set_aspect("equal"); ax.set_anchor("N")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    handles = [mpatches.Patch(facecolor=c, edgecolor="none", label=l)
               for c, l in zip(CLASS_COLS, CLASS_LBL)]
    row[-1].legend(handles=handles, loc="center left", bbox_to_anchor=(1.04, 0.5),
                   fontsize=6.0, frameon=False, handlelength=1.1, labelspacing=0.55,
                   title="winter direct-sun", title_fontsize=6.4)

    # uniform site-title row above the sun maps (ax titles drift because equal-aspect
    # boxes shrink per map; place them from the gridspec cells instead)
    for i, s in enumerate(ORDER):
        cell = gs[0, i].get_position(fig)
        fig.text((cell.x0 + cell.x1) / 2, cell.y1 + 0.004, SITE_LABELS[s],
                 ha="center", va="bottom", fontsize=7.2, fontweight="bold", color="#222222")
    # letter the whole map row-group ONCE, with the deficit read-out beneath it
    yA = gs[0, 0].get_position(fig).y1
    fig.text(0.065, yA + 0.032, "(A)  Winter direct-sun maps (four-class: 0 h / <2 h / 2–5 h / ≥5 h)",
             ha="left", va="bottom",
             fontsize=8, fontweight="bold", color="#111111")
    fig.text(0.065, yA + 0.019, "the sub-2 h deficit (dark blue) forms coherent zones across the "
             "hillside fabric; flatland sites reach the floor more often",
             ha="left", va="bottom", fontsize=6.2, style="italic", color="#555555")

    # ── Bottom-left: per-site VERTICAL histograms (sun-hours on the y-axis) ─────
    # Five site columns; sun-hours rise on y (0-10.5 h), horizontal bars = share of
    # cells per 0.5 h bin (coloured by the 4-class scheme, matching the maps). One
    # red HORIZONTAL >=2 h floor line across all sites; thin dark median line each.
    bins = np.arange(0.0, 10.5 + 0.5, 0.5)          # native 30-min slices
    centers = (bins[:-1] + bins[1:]) / 2
    bar_cols = [CLASS_COLS[int(np.digitize(c, CLASS_EDGES))] for c in centers]
    inner = gs[1, 0:4].subgridspec(1, len(ORDER), wspace=0.18)
    hist_axes = []
    for i, s in enumerate(ORDER):
        axh = fig.add_subplot(inner[0, i], sharey=hist_axes[0] if hist_axes else None)
        hist_axes.append(axh)
        h = np.clip(hours[s], 0.0, bins[-1] - 1e-9)
        frac = np.histogram(h, bins=bins)[0] / len(h)
        med = np.median(hours[s]); dep = (hours[s] < THRESHOLD_SUN_HRS).mean() * 100
        xmax = max(frac.max(), 0.01) * 1.2
        axh.barh(centers, frac, height=0.46, color=bar_cols, zorder=2)
        # red >=2 h reference floor: dashed so it survives greyscale and stays visually
        # distinct from the solid black median tick.
        axh.axhline(THRESHOLD_SUN_HRS, color="#CC0000", lw=1.2, ls=(0, (4, 2)), zorder=4)
        # median as a HALF-LENGTH tick over the bar region (not a full-width rule).
        # Rocinha's ~0 h median would sit on the axis floor and vanish, so clamp the
        # DRAWN y to a small min offset while the value label reports the true median.
        y_med = max(med, 0.15)
        axh.hlines(y_med, 0.0, xmax * 0.5, color="#1A1A1A", lw=1.2, zorder=5)
        trans = blended_transform_factory(axh.transAxes, axh.transData)
        axh.text(0.5, y_med + 0.35, f"med {med:.1f} h", transform=trans, ha="center",
                 va="bottom", fontsize=5.2, color="#1A1A1A", zorder=6)
        axh.set_ylim(0, bins[-1]); axh.set_xlim(0, xmax)
        axh.set_xticks([])
        axh.set_title(SITE_LABELS[s], fontsize=6.6, fontweight="bold", color="#333333", pad=3)
        axh.text(0.96, 0.99, f"{dep:.0f}%\n<2 h", transform=axh.transAxes, ha="right",
                 va="top", fontsize=5.8, fontweight="bold", color="#7A1F1F", linespacing=0.9)
        for sp in ("top", "right", "bottom"):
            axh.spines[sp].set_visible(False)
        if i > 0:
            axh.spines["left"].set_visible(False); axh.tick_params(labelleft=False)
    hist_axes[0].set_yticks(np.arange(0, 11, 2)); hist_axes[0].tick_params(labelsize=6)
    hist_axes[0].set_ylabel("winter direct-sun (h/day)", fontsize=7)
    yB = hist_axes[0].get_position().y1
    fig.text(0.065, yB + 0.044, "(B)  Winter direct-sun distribution by site", ha="left", va="bottom",
             fontsize=8, fontweight="bold", color="#111111")
    fig.text(0.065, yB + 0.030, "red dashed line = ≥2 h reference (exogenous); dark tick = median",
             ha="left", va="bottom", fontsize=6.2, style="italic", color="#555555")
    fig.text(0.065, 0.045, f"pooled {dep_all:.0f}% of built cells below the ≥2 h reference floor "
             f"(35–74% by site); the delivered, CFD-independent axis",
             ha="left", va="top", fontsize=6.0, color="#555555")

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
    pC = axC.get_position()
    fig.text(pC.x0, yB + 0.044, "(C)  Within-fabric inequality (Gini)", ha="left", va="bottom",
             fontsize=8, fontweight="bold", color="#111111")
    axC.legend(loc="upper left", fontsize=5.2, frameon=False, handlelength=1.1, title="Gini", title_fontsize=5.4)
    axC.text(0.0, -0.20, "inequality WITHIN each favela; sites not ranked", transform=axC.transAxes,
             ha="left", va="top", fontsize=5.6, color="#555555")
    for sp in ("top", "right"):
        axC.spines[sp].set_visible(False)

    fig.suptitle("Unequal delivery of germicidal sunlight across five favelas", x=0.075, ha="left",
                 fontsize=10, fontweight="bold")
    fig.text(0.075, 0.02, "Buildings © IPP (cadaster); ALS heights MIT/SondoTecnica. Pipeline open; "
             "input data not redistributable.", fontsize=5.6, color="#999999")
    save_fig(fig, "fig_solar_deficit")
    print("saved fig_solar_deficit  (pooled dep %.1f%%)" % dep_all)


if __name__ == "__main__":
    main()
