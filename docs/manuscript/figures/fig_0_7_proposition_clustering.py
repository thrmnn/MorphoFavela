#!/usr/bin/env python3
"""Figure 0.7 (proposition B) — Spatial clustering of compound failure.

Tests whether compound-failure cells (4-state grid from Fig 0.4) are
spatially structured or random. If structured, then targeting
interventions geographically (rather than by census boundary) is
mechanistically sound.

Panels
------
A   Global Moran's I per site with permutation null distribution
    (999 random reshuffles). Each site's observed I plotted as a
    coloured dot against a grey violin of the null. p-value annotated.
B   Local Moran's I (LISA) cluster map per site (5 panels). Cells
    classified as high-high (compound surrounded by compound) are
    highlighted; the rest fade. This identifies the "compound
    corridors" — where the failure concentrates.
C   Compound-cluster size distribution. CCDF of connected-component
    sizes (Rook adjacency) observed vs. random-spatial-permutation
    expectation. A heavy upper tail = real clustering.

Run:
    python docs/manuscript/figures/fig_0_7_proposition_clustering.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from libpysal.weights import Rook
from esda.moran import Moran, Moran_Local
from matplotlib.patches import Patch
from scipy.sparse.csgraph import connected_components

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "outputs" / "paper_figures"))
sys.path.insert(0, str(PROJECT_ROOT / "docs" / "manuscript" / "figures"))

from fig_style import (  # noqa: E402
    SITE_LABELS,
    SITE_ORDER,
    WIDTH_DOUBLE,
    add_north_arrow,
    add_scalebar,
    apply_style,
    clean_map_axes,
    load_boundary,
    load_buildings,
)
from fig_0_4_diagnostic import (  # noqa: E402
    BUILDING_COLOR,
    NOT_ASSESSED_COLOR,
    STATE_COLORS,
    TYPOLOGY_OF,
    build_diagnostic_grid,
)

EXPORTS_DIR = Path(__file__).resolve().parent / "exports"
EXPORTS_DIR.mkdir(exist_ok=True)

N_PERMUTATIONS = 999
TYPOLOGY_COLORS = {
    "hillside": "#0072B2",
    "mixed": "#009E73",
    "flatland": "#D55E00",
}
SITE_TO_TYP = {s: TYPOLOGY_OF[s].lower() for s in SITE_ORDER}


# ---------------------------------------------------------------------------
# Moran's I  / LISA / cluster sizes
# ---------------------------------------------------------------------------
def compute_moran_and_lisa(grid: gpd.GeoDataFrame) -> dict:
    """Return Moran's I, p, perm null, and LISA quadrants per cell."""
    cls = grid[grid["state"].notna()].copy()
    if len(cls) < 50:
        return {"moran": None, "lisa": None, "n": len(cls)}
    y = (cls["state"] == "compound").astype(int).to_numpy()
    if y.sum() < 5 or y.sum() == len(y):
        return {"moran": None, "lisa": None, "n": len(cls)}
    # Rook contiguity (cells sharing an edge); silence_warning to suppress
    # the unweighted "islands" warning for outlier cells.
    w = Rook.from_dataframe(cls, use_index=False, silence_warnings=True)
    w.transform = "r"  # row-standardize
    mo = Moran(y, w, permutations=N_PERMUTATIONS)
    lisa = Moran_Local(y, w, permutations=N_PERMUTATIONS)
    return {
        "moran_I": float(mo.I),
        "moran_p": float(mo.p_sim),
        "perm_null": np.array(mo.sim, dtype=float),
        "lisa_q": lisa.q,        # 1=HH, 2=LH, 3=LL, 4=HL
        "lisa_p": lisa.p_sim,
        "indices": cls.index.to_numpy(),
        "y": y,
        "n": len(cls),
    }


def cluster_size_distribution(grid: gpd.GeoDataFrame,
                              y_label: str = "compound") -> np.ndarray:
    """Connected-component sizes of cells in the given state (Rook adjacency).

    Returns array of component sizes (ints).
    """
    cls = grid[grid["state"].notna()].copy()
    mask = (cls["state"] == y_label).to_numpy()
    if mask.sum() < 2:
        return np.array([])
    sub = cls.iloc[np.where(mask)[0]]
    if len(sub) < 2:
        return np.array([])
    w = Rook.from_dataframe(sub, use_index=False, silence_warnings=True)
    n_comp, labels = connected_components(w.sparse, directed=False)
    sizes = np.bincount(labels)
    return sizes[sizes > 0]


def random_cluster_size_distribution(grid: gpd.GeoDataFrame,
                                     n_perms: int = 100) -> np.ndarray:
    """Random-permutation null for cluster sizes: reshuffle compound labels
    over classified cells and recompute connected-component sizes."""
    cls = grid[grid["state"].notna()].copy()
    n_compound = (cls["state"] == "compound").sum()
    if n_compound < 2:
        return np.array([])
    rng = np.random.default_rng(0)
    sizes_all = []
    n_cls = len(cls)
    for _ in range(n_perms):
        random_mask = np.zeros(n_cls, dtype=bool)
        idx = rng.choice(n_cls, size=int(n_compound), replace=False)
        random_mask[idx] = True
        sub = cls.iloc[np.where(random_mask)[0]]
        if len(sub) < 2:
            continue
        w = Rook.from_dataframe(sub, use_index=False, silence_warnings=True)
        _, labels = connected_components(w.sparse, directed=False)
        sizes = np.bincount(labels)
        sizes_all.append(sizes[sizes > 0])
    if not sizes_all:
        return np.array([])
    return np.concatenate(sizes_all)


# ---------------------------------------------------------------------------
# Panel A: Moran's I per site
# ---------------------------------------------------------------------------
def draw_panel_a(ax, moran_per_site: dict) -> None:
    sites = [s for s in SITE_ORDER if moran_per_site[s].get("moran_I") is not None]
    n = len(sites)
    x = np.arange(n)
    for i, s in enumerate(sites):
        d = moran_per_site[s]
        null = d["perm_null"]
        # Grey violin / null distribution (sampling thin)
        rng = np.random.default_rng(i)
        jitter = rng.uniform(-0.15, 0.15, size=len(null))
        ax.scatter(x[i] + jitter, null,
                   s=1.0, color="#cccccc", alpha=0.6, zorder=2,
                   linewidths=0)
        # Mean null (≈ 0 by construction).
        ax.scatter(x[i], np.mean(null), marker="_", color="#888",
                   s=80, linewidths=0.8, zorder=3)
        # Observed I.
        typ = SITE_TO_TYP[s]
        ax.scatter(x[i], d["moran_I"], marker="o",
                   color=TYPOLOGY_COLORS[typ], s=42,
                   edgecolor="white", linewidths=0.7, zorder=4)
        # p annotation.
        p = d["moran_p"]
        p_str = "p<0.001" if p < 0.001 else f"p={p:.3f}"
        ax.text(x[i], d["moran_I"] + 0.03,
                f"I={d['moran_I']:.2f}\n{p_str}",
                ha="center", va="bottom", fontsize=5.5,
                color="#222", fontweight="bold")

    ax.axhline(0, color="#aaa", lw=0.4, linestyle=":", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([SITE_LABELS[s] for s in sites], fontsize=6.5,
                       rotation=10)
    ax.set_ylabel("Moran's I (compound failure)", fontsize=7, labelpad=2)
    ax.tick_params(axis="y", labelsize=6, length=2, width=0.4, pad=2)
    ax.tick_params(axis="x", length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.4)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#eaeaea", linewidth=0.4, zorder=0)
    # Legend for typology colors.
    handles = [
        plt.scatter([], [], color=TYPOLOGY_COLORS[t], s=42, edgecolor="white",
                    linewidths=0.7, label=t.capitalize())
        for t in TYPOLOGY_COLORS
    ]
    handles.append(plt.scatter([], [], color="#cccccc", s=8, alpha=0.6,
                               linewidths=0, label="null (999 perms)"))
    handles.append(plt.scatter([], [], marker="_", color="#888", s=80,
                               linewidths=0.8, label="null mean"))
    leg = ax.legend(handles=handles, loc="upper right", fontsize=6,
                    frameon=False, labelspacing=0.45, handlelength=1.2,
                    borderpad=0.0)


# ---------------------------------------------------------------------------
# Panel B: LISA cluster maps
# ---------------------------------------------------------------------------
def draw_lisa_map(ax, site: str, grid: gpd.GeoDataFrame,
                  moran_data: dict, buildings: gpd.GeoDataFrame,
                  boundary: gpd.GeoDataFrame, alpha_threshold: float = 0.05
                  ) -> None:
    minx, miny, maxx, maxy = boundary.total_bounds
    pad = 0.02 * max(maxx - minx, maxy - miny)
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_aspect("equal")
    buildings.plot(ax=ax, facecolor=BUILDING_COLOR, edgecolor="none",
                   linewidth=0.0, zorder=1)

    # Background of classified cells: faint adequate / compound.
    cls = grid[grid["state"].notna()]
    if len(cls):
        cls.plot(ax=ax, facecolor=NOT_ASSESSED_COLOR, edgecolor="none",
                 alpha=0.35, zorder=2)

    if moran_data.get("lisa_q") is None:
        boundary.boundary.plot(ax=ax, color="black", linewidth=0.4,
                               linestyle=(0, (3, 2)), zorder=4, alpha=0.6)
        clean_map_axes(ax)
        ax.set_title(SITE_LABELS[site], fontsize=7.5, pad=4,
                     fontweight="bold", color="#222")
        ax.text(0.5, -0.04, "(insufficient compound cells)",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=5.5, color="#888", style="italic")
        return

    cls = grid.loc[moran_data["indices"]].copy()
    cls["lisa_q"] = moran_data["lisa_q"]
    cls["lisa_p"] = moran_data["lisa_p"]
    sig = cls[cls["lisa_p"] < alpha_threshold]
    hh = sig[sig["lisa_q"] == 1]
    ll = sig[sig["lisa_q"] == 3]
    if len(hh):
        hh.plot(ax=ax, facecolor=STATE_COLORS["compound"], edgecolor="none",
                alpha=0.95, zorder=3)
    if len(ll):
        ll.plot(ax=ax, facecolor=STATE_COLORS["adequate"], edgecolor="none",
                alpha=0.65, zorder=3)
    boundary.boundary.plot(ax=ax, color="black", linewidth=0.4,
                           linestyle=(0, (3, 2)), zorder=4, alpha=0.65)
    clean_map_axes(ax)
    ew = maxx - minx
    bar = 100 if ew < 1500 else (200 if ew < 2500 else 500)
    add_scalebar(ax, length_m=bar, loc="lower left")
    add_north_arrow(ax, loc="upper right", size=0.05)
    ax.set_title(SITE_LABELS[site], fontsize=7.5, pad=4, fontweight="bold",
                 color="#222")
    p_hh = 100.0 * len(hh) / max(len(cls), 1)
    ax.text(0.5, -0.04,
            f"compound-corridor cells {p_hh:.1f}%",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=6, color=STATE_COLORS["compound"], fontweight="bold")


# ---------------------------------------------------------------------------
# Panel C: cluster-size CCDF
# ---------------------------------------------------------------------------
def draw_panel_c(ax, observed_sizes: dict, random_sizes: dict) -> None:
    # Pool across sites for clarity; one CCDF line for observed, one for null.
    obs_all = np.concatenate([s for s in observed_sizes.values() if len(s)])
    null_all = np.concatenate([s for s in random_sizes.values() if len(s)])
    if len(obs_all) == 0 or len(null_all) == 0:
        ax.text(0.5, 0.5, "(insufficient data)", ha="center", va="center",
                transform=ax.transAxes, fontsize=7, color="#888")
        return

    def ccdf(arr):
        s = np.sort(arr)
        p = 1.0 - np.arange(1, len(s) + 1) / len(s)
        return s, p

    xo, po = ccdf(obs_all)
    xn, pn = ccdf(null_all)
    ax.step(xo, po, where="post", color=STATE_COLORS["compound"], lw=1.4,
            label="Observed", zorder=4)
    ax.step(xn, pn, where="post", color="#888", lw=1.0, linestyle=(0, (3, 2)),
            label="Random null", zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Cluster size (cells)", fontsize=7, labelpad=2)
    ax.set_ylabel("P(size ≥ x)", fontsize=7, labelpad=2)
    ax.tick_params(labelsize=6, length=2, width=0.4, pad=2)
    ax.set_xlim(1, max(xo.max(), xn.max()) * 1.2)
    ax.set_ylim(0.5 / len(obs_all), 1.1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.4)
    ax.set_axisbelow(True)
    ax.grid(color="#eaeaea", linewidth=0.4, zorder=0)
    leg = ax.legend(loc="upper right", fontsize=6.5, frameon=False,
                    handlelength=1.4, labelspacing=0.4, borderpad=0.0)


def main() -> None:
    apply_style()
    warnings.filterwarnings("ignore", category=UserWarning, module="geopandas")
    warnings.filterwarnings("ignore", category=UserWarning, module="libpysal")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    print("Building diagnostic grids ...")
    grids = {}
    moran = {}
    obs_sizes = {}
    null_sizes = {}
    for site in SITE_ORDER:
        g = build_diagnostic_grid(site)
        grids[site] = g
        moran[site] = compute_moran_and_lisa(g)
        if moran[site].get("moran_I") is not None:
            d = moran[site]
            print(f"  {site:<22} I = {d['moran_I']:.3f}  "
                  f"p = {d['moran_p']:.3f}  n_cells = {d['n']}")
            obs_sizes[site] = cluster_size_distribution(g)
            null_sizes[site] = random_cluster_size_distribution(g, n_perms=20)
            if len(obs_sizes[site]):
                print(f"    observed clusters: {len(obs_sizes[site])}  "
                      f"max size {obs_sizes[site].max()}")
        else:
            print(f"  {site:<22} -- insufficient classified cells "
                  f"({moran[site]['n']})")

    boundaries = {s: load_boundary(s) for s in SITE_ORDER}
    buildings = {s: load_buildings(s, extended=False) for s in SITE_ORDER}

    fig = plt.figure(figsize=(WIDTH_DOUBLE, WIDTH_DOUBLE * 1.05))
    outer = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[1.4, 2.4, 1.4],
        hspace=0.32,
        left=0.05, right=0.96, top=0.93, bottom=0.06,
    )

    # Row 0: panel A.
    a_gs = outer[0].subgridspec(1, 2, width_ratios=[0.68, 0.32], wspace=0.10)
    ax_a = fig.add_subplot(a_gs[0, 0])
    draw_panel_a(ax_a, moran)

    ax_cap_a = fig.add_subplot(a_gs[0, 1])
    ax_cap_a.set_xticks([])
    ax_cap_a.set_yticks([])
    for sp in ax_cap_a.spines.values():
        sp.set_visible(False)
    cap_a = (
        "Global Moran's I on the binary compound-failure indicator,\n"
        "Rook contiguity (10 m cells sharing an edge), row-standardized.\n"
        "Null obtained by random label permutation (999 reshuffles).\n"
        "I > 0 → compound cells co-locate more than chance; I ≈ 0 →\n"
        "spatially random; I < 0 → regularly dispersed."
    )
    ax_cap_a.text(0.0, 0.95, cap_a, transform=ax_cap_a.transAxes,
                  ha="left", va="top", fontsize=5.8, color="#333",
                  linespacing=1.4)

    # Row 1: LISA maps.
    maps_gs = outer[1].subgridspec(1, len(SITE_ORDER), wspace=0.02)
    for j, site in enumerate(SITE_ORDER):
        ax = fig.add_subplot(maps_gs[0, j])
        draw_lisa_map(ax, site, grids[site], moran[site],
                      buildings[site], boundaries[site])

    # Row 2: panel C CCDF.
    c_gs = outer[2].subgridspec(1, 2, width_ratios=[0.55, 0.45], wspace=0.10)
    ax_c = fig.add_subplot(c_gs[0, 0])
    draw_panel_c(ax_c, obs_sizes, null_sizes)

    ax_cap_c = fig.add_subplot(c_gs[0, 1])
    ax_cap_c.set_xticks([])
    ax_cap_c.set_yticks([])
    for sp in ax_cap_c.spines.values():
        sp.set_visible(False)
    cap_c = (
        "Cluster sizes pooled across sites. Components found by Rook\n"
        "adjacency on the subset of cells classified as compound failure.\n"
        "Random null: same number of compound labels permuted uniformly\n"
        "over classified cells (20 reshuffles × site). A heavy upper tail\n"
        "in the observed CCDF, relative to the null, is the signature of\n"
        "spatial clustering."
    )
    ax_cap_c.text(0.0, 0.95, cap_c, transform=ax_cap_c.transAxes,
                  ha="left", va="top", fontsize=5.8, color="#333",
                  linespacing=1.4)

    # Panel labels.
    pos_a = ax_a.get_position()
    fig.text(pos_a.x0 - 0.025, pos_a.y1 + 0.005, "a",
             fontsize=9, fontweight="bold", va="bottom", ha="left")
    first_map = fig.axes[2]  # 0=A, 1=cap_a, 2=first map
    pos_m = first_map.get_position()
    fig.text(pos_m.x0 - 0.025, pos_m.y1 + 0.015, "b",
             fontsize=9, fontweight="bold", va="bottom", ha="left")
    pos_c = ax_c.get_position()
    fig.text(pos_c.x0 - 0.025, pos_c.y1 + 0.005, "c",
             fontsize=9, fontweight="bold", va="bottom", ha="left")

    fig.text(0.5, 0.985,
             "Spatial clustering: compound failure forms coherent corridors, not random pixels",
             ha="center", va="top", fontsize=8.5, fontweight="bold",
             color="#1a1a1a")
    fig.text(0.5, 0.968,
             "PROPOSITION B — Moran's I + LISA on the Fig 0.4 diagnostic. Synthetic CFD; pattern conserved on real CFD if morphology dominates.",
             ha="center", va="top", fontsize=5.5, style="italic", color="#a0522d")

    out_png = EXPORTS_DIR / "fig_0_7_proposition_clustering.png"
    out_svg = EXPORTS_DIR / "fig_0_7_proposition_clustering.svg"
    print(f"Saving {out_png.name} + {out_svg.name} ...")
    fig.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.05,
                facecolor="white")
    fig.savefig(out_svg, format="svg", bbox_inches="tight", pad_inches=0.05,
                facecolor="white")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
