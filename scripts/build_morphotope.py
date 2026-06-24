"""WS — block-scale morphotopes (the favela signature at tissue scale).

Per cell → its ~50 m morphotype-composition → GMM (k by BIC) → morphotopes →
cross-site recurrence + the cell-type profile of each tissue + per-favela maps.
Writes a `morphotope` column back to features_grid. Decisions:
docs/morpho_signature_decisions.md.

    python scripts/build_morphotope.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from src.morphometry.morphotope import (  # noqa: E402
    N_TYPES,
    cell_type_profile,
    fit_morphotopes,
    neighborhood_composition,
    select_k_bic,
)
from src.morphometry.signature import (  # noqa: E402
    CAMPAIGN_SITES,
    choose_k_elbow,
    recurrence_flags,
    recurrence_matrix,
)
from src.svf_v2.io import _git_sha  # noqa: E402
from src.viz.signature_style import (  # noqa: E402
    MORPHOTOPE_LABEL,
    NULL_COLOR,
    TYPE_COLORS,
    TYPE_LABEL,
)

OUT = ROOT / "outputs" / "cross_site" / "morphotope"
FIGS = ROOT / "outputs" / "cross_site" / "signature" / "figures_v2"
SITE_NAMES = {"vidigal": "Vidigal", "rocinha": "Rocinha", "riodaspedras": "Rio das Pedras",
              "complexo_do_alemao": "Complexo do Alemão", "maré": "Maré"}
RADIUS = 50.0
# morphotope palette — distinct from the cell-type Okabe–Ito (greens→browns, ordered)
MT_COLORS = ["#d9f0d3", "#a6dba0", "#5aae61", "#9970ab", "#762a83", "#40004b"]


def load() -> pd.DataFrame:
    frames = []
    for s in CAMPAIGN_SITES:
        p = ROOT / "outputs" / s / "features" / "features_grid.parquet"
        if not p.exists():
            continue
        g = gpd.read_parquet(p)
        col = "morphotype_smooth" if "morphotype_smooth" in g else "morphotype"
        d = pd.DataFrame(g.drop(columns="geometry"))
        d["site"] = s
        d = d.rename(columns={col: "morphotype_smooth"})
        frames.append(d[["site", "zone_id", "centroid_x", "centroid_y",
                         "morphotype_smooth"]])
    return pd.concat(frames, ignore_index=True)


def write_back(comp, labels):
    lut = comp[["site", "zone_id"]].copy()
    lut["morphotope"] = labels
    for site, grp in lut.groupby("site"):
        p = ROOT / "outputs" / site / "features" / "features_grid.parquet"
        g = gpd.read_parquet(p).drop(columns="morphotope", errors="ignore")
        g = g.merge(grp.drop(columns="site"), on="zone_id", how="left")
        g["morphotope"] = g["morphotope"].astype("Int64")
        g.to_parquet(p)


def fig_cell_profile(prof, k):
    """What cell-types each morphotope is made of (stacked) — shows where T2/T3 live."""
    fig, ax = plt.subplots(figsize=(7, 3.4))
    bottom = np.zeros(k)
    x = np.arange(k)
    for c in range(N_TYPES):
        vals = prof[f"comp_{c}"].to_numpy() * 100
        ax.bar(x, vals, bottom=bottom, color=TYPE_COLORS[c], width=0.8,
               edgecolor="white", linewidth=1.2, label=f"T{c} {TYPE_LABEL[c]}")
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([f"M{m}" for m in range(k)])
    ax.set_ylabel("% of cells in tissue")
    ax.set_ylim(0, 100)
    ax.set_title("What each MORPHOTOPE (block tissue) is made of — its mix of cell MORPHOTYPES "
                 "(T0–T5). T2/T3 cells concentrate in specific tissues, not noise.", fontsize=8)
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              frameon=False)
    fig.tight_layout()
    fig.savefig(FIGS / "morphotope_profile.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_recurrence(shares, flags, k):
    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(shares.to_numpy(), cmap="YlGnBu", vmin=0,
                   vmax=float(np.nanmax(shares.to_numpy())))
    ax.set_xticks(range(k))
    ax.set_xticklabels([f"M{m}" for m in range(k)])
    ax.set_yticks(range(len(shares.index)))
    ax.set_yticklabels([SITE_NAMES.get(s, s) for s in shares.index])
    for (i, j), v in np.ndenumerate(shares.to_numpy()):
        if not np.isnan(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="black" if v < 0.3 else "white")
    ax.set_title("Morphotope (block tissue) recurrence — tissue share per favela", fontsize=9)
    fig.colorbar(im, ax=ax, label="share", shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIGS / "morphotope_recurrence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_maps(k):
    cmap = ListedColormap(MT_COLORS[:k])
    camp = [(s, ROOT / "outputs" / s / "features" / "features_grid.parquet")
            for s in CAMPAIGN_SITES]
    camp = [(s, gpd.read_parquet(p)) for s, p in camp if p.exists()]
    widths = [g.total_bounds[2] - g.total_bounds[0] for _, g in camp]
    fig, axes = plt.subplots(1, len(camp), figsize=(2.7 * len(camp), 3.6),
                             gridspec_kw={"width_ratios": widths})
    for ax, (s, g) in zip(np.atleast_1d(axes), camp):
        g.plot(ax=ax, color=NULL_COLOR, linewidth=0)
        gg = g.dropna(subset=["morphotope"])
        gg.plot(ax=ax, column="morphotope", categorical=True, cmap=cmap,
                vmin=0, vmax=k - 1, linewidth=0)
        gpd.GeoSeries([g.geometry.union_all()], crs=g.crs).boundary.plot(
            ax=ax, color="0.25", lw=0.5)
        ax.set_xlim(g.total_bounds[0], g.total_bounds[2])
        ax.set_ylim(g.total_bounds[1], g.total_bounds[3])
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(SITE_NAMES.get(s, s), fontsize=9)
    handles = [Rectangle((0, 0), 1, 1, color=MT_COLORS[m]) for m in range(k)]
    fig.legend(handles, [f"M{m} {MORPHOTOPE_LABEL.get(m, '')}" for m in range(k)], loc="lower center", ncol=k,
               fontsize=8, frameon=False)
    fig.suptitle("MORPHOTOPES (M0–M4) — block-scale favela tissue, distinct from the cell "
                 "MORPHOTYPES (T0–T5); 50 m composition window", fontsize=9)
    fig.savefig(FIGS / "morphotope_maps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    comp = neighborhood_composition(df, radius=RADIUS)
    sel = select_k_bic(comp)
    k = choose_k_elbow(sel)
    labels = fit_morphotopes(comp, k, random_state=0)
    write_back(comp, labels)

    shares = recurrence_matrix(comp.assign(_l=labels).rename(columns={"_l": "x"})
                               .drop(columns=[c for c in comp.columns if c.startswith("comp_")]),
                               labels)
    flags = recurrence_flags(shares)
    prof = cell_type_profile(comp, labels)

    sel.to_csv(OUT / "k_selection.csv", index=False)
    shares.to_csv(OUT / "recurrence.csv")
    prof.to_csv(OUT / "cell_type_profile.csv")
    fig_cell_profile(prof, k)
    fig_recurrence(shares, flags, k)
    fig_maps(k)
    (OUT / "run_meta.json").write_text(json.dumps({
        "git_sha": _git_sha(),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "radius_m": RADIUS, "k": int(k), "n_cells": int(len(comp))}, indent=2))

    print(f"k={k} morphotopes, radius={RADIUS} m, {len(comp)} cells")
    print("\nrecurrence (share per site):")
    print(shares.round(2).to_string())
    print("\nrecurs flags:")
    print(flags.to_string())
    print("\ncell-type profile (mean composition per tissue):")
    print(prof.round(2).to_string())


if __name__ == "__main__":
    main()
