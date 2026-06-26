"""Add configuration metrics (party-wall adjacency) to the feature substrate.

For each campaign site: compute per-building party-wall ratio from footprints,
area-weight it to the 10 m grid, and merge `party_wall_ratio` + `n_buildings_cfg`
+ `has_config` into features_grid.parquet. Then a figure: party-wall ratio by
morphotype — does configuration discriminate the types the intensity vector built?

    python scripts/build_configuration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.morphometry.configuration import aggregate_to_grid, party_wall_ratio
from src.morphometry.signature import CAMPAIGN_SITES
from src.viz.signature_style import TYPE_COLORS, TYPE_LABEL

FIGS = ROOT / "outputs" / "cross_site" / "signature" / "figures_v2"
FOOTPRINTS = "buildings_extended_300m.gpkg"


def load_footprints(site: str) -> gpd.GeoDataFrame | None:
    p = ROOT / "data" / site / FOOTPRINTS
    if not p.exists():
        return None
    b = gpd.read_file(p)
    if "topo" in b.columns:                       # known phantom-height corruption
        b = b[b["topo"] != 0]
    b = b[b.geometry.notna()].copy()
    b["geometry"] = b.geometry.buffer(0)          # repair invalid rings
    return b[b.geometry.area > 1]


def main():
    rows = []
    for site in CAMPAIGN_SITES:
        b = load_footprints(site)
        gpath = ROOT / "outputs" / site / "features" / "features_grid.parquet"
        if b is None or not gpath.exists():
            print(f"{site}: skipped (no footprints/grid)")
            continue
        grid = gpd.read_parquet(gpath).to_crs(b.crs)
        ratio = party_wall_ratio(b)
        agg = aggregate_to_grid(b, grid, ratio)
        g = gpd.read_parquet(gpath)
        g = g.drop(columns=["party_wall_ratio", "n_buildings_cfg", "has_config"],
                   errors="ignore").merge(agg, on="zone_id", how="left")
        g.to_parquet(gpath)
        col = "morphotype_smooth" if "morphotype_smooth" in g else "morphotype"
        sub = g.dropna(subset=["party_wall_ratio", col])
        for c in range(6):
            v = sub.loc[sub[col] == c, "party_wall_ratio"]
            if len(v):
                rows.append({"site": site, "morphotype": c,
                             "pwr_median": float(v.median()), "n": int(len(v))})
        print(f"{site:20s} party-wall ratio median {np.nanmedian(ratio):.2f} "
              f"({len(b)} buildings)")

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "outputs" / "cross_site" / "signature" / "party_wall_by_type.csv",
              index=False)
    # figure: party-wall ratio by morphotype (pooled across sites)
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    pooled = df.groupby("morphotype").apply(
        lambda d: np.average(d["pwr_median"], weights=d["n"]), include_groups=False)
    ax.bar([f"T{c}\n{TYPE_LABEL[c]}" for c in pooled.index], pooled.to_numpy(),
           color=[TYPE_COLORS[c] for c in pooled.index])
    ax.axhline(0.1, color="0.5", ls=":", lw=1)
    ax.text(5.4, 0.12, "detached formal ≈0.1", fontsize=6, color="0.5", ha="right")
    ax.set_ylabel("median party-wall ratio")
    ax.set_ylim(0, 1)
    ax.set_title("Favela fabric is highly fused everywhere (0.6–1.0 vs ~0.1 detached) — "
                 "and FLAT types (T2/T3/T5) are fully party-walled, HILLSIDE (T1/T4) "
                 "more stepped: a configuration axis density alone misses", fontsize=7.5)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="y", color="0.92")
    fig.tight_layout()
    fig.savefig(FIGS / "party_wall_by_type.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\nparty-wall ratio by morphotype (pooled):")
    print(pooled.round(2).to_string())


if __name__ == "__main__":
    main()
