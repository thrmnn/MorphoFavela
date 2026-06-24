"""Build per-cell street-network / beco configuration metrics across sites.

Reads each campaign site's road shapefile and 10 m feature grid, computes
intersection density and mean segment length per cell (no graph library — node =
shared snapped endpoint of road segments), pools the cells, summarises the
configuration axis by morphotype (T0–T5), and writes:

- outputs/cross_site/streetnet/streetnet_by_type.csv
- outputs/cross_site/signature/figures_v2/beco_by_type.png  (group: signature)

Run:
    /home/theo/miniconda3/envs/IVF/bin/python scripts/build_streetnet.py
"""

from __future__ import annotations

import os
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.morphometry.signature import CAMPAIGN_SITES  # noqa: E402
from src.morphometry.streetnet import (  # noqa: E402
    cell_streetnet_metrics,
    find_roads_path,
    summarise_by_type,
)
from src.viz.signature_style import TYPE_COLORS, TYPE_LABEL  # noqa: E402

FIG_DIR = os.path.join(ROOT, "outputs/cross_site/signature/figures_v2")
CSV_DIR = os.path.join(ROOT, "outputs/cross_site/streetnet")


def load_site_cells(site: str) -> tuple[pd.DataFrame | None, str | None]:
    """Compute per-cell streetnet metrics for one site.

    Returns ``(cells_df, None)`` on success or ``(None, reason)`` if the site
    must be skipped (missing/empty roads).
    """
    site_dir = os.path.join(ROOT, "data", site)
    roads_path = find_roads_path(site_dir)
    if roads_path is None:
        return None, f"{site}: no roads/street shapefile found"

    roads = gpd.read_file(roads_path)
    roads = roads[roads.geometry.notna() & ~roads.geometry.is_empty]
    if len(roads) == 0:
        return None, f"{site}: roads file has no usable geometry"

    grid_path = os.path.join(ROOT, "outputs", site, "features", "features_grid.parquet")
    grid = gpd.read_parquet(grid_path)
    if grid.crs != roads.crs:
        roads = roads.to_crs(grid.crs)

    cells = cell_streetnet_metrics(grid, roads)
    cells["site"] = site
    cells["morphotype_smooth"] = grid["morphotype_smooth"].to_numpy()
    return cells, None


def make_figure(summary: pd.DataFrame, out_path: str) -> None:
    """Two-panel bar chart: intersection density and segment grain by type."""
    summary = summary.sort_values("morphotype")
    types = summary["morphotype"].astype(int).tolist()
    colors = [TYPE_COLORS.get(t, "#888888") for t in types]
    labels = [TYPE_LABEL.get(t, f"T{t}") for t in types]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    ax = axes[0]
    ax.bar(range(len(types)), summary["mean_intersection_density_ha"], color=colors)
    ax.set_title("Intersection density by morphotype")
    ax.set_ylabel("junctions (deg≥3) per ha")
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)

    ax = axes[1]
    ax.bar(range(len(types)), summary["mean_segment_len_m"], color=colors)
    ax.set_title("Mean street-segment length by morphotype")
    ax.set_ylabel("mean segment length (m)  — shorter = finer beco grain")
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)

    fig.suptitle(
        "Beco (alley) configuration axis — street-network grain the intensity "
        "vector misses",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)

    frames: list[pd.DataFrame] = []
    skipped: list[str] = []
    for site in CAMPAIGN_SITES:
        cells, reason = load_site_cells(site)
        if cells is None:
            skipped.append(reason)
            print("SKIP", reason)
            continue
        frames.append(cells)
        print(
            f"{site}: {len(cells)} cells, "
            f"{cells['intersection_count'].sum()} junctions, "
            f"{(cells['n_segments'] > 0).sum()} road-touched cells"
        )

    if not frames:
        raise SystemExit("No site produced streetnet metrics; aborting.")

    pooled = pd.concat(frames, ignore_index=True)
    summary = summarise_by_type(pooled)

    csv_path = os.path.join(CSV_DIR, "streetnet_by_type.csv")
    summary.to_csv(csv_path, index=False)
    print("wrote", csv_path)

    fig_path = os.path.join(FIG_DIR, "beco_by_type.png")
    make_figure(summary, fig_path)
    print("wrote", fig_path)

    print("\n=== streetnet by morphotype ===")
    with pd.option_context("display.width", 140, "display.max_columns", None):
        print(summary.to_string(index=False))
    if skipped:
        print("\nSkipped sites:", skipped)


if __name__ == "__main__":
    main()
