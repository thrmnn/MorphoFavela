"""WS-B (geometry-first) — build the morphometrics-only prioritization map.

Scores every street observer with the pure-geometry deprivation index, aggregates
to cells by worst-decile (p90), writes priority columns back into each site's
features_grid.parquet, and renders a per-site priority map (shared tertile breaks,
NULL grey) into the signature figures_v2/ gallery.

    python scripts/build_prioritization.py
"""

from __future__ import annotations

import glob
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

from src.morphometry.prioritization import (  # noqa: E402
    CLASS_LABELS,
    WEIGHTS,
    aggregate_priority_to_cells,
    priority_score,
)
from src.morphometry.signature import CAMPAIGN_SITES  # noqa: E402
from src.svf_v2.io import _git_sha  # noqa: E402

SIG = ROOT / "outputs" / "cross_site" / "signature"
FIGS = SIG / "figures_v2"
PRIORITY_COLORS = ["#fee391", "#fe9929", "#cc4c02"]  # YlOrBr tertiles
NULL_COLOR = "#E0E0E0"


def _scalebar(ax, length_m=200):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x = x0 + 0.06 * (x1 - x0)
    y = y0 + 0.06 * (y1 - y0)
    ax.plot([x, x + length_m], [y, y], color="black", lw=2)
    ax.text(x + length_m / 2, y + 0.015 * (y1 - y0), f"{length_m} m",
            ha="center", va="bottom", fontsize=6)


def main():
    cells = {}
    for p in sorted(glob.glob(str(ROOT / "outputs/*/features/features_grid.parquet"))):
        site = Path(p).parents[1].name
        grid = gpd.read_parquet(p)
        obs = gpd.read_parquet(Path(p).parents[1] / "features" / "features_street.parquet")
        obs = obs.dropna(subset=["zone_id", "svf", "solar_hours_winter", "lambda_f_mean"])
        score = priority_score(obs)
        agg = aggregate_priority_to_cells(obs, grid, score)
        grid = grid.drop(columns=["priority_p90", "priority_p50", "priority_class"],
                         errors="ignore").merge(
            agg[["zone_id", "priority_p90", "priority_p50", "has_priority"]],
            on="zone_id", how="left")
        cells[site] = (p, grid)

    # shared tertile breaks on the pooled supported-cell p90
    pooled = pd.concat([g["priority_p90"] for _, g in cells.values()]).dropna()
    breaks = [pooled.quantile(q) for q in (1 / 3, 2 / 3)]
    edges = [-np.inf, breaks[0], breaks[1], np.inf]

    summary = []
    for site, (p, grid) in cells.items():
        cls = pd.cut(grid["priority_p90"], bins=edges, labels=list(CLASS_LABELS))
        grid["priority_class"] = cls.astype("string")
        grid.to_parquet(p)
        vc = cls.value_counts()
        summary.append({"site": site, **{c: int(vc.get(c, 0)) for c in CLASS_LABELS}})

    # per-site priority map (campaign sites), shared breaks
    camp = [(s, cells[s][1]) for s in CAMPAIGN_SITES if s in cells]
    widths = [g.total_bounds[2] - g.total_bounds[0] for _, g in camp]
    fig, axes = plt.subplots(1, len(camp), figsize=(2.6 * len(camp), 3.4),
                             gridspec_kw={"width_ratios": widths})
    cmap = ListedColormap(PRIORITY_COLORS)
    norm = BoundaryNorm([0, 1, 2, 3], cmap.N)
    code = {c: i for i, c in enumerate(CLASS_LABELS)}
    for ax, (site, g) in zip(np.atleast_1d(axes), camp):
        g.plot(ax=ax, color=NULL_COLOR, linewidth=0)
        gg = g.dropna(subset=["priority_class"]).copy()
        gg["c"] = gg["priority_class"].map(code)
        gg.plot(ax=ax, column="c", cmap=cmap, norm=norm, linewidth=0)
        gpd.GeoSeries([g.geometry.union_all()], crs=g.crs).boundary.plot(
            ax=ax, color="0.25", linewidth=0.6)
        ax.set_xlim(g.total_bounds[0], g.total_bounds[2])
        ax.set_ylim(g.total_bounds[1], g.total_bounds[3])
        ax.set_aspect("equal")
        _scalebar(ax)
        ax.set_axis_off()
        ax.set_title(site.replace("_", " "), fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=PRIORITY_COLORS[i]) for i in range(3)]
    handles.append(plt.Rectangle((0, 0), 1, 1, color=NULL_COLOR))
    fig.legend(handles, [*CLASS_LABELS, "no support"], loc="lower center",
               ncol=4, fontsize=8, frameon=False)
    fig.suptitle("Morphometrics-only priority (geometry; worst-decile per cell; "
                 "rank classes, not absolute)", fontsize=10)
    fig.savefig(FIGS / "priority_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    meta = {
        "git_sha": _git_sha(),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "weights": WEIGHTS, "tertile_breaks": [float(b) for b in breaks],
        "note": "geometry-only; equal weights provisional pending CFD calibration",
    }
    (SIG / "prioritization_meta.json").write_text(json.dumps(meta, indent=2))
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"\nbreaks (pooled p90 tertiles): {[round(b,3) for b in breaks]}")


if __name__ == "__main__":
    main()
