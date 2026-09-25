"""Per-subunit morphology — site-agnostic core.

Generalises the table half of `docs/briefs/mare/mare_subunits.py` (Maré's
per-neighbourhood morphology brief module) so any campaign site that
declares `subunits` in `config/sites.yaml` (src.sites.territory) can get
the same descriptive breakdown: per-subunit density, height, footprint and
sky-view (from the site's WP-06 grid) plus winter-sun and annual-irradiation
medians (from a WP-04 ground-point run) — geographic order (north to south
by mean grid-cell y), rows never ranked, `BETWEEN_SUBUNITS_LABEL` kept for
study-area ground inside no named subunit.

This module deliberately does NOT carry Maré's fabric-clustering refit
(`docs/briefs/mare/mare_subunits.py`'s `fit_within_mare_clusters`) — that
GMM-on-morphotype-composition analysis was scoped to Maré specifically and
stays there; nothing in this generalisation asked for it, and adding it
elsewhere would be unrequested scope, not a generalisation of what's here.

Callers: `scripts/build_subunit_morphology.py` (the CLI) and
`scripts/build_site_pages.py` (reads the CLI's run output to render a panel
per site page). `docs/briefs/mare/mare_subunits.py` is left untouched —
Maré keeps its own dedicated script; this module does not run for Maré.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from src.sites.territory import Territory, label_subunits, load_territory

#: The grid_metrics.gpkg columns this module reads besides
#: building_count/centroid_x/centroid_y — same set docs/briefs/mare's
#: mare_subunits.py used (density, height, footprint, sky-view).
GRID_METRICS = ["lambda_p", "far", "H_mean", "sigma_h", "porosity", "svf"]


class MissingSource(Exception):
    pass


def latest_wp04_run_for_site(runs_root: Path, site: str) -> Path:
    """The newest `runs/wp04_*` run directory that contains `<site>/
    ground.parquet` — generalises mare_subunits.py's
    `latest_wp04_studyarea_run` (which only ever looked at
    `wp04_mare_studyarea_*`) to whichever WP-04 run actually covers this
    site. Run directories sort lexicographically by their trailing UTC
    timestamp, so the last match is the newest, exactly like the Maré-only
    version did within its own narrower glob."""
    candidates = [d for d in sorted(runs_root.glob("wp04_*"))
                 if d.is_dir() and (d / site / "ground.parquet").exists()]
    if not candidates:
        raise MissingSource(f"no runs/wp04_*/{site}/ground.parquet under {runs_root}")
    return candidates[-1]


def grid_by_subunit(outputs_root: Path, site: str, territory: Territory) -> pd.DataFrame:
    """Density/height/footprint/sky-view per subunit, from
    outputs/<site>/morphometrics/grid/grid_metrics.gpkg (WP-06's 10 m grid,
    study-area clipped) — same source and same median-of-built-cells
    convention docs/briefs/mare/mare_subunits.py's `_grid_by_subunit` used."""
    path = outputs_root / site / "morphometrics" / "grid" / "grid_metrics.gpkg"
    if not path.exists():
        raise MissingSource(f"missing {path}")
    grid = gpd.read_file(path)
    labels = label_subunits(grid["centroid_x"].to_numpy(), grid["centroid_y"].to_numpy(), territory)
    grid = grid.assign(subunit=labels)
    grid = grid.loc[grid["subunit"].notna()].copy()
    built = grid[grid["building_count"] > 0]

    rows = []
    for name, g in grid.groupby("subunit"):
        b = built[built["subunit"] == name]
        row = {"name": name, "n_cells": int(len(g)), "n_built_cells": int(len(b)),
               "mean_y": float(g["centroid_y"].mean())}
        row["lambda_p_median"] = float(g["lambda_p"].median())
        row["svf_median"] = float(g["svf"].median())
        for col in ("far", "H_mean", "sigma_h", "porosity"):
            row[f"{col}_median"] = float(b[col].median()) if len(b) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def solar_by_subunit(run_dir: Path, site: str, territory: Territory) -> pd.DataFrame:
    """Winter-sun-hours and annual-irradiation medians per subunit, from
    <run_dir>/<site>/ground.parquet (1 m ray-cast/sun-position points) —
    same source and column set docs/briefs/mare/mare_subunits.py's
    `_solar_by_subunit` used."""
    path = run_dir / site / "ground.parquet"
    if not path.exists():
        raise MissingSource(f"missing {path}")
    pts = pd.read_parquet(path, columns=["x", "y", "hours_winter_solstice", "kwh_m2"])
    labels = label_subunits(pts["x"].to_numpy(), pts["y"].to_numpy(), territory)
    pts = pts.assign(subunit=labels)
    pts = pts.loc[pts["subunit"].notna()]
    agg = pts.groupby("subunit").agg(
        sun_winter_median_h=("hours_winter_solstice", "median"),
        kwh_m2_median=("kwh_m2", "median"),
        n_solar_points=("hours_winter_solstice", "size"),
    )
    return agg.reset_index().rename(columns={"subunit": "name"})


def build_subunit_table(site: str, outputs_root: Path, runs_root: Path,
                        root: Path | None = None) -> pd.DataFrame:
    """One row per subunit declared for `site` (config/sites.yaml) plus
    BETWEEN_SUBUNITS_LABEL, ordered geographically north to south
    (descending mean grid-cell y — a geographic fact, never a ranking
    choice, same convention as the Maré-only version). Raises
    MissingSource if `site` has no subunits declared, or if either source
    is absent on this checkout."""
    territory = load_territory(site, root=root) if root is not None else load_territory(site)
    if territory.subunits is None:
        raise MissingSource(f"{site}: no subunits declared in config/sites.yaml — nothing to break down")
    grid_tbl = grid_by_subunit(outputs_root, site, territory)
    run_dir = latest_wp04_run_for_site(runs_root, site)
    solar_tbl = solar_by_subunit(run_dir, site, territory)
    table = grid_tbl.merge(solar_tbl, on="name", how="left")
    table = table.sort_values("mean_y", ascending=False).reset_index(drop=True)
    table.attrs["wp04_run"] = run_dir.name
    return table
