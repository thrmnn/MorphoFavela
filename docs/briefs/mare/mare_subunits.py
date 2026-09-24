#!/usr/bin/env python3
"""Per-neighbourhood (subunit) Maré morphology.

Labels every grid cell / WP-04 ground point with the community it falls in
(src.sites.territory.label_subunits — 15 communities + "between
communities"; Marcílio Dias is excluded upstream by label_subunits itself,
since it contributes no area to the study area) and aggregates two
independent sources of record:

- density/height/footprint/sky-view: outputs/{site}/morphometrics/grid/
  grid_metrics.gpkg (the same file docs/briefs/mare/render_figures.py maps) —
  10 m grid, study-area clipped.
- winter sun / annual irradiation: the WP-04 study-area run's ground.parquet
  (runs/wp04_mare_studyarea_*/maré/ground.parquet) — 1 m ray-cast/sun-
  position points, the higher-fidelity source the brief's existing
  street-point figures do not carry per-community.

Also fits a Maré-internal fabric clustering: the campaign's fabric-vector
clustering (src/morphometry/morphotope.py) pools cell-composition vectors
across all five sites; this refits the same GMM restricted to Maré's own
built cells only, with k chosen by BIC (never inherited from the campaign
fit), and reports the result honestly — including if one group still
dominates.

Both are descriptive only: no favela-vs-conjunto deficit framing, and rows
are never ranked (report order is fixed north-to-south, by mean cell y).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("MORPHOFAVELA_ROOT", HERE.parent.parent.parent))
sys.path.insert(0, str(REPO_ROOT))

from src.brisa_solar import mare_study_area as msa  # noqa: E402
from src.sites.territory import BETWEEN_SUBUNITS_LABEL, load_territory, label_subunits  # noqa: E402
from src.morphometry.morphotope import (  # noqa: E402
    N_TYPES,
    fit_morphotopes,
    neighborhood_composition,
    select_k_bic,
)

# load_territory() reads data/ (gitignored: absent from a worktree
# checkout). msa.ROOT is the hardcoded main checkout, not REPO_ROOT — same
# reason collect_numbers.py's _study_area() uses it instead of REPO_ROOT.
_TERRITORY_ROOT = msa.ROOT

SITE = "maré"

GRID_METRICS = ["lambda_p", "far", "H_mean", "sigma_h", "porosity", "svf"]


class MissingSource(Exception):
    pass


def latest_wp04_studyarea_run(runs_root: Path) -> Path:
    candidates = sorted(runs_root.glob("wp04_mare_studyarea_*"))
    if not candidates:
        raise MissingSource(f"no wp04_mare_studyarea_* run under {runs_root}")
    return candidates[-1]


def _grid_by_subunit(outputs_root: Path, territory) -> pd.DataFrame:
    path = outputs_root / SITE / "morphometrics" / "grid" / "grid_metrics.gpkg"
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


def _solar_by_subunit(run_dir: Path, territory) -> pd.DataFrame:
    path = run_dir / SITE / "ground.parquet"
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


def build_subunit_table(outputs_root: Path, runs_root: Path) -> pd.DataFrame:
    """One row per subunit (15 communities + BETWEEN_SUBUNITS_LABEL), ordered
    north to south (descending mean grid-cell y — Maré's study area sits in a
    UTM zone, so y increases due north; this is a geographic fact, not a
    ranking choice)."""
    territory = load_territory(SITE, root=_TERRITORY_ROOT)
    grid_tbl = _grid_by_subunit(outputs_root, territory)
    run_dir = latest_wp04_studyarea_run(runs_root)
    solar_tbl = _solar_by_subunit(run_dir, territory)
    table = grid_tbl.merge(solar_tbl, on="name", how="left")
    table = table.sort_values("mean_y", ascending=False).reset_index(drop=True)
    table.attrs["wp04_run"] = run_dir.name
    return table


def _bic_elbow_k(bic: pd.DataFrame) -> int:
    """Geometric elbow (max perpendicular distance from the chord joining the
    curve's endpoints — the standard 'kneedle' heuristic) over the tested
    k-range. Used because src.morphometry.morphotope.select_k_bic's BIC does
    not have an interior minimum on Maré-internal data (verified: strictly
    decreasing k=2..15) — a plain argmin would just pick the range's upper
    bound, which is a range artefact, not a selection."""
    ks = bic["k"].to_numpy(dtype=float)
    vals = bic["bic"].to_numpy(dtype=float)
    x0, y0, x1, y1 = ks[0], vals[0], ks[-1], vals[-1]
    seg = np.hypot(x1 - x0, y1 - y0)
    if seg == 0:
        return int(ks[0])
    dist = np.abs((y1 - y0) * ks - (x1 - x0) * vals + x1 * y0 - y1 * x0) / seg
    return int(ks[int(np.argmax(dist))])


def fit_within_mare_clusters(outputs_root: Path, krange=range(2, 9)) -> dict:
    """Refit the campaign's fabric-vector (tissue) clustering restricted to
    Maré's own built cells only — never pooled with the other four campaign
    sites, and never the campaign's inherited k=6. `krange` matches
    select_k_bic's own default (2..8) for direct methodological parity with
    the campaign fit.

    BIC on this Maré-internal composition data is strictly decreasing over
    the tested range (no interior minimum: reported as `bic_monotonic`) —
    an argmin would just return the range's upper bound, a range artefact
    rather than a selection. k is instead chosen by the geometric elbow of
    the BIC curve (`_bic_elbow_k`), and the honesty check itself is kept in
    the returned dict rather than silently resolved.

    Returns k, the full BIC curve, and the resulting cluster shares — which
    may still be dominated by one group; reported as computed, not
    adjusted."""
    path = outputs_root / SITE / "features" / "features_grid.parquet"
    if not path.exists():
        raise MissingSource(f"missing {path}")
    df = pd.read_parquet(path, columns=[
        "zone_id", "centroid_x", "centroid_y", "built_mask", "morphotype_smooth",
    ])
    df = df.loc[df["built_mask"] & df["morphotype_smooth"].notna()].copy()
    df["site"] = SITE
    comp = neighborhood_composition(df, radius=50.0, label_col="morphotype_smooth")

    bic = select_k_bic(comp, krange=krange)
    bic_monotonic = bool((bic["bic"].diff().dropna() < 0).all())
    argmin_k = int(bic.loc[bic["bic"].idxmin(), "k"])
    k = _bic_elbow_k(bic) if bic_monotonic else argmin_k
    labels = fit_morphotopes(comp, k=k)
    shares = pd.Series(labels).value_counts(normalize=True).sort_index()

    return {
        "n_cells": int(len(comp)),
        "k_range": [int(bic["k"].min()), int(bic["k"].max())],
        "bic_monotonic": bic_monotonic,
        "bic_argmin_k": argmin_k,
        "k_selection_method": "bic_elbow (no interior BIC minimum in range)" if bic_monotonic else "bic_argmin",
        "k_selected": k,
        "bic_curve": [{"k": int(r.k), "bic": float(r.bic)} for r in bic.itertuples()],
        "shares_pct": {str(i): 100 * float(shares.get(i, 0.0)) for i in range(k)},
        "dominant_group": int(shares.idxmax()),
        "dominant_share_pct": 100 * float(shares.max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-root", type=Path, required=True)
    ap.add_argument("--runs-root", type=Path, default=REPO_ROOT / "runs")
    ap.add_argument("--out-csv", type=Path, default=HERE / "mare_subunits.csv")
    ap.add_argument("--out-clusters-json", type=Path, default=HERE / "mare_fabric_within.json")
    args = ap.parse_args()

    table = build_subunit_table(args.outputs_root, args.runs_root)
    table.to_csv(args.out_csv, index=False)
    print(f"mare_subunits: wrote {len(table)} rows -> {args.out_csv}")

    clusters = fit_within_mare_clusters(args.outputs_root)
    args.out_clusters_json.write_text(json.dumps(clusters, indent=2) + "\n")
    print(f"mare_subunits: k={clusters['k_selected']} "
          f"(dominant group {clusters['dominant_group']}, "
          f"{clusters['dominant_share_pct']:.1f}%) -> {args.out_clusters_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
