"""Prong A — repair the grid λf product (cell-clipped frontal area).

Reads each site's existing `outputs/<site>/morphometrics/grid/grid_metrics.gpkg`,
recomputes the 8-direction λf columns using the *clipped* projection in
`src.morphometry.indicators.compute_lambda_f_directional`, writes
`grid_metrics_v2.gpkg` + `grid_metrics_v2.csv` alongside, and a stats JSON
to `outputs/brisa_ventilation_fix/lambda_f_v2_stats.json`.

This is intentionally a surgical recompute (just λf), not a full grid
regeneration — BCR, FAR, SVF, slope, etc. are unchanged.

Threshold choices (cited in the stats JSON):
    λf = 0.40 — WIF→SF transition lower bound, Macdonald et al. 1998 (primary)
    λf = 0.50 — WIF→SF transition upper bound, Macdonald et al. 1998
    λf = 0.35 — legacy paper threshold (kept for comparison; not defensible)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import geopandas as gpd
import numpy as np
import pandas as pd

from src.morphometry.indicators import compute_lambda_f_directional

SITES = {
    "vidigal": "Vidigal",
    "rocinha": "Rocinha",
    "complexo_do_alemao": "Complexo do Alemão",
    "riodaspedras": "Rio das Pedras",
    "maré": "Maré",
}

THRESHOLDS = {
    "macdonald_low": 0.40,
    "macdonald_high": 0.50,
    "legacy_paper": 0.35,
}

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "brisa_ventilation_fix"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _site_buildings(site: str) -> gpd.GeoDataFrame:
    path = ROOT / "data" / site / "buildings_extended_300m.gpkg"
    b = gpd.read_file(path)
    if "height" not in b.columns and "altura" in b.columns:
        b["height"] = b["altura"]
    invalid = ~b.geometry.is_valid
    if invalid.any():
        b.loc[invalid, "geometry"] = b.loc[invalid, "geometry"].buffer(0)
    return b


def _distribution(values: pd.Series) -> dict:
    s = values.dropna()
    if s.empty:
        return {"n": 0}
    return {
        "n": int(s.size),
        "min": float(s.min()),
        "p1": float(s.quantile(0.01)),
        "p5": float(s.quantile(0.05)),
        "p25": float(s.quantile(0.25)),
        "p50": float(s.quantile(0.50)),
        "p75": float(s.quantile(0.75)),
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "std": float(s.std()),
    }


def _exceedance(values: pd.Series) -> dict:
    s = values.dropna()
    if s.empty:
        return {k: None for k in THRESHOLDS}
    return {
        name: float((s > thr).mean()) for name, thr in THRESHOLDS.items()
    }


def _correlations(df: pd.DataFrame, cols: list[str]) -> dict:
    out = {}
    for col in cols:
        if col not in df.columns:
            continue
        valid = df[["lambda_f_mean_v2", col]].dropna()
        if len(valid) < 5:
            out[col] = None
        else:
            out[col] = float(valid.corr().iloc[0, 1])
    return out


def process_site(site: str, label: str) -> dict:
    grid_path = ROOT / "outputs" / site / "morphometrics" / "grid" / "grid_metrics.gpkg"
    zones = gpd.read_file(grid_path)
    # Restore zone_area + zone_id required by the indicator functions.
    if "zone_area" not in zones.columns:
        zones["zone_area"] = zones.geometry.area
    if "zone_id" not in zones.columns:
        zones["zone_id"] = np.arange(len(zones))

    buildings = _site_buildings(site)
    # Reproject buildings if needed.
    if buildings.crs != zones.crs:
        buildings = buildings.to_crs(zones.crs)

    recomputed = compute_lambda_f_directional(buildings, zones, clip_to_zone=True)
    direction_cols = [f"lambda_f_{d}" for d in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]]

    # Keep the original (broken) columns as `_v1` so downstream code can compare.
    rename = {c: f"{c}_v1" for c in direction_cols + ["lambda_f_mean", "lambda_f_max"]
              if c in zones.columns}
    zones_v2 = zones.rename(columns=rename).copy()

    for c in direction_cols:
        zones_v2[c + "_v2"] = recomputed[c]
    zones_v2["lambda_f_mean_v2"] = recomputed["lambda_f_mean"]
    zones_v2["lambda_f_max_v2"] = recomputed["lambda_f_max"]

    # Built cells = zones with lambda_p > 0 (matches plan-doc accounting).
    built = zones_v2[zones_v2.get("lambda_p", pd.Series(np.zeros(len(zones_v2)))) > 0].copy()

    dist_v1 = _distribution(built.get("lambda_f_mean_v1", pd.Series(dtype=float)))
    dist_v2 = _distribution(built["lambda_f_mean_v2"])
    exc_v1 = _exceedance(built.get("lambda_f_mean_v1", pd.Series(dtype=float)))
    exc_v2 = _exceedance(built["lambda_f_mean_v2"])
    corrs_v2 = _correlations(built, ["building_count", "lambda_p", "H_mean"])

    out_dir = ROOT / "outputs" / site / "morphometrics" / "grid"
    csv_path = out_dir / "grid_metrics_v2.csv"
    gpkg_path = out_dir / "grid_metrics_v2.gpkg"
    pd.DataFrame(zones_v2.drop(columns="geometry")).to_csv(csv_path, index=False)
    zones_v2.to_file(gpkg_path, driver="GPKG")

    # -- Coarse-grid aggregation (50 m, 100 m) ------------------------------
    # Aggregating mean-of-clipped λf over an N×N block of 10 m cells equals
    # the clipped λf computed on the N·10 m macro-cell (clipped facade area
    # is conservatively partitioned across the 10 m grid).
    coarse_stats: dict[str, dict] = {}
    cx = zones_v2.geometry.centroid.x.values
    cy = zones_v2.geometry.centroid.y.values
    lp = zones_v2.get("lambda_p", pd.Series(np.zeros(len(zones_v2)))).values
    lf = zones_v2["lambda_f_mean_v2"].values
    for size in (50.0, 100.0):
        ix = np.floor(cx / size).astype(np.int64)
        iy = np.floor(cy / size).astype(np.int64)
        # Use lambda_p as a "built fraction" within the macro-cell and weight by it
        # to keep the macro-cell λf comparable to a clipped facade-area density.
        df = pd.DataFrame({"ix": ix, "iy": iy, "lf": lf, "lp": lp})
        # Built macro-cell: at least one constituent cell built (lp>0).
        agg = df.groupby(["ix", "iy"]).agg(
            lf_mean=("lf", "mean"), lp_mean=("lp", "mean"), n_cells=("lf", "size")
        )
        macro_built = agg[agg["lp_mean"] > 0]
        coarse_stats[f"{int(size)}m"] = {
            "n_macro_cells": int(len(agg)),
            "n_macro_built": int(len(macro_built)),
            "distribution": _distribution(macro_built["lf_mean"]),
            "exceedance": _exceedance(macro_built["lf_mean"]),
        }

    return {
        "site": site,
        "label": label,
        "n_cells": int(len(zones_v2)),
        "n_built_cells": int(len(built)),
        "distribution_v1_broken": dist_v1,
        "distribution_v2_clipped": dist_v2,
        "exceedance_v1_broken": exc_v1,
        "exceedance_v2_clipped": exc_v2,
        "corr_lambda_f_v2_vs": corrs_v2,
        "coarse_aggregations": coarse_stats,
        "grid_metrics_v2_csv": str(csv_path.relative_to(ROOT)),
        "grid_metrics_v2_gpkg": str(gpkg_path.relative_to(ROOT)),
    }


def main() -> None:
    results = []
    for site, label in SITES.items():
        print(f"[+] {label}")
        try:
            results.append(process_site(site, label))
            d = results[-1]["distribution_v2_clipped"]
            ev2 = results[-1]["exceedance_v2_clipped"]
            print(
                f"    n_built={results[-1]['n_built_cells']}  "
                f"median(v2)={d.get('p50', float('nan')):.3f}  "
                f"max(v2)={d.get('max', float('nan')):.2f}  "
                f"%>0.40={100*ev2['macdonald_low']:.1f}  "
                f"%>0.50={100*ev2['macdonald_high']:.1f}"
            )
        except Exception as e:
            print(f"    FAILED: {e}")
            results.append({"site": site, "label": label, "error": str(e)})

    summary = {
        "thresholds": THRESHOLDS,
        "citations": {
            "macdonald_low": "Macdonald, Griffiths & Hall 1998 — WIF→SF transition λf ≈ 0.4–0.5",
            "legacy_paper": "Oke 1988 H/W ≈ 0.35 (mis-cited as a λf threshold)",
        },
        "sites": results,
    }
    out_path = OUT_DIR / "lambda_f_v2_stats.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
