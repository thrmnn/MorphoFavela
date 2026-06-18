"""Prong A — repair the grid λf product (cell-clipped frontal area).

Overwrites each site's `outputs/<site>/morphometrics/grid/grid_metrics.{csv,gpkg}`
in place with the *clipped* 8-direction λf columns. The pre-overwrite
distribution of the broken values is captured into the stats JSON for the
paper trail.

This is a surgical recompute (only the eight λf direction columns plus
mean/max): BCR, FAR, SVF, slope, etc. are unchanged. Any leftover columns
from earlier `_v1`/`_v2` shadow products are dropped.

Threshold choices (cited in the stats JSON):
    λf = 0.40 — WIF→SF transition lower bound, Macdonald et al. 1998 (primary)
    λf = 0.50 — WIF→SF transition upper bound, Macdonald et al. 1998
    λf = 0.35 — legacy paper threshold (kept for comparison; NOT a defensible
                λf threshold — it's the Oke 1988 H/W aspect-ratio boundary)
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

DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
LF_COLS = [f"lambda_f_{d}" for d in DIRECTIONS] + ["lambda_f_mean", "lambda_f_max"]

OUT_DIR = _ROOT / "outputs" / "brisa_ventilation_fix"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _site_buildings(site: str) -> gpd.GeoDataFrame:
    b = gpd.read_file(_ROOT / "data" / site / "buildings_extended_300m.gpkg")
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
        return dict.fromkeys(THRESHOLDS)
    return {name: float((s > thr).mean()) for name, thr in THRESHOLDS.items()}


def _correlations(df: pd.DataFrame, target: str, cols: list[str]) -> dict:
    out = {}
    for col in cols:
        if col not in df.columns:
            continue
        valid = df[[target, col]].dropna()
        out[col] = float(valid.corr().iloc[0, 1]) if len(valid) >= 5 else None
    return out


def process_site(site: str, label: str) -> dict:
    grid_path = _ROOT / "outputs" / site / "morphometrics" / "grid" / "grid_metrics.gpkg"
    csv_path = grid_path.with_suffix(".csv")
    legacy_v2_gpkg = grid_path.with_name("grid_metrics_v2.gpkg")
    legacy_v2_csv = grid_path.with_name("grid_metrics_v2.csv")

    zones = gpd.read_file(grid_path)
    if "zone_area" not in zones.columns:
        zones["zone_area"] = zones.geometry.area
    if "zone_id" not in zones.columns:
        zones["zone_id"] = np.arange(len(zones))

    # Drop any leftover *_v1/_v2 shadow columns from earlier iterations.
    drop_cols = [c for c in zones.columns if c.endswith(("_v1", "_v2"))]
    if drop_cols:
        zones = zones.drop(columns=drop_cols)

    # Capture the broken distribution BEFORE overwriting.
    built_mask = zones.get("lambda_p", pd.Series(np.zeros(len(zones)))) > 0
    pre = zones.loc[built_mask, "lambda_f_mean"] if "lambda_f_mean" in zones.columns else pd.Series(dtype=float)
    dist_broken = _distribution(pre)
    exc_broken = _exceedance(pre)

    buildings = _site_buildings(site)
    if buildings.crs != zones.crs:
        buildings = buildings.to_crs(zones.crs)

    recomputed = compute_lambda_f_directional(buildings, zones, clip_to_zone=True)
    for c in LF_COLS:
        zones[c] = recomputed[c].values

    # Overwrite in place. Drop legacy shadow files if present.
    pd.DataFrame(zones.drop(columns="geometry")).to_csv(csv_path, index=False)
    zones.to_file(grid_path, driver="GPKG")
    for stale in (legacy_v2_gpkg, legacy_v2_csv):
        if stale.exists():
            stale.unlink()

    built = zones.loc[built_mask].copy()
    dist_fixed = _distribution(built["lambda_f_mean"])
    exc_fixed = _exceedance(built["lambda_f_mean"])
    corrs_fixed = _correlations(built, "lambda_f_mean", ["building_count", "lambda_p", "H_mean"])

    # Coarse-grid aggregation (50 m / 100 m macro-cells).
    coarse: dict[str, dict] = {}
    cx = zones.geometry.centroid.x.values
    cy = zones.geometry.centroid.y.values
    lp = zones.get("lambda_p", pd.Series(np.zeros(len(zones)))).values
    lf = zones["lambda_f_mean"].values
    for size in (50.0, 100.0):
        ix = np.floor(cx / size).astype(np.int64)
        iy = np.floor(cy / size).astype(np.int64)
        df = pd.DataFrame({"ix": ix, "iy": iy, "lf": lf, "lp": lp})
        agg = df.groupby(["ix", "iy"]).agg(
            lf_mean=("lf", "mean"), lp_mean=("lp", "mean"), n_cells=("lf", "size")
        )
        macro_built = agg[agg["lp_mean"] > 0]
        coarse[f"{int(size)}m"] = {
            "n_macro_cells": int(len(agg)),
            "n_macro_built": int(len(macro_built)),
            "distribution": _distribution(macro_built["lf_mean"]),
            "exceedance": _exceedance(macro_built["lf_mean"]),
        }

    return {
        "site": site,
        "label": label,
        "n_cells": int(len(zones)),
        "n_built_cells": int(built_mask.sum()),
        "distribution_broken_pre_overwrite": dist_broken,
        "distribution_clipped_in_place": dist_fixed,
        "exceedance_broken_pre_overwrite": exc_broken,
        "exceedance_clipped_in_place": exc_fixed,
        "corr_lambda_f_clipped_vs": corrs_fixed,
        "coarse_aggregations": coarse,
        "grid_metrics_csv": str(csv_path.relative_to(_ROOT)),
        "grid_metrics_gpkg": str(grid_path.relative_to(_ROOT)),
    }


def main() -> None:
    results = []
    for site, label in SITES.items():
        print(f"[+] {label}")
        try:
            r = process_site(site, label)
            results.append(r)
            d = r["distribution_clipped_in_place"]
            db = r["distribution_broken_pre_overwrite"]
            e = r["exceedance_clipped_in_place"]
            print(
                f"    n_built={r['n_built_cells']}  "
                f"median(broken→clipped)={db.get('p50', float('nan')):.2f}→{d.get('p50', float('nan')):.2f}  "
                f"max={d.get('max', float('nan')):.2f}  "
                f"%>0.40={100 * e['macdonald_low']:.1f}  "
                f"%>0.50={100 * e['macdonald_high']:.1f}"
            )
        except Exception as e:
            print(f"    FAILED: {e}")
            results.append({"site": site, "label": label, "error": str(e)})

    payload = {
        "thresholds": THRESHOLDS,
        "citations": {
            "macdonald": "Macdonald, Griffiths & Hall 1998 — WIF→SF transition λf ≈ 0.4–0.5",
            "legacy_misattribution": "Oke 1988 H/W ≈ 0.35 (mis-cited as a λf threshold in the legacy pipeline)",
        },
        "sites": results,
    }
    out_path = OUT_DIR / "lambda_f_repair_stats.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
