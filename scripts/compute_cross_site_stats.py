#!/usr/bin/env python3
"""Cross-site diagnostic statistics for the BRISA Nature Cities paper.

Reads per-site grid_metrics + svf_streets_solar, runs the four-state
classification, and emits a single JSON consolidating every number the
manuscript needs:

    - per-site state shares + counts
    - hillside vs flatland aggregates
    - cross-site SVF / lambda_p / lambda_f ranges
    - sigma_H ceiling per site (low-rise check)
    - slope-Pearson independence (slope vs SVF/λp/λf/porosity/σH/entropy)
    - sunlight-fail % cross-site range
    - lambda_p ↔ porosity redundancy check

Output:
    outputs/paper_figures/cross_site_stats.json
    outputs/paper_figures/cross_site_stats.md   (human-readable summary)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SITES = ["vidigal", "rocinha", "complexo_do_alemao", "maré", "riodaspedras"]
HILLSIDE = {"vidigal", "rocinha", "complexo_do_alemao"}
FLATLAND = {"maré", "riodaspedras"}

THRESHOLD_SUN_HRS = 2.0
THRESHOLD_LAMBDA_F = 0.35


def aggregate_solar(grid: gpd.GeoDataFrame, solar_path: Path,
                    max_radius: float = 25.0, k: int = 3) -> gpd.GeoDataFrame:
    sol = gpd.read_file(solar_path)[["solar_hours_winter", "geometry"]]
    j = gpd.sjoin(sol, grid[["zone_id", "geometry"]], how="inner", predicate="within")
    primary = j.groupby("zone_id")["solar_hours_winter"].median().reset_index()
    grid = grid.merge(primary, on="zone_id", how="left")
    missing = grid["solar_hours_winter"].isna()
    if missing.any():
        sol_pts = np.array([(g.x, g.y) for g in sol.geometry])
        sol_vals = sol["solar_hours_winter"].to_numpy()
        tree = cKDTree(sol_pts)
        cents = grid.loc[missing, ["centroid_x", "centroid_y"]].to_numpy()
        dists, idxs = tree.query(cents, k=k, distance_upper_bound=max_radius)
        with np.errstate(invalid="ignore"):
            valid = np.isfinite(dists)
        idxs = np.where(valid, idxs, 0)
        vals = sol_vals[idxs]
        vals = np.where(valid, vals, np.nan)
        with np.errstate(invalid="ignore"):
            cell_vals = np.nanmedian(vals, axis=1)
        grid.loc[missing, "solar_hours_winter"] = cell_vals
    return grid


def classify(grid: gpd.GeoDataFrame) -> pd.Series:
    sun = grid["solar_hours_winter"]
    vent = grid["lambda_f_mean"]
    built = (grid["lambda_p"].fillna(0) > 0.01) | (grid["building_count"] > 0)
    sun_known = sun.notna()
    vent_known = vent.notna()
    both = built & sun_known & vent_known
    state = pd.Series("nodata", index=grid.index, dtype=object)
    state[both & (sun >= THRESHOLD_SUN_HRS) & (vent <= THRESHOLD_LAMBDA_F)] = "adequate"
    state[both & (sun <  THRESHOLD_SUN_HRS) & (vent <= THRESHOLD_LAMBDA_F)] = "sun_only"
    state[both & (sun >= THRESHOLD_SUN_HRS) & (vent >  THRESHOLD_LAMBDA_F)] = "vent_only"
    state[both & (sun <  THRESHOLD_SUN_HRS) & (vent >  THRESHOLD_LAMBDA_F)] = "compound"
    return state


def percentile_or_none(s: pd.Series, q: float) -> float | None:
    s = s.dropna()
    if len(s) == 0:
        return None
    return float(np.nanpercentile(s, q))


def quick_pearson(a: pd.Series, b: pd.Series) -> float | None:
    """Pearson r, ignoring NaNs and zero-variance edge cases."""
    mask = a.notna() & b.notna()
    if mask.sum() < 50:
        return None
    if a[mask].std() == 0 or b[mask].std() == 0:
        return None
    try:
        r, _ = pearsonr(a[mask], b[mask])
        return float(r)
    except Exception:
        return None


def stats_for_site(site: str) -> dict[str, Any]:
    grid_path = PROJECT_ROOT / "outputs" / site / "morphometrics" / "grid" / "grid_metrics.gpkg"
    solar_path = PROJECT_ROOT / "outputs" / site / "morphometrics" / "svf" / "svf_streets_solar.gpkg"
    grid = gpd.read_file(grid_path)
    grid = aggregate_solar(grid, solar_path)
    state = classify(grid)
    built_mask = state != "nodata"
    built = grid[built_mask].copy()
    built["state"] = state[built_mask]

    counts = built["state"].value_counts().to_dict()
    total = int(built_mask.sum())
    shares = {k: counts.get(k, 0) / total if total else 0.0
              for k in ("adequate", "sun_only", "vent_only", "compound")}

    descriptors = {}
    for col in ("svf", "lambda_p", "lambda_f_mean", "porosity", "H_mean", "sigma_h", "slope_deg",
                "street_orientation_entropy"):
        if col in built.columns:
            s = built[col].dropna()
            descriptors[col] = {
                "min": float(s.min()) if len(s) else None,
                "p05": percentile_or_none(s, 5),
                "p25": percentile_or_none(s, 25),
                "median": percentile_or_none(s, 50),
                "p75": percentile_or_none(s, 75),
                "p95": percentile_or_none(s, 95),
                "max": float(s.max()) if len(s) else None,
                "mean": float(s.mean()) if len(s) else None,
                "std": float(s.std()) if len(s) else None,
                "n": int(len(s)),
            }

    slope_corr = {}
    if "slope_deg" in built.columns:
        for other in ("svf", "lambda_p", "lambda_f_mean", "porosity", "sigma_h", "street_orientation_entropy"):
            if other in built.columns:
                slope_corr[other] = quick_pearson(built["slope_deg"], built[other])
    lambda_porosity_corr = (
        quick_pearson(built["lambda_p"], built["porosity"])
        if {"lambda_p", "porosity"}.issubset(built.columns) else None
    )

    pct_below_2h = None
    if "solar_hours_winter" in built.columns:
        s = built["solar_hours_winter"].dropna()
        if len(s):
            pct_below_2h = float((s < THRESHOLD_SUN_HRS).mean())

    pct_lambda_f_above = None
    if "lambda_f_mean" in built.columns:
        s = built["lambda_f_mean"].dropna()
        if len(s):
            pct_lambda_f_above = float((s > THRESHOLD_LAMBDA_F).mean())

    return {
        "site": site,
        "typology": "hillside" if site in HILLSIDE else "flatland",
        "n_built_cells": total,
        "counts": {k: counts.get(k, 0) for k in ("adequate", "sun_only", "vent_only", "compound")},
        "shares": shares,
        "pct_sun_below_2h": pct_below_2h,
        "pct_lambda_f_above_0_35": pct_lambda_f_above,
        "descriptors": descriptors,
        "slope_pearson_abs_max": (
            max((abs(v) for v in slope_corr.values() if v is not None), default=None)
        ),
        "slope_pearson": slope_corr,
        "lambda_p_porosity_pearson": lambda_porosity_corr,
    }


def aggregate(per_site: list[dict[str, Any]]) -> dict[str, Any]:
    def pooled(group_sites: set[str]) -> dict[str, Any]:
        total = sum(s["n_built_cells"] for s in per_site if s["site"] in group_sites)
        counts = {k: sum(s["counts"][k] for s in per_site if s["site"] in group_sites)
                  for k in ("adequate", "sun_only", "vent_only", "compound")}
        shares = {k: (v / total if total else 0.0) for k, v in counts.items()}
        return {"n_built_cells": total, "counts": counts, "shares": shares}

    all_sites = {s["site"] for s in per_site}
    return {
        "all": pooled(all_sites),
        "hillside": pooled(HILLSIDE),
        "flatland": pooled(FLATLAND),
    }


def cross_site_ranges(per_site: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("svf", "lambda_p", "lambda_f_mean", "porosity", "sigma_h", "slope_deg"):
        mins, maxs, p05, p95 = [], [], [], []
        for s in per_site:
            d = s["descriptors"].get(key)
            if d is None:
                continue
            mins.append(d["min"]); maxs.append(d["max"])
            p05.append(d["p05"]); p95.append(d["p95"])
        if mins:
            out[key] = {
                "site_min_min": float(min(mins)),
                "site_max_max": float(max(maxs)),
                "site_p05_min": float(min(p05)),
                "site_p95_max": float(max(p95)),
            }
    out["pct_sun_below_2h_range"] = {
        "min": float(min(s["pct_sun_below_2h"] for s in per_site if s["pct_sun_below_2h"] is not None)),
        "max": float(max(s["pct_sun_below_2h"] for s in per_site if s["pct_sun_below_2h"] is not None)),
    }
    out["pct_lambda_f_above_0_35_range"] = {
        "min": float(min(s["pct_lambda_f_above_0_35"] for s in per_site if s["pct_lambda_f_above_0_35"] is not None)),
        "max": float(max(s["pct_lambda_f_above_0_35"] for s in per_site if s["pct_lambda_f_above_0_35"] is not None)),
    }
    return out


def write_summary_md(stats: dict[str, Any], path: Path) -> None:
    lines = ["# Cross-site diagnostic statistics", ""]
    lines.append(f"_Computed from {len(stats['per_site'])} sites at 10 m grid resolution._")
    lines.append("")
    lines.append("## State shares per site")
    lines.append("")
    lines.append("| Site | Typology | Built cells | Adequate | Sun-only | Vent-only | Compound |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for s in stats["per_site"]:
        sh = s["shares"]
        lines.append(
            f"| {s['site']} | {s['typology']} | {s['n_built_cells']:,} | "
            f"{sh['adequate']*100:.1f}% | {sh['sun_only']*100:.1f}% | "
            f"{sh['vent_only']*100:.1f}% | {sh['compound']*100:.1f}% |"
        )
    lines.append("")
    lines.append("## Aggregates")
    lines.append("")
    for k in ("all", "hillside", "flatland"):
        a = stats["aggregates"][k]
        sh = a["shares"]
        lines.append(
            f"- **{k}**: {a['n_built_cells']:,} cells — "
            f"adequate {sh['adequate']*100:.1f}%, sun-only {sh['sun_only']*100:.1f}%, "
            f"vent-only {sh['vent_only']*100:.1f}%, compound {sh['compound']*100:.1f}%"
        )
    lines.append("")
    lines.append("## Cross-site ranges")
    lines.append("")
    for key, r in stats["cross_site_ranges"].items():
        if isinstance(r, dict) and "site_min_min" in r:
            lines.append(f"- **{key}**: range across sites = "
                         f"[{r['site_min_min']:.3f}, {r['site_max_max']:.3f}]  "
                         f"(P5–P95 envelope = [{r['site_p05_min']:.3f}, {r['site_p95_max']:.3f}])")
        elif isinstance(r, dict) and "min" in r:
            lines.append(f"- **{key}**: per-site range = "
                         f"[{r['min']*100:.1f}%, {r['max']*100:.1f}%]")
    lines.append("")
    lines.append("## Slope independence (Pearson r vs slope, by site)")
    lines.append("")
    lines.append("| Site | r(slope, SVF) | r(slope, λp) | r(slope, λf) | r(slope, porosity) | r(slope, σH) | r(slope, entropy) | abs-max |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in stats["per_site"]:
        sp = s["slope_pearson"]
        def fmt(v): return "—" if v is None else f"{v:+.3f}"
        lines.append(
            f"| {s['site']} | {fmt(sp.get('svf'))} | {fmt(sp.get('lambda_p'))} | "
            f"{fmt(sp.get('lambda_f_mean'))} | {fmt(sp.get('porosity'))} | "
            f"{fmt(sp.get('sigma_h'))} | {fmt(sp.get('street_orientation_entropy'))} | "
            f"{fmt(s['slope_pearson_abs_max'])} |"
        )
    lines.append("")
    lines.append("## λp ↔ porosity redundancy")
    lines.append("")
    for s in stats["per_site"]:
        v = s["lambda_p_porosity_pearson"]
        v_s = "—" if v is None else f"{v:+.3f}"
        lines.append(f"- {s['site']}: r = {v_s}")
    path.write_text("\n".join(lines))


def main() -> None:
    per_site = [stats_for_site(s) for s in SITES]
    stats = {
        "thresholds": {
            "sun_hours_winter_min": THRESHOLD_SUN_HRS,
            "lambda_f_max": THRESHOLD_LAMBDA_F,
        },
        "per_site": per_site,
        "aggregates": aggregate(per_site),
        "cross_site_ranges": cross_site_ranges(per_site),
    }
    out_dir = PROJECT_ROOT / "outputs" / "paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cross_site_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    write_summary_md(stats, out_dir / "cross_site_stats.md")
    print(f"wrote {out_dir/'cross_site_stats.json'}")
    print(f"wrote {out_dir/'cross_site_stats.md'}")


if __name__ == "__main__":
    main()
