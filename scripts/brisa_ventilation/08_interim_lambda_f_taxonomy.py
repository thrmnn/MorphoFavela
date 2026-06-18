"""Definitive interim λf ventilation taxonomy under the corrected
(cell-clipped) λf product.

For each of the five campaign sites this loads:
  - corrected λf from outputs/<site>/morphometrics/grid/grid_metrics.gpkg
  - winter direct-sun hours from svf_streets_solar.gpkg (median-within-cell
    + nearest-k fallback at 25 m, identical to build_diagnostic_map.py)

Built cells are masked as ``(lambda_p > 0.01) | (building_count > 0)``.
A cell is **classifiable** when it is built AND both sun and λf are known
after the fallback. Two count families exist:

  n_built_cells       = built cells regardless of data coverage
  n_classified_cells  = built cells with sun+λf known (used as the
                        canonical share denominator)

The interim taxonomy uses a SINGLE pooled λf threshold across all five
sites, chosen as the percentile of the pooled built-cell distribution
that yields a pooled adequate share in the [target_lo, target_hi] band.
This is an explicitly RELATIVE pre-screen, pending CFD-ACH + LMA.

Output: outputs/brisa_ventilation_fix/taxonomy_interim_lambda_f.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

SITES = [
    ("vidigal", "Vidigal", "hillside"),
    ("rocinha", "Rocinha", "hillside"),
    ("complexo_do_alemao", "Complexo do Alemão", "hillside"),
    ("riodaspedras", "Rio das Pedras", "flatland"),
    ("maré", "Maré", "flatland"),
]

SUN_THR = 2.0
ADEQUATE_TARGET_LO = 0.40
ADEQUATE_TARGET_HI = 0.55

OUT_DIR = _ROOT / "outputs" / "brisa_ventilation_fix"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def aggregate_solar(grid: gpd.GeoDataFrame, sol_path: Path,
                    max_radius: float = 25.0, k: int = 3) -> gpd.GeoDataFrame:
    sol = gpd.read_file(sol_path)[["solar_hours_winter", "geometry"]]
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


def load_site(site: str) -> pd.DataFrame:
    grid_path = _ROOT / "outputs" / site / "morphometrics" / "grid" / "grid_metrics.gpkg"
    sol_path = _ROOT / "outputs" / site / "morphometrics" / "svf" / "svf_streets_solar.gpkg"
    grid = gpd.read_file(grid_path)
    grid = aggregate_solar(grid, sol_path)
    lp = grid.get("lambda_p", pd.Series(np.zeros(len(grid)))).fillna(0)
    bc = grid.get("building_count", pd.Series(np.zeros(len(grid))))
    built = (lp > 0.01) | (bc > 0)
    sun = grid["solar_hours_winter"].astype(float).values
    lf = grid["lambda_f_mean"].astype(float).values
    df = pd.DataFrame({
        "site": site,
        "built": built.values,
        "sun_known": np.isfinite(sun),
        "lf_known": np.isfinite(lf),
        "sun": sun,
        "lf": lf,
    })
    return df


def shares(df: pd.DataFrame, lf_thr: float) -> dict:
    classified = df[df["built"] & df["sun_known"] & df["lf_known"]]
    n_built = int(df["built"].sum())
    n_class = int(len(classified))
    if n_class == 0:
        return {"n_built": n_built, "n_classified": n_class}
    sf = (classified["sun"] < SUN_THR).values
    vf = (classified["lf"] > lf_thr).values
    counts = {
        "adequate": int((~sf & ~vf).sum()),
        "sunlight_constraint": int((sf & ~vf).sum()),
        "ventilation_constraint": int((~sf & vf).sum()),
        "compound_constraint": int((sf & vf).sum()),
    }
    sh = {k: v / n_class for k, v in counts.items()}
    return {
        "n_built": n_built,
        "n_classified": n_class,
        "counts": counts,
        "shares": sh,
    }


def reconcile_built_counts(site_dfs: dict[str, pd.DataFrame]) -> dict:
    """Explain the 56 657 (cross_site_stats) vs 64 389 (taxonomy_shares)
    discrepancy per site, under the corrected mask."""
    rows = {}
    pooled = {"built": 0, "built_sun_known": 0, "built_sun_and_lf_known": 0}
    for site, df in site_dfs.items():
        b = int(df["built"].sum())
        bs = int((df["built"] & df["sun_known"]).sum())
        bsl = int((df["built"] & df["sun_known"] & df["lf_known"]).sum())
        rows[site] = {
            "n_built": b,
            "n_built_with_sun": bs,
            "n_built_with_sun_and_lf": bsl,
            "n_sun_missing_in_built": b - bs,
        }
        pooled["built"] += b
        pooled["built_sun_known"] += bs
        pooled["built_sun_and_lf_known"] += bsl
    return {"per_site": rows, "pooled": pooled}


def main() -> None:
    site_dfs = {}
    for site, _, _ in SITES:
        site_dfs[site] = load_site(site)

    reconcile = reconcile_built_counts(site_dfs)

    # Pool the corrected λf across all built+classified cells.
    pooled_lf = np.concatenate([
        d.loc[d["built"] & d["sun_known"] & d["lf_known"], "lf"].values
        for d in site_dfs.values()
    ])
    pooled_lf = pooled_lf[np.isfinite(pooled_lf)]

    # Search percentiles 50..95 → pick the one giving pooled adequate share
    # within [ADEQUATE_TARGET_LO, ADEQUATE_TARGET_HI]. If no exact hit, pick
    # the percentile that gets adequate closest to the band's midpoint.
    target_mid = (ADEQUATE_TARGET_LO + ADEQUATE_TARGET_HI) / 2
    candidates = []
    for pct in np.arange(50.0, 96.0, 1.0):
        thr = float(np.percentile(pooled_lf, pct))
        per_site = {site: shares(df, thr) for site, df in site_dfs.items()}
        pooled = {
            "adequate": 0, "sunlight_constraint": 0,
            "ventilation_constraint": 0, "compound_constraint": 0,
        }
        n = 0
        for _, r in per_site.items():
            if "counts" not in r:
                continue
            for k in pooled:
                pooled[k] += r["counts"][k]
            n += r["n_classified"]
        pooled_shares = {k: v / n for k, v in pooled.items()} if n else None
        candidates.append({
            "percentile": pct, "threshold": thr,
            "pooled_shares": pooled_shares,
            "pooled_n": n,
        })

    in_band = [
        c for c in candidates
        if c["pooled_shares"] is not None
        and ADEQUATE_TARGET_LO <= c["pooled_shares"]["adequate"] <= ADEQUATE_TARGET_HI
    ]
    if in_band:
        # Pick the percentile whose pooled adequate share is CLOSEST to the
        # band midpoint — avoids edge effects on either side of the band.
        chosen = min(in_band, key=lambda c: abs(c["pooled_shares"]["adequate"] - target_mid))
    else:
        chosen = min(
            candidates,
            key=lambda c: abs((c["pooled_shares"] or {"adequate": 0})["adequate"] - target_mid),
        )

    lf_thr = chosen["threshold"]
    chosen_pct = chosen["percentile"]

    # Final taxonomy under the chosen single pooled threshold.
    per_site_final = {site: shares(df, lf_thr) for site, df in site_dfs.items()}

    # Pooled + hillside / flatland aggregates.
    site_terrain = {s: t for s, _, t in SITES}
    site_label = {s: lbl for s, lbl, _ in SITES}

    def pool(group: set[str]) -> dict:
        pooled = {"adequate": 0, "sunlight_constraint": 0,
                  "ventilation_constraint": 0, "compound_constraint": 0}
        n = 0
        n_built_sum = 0
        for s, r in per_site_final.items():
            if s not in group or "counts" not in r:
                continue
            for k in pooled:
                pooled[k] += r["counts"][k]
            n += r["n_classified"]
            n_built_sum += r["n_built"]
        sh = {k: v / n for k, v in pooled.items()} if n else None
        return {
            "n_built_cells": n_built_sum,
            "n_classified_cells": n,
            "counts": pooled,
            "shares": sh,
        }

    all_sites = {s for s, _, _ in SITES}
    hillside = {s for s, _, t in SITES if t == "hillside"}
    flatland = {s for s, _, t in SITES if t == "flatland"}

    aggregates = {
        "all": pool(all_sites),
        "hillside": pool(hillside),
        "flatland": pool(flatland),
    }

    # Compound contrast.
    hill_comp = aggregates["hillside"]["shares"]["compound_constraint"]
    flat_comp = aggregates["flatland"]["shares"]["compound_constraint"]
    compound_contrast = {
        "hillside_compound_share": hill_comp,
        "flatland_compound_share": flat_comp,
        "absolute_gap_pp": (hill_comp - flat_comp) * 100,
        "relative_ratio": hill_comp / flat_comp if flat_comp > 0 else None,
    }

    # Pooled λf-distribution descriptors.
    lf_descriptors = {
        "n_pooled_built_classified": int(len(pooled_lf)),
        "p05": float(np.percentile(pooled_lf, 5)),
        "p25": float(np.percentile(pooled_lf, 25)),
        "p50": float(np.percentile(pooled_lf, 50)),
        "p75": float(np.percentile(pooled_lf, 75)),
        "p90": float(np.percentile(pooled_lf, 90)),
        "p95": float(np.percentile(pooled_lf, 95)),
        "max": float(pooled_lf.max()),
        "mean": float(pooled_lf.mean()),
        "std": float(pooled_lf.std()),
    }

    payload = {
        "status": "interim — relative pre-screen pending CFD-ACH + LMA",
        "lambda_f_product": (
            "outputs/<site>/morphometrics/grid/grid_metrics.gpkg "
            "(cell-clipped λf, fix landed in commit e2252f4)"
        ),
        "thresholds": {
            "sun_hours_winter_min": SUN_THR,
            "lambda_f_max": lf_thr,
            "lambda_f_max_percentile_of_pooled_corrected_distribution": chosen_pct,
            "adequate_target_band": [ADEQUATE_TARGET_LO, ADEQUATE_TARGET_HI],
            "pooled_adequate_under_threshold": chosen["pooled_shares"]["adequate"],
        },
        "pooled_lambda_f_distribution_corrected": lf_descriptors,
        "built_cell_reconciliation": {
            "explanation": (
                "Two count families exist. n_built = (lambda_p > 0.01 OR "
                "building_count > 0). n_classified = n_built ∩ {sun known} "
                "∩ {λf known}. The previously-cited 56 657 figure is "
                "n_classified (the denominator used in cross_site_stats.json "
                "and in build_diagnostic_map.py). The 64 389 figure was the "
                "raw n_built from 04_recompute_taxonomy.py — which did NOT "
                "drop cells with missing sun observation. The gap is the "
                "built cells lying further than 25 m from any street solar "
                "sample (mostly large-block interiors and Maré low-density "
                "edges). Per-site reconciliation below."
            ),
            **reconcile,
            "canonical_denominator": (
                "n_classified (built ∩ sun-known ∩ λf-known) — used in all "
                "shares reported here."
            ),
        },
        "per_site": {
            s: {
                "label": site_label[s],
                "terrain": site_terrain[s],
                **per_site_final[s],
            }
            for s in [s_ for s_, _, _ in SITES]
        },
        "aggregates": aggregates,
        "compound_contrast_hillside_vs_flatland": compound_contrast,
        "candidate_thresholds_searched": candidates,
    }

    out_path = OUT_DIR / "taxonomy_interim_lambda_f.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Wrote {out_path}")

    print(f"\nChosen pooled λf threshold = {lf_thr:.3f} "
          f"(p{chosen_pct:.0f} of pooled corrected distribution)")
    print(f"Pooled adequate share at threshold = "
          f"{chosen['pooled_shares']['adequate'] * 100:.1f}%")
    print()
    print(f"{'Site':<22s} {'Terrain':<10s} {'n_class':>8s} "
          f"{'Ade':>6s} {'Sun':>6s} {'Vent':>6s} {'Comp':>6s}")
    for s, lbl, t in SITES:
        r = per_site_final[s]
        sh = r["shares"]
        print(f"{lbl:<22s} {t:<10s} {r['n_classified']:>8d} "
              f"{100*sh['adequate']:>5.1f}% "
              f"{100*sh['sunlight_constraint']:>5.1f}% "
              f"{100*sh['ventilation_constraint']:>5.1f}% "
              f"{100*sh['compound_constraint']:>5.1f}%")
    for grp in ("all", "hillside", "flatland"):
        a = aggregates[grp]
        sh = a["shares"]
        print(f"{grp.upper():<22s} {'':<10s} {a['n_classified_cells']:>8d} "
              f"{100*sh['adequate']:>5.1f}% "
              f"{100*sh['sunlight_constraint']:>5.1f}% "
              f"{100*sh['ventilation_constraint']:>5.1f}% "
              f"{100*sh['compound_constraint']:>5.1f}%")
    print(f"\nCompound contrast: hillside {100*hill_comp:.1f}% vs "
          f"flatland {100*flat_comp:.1f}% — gap {compound_contrast['absolute_gap_pp']:+.1f} pp")


if __name__ == "__main__":
    main()
