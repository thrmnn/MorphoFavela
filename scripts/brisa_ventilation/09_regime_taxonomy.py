"""Regime-stratified four-state ventilation taxonomy (PI-adopted 2026-06-25).

Supersedes the interim relative pre-screen in
``08_interim_lambda_f_taxonomy.py``. Both PI decisions are now ADOPTED
(brisaverse ``shared/facts/morphofavela_sync_2026-06-25.md``):

  1. Dissolved (party-wall-corrected) λf is canonical.
  2. The taxonomy ventilation axis is re-derived **regime-stratified**:
     ventilation constraint ≡ skimming flow regime, i.e. an ABSOLUTE
     boundary ``dissolved λf > 0.65`` (Grimmond & Oke 1999), replacing the
     deprecated pooled-p75 ``λf > 2.75`` relative pre-screen.

For every campaign site this loads, on the canonical 10 m grid:
  - dissolved λf (``lambda_f_mean``) from grid_metrics.gpkg
  - winter direct-sun hours from svf_streets_solar.gpkg (median-within-cell
    + nearest-3 fallback within 25 m — identical aggregation to script 08)
  - street H/W from canyon/hw_streets.gpkg (street sample points, same
    nearest-3-within-25 m aggregation) — the λf-product-INDEPENDENT
    robustness cross-check, Oke (1988) skimming boundary H/W > 0.65.

Axes:
  sunlight constraint    winter direct sun < 2 h        (canonical, unchanged)
  ventilation constraint dissolved λf > 0.65 (skimming) PRIMARY
                         H/W > 0.65 (Oke)               cross-check (own denom)

Denominator: ``n_classified`` = built ∩ sun-known ∩ λf-known (the canonical
56 657, NOT the 64 389 raw built mask). The H/W cross-check carries its own,
sparser denominator (cells with an H/W sample within 25 m) and is reported
separately — never mixed into the primary shares.

Geometry classifies the *regime*; per-cell ventilation *adequacy* is carried
by CFD age-of-air (gated), not read off λf. No τ here.

Output:
  outputs/brisa_ventilation_fix/taxonomy_regime.json
  outputs/brisa_ventilation_fix/methods_morphometric_row.csv
"""

from __future__ import annotations

import json
import sys
import warnings
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

SUN_THR = 2.0          # winter direct-sun hours floor (canonical)
SKIMMING_LF = 0.65     # Grimmond & Oke (1999) skimming-flow boundary (dissolved λf)
SKIMMING_HW = 0.65     # Oke (1988) skimming boundary on street H/W (cross-check)
MACDONALD_LF = 0.40    # reported as a secondary λf variant (situates vs sync table)

METHODS_METRICS = ["svf", "lambda_p", "lambda_f_mean", "H_mean", "sigma_h", "slope_deg"]

OUT_DIR = _ROOT / "outputs" / "brisa_ventilation_fix"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _aggregate_points(grid: gpd.GeoDataFrame, pts: gpd.GeoDataFrame, value_col: str,
                      out_col: str, max_radius: float = 25.0, k: int = 3) -> gpd.GeoDataFrame:
    """Median-within-cell, then nearest-k fallback within ``max_radius``.

    Mirrors script 08's solar aggregation so the H/W cross-check uses the
    identical spatial support rule as the primary sun/λf axes."""
    pts = pts[[value_col, "geometry"]].dropna(subset=[value_col])
    j = gpd.sjoin(pts, grid[["zone_id", "geometry"]], how="inner", predicate="within")
    primary = j.groupby("zone_id")[value_col].median().rename(out_col).reset_index()
    grid = grid.merge(primary, on="zone_id", how="left")
    missing = grid[out_col].isna()
    if missing.any() and len(pts):
        pt_xy = np.array([(g.x, g.y) for g in pts.geometry])
        pt_val = pts[value_col].to_numpy()
        tree = cKDTree(pt_xy)
        cents = grid.loc[missing, ["centroid_x", "centroid_y"]].to_numpy()
        dists, idxs = tree.query(cents, k=k, distance_upper_bound=max_radius)
        if k == 1:
            dists, idxs = dists[:, None], idxs[:, None]
        with np.errstate(invalid="ignore"):
            valid = np.isfinite(dists)
        idxs = np.where(valid, idxs, 0)
        vals = np.where(valid, pt_val[idxs], np.nan)
        with warnings.catch_warnings():  # all-NaN rows = cells with no sample in radius → stay NaN
            warnings.simplefilter("ignore", RuntimeWarning)
            grid.loc[missing, out_col] = np.nanmedian(vals, axis=1)
    return grid


def load_site(site: str) -> pd.DataFrame:
    base = _ROOT / "outputs" / site / "morphometrics"
    grid = gpd.read_file(base / "grid" / "grid_metrics.gpkg")

    sol = gpd.read_file(base / "svf" / "svf_streets_solar.gpkg")
    grid = _aggregate_points(grid, sol, "solar_hours_winter", "solar_hours_winter")

    hw_path = base / "canyon" / "hw_streets.gpkg"
    if hw_path.exists():
        hw_pts = gpd.read_file(hw_path)
        grid = _aggregate_points(grid, hw_pts, "HW", "hw_cell")
    else:
        grid["hw_cell"] = np.nan

    lp = grid.get("lambda_p", pd.Series(np.zeros(len(grid)))).fillna(0)
    bc = grid.get("building_count", pd.Series(np.zeros(len(grid))))
    built = (lp > 0.01) | (bc > 0)
    sun = grid["solar_hours_winter"].astype(float).to_numpy()
    lf = grid["lambda_f_mean"].astype(float).to_numpy()
    hw = grid["hw_cell"].astype(float).to_numpy()

    df = pd.DataFrame({
        "site": site,
        "built": built.to_numpy(),
        "sun_known": np.isfinite(sun),
        "lf_known": np.isfinite(lf),
        "hw_known": np.isfinite(hw),
        "sun": sun,
        "lf": lf,
        "hw": hw,
    })
    for m in METHODS_METRICS:
        df[m] = grid[m].astype(float).to_numpy() if m in grid.columns else np.nan
    return df


def _four_state(sun_fail: np.ndarray, vent_fail: np.ndarray, n: int) -> dict:
    counts = {
        "adequate": int((~sun_fail & ~vent_fail).sum()),
        "sunlight_constraint": int((sun_fail & ~vent_fail).sum()),
        "ventilation_constraint": int((~sun_fail & vent_fail).sum()),
        "compound_constraint": int((sun_fail & vent_fail).sum()),
    }
    return {"counts": counts, "shares": {k: v / n for k, v in counts.items()}}


def shares_lf(df: pd.DataFrame) -> dict:
    """PRIMARY axis: dissolved λf > 0.65 on the canonical denominator."""
    c = df[df["built"] & df["sun_known"] & df["lf_known"]]
    n = len(c)
    if n == 0:
        return {"n_built": int(df["built"].sum()), "n_classified": 0}
    res = _four_state((c["sun"] < SUN_THR).to_numpy(),
                      (c["lf"] > SKIMMING_LF).to_numpy(), n)
    res_macd = _four_state((c["sun"] < SUN_THR).to_numpy(),
                           (c["lf"] > MACDONALD_LF).to_numpy(), n)
    return {
        "n_built": int(df["built"].sum()),
        "n_classified": n,
        **res,
        "macdonald_lf040": res_macd,
    }


def shares_hw(df: pd.DataFrame) -> dict:
    """Cross-check axis: street H/W > 0.65 (Oke), own sparser denominator."""
    c = df[df["built"] & df["sun_known"] & df["hw_known"]]
    n = len(c)
    if n == 0:
        return {"n_classified_hw": 0}
    res = _four_state((c["sun"] < SUN_THR).to_numpy(),
                      (c["hw"] > SKIMMING_HW).to_numpy(), n)
    return {"n_classified_hw": n, **res}


def pool(per_site: dict, group: set[str], key: str, n_key: str) -> dict:
    agg = {"adequate": 0, "sunlight_constraint": 0,
           "ventilation_constraint": 0, "compound_constraint": 0}
    n = 0
    for s, r in per_site.items():
        if s not in group or "counts" not in r:
            continue
        for k in agg:
            agg[k] += r["counts"][k]
        n += r[n_key]
    sh = {k: v / n for k, v in agg.items()} if n else None
    return {n_key: n, "counts": agg, "shares": sh}


def methods_row(df: pd.DataFrame) -> dict:
    """Per-site medians + means over the canonical classified cells."""
    c = df[df["built"] & df["sun_known"] & df["lf_known"]]
    out = {"n": int(len(c))}
    for m in METHODS_METRICS:
        v = c[m].to_numpy()
        v = v[np.isfinite(v)]
        out[m] = {"median": float(np.median(v)), "mean": float(np.mean(v))} if len(v) else None
    return out


def main() -> None:
    site_dfs = {s: load_site(s) for s, _, _ in SITES}
    label = {s: lbl for s, lbl, _ in SITES}
    terrain = {s: t for s, _, t in SITES}

    per_site_lf = {s: shares_lf(df) for s, df in site_dfs.items()}
    per_site_hw = {s: shares_hw(df) for s, df in site_dfs.items()}
    methods = {s: methods_row(df) for s, df in site_dfs.items()}

    allg = {s for s, _, _ in SITES}
    hill = {s for s, _, t in SITES if t == "hillside"}
    flat = {s for s, _, t in SITES if t == "flatland"}

    def contrast(per_site, n_key):
        h = pool(per_site, hill, "shares", n_key)["shares"]["compound_constraint"]
        f = pool(per_site, flat, "shares", n_key)["shares"]["compound_constraint"]
        return {
            "hillside_compound_share": h,
            "flatland_compound_share": f,
            "absolute_gap_pp": (h - f) * 100,
            "relative_ratio": h / f if f > 0 else None,
            "inversion_old_finding_holds": h > f,  # old finding: flatland > hillside
        }

    payload = {
        "status": "regime-stratified — PI-adopted 2026-06-25 (supersedes interim λf>2.75)",
        "lambda_f_product": "dissolved (party-wall corrected), grid_metrics.gpkg lambda_f_mean",
        "axes": {
            "sunlight_constraint": f"winter direct sun < {SUN_THR} h",
            "ventilation_constraint_primary": f"dissolved λf > {SKIMMING_LF} (Grimmond & Oke 1999 skimming)",
            "ventilation_constraint_crosscheck": f"street H/W > {SKIMMING_HW} (Oke 1988 skimming)",
            "ventilation_constraint_macdonald": f"dissolved λf > {MACDONALD_LF} (Macdonald 1998)",
        },
        "denominator": (
            "n_classified = built ∩ sun-known ∩ λf-known (canonical ~56 657). "
            "H/W cross-check uses its own sparser denominator (n_classified_hw)."
        ),
        "regime_note": (
            "Geometry classifies the flow REGIME; per-cell ventilation ADEQUACY "
            "is carried by CFD age-of-air (gated). No τ here."
        ),
        "per_site": {
            s: {"label": label[s], "terrain": terrain[s],
                "lambda_f_axis": per_site_lf[s], "hw_axis": per_site_hw[s]}
            for s in allg
        },
        "aggregates_lambda_f": {
            "all": pool(per_site_lf, allg, "shares", "n_classified"),
            "hillside": pool(per_site_lf, hill, "shares", "n_classified"),
            "flatland": pool(per_site_lf, flat, "shares", "n_classified"),
        },
        "aggregates_hw": {
            "all": pool(per_site_hw, allg, "shares", "n_classified_hw"),
            "hillside": pool(per_site_hw, hill, "shares", "n_classified_hw"),
            "flatland": pool(per_site_hw, flat, "shares", "n_classified_hw"),
        },
        "compound_contrast_lambda_f": contrast(per_site_lf, "n_classified"),
        "compound_contrast_hw": contrast(per_site_hw, "n_classified_hw"),
        "methods_morphometric_row": methods,
    }

    out_json = OUT_DIR / "taxonomy_regime.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Wrote {out_json}")

    # Methods-table CSV (medians + means).
    rows = []
    for s, _, t in SITES:
        m = methods[s]
        row = {"site": label[s], "terrain": t, "n": m["n"]}
        for met in METHODS_METRICS:
            row[f"{met}_median"] = m[met]["median"] if m[met] else None
            row[f"{met}_mean"] = m[met]["mean"] if m[met] else None
        rows.append(row)
    csv_path = OUT_DIR / "methods_morphometric_row.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    # Console summary.
    def show(per_site, aggs, n_key, title):
        print(f"\n=== {title} ===")
        print(f"{'Site':<22s} {'Terrain':<10s} {n_key:>10s} "
              f"{'Ade':>6s} {'Sun':>6s} {'Vent':>6s} {'Comp':>6s}")
        for s, lbl, t in SITES:
            r = per_site[s]
            if "shares" not in r:
                continue
            sh = r["shares"]
            print(f"{lbl:<22s} {t:<10s} {r[n_key]:>10d} "
                  f"{100*sh['adequate']:>5.1f}% {100*sh['sunlight_constraint']:>5.1f}% "
                  f"{100*sh['ventilation_constraint']:>5.1f}% {100*sh['compound_constraint']:>5.1f}%")
        for grp in ("all", "hillside", "flatland"):
            a = aggs[grp]
            if a["shares"] is None:
                continue
            sh = a["shares"]
            print(f"{grp.upper():<22s} {'':<10s} {a[n_key]:>10d} "
                  f"{100*sh['adequate']:>5.1f}% {100*sh['sunlight_constraint']:>5.1f}% "
                  f"{100*sh['ventilation_constraint']:>5.1f}% {100*sh['compound_constraint']:>5.1f}%")

    show(per_site_lf, payload["aggregates_lambda_f"], "n_classified",
         "PRIMARY  dissolved λf > 0.65 (skimming)")
    show(per_site_hw, payload["aggregates_hw"], "n_classified_hw",
         "CROSS-CHECK  H/W > 0.65 (Oke)")

    cl = payload["compound_contrast_lambda_f"]
    print(f"\nCompound contrast (λf): hillside {100*cl['hillside_compound_share']:.1f}% vs "
          f"flatland {100*cl['flatland_compound_share']:.1f}% — gap {cl['absolute_gap_pp']:+.1f} pp; "
          f"old flatland>hillside inversion holds: {not cl['inversion_old_finding_holds']}")


if __name__ == "__main__":
    main()
