"""Phase 0.5 audit: 119 patches against rectangular per-direction CFD criteria.

Applies AIJ Tominaga (2008) / Blocken (2015) rectangular-domain rules to
the existing campaign and reports per-stratum pass rate. Read-only against
``outputs/{site}/cfd_analysis/per_patch_indicators.csv`` and
``data/{site}/buildings_extended_300m.gpkg``; writes a single combined
audit CSV under ``outputs/comparative/cfd_methodology/``.

Methodology
-----------

1. **Patch-scale λ_p (diagnostic only).** Recomputed as
   ``footprint_in_disk / disk_area`` for each patch's 100 m analysis
   circle. The existing ``lambda_p`` column in ``per_patch_indicators.csv``
   is the 10 m grid-cell BCR at the patch centre (clipped to 1.0):
   right for stratification, wrong for blockage.

2. **λ_F per direction (diagnostic only).** Computed via
   ``compute_frontal_area_ratio`` on patch-disk zones:
   ``λ_F = Σ(projected_width × height) / disk_area``. This is the
   literature canopy-parameterisation quantity — sum of all building
   facades — and routinely exceeds 1.0 for dense urban clusters. NOT
   the right quantity for a CFD blockage gate, but reported for the
   §4 morphometric tables.

3. **Silhouette-envelope blockage (gating).** For wide-cluster CFD
   the standard blockage envelope (AIJ benchmark convention) treats
   the analysis disk as a single solid block::

       F_silhouette = D · H_max   (D = 100 m)
       cross-section = (2 · lateral) · top
       B = F / cross-section      < 0.05  (AIJ Tominaga 2008)

   This is the upper bound on the patch's projected obstacle area in
   any wind direction — buildings can hide behind each other but the
   silhouette can't exceed the bounding disk. Conservative *and*
   physically sound.

4. **Wide-obstacle lateral rule (Blocken 2015 §3.3).** For H/W < 1
   clusters, lateral extent scales with cluster width, not building
   height. With patch diameter W = 100 m::

       upstream    = 5 · H_max  + R_patch
       downstream  = 15 · H_max + R_patch
       lateral     = max(5 · H_max + R_patch,  5 · W_patch) = max(5H+50, 500)
       top         = 5 · H_max

   For any patch with H_max < 90 m the 5W floor of 500 m dominates,
   which conveniently makes blockage independent of H_max — uniform
   2 % envelope across the campaign.

Run::

    python scripts/audit_rectangular_domain.py

Outputs::

    outputs/comparative/cfd_methodology/audit_v1.csv
    outputs/comparative/cfd_methodology/audit_v1_pivot.csv
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.urban_morphology import compute_frontal_area_ratio  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SITES = ["vidigal", "riodaspedras", "rocinha", "complexo_do_alemao", "maré"]
PATCH_DIAMETER_M = 100.0
PATCH_RADIUS_M = PATCH_DIAMETER_M / 2.0  # 50 m — analysis circle radius
PATCH_AREA_M2 = math.pi * PATCH_RADIUS_M**2  # 7854 m²
BLOCKAGE_GATE = 0.05  # AIJ Tominaga 2008
SAFETY_MARGIN_M = 50.0  # additive safety on rotated-rect diagonal
SITE_CRS = "EPSG:31983"  # SIRGAS 2000 / UTM 23S


def _largest_extended_buffer(site: str) -> int:
    """Return the largest buffer (m) for which buildings_extended_{N}m.gpkg
    exists for this site. Defaults to 300 if none found."""
    site_dir = PROJECT_ROOT / "data" / site
    buffers = []
    for path in site_dir.glob("buildings_extended_*m.gpkg"):
        try:
            n = int(path.stem.replace("buildings_extended_", "").replace("m", ""))
            buffers.append(n)
        except ValueError:
            continue
    return max(buffers) if buffers else 300


WIND_DIRECTIONS = {
    "N": 0.0,
    "NE": 45.0,
    "E": 90.0,
    "SE": 135.0,
    "S": 180.0,
    "SW": 225.0,
    "W": 270.0,
    "NW": 315.0,
}
OUT_DIR = PROJECT_ROOT / "outputs" / "comparative" / "cfd_methodology"


def _load_indicators(site: str) -> pd.DataFrame:
    """Load per-patch covariates from campaign_patches.csv (the canonical
    source of truth for centers + covariates, per analyze_cfd_results.py).
    The downstream per_patch_indicators.csv only carries 22 rows per site
    on three of the five sites (CDA / Maré / Rocinha were topped up to 25
    after the synthetic CFD pass), so we read campaign_patches directly
    to cover all 119 patches."""
    path = (
        PROJECT_ROOT
        / "outputs"
        / site
        / "sampling_cfd"
        / "campaign_sampling"
        / "campaign_patches.csv"
    )
    df = pd.read_csv(path)
    df.insert(0, "site", site)
    return df


def _load_buildings(site: str, buffer_m: int) -> gpd.GeoDataFrame:
    path = PROJECT_ROOT / "data" / site / f"buildings_extended_{buffer_m}m.gpkg"
    bld = gpd.read_file(path)
    if bld.crs is None or str(bld.crs).upper() != SITE_CRS:
        bld = bld.to_crs(SITE_CRS)
    if "height" not in bld.columns and "altura" in bld.columns:
        bld["height"] = bld["altura"]
    return bld


def _build_patch_zones(df: pd.DataFrame) -> gpd.GeoDataFrame:
    geoms = [Point(x, y).buffer(PATCH_RADIUS_M) for x, y in zip(df["center_x"], df["center_y"])]
    zones = gpd.GeoDataFrame(
        {
            "zone_id": list(range(len(df))),
            "patch_id": df["patch_id"].values,
            "geometry": geoms,
        },
        crs=SITE_CRS,
    )
    zones["zone_area"] = PATCH_AREA_M2
    return zones


def _compute_patch_lambda_p(bld: gpd.GeoDataFrame, zones: gpd.GeoDataFrame) -> pd.Series:
    sindex = bld.sindex
    out = []
    for _, row in zones.iterrows():
        circle = row.geometry
        candidates = list(sindex.intersection(circle.bounds))
        total = 0.0
        for idx in candidates:
            geom = bld.geometry.iloc[idx]
            if geom is None or geom.is_empty:
                continue
            inter = geom.intersection(circle)
            if not inter.is_empty:
                total += inter.area
        out.append(min(total / PATCH_AREA_M2, 1.0))
    return pd.Series(out, index=zones.index, name="lambda_p_patch")


def _compute_lambda_f(bld: gpd.GeoDataFrame, zones: gpd.GeoDataFrame) -> pd.DataFrame:
    """Per-patch λ_F for each of 8 wind directions, plus mean and max."""
    out = pd.DataFrame(index=zones.index)
    for label, bearing in WIND_DIRECTIONS.items():
        result = compute_frontal_area_ratio(bld, zones, wind_dir=bearing)
        out[f"lambda_f_{label}"] = result["lambda_f"].values
    cols = [f"lambda_f_{lbl}" for lbl in WIND_DIRECTIONS]
    out["lambda_f_mean"] = out[cols].mean(axis=1)
    out["lambda_f_max"] = out[cols].max(axis=1)
    out["lambda_f_max_dir"] = out[cols].idxmax(axis=1).str.replace("lambda_f_", "", regex=False)
    return out


def _compute_gates(df: pd.DataFrame) -> pd.DataFrame:
    h = df["H_max_analysis"].astype(float)
    lf_max = df["lambda_f_max"].astype(float)
    R = PATCH_RADIUS_M
    W = PATCH_DIAMETER_M

    # Blocken 2015 wide-obstacle: lateral takes the larger of (5H+R) and 5W.
    df["domain_upstream_m"] = 5.0 * h + R
    df["domain_downstream_m"] = 15.0 * h + R
    df["domain_lateral_m"] = (5.0 * h + R).clip(lower=5.0 * W)
    df["domain_top_m"] = 5.0 * h

    # Silhouette-envelope blockage: treat the patch as a solid block of
    # dimensions D × H_max — the AIJ benchmark convention for wide-cluster
    # CFD. This is the actual gating quantity. λ_F-based blockage is also
    # reported, for diagnostics, but is the canopy-parameterisation quantity
    # (sum of facades) and routinely exceeds 1.0 in dense favela patches.
    df["domain_blockage_frontal_m2"] = PATCH_DIAMETER_M * h
    df["domain_blockage_cross_section_m2"] = 2.0 * df["domain_lateral_m"] * df["domain_top_m"]
    df["domain_blockage_ratio"] = (
        df["domain_blockage_frontal_m2"] / df["domain_blockage_cross_section_m2"]
    )
    df["domain_blockage_ok"] = df["domain_blockage_ratio"] < BLOCKAGE_GATE

    # Diagnostic: blockage if we used λ_F × disk_area (literature canopy-
    # parameterisation quantity). Reported alongside but not used to gate.
    df["lambda_f_blockage_diag"] = lf_max * PATCH_AREA_M2 / df["domain_blockage_cross_section_m2"]

    # Source-data extent: half-diagonal of the rotated box
    # (20 H + 2 R) long × 2 · lateral wide, plus safety. One radius covers all 8 dirs.
    half_long = 10.0 * h + R  # half of the 20H + 2R length
    half_short = df["domain_lateral_m"]  # half of the 2 × lateral width
    df["source_data_required_m"] = (half_long**2 + half_short**2).pow(0.5).apply(
        math.ceil
    ) + SAFETY_MARGIN_M
    # source_data_extent_m is set per-site by the caller (see main()).
    df["source_data_ok"] = df["source_data_extent_m"] >= df["source_data_required_m"]

    df["eligible"] = df["domain_blockage_ok"] & df["source_data_ok"]
    return df


def _pivot_pass_rate(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["site", "stratum_id"], as_index=False).agg(
        n=("patch_id", "count"),
        n_blockage_ok=("domain_blockage_ok", "sum"),
        n_source_ok=("source_data_ok", "sum"),
        n_eligible=("eligible", "sum"),
        mean_lambda_p_patch=("lambda_p_patch", "mean"),
        mean_lambda_f_max=("lambda_f_max", "mean"),
        mean_H_max=("H_max_analysis", "mean"),
        mean_blockage=("domain_blockage_ratio", "mean"),
    )
    g["pass_rate"] = g["n_eligible"] / g["n"]
    return g.sort_values(["site", "stratum_id"])


def _print_report(df: pd.DataFrame, pivot: pd.DataFrame) -> None:
    n = len(df)
    n_block = int(df["domain_blockage_ok"].sum())
    n_src = int(df["source_data_ok"].sum())
    n_elig = int(df["eligible"].sum())

    print("=" * 78)
    print("  PHASE 0.5 AUDIT — rectangular domain (real λ_F + Blocken wide-obstacle + AIJ 5%)")
    print(f"  n = {n} patches across 5 sites")
    print("=" * 78)
    print()
    print(
        f"  Blockage gate  (B < {BLOCKAGE_GATE:.2f}):   {n_block}/{n}  ({100 * n_block / n:5.1f}%)"
    )
    print(f"  Source-data gate (300 m):       {n_src}/{n}  ({100 * n_src / n:5.1f}%)")
    print(f"  ELIGIBLE (both gates):          {n_elig}/{n}  ({100 * n_elig / n:5.1f}%)")
    print()

    print("  λ_F max (patch-scale, worst of 8 directions): distribution")
    for q in [0.10, 0.25, 0.50, 0.75, 0.90]:
        v = df["lambda_f_max"].quantile(q)
        print(f"    p{int(100 * q):02d} = {v:.3f}")
    print(f"    min = {df['lambda_f_max'].min():.3f}   max = {df['lambda_f_max'].max():.3f}")
    print()

    print("  Per-site eligibility:")
    by_site = df.groupby("site").agg(
        n=("patch_id", "count"),
        n_block=("domain_blockage_ok", "sum"),
        n_src=("source_data_ok", "sum"),
        n_eligible=("eligible", "sum"),
    )
    by_site["pass_rate"] = by_site["n_eligible"] / by_site["n"]
    for site, row in by_site.iterrows():
        print(
            f"    {site:24s} elig={int(row['n_eligible']):2d}/{int(row['n']):2d}  "
            f"(blockage={int(row['n_block']):2d}, src={int(row['n_src']):2d})  "
            f"{100 * row['pass_rate']:5.1f}%"
        )
    print()

    print("  Per-stratum eligibility, sorted worst-first:")
    print()
    cols = [
        "site",
        "stratum_id",
        "n",
        "n_eligible",
        "pass_rate",
        "mean_lambda_p_patch",
        "mean_lambda_f_max",
        "mean_H_max",
        "mean_blockage",
    ]
    p = pivot.sort_values("pass_rate")[cols]
    print(p.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()

    fail = df[~df["eligible"]].sort_values("domain_blockage_ratio", ascending=False)
    if len(fail):
        print(f"  Failing patches ({len(fail)}):")
        for _, r in fail.iterrows():
            why = []
            if not r["domain_blockage_ok"]:
                why.append(
                    f"B={r['domain_blockage_ratio']:.3f} (worst dir={r['lambda_f_max_dir']})"
                )
            if not r["source_data_ok"]:
                why.append(
                    f"src={int(r['source_data_required_m'])}m>{int(r['source_data_extent_m'])}m"
                )
            print(
                f"    {r['site']:18s} {r['patch_id']:8s}  stratum={r['stratum_id']:14s}  "
                f"λf_max={r['lambda_f_max']:.3f}  H_max={r['H_max_analysis']:5.1f}m   "
                f"({', '.join(why)})"
            )
        print()

    # If failures are all source-data-only, surface that — re-stratification
    # is not needed; a buffer extension fixes all patches at once.
    block_only_fail = ((~df["domain_blockage_ok"]) & df["source_data_ok"]).sum()
    src_only_fail = (df["domain_blockage_ok"] & (~df["source_data_ok"])).sum()
    both_fail = ((~df["domain_blockage_ok"]) & (~df["source_data_ok"])).sum()
    print("  Failure-mode breakdown:")
    print(f"    blockage-only failure:   {block_only_fail}/{n}")
    print(f"    source-data-only fail:   {src_only_fail}/{n}")
    print(f"    both gates fail:         {both_fail}/{n}")
    print()

    if src_only_fail and not block_only_fail and not both_fail:
        max_needed = int(df["source_data_required_m"].max())
        suggested = ((max_needed + 49) // 50 + 1) * 50  # round up to next 50 + buffer
        print("  ALL FAILURES ARE SOURCE-DATA EXTENT ONLY.")
        print(f"  Max required: {max_needed} m. Suggested extension: {suggested} m.")
        print("  No re-stratification needed — buffer extension lifts the entire")
        print("  campaign to 100% eligibility.")
        print()
    else:
        risky = pivot.groupby("stratum_id").agg(n=("n", "sum"), n_eligible=("n_eligible", "sum"))
        risky["pass_rate"] = risky["n_eligible"] / risky["n"]
        dropped = risky[risky["n_eligible"] == 0]
        if len(dropped):
            print("  STRATA AT RISK — zero eligible patches across all sites:")
            for stratum, row in dropped.iterrows():
                print(f"    {stratum}: 0/{int(row['n'])} eligible")
            print()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parts = []
    for site in SITES:
        buffer_m = _largest_extended_buffer(site)
        df = _load_indicators(site)
        bld = _load_buildings(site, buffer_m)
        zones = _build_patch_zones(df)
        df["lambda_p_patch"] = _compute_patch_lambda_p(bld, zones).values
        lf = _compute_lambda_f(bld, zones)
        for col in lf.columns:
            df[col] = lf[col].values
        df = df.rename(columns={"lambda_p": "lambda_p_grid_center"})
        df["source_data_extent_m"] = float(buffer_m)
        parts.append(df)
        print(f"  {site:24s} buffer={buffer_m}m  λ_F for {len(df)} patches × 8 directions")
    print()

    df = pd.concat(parts, ignore_index=True)
    df = _compute_gates(df)

    keep_cols = [
        "site",
        "patch_id",
        "is_pilot",
        "stratum_id",
        "lambda_p_grid_center",
        "lambda_p_patch",
        *[f"lambda_f_{lbl}" for lbl in WIND_DIRECTIONS],
        "lambda_f_mean",
        "lambda_f_max",
        "lambda_f_max_dir",
        "H_max_analysis",
        "H_mean",
        "domain_upstream_m",
        "domain_downstream_m",
        "domain_lateral_m",
        "domain_top_m",
        "domain_blockage_frontal_m2",
        "domain_blockage_cross_section_m2",
        "domain_blockage_ratio",
        "domain_blockage_ok",
        "lambda_f_blockage_diag",
        "source_data_required_m",
        "source_data_extent_m",
        "source_data_ok",
        "eligible",
    ]
    audit = df[keep_cols].copy()
    audit_path = OUT_DIR / "audit_v1.csv"
    audit.to_csv(audit_path, index=False)

    pivot = _pivot_pass_rate(df)
    pivot_path = OUT_DIR / "audit_v1_pivot.csv"
    pivot.to_csv(pivot_path, index=False)

    _print_report(df, pivot)
    print(f"  wrote: {audit_path.relative_to(PROJECT_ROOT)}")
    print(f"  wrote: {pivot_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
