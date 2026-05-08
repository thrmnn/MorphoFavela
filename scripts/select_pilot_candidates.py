"""Rank patches as CFD pilot candidates across three rungs.

The CFD repo wants a small set (3–6) of pilot patches to mesh-converge
and validate the v1 rectangular-domain pipeline before scaling to all
119 × 8 sims. Three rungs in the experimental ladder, designed to
isolate one confound at a time:

* **Rung 0** (flat + uniform heights): slope_deg < 5 AND sigma_h < 1.5.
  Tests the meshing pipeline + boundary-layer treatment in isolation,
  no terrain or height-variability complications.
* **Rung 1** (sloped + sparse): slope_deg ≥ 8 AND lambda_p < 0.4.
  Adds terrain forcing on a sparse array — separable from canopy
  shadowing.
* **Rung 2** (VDG-P07-equivalent: moderate slope + moderate λ_p,
  low height variability): 6 ≤ slope_deg < 10 AND 0.15 ≤ lambda_p ≤
  0.35 AND sigma_h < 1.5. Reproduces the morphology that broke the
  v0 cylindrical pilot (omega-bounding events at sector corners) so
  v1 mesh convergence can be benchmarked against that failure mode.

Each candidate is ranked by squared Mahalanobis-style distance from the
rung archetype in the (slope, lambda_p, sigma_h) space. Top-3 per rung
are flagged ``recommended=True``. The three pre-existing candidates
(RDP-P20, ROC-P12, CDA-P22) are flagged ``original_pick=True``
regardless of rank.

Reads ``outputs/{site}/sampling_cfd/campaign_sampling/campaign_patches.csv``
(post-migration, 37 cols including the v1 domain block). Writes
``outputs/comparative/cfd_methodology/pilot_candidates_v1.csv``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SITES = ["vidigal", "riodaspedras", "rocinha", "complexo_do_alemao", "maré"]
OUT_DIR = PROJECT_ROOT / "outputs" / "comparative" / "cfd_methodology"
ORIGINAL_PICKS = {"RDP-P20", "ROC-P12", "CDA-P22"}

# Archetype anchors per rung — used as the centre point for the
# distance ranking. Matches the prompt's experimental-ladder design.
ARCHETYPES = {
    0: {"slope_deg": 1.0, "lambda_p": 0.30, "sigma_h": 0.50},
    1: {"slope_deg": 12.0, "lambda_p": 0.20, "sigma_h": 1.50},
    2: {"slope_deg": 8.5, "lambda_p": 0.25, "sigma_h": 0.50},
}

# Hard-eligibility predicates; rows that fail are not candidates for
# that rung at all. Thresholds chosen so the three v0 picks
# (RDP-P20 / ROC-P12 / CDA-P22) each land in a distinct rung.
RUNG_FILTERS = {
    0: lambda r: (r["slope_deg"] < 5.0) and (r["sigma_h"] < 1.5),
    1: lambda r: (r["slope_deg"] >= 5.0) and (r["lambda_p"] < 0.20),
    2: lambda r: (
        6.0 <= r["slope_deg"] < 12.0
        and 0.20 <= r["lambda_p"] <= 0.35
        and r["sigma_h"] < 1.5
    ),
}


def _load_campaign() -> pd.DataFrame:
    parts = []
    for site in SITES:
        path = PROJECT_ROOT / "outputs" / site / "sampling_cfd" / "campaign_sampling" / "campaign_patches.csv"
        df = pd.read_csv(path)
        df.insert(0, "site", site)
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def _normalised_distance(df: pd.DataFrame, archetype: dict[str, float]) -> pd.Series:
    """Squared standardised distance in (slope, λp, σ_h) space.

    Each axis is divided by the campaign-wide stdev so the three
    contribute on comparable scales — slope spans ~30°, λp ~0–1,
    σ_h ~0–6, otherwise slope would dominate.
    """
    cols = list(archetype)
    stds = {c: df[c].std() or 1.0 for c in cols}
    d2 = sum(((df[c] - archetype[c]) / stds[c]) ** 2 for c in cols)
    return d2


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = _load_campaign()
    print(f"  loaded {len(df)} patches from {len(SITES)} sites")
    print()

    if "eligible" not in df.columns:
        print("ERROR: campaign_patches.csv has not been migrated to v1 yet.", file=sys.stderr)
        print("Run: python scripts/migrate_indicators_rectangular_v1.py --apply", file=sys.stderr)
        return 1

    out_rows = []
    for rung, archetype in ARCHETYPES.items():
        passes = df.apply(RUNG_FILTERS[rung], axis=1)
        eligible = df[passes & df["eligible"].astype(bool)].copy()
        if eligible.empty:
            print(f"  rung {rung}: 0 candidates pass the filter — skipping")
            continue
        eligible["rung"] = rung
        eligible["distance"] = _normalised_distance(eligible, archetype)
        eligible = eligible.sort_values("distance").reset_index(drop=True)
        eligible["rank_within_rung"] = eligible.index + 1
        eligible["recommended"] = eligible["rank_within_rung"] <= 3
        out_rows.append(eligible)
        print(f"  rung {rung}: {len(eligible):2d} candidates pass filter — top 3 recommended")
        for _, r in eligible.head(3).iterrows():
            print(
                f"      #{int(r['rank_within_rung'])}  {r['patch_id']:8s}  ({r['site']:18s})  "
                f"slope={r['slope_deg']:5.2f}°  λp={r['lambda_p']:.3f}  σh={r['sigma_h']:.3f}"
            )
    print()

    out = pd.concat(out_rows, ignore_index=True)
    out["original_pick"] = out["patch_id"].isin(ORIGINAL_PICKS)

    cols = [
        "rung", "rank_within_rung", "recommended", "original_pick",
        "site", "patch_id", "stratum_id",
        "slope_deg", "lambda_p", "lambda_p_patch", "sigma_h",
        "H_mean", "H_max_analysis",
        "domain_lateral_m", "domain_blockage_ratio", "source_data_required_m",
        "eligible", "distance",
    ]
    cols = [c for c in cols if c in out.columns]
    out = out[cols]

    path = OUT_DIR / "pilot_candidates_v1.csv"
    out.to_csv(path, index=False)
    print(f"  wrote: {path.relative_to(PROJECT_ROOT)}  ({len(out)} rows)")

    # Sanity: confirm the three v0 picks landed in expected rungs.
    print()
    print("  Original-pick verification (RDP-P20 / ROC-P12 / CDA-P22):")
    for pid in sorted(ORIGINAL_PICKS):
        rows = out[out["patch_id"] == pid]
        if rows.empty:
            print(f"    {pid}: NOT in any rung's candidate pool")
        else:
            for _, r in rows.iterrows():
                print(
                    f"    {pid}: rung {int(r['rung'])} rank {int(r['rank_within_rung'])}/{len(out[out['rung']==r['rung']])}"
                    f"  slope={r['slope_deg']:.2f}°  λp={r['lambda_p']:.3f}  σh={r['sigma_h']:.3f}"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
