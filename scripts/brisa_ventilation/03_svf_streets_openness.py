"""Prong C — street-level SVF as the ventilation-openness anchor.

Standardises the ventilation-openness story on **street SVF**
(`outputs/<site>/morphometrics/svf/svf_streets_solar.gpkg`), which is UMEP-
validated and excludes the degenerate rooftop/interior cells that pollute
grid SVF.

For each site, writes:
    outputs/<site>/morphometrics/svf/ventilation_openness_streets.gpkg
        (street SVF points with categorical openness class)
    outputs/brisa_ventilation_fix/svf_streets_stats.json

Openness classes (literature-anchored to Yang et al. 2022 SVF↔wind-velocity
ratio; thresholds chosen per Rio favela context, not from a single canonical
benchmark — the CFD-ACH campaign supersedes these when it lands):
    OPEN       SVF > 0.50   high pedestrian-level ventilation potential
    INTERMEDIATE 0.30–0.50 mixed openness
    SHELTERED  0.15–0.30   limited pedestrian ventilation
    DEEP_CANYON SVF < 0.15  near-still skimming pockets
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

SITES = {
    "vidigal": "Vidigal",
    "rocinha": "Rocinha",
    "complexo_do_alemao": "Complexo do Alemão",
    "riodaspedras": "Rio das Pedras",
    "maré": "Maré",
}

CLASSES = [
    ("DEEP_CANYON", -np.inf, 0.15),
    ("SHELTERED", 0.15, 0.30),
    ("INTERMEDIATE", 0.30, 0.50),
    ("OPEN", 0.50, np.inf),
]


def _classify(svf: float) -> str:
    for name, lo, hi in CLASSES:
        if lo <= svf < hi:
            return name
    return "UNKNOWN"


def process_site(site: str, label: str) -> dict:
    street_svf_path = _ROOT / "outputs" / site / "morphometrics" / "svf" / "svf_streets_solar.gpkg"
    grid_path = _ROOT / "outputs" / site / "morphometrics" / "grid" / "grid_metrics.gpkg"

    streets = gpd.read_file(street_svf_path)
    if "svf" not in streets.columns:
        return {"site": site, "label": label, "error": "no svf column in street file"}

    streets["openness_class"] = streets["svf"].map(_classify)
    out_dir = _ROOT / "outputs" / site / "morphometrics" / "svf"
    open_path = out_dir / "ventilation_openness_streets.gpkg"
    streets[["svf", "openness_class", "geometry"]].to_file(open_path, driver="GPKG")

    sv_st = streets["svf"].dropna()
    class_shares = (
        streets["openness_class"].value_counts(normalize=True).to_dict()
    )

    # Comparator: grid SVF (built cells) — same dist stats so we can show
    # WHY street SVF is the right anchor.
    grid_compare = None
    if grid_path.exists():
        g = gpd.read_file(grid_path)
        if "svf" in g.columns:
            built = g[g.get("lambda_p", pd.Series(np.zeros(len(g)))) > 0]
            sv_g = built["svf"].dropna()
            if len(sv_g):
                grid_compare = {
                    "n": int(sv_g.size),
                    "p5": float(sv_g.quantile(0.05)),
                    "p50": float(sv_g.quantile(0.50)),
                    "p95": float(sv_g.quantile(0.95)),
                    "frac_lt_0_15 (rooftop-contaminated DEEP)": float((sv_g < 0.15).mean()),
                    "frac_gt_0_80 (rooftop/open-sky)": float((sv_g > 0.80).mean()),
                }

    return {
        "site": site,
        "label": label,
        "n_street_points": int(len(streets)),
        "n_with_svf": int(sv_st.size),
        "street_svf": {
            "p5": float(sv_st.quantile(0.05)),
            "p25": float(sv_st.quantile(0.25)),
            "p50": float(sv_st.quantile(0.50)),
            "p75": float(sv_st.quantile(0.75)),
            "p95": float(sv_st.quantile(0.95)),
            "mean": float(sv_st.mean()),
            "std": float(sv_st.std()),
        },
        "openness_class_share": class_shares,
        "grid_svf_comparator_built_cells": grid_compare,
        "ventilation_openness_gpkg": str(open_path.relative_to(_ROOT)),
    }


def main() -> None:
    out_root = _ROOT / "outputs" / "brisa_ventilation_fix"
    out_root.mkdir(parents=True, exist_ok=True)
    results = []
    for site, label in SITES.items():
        print(f"[+] {label}")
        try:
            r = process_site(site, label)
            results.append(r)
            sv = r.get("street_svf", {})
            shares = r.get("openness_class_share", {})
            print(
                f"    n={r['n_with_svf']}  street SVF p5/p50/p95 = "
                f"{sv.get('p5',0):.2f}/{sv.get('p50',0):.2f}/{sv.get('p95',0):.2f}  "
                f"DEEP={100*shares.get('DEEP_CANYON',0):.1f}%  "
                f"SHELTERED={100*shares.get('SHELTERED',0):.1f}%  "
                f"OPEN={100*shares.get('OPEN',0):.1f}%"
            )
        except Exception as e:
            print(f"    FAILED: {e}")
            results.append({"site": site, "label": label, "error": str(e)})

    rationale = (
        "Street SVF is sampled only at observer points on the pedestrian network — "
        "the locus where ventilation matters. Grid SVF aggregates point samples to "
        "10 m cells regardless of whether the cell is roof, courtyard, or street; "
        "the resulting cell mean is contaminated by rooftop opening-sky readings "
        "(SVF→1) and degenerate interior cells (SVF→0). For the ventilation-openness "
        "interpretation in BRISA+, street SVF is the physically correct anchor. "
        "Yang et al. 2022 provide the SVF↔wind-velocity-ratio link; the CFD-ACH "
        "campaign supersedes this geometric proxy when it lands."
    )

    payload = {
        "openness_classes": [
            {"name": n, "svf_min": float(lo) if np.isfinite(lo) else None,
             "svf_max": float(hi) if np.isfinite(hi) else None}
            for n, lo, hi in CLASSES
        ],
        "rationale_street_vs_grid_svf": rationale,
        "citations": {
            "yang_2022": "Yang et al. 2022 — SVF↔ground-level wind velocity ratio",
            "umep_validation": "scripts/validate_svf_against_umep.py confirms street SVF replicates UMEP within ~0.05 absolute on Vidigal pilot",
        },
        "sites": results,
    }
    (out_root / "svf_streets_stats.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False)
    )
    print(f"\nWrote {out_root / 'svf_streets_stats.json'}")


if __name__ == "__main__":
    main()
