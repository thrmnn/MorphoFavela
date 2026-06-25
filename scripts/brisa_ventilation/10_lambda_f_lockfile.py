"""Canonical dissolved-λf lock file — the single authoritative source.

Produced so the brisaverse agent integrates ONE traceable λf reference
instead of reconciling three (lambda_f_regime.json, methods_morphometric_row.csv,
taxonomy_regime.json), which carry different denominators and groupings.

CANONICAL DENOMINATOR for the λf *morphometric descriptor* = the full built
mask ``building_count > 0`` (pooled n = 64,389). This is the denominator the
TR §5.2 Methods-table λf row uses and the one ``lambda_f_regime.json`` uses.
It is NOT the taxonomy's ``n_classified`` (~56,631 = built ∩ sun-known ∩
λf-known); that smaller denominator is taxonomy-specific (it additionally
needs a winter-sun observation) and is correct only inside the four-state
taxonomy, never for the standalone λf descriptor.

TWO LEGITIMATE GROUPINGS — do not conflate:
  signature_family (ternary, TR §5.2): Hillside {Vid, Roc} · Mixed {CDA} ·
      Flatland {RdP, Maré}. The morphometric-signature family split.
  terrain_binary (taxonomy four-state): hillside {Vid, Roc, CDA} ·
      flatland {RdP, Maré}. CDA folds into its predominant hillside morro;
      this is the split behind the compound contrast (hillside 42.2% >
      flatland 37.3%).

λf is dissolved (party-wall corrected): touching footprints are unioned into
physical blocks before projecting, so interior party walls don't double-count.
``lambda_f_mean_summed`` (the pre-fix per-building sum) is preserved on every
grid; the over-count factor is median(summed)/median(dissolved).

Output: outputs/brisa_ventilation_fix/lambda_f_canonical.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

SITES = [
    ("vidigal", "Vidigal"),
    ("rocinha", "Rocinha"),
    ("complexo_do_alemao", "Complexo do Alemão"),
    ("riodaspedras", "Rio das Pedras"),
    ("maré", "Maré"),
]

SIGNATURE_FAMILY = {  # TR §5.2 ternary
    "Hillside": ["vidigal", "rocinha"],
    "Mixed": ["complexo_do_alemao"],
    "Flatland": ["riodaspedras", "maré"],
}
TERRAIN_BINARY = {  # taxonomy four-state
    "hillside": ["vidigal", "rocinha", "complexo_do_alemao"],
    "flatland": ["riodaspedras", "maré"],
}

COMPANIONS = ["svf", "lambda_p", "H_mean", "sigma_h", "slope_deg"]
ISOLATED_MAX, SKIMMING_MIN = 0.15, 0.65

OUT = _ROOT / "outputs" / "brisa_ventilation_fix" / "lambda_f_canonical.json"


def built_cells(site: str) -> gpd.GeoDataFrame:
    g = gpd.read_file(_ROOT / "outputs" / site / "morphometrics" / "grid" / "grid_metrics.gpkg")
    return g[g["building_count"] > 0]


def _stats(v: np.ndarray) -> dict:
    v = v[np.isfinite(v)]
    return {"n": int(v.size), "median": float(np.median(v)), "mean": float(np.mean(v))}


def regime_shares(lf: np.ndarray) -> dict:
    lf = lf[np.isfinite(lf)]
    iso = float((lf < ISOLATED_MAX).mean())
    sky = float((lf >= SKIMMING_MIN).mean())
    return {"isolated": iso, "wake": 1.0 - iso - sky, "skimming": sky}


def main() -> None:
    grids = {s: built_cells(s) for s, _ in SITES}
    lf = {s: grids[s]["lambda_f_mean"].to_numpy() for s, _ in SITES}
    lfs = {s: grids[s]["lambda_f_mean_summed"].to_numpy() for s, _ in SITES}

    per_site = {}
    for s, lbl in SITES:
        d = _stats(lf[s])
        summ = _stats(lfs[s])
        comp = {}
        for c in COMPANIONS:
            comp[c] = _stats(grids[s][c].to_numpy()) if c in grids[s].columns else None
        per_site[s] = {
            "label": lbl,
            "lambda_f_dissolved": d,
            "lambda_f_summed": summ,
            "over_count_factor_median": summ["median"] / d["median"],
            "companions": comp,
        }

    def group(members: list[str]) -> dict:
        v = np.concatenate([lf[s] for s in members])
        return _stats(v)

    groupings = {
        "signature_family": {k: group(v) for k, v in SIGNATURE_FAMILY.items()},
        "terrain_binary": {k: group(v) for k, v in TERRAIN_BINARY.items()},
        "pooled": group([s for s, _ in SITES]),
    }

    pooled_lf = np.concatenate([lf[s] for s, _ in SITES])
    payload = {
        "title": "Canonical dissolved λf — single authoritative source (2026-06-25)",
        "lambda_f_definition": (
            "dissolved (party-wall corrected): touching footprints unioned into "
            "blocks before projecting frontal area; mean of the 4 distinct "
            "cross-wind axes (lambda_f_mean). Pre-fix per-building sum preserved "
            "as lambda_f_mean_summed."
        ),
        "canonical_denominator": (
            "full built mask building_count > 0 (pooled n = 64,389). Matches TR "
            "§5.2 and lambda_f_regime.json. NOT the taxonomy n_classified (~56,631), "
            "which additionally requires a winter-sun observation and is "
            "taxonomy-specific."
        ),
        "groupings_note": (
            "signature_family (ternary, TR §5.2): Hillside{Vid,Roc} Mixed{CDA} "
            "Flatland{RdP,Maré}. terrain_binary (taxonomy): hillside{Vid,Roc,CDA} "
            "flatland{RdP,Maré} — CDA folds into hillside. Do not conflate."
        ),
        "flow_regime_thresholds": {"isolated_max": ISOLATED_MAX, "skimming_min": SKIMMING_MIN,
                                   "reference": "Oke (1988) / Grimmond & Oke (1999)"},
        "flow_regime_pooled": regime_shares(pooled_lf),
        "per_site": per_site,
        "groupings": groupings,
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Wrote {OUT.relative_to(_ROOT)}")

    print(f"\n{'Site':<22s} {'n':>7s} {'λf_med':>7s} {'λf_mean':>8s} {'summed_med':>11s} {'over×':>6s}")
    for s, lbl in SITES:
        p = per_site[s]
        print(f"{lbl:<22s} {p['lambda_f_dissolved']['n']:>7d} "
              f"{p['lambda_f_dissolved']['median']:>7.3f} {p['lambda_f_dissolved']['mean']:>8.3f} "
              f"{p['lambda_f_summed']['median']:>11.3f} {p['over_count_factor_median']:>5.2f}x")
    print("\nsignature_family (TR §5.2):")
    for k, v in groupings["signature_family"].items():
        print(f"  {k:<10s} median {v['median']:.3f}  mean {v['mean']:.3f}  n {v['n']}")
    print("terrain_binary (taxonomy):")
    for k, v in groupings["terrain_binary"].items():
        print(f"  {k:<10s} median {v['median']:.3f}  n {v['n']}")
    pr = payload["flow_regime_pooled"]
    print(f"pooled n {groupings['pooled']['n']}  median {groupings['pooled']['median']:.3f}  "
          f"skimming {pr['skimming']*100:.1f}% wake {pr['wake']*100:.1f}% isolated {pr['isolated']*100:.1f}%")


if __name__ == "__main__":
    main()
