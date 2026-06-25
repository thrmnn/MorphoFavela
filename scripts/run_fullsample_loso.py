#!/usr/bin/env python3
"""Full-sample, geometry-only LOSO robustness bound for the sunlight-constraint
predictor.

Why this exists
---------------
The canonical predictor (``run_predictor_analysis.py``) is complete-case: it
keeps only built cells where every feature — crucially **SVF** and
**street-orientation entropy** — is present. Those two features are measured at
street points, so ~50% of built cells (interior / non-street-adjacent fabric)
are dropped. The surviving 22,238-cell subset is biased toward street-adjacent
cells and has a higher sunlight-failure prevalence (56%) than the full built
sample (46%).

The brisaverse handoff asked to bound that caveat by "imputing aspect" on the
full 56,631-cell sample. That premise does not hold for this data: aspect is
~complete (≤ 8 NaN cells). The genuine missingness is SVF (≈50%) and
street-entropy (≈54%). Imputing SVF — the single strongest predictor — would
inflate the AUC with fabricated signal, not bound it.

So the honest robustness check is the opposite: drop the two high-missingness,
street-sampled features and refit LOSO on the **full** 56,631-cell sample using
only geometry that is defined everywhere (slope, aspect, λp, λf, σH). If the
signal survives, the headline transfers beyond the street-adjacent subset.

Result (see the written JSON): full-sample reduced-feature LOSO AUC ≈ 0.75–0.84
(mean 0.78), vs the complete-case full-feature headline 0.87–0.93. The morphology
-only signal is clearly transferable on the unbiased sample; the extra ~0.12 AUC
in the headline is the SVF contribution on the cells where SVF is observed.

Output: outputs/paper_figures/fullsample_loso.json
"""

from __future__ import annotations

import json
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from scripts.run_predictor_analysis import (
    FEATURES,
    PROJECT_ROOT,
    SITES,
    THRESHOLD_SUN_HRS,
    aggregate_solar,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# Features defined on (nearly) every built cell — no street-sampled SVF or
# street-orientation entropy, which carry the ~50% missingness.
REDUCED_FEATURES = ["slope_deg", "southness", "eastness", "lambda_p", "lambda_f_mean", "sigma_h"]

OUT_JSON = PROJECT_ROOT / "outputs" / "paper_figures" / "fullsample_loso.json"


def build_fullsample_table() -> tuple[pd.DataFrame, dict]:
    """All built cells with a known sun/vent state; aspect→neutral and σ_H→0
    imputed for the few cells where they are undefined (flat / single-building).
    Returns the table plus a per-feature missingness audit over the keep set."""
    frames = []
    missing = {f: 0 for f in FEATURES}
    keep_n = 0
    for site in SITES:
        grid = gpd.read_file(
            PROJECT_ROOT / "outputs" / site / "morphometrics" / "grid" / "grid_metrics.gpkg"
        )
        grid = aggregate_solar(
            grid,
            PROJECT_ROOT / "outputs" / site / "morphometrics" / "svf" / "svf_streets_solar.gpkg",
        )
        built = (grid["lambda_p"].fillna(0) > 0.01) | (grid["building_count"] > 0)
        keep = built & grid["solar_hours_winter"].notna() & grid["lambda_f_mean"].notna()
        g = grid[keep].copy()
        keep_n += len(g)
        aspect = np.deg2rad(g["aspect_deg"])
        g["southness"] = -np.cos(aspect)
        g["eastness"] = np.sin(aspect)
        for f in FEATURES:
            missing[f] += int(g[f].isna().sum())
        # Flat cell → undefined aspect → neutral directional signal; single-
        # building cell → undefined height spread → zero.
        flat = g["aspect_deg"].isna()
        g.loc[flat, ["southness", "eastness", "slope_deg"]] = 0.0
        g["sigma_h"] = g["sigma_h"].fillna(0.0)
        g["sun_fail"] = (g["solar_hours_winter"] < THRESHOLD_SUN_HRS).astype(int)
        g["site"] = site
        frames.append(g[REDUCED_FEATURES + ["site", "sun_fail"]])
    df = pd.concat(frames, ignore_index=True)
    audit = {
        "keep_n": keep_n,
        "missing_per_feature": dict(sorted(missing.items(), key=lambda x: -x[1])),
        "missing_frac_per_feature": {f: missing[f] / keep_n for f in missing},
    }
    return df, audit


def loso_reduced(df: pd.DataFrame, feats: list[str], sites: list[str]) -> dict:
    """Leave-one-site-out RF AUC on the reduced feature set. Pure: operates on
    whatever frame/sites are passed (so it is testable on synthetic data)."""
    per_site = {}
    for held_out in sites:
        tr = df[df["site"] != held_out]
        te = df[df["site"] == held_out]
        if te["sun_fail"].nunique() < 2 or tr["sun_fail"].nunique() < 2:
            continue
        rf = RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=50,
            class_weight="balanced", n_jobs=-1, random_state=0,
        )
        rf.fit(tr[feats], tr["sun_fail"])
        proba = rf.predict_proba(te[feats])[:, 1]
        per_site[held_out] = float(roc_auc_score(te["sun_fail"], proba))
    vals = list(per_site.values())
    return {
        "per_site_auc": per_site,
        "auc_mean": float(np.mean(vals)) if vals else float("nan"),
        "auc_min": float(min(vals)) if vals else float("nan"),
        "auc_max": float(max(vals)) if vals else float("nan"),
    }


def main() -> None:
    df, audit = build_fullsample_table()
    print(f"  full sample n={len(df):,}  sun_fail prevalence={df['sun_fail'].mean():.3f}")
    loso = loso_reduced(df, REDUCED_FEATURES, SITES)
    for s, a in loso["per_site_auc"].items():
        print(f"    {s:20s} AUC={a:.3f}")
    print(f"  reduced-feature LOSO AUC {loso['auc_min']:.3f}–{loso['auc_max']:.3f} "
          f"(mean {loso['auc_mean']:.3f})")

    canonical = {}
    canon_path = PROJECT_ROOT / "outputs" / "paper_figures" / "rf_predictor_stats.json"
    if canon_path.exists():
        cj = json.loads(canon_path.read_text())
        canonical = {
            "complete_case_n": cj.get("pooled_n"),
            "complete_case_prevalence": cj.get("sun_fail_rate"),
            "full_feature_auc_min": cj["loso"]["auc_min"],
            "full_feature_auc_max": cj["loso"]["auc_max"],
            "full_feature_auc_mean": cj["loso"]["auc_mean"],
        }

    out = {
        "purpose": (
            "Full-sample, geometry-only LOSO bound on the sunlight-constraint "
            "predictor. Tests whether the headline transfers beyond the "
            "street-adjacent complete-case subset. Aspect imputation is a no-op "
            "here (aspect is ~complete); SVF + street-entropy carry the ~50% "
            "missingness and are deliberately excluded rather than imputed."
        ),
        "reduced_features": REDUCED_FEATURES,
        "full_sample_n": int(len(df)),
        "full_sample_prevalence": float(df["sun_fail"].mean()),
        "missingness_audit": audit,
        "reduced_feature_loso": loso,
        "canonical_complete_case_full_feature": canonical,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"  wrote {OUT_JSON.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
