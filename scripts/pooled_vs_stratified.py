#!/usr/bin/env python3
"""Pooled-vs-stratified logistic comparison + spatial-block-robust CIs.

Closes two brisaverse-handoff caveats on the sunlight-constraint predictor
(companion to ``run_predictor_analysis.py``, whose model spec this reuses):

  item 2 — does ONE pooled slope-by-morphology logistic actually span hillside
           and flatland, or would two typology-stratified models fit better?
           Answered by Delta-AIC (pooled vs hillside+flatland) plus the
           interaction-sign check within each typology.

  item 4 — the naive parametric 95% CIs from ``run_predictor_analysis`` assume
           independent cells; on the contiguous 10 m grid they are
           anticonservative. Recompute cluster-robust CIs with spatial blocks
           (site x 200 m tiles) as the clustering unit.

Model spec is identical to run_predictor_analysis.pooled_logit_with_interactions:
standardized features + slope_x_southness + slope_x_svf, statsmodels Logit.
Note: ``southness`` = -cos(aspect), so the manuscript's "slope x northness"
coefficient is the negation of ``slope_x_southness`` here.

Output: outputs/paper_figures/pooled_vs_stratified.json
"""

from __future__ import annotations

import json
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from scripts.run_predictor_analysis import (
    FEATURES,
    PROJECT_ROOT,
    SITES,
    THRESHOLD_SUN_HRS,
    aggregate_solar,
)

warnings.filterwarnings("ignore")

HILLSIDE = ["vidigal", "rocinha", "complexo_do_alemao"]
FLATLAND = ["maré", "riodaspedras"]
BLOCK_M = 200.0  # spatial-block edge (m) for the cluster-robust covariance
INTERACTIONS = ["slope_x_southness", "slope_x_svf"]
OUT_JSON = PROJECT_ROOT / "outputs" / "paper_figures" / "pooled_vs_stratified.json"


def load_site_xy(site: str) -> pd.DataFrame:
    """Faithful mirror of run_predictor_analysis.load_site, retaining the cell
    centroid coordinates that the canonical loader drops."""
    grid = gpd.read_file(
        PROJECT_ROOT / "outputs" / site / "morphometrics" / "grid" / "grid_metrics.gpkg"
    )
    grid = aggregate_solar(
        grid, PROJECT_ROOT / "outputs" / site / "morphometrics" / "svf" / "svf_streets_solar.gpkg"
    )
    built = (grid["lambda_p"].fillna(0) > 0.01) | (grid["building_count"] > 0)
    keep = built & grid["solar_hours_winter"].notna() & grid["lambda_f_mean"].notna()
    g = grid[keep].copy()
    aspect = np.deg2rad(g["aspect_deg"])
    g["southness"] = -np.cos(aspect)
    g["eastness"] = np.sin(aspect)
    g["site"] = site
    g["sun_fail"] = (g["solar_hours_winter"] < THRESHOLD_SUN_HRS).astype(int)
    needed = FEATURES + ["site", "sun_fail", "centroid_x", "centroid_y"]
    return g[needed].dropna(subset=FEATURES)


def design(df: pd.DataFrame) -> pd.DataFrame:
    fs = df[FEATURES].copy()
    fs = (fs - fs.mean()) / fs.std()
    fs["slope_x_southness"] = fs["slope_deg"] * fs["southness"]
    fs["slope_x_svf"] = fs["slope_deg"] * fs["svf"]
    return sm.add_constant(fs)


def fit_logit(df: pd.DataFrame, cov_type=None, groups=None):
    X = design(df)
    y = df["sun_fail"].values
    kw = {"disp": False, "maxiter": 200}
    if cov_type == "cluster":
        kw["cov_type"] = "cluster"
        kw["cov_kwds"] = {"groups": groups}
    return sm.Logit(y, X).fit(**kw)


def coef_block(model, names) -> dict:
    ci = model.conf_int(alpha=0.05)
    out = {}
    for n in names:
        out[n] = {
            "beta": float(model.params[n]),
            "se": float(model.bse[n]),
            "ci_low": float(ci.loc[n, 0]),
            "ci_high": float(ci.loc[n, 1]),
        }
    return out


def main() -> None:
    print("loading per-site data with coordinates ...")
    df = pd.concat([load_site_xy(s) for s in SITES], ignore_index=True)
    print(f"  pooled n={len(df):,}  sun_fail rate={df['sun_fail'].mean():.4f}")

    report_names = ["slope_x_southness", "slope_x_svf", "svf", "slope_deg", "southness", "const"]

    # ---- item 2: pooled vs typology-stratified (Delta-AIC) --------------------
    print("fitting pooled + stratified logistic ...")
    m_pool = fit_logit(df)
    df_h = df[df["site"].isin(HILLSIDE)]
    df_f = df[df["site"].isin(FLATLAND)]
    m_h = fit_logit(df_h)
    m_f = fit_logit(df_f)

    aic_pool = float(m_pool.aic)
    aic_h = float(m_h.aic)
    aic_f = float(m_f.aic)
    aic_strat = aic_h + aic_f
    delta_aic = aic_pool - aic_strat  # <0 => pooled preferred; >0 => stratified

    # stratified held-out AUC: pooled model vs typology-specific models, each
    # scored by leave-one-SITE-out (train excludes the held-out site).
    def loso_auc(train_sites_fn) -> dict:
        per = {}
        for held in SITES:
            tr = df[(df["site"] != held) & (df["site"].isin(train_sites_fn(held)))]
            te = df[df["site"] == held]
            if te["sun_fail"].nunique() < 2 or len(tr) < 50:
                continue
            rf = RandomForestClassifier(
                n_estimators=300, min_samples_leaf=10, n_jobs=-1,
                random_state=42, class_weight="balanced",
            )
            rf.fit(tr[FEATURES].values, tr["sun_fail"].values)
            p = rf.predict_proba(te[FEATURES].values)[:, 1]
            per[held] = float(roc_auc_score(te["sun_fail"].values, p))
        return per

    auc_pooled = loso_auc(lambda held: SITES)  # train on all other sites
    auc_strat = loso_auc(  # train only on same-typology other sites
        lambda held: HILLSIDE if held in HILLSIDE else FLATLAND
    )

    # interaction signs within each typology (the real "does it hold in both")
    strat_coefs = {
        "hillside": coef_block(m_h, INTERACTIONS),
        "flatland": coef_block(m_f, INTERACTIONS),
    }

    # ---- item 4: spatial-block cluster-robust CIs ----------------------------
    print("spatial-block cluster-robust refit ...")
    bx = np.floor(df["centroid_x"].to_numpy() / BLOCK_M).astype(int)
    by = np.floor(df["centroid_y"].to_numpy() / BLOCK_M).astype(int)
    block_id = df["site"].astype(str) + "_" + bx.astype(str) + "_" + by.astype(str)
    n_blocks = int(block_id.nunique())
    m_robust = fit_logit(df, cov_type="cluster", groups=block_id.values)

    naive_ci = coef_block(m_pool, report_names)
    robust_ci = coef_block(m_robust, report_names)

    out = {
        "purpose": "Pooled-vs-stratified Delta-AIC + spatial-block cluster-robust CIs "
                   "for the sunlight-constraint logistic. Closes brisaverse items 2 and 4.",
        "n": int(len(df)),
        "n_hillside": int(len(df_h)),
        "n_flatland": int(len(df_f)),
        "sun_fail_rate": float(df["sun_fail"].mean()),
        "item2_pooled_vs_stratified": {
            "aic_pooled": aic_pool,
            "aic_hillside": aic_h,
            "aic_flatland": aic_f,
            "aic_stratified_sum": aic_strat,
            "delta_aic_pooled_minus_stratified": delta_aic,
            "preferred": "pooled" if delta_aic < 0 else "stratified",
            "pseudo_r2_pooled": float(m_pool.prsquared),
            "pseudo_r2_hillside": float(m_h.prsquared),
            "pseudo_r2_flatland": float(m_f.prsquared),
            "interaction_signs_by_typology": strat_coefs,
            "loso_auc_pooled_training": auc_pooled,
            "loso_auc_pooled_mean": float(np.mean(list(auc_pooled.values()))) if auc_pooled else None,
            "loso_auc_stratified_training": auc_strat,
            "loso_auc_stratified_mean": float(np.mean(list(auc_strat.values()))) if auc_strat else None,
        },
        "item4_spatial_block_robust_ci": {
            "block_edge_m": BLOCK_M,
            "n_blocks": n_blocks,
            "naive_parametric": naive_ci,
            "spatial_block_cluster_robust": robust_ci,
        },
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    # ---- console summary -----------------------------------------------------
    print("\n=== item 2: pooled vs stratified ===")
    print(f"  AIC pooled     = {aic_pool:,.1f}  (11 params, n={len(df):,})")
    print(f"  AIC hillside   = {aic_h:,.1f}")
    print(f"  AIC flatland   = {aic_f:,.1f}")
    print(f"  AIC stratified = {aic_strat:,.1f}  (22 params)")
    print(f"  Delta-AIC (pooled - stratified) = {delta_aic:,.1f}  -> {out['item2_pooled_vs_stratified']['preferred']} preferred")
    print(f"  LOSO AUC  pooled-train mean = {out['item2_pooled_vs_stratified']['loso_auc_pooled_mean']}")
    print(f"  LOSO AUC  strat-train  mean = {out['item2_pooled_vs_stratified']['loso_auc_stratified_mean']}")
    for typ, cb in strat_coefs.items():
        s = "  ".join(f"{k}={v['beta']:+.3f}[{v['ci_low']:+.3f},{v['ci_high']:+.3f}]" for k, v in cb.items())
        print(f"    {typ:9s}: {s}")
    print("\n=== item 4: spatial-block cluster-robust CIs ===")
    print(f"  n_blocks = {n_blocks} ({BLOCK_M:.0f} m tiles)")
    for n in INTERACTIONS + ["svf"]:
        a, b = naive_ci[n], robust_ci[n]
        print(f"    {n:18s} beta={a['beta']:+.3f}  naive CI[{a['ci_low']:+.3f},{a['ci_high']:+.3f}]  "
              f"robust CI[{b['ci_low']:+.3f},{b['ci_high']:+.3f}]  (SE {a['se']:.3f}->{b['se']:.3f})")
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
