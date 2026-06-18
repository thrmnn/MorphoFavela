#!/usr/bin/env python3
"""Random-forest + pooled-logistic predictor analysis for sunlight constraint.

Pipeline:
  1. Load per-site grid + solar layers (same engine as build_diagnostic_map.py).
  2. Pool built cells across all 5 sites.
  3. Target: sunlight constraint (winter_sun < 2 h, binary).
  4. Features: SVF, slope_deg, northness, eastness, lambda_p, lambda_f_mean,
              sigma_h, street_orientation_entropy.
  5. Random forest classifier, leave-one-site-out 5-fold CV. Per-fold ROC-AUC
     and permutation importance. Pooled refit for final importance ranking.
  6. Partial-dependence curves for top 3 features (pooled refit).
  7. Pooled logistic regression with slope x northness and slope x SVF
     interaction terms (statsmodels Logit, robust SEs). Plus 5 LOSO refits
     for coefficient stability.

Outputs:
  outputs/paper_figures/rf_predictor_stats.json   — numerics for main.tex
  outputs/paper_figures/rf_pooled_data.parquet    — feature/target table
  outputs/paper_figures/rf_pd_curves.json         — partial-dependence values
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SITES = ["vidigal", "rocinha", "complexo_do_alemao", "maré", "riodaspedras"]

THRESHOLD_SUN_HRS = 2.0
THRESHOLD_LAMBDA_F = 0.35

FEATURES = [
    "svf",
    "slope_deg",
    "southness",
    "eastness",
    "lambda_p",
    "lambda_f_mean",
    "sigma_h",
    "street_orientation_entropy",
]
FEATURE_LABELS = {
    "svf": "SVF",
    "slope_deg": "slope",
    "southness": "southness",
    "eastness": "eastness",
    "lambda_p": r"$\lambda_p$",
    "lambda_f_mean": r"$\lambda_f$",
    "sigma_h": r"$\sigma_H$",
    "street_orientation_entropy": "street entropy",
}


def aggregate_solar(
    grid: gpd.GeoDataFrame, solar_path: Path, max_radius: float = 25.0, k: int = 3
) -> gpd.GeoDataFrame:
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


def load_site(site: str) -> pd.DataFrame:
    grid = gpd.read_file(
        PROJECT_ROOT / "outputs" / site / "morphometrics" / "grid" / "grid_metrics.gpkg"
    )
    grid = aggregate_solar(
        grid, PROJECT_ROOT / "outputs" / site / "morphometrics" / "svf" / "svf_streets_solar.gpkg"
    )
    built = (grid["lambda_p"].fillna(0) > 0.01) | (grid["building_count"] > 0)
    sun_known = grid["solar_hours_winter"].notna()
    vent_known = grid["lambda_f_mean"].notna()
    keep = built & sun_known & vent_known
    g = grid[keep].copy()
    aspect = np.deg2rad(g["aspect_deg"])
    # Southern-hemisphere convention: south-facing slopes get less winter sun,
    # north-facing get more. We expose "southness" = -cos(aspect) so the sign
    # of the coefficient is intuitive for a northern-hemisphere audience
    # (higher southness -> higher P(sunlight constraint)).
    g["southness"] = -np.cos(aspect)
    g["eastness"] = np.sin(aspect)
    g["site"] = site
    g["sun_fail"] = (g["solar_hours_winter"] < THRESHOLD_SUN_HRS).astype(int)
    g["vent_fail"] = (g["lambda_f_mean"] > THRESHOLD_LAMBDA_F).astype(int)
    needed = FEATURES + ["site", "sun_fail", "vent_fail", "solar_hours_winter"]
    out = g[needed].dropna(subset=FEATURES)
    return out


def pooled_dataset() -> pd.DataFrame:
    frames = [load_site(s) for s in SITES]
    df = pd.concat(frames, ignore_index=True)
    return df


def loso_cv_rf(df: pd.DataFrame, target: str = "sun_fail") -> dict:
    fold_results = []
    importance_runs = []
    for held_out in SITES:
        train = df[df["site"] != held_out]
        test = df[df["site"] == held_out]
        if test["sun_fail"].nunique() < 2:
            continue
        X_train = train[FEATURES].values
        y_train = train[target].values
        X_test = test[FEATURES].values
        y_test = test[target].values
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=10,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced",
        )
        rf.fit(X_train, y_train)
        proba = rf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        rf_score = rf.score(X_test, y_test)
        perm = permutation_importance(
            rf,
            X_test,
            y_test,
            n_repeats=10,
            random_state=42,
            n_jobs=-1,
        )
        importance_runs.append({f: float(v) for f, v in zip(FEATURES, perm.importances_mean)})
        fold_results.append(
            {
                "held_out": held_out,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "test_accuracy": float(rf_score),
                "test_roc_auc": float(auc),
                "permutation_importance": {
                    f: float(v) for f, v in zip(FEATURES, perm.importances_mean)
                },
            }
        )
    return {"fold_results": fold_results, "importance_runs": importance_runs}


def pooled_rf_with_pd(df: pd.DataFrame, target: str = "sun_fail") -> dict:
    X = df[FEATURES].values
    y = df[target].values
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=10,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    )
    rf.fit(X, y)
    perm = permutation_importance(
        rf,
        X,
        y,
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
        max_samples=10000,
    )
    importance = sorted(
        [(f, float(v)) for f, v in zip(FEATURES, perm.importances_mean)],
        key=lambda kv: -kv[1],
    )
    top3 = [f for f, _ in importance[:3]]
    pd_curves = {}
    for feat in top3:
        idx = FEATURES.index(feat)
        try:
            pd_result = partial_dependence(
                rf,
                X,
                [idx],
                kind="average",
                grid_resolution=40,
            )
            pd_curves[feat] = {
                "grid": pd_result["grid_values"][0].tolist(),
                "values": pd_result["average"][0].tolist(),
            }
        except Exception as e:
            pd_curves[feat] = {"error": str(e)}
    return {
        "n": int(len(df)),
        "permutation_importance": importance,
        "pd_curves": pd_curves,
        "top3": top3,
    }


def pooled_logit_with_interactions(df: pd.DataFrame, target: str = "sun_fail") -> dict:
    import statsmodels.api as sm

    feats_std = df[FEATURES].copy()
    feats_std = (feats_std - feats_std.mean()) / feats_std.std()
    feats_std["slope_x_southness"] = feats_std["slope_deg"] * feats_std["southness"]
    feats_std["slope_x_svf"] = feats_std["slope_deg"] * feats_std["svf"]
    X = sm.add_constant(feats_std)
    y = df[target].values
    try:
        model = sm.Logit(y, X).fit(disp=False, maxiter=200)
        out = {
            "n": int(len(df)),
            "converged": bool(model.mle_retvals["converged"]),
            "pseudo_r2": float(model.prsquared),
            "coefficients": {},
        }
        ci = model.conf_int(alpha=0.05)
        for name in X.columns:
            out["coefficients"][name] = {
                "estimate": float(model.params[name]),
                "se": float(model.bse[name]),
                "z": float(model.tvalues[name]),
                "p": float(model.pvalues[name]),
                "ci_low": float(ci.loc[name, 0]),
                "ci_high": float(ci.loc[name, 1]),
            }
        loso_coefs = []
        for held_out in SITES:
            sub = df[df["site"] != held_out]
            fs = sub[FEATURES].copy()
            fs = (fs - fs.mean()) / fs.std()
            fs["slope_x_southness"] = fs["slope_deg"] * fs["southness"]
            fs["slope_x_svf"] = fs["slope_deg"] * fs["svf"]
            Xs = sm.add_constant(fs)
            ys = sub[target].values
            try:
                m = sm.Logit(ys, Xs).fit(disp=False, maxiter=200)
                loso_coefs.append(
                    {
                        "held_out": held_out,
                        "coefficients": {name: float(m.params[name]) for name in Xs.columns},
                    }
                )
            except Exception:
                continue
        out["loso_folds"] = loso_coefs
        return out
    except Exception as e:
        return {"error": str(e)}


def derive_svf_threshold(pd_svf: dict) -> float | None:
    """Find SVF value where the partial-dependence curve crosses 0.5
    failure probability (downward — high SVF → low fail prob).
    Returns the SVF value where PD ≤ 0.5 occurs (interpolated).
    """
    if "error" in pd_svf:
        return None
    grid = np.array(pd_svf["grid"])
    vals = np.array(pd_svf["values"])
    order = np.argsort(grid)
    grid = grid[order]
    vals = vals[order]
    above = vals > 0.5
    transitions = np.where(np.diff(above.astype(int)) != 0)[0]
    if len(transitions) == 0:
        return None
    i = transitions[-1]
    if vals[i] == vals[i + 1]:
        return float(grid[i])
    t = (0.5 - vals[i]) / (vals[i + 1] - vals[i])
    return float(grid[i] + t * (grid[i + 1] - grid[i]))


def main() -> None:
    out_dir = PROJECT_ROOT / "outputs" / "paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading per-site data ...")
    df = pooled_dataset()
    print(f"  pooled n={len(df):,}  sun_fail rate={df['sun_fail'].mean():.3f}")
    df.to_parquet(out_dir / "rf_pooled_data.parquet")

    print("LOSO CV ...")
    loso = loso_cv_rf(df)
    aucs = [r["test_roc_auc"] for r in loso["fold_results"]]
    accs = [r["test_accuracy"] for r in loso["fold_results"]]
    print(f"  LOSO ROC-AUC range: {min(aucs):.3f} - {max(aucs):.3f}  (mean {np.mean(aucs):.3f})")
    print(f"  LOSO accuracy range: {min(accs):.3f} - {max(accs):.3f}  (mean {np.mean(accs):.3f})")

    print("pooled RF + partial dependence ...")
    pooled_rf = pooled_rf_with_pd(df)
    print("  Top features by permutation importance:")
    for f, v in pooled_rf["permutation_importance"][:5]:
        print(f"    {f:34s} {v:+.4f}")

    svf_thresh = None
    if "svf" in pooled_rf["pd_curves"]:
        svf_thresh = derive_svf_threshold(pooled_rf["pd_curves"]["svf"])
        if svf_thresh is not None:
            print(f"  SVF threshold where PD crosses 0.5: {svf_thresh:.3f}")

    print("pooled logistic + interactions ...")
    logit = pooled_logit_with_interactions(df)
    if "error" not in logit:
        print(f"  pseudo R^2 = {logit['pseudo_r2']:.3f}")
        for name in ("slope_x_southness", "slope_x_svf", "slope_deg", "southness", "svf"):
            if name in logit["coefficients"]:
                c = logit["coefficients"][name]
                print(
                    f"    {name:24s} beta={c['estimate']:+.3f}  p={c['p']:.3g}  "
                    f"95% CI [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]"
                )
    else:
        print(f"  logit ERROR: {logit['error']}")

    stats = {
        "thresholds": {"sun_hours_winter_min": THRESHOLD_SUN_HRS},
        "pooled_n": int(len(df)),
        "sun_fail_rate": float(df["sun_fail"].mean()),
        "loso": {
            "fold_results": loso["fold_results"],
            "auc_mean": float(np.mean(aucs)),
            "auc_min": float(min(aucs)),
            "auc_max": float(max(aucs)),
            "acc_mean": float(np.mean(accs)),
            "acc_min": float(min(accs)),
            "acc_max": float(max(accs)),
        },
        "pooled_rf": {
            "permutation_importance": pooled_rf["permutation_importance"],
            "top3": pooled_rf["top3"],
            "svf_threshold_pd_0_5": svf_thresh,
        },
        "logit_interactions": logit,
    }
    (out_dir / "rf_predictor_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False)
    )
    (out_dir / "rf_pd_curves.json").write_text(
        json.dumps(pooled_rf["pd_curves"], indent=2, ensure_ascii=False)
    )
    print(f"\nwrote {out_dir / 'rf_predictor_stats.json'}")
    print(f"wrote {out_dir / 'rf_pd_curves.json'}")
    print(f"wrote {out_dir / 'rf_pooled_data.parquet'}")


if __name__ == "__main__":
    main()
