#!/usr/bin/env python3
"""Diagnostic models for Fig 0.5 (predictors and typology contrast).

Produces four CSV artifacts under
``outputs/comparative/diagnostic_models/``:

1. ``rf_importance.csv``   — sklearn permutation importance (vent + sun).
2. ``pdp_curves.csv``      — partial-dependence x/y/CI for {SVF, λf,
                              slope} × {vent, sun}.
3. ``logit_coefs.csv``     — pooled logistic regression with main
                              effects + interactions, cluster-robust SE
                              on ``site``.
4. ``changepoint_svf_ach.csv`` — patch-level segmented (2-piece linear)
                                 regression of annual U_mean on SVF,
                                 with bootstrap 95% CI for the
                                 breakpoint.

The figure script ``docs/manuscript/figures/fig_0_5_predictors.py``
consumes these artifacts directly. Re-run this script when input data
changes (real CFD ingestion, new sites, etc.).

Usage::

    python scripts/run_diagnostic_models.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs" / "comparative" / "diagnostic_models"

SITES = ["vidigal", "rocinha", "complexo_do_alemao", "riodaspedras", "maré"]

PREDICTORS = [
    "svf",
    "lambda_p",
    "lambda_f_mean",
    "sigma_h",
    "street_orientation_entropy",
    "slope_deg",
    "northness",
    "eastness",
]
PRETTY = {
    "svf": "SVF",
    "lambda_p": "λp",
    "lambda_f_mean": "λf",
    "sigma_h": "σH",
    "street_orientation_entropy": "entropy",
    "slope_deg": "slope",
    "northness": "northness",
    "eastness": "eastness",
}
TARGETS = ["vent_fail", "sun_fail"]
PDP_PREDICTORS = ["svf", "lambda_f_mean", "slope_deg"]

THRESHOLD_U_VENT = 1.0  # m/s, Lawson stagnation
THRESHOLD_SUN_HRS = 2.0  # h winter solstice
ALPHA_ACH = 1.0 / 150.0  # canyon→indoor coupling
ACH_PER_U = 366.0  # synthetic-CFD calibration: ACH_canyon ≈ 366·U_mean

INTERACTIONS = [
    ("svf", "slope_deg"),  # canyon shading × terrain steepness
    ("lambda_p", "lambda_f_mean"),  # plan vs frontal density
    ("slope_deg", "northness"),  # terrain × aspect
]

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Cell-level master table
# ---------------------------------------------------------------------------
def _aggregate_solar(grid: gpd.GeoDataFrame, site: str) -> gpd.GeoDataFrame:
    sol_path = PROJECT_ROOT / "outputs" / site / "morphometrics" / "svf" / "svf_streets_solar.gpkg"
    if not sol_path.exists():
        grid = grid.copy()
        grid["solar_hours_winter"] = np.nan
        return grid
    sol = gpd.read_file(sol_path)[["solar_hours_winter", "geometry"]]
    j = gpd.sjoin(sol, grid[["zone_id", "geometry"]], how="inner", predicate="within")
    agg = j.groupby("zone_id")["solar_hours_winter"].mean().reset_index()
    return grid.merge(agg, on="zone_id", how="left")


def build_cell_table() -> pd.DataFrame:
    rows = []
    for site in SITES:
        gm = gpd.read_file(
            PROJECT_ROOT / "outputs" / site / "morphometrics" / "grid" / "grid_metrics.gpkg"
        )
        cfd = gpd.read_file(PROJECT_ROOT / "outputs" / site / "cfd_analysis" / "grid_with_cfd.gpkg")
        merged = gm.merge(
            cfd[["zone_id", "annual_cfd_U_mean", "annual_cfd_ach"]], on="zone_id", how="inner"
        )
        merged = _aggregate_solar(merged, site)
        merged["site"] = site
        # Derived predictors.
        merged["northness"] = np.cos(np.deg2rad(merged["aspect_deg"]))
        merged["eastness"] = np.sin(np.deg2rad(merged["aspect_deg"]))
        # Targets.
        merged["vent_fail"] = (merged["annual_cfd_U_mean"] < THRESHOLD_U_VENT).astype(int)
        merged["sun_fail"] = (merged["solar_hours_winter"] < THRESHOLD_SUN_HRS).astype(int)
        rows.append(
            merged[
                ["site", "zone_id"]
                + PREDICTORS
                + [
                    "annual_cfd_U_mean",
                    "annual_cfd_ach",
                    "solar_hours_winter",
                    "vent_fail",
                    "sun_fail",
                ]
            ]
        )
    df = pd.concat(rows, ignore_index=True)
    return df


# ---------------------------------------------------------------------------
# Random forest + permutation importance
# ---------------------------------------------------------------------------
def fit_rf(df: pd.DataFrame, target: str) -> tuple[RandomForestClassifier, pd.DataFrame, float]:
    sub = df.dropna(subset=PREDICTORS + [target])
    X = sub[PREDICTORS].to_numpy()
    y = sub[target].to_numpy()
    if y.sum() < 5 or y.sum() == len(y):
        raise RuntimeError(f"target '{target}' has no failure variation.")
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=10,
        random_state=0,
        n_jobs=-1,
        class_weight="balanced",
    )
    # Honest AUC via 5-fold stratified CV.
    auc = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for tr, te in skf.split(X, y):
        rf2 = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=10,
            random_state=0,
            n_jobs=-1,
            class_weight="balanced",
        ).fit(X[tr], y[tr])
        auc.append(roc_auc_score(y[te], rf2.predict_proba(X[te])[:, 1]))
    rf.fit(X, y)
    perm = permutation_importance(rf, X, y, n_repeats=20, random_state=0, n_jobs=-1)
    imp_df = pd.DataFrame(
        {
            "predictor": PREDICTORS,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    return rf, imp_df, float(np.mean(auc))


def compute_rf_importance(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    out = []
    for tgt in TARGETS:
        rf, imp_df, auc = fit_rf(df, tgt)
        models[tgt] = rf
        imp_df = imp_df.assign(target=tgt, auc_cv5=auc)
        out.append(imp_df)
        print(f"  RF {tgt}: 5-fold AUC = {auc:.3f}")
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------
# Partial dependence with bootstrap CI
# ---------------------------------------------------------------------------
def _pdp_one(rf, X: np.ndarray, feature_idx: int, grid: np.ndarray) -> np.ndarray:
    """Wrap sklearn.partial_dependence -> 1D probability curve."""
    pd_res = partial_dependence(
        rf,
        X,
        [feature_idx],
        kind="average",
        grid_resolution=len(grid),
        percentiles=(0.0, 1.0),
    )
    # sklearn returns response in {average, values}. With a binary classifier
    # the average is the predicted probability of class 1.
    avg = pd_res["average"][0]
    return avg


def compute_pdp(df: pd.DataFrame, n_boot: int = 50) -> pd.DataFrame:
    """Partial-dependence curves with bootstrap CI for each
    (target, predictor) ∈ TARGETS × PDP_PREDICTORS."""
    out = []
    sub = df.dropna(subset=PREDICTORS + TARGETS)
    X_full = sub[PREDICTORS].to_numpy()
    n = len(sub)

    for tgt in TARGETS:
        y_full = sub[tgt].to_numpy()
        for pred in PDP_PREDICTORS:
            j = PREDICTORS.index(pred)
            xrange = np.linspace(
                np.quantile(X_full[:, j], 0.02),
                np.quantile(X_full[:, j], 0.98),
                25,
            )
            # Point estimate.
            rf = RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=10,
                random_state=0,
                n_jobs=-1,
                class_weight="balanced",
            ).fit(X_full, y_full)
            base = partial_dependence(
                rf,
                X_full,
                [j],
                kind="average",
                grid_resolution=25,
            )
            xs = base["grid_values"][0]
            ys = base["average"][0]
            # Bootstrap.
            boots = np.zeros((n_boot, len(xs)))
            for b in range(n_boot):
                idx = RNG.integers(0, n, size=n)
                rfb = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=12,
                    min_samples_leaf=10,
                    random_state=b + 1,
                    n_jobs=-1,
                    class_weight="balanced",
                ).fit(X_full[idx], y_full[idx])
                pdb = partial_dependence(
                    rfb,
                    X_full[idx],
                    [j],
                    kind="average",
                    grid_resolution=25,
                )
                # Resample to the base grid via interpolation.
                boots[b] = np.interp(xs, pdb["grid_values"][0], pdb["average"][0])
            lo = np.quantile(boots, 0.025, axis=0)
            hi = np.quantile(boots, 0.975, axis=0)
            for k in range(len(xs)):
                out.append(
                    {
                        "target": tgt,
                        "predictor": pred,
                        "x": float(xs[k]),
                        "y": float(ys[k]),
                        "lo": float(lo[k]),
                        "hi": float(hi[k]),
                    }
                )
            print(
                f"  PDP {tgt} × {pred}: x∈[{xs.min():.2f}, {xs.max():.2f}] "
                f"y∈[{ys.min():.3f}, {ys.max():.3f}]"
            )
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Logistic regression with interactions, cluster-robust SE on site
# ---------------------------------------------------------------------------
def compute_logit(df: pd.DataFrame) -> pd.DataFrame:
    sub = df.dropna(subset=PREDICTORS + TARGETS).copy()
    # Standardize continuous predictors so coefficients are comparable.
    for p in PREDICTORS:
        sub[p + "_z"] = (sub[p] - sub[p].mean()) / sub[p].std(ddof=0)
    rows = []
    for tgt in TARGETS:
        terms = [p + "_z" for p in PREDICTORS]
        # Add interactions.
        ix_names = []
        for a, b in INTERACTIONS:
            n = f"{a}:{b}"
            sub[n + "_z"] = sub[a + "_z"] * sub[b + "_z"]
            ix_names.append(n + "_z")
            terms.append(n + "_z")
        X = sm.add_constant(sub[terms])
        y = sub[tgt]
        try:
            res = sm.GLM(y, X, family=sm.families.Binomial()).fit(
                cov_type="cluster",
                cov_kwds={"groups": sub["site"].astype("category").cat.codes},
            )
        except Exception as e:  # pragma: no cover
            print(f"  LOGIT {tgt}: ERR {e}")
            continue
        params = res.params.drop("const")
        ses = res.bse.drop("const")
        ps = res.pvalues.drop("const")
        for term in params.index:
            base = term.replace("_z", "")
            kind = "interaction" if ":" in base else "main"
            rows.append(
                {
                    "target": tgt,
                    "term": base,
                    "kind": kind,
                    "beta": float(params[term]),
                    "se": float(ses[term]),
                    "p": float(ps[term]),
                    "ci_lo": float(params[term] - 1.96 * ses[term]),
                    "ci_hi": float(params[term] + 1.96 * ses[term]),
                }
            )
        print(f"  LOGIT {tgt}: pseudo-R² = {1 - res.deviance / res.null_deviance:.3f}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Changepoint regression (2-segment piecewise linear)
# ---------------------------------------------------------------------------
def _fit_2seg(x: np.ndarray, y: np.ndarray, bp: float) -> tuple[np.ndarray, float]:
    """Fit y = a0 + a1·x + a2·max(0, x-bp), return (params, RSS)."""
    X = np.column_stack([np.ones_like(x), x, np.maximum(0.0, x - bp)])
    p, *_ = np.linalg.lstsq(X, y, rcond=None)
    rss = float(((y - X @ p) ** 2).sum())
    return p, rss


def changepoint_svf_ach(n_boot: int = 500) -> pd.DataFrame:
    rows = []
    for site in SITES:
        df = pd.read_csv(
            PROJECT_ROOT / "outputs" / site / "cfd_analysis" / "per_patch_indicators.csv"
        )
        df["site"] = site
        rows.append(df[["site", "patch_id", "svf", "annual_U_mean"]])
    pdf = pd.concat(rows, ignore_index=True).dropna()
    x = pdf["svf"].to_numpy()
    y = pdf["annual_U_mean"].to_numpy()

    bp_grid = np.linspace(0.10, 0.55, 91)

    def best_bp(xx: np.ndarray, yy: np.ndarray) -> tuple[float, np.ndarray, float]:
        rss = []
        for b in bp_grid:
            _, r = _fit_2seg(xx, yy, b)
            rss.append(r)
        idx = int(np.argmin(rss))
        bp = float(bp_grid[idx])
        p, r = _fit_2seg(xx, yy, bp)
        return bp, p, r

    bp_hat, params_hat, rss_hat = best_bp(x, y)

    # Bootstrap.
    bps = []
    for b in range(n_boot):
        idx = RNG.integers(0, len(x), size=len(x))
        try:
            bp_b, _, _ = best_bp(x[idx], y[idx])
            bps.append(bp_b)
        except Exception:
            pass
    bps = np.array(bps)
    bp_lo, bp_hi = np.quantile(bps, [0.025, 0.975])

    out = pd.DataFrame(
        {
            "patch_id": pdf["patch_id"].to_numpy(),
            "site": pdf["site"].to_numpy(),
            "svf": x,
            "annual_U_mean": y,
        }
    )
    out.attrs["bp_hat"] = bp_hat
    out.attrs["bp_lo"] = float(bp_lo)
    out.attrs["bp_hi"] = float(bp_hi)
    out.attrs["params"] = params_hat.tolist()

    # Stash fit + CI in a side CSV (params row + per-patch rows).
    meta = pd.DataFrame(
        [
            {
                "patch_id": "__fit__",
                "site": "__fit__",
                "svf": np.nan,
                "annual_U_mean": np.nan,
                "bp_hat": bp_hat,
                "bp_lo": float(bp_lo),
                "bp_hi": float(bp_hi),
                "a0": float(params_hat[0]),
                "a1": float(params_hat[1]),
                "a2": float(params_hat[2]),
            }
        ]
    )
    out["bp_hat"] = bp_hat
    out["bp_lo"] = float(bp_lo)
    out["bp_hi"] = float(bp_hi)
    out["a0"] = float(params_hat[0])
    out["a1"] = float(params_hat[1])
    out["a2"] = float(params_hat[2])

    print(
        f"  Changepoint SVF→U: bp = {bp_hat:.3f} "
        f"(95% CI {bp_lo:.3f}–{bp_hi:.3f}), "
        f"slopes a1={params_hat[1]:.2f}, a2={params_hat[2]:.2f}"
    )
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore")

    print("Building cell-level table ...")
    df = build_cell_table()
    n_cls = df.dropna(subset=PREDICTORS + TARGETS)
    print(f"  pooled cells: {len(df)}  classified: {len(n_cls)}")
    print(
        f"  vent_fail rate: {n_cls['vent_fail'].mean():.2%}, "
        f"sun_fail rate: {n_cls['sun_fail'].mean():.2%}"
    )

    print("\nFitting random forests + permutation importance ...")
    rf_models: dict[str, RandomForestClassifier] = {}
    rf_imp = compute_rf_importance(df, rf_models)
    rf_imp.to_csv(OUT_DIR / "rf_importance.csv", index=False)

    print("\nComputing partial-dependence curves with bootstrap CI ...")
    pdp = compute_pdp(df, n_boot=50)
    pdp.to_csv(OUT_DIR / "pdp_curves.csv", index=False)

    print("\nFitting pooled logistic regressions with interactions ...")
    logit = compute_logit(df)
    logit.to_csv(OUT_DIR / "logit_coefs.csv", index=False)

    print("\nFitting SVF → U_mean changepoint regression ...")
    cp = changepoint_svf_ach(n_boot=500)
    cp.to_csv(OUT_DIR / "changepoint_svf_ach.csv", index=False)

    print(f"\nWrote artifacts to {OUT_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
