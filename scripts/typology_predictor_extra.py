"""Typology → environmental-failure predictor: the two payoff deliverables.

Companion to ``scripts/analyze_typology_predictor.py`` (which does Steps 1–4:
the per-type lookup, parsimony, LOSO transfer, and the base calibration curve).
This script adds the two pieces that close the loop in
``docs/typology_predictor_plan.md``:

1. **Isotonic recalibration** of the leave-one-site-out (LOSO) WHO-2 h failure
   model (the ``both`` predictor set: morphotype + morphotope one-hot + the
   continuous fabric vector). The base LOSO model is slightly under-confident;
   we refit an isotonic calibrator *trained only on the 4 training sites per
   fold* (an inner LOSO over those 4 supplies the out-of-fold scores the
   isotonic map is fit on) and apply it to the held-out 5th site. Output is a
   before/after reliability diagram + the Expected Calibration Error (ECE)
   before and after. Figure: ``typology_calibration_isotonic.png``.

2. **Blind risk map** on the 3 *calibration* favelas
   (borel, jacarezinho, morro_do_juramento) — held entirely out of every model
   above (their grids carry the fabric vector but no morphotype/morphotope).
   We (a) refit the canonical cell GMM on the 5 campaign sites with the lean
   signature vector + assignment logic in ``src.morphometry.signature``,
   (b) project each calibration site's built cells through the FROZEN
   standardiser + GMM to assign a morphotype, (c) apply the per-morphotype
   WHO-2 h failure rate learned on the campaign sites, and (d) render a
   per-favela choropleth (3 panels, shared scale, NULL = grey). Morphology in →
   prioritised risk out, for favelas the model never saw.
   Figure: ``typology_blind_riskmap.png``.

    python scripts/typology_predictor_extra.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

from src.morphometry.signature import (  # noqa: E402
    CAMPAIGN_SITES,
    SIGNATURE_FEATURES,
    assemble_signature_matrix,
    standardize,
)
from src.viz.signature_style import TYPE_LABEL  # noqa: E402

FIGS = ROOT / "outputs" / "cross_site" / "signature" / "figures_v2"
OUT = ROOT / "outputs" / "cross_site" / "typology_predictor_extra"
TARGET = "solar_winter_frac_below2h"
FAIL_THRESH = 0.5
# See analyze_typology_predictor.CONT_FEATURES: party_wall_ratio is a separate
# per-building analysis, not a grid feature, and the dissolved λf already encodes
# the party-wall structure — so the continuous vector is the 5 canonical columns.
CONT_FEATURES = ["lambda_p", "lambda_f_mean", "H_mean", "sigma_h", "slope_deg"]
CALIBRATION_SITES = ["borel", "jacarezinho", "morro_do_juramento"]
K = 6


# --------------------------------------------------------------------------- #
# Shared data loading
# --------------------------------------------------------------------------- #
def load_full() -> pd.DataFrame:
    """Supported campaign cells with the binary WHO-2 h target + features + type.

    Mirrors ``analyze_typology_predictor.load_full`` so the recalibration
    operates on the exact same rows as the base LOSO model it is correcting.
    """
    frames = []
    for s in CAMPAIGN_SITES:
        p = ROOT / "outputs" / s / "features" / "features_grid.parquet"
        if not p.exists():
            continue
        g = pd.DataFrame(gpd.read_parquet(p).drop(columns="geometry"))
        g = g[g.get("has_street_support", False) & g["morphotype_smooth"].notna()]
        g = g.dropna(subset=[TARGET])
        g["site"] = s
        g["fail"] = (g[TARGET] > FAIL_THRESH).astype(int)
        g["sigma_h"] = g["sigma_h"].fillna(0.0)
        frames.append(g)
    df = pd.concat(frames, ignore_index=True)
    return df.dropna(subset=CONT_FEATURES)


def _design_both(df: pd.DataFrame) -> np.ndarray:
    """The 'both' predictor matrix: type one-hot + continuous fabric vector.

    Columns are aligned to the full label sets (0..5 morphotype, 0..4
    morphotope) so train/test designs never disagree on width across folds.
    """
    mt = pd.get_dummies(df["morphotype_smooth"].astype(int), prefix="mt")
    mt = mt.reindex(columns=[f"mt_{i}" for i in range(6)], fill_value=0)
    mtop = pd.get_dummies(df["morphotope"].astype("Int64"), prefix="mtop")
    mtop = mtop.reindex(columns=[f"mtop_{i}" for i in range(5)], fill_value=0)
    cont = df[CONT_FEATURES].reset_index(drop=True)
    X = pd.concat([mt.reset_index(drop=True), mtop.reset_index(drop=True), cont],
                  axis=1)
    return X.astype(float).to_numpy()


# --------------------------------------------------------------------------- #
# Deliverable 1 — isotonic recalibration of the LOSO 'both' model
# --------------------------------------------------------------------------- #
def _fit_logreg(Xtr, ytr):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(sc.transform(Xtr), ytr)
    return sc, clf


def _score(sc, clf, X):
    return clf.predict_proba(sc.transform(X))[:, 1]


def loso_with_isotonic(df: pd.DataFrame):
    """LOSO over the 5 campaign sites returning RAW and ISOTONIC-recalibrated
    out-of-fold probabilities for the held-out site.

    Per outer fold (held-out site = the 5th favela):
      * fit the base logistic model on the 4 training sites,
      * RAW score = that model on the held-out site,
      * to recalibrate *without touching the held-out site*, run an inner LOSO
        across the 4 training sites to get honest out-of-fold scores there, fit
        an IsotonicRegression on (inner-oof score → fail), then map the
        held-out site's raw scores through it.
    """
    from sklearn.isotonic import IsotonicRegression

    X = _design_both(df)
    y = df["fail"].to_numpy()
    site = df["site"].to_numpy()
    raw = np.full(len(y), np.nan)
    iso = np.full(len(y), np.nan)

    for s in CAMPAIGN_SITES:
        tr, te = site != s, site == s
        if te.sum() == 0 or y[tr].sum() == 0:
            continue
        sc, clf = _fit_logreg(X[tr], y[tr])
        raw[te] = _score(sc, clf, X[te])

        # inner LOSO over the 4 training sites → out-of-fold scores for isotonic
        train_sites = [t for t in CAMPAIGN_SITES if t != s]
        inner_oof = np.full(tr.sum(), np.nan)
        Xtr, ytr, str_ = X[tr], y[tr], site[tr]
        for u in train_sites:
            itr, ite = str_ != u, str_ == u
            if ite.sum() == 0 or ytr[itr].sum() == 0:
                continue
            isc, iclf = _fit_logreg(Xtr[itr], ytr[itr])
            inner_oof[ite] = _score(isc, iclf, Xtr[ite])
        m = np.isfinite(inner_oof)
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        ir.fit(inner_oof[m], ytr[m])
        iso[te] = ir.predict(raw[te])

    return raw, iso, y


def expected_calibration_error(y, p, n_bins=10):
    """ECE with equal-width bins (the standard reliability-diagram weighting)."""
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(y[m].mean() - p[m].mean())
    return ece


def _reliability(ax, y, p, color, label):
    from sklearn.calibration import calibration_curve

    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    ax.plot(mean_pred, frac_pos, "o-", color=color, label=label, ms=4)


def fig_isotonic(df: pd.DataFrame) -> dict:
    raw, iso, y = loso_with_isotonic(df)
    ok = np.isfinite(raw) & np.isfinite(iso)
    yk, rk, ik = y[ok], raw[ok], iso[ok]
    ece_raw = expected_calibration_error(yk, rk)
    ece_iso = expected_calibration_error(yk, ik)

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([0, 1], [0, 1], ":", color="0.6", lw=1, label="perfect")
    _reliability(ax, yk, rk, "#d6604d", f"raw LOSO (ECE {ece_raw:.3f})")
    _reliability(ax, yk, ik, "#1a6fb5", f"isotonic recal (ECE {ece_iso:.3f})")
    ax.set(xlabel="predicted failure probability",
           ylabel="observed failure rate", xlim=(0, 1), ylim=(0, 1))
    verdict = ("raw is already well-calibrated; isotonic adds nothing"
               if ece_iso >= ece_raw - 0.002 else
               f"isotonic improves ECE by {ece_raw - ece_iso:.3f}")
    ax.set_title("Isotonic recalibration of the LOSO WHO-2h model\n"
                 f"(both set, n={ok.sum()} OOF cells) — {verdict}",
                 fontsize=8.5)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(color="0.93")
    fig.tight_layout()
    fig.savefig(FIGS / "typology_calibration_isotonic.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    return {"ece_raw": ece_raw, "ece_iso": ece_iso, "n_oof": int(ok.sum())}


# --------------------------------------------------------------------------- #
# Deliverable 2 — blind risk map on the 3 calibration favelas
# --------------------------------------------------------------------------- #
def fit_canonical_gmm():
    """Refit the canonical k=6 cell GMM on the 5 campaign sites and return the
    frozen standardiser stats + a predict function over the lean signature
    vector. Labels are ordered by ascending lambda_p (same convention as the
    signature module), so the per-type meaning is stable.
    """
    from sklearn.mixture import GaussianMixture

    frames = []
    for s in CAMPAIGN_SITES:
        g = pd.DataFrame(
            gpd.read_parquet(ROOT / "outputs" / s / "features"
                             / "features_grid.parquet").drop(columns="geometry"))
        g["site"] = s
        frames.append(g)
    df = pd.concat(frames, ignore_index=True)
    mat = assemble_signature_matrix(df)
    Xz, stats = standardize(mat)

    # Refit the GMM with the SAME label-ordering convention as fit_morphotypes,
    # but keep the fitted object (not just labels) so we can project new cells.
    gm = GaussianMixture(n_components=K, covariance_type="full",
                         random_state=0, n_init=4).fit(Xz)
    raw = gm.predict(Xz)
    order = np.argsort([Xz[raw == c, 0].mean() for c in range(K)])
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[c] for c in raw])

    mu = stats["mean"].to_numpy()
    sd = stats["std"].to_numpy()

    def assign(feat_df: pd.DataFrame) -> np.ndarray:
        Xn = (feat_df[SIGNATURE_FEATURES].to_numpy(float) - mu) / sd
        rawp = gm.predict(Xn)
        return np.array([remap[c] for c in rawp])

    return assign, labels, mat, df


def per_type_failure_rate(mat: pd.DataFrame, labels: np.ndarray,
                          pooled: pd.DataFrame) -> dict:
    """Per-morphotype WHO-2 h failure rate learned on the campaign sites.

    Uses the same fresh GMM labels (so the rate matches the assignment the blind
    sites receive). Rate = mean of the continuous WHO-2 h failure intensity over
    supported cells of that type, equal-site-weighted to avoid the largest
    favela dominating the lookup.
    """
    key = mat[["site", "zone_id"]].copy()
    key["mt_fresh"] = labels
    j = key.merge(pooled[["site", "zone_id", TARGET, "has_street_support"]],
                  on=["site", "zone_id"], how="left")
    j = j[j["has_street_support"].fillna(False) & j[TARGET].notna()]
    rate = {}
    for c in range(K):
        sub = j[j["mt_fresh"] == c]
        if len(sub) == 0:
            rate[c] = np.nan
            continue
        # equal-site-weight: mean of per-site means
        per_site = sub.groupby("site")[TARGET].mean()
        rate[c] = float(per_site.mean())
    return rate


def blind_riskmap(assign, type_rate: dict) -> pd.DataFrame:
    """Project each calibration favela through the frozen GMM, map type→rate,
    render the 3-panel choropleth, and return per-site summary stats."""
    panels = []
    summary_rows = []
    for s in CALIBRATION_SITES:
        g = gpd.read_parquet(ROOT / "outputs" / s / "features"
                             / "features_grid.parquet")
        g = g.copy()
        # derive lambda_f_aniso + impute sigma_h exactly as the campaign cells
        # were prepared, so the frozen standardiser sees identical features.
        g["lambda_f_aniso"] = (g["lambda_f_max"] - g["lambda_f_mean"]).clip(lower=0)
        g["sigma_h"] = g["sigma_h"].fillna(0.0)
        built = g["built_mask"].fillna(False)
        ok = built & g[SIGNATURE_FEATURES].notna().all(axis=1)
        g["mt_blind"] = np.nan
        g.loc[ok, "mt_blind"] = assign(g.loc[ok, SIGNATURE_FEATURES])
        g["pred_fail"] = g["mt_blind"].map(
            lambda c: type_rate.get(int(c), np.nan) if pd.notna(c) else np.nan)
        n_built = int(built.sum())
        n_assign = int(ok.sum())
        mean_pred = float(g.loc[ok, "pred_fail"].mean())
        panels.append((s, g))
        summary_rows.append({
            "site": s, "n_built": n_built, "n_assignable": n_assign,
            "frac_assignable": n_assign / max(n_built, 1),
            "mean_pred_failure": mean_pred,
        })

    # shared colour scale across all 3 panels
    all_vals = np.concatenate([g.loc[g["pred_fail"].notna(), "pred_fail"].to_numpy()
                               for _, g in panels])
    norm = Normalize(vmin=float(np.nanmin(all_vals)), vmax=float(np.nanmax(all_vals)))
    cmap = plt.cm.inferno_r

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.2))
    for ax, (s, g) in zip(axes, panels):
        null = g[g["pred_fail"].isna()]
        if len(null):
            null.plot(ax=ax, color="0.85", edgecolor="none", linewidth=0)
        ok = g[g["pred_fail"].notna()]
        ok.plot(ax=ax, column="pred_fail", cmap=cmap, norm=norm,
                edgecolor="none", linewidth=0)
        row = next(r for r in summary_rows if r["site"] == s)
        ax.set_title(f"{s}\nmean p̂ = {row['mean_pred_failure']*100:.0f}%  "
                     f"({row['n_assignable']}/{row['n_built']} cells)", fontsize=9)
        ax.set_axis_off()
        ax.set_aspect("equal")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.25, 0.06, 0.5, 0.025])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("predicted WHO-2h winter-sun failure (morphology-only, blind)",
                 fontsize=8)
    cb.ax.xaxis.set_major_formatter(lambda v, _: f"{v*100:.0f}%")
    fig.suptitle("Blind risk map: morphology in → prioritised WHO-2h sun-failure "
                 "out, for 3 favelas the model never saw\n(frozen campaign GMM "
                 "morphotype → per-type failure-rate lookup; grey = unbuilt / "
                 "unassignable)", fontsize=10, y=0.99)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.12, wspace=0.02)
    fig.savefig(FIGS / "typology_blind_riskmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(summary_rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    # Deliverable 1
    full = load_full()
    cal = fig_isotonic(full)
    pd.DataFrame([cal]).to_csv(OUT / "isotonic_calibration_ece.csv", index=False)
    print(f"[1] Isotonic recalibration (n={cal['n_oof']} OOF cells): "
          f"ECE raw {cal['ece_raw']:.4f} → isotonic {cal['ece_iso']:.4f}")

    # Deliverable 2
    assign, labels, mat, pooled = fit_canonical_gmm()
    type_rate = per_type_failure_rate(mat, labels, pooled)
    print("[2] Per-morphotype WHO-2h failure rate (campaign, equal-site-weight):")
    for c in range(K):
        r = type_rate[c]
        print(f"    T{c} {TYPE_LABEL.get(c, ''):<20} "
              f"{'nan' if np.isnan(r) else f'{r*100:5.1f}%'}")
    pd.DataFrame([{"morphotype": c, "label": TYPE_LABEL.get(c, ""),
                   "who2h_fail_rate": type_rate[c]} for c in range(K)]).to_csv(
        OUT / "per_type_failure_rate.csv", index=False)

    summary = blind_riskmap(assign, type_rate)
    summary.to_csv(OUT / "blind_riskmap_summary.csv", index=False)
    print("[2] Blind risk map per calibration site:")
    print(summary.assign(
        mean_pred_failure=(summary["mean_pred_failure"] * 100).round(1),
        frac_assignable=(summary["frac_assignable"] * 100).round(1)
    ).to_string(index=False))


if __name__ == "__main__":
    main()
