"""Typology as predictor of environmental failure — Step 1 (the lookup figure).

Per morphotype / morphotope: the WHO-2 h winter-sun failure rate (the held-out,
ray-cast target), pooled across the 5 campaign favelas, with the per-site rates
overlaid so regime behaviour (does failure jump with type?) and transfer (do sites
agree?) are visible in one glance. Site-clustered bootstrap CIs (resample sites, not
cells — honest under spatial autocorrelation). The full plan (parsimony test, LOSO,
calibration, variance decomposition, blind risk map) is in
docs/typology_predictor_plan.md; this is its first, self-contained step.

    python scripts/analyze_typology_predictor.py
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

from src.morphometry.signature import CAMPAIGN_SITES  # noqa: E402
from src.viz.signature_style import (  # noqa: E402
    MORPHOTOPE_LABEL,
    TYPE_COLORS,
    TYPE_LABEL,
)

FIGS = ROOT / "outputs" / "cross_site" / "signature" / "figures_v2"
OUT = ROOT / "outputs" / "cross_site" / "typology_predictor"
TARGET = "solar_winter_frac_below2h"  # held-out WHO-2h failure intensity per cell


def load() -> pd.DataFrame:
    frames = []
    for s in CAMPAIGN_SITES:
        p = ROOT / "outputs" / s / "features" / "features_grid.parquet"
        if not p.exists():
            continue
        g = gpd.read_parquet(p)
        d = pd.DataFrame(g.drop(columns="geometry"))
        d = d[d.get("has_street_support", False) & d["morphotype_smooth"].notna()]
        d = d.dropna(subset=[TARGET])
        d["site"] = s
        frames.append(d[["site", "morphotype_smooth", "morphotope", TARGET]])
    return pd.concat(frames, ignore_index=True)


def site_clustered_ci(df, key, n_boot=2000, seed=0):
    """Per-class mean failure + 95% CI by resampling SITES (cluster-robust)."""
    rng = np.random.default_rng(seed)
    sites = df["site"].unique()
    classes = sorted(df[key].dropna().unique())
    per_site = {(s, c): df[(df.site == s) & (df[key] == c)][TARGET].mean()
                for s in sites for c in classes}
    rows = []
    for c in classes:
        # equal-site-weight point estimate (matches the site-resampling bootstrap)
        point = np.nanmean([per_site[(s, c)] for s in sites])
        boots = []
        for _ in range(n_boot):
            samp = rng.choice(sites, size=len(sites), replace=True)
            vals = [per_site[(s, c)] for s in samp if not np.isnan(per_site[(s, c)])]
            if vals:
                boots.append(np.mean(vals))
        lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
        rows.append({key: int(c), "mean": point, "lo": lo, "hi": hi,
                     "n_cells": int((df[key] == c).sum())})
    return pd.DataFrame(rows), per_site, classes, sites


def lookup_figure(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    cfg = [
        ("morphotype_smooth", "T", TYPE_LABEL.get, lambda c: TYPE_COLORS[c]),
        ("morphotope", "M", MORPHOTOPE_LABEL.get, lambda c: "#5aae61"),
    ]
    for ax, (key, prefix, namer, colorer) in zip(axes, cfg):
        tab, per_site, classes, sites = site_clustered_ci(df, key)
        x = np.arange(len(classes))
        lo_err = np.clip((tab["mean"] - tab["lo"]) * 100, 0, None)
        hi_err = np.clip((tab["hi"] - tab["mean"]) * 100, 0, None)
        ax.bar(x, tab["mean"] * 100, color=[colorer(c) for c in classes],
               yerr=[lo_err, hi_err], capsize=3, alpha=0.85,
               error_kw=dict(lw=1, ecolor="0.3"))
        for s in sites:
            ys = [per_site[(s, c)] * 100 for c in classes]
            ax.plot(x, ys, "o", ms=3, color="0.25", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{prefix}{c}\n{namer(c)}" for c in classes], fontsize=7)
        ax.set_ylabel("WHO-2h winter-sun failure (%)")
        ax.set_ylim(0, 100)
        ax.set_title(("Cell morphotype" if key == "morphotype_smooth" else
                      "Block morphotope") + " → sun-failure rate", fontsize=9)
        ax.grid(axis="y", color="0.93")
    fig.suptitle("Typology → environmental failure: per-type WHO-2h sun-failure rate "
                 "(bars = pooled ±95% site-bootstrap CI; dots = per-favela rates)",
                 fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIGS / "typology_failure_lookup.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return tab


FAIL_THRESH = 0.5  # cell "fails" WHO-2h if a majority of its observers fall below 2h
# Continuous fabric vector = the canonical grid columns. party_wall_ratio is NOT
# here: it is a separate per-building configuration analysis (party_wall_by_type.csv),
# never a grid feature, and the dissolved λf already encodes party-wall structure
# (touching footprints are unioned before projecting), so it would be redundant.
CONT_FEATURES = ["lambda_p", "lambda_f_mean", "H_mean", "sigma_h", "slope_deg"]


def load_full() -> pd.DataFrame:
    """Supported cells with the binary WHO-2h target + continuous features + type."""
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


def _design(df, which):
    """Predictor matrix: 'type' (morphotype+morphotope one-hot), 'vector'
    (continuous fabric), or 'both'."""
    parts = []
    if which in ("type", "both"):
        parts.append(pd.get_dummies(df["morphotype_smooth"].astype(int), prefix="mt"))
        parts.append(pd.get_dummies(df["morphotope"].astype("Int64"), prefix="mtop"))
    if which in ("vector", "both"):
        parts.append(df[CONT_FEATURES].reset_index(drop=True))
    X = pd.concat([p.reset_index(drop=True) for p in parts], axis=1)
    return X.astype(float).to_numpy()


def loso(df, which):
    """Leave-one-site-out: train on 4 favelas, predict the 5th. Returns per-fold
    AUC-PR/ROC/Brier + pooled out-of-fold predictions (the honest transfer test)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    X, y, site = _design(df, which), df["fail"].to_numpy(), df["site"].to_numpy()
    rows, oof_p, oof_y = [], np.full(len(y), np.nan), y.copy()
    for s in CAMPAIGN_SITES:
        tr, te = site != s, site == s
        if te.sum() == 0 or y[tr].sum() == 0:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(sc.transform(X[tr]), y[tr])
        p = clf.predict_proba(sc.transform(X[te]))[:, 1]
        oof_p[te] = p
        rows.append({"site": s, "n": int(te.sum()), "prevalence": float(y[te].mean()),
                     "auc_pr": average_precision_score(y[te], p),
                     "auc_roc": roc_auc_score(y[te], p) if len(set(y[te])) > 1 else np.nan,
                     "brier": brier_score_loss(y[te], p)})
    return pd.DataFrame(rows), oof_p, oof_y


def loso_block_bootstrap(df, which, n_boot=2000, seed=0):
    """Site-block bootstrap CI on the pooled out-of-fold AUC-PR. Each LOSO fold's
    held-out site is already fully out-of-sample (the strongest spatial blocking);
    this resamples the 5 SITES with replacement and re-pools their OOF predictions
    to put an honest uncertainty band on the transfer metric (n=5 → wide is correct).
    """
    from sklearn.metrics import average_precision_score

    _, oof_p, oof_y = loso(df, which)
    site = df["site"].to_numpy()
    ok = np.isfinite(oof_p)
    by = {}
    for s in CAMPAIGN_SITES:
        m = ok & (site == s)
        if m.sum() and len(set(oof_y[m])) > 1:
            by[s] = (oof_y[m], oof_p[m])
    keys = list(by)
    point = float(average_precision_score(oof_y[ok], oof_p[ok]))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        samp = rng.choice(keys, size=len(keys), replace=True)
        yy = np.concatenate([by[s][0] for s in samp])
        pp = np.concatenate([by[s][1] for s in samp])
        if len(set(yy)) > 1:
            boots.append(average_precision_score(yy, pp))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"point": point, "lo": float(lo), "hi": float(hi), "n_boot": len(boots)}


def vif_report(df):
    """Variance-inflation factors on the standardized continuous fabric vector —
    answers the 'is the 0.84 just one collinear feature re-badged' referee question.
    Note the vector contains NO SVF and NO solar term; it is pure fabric geometry."""
    from sklearn.preprocessing import StandardScaler
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    x = StandardScaler().fit_transform(df[CONT_FEATURES].to_numpy())
    return {f: float(variance_inflation_factor(x, i)) for i, f in enumerate(CONT_FEATURES)}


def fig_parsimony(results, cis=None):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    sets = ["type", "vector", "both"]
    labels = ["type only\n(morphotype+morphotope)", "continuous\nfabric vector", "both"]
    x = np.arange(3)
    means = [results[s][0]["auc_pr"].mean() for s in sets]
    # Bars sit at the pooled-OOF point (what the block-bootstrap CI bands) when CIs
    # are supplied, so bar height and error bar are the same quantity; the per-fold
    # AUC-PR dots overlay to show the 5 held-out-site spread.
    if cis is not None:
        bar_vals = [cis[s]["point"] for s in sets]
        yerr = np.clip(np.array([[bar_vals[i] - cis[s]["lo"], cis[s]["hi"] - bar_vals[i]]
                                 for i, s in enumerate(sets)]).T, 0, None)
        ax.bar(x, bar_vals, color=["#5aae61", "#9970ab", "#1a6fb5"], alpha=0.85,
               yerr=yerr, capsize=5, error_kw=dict(lw=1.2, ecolor="0.2"))
    else:
        ax.bar(x, means, color=["#5aae61", "#9970ab", "#1a6fb5"], alpha=0.85)
    for s, xi in zip(sets, x):
        ax.plot([xi] * len(results[s][0]), results[s][0]["auc_pr"], "o", ms=4,
                color="0.25", alpha=0.6)
    prev = results["type"][0]["prevalence"].mean()
    ax.axhline(prev, ls=":", color="crimson", lw=1)
    ax.text(2.4, prev + 0.01, f"no-skill (prevalence {prev:.2f})", fontsize=7,
            color="crimson", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("AUC-PR (leave-one-site-out)")
    ax.set_ylim(0, 1)
    gap = means[1] - means[0]
    # Data-driven verdict: under the dissolved-λf re-baseline the discrete
    # typology alone is only modestly skillful and the continuous vector carries
    # most of the transferable signal — so phrase by the measured gap, never a
    # fixed "type keeps most of the signal" claim that the numbers may contradict.
    verdict = ("the discrete code keeps most of the signal"
               if gap < 0.10 else
               "the continuous fabric vector carries most of the transferable signal")
    if cis is not None:
        # Honest test: does the vector>type gap survive the site-block-bootstrap CI?
        sep = "separated" if cis["type"]["hi"] < cis["vector"]["lo"] else "CI-overlapping"
        ci_txt = (f"\nsite-block-bootstrap 95% CI: type {cis['type']['lo']:.2f}–"
                  f"{cis['type']['hi']:.2f} vs vector {cis['vector']['lo']:.2f}–"
                  f"{cis['vector']['hi']:.2f} ({sep})")
    else:
        ci_txt = ""
    ax.set_title(f"Parsimony (leave-one-site-out): type-only transfers at AUC-PR "
                 f"{means[0]:.2f} vs {means[1]:.2f} for the full vector "
                 f"(baseline {prev:.2f}) — {verdict}; Δ{gap:+.3f}{ci_txt}", fontsize=7.5)
    ax.grid(axis="y", color="0.93")
    fig.tight_layout()
    fig.savefig(FIGS / "typology_parsimony.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_calibration(df):
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import average_precision_score, precision_recall_curve
    _, oof_p, oof_y = loso(df, "both")
    ok = np.isfinite(oof_p)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    frac_pos, mean_pred = calibration_curve(oof_y[ok], oof_p[ok], n_bins=10,
                                            strategy="quantile")
    axes[0].plot([0, 1], [0, 1], ":", color="0.6")
    axes[0].plot(mean_pred, frac_pos, "o-", color="#1a6fb5")
    axes[0].set(xlabel="predicted failure prob", ylabel="observed failure rate",
                title="Calibration (LOSO out-of-fold)", xlim=(0, 1), ylim=(0, 1))
    axes[0].grid(color="0.93")
    prec, rec, _ = precision_recall_curve(oof_y[ok], oof_p[ok])
    ap = average_precision_score(oof_y[ok], oof_p[ok])
    axes[1].plot(rec, prec, color="#5aae61")
    axes[1].axhline(oof_y[ok].mean(), ls=":", color="crimson", lw=1)
    axes[1].set(xlabel="recall", ylabel="precision",
                title=f"PR curve (LOSO), AP={ap:.2f}", xlim=(0, 1), ylim=(0, 1))
    axes[1].grid(color="0.93")
    fig.suptitle("Transfer is honest: calibrated out-of-site predictions + PR vs "
                 "the prevalence baseline", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIGS / "typology_calibration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_variance(df):
    """C1 — is failure driven by WHICH TYPES a favela has, or by SITE effects?
    Partition explained variance of cell failure into type / site / site×type via a
    two-way ANOVA (partial η²). Large site×type ⇒ the type→failure mapping shifts by
    site (transfer moderation); large between-type ⇒ a clean transferable proxy."""
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm

    d = df[["fail", "site"]].copy()
    d["mt"] = df["morphotype_smooth"].astype(int).astype(str)
    model = smf.ols("fail ~ C(mt) + C(site) + C(mt):C(site)", data=d).fit()
    aov = anova_lm(model, typ=2)
    ss = aov["sum_sq"]
    parts = {"morphotype\n(the proxy signal)": ss["C(mt)"],
             "site\n(topography the type misses)": ss["C(site)"],
             "site × type\n(transfer moderation)": ss["C(mt):C(site)"],
             "residual\n(within-group)": ss["Residual"]}
    tot = sum(parts.values())
    frac = {k: v / tot for k, v in parts.items()}
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    colors = ["#5aae61", "#d6604d", "#9970ab", "#cccccc"]
    left = 0
    for (k, v), c in zip(frac.items(), colors):
        ax.barh(0, v, left=left, color=c, edgecolor="white")
        if v > 0.03:
            ax.text(left + v / 2, 0, f"{v*100:.0f}%", ha="center", va="center",
                    fontsize=8, color="white" if c != "#cccccc" else "0.3",
                    fontweight="bold")
        left += v
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("share of explained + residual variance in cell WHO-2h failure")
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in colors],
              list(parts.keys()), fontsize=7, loc="upper center",
              bbox_to_anchor=(0.5, -0.35), ncol=2, frameon=False)
    f = list(frac.values())
    syst = f[0] + f[1] + f[2]
    # Data-driven: name whichever of type/site leads (they are now comparable),
    # and read the interaction term honestly — a tiny site×type still means the
    # (modest) type signal transfers, even though most variance is within-type.
    lead = "morphotype leads" if f[0] > f[1] else "site leads"
    transfer = ("negligible interaction ⇒ what type-signal exists transfers"
                if f[2] < 0.02 else
                "a sizeable site×type interaction ⇒ the type→failure mapping shifts by site")
    ax.set_title(f"Of the SYSTEMATIC variance ({syst*100:.0f}%; rest is within-group "
                 f"cell noise), {lead} — type {f[0]*100:.0f}% vs site "
                 f"{f[1]*100:.0f}% vs site×type {f[2]*100:.1f}%.\n{transfer}.", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIGS / "typology_variance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return frac


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    for key in ("morphotype_smooth", "morphotope"):
        tab, *_ = site_clustered_ci(df, key)
        tab.to_csv(OUT / f"failure_by_{key.replace('_smooth','')}.csv", index=False)
        print(f"\nWHO-2h failure rate by {key}:")
        print(tab.assign(**{c: (tab[c] * 100).round(1) for c in ("mean", "lo", "hi")})
              .to_string(index=False))
    lookup_figure(df)

    # Steps 2–4: parsimony + LOSO transfer + calibration
    full = load_full()
    results = {w: loso(full, w) for w in ("type", "vector", "both")}
    summary = pd.DataFrame({w: {"auc_pr": results[w][0]["auc_pr"].mean(),
                                "auc_roc": results[w][0]["auc_roc"].mean(),
                                "brier": results[w][0]["brier"].mean()}
                            for w in results}).T
    summary.to_csv(OUT / "loso_parsimony.csv")

    # Rank-1 (council): site-block-bootstrap CIs on the pooled-OOF AUC-PR + VIF audit.
    cis = {w: loso_block_bootstrap(full, w) for w in ("type", "vector", "both")}
    vif = vif_report(full)
    spatial_cv = {
        "note": ("Site-block-bootstrap 95% CI on pooled out-of-fold AUC-PR (resample "
                 "the 5 favelas with replacement). LOSO already holds each test site "
                 "fully out; this bands the transfer metric. VIF on the standardized "
                 "continuous fabric vector (no SVF, no solar term)."),
        "auc_pr_ci": cis,
        "gap_vector_minus_type_ci_separated": bool(cis["type"]["hi"] < cis["vector"]["lo"]),
        "vif_continuous_vector": vif,
    }
    import json as _json
    (OUT / "spatial_cv.json").write_text(_json.dumps(spatial_cv, indent=2))
    print("\nsite-block-bootstrap AUC-PR 95% CI:")
    for w in ("type", "vector", "both"):
        print(f"  {w:<7s} {cis[w]['point']:.3f}  [{cis[w]['lo']:.3f}, {cis[w]['hi']:.3f}]")
    print(f"  vector>type CI-separated: {spatial_cv['gap_vector_minus_type_ci_separated']}")
    print("VIF (continuous fabric vector):",
          {k: round(v, 1) for k, v in vif.items()})

    fig_parsimony(results, cis)
    fig_calibration(full)
    try:
        frac = fig_variance(full)
        print("\nvariance partition (share):",
              {k.split(chr(10))[0]: round(v, 3) for k, v in frac.items()})
    except ModuleNotFoundError:
        print("\n[variance partition skipped — needs statsmodels (queued)]")
    print(f"\nLOSO transfer (n={len(full)} cells, prevalence "
          f"{full['fail'].mean():.2f}):")
    print(summary.round(3).to_string())
    print(f"\nparsimony ΔAUC-PR (vector − type) = "
          f"{summary.loc['vector','auc_pr'] - summary.loc['type','auc_pr']:+.3f}")


if __name__ == "__main__":
    main()
