"""Why does the blind morphotype→failure lookup transfer to jacarezinho (AUC-PR 0.82)
but fail on morro_do_juramento (0.39) even though morro is *inside* the training
envelope (only 8% out-of-envelope)?

FINDING (the data overturned the first hypothesis — kept here as the record). The blind
predictor is a 6-level per-morphotype WHO-2h winter-sun failure lookup
(typology_predictor_extra.blind_riskmap) over a geometry-only signature
(SIGNATURE_FEATURES) that carries slope MAGNITUDE (slope_deg) but NOT slope DIRECTION
(aspect_deg). Three candidate causes were tested:

  (a) type collapse — REJECTED: morro spreads across ~3.8 effective types (n_eff), good
      diversity, so it is not that every cell gets the same p̂.
  (b) within-site aspect gradient — REJECTED for morro: r(northness, fail) = -0.03 and
      adding aspect+slope does NOT recover ranking (0.39 → 0.36). (It DOES help borel,
      r=-0.40 — a mixed-orientation steep site.)
  (c) site×type interaction from a site-level ORIENTATION offset — CONFIRMED: morro is a
      coherently north-facing hill (mean northness 0.68; 79% of cells face N-ish =
      equator-facing = winter-lit in Rio, ~22.9°S). It is steep (median 25°) yet only 23%
      fail, whereas the campaign's steep types fail 58-66%. So the campaign lookup, trained
      where steep fabric is often shadowed, assigns morro's steep-but-sunny fabric ~60%
      failure; reality is ~20% (T3: 58%→21% on n=214, the dominant type). The morphotype
      transfers the density→failure ORDERING but not the orientation→failure OFFSET; on a
      uniformly-oriented hill that offset dominates and the coarse type ranking collapses.

Implication (feeds plan C3): the transferable fix is a physically-deterministic winter-sun
potential from (latitude, slope, aspect) — a per-cell astronomical moderator the geometry-
only morphotype omits — not a within-site aspect regression.

    python scripts/diagnose_morro_transfer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from src.morphometry.signature import SIGNATURE_FEATURES
from typology_predictor_extra import (  # noqa: E402
    CALIBRATION_SITES,
    FAIL_THRESH,
    K,
    TARGET,
    fit_canonical_gmm,
    per_type_failure_rate,
)

OUT = ROOT / "outputs" / "cross_site" / "typology_predictor_extra"
FIGS = ROOT / "outputs" / "cross_site" / "signature" / "figures_v2"


def _prep(site):
    import geopandas as gpd

    g = gpd.read_parquet(ROOT / "outputs" / site / "features" / "features_grid.parquet").copy()
    g["lambda_f_aniso"] = (g["lambda_f_max"] - g["lambda_f_mean"]).clip(lower=0)
    g["sigma_h"] = g["sigma_h"].fillna(0.0)
    built = g["built_mask"].fillna(False)
    ok = built & g[SIGNATURE_FEATURES].notna().all(axis=1) & g[TARGET].notna() & g["aspect_deg"].notna()
    return g.loc[ok].reset_index(drop=True)


def _auc_pr(y, s):
    return average_precision_score(y, s) if len(set(y)) > 1 else np.nan


def main():
    assign, labels, mat, pooled = fit_canonical_gmm()
    type_rate = per_type_failure_rate(mat, labels, pooled)
    print("Campaign per-type WHO-2h failure lookup (the blind predictor):")
    print("  " + "  ".join(f"T{c}={type_rate[c]*100:.0f}%" for c in range(K)))

    rows = []
    per_site = {}
    for s in CALIBRATION_SITES:
        g = _prep(s)
        mt = assign(g[SIGNATURE_FEATURES]).astype(int)
        phat = np.array([type_rate[c] for c in mt])
        y = (g[TARGET].to_numpy() > FAIL_THRESH).astype(int)
        # aspect → "northness" = cos(aspect); +1 = equator-facing (winter-lit), −1 = shadowed
        northness = np.cos(np.deg2rad(g["aspect_deg"].to_numpy()))
        slope = g["slope_deg"].to_numpy()

        # (a) type collapse? distribution + effective distinct predictions
        shares = pd.Series(mt).value_counts(normalize=True).sort_index()
        top_share = shares.max()
        n_eff = 1.0 / (shares**2).sum()  # inverse-Simpson: effective # of types

        # (b) site×type shift: observed within-type failure here vs the campaign lookup
        obs_rate = {c: float(y[mt == c].mean()) for c in np.unique(mt) if (mt == c).sum() >= 20}
        shift = np.nanmean([abs(obs_rate[c] - type_rate[c]) for c in obs_rate]) if obs_rate else np.nan

        # (c) aspect mechanism: does northness rank failure the lookup can't?
        #   lookup-only vs lookup+aspect+slope logistic (in-site fit → optimistic ceiling,
        #   used only to see how much *recoverable* signal aspect carries)
        aucpr_lookup = _auc_pr(y, phat)
        r_north = np.corrcoef(northness, y)[0, 1]
        # within the dominant type, does aspect still separate failure?
        dom = int(shares.idxmax())
        d = mt == dom
        r_north_wt = np.corrcoef(northness[d], y[d])[0, 1] if d.sum() > 20 and len(set(y[d])) > 1 else np.nan
        Xa = StandardScaler().fit_transform(np.c_[phat, northness, slope])
        aucpr_aspect = _auc_pr(
            y, LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xa, y).predict_proba(Xa)[:, 1]
        ) if len(set(y)) > 1 else np.nan

        rows.append({
            "site": s, "n": len(y), "prevalence": round(float(y.mean()), 3),
            "median_slope": round(float(np.median(slope)), 1),
            "top_type_share": round(float(top_share), 2), "n_eff_types": round(float(n_eff), 2),
            "sitextype_shift": round(float(shift), 3) if not np.isnan(shift) else np.nan,
            "auc_pr_lookup": round(float(aucpr_lookup), 3),
            "auc_pr_+aspect": round(float(aucpr_aspect), 3),
            "r(northness,fail)": round(float(r_north), 3),
            "r_within_domtype": round(float(r_north_wt), 3) if not np.isnan(r_north_wt) else np.nan,
        })
        per_site[s] = (g, mt, y, northness, phat)

    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "morro_transfer_diagnosis.csv", index=False)
    print("\nBlind-transfer diagnosis (3 calibration favelas):")
    print(tab.to_string(index=False))

    # per-type shift table for morro (campaign lookup vs morro observed) — the site×type
    # interaction that breaks ranking.
    g, mt, y, north, phat = per_site["morro_do_juramento"]
    shift_rows = []
    for c in range(K):
        m = mt == c
        if m.sum() == 0:
            continue
        shift_rows.append({"morphotype": c, "n": int(m.sum()),
                           "campaign_rate": round(type_rate[c], 3),
                           "morro_observed": round(float(y[m].mean()), 3)})
    pd.DataFrame(shift_rows).to_csv(OUT / "morro_per_type_shift.csv", index=False)

    # figure — the confirmed mechanism:
    #  (A) per-type failure: campaign lookup (what the blind model assigns) vs morro
    #      observed → the lookup over-predicts everywhere, worst on the dominant steep type.
    #  (B) morro aspect distribution → a coherently N-facing (winter-lit) hill, so the
    #      orientation offset is site-level, not a within-site gradient the model could learn.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    present = [r["morphotype"] for r in shift_rows]
    xc = np.arange(len(present))
    camp = [type_rate[c] * 100 for c in present]
    obs = [dict((r["morphotype"], r["morro_observed"]) for r in shift_rows)[c] * 100 for c in present]
    ns = [dict((r["morphotype"], r["n"]) for r in shift_rows)[c] for c in present]
    axes[0].bar(xc - 0.2, camp, 0.4, label="campaign lookup (assigned p̂)", color="#b2182b")
    axes[0].bar(xc + 0.2, obs, 0.4, label="morro observed", color="#2166ac")
    for i, n in enumerate(ns):
        axes[0].text(xc[i], max(camp[i], obs[i]) + 2, f"n={n}", ha="center", fontsize=7)
    axes[0].set_xticks(xc)
    axes[0].set_xticklabels([f"T{c}" for c in present])
    axes[0].set(ylabel="WHO-2h failure rate (%)", ylim=(0, 100),
                title="Same morphotype fails FAR less in morro than the campaign lookup assigns\n"
                      "(site×type interaction — worst on dominant steep T3: 58%→21%)")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", color="0.93")

    asp = g["aspect_deg"].to_numpy()
    axes[1].hist(asp, bins=np.arange(0, 361, 30), color="#f4a582", edgecolor="0.4")
    axes[1].axvspan(90, 270, color="0.9", zorder=0)
    axes[1].set(xlabel="aspect (° from N)  —  grey = pole-facing (winter-shadowed)",
                ylabel="morro cells", xlim=(0, 360), xticks=[0, 90, 180, 270, 360],
                title=f"morro is a coherently N-facing (winter-lit) hill\n"
                      f"mean northness {north.mean():.2f}, {((asp<60)|(asp>300)).mean()*100:.0f}% face N-ish "
                      f"→ steep yet only {y.mean()*100:.0f}% fail")
    fig.suptitle("Morro transfer-breakdown: an in-envelope, N-facing hillside the geometry-only "
                 "morphotype mis-maps\n(it lacks aspect — the campaign's steep-and-shadowed prior "
                 "over-predicts failure on morro's steep-but-sunny fabric)", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIGS / "morro_transfer_diagnosis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nfigure → {FIGS / 'morro_transfer_diagnosis.png'}")


if __name__ == "__main__":
    main()
