#!/usr/bin/env python3
"""E5 — unified cross-site WHO-2h winter-sun-failure risk surface (2026-06-28).

The §5.5 finding is that the CONTINUOUS fabric vector (not the discrete morphotype
code) carries the transferable failure signal. Two risk products existed separately:
campaign-site out-of-fold LOSO scores, and a morphotype-RATE blind map on the three
calibration favelas. This unifies them into ONE per-cell risk surface across all
eight favelas, using the SAME continuous-vector logistic predictor everywhere, with
honest provenance per cell:

  CAMPAIGN (5 sites)    — out-of-fold LOSO probability (the cell's site is held out
                          of training; honest out-of-sample).
  CALIBRATION (3 sites) — blind probability from a model fit on all 5 campaign
                          sites; flagged out-of-envelope where any feature leaves the
                          campaign [p1,p99] training range (extrapolation, a coarse
                          prioritiser, NOT a per-building guarantee).

STILL morphology-only: predicts the winter-sun (solar-geometry) outcome, not air
exchange; ventilation adequacy stays CFD-gated.

Outputs:
  outputs/cross_site/risk_map/cross_site_risk.json
  outputs/paper_figures/exports/cross_site_riskmap.png (+ .svg)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "outputs" / "paper_figures"))
from fig_style import add_provenance, apply_style, save_fig  # noqa: E402

CAMPAIGN = ["vidigal", "rocinha", "complexo_do_alemao", "riodaspedras", "maré"]
CALIBRATION = ["borel", "jacarezinho", "morro_do_juramento"]
CONT = ["lambda_p", "lambda_f_mean", "H_mean", "sigma_h", "slope_deg"]
TARGET = "solar_winter_frac_below2h"
FAIL = 0.5
LABELS = {"vidigal": "Vidigal", "rocinha": "Rocinha", "complexo_do_alemao": "C. do Alemão",
          "riodaspedras": "Rio das Pedras", "maré": "Maré", "borel": "Borel",
          "jacarezinho": "Jacarezinho", "morro_do_juramento": "M. do Juramento"}


def out_of_envelope_mask(X: np.ndarray, env: list[tuple[float, float]]) -> np.ndarray:
    """Pure: per-row True if ANY feature leaves its campaign [p1,p99] range.
    X is (n, n_features); env is a per-feature (lo, hi) list in column order."""
    out = np.zeros(len(X), dtype=bool)
    for i, (lo, hi) in enumerate(env):
        out |= (X[:, i] < lo) | (X[:, i] > hi)
    return out


def _load(site: str, need_target: bool) -> gpd.GeoDataFrame:
    g = gpd.read_parquet(ROOT / "outputs" / site / "features" / "features_grid.parquet").copy()
    if "built_mask" in g:
        g = g[g["built_mask"].fillna(False)]
    else:
        g = g[g["building_count"] > 0]
    g["sigma_h"] = g["sigma_h"].fillna(0.0)
    g = g.dropna(subset=CONT)
    if need_target:
        if "has_street_support" in g:
            g = g[g["has_street_support"].fillna(False)]
        g = g.dropna(subset=[TARGET])
    g["site"] = site
    return g


def _fit(X, y):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X), y)
    return sc, clf


def _score(sc, clf, X):
    return clf.predict_proba(sc.transform(X))[:, 1]


def main() -> None:
    camp = {s: _load(s, need_target=True) for s in CAMPAIGN}
    pooled = pd.concat([pd.DataFrame(g.drop(columns="geometry")) for g in camp.values()],
                       ignore_index=True)
    env = [(float(np.percentile(pooled[f], 1)), float(np.percentile(pooled[f], 99))) for f in CONT]

    per_site = {}
    risk = {}  # site -> per-cell probability aligned to camp[s]/calib[s] index order

    # --- campaign: out-of-fold LOSO -------------------------------------------
    for s in CAMPAIGN:
        tr = pooled[pooled["site"] != s]
        Xtr, ytr = tr[CONT].to_numpy(float), (tr[TARGET] > FAIL).astype(int).to_numpy()
        sc, clf = _fit(Xtr, ytr)
        Xte = camp[s][CONT].to_numpy(float)
        p = _score(sc, clf, Xte)
        y = (camp[s][TARGET] > FAIL).astype(int).to_numpy()
        risk[s] = p
        per_site[s] = {
            "role": "campaign", "provenance": "loso_out_of_fold", "n": int(len(p)),
            "mean_risk": float(p.mean()), "prevalence": float(y.mean()),
            "auc_pr": float(average_precision_score(y, p)) if len(set(y)) > 1 else None,
            "frac_out_envelope": 0.0,
        }

    # --- calibration: blind, fit on ALL campaign ------------------------------
    Xall, yall = pooled[CONT].to_numpy(float), (pooled[TARGET] > FAIL).astype(int).to_numpy()
    sc_all, clf_all = _fit(Xall, yall)
    calib = {}
    for s in CALIBRATION:
        g = _load(s, need_target=False)
        calib[s] = g
        X = g[CONT].to_numpy(float)
        p = _score(sc_all, clf_all, X)
        risk[s] = p
        oo = out_of_envelope_mask(X, env)
        rec = {"role": "calibration", "provenance": "blind_fit_on_all_campaign",
               "n": int(len(p)), "mean_risk": float(p.mean()),
               "frac_out_envelope": float(oo.mean())}
        if TARGET in g and g[TARGET].notna().any():
            ok = g[TARGET].notna().to_numpy()
            y = (g.loc[ok, TARGET] > FAIL).astype(int).to_numpy()
            if len(set(y)) > 1:
                rec["auc_pr"] = float(average_precision_score(y, p[ok]))
                rec["prevalence"] = float(y.mean())
        per_site[s] = rec

    payload = {
        "title": "Unified cross-site WHO-2h winter-sun-failure risk surface (E5, 2026-06-28)",
        "predictor": "continuous fabric vector (lambda_p, lambda_f_mean, H_mean, sigma_h, slope_deg) logistic",
        "method": {"campaign": "out-of-fold LOSO probability (site held out)",
                   "calibration": "blind, model fit on all 5 campaign sites; out-of-envelope flagged"},
        "status": "morphology-only winter-sun (solar-geometry) risk; NOT ventilation adequacy (τ CFD-gated)",
        "envelope_cont_p1_p99": {f: env[i] for i, f in enumerate(CONT)},
        "per_site": per_site,
    }
    OUT_DIR = ROOT / "outputs" / "cross_site" / "risk_map"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "cross_site_risk.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Wrote {(OUT_DIR / 'cross_site_risk.json').relative_to(ROOT)}\n")
    print(f"{'site':<18s} {'role':<12s} {'n':>6s} {'mean_p':>7s} {'AUC-PR':>7s} {'out_env':>8s}")
    for s in CAMPAIGN + CALIBRATION:
        r = per_site[s]
        ap = f"{r['auc_pr']:.2f}" if r.get("auc_pr") is not None else "  — "
        print(f"{LABELS[s]:<18s} {r['role']:<12s} {r['n']:>6d} {r['mean_risk']:>7.3f} "
              f"{ap:>7s} {r['frac_out_envelope']*100:>7.0f}%")

    # --- figure: 8 panels, campaign (OOF) then calibration (blind) ------------
    apply_style()
    sites = CAMPAIGN + CALIBRATION
    grids = {**camp, **calib}
    fig = plt.figure(figsize=(7.09, 3.7))
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 1], hspace=0.32, wspace=0.06,
                          top=0.84, bottom=0.04, left=0.01, right=0.93)
    norm = Normalize(0, 1)
    slots = [(0, i) for i in range(5)] + [(1, i) for i in range(3)]
    for (r, c), s in zip(slots, sites):
        ax = fig.add_subplot(gs[r, c])
        g = grids[s]
        g.assign(_risk=risk[s]).plot(ax=ax, column="_risk", cmap="inferno", norm=norm,
                                     edgecolor="none", linewidth=0.0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        tag = "OOF" if per_site[s]["role"] == "campaign" else f"blind·{per_site[s]['frac_out_envelope']*100:.0f}%oe"
        ax.set_title(f"{LABELS[s]}\n{tag} · p̄ {per_site[s]['mean_risk']:.2f}", fontsize=6.0, pad=2)
    cax = fig.add_subplot(gs[1, 4])
    pos = cax.get_position()
    cax.set_position([pos.x0 + 0.01, pos.y0 + 0.18, 0.015, pos.height * 0.55])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap="inferno"), cax=cax)
    cb.set_label("p̂(WHO-2h winter-sun failure)", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    fig.text(0.5, 0.965,
             "Unified cross-site winter-sun-failure risk — continuous fabric-vector predictor\n"
             "(top: 5 campaign favelas, out-of-fold LOSO; bottom: 3 calibration favelas, blind + "
             "out-of-envelope %; morphology-only, not ventilation adequacy)",
             ha="center", va="top", fontsize=7.0)
    add_provenance(fig)
    save_fig(fig, "cross_site_riskmap", gate=True)


if __name__ == "__main__":
    main()
