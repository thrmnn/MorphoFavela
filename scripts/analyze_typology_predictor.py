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
    print(f"\nfigure → {FIGS/'typology_failure_lookup.png'}  (n={len(df)} supported cells)")


if __name__ == "__main__":
    main()
