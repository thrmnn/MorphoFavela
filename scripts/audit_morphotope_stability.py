"""Bootstrap stability audit of the block-scale morphotope clustering (k=5).

Rebuilds the per-cell 50 m morphotype-composition vectors over the 5 campaign
sites, then defends the morphotope GMM on two axes:

  (1) BIC-vs-k curve (k=2..8) with the elbow choice marked — why k=5;
  (2) bootstrap stability — N=20 GMM refits on 80% subsamples, each compared
      to the reference k=5 labels via Adjusted Rand Index (ARI) on the shared
      cells; report mean / sd / min ARI.

This does NOT edit build_morphotope.py; it reuses the same primitives from
src.morphometry.morphotope so the reference labels are bit-for-bit the canonical
ones (random_state=0, n_init=4).

    python scripts/audit_morphotope_stability.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402

from src.morphometry.morphotope import (  # noqa: E402
    N_TYPES,
    fit_morphotopes,
    neighborhood_composition,
    select_k_bic,
)
from src.morphometry.signature import CAMPAIGN_SITES, choose_k_elbow  # noqa: E402

OUT = ROOT / "outputs" / "cross_site" / "morphotope_stability"
FIGS = ROOT / "outputs" / "cross_site" / "signature" / "figures_v2"
RADIUS = 50.0
N_BOOT = 20
SUBSAMPLE_FRAC = 0.80
REF_K = 5
COMP_COLS = [f"comp_{c}" for c in range(N_TYPES)] + ["diversity"]


def load() -> pd.DataFrame:
    frames = []
    for s in CAMPAIGN_SITES:
        p = ROOT / "outputs" / s / "features" / "features_grid.parquet"
        if not p.exists():
            continue
        g = gpd.read_parquet(p)
        col = "morphotype_smooth" if "morphotype_smooth" in g else "morphotype"
        d = pd.DataFrame(g.drop(columns="geometry"))
        d["site"] = s
        d = d.rename(columns={col: "morphotype_smooth"})
        frames.append(d[["site", "zone_id", "centroid_x", "centroid_y",
                         "morphotype_smooth"]])
    return pd.concat(frames, ignore_index=True)


def bootstrap_ari(comp: pd.DataFrame, ref_labels: np.ndarray, k: int):
    """N_BOOT GMM refits on SUBSAMPLE_FRAC random subsamples of the composition
    vectors; ARI of each refit against the reference labels on the shared cells.

    Each bootstrap seeds both the subsample draw and the GMM, so the run is
    reproducible. The reference labels (full-data fit) are held fixed.
    """
    X = comp[COMP_COLS].to_numpy()
    n = len(X)
    m = int(round(SUBSAMPLE_FRAC * n))
    aris = []
    for b in range(N_BOOT):
        rng = np.random.default_rng(b)
        sub = rng.choice(n, size=m, replace=False)
        gm = GaussianMixture(n_components=k, covariance_type="full",
                             random_state=b, n_init=4).fit(X[sub])
        boot_lab = gm.predict(X[sub])
        aris.append(adjusted_rand_score(ref_labels[sub], boot_lab))
    return np.array(aris)


def fig_stability(sel: pd.DataFrame, elbow_k: int, bic_min_k: int, aris: np.ndarray):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    # (1) BIC vs k
    ax = axes[0]
    k = sel["k"].to_numpy()
    b = sel["bic"].to_numpy()
    ax.plot(k, b, "-o", color="#5aae61", lw=1.8, ms=6)
    ax.axvline(elbow_k, color="#762a83", ls="--", lw=1.4,
               label=f"elbow → k={elbow_k} (reference)")
    if bic_min_k != elbow_k:
        ax.axvline(bic_min_k, color="0.6", ls=":", lw=1.2,
                   label=f"BIC argmin → k={bic_min_k}")
    ax.set_xlabel("k (number of morphotopes)")
    ax.set_ylabel("GMM BIC")
    ax.set_title("Model selection — why k=5", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)

    # (2) bootstrap ARI distribution
    ax = axes[1]
    ax.hist(aris, bins=np.linspace(0, 1, 21), color="#a6dba0",
            edgecolor="#5aae61")
    ax.axvline(aris.mean(), color="#762a83", lw=1.6,
               label=f"mean ARI = {aris.mean():.3f}")
    ax.axvline(aris.min(), color="0.4", ls="--", lw=1.2,
               label=f"min ARI = {aris.min():.3f}")
    ax.set_xlim(0, 1)
    ax.set_xlabel("ARI vs reference k=5 labels")
    ax.set_ylabel(f"count (N={N_BOOT} refits)")
    ax.set_title(f"Bootstrap stability — {int(SUBSAMPLE_FRAC * 100)}% subsamples",
                 fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)

    fig.suptitle("Morphotope clustering stability (block-scale, 50 m composition window)",
                 fontsize=10)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "morphotope_stability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    comp = neighborhood_composition(df, radius=RADIUS)

    sel = select_k_bic(comp, krange=range(2, 9))
    elbow_k = choose_k_elbow(sel)
    bic_min_k = int(sel.loc[sel["bic"].idxmin(), "k"])

    ref_labels = fit_morphotopes(comp, REF_K, random_state=0, n_init=4)
    aris = bootstrap_ari(comp, ref_labels, REF_K)

    sel.to_csv(OUT / "bic_curve.csv", index=False)
    pd.DataFrame({"boot": range(N_BOOT), "ari": aris}).to_csv(
        OUT / "bootstrap_ari.csv", index=False)
    fig_stability(sel, elbow_k, bic_min_k, aris)

    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "radius_m": RADIUS,
        "n_cells": int(len(comp)),
        "reference_k": REF_K,
        "elbow_k": int(elbow_k),
        "bic_argmin_k": bic_min_k,
        "n_bootstrap": N_BOOT,
        "subsample_frac": SUBSAMPLE_FRAC,
        "ari_mean": float(aris.mean()),
        "ari_sd": float(aris.std(ddof=1)),
        "ari_min": float(aris.min()),
        "ari_max": float(aris.max()),
    }
    (OUT / "stability_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"n_cells={len(comp)}  reference_k={REF_K}")
    print(f"elbow k={elbow_k}  BIC-argmin k={bic_min_k}")
    print("BIC curve:")
    print(sel.to_string(index=False))
    print(f"\nbootstrap ARI (N={N_BOOT}, {int(SUBSAMPLE_FRAC*100)}% subsamples): "
          f"mean={aris.mean():.3f} sd={aris.std(ddof=1):.3f} "
          f"min={aris.min():.3f} max={aris.max():.3f}")


if __name__ == "__main__":
    main()
