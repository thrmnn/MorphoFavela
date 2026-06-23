"""Favela morphometric signature (WS-A).

Clusters the *fabric* feature vector (cell-native morphometrics only) into
morphotypes, validates them by cross-site recurrence, and profiles each type's
experienced conditions (the fabric×experience contingency). The street/grid
support separation from WS-0 is preserved: clustering is on fabric; experience
enters only as a downstream per-type profile, never as a clustering input.

Feature and method decisions are logged in
``docs/morpho_signature_decisions.md``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Lean, low-redundancy fabric signature vector. Rationale in the decision log:
# porosity (r=-0.97 w/ lambda_p) and far (collinear w/ lambda_p*H_mean) dropped;
# street_orientation_entropy dropped (59% NaN, ~0 variance); orientation
# (northness/eastness) excluded so the signature is rotation-invariant; the 8
# directional lambda_f collapsed to magnitude + anisotropy.
SIGNATURE_FEATURES = [
    "lambda_p",        # plan-area density / coverage
    "H_mean",          # mean building height
    "sigma_h",         # height variability (0 for single-building cells)
    "lambda_f_mean",   # frontal-area density (roughness magnitude)
    "lambda_f_aniso",  # directional anisotropy = lambda_f_max - lambda_f_mean
    "slope_deg",       # terrain
]

# Experience columns profiled per morphotype (support-aware, from WS-0).
EXPERIENCE_COLS = [
    "svf_p50", "solar_winter_p50", "solar_winter_frac_below2h",
    "frac_deep_canyon", "hw_p50",
]

CAMPAIGN_SITES = [
    "vidigal", "rocinha", "riodaspedras", "complexo_do_alemao", "maré",
]


def assemble_signature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Built-cell fabric matrix with documented imputation.

    Expects a pooled frame with a ``site`` column and the WS-0 fabric columns.
    Filters to built cells, derives ``lambda_f_aniso``, imputes ``sigma_h`` = 0
    for single-building cells, and drops any row still carrying a NaN feature
    (count returned via ``.attrs['n_dropped']``).
    """
    out = df[df["built_mask"]].copy()
    out["lambda_f_aniso"] = (out["lambda_f_max"] - out["lambda_f_mean"]).clip(lower=0)
    out["sigma_h"] = out["sigma_h"].fillna(0.0)

    keep = ["site", "zone_id"] + SIGNATURE_FEATURES
    mat = out[keep].copy()
    before = len(mat)
    mat = mat.dropna(subset=SIGNATURE_FEATURES)
    mat.attrs["n_dropped"] = before - len(mat)
    return mat


def standardize(mat: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Z-score on the pooled built cells so a type means the same across sites."""
    X = mat[SIGNATURE_FEATURES].to_numpy(dtype=float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    stats = pd.DataFrame({"feature": SIGNATURE_FEATURES, "mean": mu, "std": sd})
    return (X - mu) / sd, stats


def select_k(
    Xz: np.ndarray,
    krange=range(2, 13),
    random_state: int = 0,
    silhouette_sample: int = 5000,
) -> pd.DataFrame:
    """Model-selection battery per k: BIC (primary), silhouette, CH, DB.

    Silhouette is computed on a fixed random subsample (it is O(n^2)). BIC is the
    GMM-native primary criterion; the others are reported as a cross-check.
    """
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )
    from sklearn.mixture import GaussianMixture

    rng = np.random.default_rng(random_state)
    sub = (
        rng.choice(len(Xz), size=silhouette_sample, replace=False)
        if len(Xz) > silhouette_sample
        else np.arange(len(Xz))
    )
    rows = []
    for k in krange:
        gm = GaussianMixture(
            n_components=k, covariance_type="full",
            random_state=random_state, n_init=1,
        ).fit(Xz)
        lab = gm.predict(Xz)
        rows.append({
            "k": k,
            "bic": gm.bic(Xz),
            "silhouette": silhouette_score(Xz[sub], lab[sub]),
            "calinski_harabasz": calinski_harabasz_score(Xz, lab),
            "davies_bouldin": davies_bouldin_score(Xz, lab),
        })
    return pd.DataFrame(rows)


def choose_k_elbow(sel: pd.DataFrame) -> int:
    """Kneedle-style elbow on the BIC curve (max distance to the chord).

    GMM BIC often keeps decreasing on large n; the elbow gives a parsimonious,
    reproducible choice rather than argmin at the krange boundary.
    """
    k = sel["k"].to_numpy(dtype=float)
    b = sel["bic"].to_numpy(dtype=float)
    kn = (k - k.min()) / (k.max() - k.min())
    bn = (b - b.min()) / (b.max() - b.min())
    # distance of each point from the chord (first→last)
    x0, y0, x1, y1 = kn[0], bn[0], kn[-1], bn[-1]
    num = np.abs((y1 - y0) * kn - (x1 - x0) * bn + x1 * y0 - y1 * x0)
    den = np.hypot(y1 - y0, x1 - x0)
    return int(sel["k"].to_numpy()[np.argmax(num / den)])


def fit_morphotypes(
    Xz: np.ndarray, k: int, random_state: int = 0, n_init: int = 4
) -> np.ndarray:
    """Final GMM fit; labels reordered by ascending mean lambda_p (col 0).

    Stable, interpretable labels: type 0 = sparsest fabric, type k-1 = densest.
    """
    from sklearn.mixture import GaussianMixture

    gm = GaussianMixture(
        n_components=k, covariance_type="full",
        random_state=random_state, n_init=n_init,
    ).fit(Xz)
    raw = gm.predict(Xz)
    order = np.argsort([Xz[raw == c, 0].mean() for c in range(k)])
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[c] for c in raw])


def centroid_linkage(Xz: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Ward linkage over morphotype centroids → the taxonomy dendrogram."""
    from scipy.cluster.hierarchy import linkage

    cents = np.vstack([Xz[labels == c].mean(axis=0) for c in sorted(set(labels))])
    return linkage(cents, method="ward")


def recurrence_matrix(
    mat: pd.DataFrame, labels: np.ndarray, sites=CAMPAIGN_SITES
) -> pd.DataFrame:
    """Per-site share of each morphotype (site × type), restricted to ``sites``.

    Cross-site recurrence is our operational signature test: a type that recurs
    across sites is a favela signature, not a site artefact.
    """
    d = mat.copy()
    d["morphotype"] = labels
    d = d[d["site"].isin(sites)]
    counts = d.groupby(["site", "morphotype"]).size().unstack(fill_value=0)
    shares = counts.div(counts.sum(axis=1), axis=0)
    return shares.reindex(sites)


def recurrence_flags(shares: pd.DataFrame, min_share=0.05, min_sites=3) -> pd.DataFrame:
    """Per type: in how many sites it clears ``min_share``, and the recurs flag."""
    n = (shares >= min_share).sum(axis=0)
    return pd.DataFrame({
        "n_sites_present": n.astype(int),
        "recurs": (n >= min_sites),
    })


def queen_neighbors(gdf) -> dict:
    """Positional Queen-contiguity neighbour map for a site's grid cells."""
    from libpysal.weights import Queen

    w = Queen.from_dataframe(gdf, use_index=False, silence_warnings=True)
    return w.neighbors


def spatial_mode_smooth(
    labels: np.ndarray, neighbors: dict, passes: int = 2
) -> np.ndarray:
    """Contiguity mode filter: relabel each cell to the majority type among
    itself + its neighbours, iterated.

    Dissolves salt-and-pepper while keeping the 6 *global* type semantics (this
    is denoising, not regionalization — full ``spopt`` regionalization is the
    rigorous follow-up, see the decision log). NaN labels (unbuilt cells) are
    left untouched and excluded from voting. Ties keep the current label, so the
    filter is conservative and deterministic.
    """
    import math
    from collections import Counter

    lab = np.array(labels, dtype=float)
    for _ in range(passes):
        new = lab.copy()
        for i in range(len(lab)):
            if math.isnan(lab[i]):
                continue
            votes = [lab[j] for j in neighbors.get(i, []) if not math.isnan(lab[j])]
            votes.append(lab[i])
            counts = Counter(votes)
            top = max(counts.values())
            winners = sorted(v for v, c in counts.items() if c == top)
            new[i] = lab[i] if lab[i] in winners else winners[0]
        lab = new
    return lab


def spatial_purity(labels: np.ndarray, neighbors: dict) -> float:
    """Fraction of same-type adjacencies — the salt-and-pepper metric (0..1)."""
    import math

    same = total = 0
    for i, neigh in neighbors.items():
        if math.isnan(labels[i]):
            continue
        for j in neigh:
            if math.isnan(labels[j]):
                continue
            total += 1
            same += labels[i] == labels[j]
    return same / total if total else float("nan")


def bootstrap_stability(
    Xz: np.ndarray, reference: np.ndarray, k: int,
    n_boot: int = 20, frac: float = 0.8, random_state: int = 0,
) -> np.ndarray:
    """ARI of bootstrap GMM refits vs the reference labels — is k=6 fragile?"""
    from sklearn.metrics import adjusted_rand_score
    from sklearn.mixture import GaussianMixture

    rng = np.random.default_rng(random_state)
    aris = []
    for b in range(n_boot):
        idx = rng.choice(len(Xz), size=int(frac * len(Xz)), replace=False)
        gm = GaussianMixture(
            n_components=k, covariance_type="full", random_state=b, n_init=1
        ).fit(Xz[idx])
        aris.append(adjusted_rand_score(reference, gm.predict(Xz)))
    return np.array(aris)


def experience_profile(
    df: pd.DataFrame, mat: pd.DataFrame, labels: np.ndarray
) -> pd.DataFrame:
    """Per-morphotype experienced-condition profile (fabric×experience link).

    Computed only over cells with street support; ``support_coverage`` reports
    what fraction of each type's cells actually carry observers.
    """
    key = mat[["site", "zone_id"]].copy()
    key["morphotype"] = labels
    # drop any pre-existing label columns so the merge can't collide (morphotype_x/_y)
    df = df.drop(columns=["morphotype", "morphotype_smooth"], errors="ignore")
    j = key.merge(df, on=["site", "zone_id"], how="left")
    rows = []
    for c in sorted(set(labels)):
        sub = j[j["morphotype"] == c]
        sup = sub[sub["has_street_support"]]
        rec = {"morphotype": c, "n_cells": len(sub),
               "support_coverage": len(sup) / max(len(sub), 1)}
        for col in EXPERIENCE_COLS:
            rec[col] = sup[col].median() if col in sup else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)
