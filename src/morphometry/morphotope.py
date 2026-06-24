"""Block-scale morphotope — the favela signature at tissue, not cell, scale.

The council's strongest move: a cell-level morphotype answers "what does this 10 m
patch look like"; a *morphotope* answers "is there a recurrent favela tissue". A
tissue is defined by the **composition and mixing of cell-types over a block-scale
window**, not by any single cell. This both lifts the recurrence claim to the
defensible scale and resolves the T2/T3 "conditional cell-type" critique — if those
types only ever appear *embedded within* particular tissues, they are real tissue
states, not clustering noise.

Pipeline: per cell → the morphotype-composition (+ Shannon diversity) of its ~R-metre
neighbourhood → cluster those vectors into morphotopes → recurrence at tissue scale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_TYPES = 6


def neighborhood_composition(
    df: pd.DataFrame, radius: float = 50.0, label_col: str = "morphotype_smooth"
) -> pd.DataFrame:
    """Per built cell: the fraction of each morphotype among cells within ``radius``
    (its tissue mix) + Shannon diversity + neighbour count.

    Computed per site (windows do not cross settlements). Expects columns
    ``site, zone_id, centroid_x, centroid_y`` and the cell morphotype label.
    Returns one row per built cell with ``comp_0..comp_5``, ``diversity``,
    ``n_neighbors``.
    """
    from scipy.spatial import cKDTree

    out = []
    for site, g in df.dropna(subset=[label_col]).groupby("site"):
        xy = g[["centroid_x", "centroid_y"]].to_numpy()
        lab = g[label_col].astype(int).to_numpy()
        tree = cKDTree(xy)
        neigh = tree.query_ball_point(xy, r=radius)
        rows = []
        for i, idx in enumerate(neigh):
            counts = np.bincount(lab[idx], minlength=N_TYPES).astype(float)
            tot = counts.sum()
            frac = counts / tot
            p = frac[frac > 0]
            div = float(-(p * np.log(p)).sum() / np.log(N_TYPES))  # 0..1
            rows.append((*frac, div, len(idx)))
        sub = pd.DataFrame(
            rows, columns=[f"comp_{c}" for c in range(N_TYPES)] + ["diversity", "n_neighbors"])
        sub["site"] = site
        sub["zone_id"] = g["zone_id"].to_numpy()
        out.append(sub)
    return pd.concat(out, ignore_index=True)


def fit_morphotopes(
    comp: pd.DataFrame, k: int, random_state: int = 0, n_init: int = 4
) -> np.ndarray:
    """GMM over the composition+diversity vector → morphotope labels, reordered by
    ascending Saturated-Core (comp_5) share so labels are stable and interpretable
    (M0 = least dense-core tissue → M{k-1} = most)."""
    from sklearn.mixture import GaussianMixture

    X = comp[[f"comp_{c}" for c in range(N_TYPES)] + ["diversity"]].to_numpy()
    gm = GaussianMixture(n_components=k, covariance_type="full",
                         random_state=random_state, n_init=n_init).fit(X)
    raw = gm.predict(X)
    order = np.argsort([X[raw == c, 5].mean() for c in range(k)])
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[c] for c in raw])


def select_k_bic(comp: pd.DataFrame, krange=range(2, 9), random_state: int = 0):
    """BIC over k for the morphotope GMM (model-selection defence)."""
    from sklearn.mixture import GaussianMixture

    X = comp[[f"comp_{c}" for c in range(N_TYPES)] + ["diversity"]].to_numpy()
    rows = []
    for k in krange:
        gm = GaussianMixture(n_components=k, covariance_type="full",
                             random_state=random_state, n_init=1).fit(X)
        rows.append({"k": k, "bic": gm.bic(X)})
    return pd.DataFrame(rows)


def cell_type_profile(comp: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Mean cell-type composition of each morphotope — what tissue each is made of
    (the figure that shows where T2/T3 actually live)."""
    d = comp.copy()
    d["morphotope"] = labels
    prof = d.groupby("morphotope")[[f"comp_{c}" for c in range(N_TYPES)]].mean()
    prof["diversity"] = d.groupby("morphotope")["diversity"].mean()
    prof["n_cells"] = d.groupby("morphotope").size()
    return prof
