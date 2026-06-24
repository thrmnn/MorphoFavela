"""Model-selection rigor for the CELL morphotype clustering (k=6).

The defence a reviewer demands for the chosen number of GMM components. Two
independent lines of evidence on the pooled 5-campaign-site built-cell fabric
matrix (the same ``assemble_signature_matrix`` + ``standardize`` the production
``build_signature.py`` uses):

1. A model-selection battery across k = 2..10 — BIC, silhouette (fixed 5000-cell
   subsample), Calinski-Harabasz, Davies-Bouldin — with k=6 marked on every
   panel. We report where the BIC elbow and the silhouette peak actually land,
   honestly, rather than assuming k=6 wins.

2. Leave-one-site-out cluster STABILITY. For each held-out site we refit the GMM
   on the other 4 sites, predict labels on *all* cells, Hungarian-match those
   labels to the full-data labels, and score the held-out site's cells with the
   Adjusted Rand Index. Mean/min ARI across the 5 folds answers: does the k=6
   partition reproduce when any one site is removed?

    python scripts/audit_k_selection.py
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    confusion_matrix,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # win over the stale editable install

from src.morphometry.signature import (  # noqa: E402
    CAMPAIGN_SITES,
    assemble_signature_matrix,
    standardize,
)

FIG_DIR = ROOT / "outputs" / "cross_site" / "signature" / "figures_v2"
CSV_DIR = ROOT / "outputs" / "cross_site" / "audit_k_selection"
RANDOM_STATE = 0
KRANGE = range(2, 11)
K_CHOSEN = 6
SILHOUETTE_SAMPLE = 5000


def load_pooled() -> pd.DataFrame:
    """Pool the 5 campaign sites' feature grids (calibration sites excluded)."""
    frames = []
    for p in sorted(glob.glob(str(ROOT / "outputs/*/features/features_grid.parquet"))):
        site = Path(p).parents[1].name
        if site not in CAMPAIGN_SITES:
            continue
        g = gpd.read_parquet(p)
        g["site"] = site
        frames.append(pd.DataFrame(g.drop(columns="geometry")))
    return pd.concat(frames, ignore_index=True)


def battery(Xz: np.ndarray) -> pd.DataFrame:
    """BIC / silhouette / CH / DB across KRANGE on a single shared standardization.

    Silhouette uses a fixed 5000-cell subsample (it is O(n^2)); the same subsample
    indices are reused for every k so the curve is internally comparable.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    sub = (
        rng.choice(len(Xz), size=SILHOUETTE_SAMPLE, replace=False)
        if len(Xz) > SILHOUETTE_SAMPLE
        else np.arange(len(Xz))
    )
    rows = []
    for k in KRANGE:
        gm = GaussianMixture(
            n_components=k, covariance_type="full",
            random_state=RANDOM_STATE, n_init=4,
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


def bic_elbow_k(sel: pd.DataFrame) -> int:
    """Kneedle-style elbow on the BIC curve (max chord distance) — same rule as
    the production ``choose_k_elbow``, recomputed here over KRANGE."""
    k = sel["k"].to_numpy(dtype=float)
    b = sel["bic"].to_numpy(dtype=float)
    kn = (k - k.min()) / (k.max() - k.min())
    bn = (b - b.min()) / (b.max() - b.min())
    x0, y0, x1, y1 = kn[0], bn[0], kn[-1], bn[-1]
    num = np.abs((y1 - y0) * kn - (x1 - x0) * bn + x1 * y0 - y1 * x0)
    den = np.hypot(y1 - y0, x1 - x0)
    return int(sel["k"].to_numpy()[np.argmax(num / den)])


def fit_labels(Xz: np.ndarray, k: int, random_state: int = RANDOM_STATE) -> np.ndarray:
    """Plain GMM predict (no lambda_p reordering — ARI is permutation-invariant
    and we Hungarian-match before any per-cluster comparison anyway)."""
    gm = GaussianMixture(
        n_components=k, covariance_type="full",
        random_state=random_state, n_init=4,
    ).fit(Xz)
    return gm.predict(Xz)


def match_labels(reference: np.ndarray, candidate: np.ndarray, k: int) -> np.ndarray:
    """Hungarian-match ``candidate`` cluster ids onto ``reference`` via the
    confusion matrix (maximise overlap), so the two labelings share a vocabulary
    before we report any per-fold confusion. ARI itself does not need this, but
    the matched labels make the fold diagnostics interpretable."""
    cm = confusion_matrix(reference, candidate, labels=list(range(k)))
    row, col = linear_sum_assignment(-cm)
    remap = {int(c): int(r) for r, c in zip(row, col)}
    return np.array([remap.get(int(c), int(c)) for c in candidate])


def loso_stability(
    mat: pd.DataFrame, Xz: np.ndarray, full_labels: np.ndarray, k: int
) -> pd.DataFrame:
    """Leave-one-site-out ARI. Refit on the 4 retained sites, predict on all
    cells, match to full labels, then score ARI on the held-out site's cells."""
    sites = mat["site"].to_numpy()
    rows = []
    for held in CAMPAIGN_SITES:
        train = sites != held
        gm = GaussianMixture(
            n_components=k, covariance_type="full",
            random_state=RANDOM_STATE, n_init=4,
        ).fit(Xz[train])
        pred_all = match_labels(full_labels, gm.predict(Xz), k)
        held_mask = sites == held
        ari = adjusted_rand_score(full_labels[held_mask], pred_all[held_mask])
        rows.append({
            "held_out_site": held,
            "n_cells_held": int(held_mask.sum()),
            "ari_vs_full": ari,
        })
    return pd.DataFrame(rows)


def plot_battery(sel: pd.DataFrame, elbow_k: int, sil_k: int) -> Path:
    panels = [
        ("bic", "BIC (lower = better)", "min"),
        ("silhouette", "Silhouette (higher = better)", "max"),
        ("calinski_harabasz", "Calinski-Harabasz (higher = better)", "max"),
        ("davies_bouldin", "Davies-Bouldin (lower = better)", "min"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ks = sel["k"].to_numpy()
    for ax, (col, title, _sense) in zip(axes.ravel(), panels):
        y = sel[col].to_numpy()
        ax.plot(ks, y, "-o", color="#0072B2", lw=1.8, ms=5, zorder=2)
        ax.axvline(K_CHOSEN, color="#D55E00", ls="--", lw=1.4, zorder=1,
                   label=f"chosen k={K_CHOSEN}")
        if col == "bic":
            ax.scatter([elbow_k], [sel.loc[sel.k == elbow_k, col].iloc[0]],
                       s=120, facecolors="none", edgecolors="#009E73",
                       lw=2.2, zorder=3, label=f"BIC elbow k={elbow_k}")
        if col == "silhouette":
            ax.scatter([sil_k], [sel.loc[sel.k == sil_k, col].iloc[0]],
                       s=120, facecolors="none", edgecolors="#009E73",
                       lw=2.2, zorder=3, label=f"silhouette peak k={sil_k}")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("k (GMM components)")
        ax.set_xticks(list(ks))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(
        "Cell-morphotype k-selection battery — pooled 5-site fabric (GMM, full cov)",
        fontsize=13, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIG_DIR / "k_selection_rigor.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    df = load_pooled()
    mat = assemble_signature_matrix(df)
    Xz, _stats = standardize(mat)
    n_cells = len(Xz)

    sel = battery(Xz)
    sel.to_csv(CSV_DIR / "k_battery.csv", index=False)
    elbow_k = bic_elbow_k(sel)
    sil_k = int(sel.loc[sel["silhouette"].idxmax(), "k"])
    ch_k = int(sel.loc[sel["calinski_harabasz"].idxmax(), "k"])
    db_k = int(sel.loc[sel["davies_bouldin"].idxmin(), "k"])

    full_labels = fit_labels(Xz, K_CHOSEN)
    loso = loso_stability(mat, Xz, full_labels, K_CHOSEN)
    loso.to_csv(CSV_DIR / "loso_stability.csv", index=False)

    fig_path = plot_battery(sel, elbow_k, sil_k)

    mean_ari = float(loso["ari_vs_full"].mean())
    min_ari = float(loso["ari_vs_full"].min())
    worst = loso.loc[loso["ari_vs_full"].idxmin(), "held_out_site"]

    print(f"n_cells clustered      : {n_cells}")
    print(f"BIC elbow (kneedle)    : k={elbow_k}")
    print(f"silhouette peak        : k={sil_k}")
    print(f"Calinski-Harabasz peak : k={ch_k}")
    print(f"Davies-Bouldin best    : k={db_k}")
    print(f"LOSO ARI (k={K_CHOSEN}) mean    : {mean_ari:.3f}")
    print(f"LOSO ARI (k={K_CHOSEN}) min     : {min_ari:.3f}  (worst fold: {worst})")
    print("\nLOSO per-fold:")
    print(loso.to_string(index=False))
    print(f"\nfigure  -> {fig_path}")
    print(f"battery -> {CSV_DIR / 'k_battery.csv'}")
    print(f"loso    -> {CSV_DIR / 'loso_stability.csv'}")


if __name__ == "__main__":
    main()
