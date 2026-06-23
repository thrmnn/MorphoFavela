"""Build the favela morphometric signature (WS-A).

Pools the WS-0 fabric tables across all sites, clusters the 6-feature fabric
vector into morphotypes (GMM, k by BIC elbow), validates by cross-site
recurrence, profiles each type's experienced conditions, writes the morphotype
label back into each site's ``features_grid.parquet``, and renders the core
figures. Decisions logged in ``docs/morpho_signature_decisions.md``.

    python scripts/build_signature.py
"""

from __future__ import annotations

import glob
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # win over the stale brisa-0.1.0 editable install

from src.morphometry.signature import (  # noqa: E402
    CAMPAIGN_SITES,
    SIGNATURE_FEATURES,
    assemble_signature_matrix,
    centroid_linkage,
    choose_k_elbow,
    experience_profile,
    fit_morphotypes,
    recurrence_flags,
    recurrence_matrix,
    select_k,
    standardize,
)
from src.svf_v2.io import _git_sha  # noqa: E402

OUT = ROOT / "outputs" / "cross_site" / "signature"
RANDOM_STATE = 0


def load_pooled() -> pd.DataFrame:
    """Pool the 5 CAMPAIGN sites only. The 3 calibration sites (borel,
    jacarezinho, morro_do_juramento) are kept aside — they have feature tables on
    disk but do not define the signature or appear in any figure."""
    frames = []
    for p in sorted(glob.glob(str(ROOT / "outputs/*/features/features_grid.parquet"))):
        site = Path(p).parents[1].name
        if site not in CAMPAIGN_SITES:
            continue
        g = gpd.read_parquet(p)
        g["site"] = site
        frames.append(pd.DataFrame(g.drop(columns="geometry")))
    return pd.concat(frames, ignore_index=True)


def clear_calibration_labels() -> None:
    """Drop any stale morphotype labels from the calibration sites' parquets."""
    for p in glob.glob(str(ROOT / "outputs/*/features/features_grid.parquet")):
        if Path(p).parents[1].name in CAMPAIGN_SITES:
            continue
        g = gpd.read_parquet(p)
        drop = [c for c in ("morphotype", "morphotype_smooth") if c in g.columns]
        if drop:
            g.drop(columns=drop).to_parquet(p)


def write_labels_back(mat: pd.DataFrame, labels: np.ndarray) -> None:
    lut = mat[["site", "zone_id"]].copy()
    lut["morphotype"] = labels
    for site, grp in lut.groupby("site"):
        path = ROOT / "outputs" / site / "features" / "features_grid.parquet"
        g = gpd.read_parquet(path)
        g = g.drop(columns="morphotype", errors="ignore").merge(
            grp.drop(columns="site"), on="zone_id", how="left"
        )
        g["morphotype"] = g["morphotype"].astype("Int64")
        g.to_parquet(path)


def figures(sel, k, Xz, labels, link, shares, profile, stats) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(c) for c in range(k)]

    def _save(fig, name):
        fig.tight_layout()
        fig.savefig(fig_dir / name, dpi=150)
        plt.close(fig)

    # 1. BIC curve with chosen k
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(sel["k"], sel["bic"], "o-")
    ax.axvline(k, color="crimson", ls="--", label=f"elbow k={k}")
    ax.set(xlabel="k", ylabel="BIC", title="GMM model selection")
    ax.legend()
    _save(fig, "bic_curve.png")

    # 2. dendrogram of morphotype centroids
    fig, ax = plt.subplots(figsize=(5, 3.2))
    dendrogram(link, labels=[f"T{c}" for c in range(k)], ax=ax)
    ax.set_title("Morphotype taxonomy (Ward on centroids)")
    _save(fig, "dendrogram.png")

    # 3. recurrence heatmap (site × type)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    im = ax.imshow(shares.to_numpy(), aspect="auto", cmap="viridis", vmin=0)
    ax.set_xticks(range(shares.shape[1]))
    ax.set_xticklabels([f"T{c}" for c in shares.columns])
    ax.set_yticks(range(shares.shape[0]))
    ax.set_yticklabels(shares.index)
    ax.set_title("Morphotype share per site (recurrence)")
    fig.colorbar(im, ax=ax, label="share")
    _save(fig, "recurrence_heatmap.png")

    # 4. fingerprint radar — standardized centroid per type
    angles = np.linspace(0, 2 * np.pi, len(SIGNATURE_FEATURES), endpoint=False)
    angles = np.concatenate([angles, angles[:1]])
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"polar": True})
    for c in range(k):
        v = Xz[labels == c].mean(axis=0)
        v = np.concatenate([v, v[:1]])
        ax.plot(angles, v, color=colors[c], label=f"T{c}")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(SIGNATURE_FEATURES, fontsize=8)
    ax.set_title("Morphotype fingerprints (z-scored)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    _save(fig, "fingerprints.png")

    # 5. per-site morphotype choropleths
    for p in sorted(glob.glob(str(ROOT / "outputs/*/features/features_grid.parquet"))):
        g = gpd.read_parquet(p)
        if "morphotype" not in g or g["morphotype"].isna().all():
            continue
        site = Path(p).parents[1].name
        fig, ax = plt.subplots(figsize=(5, 5))
        g.plot(ax=ax, color="0.92", linewidth=0)
        gg = g.dropna(subset=["morphotype"])
        gg.plot(ax=ax, column="morphotype", categorical=True,
                cmap="tab10", vmin=0, vmax=9, linewidth=0)
        ax.set_axis_off()
        ax.set_title(f"{site} — morphotypes")
        _save(fig, f"map_{site}.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_pooled()
    mat = assemble_signature_matrix(df)
    Xz, stats = standardize(mat)
    sel = select_k(Xz, random_state=RANDOM_STATE)
    k = choose_k_elbow(sel)
    labels = fit_morphotypes(Xz, k, random_state=RANDOM_STATE)
    link = centroid_linkage(Xz, labels)
    shares = recurrence_matrix(mat, labels)
    flags = recurrence_flags(shares)
    profile = experience_profile(df, mat, labels)

    write_labels_back(mat, labels)
    clear_calibration_labels()

    sel.to_csv(OUT / "k_selection.csv", index=False)
    stats.to_csv(OUT / "standardization.csv", index=False)
    shares.to_csv(OUT / "recurrence_matrix.csv")
    flags.to_csv(OUT / "recurrence_flags.csv")
    profile.to_csv(OUT / "experience_profile.csv", index=False)
    # fabric centroids in original units, per type
    cent = mat.copy()
    cent["morphotype"] = labels
    cent.groupby("morphotype")[SIGNATURE_FEATURES].median().to_csv(
        OUT / "fabric_centroids.csv"
    )
    figures(sel, k, Xz, labels, link, shares, profile, stats)

    meta = {
        "git_sha": _git_sha(),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_cells_clustered": int(len(mat)),
        "n_dropped_nan": int(mat.attrs.get("n_dropped", 0)),
        "k": int(k),
        "features": SIGNATURE_FEATURES,
        "random_state": RANDOM_STATE,
    }
    (OUT / "run_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"k={k}  clustered={len(mat)} cells (dropped {mat.attrs.get('n_dropped',0)} NaN)")
    print("\nrecurrence (share per site):")
    print(shares.round(2).to_string())
    print("\nrecurrence flags:")
    print(flags.to_string())
    print("\nexperience profile:")
    print(profile.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
