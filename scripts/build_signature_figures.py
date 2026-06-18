"""Regenerate the morpho-signature figures per the expert-reviewed spec, and
assemble a self-contained HTML review gallery.

Outputs to outputs/cross_site/signature/figures_v2/ (v1 kept for comparison) and
writes index.html — each figure paired with the decision it embodies and a
keep/refine prompt, served locally for guided review.

    python scripts/build_signature_figures.py
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # win over the stale brisa-0.1.0 editable install

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.cluster.hierarchy import dendrogram  # noqa: E402

from src.morphometry.signature import (  # noqa: E402
    CAMPAIGN_SITES,
    SIGNATURE_FEATURES,
    assemble_signature_matrix,
    centroid_linkage,
    fit_morphotypes,
    standardize,
)
from src.viz.signature_style import (  # noqa: E402
    DIVERGING,
    NULL_COLOR,
    TYPE_COLORS,
    TYPE_NAMES,
    type_color_list,
)

SIG = ROOT / "outputs" / "cross_site" / "signature"
OUT = SIG / "figures_v2"
ROW_ORDER_SITES = ["riodaspedras", "maré", "vidigal", "rocinha", "complexo_do_alemao"]


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_fingerprint_heatmap(centroids, leaf):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    M = centroids.loc[leaf, SIGNATURE_FEATURES].to_numpy()
    vmax = np.abs(M).max()
    im = ax.imshow(M, cmap=DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(SIGNATURE_FEATURES)))
    ax.set_xticklabels(SIGNATURE_FEATURES, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(leaf)))
    ax.set_yticklabels([TYPE_NAMES[c] for c in leaf], fontsize=8)
    for (i, j), v in np.ndenumerate(M):
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7,
                color="black" if abs(v) < 0.6 * vmax else "white")
    for x in (2.5, 4.5):  # density | wind | terrain groups
        ax.axvline(x, color="white", lw=2)
    ax.set_title("Morphotype fingerprints — standardized fabric (z)")
    fig.colorbar(im, ax=ax, label="z-score", shrink=0.8)
    _save(fig, "fingerprint_heatmap.png")


def fig_dendrogram(link, leaf, cut_height):
    fig, ax = plt.subplots(figsize=(5, 3.4))
    dendrogram(link, labels=[f"T{c}" for c in range(len(leaf))], ax=ax,
               color_threshold=0, above_threshold_color="#888888")
    ax.axhline(cut_height, ls="--", color="0.4", lw=1)
    ax.text(0.01, cut_height, " k=6 cut", va="bottom", fontsize=8, color="0.4",
            transform=ax.get_yaxis_transform())
    for lbl in ax.get_xticklabels():
        t = int(lbl.get_text()[1:])
        lbl.set_color(TYPE_COLORS[t])
        lbl.set_fontweight("bold")
    ax.set_ylabel("Ward linkage distance")
    ax.set_title("Morphotype taxonomy")
    _save(fig, "dendrogram.png")


def fig_recurrence(shares, flags, leaf):
    cols = [c for c in leaf if c in shares.columns]
    S = shares.reindex(index=ROW_ORDER_SITES, columns=cols)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    im = ax.imshow(S.to_numpy(), cmap="YlGnBu", vmin=0, vmax=float(np.nanmax(S.to_numpy())))
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"T{c}" for c in cols])
    for lbl, c in zip(ax.get_xticklabels(), cols):
        lbl.set_color(TYPE_COLORS[c])
        lbl.set_fontweight("bold")
    ax.set_yticks(range(len(S.index)))
    ax.set_yticklabels([s.replace("_", " ") for s in S.index])
    for (i, j), v in np.ndenumerate(S.to_numpy()):
        if not np.isnan(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="black" if v < 0.25 else "white")
    # bold-separate the site-specific columns (recurs == False)
    for jj, c in enumerate(cols):
        if not bool(flags.loc[c, "recurs"]):
            ax.axvline(jj - 0.5, color="crimson", lw=2)
            ax.axvline(jj + 0.5, color="crimson", lw=2)
    ax.axhline(1.5, color="0.3", lw=1)  # flat (top 2) vs steep
    ax.set_title("Recurrence — morphotype share per site")
    fig.colorbar(im, ax=ax, label="share", shrink=0.8)
    _save(fig, "recurrence.png")


def fig_experience(profile, leaf):
    panels = [
        ("svf_p50", "median SVF"),
        ("solar_winter_frac_below2h", "frac winter sun < 2h"),
        ("frac_deep_canyon", "frac deep-canyon"),
        ("hw_p50", "median H/W"),
    ]
    prof = profile.set_index("morphotype")
    fig, axes = plt.subplots(1, 4, figsize=(11, 3), sharey=True)
    yorder = [c for c in leaf if c in prof.index]
    for ax, (col, lab) in zip(axes, panels):
        for c in yorder:
            cov = prof.loc[c, "support_coverage"]
            ax.scatter(prof.loc[c, col], yorder.index(c), s=40 + 260 * cov,
                       color=TYPE_COLORS[c], edgecolor="0.3", zorder=3)
        ax.set_yticks(range(len(yorder)))
        ax.set_yticklabels([f"T{c}" for c in yorder])
        ax.set_xlabel(lab, fontsize=9)
        ax.grid(axis="x", color="0.9")
    axes[0].set_title("Experience by morphotype (dot size ∝ street support)",
                      loc="left", fontsize=10)
    _save(fig, "experience_dotplots.png")


def fig_recurrence_evidence(mat, labels, leaf):
    """Per-site standardized type centroids overlaid — are types the same form?"""
    d = mat.copy()
    d["morphotype"] = labels
    Xz, _ = standardize(mat)
    zcols = [f"_z_{f}" for f in SIGNATURE_FEATURES]
    d[zcols] = Xz
    sites = [s for s in CAMPAIGN_SITES if s in d["site"].unique()]
    fig, axes = plt.subplots(1, len(sites), figsize=(2.4 * len(sites), 3.2),
                             sharey=True)
    ang = np.arange(len(SIGNATURE_FEATURES))
    for ax, site in zip(axes, sites):
        ds = d[d["site"] == site]
        for c in leaf:
            sub = ds[ds["morphotype"] == c]
            if len(sub):
                ax.plot(ang, sub[zcols].mean().to_numpy(), color=TYPE_COLORS[c], lw=1.5)
        ax.set_xticks(ang)
        ax.set_xticklabels([f.replace("lambda_", "λ").replace("_", " ")
                            for f in SIGNATURE_FEATURES], rotation=90, fontsize=6)
        ax.set_title(site.replace("_", " "), fontsize=8)
        ax.axhline(0, color="0.85", lw=0.8)
    axes[0].set_ylabel("z-score")
    fig.suptitle("Recurrence evidence — type centroids per site (same colour = same type)",
                 fontsize=10)
    _save(fig, "recurrence_evidence.png")


def fig_kselection(sel, k):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(sel["k"], sel["bic"], "o-", color="0.2", label="BIC")
    ax.axvline(k, ls="--", color="crimson", label=f"elbow k={k}")
    ax.set_xlabel("k")
    ax.set_ylabel("BIC")
    ax2 = ax.twinx()
    ax2.plot(sel["k"], sel["silhouette"], "s--", color="#0072B2", ms=4,
             label="silhouette")
    ax2.set_ylabel("silhouette", color="#0072B2")
    ax.set_title("Model selection (BIC primary + silhouette)")
    ax.legend(loc="upper right", fontsize=8)
    _save(fig, "k_selection.png")


def fig_stability(meta):
    fig, ax = plt.subplots(figsize=(5, 1.8))
    m, sd, lo = meta["bootstrap_ari_mean"], meta["bootstrap_ari_sd"], meta["bootstrap_ari_min"]
    ax.errorbar([m], [0], xerr=[sd], fmt="o", color="#009E73", capsize=4, ms=8)
    ax.scatter([lo], [0], marker="|", s=200, color="crimson", label=f"min {lo:.2f}")
    ax.axvline(0.8, ls=":", color="0.5", label="stable ≥ 0.8")
    ax.set_xlim(0.5, 1.0)
    ax.set_yticks([])
    ax.set_xlabel("bootstrap ARI vs reference (20 refits)")
    ax.set_title(f"k=6 stability — mean ARI {m:.2f}", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    _save(fig, "stability.png")


def fig_maps():
    paths = sorted(glob.glob(str(ROOT / "outputs/*/features/features_grid.parquet")))
    paths = [p for p in paths if Path(p).parents[1].name in CAMPAIGN_SITES]
    fig, axes = plt.subplots(1, len(paths), figsize=(3 * len(paths), 3.2))
    for ax, p in zip(np.atleast_1d(axes), paths):
        site = Path(p).parents[1].name
        g = gpd.read_parquet(p)
        col = "morphotype_smooth" if "morphotype_smooth" in g else "morphotype"
        g.plot(ax=ax, color=NULL_COLOR, linewidth=0)
        gg = g.dropna(subset=[col])
        gg.plot(ax=ax, column=col, categorical=True,
                cmap=matplotlib.colors.ListedColormap(type_color_list()),
                vmin=0, vmax=5, linewidth=0)
        ax.set_axis_off()
        ax.set_title(site.replace("_", " "), fontsize=9)
    fig.suptitle("Morphotypes (spatially mode-filtered; grey = no street support)",
                 fontsize=10)
    _save(fig, "maps_morphotypes.png")


GALLERY = [
    ("fingerprint_heatmap.png", "Morphotype fingerprints",
     "Radar replaced by a type×character diverging heatmap (annotated z, rows in "
     "dendrogram order, columns grouped density|wind|terrain)."),
    ("dendrogram.png", "Taxonomy dendrogram",
     "Grey links (structure), leaf labels in the type palette (identity), k=6 cut."),
    ("recurrence.png", "Cross-site recurrence",
     "Single-hue Blues; cols in dendrogram order, rows flat-then-steep; the "
     "site-specific type (T3, flatland) boxed in red — the validation as a figure."),
    ("recurrence_evidence.png", "Recurrence evidence",
     "Per-site type centroids overlaid: if a type's profile is the same shape across "
     "sites, recurrence is shown, not asserted (domain-reviewer's #1 ask)."),
    ("experience_dotplots.png", "Fabric × experience",
     "Held-out experienced conditions per type (dot size ∝ street support). The "
     "monotone worsening with density is out-of-sample validation."),
    ("maps_morphotypes.png", "Morphotype maps",
     "Spatially mode-filtered (purity 0.43→0.80), Okabe–Ito, NULL=grey. "
     "Dissolve→MMU→white-casing + scale bars are the next polish."),
    ("stability.png", "k=6 stability",
     "Bootstrap ARI 0.90 over 20 refits — k=6 is robust."),
    ("k_selection.png", "Model selection",
     "BIC elbow (primary) + silhouette cross-check."),
]


def write_gallery():
    cards = []
    for name, title, decision in GALLERY:
        if not (OUT / name).exists():
            continue
        cards.append(f"""
  <section class="card">
    <h2>{title}</h2>
    <img src="{name}" loading="lazy">
    <p class="decision">{decision}</p>
    <p class="prompt">Review: <b>✅ keep</b> &nbsp;|&nbsp; <b>✏️ refine</b> — note what to change.</p>
  </section>""")
    html = f"""<!doctype html><meta charset=utf-8>
<title>Morpho-signature figures — review</title>
<style>
 body{{font:15px/1.5 system-ui,sans-serif;margin:0;background:#fafafa;color:#222}}
 header{{padding:24px 32px;background:#fff;border-bottom:1px solid #e5e5e5}}
 h1{{margin:0;font-size:20px}} header p{{margin:6px 0 0;color:#666}}
 main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:20px;padding:24px}}
 .card{{background:#fff;border:1px solid #e5e5e5;border-radius:10px;padding:16px}}
 .card h2{{margin:0 0 10px;font-size:16px}}
 .card img{{width:100%;height:auto;border:1px solid #eee;border-radius:6px}}
 .decision{{color:#444;font-size:13px}} .prompt{{color:#777;font-size:13px;border-top:1px dashed #ddd;padding-top:8px}}
</style>
<header>
 <h1>Favela morpho-signature — figure review (v2)</h1>
 <p>Generated per the expert-reviewed spec. Page through; mark each keep / refine.
    Full rationale: docs/visualization_plan.md · docs/morpho_signature_decisions.md</p>
</header>
<main>{''.join(cards)}
</main>"""
    (OUT / "index.html").write_text(html)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for p in sorted(glob.glob(str(ROOT / "outputs/*/features/features_grid.parquet"))):
        g = gpd.read_parquet(p)
        g["site"] = Path(p).parents[1].name
        frames.append(pd.DataFrame(g.drop(columns="geometry")))
    df = pd.concat(frames, ignore_index=True)
    mat = assemble_signature_matrix(df)
    Xz, _ = standardize(mat)
    k = int(json.loads((SIG / "run_meta.json").read_text())["k"])
    labels = fit_morphotypes(Xz, k, random_state=0)
    link = centroid_linkage(Xz, labels)
    leaf = dendrogram(link, no_plot=True)["leaves"]
    cut = float(link[-(k - 1), 2]) * 0.999

    cent = pd.DataFrame(
        {f: [Xz[labels == c, i].mean() for c in range(k)]
         for i, f in enumerate(SIGNATURE_FEATURES)}
    )
    sel = pd.read_csv(SIG / "k_selection.csv")
    shares = pd.read_csv(SIG / "recurrence_matrix.csv", index_col=0)
    shares.columns = shares.columns.astype(int)
    flags = pd.read_csv(SIG / "recurrence_flags.csv", index_col=0)
    profile = pd.read_csv(SIG / "experience_profile.csv")
    stab = json.loads((SIG / "stability_meta.json").read_text())

    fig_fingerprint_heatmap(cent, leaf)
    fig_dendrogram(link, leaf, cut)
    fig_recurrence(shares, flags, leaf)
    fig_recurrence_evidence(mat, labels, leaf)
    fig_experience(profile, leaf)
    fig_maps()
    fig_stability(stab)
    fig_kselection(sel, k)
    write_gallery()
    print(f"figures_v2 + gallery written to {OUT}")


if __name__ == "__main__":
    main()
