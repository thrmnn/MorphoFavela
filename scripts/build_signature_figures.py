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


def _scalebar(ax, length_m=200):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x = x0 + 0.06 * (x1 - x0)
    y = y0 + 0.06 * (y1 - y0)
    ax.plot([x, x + length_m], [y, y], color="black", lw=2, solid_capstyle="butt")
    ax.text(x + length_m / 2, y + 0.015 * (y1 - y0), f"{length_m} m",
            ha="center", va="bottom", fontsize=6)


def fig_maps(mmu_cells=10):
    """Mode-filtered morphotypes, dissolved to regions: islands below the MMU
    dropped, thin white casing between regions, site outline, constant ground
    scale (panel width ∝ site bbox width), scale bar. NULL = grey."""
    paths = sorted(glob.glob(str(ROOT / "outputs/*/features/features_grid.parquet")))
    paths = [p for p in paths if Path(p).parents[1].name in CAMPAIGN_SITES]
    grids = {Path(p).parents[1].name: gpd.read_parquet(p) for p in paths}
    widths = [g.total_bounds[2] - g.total_bounds[0] for g in grids.values()]
    fig, axes = plt.subplots(
        1, len(grids), figsize=(2.6 * len(grids), 3.4),
        gridspec_kw={"width_ratios": widths},
    )
    cmap = matplotlib.colors.ListedColormap(type_color_list())
    for ax, (site, g) in zip(np.atleast_1d(axes), grids.items()):
        col = "morphotype_smooth" if "morphotype_smooth" in g else "morphotype"
        cell_area = float(g.geometry.area.median())
        g.plot(ax=ax, color=NULL_COLOR, linewidth=0)
        gg = g.dropna(subset=[col])
        diss = gg.dissolve(by=col).reset_index().explode(index_parts=False)
        diss = diss[diss.geometry.area >= mmu_cells * cell_area]
        diss.plot(ax=ax, column=col, categorical=True, cmap=cmap,
                  vmin=0, vmax=5, edgecolor="white", linewidth=0.3)
        gpd.GeoSeries([g.geometry.union_all()], crs=g.crs).boundary.plot(
            ax=ax, color="0.25", linewidth=0.6)
        ax.set_xlim(g.total_bounds[0], g.total_bounds[2])
        ax.set_ylim(g.total_bounds[1], g.total_bounds[3])
        ax.set_aspect("equal")
        _scalebar(ax)
        ax.set_axis_off()
        ax.set_title(site.replace("_", " "), fontsize=9)
    fig.suptitle("Morphotypes — mode-filtered, dissolved (MMU "
                 f"{mmu_cells} cells); grey = no street support", fontsize=10)
    _save(fig, "maps_morphotypes.png")


def fig_naive_vs_support(site="vidigal"):
    """Why the change-of-support matters: naive nearest-k mean SVF (fills every
    cell) vs support-aware p10 with NULL grey (honest about the ~65% unsupported)."""
    g = gpd.read_parquet(ROOT / "outputs" / site / "features" / "features_grid.parquet")
    obs = gpd.read_parquet(ROOT / "outputs" / site / "features" / "features_street.parquet")
    cent = g.copy()
    cent["geometry"] = g.geometry.centroid
    near = gpd.sjoin_nearest(cent[["geometry"]], obs[["svf", "geometry"]], how="left")
    naive = near.groupby(near.index)["svf"].mean().reindex(range(len(g)))

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    g.assign(naive=naive.to_numpy()).plot(
        ax=axes[0], column="naive", cmap="YlOrBr_r", linewidth=0, legend=True,
        legend_kwds={"shrink": 0.6, "label": "SVF"})
    axes[0].set_title("Naive nearest-k mean\n(every cell filled)", fontsize=9)
    g.plot(ax=axes[1], color=NULL_COLOR, linewidth=0)
    g.dropna(subset=["svf_p10"]).plot(
        ax=axes[1], column="svf_p10", cmap="YlOrBr_r", linewidth=0, legend=True,
        legend_kwds={"shrink": 0.6, "label": "SVF p10"})
    axes[1].set_title("Support-aware p10\n(grey = no observer, not imputed)", fontsize=9)
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_axis_off()
    fig.suptitle(f"{site} — naive aggregation invents values the support-aware "
                 "table withholds", fontsize=10)
    _save(fig, "naive_vs_support.png")


def fig_terrain_sensitivity(leaf):
    """Does the flat-datum error track the prioritized types? Per morphotype:
    slope (error driver) and the gap between measured SVF and an analytic
    flat-canyon SVF from H/W (proxy for the terrain/3D correction)."""
    rows = []
    for p in sorted(glob.glob(str(ROOT / "outputs/*/features/features_grid.parquet"))):
        site = Path(p).parents[1].name
        if site not in CAMPAIGN_SITES:
            continue
        g = gpd.read_parquet(p)[["zone_id", "morphotype_smooth"]]
        obs = gpd.read_parquet(Path(p).parents[1] / "features" / "features_street.parquet")
        o = obs[obs["has_hw"]].merge(g, on="zone_id", how="left")
        o = o.dropna(subset=["morphotype_smooth", "HW", "svf", "slope_deg"])
        o["svf_flat"] = np.cos(np.arctan(2 * o["HW"]))
        o["svf_gap"] = o["svf"] - o["svf_flat"]
        rows.append(o[["morphotype_smooth", "slope_deg", "svf_gap"]])
    d = pd.concat(rows, ignore_index=True)
    order = [c for c in leaf if c in d["morphotype_smooth"].unique()]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, (col, lab) in zip(axes, [("slope_deg", "slope (°)"),
                                     ("svf_gap", "SVF gap (measured − flat-canyon)")]):
        data = [d[d["morphotype_smooth"] == c][col].to_numpy() for c in order]
        bp = ax.boxplot(data, vert=False, showfliers=False, patch_artist=True)
        for patch, c in zip(bp["boxes"], order):
            patch.set_facecolor(TYPE_COLORS[c])
            patch.set_alpha(0.8)
        ax.set_yticks(range(1, len(order) + 1))
        ax.set_yticklabels([f"T{c}" for c in order])
        ax.set_xlabel(lab, fontsize=9)
        ax.grid(axis="x", color="0.92")
    axes[0].axvline(20, ls=":", color="crimson")
    axes[0].text(20, 0.5, " flat-datum risk →", color="crimson", fontsize=7)
    fig.suptitle("Terrain sensitivity by morphotype — prioritized types T4/T5 "
                 "(steep/dense) vs the known flat-datum error", fontsize=10)
    _save(fig, "terrain_sensitivity.png")


TYPE_SITE_ORDER = ["vidigal", "rocinha", "complexo_do_alemao", "riodaspedras", "maré"]
SITE_CLEAN = {
    "vidigal": "Vidigal", "rocinha": "Rocinha", "complexo_do_alemao": "C. do Alemão",
    "riodaspedras": "R. das Pedras", "maré": "Maré",
}


def fig_type_site_fingerprint(shares, leaf):
    """Morphotype × site composition fingerprint, commonality-forward (for P4/E2).

    Unlike ``recurrence.png`` (which boxes the site-specific exception in red),
    this foregrounds the SHARED fabric: every type's share in each favela, plus a
    per-type count of how many favelas carry it (≥ 5%). The point is that the
    five favelas draw on one recurrent, classifiable type set."""
    types = [c for c in leaf if c in shares.columns]
    M = shares.reindex(index=TYPE_SITE_ORDER, columns=types).to_numpy().T  # types × sites
    n_present = (M >= 0.05).sum(axis=1)
    vmax = float(np.nanmax(M))
    n_sites = len(TYPE_SITE_ORDER)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    im = ax.imshow(M, cmap="YlGnBu", vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(n_sites))
    ax.set_xticklabels([SITE_CLEAN[s] for s in TYPE_SITE_ORDER], rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(len(types)))
    ax.set_yticklabels([TYPE_NAMES[c] for c in types], fontsize=8)
    for lbl, c in zip(ax.get_yticklabels(), types):
        lbl.set_color(TYPE_COLORS[c])
        lbl.set_fontweight("bold")
    for (i, j), v in np.ndenumerate(M):
        if not np.isnan(v):
            ax.text(j, i, f"{v * 100:.0f}%", ha="center", va="center", fontsize=7,
                    color="white" if v > 0.5 * vmax else "#222222")
    # Right-margin recurrence count per type (# favelas carrying the type ≥5%).
    ax.text(n_sites - 0.3, -0.62, "in", ha="left", va="center", fontsize=6.5, color="#333333")
    for i in range(len(types)):
        recur = n_present[i] >= n_sites - 1  # present in ≥4 of 5
        ax.text(n_sites - 0.3, i, f"{n_present[i]}/{n_sites}", ha="left", va="center",
                fontsize=7.5, color="#1A7F37" if recur else "#999999", fontweight="bold")
    ax.set_xlim(-0.5, n_sites - 0.5 + 1.1)
    n_ge4 = int((n_present >= n_sites - 1).sum())
    universal = [f"T{types[i]}" for i in range(len(types)) if n_present[i] == n_sites]
    conditional = [f"T{types[i]}" for i in range(len(types)) if n_present[i] <= 2]
    uni_txt = " & ".join(universal) + (" universal" if universal else "")
    cond_txt = ("/".join(conditional) + " terrain-conditional") if conditional else ""
    detail = "; ".join(t for t in (uni_txt, cond_txt) if t)
    ax.set_title(
        f"Morphotype × site fingerprint — a recurrent type set\n({n_ge4}/{len(types)} types "
        f"in ≥4 favelas; {detail})",
        fontsize=8.5, pad=6,
    )
    fig.colorbar(im, ax=ax, label="share of favela's cells", shrink=0.8, pad=0.12)
    _save(fig, "type_site_fingerprint.png")


# Themed groups → each becomes an anchored gallery section with sidebar nav.
GALLERY_GROUPS = [
    ("Signature", "signature", [
        ("morphotype_schematics.png", "What the 6 types look like",
         "Idealized 40 m street section per morphotype (driven by the centroids): "
         "density, height spread, slope and canyon depth at a glance."),
        ("composition_by_site.png", "Composition per favela",
         "% of each morphotype per site — hillside favelas are T4 Hillside Core-"
         "dominated, flat ones T5 Saturated Core; topography drives the mix."),
        ("party_wall_by_type.png", "Configuration: party-wall ratio",
         "Shared-wall ratio by morphotype — a relational metric the intensity vector "
         "never saw. Favela fabric is highly fused everywhere (0.6–1.0 vs ~0.1 for "
         "detached formal blocks); flat types are fully party-walled, hillside types "
         "more stepped — a new configuration axis (council's top 'what's missing')."),
        ("fingerprint_heatmap.png", "Morphotype fingerprints",
         "Type×character diverging heatmap (annotated z, dendrogram order, columns "
         "grouped density|wind|terrain)."),
        ("dendrogram.png", "Taxonomy dendrogram",
         "Grey links (structure), leaf labels in the type palette, k=6 cut."),
        ("recurrence.png", "Cross-site recurrence",
         "Single-hue Blues; the flatland-specific types (T1, T5) boxed red — the "
         "validation as a figure."),
        ("type_site_fingerprint.png", "Morphotype × site fingerprint (P4/E2)",
         "Commonality-forward composition heatmap with full type names + a per-type "
         "count of how many favelas carry it (≥5%): the five favelas draw on one "
         "recurrent, classifiable type set — evidence for the informal-typology claim."),
        ("recurrence_evidence.png", "Recurrence evidence",
         "Per-site type centroids overlaid: same profile shape across sites = "
         "recurrence shown, not asserted."),
        ("maps_morphotypes.png", "Morphotype maps",
         "Mode-filtered + dissolved to regions; Okabe–Ito, NULL=grey, scale bars."),
        ("stability.png", "k=6 stability",
         "Bootstrap ARI 0.90 over 20 refits — k=6 is robust."),
        ("k_selection_rigor.png", "k-selection rigor (honest)",
         "The honest verdict: internal indices (silhouette, CH, DB) favour k=2–3, BIC "
         "elbow→k=3 — NO index peaks at k=6. k=6 is a domain-driven interpretability "
         "choice, supported by cross-site reproducibility (leave-one-site-out ARI "
         "0.78), not by a distance-based optimum."),
        ("beco_by_type.png", "Beco grain (street-network config)",
         "Circulation by morphotype: T4 Hillside Core has the FINEST alley grain "
         "(9.5 junctions/ha), not the densest T5 (3.0/ha) — so the street wiring is "
         "orthogonal information the density field misses (council's 2nd config feature)."),
    ]),
    ("Morphotopes (tissue scale)", "morphotope", [
        ("morphotope_maps.png", "Morphotope maps (block scale)",
         "The favela signature at ~50 m tissue scale, not the 10 m cell — coherent "
         "regions, not salt-and-pepper. Vidigal = fringe tissue, Rocinha = dense "
         "hillside-core, flat sites have dark flat-core tissue."),
        ("morphotope_profile.png", "What each tissue is made of",
         "Cell-type composition per morphotope. Answers the T2/T3 critique: T3 isn't "
         "noise — it concentrates in the flat dense-core tissue M4, a coherent "
         "recurring tissue."),
        ("morphotope_recurrence.png", "Tissue recurrence",
         "Morphotope share per favela — 4 of 5 tissues recur across ≥3 sites; the "
         "dense hillside-core tissue dominates the hillside favelas. A stronger "
         "recurrence claim than the cell scale."),
        ("morphotope_stability.png", "Tissue stability (k=5)",
         "k=5 morphotopes are highly stable — bootstrap ARI 0.994 (min 0.965) over 20 "
         "refits. k chosen by BIC elbow (a parsimony argument; BIC keeps decreasing, "
         "so it's the elbow, not an optimum)."),
    ]),
    ("Roughness (track/roughness)", "roughness", [
        ("roughness_validity.png", "⚠ Physical validity (read first)",
         "The headline: morphometric z0/zd is physically INVALID in 53–75% of "
         "cells — zd exceeds the tallest building, or z0 collapses to ~0 (skimming "
         "asymptote). λp>0.5 is past every method's calibration. Per-cell estimate "
         "is not trustworthy; the patch scale + CFD (R-C) is the path."),
        ("roughness_methods.png", "Cross-method z0 envelope (the result)",
         "Macdonald/Raupach/Millward-Hopkins/Kanda diverge up to ~20× in the favela "
         "regime. With no validation, the envelope IS the result: morphometry can't "
         "constrain favela z0 to better than ~1.5 orders — CFD is required."),
        ("roughness_anisotropy.png", "Per-cell λf anisotropy",
         "Individual cells are directionally anisotropic (median ~0.37) — the "
         "honest directional statistic. The site-median rose cancels it (varying "
         "cell orientations), which is why the rose looks round."),
        ("roughness_rose.png", "Directional z0(θ) rose",
         "180°-symmetric by construction (frontal area is the same for opposite "
         "wind) — z0(θ)=z0(θ+180°); morphometry can never break N/S symmetry, only "
         "CFD can. Radial axis zoomed to the (compressed) modulation band."),
        ("roughness_zd_ratio.png", "Displacement zd / H_mean",
         "Diverging map centred at 1: red = displacement exceeds mean building "
         "height — but note zd>H_max (impossible) in many cells (see validity)."),
        ("roughness_map.png", "Roughness z0 map (illustrative)",
         "Per-cell Kanda z0. Physically ill-posed at 10 m (z0 is a blended quantity "
         "over many elements) — illustrative only; the 100 m patch is the smallest "
         "defensible scale."),
        ("roughness_slope.png", "Roughness vs slope (confound)",
         "z0 rises on steep slopes — a confound, NOT a finding: flat-datum λf/σH "
         "absorbs the hillside. Terrain-following datum is the fix; slope stays out "
         "of the CFD inlet z0."),
    ]),
    ("Prioritization & honesty", "prioritization", [
        ("typology_failure_lookup.png", "Typology → environmental failure",
         "The money figure: per-type WHO-2h winter-sun failure rate (held-out, "
         "ray-cast). Monotone with a regime jump at T3→T4, saturating at T4/T5 "
         "(71–73%); tight per-favela dots at T4/T5 = good transfer."),
        ("typology_parsimony.png", "Parsimony (LOSO)",
         "Type-only transfers out-of-site at AUC-PR 0.77 (vs 0.85 full vector, 0.64 "
         "baseline) — the discrete code keeps most of the signal at far lower "
         "dimension; the continuous vector adds a modest Δ0.09."),
        ("typology_calibration.png", "Transfer calibration + PR (LOSO)",
         "Leave-one-site-out: calibrated (slightly under-confident — isotonic recal "
         "queued) out-of-site predictions + PR curve (AP 0.88) vs the prevalence "
         "baseline. The honest transfer evidence."),
        ("typology_variance.png", "Variance: type vs site vs interaction",
         "Of the systematic variance in failure, morphotype dominates (17% vs 2% "
         "site vs 0.7% site×type). The tiny site×type interaction = the type→failure "
         "mapping transfers across favelas. Is it the types a favela has, or where "
         "it is? — the types."),
        ("typology_blind_riskmap.png", "Blind risk map (the payoff)",
         "Morphology-in → prioritised WHO-2h sun-failure-out for the 3 calibration "
         "favelas the model NEVER saw: cells projected through the frozen campaign "
         "GMM → per-type failure rate. Jacarezinho 67%, Borel/Morro 45%; 100% of "
         "built cells assignable."),
        ("typology_calibration_isotonic.png", "Isotonic recalibration",
         "Raw LOSO is under-confident (ECE 0.090); isotonic recal (fit on training "
         "sites only) pulls it onto the diagonal (ECE 0.026, 3.5× better). The "
         "predictor is now calibrated for prioritisation."),
        ("priority_map.png", "Morphometrics-only priority (WS-B)",
         "Geometry-only deprivation index, worst-decile per cell, rank classes. "
         "No CFD; weights provisional."),
        ("experience_dotplots.png", "Fabric × experience",
         "Held-out experienced conditions per type (dot size ∝ support). Monotone "
         "worsening with density = out-of-sample validation."),
        ("naive_vs_support.png", "Naive vs support-aware",
         "Nearest-k mean invents a value for every cell; support-aware p10 leaves "
         "the ~65% unsupported cells grey."),
        ("terrain_sensitivity.png", "Terrain sensitivity",
         "Does the flat-datum error track the prioritized types? Slope + the "
         "measured-vs-flat-canyon SVF gap per morphotype."),
    ]),
]


def write_gallery():
    blocks = []
    toc = []
    for group, anchor, items in GALLERY_GROUPS:
        cards = []
        for name, title, decision in items:
            if not (OUT / name).exists():
                continue
            fid = "fig-" + name.replace(".png", "")
            cards.append(f"""
    <section class="card" id="{fid}">
      <h3>{title}</h3>
      <img src="{name}" loading="lazy" onclick="zoom('{name}','{title}')">
      <p class="decision">{decision}</p>
      <p class="prompt">Review: <b>✅ keep</b> &nbsp;|&nbsp; <b>✏️ refine</b></p>
    </section>""")
        if cards:
            toc.append(f'<a href="#{anchor}">{group}</a>')
            blocks.append(f'<h2 id="{anchor}">{group}</h2>'
                          f'<div class="grid">{"".join(cards)}</div>')
    sidebar = ('<nav class="toc"><div class="toc-h">Sections</div>'
               + "".join(toc) + "</nav>")
    html = f"""<!doctype html><meta charset=utf-8>
<title>Morpho-signature figures — review</title>
<style>
 body{{font:15px/1.5 system-ui,sans-serif;margin:0;background:#fafafa;color:#222}}
 .top{{position:sticky;top:0;background:#fffd;backdrop-filter:blur(6px);padding:10px 24px;border-bottom:1px solid #e5e5e5;font-weight:600;font-size:14px;z-index:30}}
 .top a{{color:#1a6fb5;text-decoration:none}}
 .layout{{display:flex;align-items:flex-start;max-width:1400px;margin:0 auto}}
 .layout aside{{position:sticky;top:42px;flex:0 0 220px;max-height:calc(100vh - 42px);overflow:auto;padding:20px 8px 20px 24px}}
 .toc{{font-size:13px;border-left:2px solid #e5e5e5;padding-left:12px}}
 .toc-h{{font-weight:700;color:#888;text-transform:uppercase;letter-spacing:.04em;font-size:11px;margin-bottom:8px}}
 .toc a{{display:block;color:#666;text-decoration:none;padding:4px 0}} .toc a:hover{{color:#1a6fb5}}
 main{{flex:1 1 auto;min-width:0;padding:18px 24px 60px}}
 h1{{font-size:20px;margin:8px 0}} main>h2{{font-size:16px;border-bottom:1px solid #e5e5e5;padding-bottom:5px;margin:26px 0 12px}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}}
 .card{{background:#fff;border:1px solid #e5e5e5;border-radius:10px;padding:16px}}
 .card h3{{margin:0 0 10px;font-size:15px}}
 .card img{{width:100%;height:auto;border:1px solid #eee;border-radius:6px;cursor:zoom-in}}
 .decision{{color:#444;font-size:13px}} .prompt{{color:#777;font-size:13px;border-top:1px dashed #ddd;padding-top:8px}}
 html{{scroll-behavior:smooth}} :target{{scroll-margin-top:52px}}
 #lb{{display:none;position:fixed;inset:0;background:rgba(0,0,0,.9);z-index:99;cursor:zoom-out;flex-direction:column;align-items:center;justify-content:center}}
 #lb img{{max-width:96vw;max-height:90vh}} #lb p{{color:#eee;margin:10px;font-size:15px}}
 @media(max-width:820px){{.layout{{display:block}}.layout aside{{position:static;max-height:none;flex-basis:auto;padding:12px 24px 0}}.toc{{border-left:none}}.toc a{{display:inline-block;margin-right:14px}}}}
</style>
<div class="top"><a href="/outputs/_hub/index.html">← Project hub</a>
 <span style="color:#999;margin:0 8px">›</span><span style="color:#666">Signature &amp; roughness figures</span></div>
<div class="layout">
<aside>{sidebar}</aside>
<main>
 <h1>Favela morpho-signature &amp; roughness — figure review</h1>
 <p style="color:#666;margin:0 0 6px">Click any figure to enlarge. Rationale:
    <a href="/outputs/_hub/docs/visualization_plan.html">visualization plan</a> ·
    <a href="/outputs/_hub/docs/morpho_signature_decisions.html">signature decisions</a> ·
    <a href="/outputs/_hub/docs/roughness_decisions.html">roughness decisions</a></p>
 {''.join(blocks)}
</main></div>
<div id="lb" onclick="this.style.display='none'"><img id="lbimg"><p id="lbcap"></p></div>
<script>
 function zoom(src,cap){{lbimg.src=src;lbcap.textContent=cap;lb.style.display='flex'}}
 addEventListener('keydown',e=>{{if(e.key==='Escape')lb.style.display='none'}});
</script>"""
    (OUT / "index.html").write_text(html)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for p in sorted(glob.glob(str(ROOT / "outputs/*/features/features_grid.parquet"))):
        site = Path(p).parents[1].name
        if site not in CAMPAIGN_SITES:  # calibration sites kept aside (5-site viz)
            continue
        g = gpd.read_parquet(p)
        g["site"] = site
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
    fig_type_site_fingerprint(shares, leaf)
    fig_recurrence_evidence(mat, labels, leaf)
    fig_experience(profile, leaf)
    fig_maps()
    fig_naive_vs_support()
    fig_terrain_sensitivity(leaf)
    fig_stability(stab)
    fig_kselection(sel, k)
    write_gallery()
    print(f"figures_v2 + gallery written to {OUT}")


if __name__ == "__main__":
    main()
