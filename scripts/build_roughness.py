"""R-A — per-cell aerodynamic roughness from morphometry, UMEP-driven.

For each site: derive H_max per cell from the building heights, then compute
Kanda z0/zd (mean + 8 directional sectors), the cross-method z0 spread
(Kan/Mho/Mac/Rau = morphometric uncertainty), and favela extrapolation flags.
Writes features_roughness.parquet per site and a directional roughness-rose figure
into the signature gallery. Decisions: docs/roughness_plan.md.

    python scripts/build_roughness.py
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
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.morphometry.roughness import (  # noqa: E402
    DIRS,
    extrapolation_flags,
    method_spread_z0,
    roughness_vec,
)
from src.morphometry.signature import CAMPAIGN_SITES  # noqa: E402
from src.svf_v2.io import _git_sha  # noqa: E402

FIGS = ROOT / "outputs" / "cross_site" / "signature" / "figures_v2"
HEIGHT_COLS = ["height", "altura", "H", "mean_height", "building_height", "bld_h", "z"]


def derive_hmax(grid: gpd.GeoDataFrame, site_dir: Path) -> np.ndarray:
    """Max building height per cell; fall back to H_mean + 2.5σH where unavailable."""
    fallback = grid["H_mean"].to_numpy() + 2.5 * grid["sigma_h"].fillna(0).to_numpy()
    bpath = site_dir / "morphometrics" / "buildings" / "buildings_with_morphology_metrics.gpkg"
    if not bpath.exists():
        return fallback
    b = gpd.read_file(bpath)
    hcol = next((c for c in HEIGHT_COLS if c in b.columns), None)
    if hcol is None:
        return fallback
    j = gpd.sjoin(b[[hcol, "geometry"]].to_crs(grid.crs),
                  grid[["zone_id", "geometry"]], predicate="intersects")
    zmax = j.groupby("zone_id")[hcol].max()
    return grid["zone_id"].map(zmax).fillna(pd.Series(fallback, index=grid.index)).to_numpy()


def _scalebar(ax, length_m=200):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x, y = x0 + 0.06 * (x1 - x0), y0 + 0.06 * (y1 - y0)
    ax.plot([x, x + length_m], [y, y], color="black", lw=2)
    ax.text(x + length_m / 2, y + 0.015 * (y1 - y0), f"{length_m} m",
            ha="center", va="bottom", fontsize=6)


def main():
    summary = []
    rose = {}
    geoms = {}
    method_med = {}  # site -> {method: median z0}
    for p in sorted(glob.glob(str(ROOT / "outputs/*/features/features_grid.parquet"))):
        site = Path(p).parents[1].name
        g = gpd.read_parquet(p)
        zH = g["H_mean"].to_numpy()
        pai = g["lambda_p"].to_numpy()
        sdev = g["sigma_h"].fillna(0).to_numpy()
        zMax = derive_hmax(g, Path(p).parents[1])
        fai = g["lambda_f_mean"].to_numpy()

        zd_kan, z0_kan = roughness_vec("Kan", zH, fai, pai, zMax, sdev)
        out = g[["zone_id", "geometry"]].copy()
        out["H_max"] = zMax
        out["z0_kan"] = z0_kan
        out["zd_kan"] = zd_kan
        out["zd_exceeds_Hmean"] = zd_kan > zH
        out["zd_over_Hmean"] = np.where(zH > 0, zd_kan / zH, np.nan)
        out["H_mean"] = zH
        out["slope_deg"] = g["slope_deg"].to_numpy()
        for d in DIRS:
            out[f"z0_kan_{d}"] = roughness_vec("Kan", zH, g[f"lambda_f_{d}"].to_numpy(),
                                               pai, zMax, sdev)[1]
        z0s, spread = method_spread_z0(zH, fai, pai, zMax, sdev)
        for m, arr in z0s.items():
            out[f"z0_{m.lower()}"] = arr
        out["z0_method_spread"] = spread
        for k, v in extrapolation_flags(zH, pai, zMax, sdev).items():
            out[k] = v
        out.to_parquet(Path(p).parents[1] / "features" / "features_roughness.parquet")

        m = np.isfinite(z0_kan)
        summary.append({
            "site": site, "n": int(m.sum()),
            "z0_kan_med": float(np.nanmedian(z0_kan)),
            "zd_kan_med": float(np.nanmedian(zd_kan)),
            "frac_zd_gt_Hmean": float(np.nanmean((zd_kan > zH)[m])),
            "frac_pai_over_0.5": float(np.nanmean(out["flag_pai_over_envelope"][m])),
            "z0_spread_med": float(np.nanmedian(spread)),
        })
        method_med[site] = {mm: float(np.nanmedian(out[f"z0_{mm.lower()}"]))
                            for mm in ("Kan", "Mho", "Mac", "Rau")}
        if site in CAMPAIGN_SITES:
            rose[site] = [np.nanmedian(out[f"z0_kan_{d}"]) for d in DIRS]
            geoms[site] = out

    # directional roughness rose (campaign sites), shared radial scale
    sites = [s for s in CAMPAIGN_SITES if s in rose]
    ang = np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315])
    ang = np.concatenate([ang, ang[:1]])
    rmax = max(max(v) for v in rose.values())
    fig, axes = plt.subplots(1, len(sites), figsize=(2.5 * len(sites), 3),
                             subplot_kw={"polar": True})
    for ax, s in zip(np.atleast_1d(axes), sites):
        r = rose[s] + rose[s][:1]
        ax.plot(ang, r, color="#1a6fb5")
        ax.fill(ang, r, color="#1a6fb5", alpha=0.25)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(0, rmax)
        ax.set_xticks(ang[:-1])
        ax.set_xticklabels(DIRS, fontsize=6)
        ax.set_yticklabels([])
        ax.set_title(s.replace("_", " "), fontsize=8)
    fig.suptitle("Directional roughness z0(θ) — median per sector (Kanda 2013)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "roughness_rose.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # spatial z0 map (campaign sites), shared robust scale; cividis (no palette clash)
    camp = [(s, geoms[s]) for s in CAMPAIGN_SITES if s in geoms]
    pooled = np.concatenate([g["z0_kan"].dropna().to_numpy() for _, g in camp])
    vmax = float(np.nanpercentile(pooled, 95))
    widths = [g.total_bounds[2] - g.total_bounds[0] for _, g in camp]
    fig, axes = plt.subplots(1, len(camp), figsize=(2.6 * len(camp), 3.4),
                             gridspec_kw={"width_ratios": widths})
    for ax, (s, g) in zip(np.atleast_1d(axes), camp):
        g.plot(ax=ax, color="#E0E0E0", linewidth=0)
        gg = g.dropna(subset=["z0_kan"])
        gg.plot(ax=ax, column="z0_kan", cmap="cividis", vmin=0, vmax=vmax, linewidth=0)
        gpd.GeoSeries([g.geometry.union_all()], crs=g.crs).boundary.plot(
            ax=ax, color="0.25", linewidth=0.6)
        ax.set_xlim(g.total_bounds[0], g.total_bounds[2])
        ax.set_ylim(g.total_bounds[1], g.total_bounds[3])
        ax.set_aspect("equal")
        _scalebar(ax)
        ax.set_axis_off()
        ax.set_title(s.replace("_", " "), fontsize=9)
    sm = plt.cm.ScalarMappable(cmap="cividis",
                               norm=plt.Normalize(vmin=0, vmax=vmax))
    fig.colorbar(sm, ax=list(np.atleast_1d(axes)), shrink=0.6, label="z0 (m), Kanda")
    fig.suptitle("Aerodynamic roughness z0 (Kanda 2013); most cells λp>0.5 = "
                 "out of calibration envelope (see flags)", fontsize=9)
    fig.savefig(FIGS / "roughness_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # cross-method comparison — the "σH premium" (Kanda/MHN vs Macdonald baseline)
    msites = list(method_med)
    methods = ["Mac", "Rau", "Mho", "Kan"]
    mcolor = {"Mac": "#999999", "Rau": "#56B4E9", "Mho": "#E69F00", "Kan": "#0072B2"}
    x = np.arange(len(msites))
    w = 0.2
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    for i, mm in enumerate(methods):
        ax.bar(x + (i - 1.5) * w, [method_med[s][mm] for s in msites], w,
               label=mm, color=mcolor[mm])
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", " ") for s in msites], rotation=25, ha="right",
                       fontsize=8)
    ax.set_ylabel("median z0 (m)")
    ax.set_title("Cross-method z0 disagreement — the four methods span up to ~20× "
                 "in the λp>0.5 favela regime (which is right is unknown w/o CFD)",
                 fontsize=8.5)
    ax.legend(fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(FIGS / "roughness_methods.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(method_med).T.to_csv(
        ROOT / "outputs" / "cross_site" / "roughness" / "method_medians.csv")

    # zd/H_mean ratio map — the headline finding spatially (where zd exceeds H_mean)
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vcenter=1.0, vmin=0.0, vmax=2.0)
    fig, axes = plt.subplots(1, len(camp), figsize=(2.6 * len(camp), 3.4),
                             gridspec_kw={"width_ratios": widths})
    for ax, (s, g) in zip(np.atleast_1d(axes), camp):
        g.plot(ax=ax, color="#E0E0E0", linewidth=0)
        gg = g.dropna(subset=["zd_over_Hmean"])
        gg.plot(ax=ax, column="zd_over_Hmean", cmap="RdBu_r", norm=norm, linewidth=0)
        gpd.GeoSeries([g.geometry.union_all()], crs=g.crs).boundary.plot(
            ax=ax, color="0.25", linewidth=0.6)
        ax.set_xlim(g.total_bounds[0], g.total_bounds[2])
        ax.set_ylim(g.total_bounds[1], g.total_bounds[3])
        ax.set_aspect("equal")
        _scalebar(ax)
        ax.set_axis_off()
        ax.set_title(s.replace("_", " "), fontsize=9)
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
    fig.colorbar(sm, ax=list(np.atleast_1d(axes)), shrink=0.6,
                 label="zd / H_mean  (red = displacement exceeds mean height)")
    fig.suptitle("Displacement height relative to mean building height — "
                 "zd>H_mean in 70–93% of cells (heterogeneity signature)", fontsize=8.5)
    fig.savefig(FIGS / "roughness_zd_ratio.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # z0 vs slope — the terrain confound no morphometric method separates.
    # Bins need >=MIN_N cells or the steep tail is a few noisy edge cells.
    MIN_N = 30
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    bins = np.array([0, 5, 10, 15, 20, 25, 30, 35])  # cap at 35°: above is sparse
    centers = 0.5 * (bins[:-1] + bins[1:])
    for s, g in camp:
        d = g.dropna(subset=["z0_kan", "slope_deg"])
        idx = np.digitize(d["slope_deg"].to_numpy(), bins) - 1
        med = [np.nanmedian(d["z0_kan"].to_numpy()[idx == b])
               if (idx == b).sum() >= MIN_N else np.nan
               for b in range(len(bins) - 1)]
        ax.plot(centers, med, "o-", ms=3, label=s.replace("_", " "))
    ax.set_xlabel("terrain slope (°)")
    ax.set_ylabel("median z0 (m), Kanda")
    ax.set_title("Morphometric z0 rises on steep slopes — but flat-datum λf/σH "
                 "absorbs the hillside; no method separates terrain from fabric",
                 fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(color="0.92")
    fig.tight_layout()
    fig.savefig(FIGS / "roughness_slope.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    (ROOT / "outputs" / "cross_site" / "signature" / "roughness_meta.json").write_text(
        json.dumps({"git_sha": _git_sha(),
                    "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "method_primary": "Kan", "note": "UMEP RoughnessCalc; H_max per cell"},
                   indent=2))
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
