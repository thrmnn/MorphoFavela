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


def main():
    summary = []
    rose = {}
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
        if site in CAMPAIGN_SITES:
            rose[site] = [np.nanmedian(out[f"z0_kan_{d}"]) for d in DIRS]

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

    (ROOT / "outputs" / "cross_site" / "signature" / "roughness_meta.json").write_text(
        json.dumps({"git_sha": _git_sha(),
                    "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "method_primary": "Kan", "note": "UMEP RoughnessCalc; H_max per cell"},
                   indent=2))
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
