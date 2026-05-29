#!/usr/bin/env python3
"""Validate the MorphoFavela Tregenza-145 SVF engine against UMEP's shadow-casting SVF.

This closes the §10.3 limitation in `docs/technical_report/technical_report.md`:
"SVF validation against benchmark tools pending."

The two engines use algorithmically distinct approaches:

* **MorphoFavela (this repo).** Ray-casting on a triangle mesh through 145 Tregenza
  hemispherical patches; sampling at street-level (z = 1.5 m above ground).
* **UMEP** (Lindberg & Holmer 2010, used in SOLWEIG). Shadow-casting on a
  digital surface model through 153 hemispherical patches; integrated at
  the DSM surface (effectively z = 0 above ground for street pixels).

Agreement between the two is what we want to bank for the methods chapter.
A systematic offset is expected (UMEP at ground-level vs MorphoFavela at 1.5 m);
the slope and r² of the per-cell scatter is what reviewers care about.

UMEP source is auto-fetched on first run into ``vendor/umep_processing/``
(gitignored) and patched to bypass its QGIS-only ``util/__init__.py``.

Usage:
    python scripts/validate_svf_against_umep.py --site vidigal
    python scripts/validate_svf_against_umep.py --site vidigal --pixel-size 1.0
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

logger = logging.getLogger(__name__)

UMEP_REPO = "https://github.com/UMEP-dev/UMEP-processing.git"
UMEP_VENDOR = PROJECT_ROOT / "vendor" / "umep_processing"


def ensure_umep() -> Path:
    """Clone UMEP-processing to vendor/ if missing; patch QGIS-only __init__'s."""
    if not UMEP_VENDOR.exists():
        logger.info("Cloning UMEP-processing to %s ...", UMEP_VENDOR)
        UMEP_VENDOR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(
            ["git", "clone", "--depth", "1", UMEP_REPO, str(UMEP_VENDOR)],
            stdout=subprocess.DEVNULL,
        )
    for stub in ("__init__.py", "util/__init__.py", "functions/__init__.py"):
        (UMEP_VENDOR / stub).write_text("")
    return UMEP_VENDOR


def build_dsm(
    site: str,
    pixel_size: float,
    crop_bounds: tuple[float, float, float, float] | None = None,
    observer_height: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """Build a (DTM + building heights) raster from MorphoFavela site inputs.

    Returns (dsm_array, profile) where profile has rasterio-compatible
    transform + crs + shape so we can re-project sample points later.

    ``observer_height`` shifts the effective integration plane up by
    that many meters at street pixels. UMEP integrates SVF at the DSM
    surface; the equivalent of "observer at ground+h" is to lower
    every building by ``h``. Buildings shorter than ``h`` collapse to
    ground (the observer looks over them). Set to 1.5 to match MorphoFavela's
    pedestrian-height sampling; set to 0 for the legacy z=0 comparison.
    """
    dtm_path = PROJECT_ROOT / "data" / site / "dtm_extended_300m.tif"
    bld_path = PROJECT_ROOT / "data" / site / "buildings_extended_300m.gpkg"

    with rasterio.open(dtm_path) as src:
        if crop_bounds is None:
            left, bottom, right, top = (
                src.bounds.left,
                src.bounds.bottom,
                src.bounds.right,
                src.bounds.top,
            )
        else:
            left, bottom, right, top = crop_bounds

        width = int(np.ceil((right - left) / pixel_size))
        height = int(np.ceil((top - bottom) / pixel_size))
        dst_transform = from_origin(left, top, pixel_size, pixel_size)
        dst = np.empty((height, width), dtype=np.float32)
        nodata_sentinel = np.float32(-9999.0)

        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=src.crs,
            src_nodata=src.nodata,
            dst_nodata=nodata_sentinel,
            resampling=Resampling.bilinear,
        )
        invalid = dst == nodata_sentinel
        if invalid.all():
            raise RuntimeError("DTM reproject produced all-nodata output")
        median_elev = float(np.median(dst[~invalid]))
        dst = np.where(invalid, median_elev, dst).astype(np.float32)
        crs = src.crs

    bld = gpd.read_file(bld_path)
    if bld.crs != crs:
        bld = bld.to_crs(crs)
    height_col = "altura" if "altura" in bld.columns else "H_mean"
    bld = bld[bld[height_col].fillna(0) > 0]
    shapes = list(zip(bld.geometry, bld[height_col].astype(float)))

    bld_raster = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=dst_transform,
        fill=0.0,
        all_touched=True,
        dtype=np.float32,
    )
    is_building = bld_raster > 0  # cells where a building footprint hits this pixel
    if observer_height > 0:
        bld_effective = np.maximum(bld_raster - observer_height, 0.0).astype(np.float32)
    else:
        bld_effective = bld_raster
    dsm = dst + bld_effective
    profile = {
        "transform": dst_transform,
        "crs": crs,
        "shape": (height, width),
        "bounds": (left, bottom, right, top),
        "pixel_size": pixel_size,
        "is_building": is_building,
        "observer_height": float(observer_height),
    }
    logger.info(
        "DSM built for %s @ z=%.1f m: %d × %d cells at %.1f m, dsm range %.1f–%.1f m",
        site,
        observer_height,
        height,
        width,
        pixel_size,
        float(dsm.min()),
        float(dsm.max()),
    )
    return dsm, profile


def run_umep_svf(dsm: np.ndarray, pixel_size: float) -> np.ndarray:
    """Call UMEP's shadow-casting SVF processor on the DSM."""
    sys.path.insert(0, str(UMEP_VENDOR.parent))
    from umep_processing.functions import svf_functions

    vegdem = np.zeros_like(dsm)
    vegdem2 = np.zeros_like(dsm)
    scale = 1.0 / pixel_size

    class _NullFeedback:
        def setProgress(self, *a, **k):
            pass

        def isCanceled(self):
            return False

        def pushInfo(self, *a, **k):
            pass

    logger.info("Running UMEP svfForProcessing153 (this can take a few minutes)...")
    out = svf_functions.svfForProcessing153(
        dsm=dsm.astype(np.float32),
        vegdem=vegdem,
        vegdem2=vegdem2,
        scale=scale,
        usevegdem=0,
        pixel_resolution=pixel_size,
        wallScheme=0,
        demlayer=None,
        feedback=_NullFeedback(),
    )
    return np.asarray(out["svf"], dtype=np.float32)


def aggregate_umep_to_grid(
    svf_raster: np.ndarray,
    profile: dict,
    grid: gpd.GeoDataFrame,
    cell_size: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate UMEP SVF over each 10 m grid cell, masking out building pixels.

    For an apples-to-apples comparison with MorphoFavela (whose grid-cell SVF is the
    mean of passage-level samples — never on rooftops), we average UMEP SVF
    only over the non-building DSM pixels inside each cell's footprint.

    Returns (umep_svf_mean, n_pixels_used) per cell.
    """
    transform = profile["transform"]
    rows, cols = profile["shape"]
    is_building = profile["is_building"]
    inv = ~transform

    half = cell_size / 2
    out_svf = np.full(len(grid), np.nan, dtype=np.float32)
    out_n = np.zeros(len(grid), dtype=np.int32)
    for i, (cx, cy) in enumerate(grid[["centroid_x", "centroid_y"]].values):
        col_min, row_max = inv * (cx - half, cy - half)
        col_max, row_min = inv * (cx + half, cy + half)
        c0 = max(0, int(np.floor(min(col_min, col_max))))
        c1 = min(cols, int(np.ceil(max(col_min, col_max))))
        r0 = max(0, int(np.floor(min(row_min, row_max))))
        r1 = min(rows, int(np.ceil(max(row_min, row_max))))
        if r0 >= r1 or c0 >= c1:
            continue
        window = svf_raster[r0:r1, c0:c1]
        b_window = is_building[r0:r1, c0:c1]
        valid = ~b_window & np.isfinite(window)
        if valid.any():
            out_svf[i] = float(window[valid].mean())
            out_n[i] = int(valid.sum())
    return out_svf, out_n


def compare(
    grid: gpd.GeoDataFrame,
    umep_svf: np.ndarray,
    profile: dict,
    min_svf_count: int = 5,
    min_umep_pixels: int = 4,
) -> pd.DataFrame:
    """Build a per-cell comparison table."""
    umep_mean, umep_n = aggregate_umep_to_grid(umep_svf, profile, grid)
    grid = grid.copy()
    grid["umep_svf"] = umep_mean
    grid["umep_n_pixels"] = umep_n
    df = grid[
        ["zone_id", "centroid_x", "centroid_y", "svf", "svf_count", "umep_svf", "umep_n_pixels"]
    ].copy()
    df["delta"] = df["umep_svf"] - df["svf"]
    keep = (
        df["svf"].notna()
        & df["umep_svf"].notna()
        & (df["svf_count"] >= min_svf_count)
        & (df["svf"] > 0)
        & (df["umep_n_pixels"] >= min_umep_pixels)
    )
    df["used_in_fit"] = keep
    return df


def regress(df: pd.DataFrame) -> dict:
    sub = df[df["used_in_fit"]]
    y = sub["umep_svf"].values
    x = sub["svf"].values
    if len(x) < 3:
        return {"n": int(len(x)), "r2": np.nan, "slope": np.nan, "intercept": np.nan}
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {
        "n": int(len(x)),
        "r2": float(r2),
        "slope": float(slope),
        "intercept": float(intercept),
        "rmse": float(np.sqrt(np.mean((y - x) ** 2))),
        "mae": float(np.mean(np.abs(y - x))),
        "bias_umep_minus_ivf": float(np.mean(y - x)),
        "ivf_mean": float(x.mean()),
        "umep_mean": float(y.mean()),
    }


def plot(
    df: pd.DataFrame, stats: dict, site: str, out_path: Path, observer_height: float = 0.0
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub = df[df["used_in_fit"]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    ax = axes[0]
    ax.scatter(sub["svf"], sub["umep_svf"], s=4, alpha=0.3, color="#2A6EBB", linewidths=0)
    lim = (0.0, 1.0)
    ax.plot(lim, lim, "k--", lw=0.8, alpha=0.6, label="1:1")
    xs = np.linspace(0, 1, 50)
    ax.plot(xs, stats["slope"] * xs + stats["intercept"], "r-", lw=1.2, label="OLS fit")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("MorphoFavela Tregenza-145 SVF (z = 1.5 m)")
    ax.set_ylabel(f"UMEP shadow-cast SVF (z = {observer_height:g} m)")
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.set_title(
        f"n = {stats['n']:,}   r² = {stats['r2']:.3f}\n"
        f"slope = {stats['slope']:.2f}   bias = {stats['bias_umep_minus_ivf']:+.3f}",
        fontsize=9,
    )

    ax = axes[1]
    delta = sub["delta"].values
    ax.hist(delta, bins=40, color="#2A6EBB", edgecolor="white")
    ax.axvline(0, color="k", lw=0.6, ls="--")
    ax.axvline(
        stats["bias_umep_minus_ivf"],
        color="r",
        lw=1.0,
        label=f"bias = {stats['bias_umep_minus_ivf']:+.3f}",
    )
    ax.set_xlabel("UMEP − MorphoFavela (SVF units)")
    ax.set_ylabel("cells")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(f"RMSE = {stats['rmse']:.3f}   MAE = {stats['mae']:.3f}", fontsize=9)

    fig.suptitle(f"SVF cross-validation against UMEP — {site}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--site", required=True, help="Site name (e.g., 'vidigal').")
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=2.0,
        help="DSM pixel size in meters (default 2.0). 1.0 is more accurate but slower.",
    )
    parser.add_argument(
        "--min-svf-count",
        type=int,
        default=5,
        help="Drop cells with fewer than N MorphoFavela sample points (default 5).",
    )
    parser.add_argument(
        "--observer-height",
        type=float,
        default=1.5,
        help="Effective integration height in meters (default 1.5 to match "
        "MorphoFavela pedestrian-height sampling). 0 = legacy z=0 comparison.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output dir (default outputs/{site}/morphometrics/svf/umep_validation/).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ensure_umep()

    grid_path = (
        PROJECT_ROOT / "outputs" / args.site / "morphometrics" / "grid" / "grid_metrics.gpkg"
    )
    if not grid_path.exists():
        raise FileNotFoundError(f"grid_metrics.gpkg not found: {grid_path}")
    grid = gpd.read_file(grid_path)
    minx, miny, maxx, maxy = grid.total_bounds
    pad = 50.0
    crop = (minx - pad, miny - pad, maxx + pad, maxy + pad)

    dsm, profile = build_dsm(args.site, args.pixel_size, crop, observer_height=args.observer_height)
    umep_svf = run_umep_svf(dsm, args.pixel_size)

    df = compare(grid, umep_svf, profile, min_svf_count=args.min_svf_count)
    stats = regress(df)

    out_dir = (
        Path(args.out)
        if args.out
        else PROJECT_ROOT / "outputs" / args.site / "morphometrics" / "svf" / "umep_validation"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / "per_cell_comparison.csv", index=False)
    pd.DataFrame([stats]).to_csv(out_dir / "summary_stats.csv", index=False)
    plot(df, stats, args.site, out_dir / "scatter.png", observer_height=args.observer_height)

    logger.info("Saved %s", out_dir / "per_cell_comparison.csv")
    logger.info("Saved %s", out_dir / "summary_stats.csv")
    logger.info("Saved %s", out_dir / "scatter.png")
    print(
        f"site={args.site} z={args.observer_height:g}m px={args.pixel_size:g}m "
        f"n={stats['n']} r2={stats['r2']:.3f} "
        f"slope={stats['slope']:.2f} rmse={stats['rmse']:.3f} "
        f"bias={stats['bias_umep_minus_ivf']:+.3f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
