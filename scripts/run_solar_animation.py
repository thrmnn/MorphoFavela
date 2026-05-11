#!/usr/bin/env python3
"""Per-hour sunlit GIS layer for dataviz animations.

Produces a wide-format GeoPackage on the **same observer points** as
``outputs/{site}/morphometrics/svf/svf_streets_solar.gpkg`` — every existing
column (``svf``, ``solar_hours_winter``, ``solar_hours_annual``, …) is
preserved, plus one ``lit_T{HHMM}`` boolean column per timestep and a
``lit_hours_total`` convenience float. The dataviz team uses this file to
animate the winter-solstice shadow sweep frame-by-frame; because the
geometry is identical to the seasonal-envelope layer, overlays, joins, and
transitions between aggregated and animated views align exactly.

This is a *dataviz deliverable* — it does **not** replace the existing
``svf_streets_solar.gpkg`` or the aggregated ``solar_hours_winter`` field.
New artefacts land under ``outputs/{site}/dataviz/solar/`` and the existing
``morphometrics/svf/`` tree is untouched.

Usage::

    python scripts/run_solar_animation.py --site vidigal
    python scripts/run_solar_animation.py --site vidigal --interval 30 --n-jobs -1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.solar.animation import (  # noqa: E402
    build_animation_manifest,
    compute_sun_positions_with_times,
    sunlit_matrix_to_wide_gdf,
)
from src.solar.compute import compute_sunlit_matrix  # noqa: E402
from src.solar.sun import DEFAULT_LATITUDE, DEFAULT_LONGITUDE  # noqa: E402
from src.svf_v2.paths import resolve_paths  # noqa: E402
from src.svf_v2.scene import build_scene  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("run_solar_animation")

DEFAULT_DATE = "2026-06-21"  # Southern-Hemisphere winter solstice
DEFAULT_TIMEZONE = "America/Sao_Paulo"


def resolve_scene_mesh(site: str) -> pv.PolyData:
    """Reuse cached ``scene.stl`` if it exists, else rebuild from inputs.

    The cached STL is the same geometry used by ``run_street_solar.py`` and
    ``run_svf_v2.py``, so the per-hour layer's shading agrees with the
    seasonal envelope.
    """
    stl_path = PROJECT_ROOT / "outputs" / site / "morphometrics" / "svf" / "scene.stl"
    if stl_path.exists():
        logger.info("Loading cached scene mesh: %s", stl_path)
        return pv.read(str(stl_path))

    logger.info("scene.stl missing for %s — rebuilding from DTM + footprints", site)
    dtm_path, footprints_path, _roads_path = resolve_paths(site)
    scene_mesh, _terrain, _gdf = build_scene(
        dtm_path=dtm_path,
        footprints_path=footprints_path,
        area=site,
    )
    stl_path.parent.mkdir(parents=True, exist_ok=True)
    scene_mesh.save(str(stl_path))
    logger.info("  Wrote new scene.stl (%d cells)", scene_mesh.n_cells)
    return scene_mesh


def load_street_observers(streets_path: Path) -> tuple[gpd.GeoDataFrame, np.ndarray]:
    """Load svf_streets_solar.gpkg and return ``(base_gdf, (N, 3) observers)``.

    The returned ``base_gdf`` is the *full* source GeoDataFrame so the
    dataviz file can be a strict superset of the seasonal envelope. The
    observer xyz array uses ``z_observer`` (z_terrain + pedestrian height
    set during SVF sampling), falling back to ``z`` if necessary so the
    function is portable to older street files.
    """
    if not streets_path.exists():
        raise FileNotFoundError(f"Street solar GPKG not found: {streets_path}")
    gdf = gpd.read_file(streets_path).reset_index(drop=True)

    xs = np.array([p.x for p in gdf.geometry], dtype=np.float64)
    ys = np.array([p.y for p in gdf.geometry], dtype=np.float64)
    if "z_observer" in gdf.columns:
        zs = gdf["z_observer"].to_numpy(dtype=np.float64)
    elif "z" in gdf.columns:
        zs = gdf["z"].to_numpy(dtype=np.float64) + 1.5
        logger.warning("z_observer missing — falling back to z + 1.5 m")
    else:
        raise ValueError(
            f"{streets_path} has neither 'z_observer' nor 'z' — cannot place observers"
        )
    return gdf, np.column_stack([xs, ys, zs])


def render_preview(
    gdf: gpd.GeoDataFrame,
    frame_records: list[dict],
    output_path: Path,
    max_frames: int = 12,
) -> None:
    """Render a small preview grid showing the sunlit sweep across the day."""
    n_frames = min(len(frame_records), max_frames)
    if n_frames == 0:
        logger.warning("No frames to preview — skipping preview.png")
        return

    cols = min(4, n_frames)
    rows = int(np.ceil(n_frames / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.4), squeeze=False)

    xs = np.array([p.x for p in gdf.geometry])
    ys = np.array([p.y for p in gdf.geometry])
    extent_pad = 10.0
    xlim = (xs.min() - extent_pad, xs.max() + extent_pad)
    ylim = (ys.min() - extent_pad, ys.max() + extent_pad)

    indices = np.linspace(0, len(frame_records) - 1, n_frames).round().astype(int)

    for ax_idx, frame_idx in enumerate(indices):
        ax = axes[ax_idx // cols][ax_idx % cols]
        frame = frame_records[frame_idx]
        lit = gdf[frame["col"]].to_numpy(dtype=bool)
        ax.scatter(xs[~lit], ys[~lit], s=2.0, c="#1f1f1f", alpha=0.35, marker=".", linewidths=0)
        ax.scatter(xs[lit], ys[lit], s=2.0, c="#f4a300", alpha=0.95, marker=".", linewidths=0)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"{frame['time_local'][11:16]}  ({frame['n_sunlit']} lit)",
            fontsize=9,
        )

    for ax_idx in range(len(indices), rows * cols):
        axes[ax_idx // cols][ax_idx % cols].axis("off")

    fig.suptitle("Sunlit street observers — winter solstice sweep", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote preview PNG → %s", output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--site", required=True, help="Site name, e.g. 'vidigal'.")
    parser.add_argument(
        "--date",
        default=DEFAULT_DATE,
        help=f"ISO date (default: {DEFAULT_DATE}, Southern-Hemisphere winter solstice).",
    )
    parser.add_argument(
        "--interval", type=int, default=60, help="Frame interval (minutes; default 60)."
    )
    parser.add_argument(
        "--hour-start", type=int, default=7, help="Earliest local hour (default 7)."
    )
    parser.add_argument("--hour-end", type=int, default=17, help="Latest local hour (default 17).")
    parser.add_argument(
        "--latitude", type=float, default=DEFAULT_LATITUDE, help="Site latitude (deg)."
    )
    parser.add_argument(
        "--longitude", type=float, default=DEFAULT_LONGITUDE, help="Site longitude (deg)."
    )
    parser.add_argument(
        "--timezone",
        default=DEFAULT_TIMEZONE,
        help="IANA timezone used to label frames (default: America/Sao_Paulo).",
    )
    parser.add_argument(
        "--streets-source",
        type=Path,
        default=None,
        help=(
            "Override path to street observer GPKG "
            "(default: outputs/{site}/morphometrics/svf/svf_streets_solar.gpkg)."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel workers for ray-casting (default: all cores).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output dir (default: outputs/{site}/dataviz/solar/{date_label}).",
    )
    parser.add_argument(
        "--no-preview", action="store_true", help="Skip the preview PNG."
    )
    args = parser.parse_args()

    t0 = time.time()

    # ---- Output dir ---------------------------------------------------
    date_label = f"winter_solstice_{args.date}" if args.date == DEFAULT_DATE else args.date
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = (
            PROJECT_ROOT / "outputs" / args.site / "dataviz" / "solar" / date_label
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Site: %s", args.site)
    logger.info("Date: %s  (timezone: %s)", args.date, args.timezone)
    logger.info("Output dir: %s", output_dir)

    # ---- Street observers --------------------------------------------
    if args.streets_source is not None:
        streets_path = args.streets_source
    else:
        streets_path = (
            PROJECT_ROOT
            / "outputs"
            / args.site
            / "morphometrics"
            / "svf"
            / "svf_streets_solar.gpkg"
        )
    logger.info("Observer source: %s", streets_path)
    base_gdf, observers = load_street_observers(streets_path)
    target_crs = base_gdf.crs
    logger.info("Street observers: %d points (crs %s)", len(observers), target_crs)

    # ---- Scene mesh (same one used by run_street_solar) --------------
    scene_mesh = resolve_scene_mesh(args.site)

    # ---- Sun positions -----------------------------------------------
    sun_frames = compute_sun_positions_with_times(
        latitude=args.latitude,
        longitude=args.longitude,
        date=args.date,
        hour_start=args.hour_start,
        hour_end=args.hour_end,
        interval_minutes=args.interval,
        timezone=args.timezone,
    )
    logger.info(
        "Sun positions above horizon: %d (%d-min interval, %02d:00 → %02d:00 local)",
        len(sun_frames),
        args.interval,
        args.hour_start,
        args.hour_end,
    )
    if not sun_frames:
        logger.error("No sun positions above horizon for the requested window — aborting.")
        return 1

    sun_dirs = np.stack([f["direction"] for f in sun_frames], axis=0)

    # ---- Sunlit matrix ------------------------------------------------
    logger.info("Computing sunlit matrix (n_jobs=%s)…", args.n_jobs)
    t_ray = time.time()
    sunlit_matrix = compute_sunlit_matrix(
        observer_points=observers,
        sun_directions=sun_dirs,
        scene_mesh=scene_mesh,
        n_jobs=args.n_jobs,
    )
    logger.info(
        "Sunlit matrix done in %.1fs (%.1f%% sunlit overall)",
        time.time() - t_ray,
        100.0 * sunlit_matrix.sum() / max(sunlit_matrix.size, 1),
    )

    # ---- Append lit_T* columns to base_gdf ---------------------------
    gdf, frame_manifest = sunlit_matrix_to_wide_gdf(
        observers=observers,
        sunlit_matrix=sunlit_matrix,
        sun_frames=sun_frames,
        interval_minutes=args.interval,
        include_solar_hours=True,
        base_gdf=base_gdf,
        # Distinct name so we don't shadow `solar_hours_winter` from the source.
        solar_hours_col="lit_hours_total",
    )

    gpkg_path = output_dir / "sunlit_wide.gpkg"
    if gpkg_path.exists():
        gpkg_path.unlink()
    gdf.to_file(gpkg_path, driver="GPKG", layer="sunlit_wide")
    logger.info(
        "Wrote GPKG → %s  (%d points × %d frames, %d total columns)",
        gpkg_path,
        len(gdf),
        len(sun_frames),
        len(gdf.columns),
    )

    manifest = build_animation_manifest(
        site=args.site,
        date=args.date,
        timezone=args.timezone,
        crs=str(target_crs),
        hour_start=args.hour_start,
        hour_end=args.hour_end,
        frame_manifest=frame_manifest,
        observer_source=str(streets_path.relative_to(PROJECT_ROOT)),
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Wrote manifest → %s", manifest_path)

    # ---- Preview ------------------------------------------------------
    if not args.no_preview:
        render_preview(gdf, manifest["frames"], output_dir / "preview.png")

    logger.info("Done in %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
