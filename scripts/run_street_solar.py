#!/usr/bin/env python3
"""Street-level seasonal solar access.

For each street SVF point in ``svf_streets.gpkg`` this runner casts
direct-sun rays against ``scene.stl`` for the four reference dates
(winter solstice, summer solstice, March equinox, September equinox)
and writes the seasonal envelope back into a single
``svf_streets_solar.gpkg`` per site.

Compared to the legacy ``compute_solar_access.py`` (winter-only,
60-min step, integer-clamped), this runner:

* uses 30-min sampling by default (configurable via ``--interval``)
* keeps 4 reference dates, then exposes:
  - ``solar_hours_winter``    -- worst case for southern hemisphere
  - ``solar_hours_summer``    -- best case
  - ``solar_hours_equinox_mar``, ``solar_hours_equinox_sep``
  - ``solar_hours_annual``    -- mean of the 4 reference dates
                                  (Tregenza-style annual proxy)
  - ``seasonal_range``         -- summer minus winter
* keeps the legacy ``solar_hours`` column == winter solstice for
  back-compatibility with §5.3 of the technical report and the two
  paper-figure scripts (``fig06_terrain_aspect.py``,
  ``figS_terrain_aspect_spatial.py``).
* preserves the existing ``svf`` column from ``svf_streets.gpkg`` so
  downstream code (aspect analysis, paper figures) does not need to
  re-join.

Usage::

    python scripts/run_street_solar.py --site vidigal --n-jobs -1
    python scripts/run_street_solar.py --site vidigal --interval 15 --n-jobs -1
    python scripts/run_street_solar.py --all-sites --n-jobs -1

Notes
-----
* Only ``vidigal`` and ``maré`` currently have a pre-built
  ``scene.stl``.  For the other three sites the runner will rebuild
  the scene (terrain + extruded footprints) the first time it runs.
* The implementation reuses ``src/solar/seasonal.py`` so the
  irradiance/sunshine-ratio plumbing is identical to ``run_facade_solar``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyvista as pv

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.solar.seasonal import compute_seasonal_solar  # noqa: E402
from src.solar.sun import REFERENCE_DATES  # noqa: E402
from src.svf_v2.paths import resolve_paths  # noqa: E402
from src.svf_v2.scene import build_scene  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("run_street_solar")

ALL_SITES = ("vidigal", "rocinha", "complexo_do_alemao", "riodaspedras", "maré")

# Site latitude / longitude / mean elevation for the irradiance model.
# Latitude/longitude only matter for sun-position calculation, not
# ray-casting — they are within ~5 km of one another, so we use a
# single Rio default and only override elevation per-site.
SITE_ELEVATION_M: dict[str, float] = {
    "vidigal": 150.0,
    "rocinha": 90.0,
    "complexo_do_alemao": 60.0,
    "riodaspedras": 30.0,
    "maré": 5.0,
}


def _resolve_scene_mesh(site: str, output_svf_dir: Path) -> pv.PolyData:
    """Load ``scene.stl`` if it exists, otherwise rebuild it via ``build_scene``.

    The STL is the same object used during the SVF run, so re-using it
    here guarantees that solar shading and SVF were computed against
    identical geometry.
    """
    stl_path = output_svf_dir / "scene.stl"
    if stl_path.exists():
        logger.info("Loading cached scene STL: %s", stl_path)
        return pv.read(str(stl_path))

    logger.info("scene.stl missing for %s -- rebuilding from DTM + footprints", site)
    dtm_path, footprints_path, _ = resolve_paths(site)
    scene_mesh, _terrain, _gdf = build_scene(
        dtm_path=dtm_path,
        footprints_path=footprints_path,
        area=site,
    )
    output_svf_dir.mkdir(parents=True, exist_ok=True)
    scene_mesh.save(str(stl_path))
    logger.info("  Wrote new scene.stl (%d cells)", scene_mesh.n_cells)
    return scene_mesh


def _load_street_observers(streets_path: Path) -> tuple[gpd.GeoDataFrame, np.ndarray]:
    """Return (street_gdf, (N,3) observer xyz).

    Observer z is taken from ``z_observer`` (z_terrain + pedestrian
    height set during SVF sampling).  Falls back to ``z + 1.5`` when
    that column is absent so the runner is portable across older
    street files.
    """
    if not streets_path.exists():
        raise FileNotFoundError(f"Street SVF gpkg not found: {streets_path}")
    gdf = gpd.read_file(streets_path)
    if "z_observer" in gdf.columns:
        zs = gdf["z_observer"].to_numpy(dtype=np.float64)
    elif "z" in gdf.columns:
        zs = gdf["z"].to_numpy(dtype=np.float64) + 1.5
    else:
        raise ValueError("street SVF gpkg has neither 'z' nor 'z_observer'")
    xs = np.array([p.x for p in gdf.geometry], dtype=np.float64)
    ys = np.array([p.y for p in gdf.geometry], dtype=np.float64)
    return gdf, np.column_stack([xs, ys, zs])


def _seasonal_columns(seasonal: dict, dates: list[str]) -> dict[str, np.ndarray]:
    """Translate the keyed-by-date seasonal dict into the column schema."""
    rows: dict[str, np.ndarray] = {}

    label_for_date = {
        REFERENCE_DATES["winter_solstice"]: "winter",
        REFERENCE_DATES["summer_solstice"]: "summer",
        REFERENCE_DATES["equinox_march"]: "equinox_mar",
        REFERENCE_DATES["equinox_september"]: "equinox_sep",
    }

    for date in dates:
        label = label_for_date.get(date, date.replace("-", ""))
        entry = seasonal["dates"][date]
        rows[f"solar_hours_{label}"] = np.asarray(entry["solar_hours"], dtype=np.float64)
        if "irradiance_wh" in entry:
            rows[f"irradiance_{label}_wh"] = np.asarray(entry["irradiance_wh"], dtype=np.float64)

    summary = seasonal["summary"]
    rows["solar_hours_annual"] = np.asarray(summary["solar_hours_mean"], dtype=np.float64)
    rows["seasonal_range"] = np.asarray(summary["seasonal_range"], dtype=np.float64)
    rows["sunshine_ratio_mean"] = np.asarray(summary["sunshine_ratio_mean"], dtype=np.float64)
    if "irradiance_mean" in summary and summary["irradiance_mean"].any():
        rows["irradiance_annual_wh"] = np.asarray(summary["irradiance_mean"], dtype=np.float64)
    return rows


def run_site(
    site: str,
    interval_minutes: int,
    n_jobs: int,
    hour_start: int,
    hour_end: int,
) -> Path:
    output_svf_dir = PROJECT_ROOT / "outputs" / site / "morphometrics" / "svf"
    streets_path = output_svf_dir / "svf_streets.gpkg"
    out_path = output_svf_dir / "svf_streets_solar.gpkg"

    logger.info("=" * 60)
    logger.info("STREET SOLAR ENVELOPE -- site=%s", site)
    logger.info("=" * 60)

    scene_mesh = _resolve_scene_mesh(site, output_svf_dir)
    street_gdf, observers = _load_street_observers(streets_path)

    if "svf" in street_gdf.columns:
        svf_arr = street_gdf["svf"].to_numpy(dtype=np.float64)
    else:
        logger.warning("street gpkg lacks 'svf' -- diffuse irradiance will use SVF=1.0")
        svf_arr = np.ones(len(street_gdf), dtype=np.float64)

    elevation_m = SITE_ELEVATION_M.get(site, 100.0)
    dates = [
        REFERENCE_DATES["winter_solstice"],
        REFERENCE_DATES["summer_solstice"],
        REFERENCE_DATES["equinox_march"],
        REFERENCE_DATES["equinox_september"],
    ]

    logger.info(
        "Observers: %d  |  Dates: %d  |  Interval: %d min  |  hours [%d, %d]  |  elevation: %.0fm",
        len(observers),
        len(dates),
        interval_minutes,
        hour_start,
        hour_end,
        elevation_m,
    )

    t0 = time.time()
    seasonal = compute_seasonal_solar(
        observer_points=observers,
        scene_mesh=scene_mesh,
        svf_array=svf_arr,
        dates=dates,
        interval_minutes=interval_minutes,
        n_jobs=n_jobs,
        site_elevation_m=elevation_m,
    )
    logger.info("Seasonal solar finished in %.1fs", time.time() - t0)

    # Assemble the column schema and write back into a single gpkg.
    cols = _seasonal_columns(seasonal, dates)

    out_gdf = street_gdf.copy()
    for name, arr in cols.items():
        out_gdf[name] = arr

    # Legacy back-compat: the report and Figure 6 read 'solar_hours' as
    # winter-solstice hours.  Keep that column wired to the same numbers.
    out_gdf["solar_hours"] = cols["solar_hours_winter"]
    out_gdf["n_sun_positions"] = int(
        len(seasonal["dates"][REFERENCE_DATES["winter_solstice"]]["sun_positions"])
    )

    if out_path.exists():
        out_path.unlink()
    out_gdf.to_file(out_path, driver="GPKG")
    logger.info("Wrote %s (%d points, %d cols)", out_path, len(out_gdf), len(out_gdf.columns))

    # Quick console summary.
    for label in ("winter", "equinox_mar", "equinox_sep", "summer"):
        col = f"solar_hours_{label}"
        vals = out_gdf[col]
        logger.info(
            "  %-13s  mean=%5.2fh  median=%5.2fh  zero-share=%4.1f%%",
            label,
            vals.mean(),
            vals.median(),
            100.0 * (vals == 0).mean(),
        )
    logger.info(
        "  %-13s  mean=%5.2fh  median=%5.2fh",
        "annual",
        out_gdf["solar_hours_annual"].mean(),
        out_gdf["solar_hours_annual"].median(),
    )
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Street-level seasonal solar envelope.")
    parser.add_argument("--site", default=None, help=f"One site, e.g. 'vidigal'. Choose from {ALL_SITES}.")
    parser.add_argument("--all-sites", action="store_true", help="Run every site in ALL_SITES.")
    parser.add_argument("--interval", type=int, default=30, help="Sun-position timestep (min).")
    parser.add_argument("--hour-start", type=int, default=5, help="Earliest hour of day (5 catches summer dawn).")
    parser.add_argument("--hour-end", type=int, default=19, help="Latest hour of day (19 catches summer dusk).")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel workers for ray-casting (default: all cores).",
    )
    args = parser.parse_args()

    if args.all_sites:
        sites = list(ALL_SITES)
    elif args.site:
        sites = [args.site]
    else:
        parser.error("Pass --site SITE or --all-sites.")

    failures: list[tuple[str, str]] = []
    for site in sites:
        try:
            run_site(
                site=site,
                interval_minutes=args.interval,
                n_jobs=args.n_jobs,
                hour_start=args.hour_start,
                hour_end=args.hour_end,
            )
        except FileNotFoundError as exc:
            logger.warning("Skipping %s: %s", site, exc)
            failures.append((site, str(exc)))
        except Exception as exc:  # pragma: no cover - surfaced for the CLI
            logger.exception("Site %s failed: %s", site, exc)
            failures.append((site, repr(exc)))

    if failures:
        logger.warning("%d site(s) skipped or failed:", len(failures))
        for site, msg in failures:
            logger.warning("  %s -- %s", site, msg)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
