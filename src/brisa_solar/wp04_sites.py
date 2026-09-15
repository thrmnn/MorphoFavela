"""WP-04: five study sites decomposed into ground / street / façade evaluation surfaces.

Spec: docs/wp04_sites_spec.md. Runs the SAME engine (wp02_horizon.patch_visibility),
cell size (1 m) and sky (wp02_sky.CumulativeSky, the one-resolution 145-patch
Tregenza sky) as the WP-05 citywide run, at three surfaces per site:

  1. ground  — every 1 m cell inside the site's Favelas_Limit_2019 polygon,
     excluding building cells
  2. street  — svf_v2.sampling.sample_street_points on the site's road network
  3. façade  — svf_v2.sampling.sample_facade_points, per storey

Direct-sun hours on the two reference days come from the marched horizon angle
(wp02_horizon.patch_visibility(..., return_horizon=True)) compared against pvlib
sun positions for the Galeão EPW site — never from binary patch visibility, which
bins a whole ~11-degree patch and cannot resolve a specific clock hour (spec).

Run: python -m src.brisa_solar.wp04_sites
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timedelta, timezone as dt_timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pvlib
import rasterio
import torch
from rasterio.features import rasterize

from . import wp02_sky
from .constants import P1_SKY_PATCHES, REPO_ROOT, load_params
from .wp02_horizon import hemisphere_mask, patch_visibility
from .wp02_surface import build_surface, load_surface
from .wp05_pilot import pack_visibility  # tile/checkpoint pattern reuse (spec: reuse WP-05)
import src.config as _svf_config
from src.svf_v2 import sampling as svf_sampling
from src.svf_v2.compute import generate_tregenza_patches
from src.svf_v2.paths import resolve_boundary, resolve_paths

CELL_M = 1.0
OBS_HEIGHT_M = 1.5
MAX_DIST_M = 500.0

#: (site_key for svf_v2.paths / data/<key>/, display name for Favelas_Limit_2019 match)
SITES: list[tuple[str, str]] = [
    ("vidigal", "Vidigal"),
    ("rocinha", "Rocinha"),
    ("complexo_do_alemao", "Complexo do Alemão"),
    ("maré", "Maré"),
    ("riodaspedras", "Rio das Pedras"),
]


def _utc_now() -> str:
    return datetime.now(dt_timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def pq_row_count(path: Path) -> int:
    """Row count from Parquet footer metadata — no data read, safe for a 10M-row file."""
    import pyarrow.parquet as pq

    return pq.ParquetFile(path).metadata.num_rows


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Favela polygon matching (reproduces WP-05's citywide distribution.json logic)
# ---------------------------------------------------------------------------

def match_favela_polygon(favelas_gdf: gpd.GeoDataFrame, display_name: str) -> tuple[gpd.GeoDataFrame, str]:
    """complexo-exact first, else a unique nome match.

    Reproduces the priority runs/wp05_full_20260914T215419Z/distribution.json used
    (verified 2026-09-15 against its match_method + matched_polygons per site): an
    "Isolada" favela (Vidigal, Rocinha) has no complexo group under its own name and
    falls through to nome; a grouped favela (Alemão, Maré, Rio das Pedras) has one
    and is matched whole rather than by its single largest-named parcel.
    """
    key = display_name.strip().lower()
    m = favelas_gdf[favelas_gdf["complexo"].str.strip().str.lower() == key]
    if len(m) > 0:
        return m, "complexo_exact"
    m = favelas_gdf[favelas_gdf["nome"].str.strip().str.lower() == key]
    if len(m) == 1:
        return m, "nome_exact_unique"
    raise ValueError(
        f"no unique Favelas_Limit_2019 match for {display_name!r}: "
        f"{len(m)} nome match(es), 0 complexo match(es)"
    )


def resolve_native_paths(site_key: str, data_root: Path) -> tuple[Path, Path, Path]:
    """svf_v2.paths.resolve_paths()/get_area_data_dir() hardcode DATA_DIR relative
    to src/config.py's OWN file location — this worktree, which carries no data/
    (data/ lives only in the main checkout, gitignored). Temporarily repoint that
    module global so the existing resolver (mandatory read: svf_v2/paths.py) can
    be reused as-is rather than re-implementing its AREA_FILES registry here.
    """
    original = _svf_config.DATA_DIR
    _svf_config.DATA_DIR = data_root / "data"
    try:
        return resolve_paths(site_key)
    finally:
        _svf_config.DATA_DIR = original


def resolve_native_boundary(site_key: str, data_root: Path) -> Path | None:
    """Same DATA_DIR repointing as resolve_native_paths, for resolve_boundary()."""
    original = _svf_config.DATA_DIR
    _svf_config.DATA_DIR = data_root / "data"
    try:
        return resolve_boundary(site_key)
    finally:
        _svf_config.DATA_DIR = original


def site_polygon(favelas_path: Path, display_name: str, target_crs):
    from shapely.ops import unary_union

    gdf = gpd.read_file(favelas_path)
    matched, method = match_favela_polygon(gdf, display_name)
    if matched.crs is not None and str(matched.crs) != str(target_crs):
        matched = matched.to_crs(target_crs)
    poly = unary_union(matched.geometry.values)
    matched_polygons = matched[["objectid", "cod_favela", "nome", "complexo"]].to_dict("records")
    return poly, method, matched_polygons


# ---------------------------------------------------------------------------
# Surface + ground grid
# ---------------------------------------------------------------------------

def build_site_surface(site_key: str, data_root: Path, cell_m: float, tmp_dir: Path):
    dtm_path = data_root / f"data/{site_key}/dtm_extended_700m.tif"
    fp_path = data_root / f"data/{site_key}/buildings_extended_700m.gpkg"
    out_stem = tmp_dir / f"{site_key}_{cell_m:g}m"
    surface_tif = build_surface(dtm_path, fp_path, cell_m, out_stem)
    is_building_tif = surface_tif.with_name(surface_tif.stem.replace("_surface", "_is_building") + ".tif")
    surface, transform, crs, is_building = load_surface(surface_tif, is_building_tif)
    return surface, transform, crs, is_building, dtm_path, fp_path


def ground_grid_points(surface: np.ndarray, transform, is_building: np.ndarray, polygon) -> pd.DataFrame:
    """Every 1 m surface cell whose centre falls inside `polygon`, excluding building cells."""
    poly_mask = rasterize(
        [(polygon, 1)], out_shape=surface.shape, transform=transform, fill=0, dtype="uint8"
    ).astype(bool)
    keep = poly_mask & ~is_building
    rows, cols = np.where(keep)
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    return pd.DataFrame({
        "row": rows, "col": cols,
        "x": np.asarray(xs), "y": np.asarray(ys),
        "z": surface[rows, cols].astype("float64"),
    })


# ---------------------------------------------------------------------------
# Façade SVF / irradiation — the general hemisphere_mask cosine-weighted formula
# ---------------------------------------------------------------------------

def facade_svf_irradiation(sky: "wp02_sky.CumulativeSky", directions, weights, visible, normals):
    """Façade SVF = Σ visible·mask·w·(d·n) / Σ w·d_z — the SAME denominator as ground
    SVF (cosine_sum), not a per-normal renormalisation: that is what makes an
    unobstructed vertical wall read 0.5 rather than 1 (measured 2026-09-15: cardinal
    normals give 0.495-0.498 against this denominator, ~1.0 against a masked one).

    Façade kWh/m² reprojects the already horizontal-plane-cosine-weighted
    `sky.patch_total_kwh` back to a per-steradian radiance proxy
    (`patch_total_kwh / cos(zenith)`, the same identity `sky.svf` uses in reverse)
    and re-integrates it against the façade's own cosine(d·n) — wp02_sky.py is not
    modified (spec's engine-extension list is wp02_horizon.py only); this is a
    downstream, WP-04-local use of its already-published patch energies.
    """
    mask = hemisphere_mask(directions, normals)          # (n, P)
    dotprod = normals @ directions.T                      # (n, P)
    cw = weights * directions[:, 2]
    denom = float(cw.sum())
    front_visible = visible & mask
    svf = (front_visible * weights[None, :] * dotprod).sum(axis=1) / denom
    cos_zenith = np.clip(directions[:, 2], 1e-6, None)
    radiance_proxy = sky.patch_total_kwh / cos_zenith
    irr = (front_visible * dotprod * radiance_proxy[None, :]).sum(axis=1)
    return svf, irr


# ---------------------------------------------------------------------------
# Direct-sun hours from marched horizon angles (never from binary patch visibility)
# ---------------------------------------------------------------------------

def patch_azimuth_deg(directions: np.ndarray) -> np.ndarray:
    """Azimuth (deg, clockwise from north) of each patch's horizontal direction —
    the same x=east/y=north convention wp02_sky.build uses for the EPW sun vector."""
    return (np.degrees(np.arctan2(directions[:, 0], directions[:, 1])) + 360.0) % 360.0


def epw_meta(epw_path: Path) -> dict:
    _df, meta = pvlib.iotools.read_epw(str(epw_path))
    return meta


def sun_positions(date_str: str, meta: dict, freq: str) -> pd.DataFrame:
    tz = dt_timezone(timedelta(hours=float(meta["TZ"])))
    day0 = pd.Timestamp(date_str, tz=tz)
    periods = {"1h": 24, "10min": 24 * 6}[freq]
    times = pd.date_range(day0, periods=periods, freq=freq)
    return pvlib.solarposition.get_solarposition(
        times, meta["latitude"], meta["longitude"], altitude=meta["altitude"],
    )


def _sun_vector(solpos: pd.DataFrame) -> np.ndarray:
    alt = np.radians(solpos["apparent_elevation"].to_numpy())
    az = np.radians(solpos["azimuth"].to_numpy())
    return np.column_stack([np.cos(alt) * np.sin(az), np.cos(alt) * np.cos(az), np.sin(alt)])


def _nearest_patch_per_time(az_deg: np.ndarray, patch_az_deg: np.ndarray) -> np.ndarray:
    diff = np.abs((az_deg[:, None] - patch_az_deg[None, :] + 180.0) % 360.0 - 180.0)
    return np.argmin(diff, axis=1)


def direct_sun_hours(
    horizon_deg: np.ndarray,
    patch_az_deg: np.ndarray,
    date_str: str,
    meta: dict,
    duration_thresholds_h: list[int],
    normals: np.ndarray | None = None,
    chunk: int = 200_000,
) -> dict:
    """Per-observer direct-sun hours on one reference day.

    Sun visible at time t iff its altitude exceeds the horizon angle at the
    NEAREST sampled patch azimuth (12-deg-band quantisation from the 145-patch
    scheme's own azimuth sampling, not a second sky — spec). Also required:
    apparent_elevation > 0 (real daylight) — the finite-max_dist_m horizon march
    on flat ground reads a few tenths of a degree BELOW the true horizontal
    (atan2(-obs_height_m, max_dist_m)), a marching artifact of a finite domain
    and the observer's own eye height, not a real sightline past the horizon;
    gating on pvlib's own daylight flag is what makes the unobstructed-ground
    test's "direct-sun hours == daylight hours" hold exactly rather than by luck.

    `normals`, when given (façade points), additionally requires the sun's own
    direction vector to be on the front side of the point's normal — a wall
    cannot receive direct sun from behind itself regardless of horizon.
    """
    hourly = sun_positions(date_str, meta, "1h")
    fine = sun_positions(date_str, meta, "10min")

    hourly_alt = hourly["apparent_elevation"].to_numpy()
    hourly_idx = _nearest_patch_per_time(hourly["azimuth"].to_numpy(), patch_az_deg)
    hourly_daylight = hourly_alt > 0.0
    hourly_sun_vec = _sun_vector(hourly) if normals is not None else None

    fine_alt = fine["apparent_elevation"].to_numpy()
    fine_idx = _nearest_patch_per_time(fine["azimuth"].to_numpy(), patch_az_deg)
    fine_daylight = fine_alt > 0.0
    fine_sun_vec = _sun_vector(fine) if normals is not None else None

    n = horizon_deg.shape[0]
    hours_count = np.zeros(n, dtype=np.int16)
    hours_fractional = np.zeros(n, dtype=np.float32)
    thresholds = {k: np.zeros(n, dtype=bool) for k in duration_thresholds_h}

    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        hz = horizon_deg[start:end]

        h_at_hour = hz[:, hourly_idx].astype(np.float32)
        vis_hour = (hourly_alt[None, :].astype(np.float32) > h_at_hour) & hourly_daylight[None, :]
        h_at_fine = hz[:, fine_idx].astype(np.float32)
        vis_fine = (fine_alt[None, :].astype(np.float32) > h_at_fine) & fine_daylight[None, :]

        if normals is not None:
            nrm = normals[start:end]
            vis_hour = vis_hour & ((hourly_sun_vec @ nrm.T).T > 0.0)
            vis_fine = vis_fine & ((fine_sun_vec @ nrm.T).T > 0.0)

        hours_count[start:end] = vis_hour.sum(axis=1)
        frac = vis_fine.sum(axis=1).astype(np.float32) / 6.0
        hours_fractional[start:end] = frac
        for k in duration_thresholds_h:
            thresholds[k][start:end] = frac >= k

    result = {
        "hours_count": hours_count,
        "hours_fractional": hours_fractional,
        "daylight_hours_pvlib": int(hourly_daylight.sum()),
    }
    for k in duration_thresholds_h:
        result[f"ge_{k}h"] = thresholds[k]
    return result


# ---------------------------------------------------------------------------
# One evaluation call, shared by ground / street / façade
# ---------------------------------------------------------------------------

def evaluate_points(
    surface, transform, is_building, obs_xy, *, directions, weights, sky,
    obs_z=None, obs_height_m=OBS_HEIGHT_M, device=None, max_dist_m=MAX_DIST_M,
    normals=None,
):
    vis, on_building, horizon_deg = patch_visibility(
        surface, transform, obs_xy, directions=directions, is_building=is_building,
        obs_height_m=obs_height_m, obs_z=obs_z, max_dist_m=max_dist_m,
        march_sampling="nearest", device=device, return_horizon=True,
    )
    if normals is None:
        svf = sky.svf(vis.astype(float))
        irr = sky.irradiation(vis.astype(float))
    else:
        svf, irr = facade_svf_irradiation(sky, directions, weights, vis, normals)
    return vis, on_building, horizon_deg, svf, irr


def add_sun_hours_columns(
    df: pd.DataFrame, horizon_deg: np.ndarray, patch_az_deg: np.ndarray, meta: dict,
    reference_days: dict, duration_thresholds_h: list[int], normals: np.ndarray | None = None,
) -> pd.DataFrame:
    df = df.copy()
    for label, date_str in reference_days.items():
        r = direct_sun_hours(horizon_deg, patch_az_deg, date_str, meta, duration_thresholds_h, normals=normals)
        df[f"hours_{label}"] = r["hours_fractional"]
        df[f"hours_count_{label}"] = r["hours_count"]
        for k in duration_thresholds_h:
            df[f"ge_{k}h_{label}"] = r[f"ge_{k}h"]
    return df


def evaluate_and_write_parquet(
    out_path: Path, base_cols: dict, obs_xy: np.ndarray, *,
    surface, transform, is_building, directions, weights, sky, meta,
    reference_days: dict, duration_thresholds_h: list[int], patch_az_deg: np.ndarray,
    device: str, obs_z: np.ndarray | None = None, normals: np.ndarray | None = None,
    obs_height_m: float = OBS_HEIGHT_M, max_dist_m: float = MAX_DIST_M,
    chunk: int = 300_000,
) -> int:
    """Evaluate `obs_xy` in chunks and stream the result to a Parquet file.

    A façade point set can run into the millions (Maré: 37,199 buildings ->
    10.3M façade points) and evaluate_points' (n, P) horizon/visibility arrays
    plus the per-point sun-hours columns do not fit in RAM all at once for a
    site that size — 2026-09-15, OOM-killed (dmesg confirmed) building the
    whole DataFrame in memory before a single write. Chunking bounds peak
    memory to one chunk regardless of site size; a running ParquetWriter means
    the full result is never held as one Python object either.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    n = obs_xy.shape[0]
    writer = None
    tmp_path = out_path.with_suffix(".parquet.tmp")
    try:
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            xy = obs_xy[start:end]
            oz = obs_z[start:end] if obs_z is not None else None
            nrm = normals[start:end] if normals is not None else None

            vis, on_building, horizon_deg, svf, irr = evaluate_points(
                surface, transform, is_building, xy,
                directions=directions, weights=weights, sky=sky, device=device,
                obs_z=oz, obs_height_m=obs_height_m, max_dist_m=max_dist_m, normals=nrm,
            )

            cols = {}
            for k, v in base_cols.items():
                v = np.asarray(v)
                cols[k] = v[start:end] if v.shape[0] == n else v
            df = pd.DataFrame(cols)
            df["on_building"] = on_building
            df["svf"] = svf
            df["kwh_m2"] = irr
            df["visibility_packed"] = [row.tobytes() for row in pack_visibility(vis)]
            df = add_sun_hours_columns(
                df, horizon_deg, patch_az_deg, meta, reference_days, duration_thresholds_h, normals=nrm
            )

            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp_path, table.schema)
            writer.write_table(table)
            del vis, on_building, horizon_deg, svf, irr, df, table
    finally:
        if writer is not None:
            writer.close()
    if n == 0 and writer is None:
        pd.DataFrame({k: np.asarray(v) for k, v in base_cols.items()}).to_parquet(tmp_path, index=False)
    tmp_path.rename(out_path)
    return n


# ---------------------------------------------------------------------------
# Citywide percentile lookup (distribution.json)
# ---------------------------------------------------------------------------

def citywide_percentile(value: float, quantile_section: dict) -> float:
    q_levels = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    q_values = [quantile_section[f"p{q}"] for q in q_levels]
    return float(np.interp(value, q_values, q_levels))


def quantiles(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype="float64")
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"n": 0}
    qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    out = {"n": int(len(v)), "median": float(np.median(v))}
    for q in qs:
        out[f"p{q}"] = float(np.percentile(v, q))
    return out


# ---------------------------------------------------------------------------
# One site
# ---------------------------------------------------------------------------

def run_site(
    site_key: str, display_name: str, run_dir: Path, *, data_root: Path,
    directions, weights, sky, meta, params, device: str,
) -> dict:
    site_dir = run_dir / site_key
    site_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = run_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    reference_days = {
        "winter_solstice": params["reference_days"]["winter_solstice"],
        "equinox": params["reference_days"]["equinox"],
    }
    duration_thresholds_h = params["reference_days"]["duration_thresholds_h"]
    patch_az_deg = patch_azimuth_deg(directions)

    favelas_path = data_root / "data/RJ/Favelas_Limit_2019.shp"
    surface, transform, crs, is_building, dtm_path, fp_path = build_site_surface(
        site_key, data_root, CELL_M, tmp_dir
    )
    polygon, match_method, matched_polygons = site_polygon(favelas_path, display_name, crs)

    report = {
        "site_key": site_key, "display_name": display_name,
        "match_method": match_method, "matched_polygons": matched_polygons,
        "surface_shape": list(surface.shape),
    }

    common = dict(
        surface=surface, transform=transform, is_building=is_building,
        directions=directions, weights=weights, sky=sky, meta=meta,
        reference_days=reference_days, duration_thresholds_h=duration_thresholds_h,
        patch_az_deg=patch_az_deg, device=device,
    )

    # --- ground ---
    ground_path = site_dir / "ground.parquet"
    if ground_path.exists():
        report["n_ground"] = int(pq_row_count(ground_path))
    else:
        obs = ground_grid_points(surface, transform, is_building, polygon)
        obs_xy = obs[["x", "y"]].to_numpy(dtype="float64")
        base_cols = {"row": obs["row"].to_numpy(), "col": obs["col"].to_numpy(),
                     "x": obs_xy[:, 0], "y": obs_xy[:, 1], "z": obs["z"].to_numpy(),
                     "site": np.full(len(obs), site_key)}
        report["n_ground"] = evaluate_and_write_parquet(ground_path, base_cols, obs_xy, **common)

    # --- street ---
    street_path = site_dir / "street.parquet"
    if street_path.exists():
        report["n_street"] = int(pq_row_count(street_path))
    else:
        native_dtm, native_fp, native_roads = resolve_native_paths(site_key, data_root)
        footprints_gdf = gpd.read_file(native_fp)
        boundary_path = resolve_native_boundary(site_key, data_root)
        # Clips road geometries to the community boundary before sampling — matches
        # what produced the accepted CPU cross-reference (Rio das Pedras: 16,905
        # points, test 5); without it, road segments outside the favela but inside
        # the shapefile add ~34% extra points that were never part of that set.
        boundary_gdf = gpd.read_file(boundary_path) if boundary_path is not None else None
        street_pts = svf_sampling.sample_street_points(
            native_roads, native_dtm, footprints_gdf=footprints_gdf, boundary_gdf=boundary_gdf,
        )
        obs_xy = np.column_stack([street_pts.geometry.x.to_numpy(), street_pts.geometry.y.to_numpy()])
        base_cols = {
            "x": obs_xy[:, 0], "y": obs_xy[:, 1],
            "z": street_pts["z"].to_numpy(), "z_observer": street_pts["z_observer"].to_numpy(),
            "street_id": street_pts["street_id"].to_numpy(),
            "distance_along": street_pts["distance_along"].to_numpy(),
            "was_offset": (street_pts["was_offset"].to_numpy() if "was_offset" in street_pts
                           else np.zeros(len(street_pts), dtype=bool)),
            "site": np.full(len(street_pts), site_key),
        }
        report["n_street"] = evaluate_and_write_parquet(street_path, base_cols, obs_xy, **common)

    # --- façade ---
    facade_path = site_dir / "facade.parquet"
    if facade_path.exists():
        report["n_facade"] = int(pq_row_count(facade_path))
    else:
        native_dtm, native_fp, _native_roads = resolve_native_paths(site_key, data_root)
        footprints_gdf = gpd.read_file(native_fp)
        facade_pts = svf_sampling.sample_facade_points(footprints_gdf, native_dtm)
        obs_xy = np.column_stack([facade_pts["x"].to_numpy(), facade_pts["y"].to_numpy()])
        obs_z = facade_pts["z"].to_numpy(dtype="float64")
        normals = np.column_stack([
            facade_pts["normal_x"].to_numpy(), facade_pts["normal_y"].to_numpy(), facade_pts["normal_z"].to_numpy(),
        ])
        base_cols = {
            "x": obs_xy[:, 0], "y": obs_xy[:, 1], "z": obs_z,
            "normal_x": normals[:, 0], "normal_y": normals[:, 1], "normal_z": normals[:, 2],
            "building_id": facade_pts["building_id"].to_numpy(),
            "facade_azimuth": facade_pts["facade_azimuth"].to_numpy(),
            "height_above_ground": facade_pts["height_above_ground"].to_numpy(),
            "site": np.full(len(facade_pts), site_key),
        }
        report["n_facade"] = evaluate_and_write_parquet(
            facade_path, base_cols, obs_xy, obs_z=obs_z, normals=normals, **common
        )

    return report


def write_site_summary(
    site_key: str, display_name: str, run_dir: Path, citywide_distribution: dict | None
) -> dict:
    site_dir = run_dir / site_key
    # Only the numeric columns needed for quantiles — never the visibility_packed
    # bytes column, which for a multi-million-row façade table is the difference
    # between a light summary pass and re-materialising the whole checkpoint.
    summary_cols = (
        ["svf", "kwh_m2"] + [f"hours_{l}" for l in ("winter_solstice", "equinox")]
        + [f"ge_{k}h_{l}" for k in (1, 2, 3, 4) for l in ("winter_solstice", "equinox")]
    )
    ground_df = pd.read_parquet(site_dir / "ground.parquet", columns=summary_cols)
    street_df = pd.read_parquet(site_dir / "street.parquet", columns=summary_cols)
    facade_df = pd.read_parquet(site_dir / "facade.parquet", columns=summary_cols)

    status = "PROVISIONAL — wp05_run_design untapped" if citywide_distribution else "PROVISIONAL — no citywide distribution on disk"

    def surface_summary(df: pd.DataFrame) -> dict:
        out = {
            "n": int(len(df)),
            "svf": quantiles(df["svf"].to_numpy()),
            "kwh_m2": quantiles(df["kwh_m2"].to_numpy()),
        }
        for label in ("winter_solstice", "equinox"):
            col = f"hours_{label}"
            if col in df.columns:
                out[f"direct_sun_hours_{label}"] = quantiles(df[col].to_numpy())
        return out

    summary = {
        "_utc": datetime.now(dt_timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": status,
        "site_key": site_key, "display_name": display_name,
        "ground": surface_summary(ground_df),
        "street": surface_summary(street_df),
        "facade": surface_summary(facade_df),
    }

    ground_thresholds = {}
    for k in (1, 2, 3, 4):
        for label in ("winter_solstice", "equinox"):
            col = f"ge_{k}h_{label}"
            if col in ground_df.columns:
                ground_thresholds[f"share_ge_{k}h_{label}"] = float(ground_df[col].mean())
    summary["ground"]["threshold_shares"] = ground_thresholds

    if citywide_distribution is not None:
        cw = citywide_distribution["citywide"]
        summary["citywide_percentile"] = {
            "status": citywide_distribution.get("status"),
            "ground_svf_median_percentile": citywide_percentile(summary["ground"]["svf"]["median"], cw["svf"]),
            "ground_kwh_m2_median_percentile": citywide_percentile(summary["ground"]["kwh_m2"]["median"], cw["kwh_m2"]),
        }

    (site_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/home/theo/SCL/SCR/MorphoFavela")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument(
        "--distribution-json",
        default="/home/theo/SCL/SCR/MorphoFavela/runs/wp05_full_20260914T215419Z/distribution.json",
    )
    ap.add_argument("--sites", default=None, help="comma-separated site keys, default all 5")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    params = load_params()

    run_dir = (
        Path(args.run_dir)
        if args.run_dir
        else data_root / "runs" / ("wp04_sites_" + datetime.now(dt_timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    epw_path = data_root / params["weather"]["primary_epw"]
    directions, weights = generate_tregenza_patches()
    sky = wp02_sky.build(epw_path)
    meta = epw_meta(epw_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    citywide_distribution = None
    dist_path = Path(args.distribution_json)
    if dist_path.exists():
        citywide_distribution = json.loads(dist_path.read_text())

    wanted = set(args.sites.split(",")) if args.sites else None
    site_reports = []
    site_summaries = {}
    for site_key, display_name in SITES:
        if wanted is not None and site_key not in wanted:
            continue
        print(f"=== {site_key} ({display_name}) ===", flush=True)
        report = run_site(
            site_key, display_name, run_dir, data_root=data_root,
            directions=directions, weights=weights, sky=sky, meta=meta, params=params, device=device,
        )
        summary = write_site_summary(site_key, display_name, run_dir, citywide_distribution)
        site_reports.append(report)
        site_summaries[site_key] = summary
        print(json.dumps(report, default=str), flush=True)

    sky_section = json.dumps(params["sky"], sort_keys=True)
    manifest = {
        "_utc": datetime.now(dt_timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sky": {"patches": int(P1_SKY_PATCHES)},
        "cell_m": CELL_M,
        "sky_model": "epw_weighted",
        "obs_height_m": OBS_HEIGHT_M,
        "max_dist_m": MAX_DIST_M,
        "march_sampling": "nearest",
        "reference_days": params["reference_days"],
        "sun_azimuth_quantisation": "nearest of the 145-patch scheme's own azimuth sampling (~12-deg bands at the lowest altitude ring); not a second sky",
        "floor_provenance": params["reference_days"]["floor_provenance"],
        "device": device,
        "torch_version": torch.__version__,
        "git_sha": _git_sha(),
        "params_sky_section_sha256": hashlib.sha256(sky_section.encode()).hexdigest()[:16],
        "sites": [{"site_key": k, "display_name": v} for k, v in SITES if wanted is None or k in wanted],
        "site_reports": site_reports,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))

    print(json.dumps({"run_dir": str(run_dir), "sites": list(site_summaries.keys())}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
