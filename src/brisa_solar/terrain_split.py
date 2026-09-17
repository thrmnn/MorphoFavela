"""WP-07 supplement — terrain- vs buildings-driven winter sun-hours lost, per
C′ study favela (the five in `src.brisa_solar.wp07_ledger.FAVELAS`).

Background: an old figure (`scripts/health/terrain_morphology_split.py` ->
`outputs/comparative/health/terrain_morphology_split/`) split street-level
winter sun-deficit into a terrain share and a morphology share, but it
predates the C′ reframe, used a different site set (including Jacarezinho),
had a caption overlapping its bars, and was never entered into the WP-07
ledger. This module recomputes the split from scratch on the CURRENT engine
(never copies a number from the old PNG) and adds the PI's specific request:
side-by-side spatial maps of terrain-only vs terrain-plus-buildings sun
hours, plus a difference panel.

Three surfaces per site (spec: reuse the machinery, don't write a second
engine):

  - open flat reference — horizon angle -inf everywhere, i.e. every daylight
    timestep is visible; no raster marching needed at all (`open_flat_hours`).
  - terrain only — `wp02_surface.build_surface`'s own `_ground.tif`
    companion output (the bare DTM on the surface grid, no buildings
    rasterised) — already produced by `wp04_sites.build_site_surface`, not
    rebuilt here.
  - real (terrain + buildings) — `wp04_sites.build_site_surface`'s
    `surface`, i.e. `max(dtm, building_top)`. This is also the surface the
    WP-04 sites run of record used, so it is the one this module's
    reproduction check compares against.

Attribution is NOT unique: (open_flat - real) splits into a terrain part and
a buildings part differently depending on which factor is subtracted first.
This module reports BOTH orderings:

  - terrain_first (headline, matches the old figure's convention):
    terrain_loss = open_flat - terrain_only; buildings_loss = terrain_only - real.
  - buildings_first (sensitivity only, in the manifest/summary, never the
    headline figure): needs a FOURTH, synthetic surface — every building
    placed on a single flat reference elevation, keeping its OWN real height
    (`build_flat_reference_surface`). First attempt used the real absolute
    `base`/`topo` attributes on a flat DTM: `wp02_surface.build_surface`
    computes building top from those columns, never from the input DTM
    array, so on Vidigal's real relief that reproduced each building's
    absolute elevation on the massif almost undisturbed — most buildings
    ended up towering unrealistically far above the single flat reference
    (mean buildings_first terrain_loss went NEGATIVE, a nonsense result,
    measured 2026-09-17 pilot). Fixed by overriding the footprints'
    `base_attr` to the flat elevation itself and dropping `top_attr` before
    rasterising, so `build_surface` falls back to `base + altura` — i.e. the
    building's real physical HEIGHT above a common flat plane, its absolute
    site elevation discarded. This is a genuine simplification (it drops
    each building's true elevation entirely) and is why this ordering is a
    disclosed sensitivity number, never the figure headline. Computed on a
    bounded random subsample of observers (`BLDG_SUBSAMPLE_N`), not the full
    grid — see the module docstring's budget note below.

Budget: the full grid (terrain + real, needed for the maps and the
reproduction check) is evaluated for every observer at every site — WP-04's
own per-site observer counts (measured 2026-09-17 against
`runs/wp04_sites_20260914T230606Z/*/summary.json`: 141k-846k ground cells per
site, ~1.72M total). The buildings-first sensitivity pass is the one lever
cut to stay in budget: it runs on a fixed-size random subsample per site
(`BLDG_SUBSAMPLE_N`) instead of the full grid, since it only ever feeds a
per-favela mean in the manifest, never a map.

Never promoted out of `runs/`: per-cell layers here are the same red line L1
as `cityhours.py` and `wp07_figures.py`'s f5/f5b maps — `release_class:
withheld`, `red_line: L1` on every map figure. The per-favela summary bars
carry no per-cell data and are classified on their own merits.

Run: python -m src.brisa_solar.terrain_split --mode pilot
     python -m src.brisa_solar.terrain_split --mode full
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
import rasterio.windows  # noqa: E402
import torch  # noqa: E402

from .constants import P1_SKY_PATCHES, load_params  # noqa: E402
from .wp02_horizon import default_device, patch_visibility  # noqa: E402
from .wp02_surface import build_surface, load_surface  # noqa: E402
from .wp04_sites import (  # noqa: E402
    CELL_M,
    MAX_DIST_M,
    OBS_HEIGHT_M,
    build_site_surface,
    direct_sun_hours,
    epw_meta,
    ground_grid_points,
    patch_azimuth_deg,
    quantiles,
    site_polygon,
    sun_positions,
)
from .wp07_figures import BOUNDARY_STROKE_PX, COLORS  # noqa: E402
from .wp07_ledger import FAVELAS, RUN_OF_RECORD, SITE_DIRS, SITES  # noqa: E402
from src.svf_v2.compute import generate_tregenza_patches  # noqa: E402

#: Headline ordering (spec: "Pick terrain-first to stay comparable with the
#: old figure, state that choice in words on the figure and in the manifest").
ATTRIBUTION_CHOICE = "terrain_first"

#: A raster below this fill fraction (within its own observers' bounding box,
#: not the wider site+halo raster) reads as "visually blank" — the same
#: defect the WP-07Z zoom windows hit this morning (2026-09-17) at a pixel
#: size finer than the sampling lattice (0.8-1.4% filled). Set comfortably
#: above that range.
MIN_MAP_FILL_FRACTION = 0.03

#: Reverse-ordering (buildings-first) sensitivity pass runs on a fixed-size
#: random subsample per site, not the full grid — see module docstring
#: "Budget". Only ever feeds a manifest mean, never a map.
BLDG_SUBSAMPLE_N = 20_000

DPI = 300


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_sha(repo_root: Path) -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=repo_root, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Open-flat reference (no raster needed — horizon angle -inf everywhere)
# ---------------------------------------------------------------------------

def open_flat_hours(date_str: str, meta: dict) -> float:
    """Sun-hours on an open flat horizon: every daylight timestep is visible
    by construction (horizon angle -inf everywhere), so this is exactly
    pvlib's own daylight-timestep count on the SAME 10-min sampling
    `wp04_sites.direct_sun_hours` uses for its fractional hours — no engine
    call needed for this reference surface at all."""
    fine = sun_positions(date_str, meta, "10min")
    return float((fine["apparent_elevation"].to_numpy() > 0.0).sum()) / 6.0


# ---------------------------------------------------------------------------
# Decomposition (both orderings — see module docstring)
# ---------------------------------------------------------------------------

def decompose_terrain_first(h_flat, hours_terrain, hours_real):
    """terrain assessed first (against the open-flat reference), buildings
    the residual against terrain-only. Headline ordering."""
    terrain_loss = h_flat - hours_terrain
    buildings_loss = hours_terrain - hours_real
    return terrain_loss, buildings_loss


def decompose_buildings_first(h_flat, hours_bldg, hours_real):
    """buildings assessed first (against the open-flat reference, using the
    flat-elevation buildings-only surface), terrain the residual against
    that. Sensitivity ordering only — never the figure headline."""
    buildings_loss = h_flat - hours_bldg
    terrain_loss = hours_bldg - hours_real
    return buildings_loss, terrain_loss


# ---------------------------------------------------------------------------
# Buildings-only flat-reference surface (for the reverse-ordering sensitivity)
# ---------------------------------------------------------------------------

def build_flat_reference_dtm(ground: np.ndarray, transform, crs, elevation_m: float, out_tif: Path) -> Path:
    """A synthetic constant-elevation DTM on the SAME grid as `ground` (the
    site's real terrain)."""
    profile = dict(
        driver="GTiff", height=ground.shape[0], width=ground.shape[1],
        count=1, dtype="float32", crs=crs, transform=transform, nodata=None,
    )
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(np.full(ground.shape, elevation_m, dtype="float32"), 1)
    return out_tif


def write_flat_footprints(fp_path: Path, elevation_m: float, out_path: Path, fp_params: dict) -> Path:
    """A copy of the site's footprints with `base_attr` overridden to a
    single flat `elevation_m` and `top_attr` dropped (NaN) — forces
    `wp02_surface.build_surface`'s `top = topo if finite&>base else
    base+altura` fallback to `elevation_m + altura`, i.e. each building's
    real physical HEIGHT above one common flat plane, with its true absolute
    site elevation discarded. `height_attr` (altura) is untouched."""
    import geopandas as gpd

    gdf = gpd.read_file(fp_path)
    gdf[fp_params["base_attr"]] = elevation_m
    gdf[fp_params["top_attr"]] = np.nan
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GPKG")
    return out_path


# ---------------------------------------------------------------------------
# Raster helpers — fill-fraction guard against the "rendered too fine, comes
# out blank" defect fixed 2026-09-17
# ---------------------------------------------------------------------------

def scatter_to_grid(rows: np.ndarray, cols: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Bounding-box raster of scattered (row, col) -> value, cropped to the
    observers' OWN extent (not the full site+700 m-halo raster) so a
    fill-fraction check measures coverage of the favela's own footprint, not
    how much of the context halo is legitimately empty."""
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    grid = np.full((r1 - r0 + 1, c1 - c0 + 1), np.nan, dtype="float32")
    grid[rows - r0, cols - c0] = values
    return grid, (r0, r1, c0, c1)


def fill_fraction(grid: np.ndarray) -> float:
    return float(np.isfinite(grid).sum()) / grid.size


def assert_not_blank(grid: np.ndarray, label: str, min_fraction: float = MIN_MAP_FILL_FRACTION) -> float:
    """Raise rather than render a raster that reads as visually blank
    (spec: "check the fill fraction of every raster you produce and assert
    it")."""
    frac = fill_fraction(grid)
    if frac < min_fraction:
        raise ValueError(
            f"{label}: raster fill fraction {frac:.4f} is below the "
            f"{min_fraction} floor — looks visually blank (the sampling-"
            "pitch defect fixed 2026-09-17); refusing to render it."
        )
    return frac


# ---------------------------------------------------------------------------
# Reproduction check against the WP-04 sites run of record
# ---------------------------------------------------------------------------

def reproduction_check(
    per_site_cells: dict[str, pd.DataFrame], repo_root: Path, labels: list[str],
    corr_min: float = 0.9999, max_abs_diff_max: float = 0.01,
) -> dict:
    """Correlate this module's real-surface (terrain+buildings) hours against
    the accepted WP-04 sites run of record, matched on (row, col) within each
    site — same discipline as `cityhours.reproduction_check`
    (tests/test_cityhours.py pins its defaults; this pins the same ones and
    is never loosened to force a pass)."""
    result = {"_utc": _utc_now(), "run_of_record": RUN_OF_RECORD["wp04"], "per_site": {}}
    passed = True
    for slug, df in per_site_cells.items():
        site_dir = SITE_DIRS[slug]
        record_path = repo_root / "runs" / RUN_OF_RECORD["wp04"] / site_dir / "ground.parquet"
        cols = ["row", "col"] + [f"hours_{l}" for l in labels]
        record = pd.read_parquet(record_path, columns=cols)
        merged = df[["row", "col"] + [f"hours_real_{l}" for l in labels]].merge(
            record, on=["row", "col"]
        )
        site_result = {"n_matched": int(len(merged)), "n_ours": int(len(df)), "n_record": int(len(record))}
        site_pass = True
        for label in labels:
            a = merged[f"hours_real_{label}"].to_numpy(dtype="float64")
            b = merged[f"hours_{label}"].to_numpy(dtype="float64")
            diff = np.abs(a - b)
            corr = float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 else float("nan")
            max_abs_diff = float(np.max(diff)) if len(diff) else float("nan")
            ok = bool(corr >= corr_min and max_abs_diff <= max_abs_diff_max)
            site_pass = site_pass and ok
            site_result[label] = {
                "corr": corr, "max_abs_diff": max_abs_diff,
                "corr_min": corr_min, "max_abs_diff_max": max_abs_diff_max, "pass": ok,
            }
        site_result["pass"] = site_pass
        passed = passed and site_pass
        result["per_site"][slug] = site_result
    result["pass"] = passed
    return result


# ---------------------------------------------------------------------------
# One site: engine passes + per-cell frames
# ---------------------------------------------------------------------------

def run_site(
    slug: str, data_root: Path, *, directions: np.ndarray, meta: dict, params: dict,
    device: str, tmp_dir: Path, bldg_subsample_n: int = BLDG_SUBSAMPLE_N, seed: int = 0,
):
    site_key = SITE_DIRS[slug]
    display = FAVELAS[slug]
    reference_days = {
        "winter_solstice": params["reference_days"]["winter_solstice"],
        "equinox": params["reference_days"]["equinox"],
    }
    duration_thresholds_h = params["reference_days"]["duration_thresholds_h"]
    patch_az_deg = patch_azimuth_deg(directions)
    favelas_path = data_root / "data/RJ/Favelas_Limit_2019.shp"

    t0 = time.perf_counter()
    surface, transform, crs, is_building, _building_id, ground, _dtm_path, fp_path = (
        build_site_surface(site_key, data_root, CELL_M, tmp_dir)
    )
    polygon, match_method, _matched = site_polygon(favelas_path, display, crs)
    obs = ground_grid_points(surface, transform, is_building, polygon)
    obs_xy = obs[["x", "y"]].to_numpy(dtype="float64")
    build_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    _vis_r, _onb_r, horizon_real = patch_visibility(
        surface, transform, obs_xy, directions=directions, is_building=is_building,
        obs_height_m=OBS_HEIGHT_M, max_dist_m=MAX_DIST_M, march_sampling="nearest",
        device=device, return_horizon=True,
    )
    real_engine_s = time.perf_counter() - t1

    t2 = time.perf_counter()
    _vis_t, _onb_t, horizon_terrain = patch_visibility(
        ground, transform, obs_xy, directions=directions, is_building=None,
        obs_height_m=OBS_HEIGHT_M, max_dist_m=MAX_DIST_M, march_sampling="nearest",
        device=device, return_horizon=True,
    )
    terrain_engine_s = time.perf_counter() - t2

    cells = obs[["row", "col", "x", "y"]].copy()
    cells["site"] = site_key
    h_flat: dict[str, float] = {}
    for label, date_str in reference_days.items():
        r_real = direct_sun_hours(horizon_real, patch_az_deg, date_str, meta, duration_thresholds_h)
        r_terr = direct_sun_hours(horizon_terrain, patch_az_deg, date_str, meta, duration_thresholds_h)
        cells[f"hours_real_{label}"] = r_real["hours_fractional"]
        cells[f"hours_terrain_{label}"] = r_terr["hours_fractional"]
        h_flat[label] = open_flat_hours(date_str, meta)
    del horizon_real, horizon_terrain

    # --- buildings-first sensitivity: flat-reference surface, subsampled ---
    n = len(obs)
    rng = np.random.default_rng(seed)
    sub_idx = rng.choice(n, size=min(bldg_subsample_n, n), replace=False)
    # A single fixed reference elevation: building HEIGHT (base+altura, base
    # overridden to this constant — see write_flat_footprints) is what
    # matters here, not the value itself, so 0.0 is as good as any other
    # constant and keeps this fully deterministic.
    flat_elev = 0.0

    flat_dtm_tif = tmp_dir / f"{site_key}_flat_ref_dtm.tif"
    build_flat_reference_dtm(ground, transform, crs, flat_elev, flat_dtm_tif)
    flat_fp_path = tmp_dir / f"{site_key}_flat_ref_footprints.gpkg"
    write_flat_footprints(fp_path, flat_elev, flat_fp_path, params["footprints"])
    bldg_out_stem = tmp_dir / f"{site_key}_bldg_only"
    t3 = time.perf_counter()
    surface_bldg_tif = build_surface(flat_dtm_tif, flat_fp_path, CELL_M, bldg_out_stem)
    is_building_bldg_tif = surface_bldg_tif.with_name(
        surface_bldg_tif.stem.replace("_surface", "_is_building") + ".tif"
    )
    surface_bldg, transform_bldg, _crs_bldg, is_building_bldg = load_surface(
        surface_bldg_tif, is_building_bldg_tif
    )
    build_bldg_s = time.perf_counter() - t3
    assert transform_bldg == transform, f"{slug}: flat-reference surface grid drifted from the real surface grid"

    obs_xy_sub = obs_xy[sub_idx]
    t4 = time.perf_counter()
    _vis_b, _onb_b, horizon_bldg = patch_visibility(
        surface_bldg, transform_bldg, obs_xy_sub, directions=directions, is_building=is_building_bldg,
        obs_height_m=OBS_HEIGHT_M, max_dist_m=MAX_DIST_M, march_sampling="nearest",
        device=device, return_horizon=True,
    )
    bldg_engine_s = time.perf_counter() - t4

    bldg_cells = obs.iloc[sub_idx][["row", "col", "x", "y"]].copy()
    bldg_cells["site"] = site_key
    for label, date_str in reference_days.items():
        r_bldg = direct_sun_hours(horizon_bldg, patch_az_deg, date_str, meta, duration_thresholds_h)
        bldg_cells[f"hours_bldg_{label}"] = r_bldg["hours_fractional"]
        bldg_cells[f"hours_real_{label}"] = cells.iloc[sub_idx][f"hours_real_{label}"].to_numpy()
    del horizon_bldg

    for f in (
        flat_dtm_tif, flat_fp_path, surface_bldg_tif, is_building_bldg_tif,
        surface_bldg_tif.with_name(surface_bldg_tif.stem.replace("_surface", "_building_id") + ".tif"),
        surface_bldg_tif.with_name(surface_bldg_tif.stem.replace("_surface", "_ground") + ".tif"),
        bldg_out_stem.with_name(bldg_out_stem.stem + "_meta.json"),
    ):
        try:
            Path(f).unlink(missing_ok=True)
        except Exception:
            pass

    report = {
        "site_key": site_key, "display_name": display, "match_method": match_method,
        "n_ground": int(n), "n_bldg_subsample": int(len(sub_idx)),
        "flat_reference_elevation_m": flat_elev,
        "flat_reference_elevation_method": (
            "buildings-only surface: footprints' base_attr overridden to a fixed "
            f"{flat_elev} m and top_attr dropped, so build_surface falls back to "
            "base + altura — each building's REAL physical height above one common "
            "flat plane, its true absolute site elevation discarded entirely. A "
            "genuine simplification (not a fabricated height — altura is read data), "
            "which is why this ordering stays a disclosed sensitivity number, never "
            "the figure headline. See write_flat_footprints."
        ),
        "h_flat": h_flat,
        "timing_s": {
            "build_s": build_s, "real_engine_s": real_engine_s, "terrain_engine_s": terrain_engine_s,
            "build_bldg_s": build_bldg_s, "bldg_engine_s": bldg_engine_s,
        },
    }
    return cells, bldg_cells, report, polygon, transform


# ---------------------------------------------------------------------------
# Per-favela aggregation (both orderings)
# ---------------------------------------------------------------------------

def aggregate_site_summary(
    slug: str, cells: pd.DataFrame, bldg_cells: pd.DataFrame, h_flat: dict, reference_days: dict,
) -> dict:
    summary = {
        "_utc": _utc_now(), "site_key": SITE_DIRS[slug], "display_name": FAVELAS[slug],
        "n_ground": int(len(cells)), "n_bldg_subsample": int(len(bldg_cells)),
        "attribution_choice_headline": ATTRIBUTION_CHOICE,
    }
    for label in reference_days:
        hf = h_flat[label]
        hours_real = cells[f"hours_real_{label}"].to_numpy()
        hours_terrain = cells[f"hours_terrain_{label}"].to_numpy()
        terrain_loss, buildings_loss = decompose_terrain_first(hf, hours_terrain, hours_real)
        total_loss = hf - hours_real
        mean_total = float(np.mean(total_loss))
        mean_terrain_tf = float(np.mean(terrain_loss))
        mean_buildings_tf = float(np.mean(buildings_loss))

        hours_bldg_sub = bldg_cells[f"hours_bldg_{label}"].to_numpy()
        hours_real_sub = bldg_cells[f"hours_real_{label}"].to_numpy()
        buildings_loss_bf, terrain_loss_bf = decompose_buildings_first(hf, hours_bldg_sub, hours_real_sub)
        mean_total_sub = float(np.mean(hf - hours_real_sub))
        mean_buildings_bf = float(np.mean(buildings_loss_bf))
        mean_terrain_bf = float(np.mean(terrain_loss_bf))

        summary[label] = {
            "h_flat": hf,
            "hours_real": quantiles(hours_real),
            "hours_terrain": quantiles(hours_terrain),
            "hours_bldg_subsample": quantiles(hours_bldg_sub),
            "terrain_first": {
                "terrain_loss_h_mean": mean_terrain_tf,
                "buildings_loss_h_mean": mean_buildings_tf,
                "total_loss_h_mean": mean_total,
                "terrain_share": mean_terrain_tf / mean_total if mean_total > 0 else float("nan"),
                "buildings_share": mean_buildings_tf / mean_total if mean_total > 0 else float("nan"),
            },
            "buildings_first_sensitivity": {
                "n_subsample": int(len(bldg_cells)),
                "buildings_loss_h_mean": mean_buildings_bf,
                "terrain_loss_h_mean": mean_terrain_bf,
                "total_loss_h_mean_subsample": mean_total_sub,
                "buildings_share": mean_buildings_bf / mean_total_sub if mean_total_sub > 0 else float("nan"),
                "terrain_share": mean_terrain_bf / mean_total_sub if mean_total_sub > 0 else float("nan"),
            },
        }
    return summary


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _save_figure(fig, fig_id: str, out_dir: Path) -> tuple[str, str]:
    svg_path = out_dir / f"{fig_id}.svg"
    png_path = out_dir / f"{fig_id}.png"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return svg_path.name, png_path.name


def render_split_bars(summaries: dict[str, dict], label: str, out_dir: Path) -> dict:
    """Horizontal stacked bars, terrain-first split, fixed site order (never
    ranked by value — same convention wp07_figures.FIGURE_SITE_ORDER uses).
    Percentage labels sit INSIDE each bar segment, site names OUTSIDE past
    the bar end with a wide margin, so nothing overlaps the bars — the exact
    defect the old figure had. Restricted to the sites actually computed in
    this run (a pilot on one site must not try to plot the other four)."""
    order = [s for s in FAVELAS if s in summaries]
    terrain_h = [summaries[s][label]["terrain_first"]["terrain_loss_h_mean"] for s in order]
    bldg_h = [summaries[s][label]["terrain_first"]["buildings_loss_h_mean"] for s in order]
    total_h = [t + b for t, b in zip(terrain_h, bldg_h)]

    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.barh(y, terrain_h, color="#8c6d46", label="terrain")
    ax.barh(y, bldg_h, left=terrain_h, color="#b5651d", label="buildings")
    for i, (th, bh, tot) in enumerate(zip(terrain_h, bldg_h, total_h)):
        if tot > 0 and th / tot > 0.12:
            ax.text(th / 2, i, f"{th / tot * 100:.0f}%", va="center", ha="center",
                    color="white", fontsize=8)
        if tot > 0 and bh / tot > 0.12:
            ax.text(th + bh / 2, i, f"{bh / tot * 100:.0f}%", va="center", ha="center",
                    color="white", fontsize=8)
        ax.text(tot + max(total_h) * 0.03, i, FAVELAS[order[i]], va="center", fontsize=8.5, color="#333")
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlabel(f"Sun-hours lost vs open-flat terrain, {label.replace('_', ' ')} (h, mean per site)")
    ax.set_title("Terrain- vs buildings-driven sun-hours lost — C′ study favelas", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.margins(x=0.32)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.5, -0.02,
        f"Attribution ordering: terrain-first ({ATTRIBUTION_CHOICE}) — terrain assessed against the open-flat "
        "reference first, buildings the residual against terrain-only. The reverse (buildings-first) ordering "
        "gives different numbers (interaction between slope shading and building shading is not additive) and "
        "is reported in this run's manifest.json, not shown here.",
        ha="center", va="top", fontsize=6.8, color="#555", wrap=True,
    )
    fig.tight_layout()
    svg_name, png_name = _save_figure(fig, "t1_terrain_buildings_split", out_dir)
    return {
        "id": "t1_terrain_buildings_split", "status": "produced",
        "svg_path": svg_name, "png_path": png_name,
        "release_class": "publishable-candidate",
        "attribution_choice_headline": ATTRIBUTION_CHOICE,
        "reference_day": label,
    }


def render_site_map(slug: str, cells: pd.DataFrame, transform, polygon, label: str, out_dir: Path) -> dict:
    """3 panels, one favela: terrain-only sun hours, terrain+buildings sun
    hours (shared colour scale), and their difference — the buildings'
    contribution, legible as a map. release_class withheld / red_line L1:
    per-cell spatial layer (spec + red_line L1, same as cityhours/wp07_figures f5).

    The favela boundary is overlaid on every panel at `BOUNDARY_STROKE_PX`
    (`wp07_figures.py`), the SAME output-pixel-derived stroke convention the
    citywide/zoom map family uses — never a second, independently-invented
    linewidth (a hardcoded point value renders to a different pixel width at
    every dpi, which is exactly the defect that convention was introduced to
    fix, 2026-09-17 PI review)."""
    rows = cells["row"].to_numpy()
    cols = cells["col"].to_numpy()
    grid_terrain, bbox = scatter_to_grid(rows, cols, cells[f"hours_terrain_{label}"].to_numpy())
    grid_real, _ = scatter_to_grid(rows, cols, cells[f"hours_real_{label}"].to_numpy())
    grid_diff = grid_terrain - grid_real

    fill_terrain = assert_not_blank(grid_terrain, f"{slug}/{label}/terrain")
    fill_real = assert_not_blank(grid_real, f"{slug}/{label}/real")
    fill_diff = fill_fraction(grid_diff)

    vmin = float(np.nanmin([grid_terrain, grid_real]))
    vmax = float(np.nanmax([grid_terrain, grid_real]))
    dmax = float(np.nanmax(grid_diff)) if np.isfinite(grid_diff).any() else 0.0

    r0, r1, c0, c1 = bbox
    window = rasterio.windows.Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1)
    windowed_transform = rasterio.windows.transform(window, transform)
    xmin, ymin, xmax, ymax = rasterio.transform.array_bounds(
        grid_terrain.shape[0], grid_terrain.shape[1], windowed_transform
    )
    boundary = gpd.GeoSeries([polygon])

    # dpi set at figure CREATION (not just at savefig) so this stroke calc,
    # which reads fig.dpi, sees the dpi this figure actually renders at.
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.0), dpi=DPI)
    boundary_lw = BOUNDARY_STROKE_PX / fig.dpi * 72.0
    boundary_color = COLORS.get(slug, "black")
    panels = (
        (axes[0], grid_terrain, "YlOrBr_r", vmin, vmax, "terrain only"),
        (axes[1], grid_real, "YlOrBr_r", vmin, vmax, "terrain + buildings"),
        (axes[2], grid_diff, "OrRd", 0.0, max(dmax, 1e-6), "difference (buildings' contribution)"),
    )
    im1 = im2 = None
    for i, (ax, grid, cmap, pvmin, pvmax, title) in enumerate(panels):
        im = ax.imshow(grid, extent=(xmin, xmax, ymin, ymax), origin="upper",
                        cmap=cmap, aspect="equal", interpolation="nearest", vmin=pvmin, vmax=pvmax)
        boundary.boundary.plot(ax=ax, color=boundary_color, linewidth=boundary_lw)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if i == 1:
            im1 = im
        elif i == 2:
            im2 = im
    cbar01 = fig.colorbar(im1, ax=axes[:2].tolist(), fraction=0.035, pad=0.02, shrink=0.85)
    cbar01.set_label(f"{label.replace('_', ' ')} sun hours (h)", fontsize=7.5)
    cbar01.ax.tick_params(labelsize=6.5)
    cbar2 = fig.colorbar(im2, ax=axes[2], fraction=0.06, pad=0.02, shrink=0.85)
    cbar2.set_label("hours lost to buildings (h)", fontsize=7.5)
    cbar2.ax.tick_params(labelsize=6.5)
    fig.suptitle(f"{FAVELAS[slug]} — {label.replace('_', ' ')} sun hours: terrain vs terrain+buildings", fontsize=10.5, y=1.02)

    fig_id = f"t2_map_{slug}"
    svg_name, png_name = _save_figure(fig, fig_id, out_dir)
    return {
        "id": fig_id, "status": "produced",
        "svg_path": svg_name, "png_path": png_name,
        "release_class": "withheld", "red_line": "L1",
        "reference_day": label,
        "raster": {
            "bbox_row_col": [r0, r1, c0, c1], "shape": list(grid_terrain.shape),
            "pitch_m": "the WP-04 run-of-record's own cell_m (read from its manifest.json, never typed)",
            "fill_fraction": {"terrain": fill_terrain, "real": fill_real, "diff": fill_diff},
            "min_fill_fraction_floor": MIN_MAP_FILL_FRACTION,
            "boundary_stroke_px": BOUNDARY_STROKE_PX,
            "boundary_stroke_source": "src.brisa_solar.wp07_figures.BOUNDARY_STROKE_PX (reused, never a second convention)",
        },
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _common_inputs(data_root: Path, params: dict):
    epw_path = data_root / params["weather"]["primary_epw"]
    return epw_path


def _wp04_run_of_record_manifest(repo_root: Path) -> dict:
    path = repo_root / "runs" / RUN_OF_RECORD["wp04"] / "manifest.json"
    return json.loads(path.read_text())


def run_all(
    slugs: list[str], run_dir: Path, data_root: Path, *,
    bldg_subsample_n: int = BLDG_SUBSAMPLE_N,
) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = run_dir / "_tmp"
    params = load_params()
    reference_days = {
        "winter_solstice": params["reference_days"]["winter_solstice"],
        "equinox": params["reference_days"]["equinox"],
    }
    epw_path = _common_inputs(data_root, params)
    directions, _weights = generate_tregenza_patches()
    meta = epw_meta(epw_path)
    device = default_device()

    wp04_manifest = _wp04_run_of_record_manifest(data_root)
    if float(wp04_manifest["cell_m"]) != float(CELL_M):
        raise ValueError(
            f"WP-04 run-of-record cell_m ({wp04_manifest['cell_m']}) != "
            f"wp04_sites.CELL_M ({CELL_M}) this module imports — refusing to run at a drifted pitch"
        )

    per_site_cells: dict[str, pd.DataFrame] = {}
    per_site_bldg: dict[str, pd.DataFrame] = {}
    site_reports = []
    summaries = {}
    figures = {}
    out_figs = run_dir / "figures"
    out_figs.mkdir(exist_ok=True)

    for slug in slugs:
        print(f"=== terrain_split: {slug} ===", flush=True)
        cells, bldg_cells, report, polygon, transform = run_site(
            slug, data_root, directions=directions, meta=meta, params=params,
            device=device, tmp_dir=tmp_dir, bldg_subsample_n=bldg_subsample_n,
        )
        site_dir = run_dir / slug
        site_dir.mkdir(exist_ok=True)
        cells.to_parquet(site_dir / "cells.parquet", index=False)
        bldg_cells.to_parquet(site_dir / "bldg_subsample.parquet", index=False)
        per_site_cells[slug] = cells
        per_site_bldg[slug] = bldg_cells
        site_reports.append(report)

        summary = aggregate_site_summary(slug, cells, bldg_cells, report["h_flat"], reference_days)
        (site_dir / "summary.json").write_text(json.dumps(summary, indent=1))
        summaries[slug] = summary

        map_fig = render_site_map(slug, cells, transform, polygon, "winter_solstice", out_figs)
        map_fig["source_cells_parquet"] = str((site_dir / "cells.parquet").relative_to(run_dir))
        figures[map_fig["id"]] = map_fig
        print(json.dumps({k: v for k, v in report.items() if k != "timing_s"}, default=str), flush=True)

    bars_fig = render_split_bars(summaries, "winter_solstice", out_figs)
    bars_fig["sources"] = [f"{slug}/summary.json" for slug in slugs]
    figures[bars_fig["id"]] = bars_fig

    repro = reproduction_check(per_site_cells, data_root, list(reference_days))
    (run_dir / "reproduction_check.json").write_text(json.dumps(repro, indent=1))

    manifest = {
        "_utc": _utc_now(),
        "git_sha": _git_sha(data_root),
        "sky": {"patches": int(P1_SKY_PATCHES)},
        "cell_m": CELL_M,
        "cell_m_source": f"runs/{RUN_OF_RECORD['wp04']}/manifest.json#/cell_m",
        "max_dist_m": MAX_DIST_M,
        "obs_height_m": OBS_HEIGHT_M,
        "march_sampling": "nearest",
        "device": device,
        "torch_version": torch.__version__,
        "reference_days": reference_days,
        "attribution_choice_headline": ATTRIBUTION_CHOICE,
        "attribution_note": (
            "terrain_first (headline, matches the old figure's convention): terrain_loss = "
            "open_flat - terrain_only; buildings_loss = terrain_only - real. buildings_first "
            "(sensitivity, never the figure): buildings_loss = open_flat - buildings_only "
            "(flat-elevation surface); terrain_loss = buildings_only - real. Both orderings sum "
            "to the same total (open_flat - real) but split it differently because terrain "
            "shading and building shading are not additive; buildings_first uses a fixed-size "
            "random subsample per site, not the full grid — see BLDG_SUBSAMPLE_N."
        ),
        "bldg_subsample_n": bldg_subsample_n,
        "floor_provenance": params["reference_days"]["floor_provenance"],
        "wp04_run_of_record": RUN_OF_RECORD["wp04"],
        "sites": slugs,
        "site_reports": site_reports,
        "reproduction_check_pass": repro["pass"],
        "reproduction_check_file": "reproduction_check.json",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))

    figure_manifest = {
        "_utc": _utc_now(),
        "git_sha": _git_sha(data_root),
        "run_manifest": "manifest.json",
        "figures": figures,
    }
    (run_dir / "figure_manifest.json").write_text(json.dumps(figure_manifest, indent=1, ensure_ascii=False))

    return {"run_dir": str(run_dir), "manifest": manifest, "figure_manifest": figure_manifest, "reproduction_check": repro}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"], required=True)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--data-root", default="/home/theo/SCL/SCR/MorphoFavela")
    ap.add_argument("--sites", default=None, help="comma-separated slugs from wp07_ledger.SITES, default one (pilot) or all (full)")
    ap.add_argument("--bldg-subsample-n", type=int, default=BLDG_SUBSAMPLE_N)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    run_dir = Path(args.run_dir) if args.run_dir else data_root / "runs" / (
        f"terrain_split_{args.mode}_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )

    if args.sites:
        slugs = args.sites.split(",")
    elif args.mode == "pilot":
        slugs = [SITES[0]]
    else:
        slugs = list(SITES)

    result = run_all(slugs, run_dir, data_root, bldg_subsample_n=args.bldg_subsample_n)
    print(json.dumps({
        "run_dir": result["run_dir"],
        "reproduction_check_pass": result["reproduction_check"]["pass"],
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
