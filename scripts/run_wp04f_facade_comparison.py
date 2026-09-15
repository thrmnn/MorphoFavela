"""WP-04F deliverable 2: four-variant measured façade comparison on Vidigal and
Rio das Pedras (spec docs/wp04f_facade_spec.md §2).

Variants, all storeys, per site:
  a. baseline            — as wp04_sites.py runs today (inset=default, no exclusion)
  b. own-building exclusion — same inset, own-building cells excluded from the march
  c. inset 1.5 m          — default inset replaced by 1.5 m, no exclusion
  d. b + c                — own-building exclusion AND inset 1.5 m

For each variant: share exactly zero, share < 0.01, median, p25/p75, and the
median by storey bin (height_above_ground: 0-3, 3-6, 6-9, >9 m).

Also carries the two physics-check numbers (unobstructed vertical facade,
opposite-parallel-wall closed form) per engine variant (exclusion off / on),
reusing the exact same synthetic construction as tests/test_wp04f_facade.py.

Writes runs/wp04f_facade_<UTC>/comparison.json and comparison.md.

Run: python3 scripts/run_wp04f_facade_comparison.py [--data-root PATH] [--run-dir PATH]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import torch
from affine import Affine

from src.brisa_solar import wp02_sky, wp04_sites
from src.brisa_solar.constants import load_params
from src.brisa_solar.wp02_horizon import patch_visibility
from src.svf_v2 import sampling as svf_sampling
from src.svf_v2.compute import generate_tregenza_patches

SITES = [("vidigal", "Vidigal"), ("riodaspedras", "Rio das Pedras")]
STOREY_BINS = [(0.0, 3.0), (3.0, 6.0), (6.0, 9.0), (9.0, float("inf"))]
DEFAULT_INSET = 0.1  # src.svf_v2.sampling.sample_facade_points's own default


def storey_medians(svf: np.ndarray, height_above_ground: np.ndarray) -> dict:
    out = {}
    for lo, hi in STOREY_BINS:
        label = f"{lo:g}-{hi:g}" if np.isfinite(hi) else f">{lo:g}"
        mask = (height_above_ground >= lo) & (height_above_ground < hi)
        out[label] = float(np.median(svf[mask])) if mask.any() else None
    return out


def variant_stats(svf: np.ndarray, height_above_ground: np.ndarray) -> dict:
    n = svf.shape[0]
    return {
        "n": int(n),
        "share_exact_zero": float((svf == 0.0).sum() / n) if n else None,
        "share_lt_0_01": float((svf < 0.01).sum() / n) if n else None,
        "median": float(np.median(svf)) if n else None,
        "p25": float(np.percentile(svf, 25)) if n else None,
        "p75": float(np.percentile(svf, 75)) if n else None,
        "median_by_storey": storey_medians(svf, height_above_ground) if n else {},
    }


def run_site_variants(
    site_key: str, data_root: Path, tmp_dir: Path, *, directions, weights, sky, device: str,
) -> dict:
    surface, transform, crs, is_building, building_id_raster, ground_surface, dtm_path, fp_path = (
        wp04_sites.build_site_surface(site_key, data_root, wp04_sites.CELL_M, tmp_dir)
    )
    native_dtm, native_fp, _native_roads = wp04_sites.resolve_native_paths(site_key, data_root)
    footprints_gdf = gpd.read_file(native_fp).reset_index(drop=True)
    ext_fp_gdf = gpd.read_file(fp_path)
    native_to_raster_id = wp04_sites.map_native_building_ids_to_raster(footprints_gdf, ext_fp_gdf)

    variants = {}
    sampled_cache: dict[float, "gpd.GeoDataFrame"] = {}

    def sampled(inset: float):
        if inset not in sampled_cache:
            sampled_cache[inset] = svf_sampling.sample_facade_points(footprints_gdf, native_dtm, inset=inset)
        return sampled_cache[inset]

    plan = [
        ("a_baseline", DEFAULT_INSET, False),
        ("b_own_building_exclusion", DEFAULT_INSET, True),
        ("c_inset_1_5m", 1.5, False),
        ("d_exclusion_and_inset_1_5m", 1.5, True),
    ]

    for label, inset, exclusion in plan:
        facade_pts = sampled(inset)
        obs_xy = np.column_stack([facade_pts["x"].to_numpy(), facade_pts["y"].to_numpy()])
        obs_z = facade_pts["z"].to_numpy(dtype="float64")
        normals = np.column_stack([
            facade_pts["normal_x"].to_numpy(), facade_pts["normal_y"].to_numpy(), facade_pts["normal_z"].to_numpy(),
        ])
        height_above_ground = facade_pts["height_above_ground"].to_numpy(dtype="float64")

        kwargs = {}
        if exclusion:
            obs_building = native_to_raster_id[facade_pts["building_id"].to_numpy()]
            kwargs = dict(
                obs_building=obs_building,
                building_id_raster=building_id_raster,
                ground_surface=ground_surface,
            )

        vis, _on_building = patch_visibility(
            surface, transform, obs_xy, directions=directions, is_building=is_building,
            obs_z=obs_z, max_dist_m=wp04_sites.MAX_DIST_M, march_sampling="nearest",
            device=device, **kwargs,
        )
        svf, _irr = wp04_sites.facade_svf_irradiation(sky, directions, weights, vis, normals)
        variants[label] = variant_stats(svf, height_above_ground)
        print(f"    {site_key}/{label}: n={variants[label]['n']} "
              f"zero={variants[label]['share_exact_zero']:.3f} "
              f"median={variants[label]['median']:.4f}", flush=True)

    return variants


def physics_checks() -> dict:
    """Reuses the exact synthetic constructions from tests/test_wp04f_facade.py."""
    directions, weights = generate_tregenza_patches()
    epw = load_params()["weather"]["primary_epw"]
    epw_path = Path("/home/theo/SCL/SCR/MorphoFavela") / epw
    sky = wp02_sky.build(epw_path) if epw_path.exists() else None

    out = {}

    # Unobstructed vertical facade, per engine variant (exclusion is geometrically
    # a no-op here: no building_id anywhere matches a nonzero obs_building).
    n_patches = len(directions)
    vis_unobstructed = np.ones((4, n_patches), dtype=bool)
    normals4 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    if sky is not None:
        svf_u, _irr_u = wp04_sites.facade_svf_irradiation(sky, directions, weights, vis_unobstructed, normals4)
        out["unobstructed_vertical_facade_svf"] = {
            "baseline": [float(x) for x in svf_u],
            "own_building_exclusion": [float(x) for x in svf_u],  # geometrically identical: no obstruction at all
            "expected": 0.5, "tolerance": 0.04,
        }

    # Opposite-parallel-wall closed form, both engine variants.
    cell, size = 0.5, 200
    origin_x, origin_y = -size * cell / 2, size * cell / 2
    transform = Affine(cell, 0, origin_x, 0, -cell, origin_y)
    ys = origin_y - (np.arange(size) + 0.5) * cell
    xs = origin_x + (np.arange(size) + 0.5) * cell
    Y, _X = np.meshgrid(ys, xs, indexing="ij")
    D, H = 20.0, 15.0
    surface = np.where(Y >= D, H, 0.0).astype("float64")
    building_id = np.zeros((size, size), dtype=np.int64)
    ground = np.zeros((size, size), dtype="float64")
    own_rows = (Y[:, 0] >= -3.0) & (Y[:, 0] < -1.0)
    own_cols = (xs >= -1.0) & (xs < 1.0)
    surface[np.ix_(own_rows, own_cols)] = 5.0
    building_id[np.ix_(own_rows, own_cols)] = 1

    obs = np.array([[0.0, -0.5]])
    obs_z = np.array([2.0])
    obs_building = np.array([1])
    max_dist_m = 40.0

    closed_horizon_deg = wp04_sites.opposite_wall_horizon_deg(directions, H, D, float(obs_z[0]))
    dx, dy = directions[:, 0], directions[:, 1]
    horiz_norm = np.hypot(dx, dy)
    hy = np.divide(dy, horiz_norm, out=np.zeros_like(dy), where=horiz_norm > 1e-9)
    facing_wall = hy >= (D / max_dist_m)

    wall_result = {}
    for label, kwargs in [
        ("baseline", {}),
        ("own_building_exclusion", dict(obs_building=obs_building, building_id_raster=building_id, ground_surface=ground)),
    ]:
        _vis, _ob, horizon_deg = patch_visibility(
            surface, transform, obs, directions=directions, obs_z=obs_z,
            max_dist_m=max_dist_m, step_m=cell, device="cpu", return_horizon=True, **kwargs,
        )
        engine_deg = horizon_deg[0].astype("float64")
        band = np.abs(engine_deg[facing_wall] - closed_horizon_deg[facing_wall])
        wall_result[label] = {
            "max_abs_deviation_deg": float(band.max()),
            "mean_abs_deviation_deg": float(band.mean()),
            "n_patches_compared": int(facing_wall.sum()),
        }
    out["opposite_wall_closed_form"] = {
        "D_m": D, "H_m": H, "observer_height_m": float(obs_z[0]),
        **wall_result,
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/home/theo/SCL/SCR/MorphoFavela")
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    run_dir = (
        Path(args.run_dir) if args.run_dir
        else data_root / "runs" / ("wp04f_facade_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = run_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    directions, weights = generate_tregenza_patches()
    epw_path = data_root / load_params()["weather"]["primary_epw"]
    sky = wp02_sky.build(epw_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    result = {
        "_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "device": device,
        "torch_version": torch.__version__,
        "default_inset_m": DEFAULT_INSET,
        "sites": {},
        "physics_checks": physics_checks(),
    }

    for site_key, display_name in SITES:
        print(f"=== {site_key} ({display_name}) ===", flush=True)
        result["sites"][site_key] = run_site_variants(
            site_key, data_root, tmp_dir, directions=directions, weights=weights, sky=sky, device=device,
        )

    (run_dir / "comparison.json").write_text(json.dumps(result, indent=1))

    md = [f"# WP-04F façade comparison — {result['_utc']}", ""]
    md.append(f"device={device}, torch={torch.__version__}, default_inset={DEFAULT_INSET} m")
    md.append("")
    for site_key, _display in SITES:
        md.append(f"## {site_key}")
        md.append("")
        md.append("| variant | n | zero share | <0.01 share | median | p25 | p75 | 0-3m | 3-6m | 6-9m | >9m |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for label, s in result["sites"][site_key].items():
            sb = s["median_by_storey"]

            def fmt(v):
                return f"{v:.4f}" if v is not None else "n/a"

            md.append(
                f"| {label} | {s['n']} | {s['share_exact_zero']:.4f} | {s['share_lt_0_01']:.4f} | "
                f"{fmt(s['median'])} | {fmt(s['p25'])} | {fmt(s['p75'])} | "
                f"{fmt(sb.get('0-3'))} | {fmt(sb.get('3-6'))} | {fmt(sb.get('6-9'))} | {fmt(sb.get('>9'))} |"
            )
        md.append("")

    md.append("## Physics checks")
    md.append("")
    md.append(json.dumps(result["physics_checks"], indent=1))
    (run_dir / "comparison.md").write_text("\n".join(md))

    print(json.dumps({"run_dir": str(run_dir)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
