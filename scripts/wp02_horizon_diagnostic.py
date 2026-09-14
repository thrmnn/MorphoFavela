"""WP-02 diagnostic: locate the +0.044 median SVF bias vs the CPU reference.

Follow-up to the horizon-engine acceptance run (docs/wp02_horizon_engine_spec.md
§Acceptance #6): the reference (src/svf_v2/scene.py) is exact polygons + a
ray-mesh, built from the SAME dtm_extended_300m.tif + buildings_extended_300m
altura/base as this engine — not a laser scan — so any bias has to be a
raster/march method difference, located here rather than attributed to
unmodelled geometry.

1. Bins the baseline (1 m, bilinear, step=1m) residual by distance-to-nearest-
   footprint-edge and by reference-SVF quartile.
2. Re-measures three single-change engine variants against the FULL 16,905-
   point reference: (A) nearest-cell march sampling, (B) step_m=0.5 with
   bilinear, (C) all_touched=True footprint rasterisation.
3. Reports r / median|delta| / p95|delta| / max|delta| for each, and for the
   best combination of the changes that individually helped.

Run: python3 -m scripts.wp02_horizon_diagnostic
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import torch

from src.brisa_solar import wp02_sky
from src.brisa_solar.constants import load_params
from src.brisa_solar.wp02_horizon import patch_visibility, svf_unweighted, write_run_manifest
from src.brisa_solar.wp02_surface import build_surface, load_surface
from src.svf_v2.compute import generate_tregenza_patches

MAIN_CHECKOUT = Path("/home/theo/SCL/SCR/MorphoFavela")
WORKTREE_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "wp02_horizon_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

_RDP_DTM = MAIN_CHECKOUT / "data/riodaspedras/dtm_extended_300m.tif"
_RDP_FOOTPRINTS = MAIN_CHECKOUT / "data/riodaspedras/buildings_extended_300m.gpkg"
_REF_PATH = MAIN_CHECKOUT / "outputs/riodaspedras/svf_v2/svf_streets.gpkg"


def _compare(measured: np.ndarray, reference: np.ndarray) -> dict:
    delta = np.abs(measured - reference)
    r = float(np.corrcoef(measured, reference)[0, 1])
    return {
        "r": r,
        "median_abs_delta": float(np.median(delta)),
        "p95_abs_delta": float(np.percentile(delta, 95)),
        "max_abs_delta": float(np.max(delta)),
        "n": int(len(reference)),
    }


def _run_variant(artifacts_dir: Path, name: str, *, cell_m=1.0, step_m=1.0,
                  march_sampling="bilinear", all_touched=False,
                  directions, obs, obs_height_m, device):
    out_stem = artifacts_dir / name
    surface_tif = build_surface(_RDP_DTM, _RDP_FOOTPRINTS, cell_m, out_stem, all_touched=all_touched)
    is_building_tif = surface_tif.with_name(surface_tif.stem.replace("_surface", "_is_building") + ".tif")
    surface, transform, _crs, is_building = load_surface(surface_tif, is_building_tif)
    vis, on_building = patch_visibility(
        surface, transform, obs, directions=directions, is_building=is_building,
        obs_height_m=obs_height_m, max_dist_m=500.0, step_m=step_m,
        march_sampling=march_sampling, device=device,
    )
    return vis, on_building


def main() -> int:
    if not _REF_PATH.exists() or not _RDP_DTM.exists() or not _RDP_FOOTPRINTS.exists():
        print(f"missing inputs: {_REF_PATH} / {_RDP_DTM} / {_RDP_FOOTPRINTS}")
        return 1

    directions, weights = generate_tregenza_patches()
    epw = MAIN_CHECKOUT / load_params()["weather"]["primary_epw"]
    sky = wp02_sky.build(epw) if epw.exists() else None

    ref = gpd.read_file(_REF_PATH)
    obs_xy = np.column_stack([ref.geometry.x.to_numpy(), ref.geometry.y.to_numpy()])
    ref_svf = ref["svf"].to_numpy(dtype="float64")
    obs_height_m = float((ref["z_observer"] - ref["z"]).median())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_dir_main = MAIN_CHECKOUT / "runs" / RUN_ID
    artifacts = run_dir_main / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    # --- distance to nearest footprint edge, per reference point -----------
    fps = gpd.read_file(_RDP_FOOTPRINTS)
    boundary = fps.boundary
    pts = gpd.GeoSeries(gpd.points_from_xy(obs_xy[:, 0], obs_xy[:, 1]), crs=fps.crs)
    _idx, edge_dist = boundary.sindex.nearest(pts, return_distance=True, return_all=False)
    edge_dist = np.asarray(edge_dist).reshape(-1)

    # --- baseline (1m, bilinear, step=1m, all_touched=False) --------------
    vis_base, onb_base = _run_variant(
        artifacts, "baseline", cell_m=1.0, step_m=1.0, march_sampling="bilinear", all_touched=False,
        directions=directions, obs=obs_xy, obs_height_m=obs_height_m, device=device,
    )
    svf_base = svf_unweighted(vis_base)
    valid = ~onb_base
    baseline_metrics = _compare(svf_base[valid], ref_svf[valid])
    signed_base = (svf_base - ref_svf)[valid]

    # --- binned diagnostic on the baseline ---------------------------------
    edge_bins = [(0, 1, "<1m"), (1, 2, "1-2m"), (2, 5, "2-5m"), (5, np.inf, ">5m")]
    ed_valid = edge_dist[valid]
    delta_valid = np.abs(signed_base)
    by_edge_distance = {}
    for lo, hi, label in edge_bins:
        mask = (ed_valid >= lo) & (ed_valid < hi)
        if mask.sum() == 0:
            by_edge_distance[label] = {"n": 0}
            continue
        by_edge_distance[label] = {
            "n": int(mask.sum()),
            "median_abs_delta": float(np.median(delta_valid[mask])),
            "p95_abs_delta": float(np.percentile(delta_valid[mask], 95)),
            "signed_median_delta": float(np.median(signed_base[mask])),
        }

    ref_valid = ref_svf[valid]
    quartile_edges = np.percentile(ref_valid, [0, 25, 50, 75, 100])
    by_svf_quartile = {}
    for i in range(4):
        lo, hi = quartile_edges[i], quartile_edges[i + 1]
        mask = (ref_valid >= lo) & (ref_valid <= hi if i == 3 else ref_valid < hi)
        label = f"Q{i + 1} [{lo:.3f},{hi:.3f}{']' if i == 3 else ')'}"
        by_svf_quartile[label] = {
            "n": int(mask.sum()),
            "median_abs_delta": float(np.median(delta_valid[mask])) if mask.sum() else None,
            "p95_abs_delta": float(np.percentile(delta_valid[mask], 95)) if mask.sum() else None,
            "signed_median_delta": float(np.median(signed_base[mask])) if mask.sum() else None,
        }

    # --- variants A, B, C ----------------------------------------------------
    variants = {}

    vis_a, onb_a = _run_variant(
        artifacts, "variantA_nearest", cell_m=1.0, step_m=1.0, march_sampling="nearest", all_touched=False,
        directions=directions, obs=obs_xy, obs_height_m=obs_height_m, device=device,
    )
    variants["A_nearest_sampling"] = _compare(
        svf_unweighted(vis_a)[~onb_a], ref_svf[~onb_a])

    vis_b, onb_b = _run_variant(
        artifacts, "variantB_step0.5", cell_m=1.0, step_m=0.5, march_sampling="bilinear", all_touched=False,
        directions=directions, obs=obs_xy, obs_height_m=obs_height_m, device=device,
    )
    variants["B_step_0.5m"] = _compare(
        svf_unweighted(vis_b)[~onb_b], ref_svf[~onb_b])

    vis_c, onb_c = _run_variant(
        artifacts, "variantC_alltouched", cell_m=1.0, step_m=1.0, march_sampling="bilinear", all_touched=True,
        directions=directions, obs=obs_xy, obs_height_m=obs_height_m, device=device,
    )
    variants["C_all_touched"] = _compare(
        svf_unweighted(vis_c)[~onb_c], ref_svf[~onb_c])

    # --- best combination: stack whichever of A/B/C individually reduced ---
    # median_abs_delta below the baseline.
    improving = [
        name for name in ("A_nearest_sampling", "B_step_0.5m", "C_all_touched")
        if variants[name]["median_abs_delta"] < baseline_metrics["median_abs_delta"]
    ]
    combo_kwargs = dict(cell_m=1.0, step_m=1.0, march_sampling="bilinear", all_touched=False)
    if "A_nearest_sampling" in improving:
        combo_kwargs["march_sampling"] = "nearest"
    if "B_step_0.5m" in improving:
        combo_kwargs["step_m"] = 0.5
    if "C_all_touched" in improving:
        combo_kwargs["all_touched"] = True

    if improving:
        vis_combo, onb_combo = _run_variant(
            artifacts, "combo_best", directions=directions, obs=obs_xy,
            obs_height_m=obs_height_m, device=device, **combo_kwargs,
        )
        combo_metrics = _compare(svf_unweighted(vis_combo)[~onb_combo], ref_svf[~onb_combo])
    else:
        combo_kwargs = None
        combo_metrics = None

    # --- decide the default -------------------------------------------------
    candidates = {"baseline": (baseline_metrics, dict(march_sampling="bilinear", step_m=1.0, all_touched=False))}
    candidates.update({
        "A_nearest_sampling": (variants["A_nearest_sampling"], dict(march_sampling="nearest", step_m=1.0, all_touched=False)),
        "B_step_0.5m": (variants["B_step_0.5m"], dict(march_sampling="bilinear", step_m=0.5, all_touched=False)),
        "C_all_touched": (variants["C_all_touched"], dict(march_sampling="bilinear", step_m=1.0, all_touched=True)),
    })
    if combo_metrics is not None:
        candidates["combo_best"] = (combo_metrics, combo_kwargs)

    qualifying = {
        name: (m, kw) for name, (m, kw) in candidates.items()
        if m["r"] >= 0.98 and m["median_abs_delta"] <= 0.03
    }
    if qualifying:
        default_name = max(qualifying, key=lambda n: qualifying[n][0]["r"])
        default_metrics, default_kwargs = qualifying[default_name]
    else:
        default_name, default_metrics, default_kwargs = None, None, None

    report = {
        "_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "reference": str(_REF_PATH),
        "reference_is_exact_polygons_not_a_scan": (
            "src/svf_v2/scene.py builds the CPU reference from the same DTM + "
            "extruded altura/base footprints this engine uses — the bias below "
            "is a raster/horizon-march method difference, not unmodelled geometry."
        ),
        "n_points": int(len(ref)),
        "obs_height_m_measured": obs_height_m,
        "baseline": {
            "params": {"cell_m": 1.0, "step_m": 1.0, "march_sampling": "bilinear", "all_touched": False},
            "metrics": baseline_metrics,
            "signed_median_delta": float(np.median(signed_base)),
        },
        "binned_diagnostic": {
            "variant": "1m, unweighted, bilinear march (baseline)",
            "by_distance_to_nearest_footprint_edge": by_edge_distance,
            "by_reference_svf_quartile": by_svf_quartile,
        },
        "variants": variants,
        "combo_best": {"kwargs": combo_kwargs, "metrics": combo_metrics, "combined_from": improving},
        "default_decision": {
            "floor": "r >= 0.98 and median_abs_delta <= 0.03",
            "qualifying": {k: v[0] for k, v in qualifying.items()},
            "chosen": default_name,
            "chosen_metrics": default_metrics,
            "chosen_kwargs": default_kwargs,
        },
    }

    run_dir_main.mkdir(parents=True, exist_ok=True)
    out_main = run_dir_main / "crossref_diagnostic.json"
    out_main.write_text(json.dumps(report, indent=1))

    run_dir_worktree = WORKTREE_ROOT / "runs" / RUN_ID
    run_dir_worktree.mkdir(parents=True, exist_ok=True)
    (run_dir_worktree / "crossref_diagnostic.json").write_text(json.dumps(report, indent=1))

    write_run_manifest(
        run_dir_main, cell_m=1.0, obs_height_m=obs_height_m, max_dist_m=500.0, step_m=1.0,
        sampling_rule="wp02_horizon_diagnostic: baseline + variants A/B/C + combo vs 16905 street points",
        device=device,
    )

    print(json.dumps(report, indent=1))
    print(f"\nwrote {out_main}\nwrote {run_dir_worktree / 'crossref_diagnostic.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
