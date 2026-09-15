#!/usr/bin/env python3
"""WP-04F2: façade SVF cross-reference against the CPU mesh raycaster.

Spec: docs/wp04f2_facade_crossref_spec.md. Ground and street already have a
CPU cross-reference (test_wp02_horizon.py's test 6); façades do not. This
draws a stratified subsample of façade points per site from the already-run
raster (wp04_sites' ``facade.parquet``) and re-evaluates the SAME points
(same x, y, z, normal) with the CPU mesh raycaster --
``src.svf_v2.compute.compute_svf_raycasting`` on ``svf_v2.scene.build_scene``'s
extruded-polygon mesh, called the exact way ``src.svf_v2.facades.compute_facade_svf``
does (normal-restricted hemisphere, UNWEIGHTED count ratio -- facades.py never
passes ``sky_weights``), except with Tregenza directions substituted for that
function's default uniform grid so the forward-patch set is directly
comparable to the raster's own Tregenza hemisphere.

The raster side reuses each point's already-computed raw per-patch visibility
(``facade.parquet``'s ``visibility_packed`` column, produced by the SAME
``patch_visibility`` call the production WP-04 run used, own-building
exclusion included) rather than re-running the raster engine -- the point set
here is a subsample of exactly those already-evaluated rows.

Data lives only in the main checkout (gitignored): read from
``data/<site>/`` and ``runs/wp04_sites_20260915T063553Z/``; written to
``runs/wp04f2_facade_<UTC>/``.

Run: python3 -m scripts.run_wp04f2_facade_crossref
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.brisa_solar.constants import P1_SKY_PATCHES, load_params
from src.brisa_solar.wp02_horizon import hemisphere_mask
from src.brisa_solar.wp05_pilot import unpack_visibility
from src.svf_v2.compute import compute_svf_raycasting, generate_tregenza_patches
from src.svf_v2.scene import build_scene

MAIN_CHECKOUT = Path("/home/theo/SCL/SCR/MorphoFavela")
SOURCE_RUN = MAIN_CHECKOUT / "runs" / "wp04_sites_20260915T063553Z"
SITES: list[str] = ["riodaspedras", "vidigal"]

#: (lo_m, hi_m, label) -- 4 height_above_ground bins, per spec deliverable 1.
HEIGHT_BIN_EDGES: list[tuple[float, float, str]] = [
    (0.0, 3.0, "0-3m"),
    (3.0, 6.0, "3-6m"),
    (6.0, 9.0, "6-9m"),
    (9.0, np.inf, ">9m"),
]
N_PER_BIN = 500
MAX_RAY_LENGTH_M = 500.0  # svf_v2.facades.compute_facade_svf's own default

#: Provisional floor, as for streets (test_wp02_horizon.py test 6) -- spec deliverable 2.
FLOOR_R = 0.95
FLOOR_MEDIAN_ABS_DELTA = 0.03

FACADE_COLUMNS = [
    "x", "y", "z", "normal_x", "normal_y", "normal_z",
    "height_above_ground", "svf", "visibility_packed",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sampling_seed() -> int:
    return int(load_params()["sampling"]["random_seed"])


# ---------------------------------------------------------------------------
# 1. Stratified subsample
# ---------------------------------------------------------------------------

def stratified_subsample(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """N_PER_BIN points per height_above_ground bin, seeded from
    ``sampling.random_seed`` in params.yaml.

    ONE rng, drawn against the bins in the fixed ``HEIGHT_BIN_EDGES`` order --
    reproducible from (df, seed) alone, no per-bin re-seeding that could
    silently draw overlapping sub-sequences. A bin with fewer than
    ``N_PER_BIN`` points keeps all of them (reported, never padded).
    """
    rng = np.random.default_rng(seed)
    height = df["height_above_ground"].to_numpy(dtype="float64")
    parts = []
    for lo, hi, label in HEIGHT_BIN_EDGES:
        idx = np.where((height >= lo) & (height < hi))[0]
        n = min(N_PER_BIN, len(idx))
        chosen = rng.choice(idx, size=n, replace=False) if n > 0 else idx
        part = df.iloc[chosen].copy()
        part["height_bin"] = label
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# 2. CPU mesh raycaster reference
# ---------------------------------------------------------------------------

def reference_svf(
    observer_points: np.ndarray,
    normals: np.ndarray,
    scene_mesh,
    tregenza_dirs: np.ndarray,
) -> np.ndarray:
    """``compute_facade_svf``'s own algorithm, Tregenza directions substituted
    for its default uniform grid: normal-restricted hemisphere, UNWEIGHTED
    count ratio (no ``sky_weights`` -- src/svf_v2/facades.py never passes one).
    """
    return compute_svf_raycasting(
        observer_points, scene_mesh, tregenza_dirs,
        max_ray_length=MAX_RAY_LENGTH_M, normals=normals, n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# 3. Raster SVF variants, on the SAME forward hemisphere the reference tests
# ---------------------------------------------------------------------------

def raster_variants(
    vis: np.ndarray, mask: np.ndarray, weights: np.ndarray, cosine_svf: np.ndarray,
) -> dict[str, np.ndarray]:
    """The 3 raster SVF conventions (docs/wp04f2_facade_crossref_spec.md
    deliverable 1: count ratio vs solid-angle vs cosine-weighted).

    ``unweighted`` and ``solid_angle`` are restricted to the forward
    hemisphere (``mask``) the reference itself tests. ``cosine_weighted``
    instead reuses the raster's own already-computed production column
    (``src.brisa_solar.wp04_sites.facade_svf_irradiation``): that formula's
    denominator is the WHOLE hemisphere's cosine sum, not the forward half,
    by design (an unobstructed wall reads ~0.5, not ~1.0 -- see that
    function's docstring), so masking it here would silently redefine the
    raster's own published number rather than just report it.
    """
    n = vis.shape[0]
    front_visible = vis & mask
    n_forward = mask.sum(axis=1)
    w_forward = (mask * weights[None, :]).sum(axis=1)
    unweighted = np.divide(
        front_visible.sum(axis=1), n_forward,
        out=np.zeros(n, dtype="float64"), where=n_forward > 0,
    )
    solid_angle = np.divide(
        (front_visible * weights[None, :]).sum(axis=1), w_forward,
        out=np.zeros(n, dtype="float64"), where=w_forward > 0,
    )
    return {
        "unweighted": unweighted,
        "solid_angle": solid_angle,
        "cosine_weighted": np.asarray(cosine_svf, dtype="float64"),
    }


# ---------------------------------------------------------------------------
# 4. Comparison stats
# ---------------------------------------------------------------------------

def _compare(measured: np.ndarray, reference: np.ndarray) -> dict:
    measured = np.asarray(measured, dtype="float64")
    reference = np.asarray(reference, dtype="float64")
    n = int(len(reference))
    if n == 0:
        return {"n": 0}
    delta = measured - reference
    abs_delta = np.abs(delta)
    if n > 1 and np.std(measured) > 0 and np.std(reference) > 0:
        r = float(np.corrcoef(measured, reference)[0, 1])
    else:
        r = float("nan")
    raster_zero = measured == 0.0
    n_raster_zero = int(raster_zero.sum())
    zero_agreement = (
        float((reference[raster_zero] == 0.0).mean()) if n_raster_zero > 0 else None
    )
    return {
        "n": n,
        "r": r,
        "median_abs_delta": float(np.median(abs_delta)),
        "p95_abs_delta": float(np.percentile(abs_delta, 95)),
        "signed_median_delta": float(np.median(delta)),
        "n_raster_zero": n_raster_zero,
        "zero_agreement_among_raster_zeros": zero_agreement,
    }


# ---------------------------------------------------------------------------
# 5. One site
# ---------------------------------------------------------------------------

def run_site(site_key: str, seed: int, run_dir: Path) -> dict:
    facade_path = SOURCE_RUN / site_key / "facade.parquet"
    df = pd.read_parquet(facade_path, columns=FACADE_COLUMNS)
    sub = stratified_subsample(df, seed)

    dtm_path = MAIN_CHECKOUT / f"data/{site_key}/dtm_extended_300m.tif"
    fp_path = MAIN_CHECKOUT / f"data/{site_key}/buildings_extended_300m.gpkg"
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    scene_mesh, _terrain, _gdf = build_scene(
        dtm_path, fp_path, cache_vtk=artifacts / f"{site_key}_scene.vtk"
    )

    tregenza_dirs, weights = generate_tregenza_patches()

    observer_points = sub[["x", "y", "z"]].to_numpy(dtype="float64")
    normals = sub[["normal_x", "normal_y", "normal_z"]].to_numpy(dtype="float64")

    ref_svf = reference_svf(observer_points, normals, scene_mesh, tregenza_dirs)

    mask = hemisphere_mask(tregenza_dirs, normals)
    packed = np.stack(
        [np.frombuffer(b, dtype=np.uint8) for b in sub["visibility_packed"].to_numpy()]
    )
    vis = unpack_visibility(packed, n_patches=P1_SKY_PATCHES)

    variants = raster_variants(vis, mask, weights, sub["svf"].to_numpy(dtype="float64"))

    bin_labels = sub["height_bin"].to_numpy()
    site_result: dict = {"n_total": int(len(sub)), "variants": {}}
    for name, measured in variants.items():
        overall = _compare(measured, ref_svf)
        by_bin = {}
        for lo, hi, label in HEIGHT_BIN_EDGES:
            m = bin_labels == label
            by_bin[label] = _compare(measured[m], ref_svf[m]) if m.any() else {"n": 0}
        site_result["variants"][name] = {"overall": overall, "by_height_bin": by_bin}

    variant_names = list(variants)
    ranked = sorted(
        variant_names,
        key=lambda nm: (
            site_result["variants"][nm]["overall"]["r"]
            if np.isfinite(site_result["variants"][nm]["overall"]["r"]) else -np.inf
        ),
        reverse=True,
    )
    best_name = ranked[0]
    best = site_result["variants"][best_name]["overall"]
    site_result["best_variant"] = best_name
    site_result["floor_pass"] = bool(
        np.isfinite(best["r"]) and best["r"] >= FLOOR_R
        and best["median_abs_delta"] <= FLOOR_MEDIAN_ABS_DELTA
    )
    return site_result


# ---------------------------------------------------------------------------
# 6. Markdown report
# ---------------------------------------------------------------------------

def _fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if not np.isfinite(v):
            return "nan"
        return f"{v:.4f}"
    return str(v)


def write_markdown(out_path: Path, results: dict) -> None:
    lines = [
        "# WP-04F2 façade cross-reference against the CPU mesh raycaster",
        "",
        f"Run: {results['_utc']}",
        "",
        "Reference: `src/svf_v2/facades.py::compute_facade_svf` on "
        "`svf_v2.scene.build_scene(dtm_extended_300m, buildings_extended_300m)` -- "
        "normal-restricted hemisphere, UNWEIGHTED count ratio (Tregenza directions).",
        "",
        f"Provisional floor: r >= {FLOOR_R} and median|Δ| <= {FLOOR_MEDIAN_ABS_DELTA} "
        "overall per site, on the best-matching raster variant.",
        "",
    ]
    for site_key, site_result in results["sites"].items():
        lines.append(f"## {site_key}")
        lines.append("")
        lines.append(
            f"n = {site_result['n_total']}, best variant = **{site_result['best_variant']}**, "
            f"floor {'PASS' if site_result['floor_pass'] else 'FAIL'}"
        )
        lines.append("")
        for variant_name, variant_result in site_result["variants"].items():
            lines.append(f"### variant: {variant_name}")
            lines.append("")
            lines.append("| bin | n | r | median\\|Δ\\| | p95\\|Δ\\| | signed median Δ | zero agreement |")
            lines.append("|---|---|---|---|---|---|---|")
            ov = variant_result["overall"]
            lines.append(
                f"| **overall** | {_fmt(ov.get('n'))} | {_fmt(ov.get('r'))} | "
                f"{_fmt(ov.get('median_abs_delta'))} | {_fmt(ov.get('p95_abs_delta'))} | "
                f"{_fmt(ov.get('signed_median_delta'))} | "
                f"{_fmt(ov.get('zero_agreement_among_raster_zeros'))} |"
            )
            for _lo, _hi, label in HEIGHT_BIN_EDGES:
                b = variant_result["by_height_bin"][label]
                lines.append(
                    f"| {label} | {_fmt(b.get('n'))} | {_fmt(b.get('r'))} | "
                    f"{_fmt(b.get('median_abs_delta'))} | {_fmt(b.get('p95_abs_delta'))} | "
                    f"{_fmt(b.get('signed_median_delta'))} | "
                    f"{_fmt(b.get('zero_agreement_among_raster_zeros'))} |"
                )
            lines.append("")
    out_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    utc = _utc_now()
    run_dir = MAIN_CHECKOUT / "runs" / f"wp04f2_facade_{utc}"
    run_dir.mkdir(parents=True, exist_ok=True)
    seed = sampling_seed()

    results: dict = {
        "_utc": utc,
        "reference": "src/svf_v2/facades.py::compute_facade_svf (Tregenza directions)",
        "sampling_seed": seed,
        "n_per_bin": N_PER_BIN,
        "floor": {"r": FLOOR_R, "median_abs_delta": FLOOR_MEDIAN_ABS_DELTA},
        "sites": {},
    }

    all_pass = True
    for site_key in SITES:
        print(f"[wp04f2] {site_key}: sampling + CPU mesh raycasting ...", flush=True)
        site_result = run_site(site_key, seed, run_dir)
        results["sites"][site_key] = site_result
        all_pass = all_pass and site_result["floor_pass"]
        best = site_result["variants"][site_result["best_variant"]]["overall"]
        print(
            f"[wp04f2] {site_key}: best={site_result['best_variant']} "
            f"r={best['r']:.4f} median|Δ|={best['median_abs_delta']:.4f} "
            f"floor={'PASS' if site_result['floor_pass'] else 'FAIL'}",
            flush=True,
        )

    out_json = run_dir / "crossref.json"
    out_json.write_text(json.dumps(results, indent=1))
    write_markdown(run_dir / "crossref.md", results)

    print(f"[wp04f2] wrote {out_json}")
    print(f"[wp04f2] all sites floor pass: {all_pass}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
