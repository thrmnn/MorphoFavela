"""WP-04F2 acceptance: façade SVF cross-reference against the CPU mesh raycaster.

Spec: docs/wp04f2_facade_crossref_spec.md. Large data (facade.parquet, DTM,
footprints) lives only in the main checkout, not this worktree -- data/,
outputs/ and runs/ are gitignored and were never copied into the worktree --
so those paths are read absolute from the main checkout rather than via
REPO_ROOT (same pattern as tests/test_wp02_horizon.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.brisa_solar.constants import load_params
from src.brisa_solar.wp02_horizon import hemisphere_mask
from src.svf_v2.compute import generate_tregenza_patches

import scripts.run_wp04f2_facade_crossref as wp04f2

MAIN_CHECKOUT = Path("/home/theo/SCL/SCR/MorphoFavela")


# ---------------------------------------------------------------------------
# (a) The subsample is seeded and stratified as specified
# ---------------------------------------------------------------------------

def test_subsample_seeded_and_stratified():
    facade_path = wp04f2.SOURCE_RUN / "riodaspedras" / "facade.parquet"
    if not facade_path.exists():
        pytest.skip(f"facade.parquet not on disk: {facade_path}")

    seed = wp04f2.sampling_seed()
    assert seed == load_params()["sampling"]["random_seed"], (
        "sampling_seed() must read sampling.random_seed from params.yaml, not a copy"
    )

    df = pd.read_parquet(facade_path, columns=["height_above_ground"])

    sub_a = wp04f2.stratified_subsample(df, seed)
    sub_b = wp04f2.stratified_subsample(df, seed)
    # same (df, seed) must reproduce the identical subsample
    pd.testing.assert_frame_equal(sub_a.reset_index(drop=True), sub_b.reset_index(drop=True))

    height = df["height_above_ground"].to_numpy(dtype="float64")
    for lo, hi, label in wp04f2.HEIGHT_BIN_EDGES:
        part = sub_a[sub_a["height_bin"] == label]
        available = int(((height >= lo) & (height < hi)).sum())
        assert len(part) == min(wp04f2.N_PER_BIN, available), (
            f"bin {label}: expected {min(wp04f2.N_PER_BIN, available)} points, got {len(part)}"
        )
        part_height = part["height_above_ground"].to_numpy(dtype="float64")
        assert (part_height >= lo).all(), f"bin {label}: a point falls below its lower edge"
        assert (part_height < hi).all(), f"bin {label}: a point falls at/above its upper edge"

    # No point qualifies for two bins at once (edges are half-open [lo, hi)) --
    # would only happen from an indexing bug in stratified_subsample itself.
    for lo, hi, label in wp04f2.HEIGHT_BIN_EDGES:
        part_height = sub_a.loc[sub_a["height_bin"] == label, "height_above_ground"].to_numpy()
        for other_lo, other_hi, other_label in wp04f2.HEIGHT_BIN_EDGES:
            if other_label == label:
                continue
            assert not (
                ((part_height >= other_lo) & (part_height < other_hi)).any()
            ), f"a point binned {label} also qualifies for {other_label}"


# ---------------------------------------------------------------------------
# (b) The reference and raster hemispheres agree (same normal -> same
#     forward patch set). No data dependency -- pure geometry, always runs.
# ---------------------------------------------------------------------------

def test_reference_and_raster_hemispheres_agree():
    directions, _weights = generate_tregenza_patches()

    rng = np.random.default_rng(20260915)
    normals = rng.normal(size=(50, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    raster_mask = hemisphere_mask(directions, normals)

    # The CPU reference's OWN forward-hemisphere test (src/svf_v2/compute.py
    # ``_svf_for_point_obb`` / ``_svf_for_point_multi_ray``): ``dots = sky_directions
    # @ normal; mask = dots > 0`` -- reproduced here independently (not imported,
    # since it is not exposed as a standalone function) so this test would catch
    # either side silently drifting to a different threshold (e.g. ``>=``).
    reference_mask = (directions[None, :, :] * normals[:, None, :]).sum(axis=-1) > 0.0

    np.testing.assert_array_equal(raster_mask, reference_mask)
    # Sanity: a real hemisphere split, not a degenerate all-True/all-False mask.
    assert 0 < raster_mask.sum() < raster_mask.size


# ---------------------------------------------------------------------------
# (c) The crossref floor -- skips cleanly if the run output is absent.
# ---------------------------------------------------------------------------

def _latest_crossref_json() -> Path | None:
    candidates = sorted(MAIN_CHECKOUT.glob("runs/wp04f2_facade_*/crossref.json"))
    return candidates[-1] if candidates else None


@pytest.mark.xfail(strict=True, reason=(
    "MEASURED FAILURE 2026-09-15 (runs/wp04f2_facade_2026-09-15T07:50:07Z): façade SVF vs the "
    "CPU mesh raycaster reads r 0.884 / median|Δ| 0.028 at Rio das Pedras and r 0.644 / 0.083 "
    "at Vidigal, worst above 9 m (rooftop-adjacent points). The façade layer is NOT ACCEPTED; "
    "the floor is kept and this xfail is strict so a future engine change that clears it is noticed."))
def test_crossref_floor():
    path = _latest_crossref_json()
    if path is None:
        pytest.skip(
            "no runs/wp04f2_facade_*/crossref.json on disk -- run "
            "scripts/run_wp04f2_facade_crossref.py first"
        )
    data = json.loads(path.read_text())
    floor = data["floor"]

    failures = []
    for site_key, site_result in data["sites"].items():
        if site_result["floor_pass"]:
            continue
        best_name = site_result["best_variant"]
        best = site_result["variants"][best_name]["overall"]
        by_bin = site_result["variants"][best_name]["by_height_bin"]
        bins_str = ", ".join(
            f"{label}: n={stats.get('n')} r={stats.get('r'):.4f} "
            f"median|Δ|={stats.get('median_abs_delta'):.4f}"
            for label, stats in by_bin.items()
            if stats.get("n")
        )
        failures.append(
            f"{site_key} (best variant '{best_name}'): "
            f"r={best['r']:.4f} median|Δ|={best['median_abs_delta']:.4f} "
            f"(floor: r >= {floor['r']} and median|Δ| <= {floor['median_abs_delta']}) "
            f"-- per bin: {bins_str}"
        )

    assert not failures, (
        "PROVISIONAL floor failed -- measured numbers in " + str(path) + ":\n"
        + "\n".join(failures)
    )
