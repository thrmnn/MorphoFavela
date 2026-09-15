"""WP-05 FULL acceptance: spec docs/wp05_full_spec.md, "Deliverables" §3.

Real DTM/footprints/EPW/Favelas_Limit_2019 live only in the main checkout
(data/ and runs/ are gitignored, never copied into this worktree) — same
convention as tests/test_wp05_pilot.py. These tests exercise the exhaustive
frame construction, tile-checkpoint consolidation and distribution math on
synthetic inputs, and the favela-name matching on a small synthetic
GeoDataFrame that reproduces the real layer's known ambiguity (a favela name
that is also a substring of an unrelated neighbouring polygon's name).
"""
from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

from src.brisa_solar import wp05_full, wp05_pilot


# ---------------------------------------------------------------------------
# (a) exhaustive mode yields exactly the frame count on a synthetic raster
# ---------------------------------------------------------------------------

def test_exhaustive_observers_equal_frame_count():
    size = 60
    cell_m = 5.0
    rng = np.random.default_rng(1)
    dtm = rng.uniform(0.0, 10.0, size=(size, size)).astype("float32")
    dtm[:5, :] = np.nan  # outside-municipality strip

    is_building = np.zeros((size, size), dtype=bool)
    is_building[30:36, 30:36] = True

    frame = wp05_pilot.build_frame(
        dtm, is_building, cell_m=cell_m,
        fabric_coverage_threshold=0.05, fabric_footprint_distance_m=15.0,
    )
    slope_deg = wp05_pilot.compute_slope_deg(dtm, cell_m)
    stratum = wp05_pilot.assign_strata(slope_deg, frame["coverage"])
    favela_id_raster = np.zeros((size, size), dtype="int32")

    from affine import Affine
    transform = Affine(cell_m, 0, 0.0, 0, -cell_m, size * cell_m)

    obs_df = wp05_full.observers_from_frame(frame["in_frame"], stratum, favela_id_raster, transform)

    # Exhaustive mode: no stratified draw, no floor, no fraction — every
    # in-frame cell is an observer, exactly.
    assert len(obs_df) == frame["frame_cells"]
    assert set(obs_df.columns) >= {"row", "col", "x", "y", "stratum", "favela_id"}
    assert len(obs_df) == len(obs_df[["row", "col"]].drop_duplicates())


# ---------------------------------------------------------------------------
# (b) per-tile checkpoints reassemble to the consolidated file, no dup (x, y)
# ---------------------------------------------------------------------------

def test_tile_checkpoints_consolidate_without_duplicate_xy(tmp_path):
    tile_dir = tmp_path / "tiles"
    tile_dir.mkdir()

    rng = np.random.default_rng(2)
    n_tiles, n_per_tile = 4, 25
    total = 0
    for t in range(n_tiles):
        n = n_per_tile
        df = pd.DataFrame({
            "x": np.arange(total, total + n, dtype="float64"),
            "y": np.full(n, float(t)),
            "tile": f"{t}_0",
            "stratum": rng.integers(0, 9, size=n).astype("int16"),
            "favela_id": rng.integers(0, 5, size=n).astype("int32"),
            "on_building": False,
            "svf": rng.uniform(0.0, 1.0, size=n),
            "kwh_m2": rng.uniform(800.0, 1800.0, size=n),
            "sky_model": "epw_weighted",
        })
        df.to_parquet(tile_dir / f"tile_{t}_0.parquet", index=False)
        total += n

    merged = wp05_full.consolidate_tiles(tile_dir, tmp_path / "consolidated.parquet")

    assert len(merged) == n_tiles * n_per_tile
    assert merged.duplicated(subset=["x", "y"]).sum() == 0
    assert (tmp_path / "consolidated.parquet").exists()
    reloaded = pd.read_parquet(tmp_path / "consolidated.parquet")
    assert len(reloaded) == len(merged)


def test_consolidate_tiles_raises_on_duplicate_xy(tmp_path):
    tile_dir = tmp_path / "tiles"
    tile_dir.mkdir()
    df = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 0.0], "svf": [0.5, 0.6], "kwh_m2": [1000.0, 1100.0]})
    df.to_parquet(tile_dir / "tile_0_0.parquet", index=False)
    df.to_parquet(tile_dir / "tile_1_0.parquet", index=False)  # same (x, y) rows, a different tile

    with pytest.raises(ValueError, match="duplicate"):
        wp05_full.consolidate_tiles(tile_dir, tmp_path / "out.parquet")


# ---------------------------------------------------------------------------
# (c) distribution.json quantiles are monotone and inside [0, 1] for svf
# ---------------------------------------------------------------------------

def test_distribution_quantiles_monotone_and_svf_bounded():
    rng = np.random.default_rng(3)
    n = 5000
    consolidated = pd.DataFrame({
        "svf": rng.uniform(0.0, 1.0, size=n),
        "kwh_m2": rng.uniform(600.0, 2000.0, size=n),
        "stratum": rng.integers(0, 9, size=n).astype("int16"),
        "favela_id": np.zeros(n, dtype="int32"),
    })
    favelas = gpd.GeoDataFrame(
        {"objectid": [], "cod_favela": [], "nome": [], "complexo": []},
        geometry=[], crs="EPSG:31983",
    )

    dist = wp05_full.compute_distribution(consolidated, favelas)

    svf_block = dist["citywide"]["svf"]
    ordered_keys = [f"p{round(q * 100)}" for q in wp05_full.QUANTILES]
    values = [svf_block[k] for k in ordered_keys]
    assert values == sorted(values), "quantiles must be monotone non-decreasing"
    assert all(0.0 <= v <= 1.0 for v in values), "svf quantiles must lie in [0, 1]"
    assert dist["citywide"]["svf"]["status"] == wp05_full.PROVISIONAL_STATUS
    assert dist["n_cells"] == n

    for s, block in dist["per_stratum"].items():
        s_values = [block["svf"][k] for k in ordered_keys]
        assert s_values == sorted(s_values)
        assert all(0.0 <= v <= 1.0 for v in s_values)

    # distribution.json round-trips through JSON without loss of the
    # monotonicity/bounds properties just checked.
    reloaded = json.loads(json.dumps(dist))
    assert reloaded["citywide"]["svf"]["p50"] == svf_block["p50"]


# ---------------------------------------------------------------------------
# (d) the five-favela match is explicit: matched polygon ids are recorded,
# not merely that a match exists — and ambiguity is flagged, not guessed.
# ---------------------------------------------------------------------------

def _square(x0, y0, s=10.0):
    return Polygon([(x0, y0), (x0 + s, y0), (x0 + s, y0 + s), (x0, y0 + s)])


def test_favela_match_records_polygon_ids_and_flags_ambiguity():
    # Reproduces the real layer's shape: "Rocinha" is a standalone (Isolada)
    # favela, but "Matinha (RA - Rocinha)" is a DIFFERENT polygon that merely
    # references it by name — a contains-match would wrongly pull it in.
    # "Complexo do Alemão" spans several polygons grouped under one complexo.
    favelas = gpd.GeoDataFrame({
        "objectid": [1, 2, 3, 4, 5],
        "cod_favela": [43, 976, 861, 860, 999],
        "nome": ["Rocinha", "Matinha (RA - Rocinha)", "Estrada do Itararé", "Rua 1 pela Ademas", "Unrelated Hill"],
        "complexo": ["Isolada", "Isolada", "Complexo do Alemão", "Complexo do Alemão", "Isolada"],
        "geometry": [_square(0, 0), _square(20, 0), _square(0, 20), _square(20, 20), _square(40, 40)],
    }, crs="EPSG:31983")

    matched, method = wp05_full.match_favela_group(favelas, "Rocinha")
    assert method == "nome_exact_unique"
    ids = [r["cod_favela"] for r in wp05_full._polygon_records(matched)]
    assert ids == [43]  # exactly Rocinha itself, NOT the Matinha cross-reference

    matched, method = wp05_full.match_favela_group(favelas, "Complexo do Alemão")
    assert method == "complexo_exact"
    ids = sorted(r["cod_favela"] for r in wp05_full._polygon_records(matched))
    assert ids == [860, 861]  # both polygons under the complexo, no others

    matched, method = wp05_full.match_favela_group(favelas, "Nonexistent Favela")
    assert method == "no_match"
    assert wp05_full._polygon_records(matched) == []

    # An ambiguous case: two DIFFERENT polygons share the same exact nome and
    # neither has a matching complexo — must be flagged, never guessed at.
    ambiguous = gpd.GeoDataFrame({
        "objectid": [10, 11],
        "cod_favela": [500, 501],
        "nome": ["Duplicate Name", "Duplicate Name"],
        "complexo": ["Isolada", "Isolada"],
        "geometry": [_square(100, 0), _square(120, 0)],
    }, crs="EPSG:31983")
    matched, method = wp05_full.match_favela_group(ambiguous, "Duplicate Name")
    assert method == "nome_exact_AMBIGUOUS"
    assert len(matched) == 2


def test_five_study_favelas_all_have_a_match_method_recorded():
    favelas = gpd.GeoDataFrame({
        "objectid": [1],
        "cod_favela": [1],
        "nome": ["Vidigal"],
        "complexo": ["Isolada"],
        "geometry": [_square(0, 0)],
    }, crs="EPSG:31983")

    for name in wp05_full.STUDY_FAVELAS:
        matched, method = wp05_full.match_favela_group(favelas, name)
        assert method in {"complexo_exact", "nome_exact_unique", "nome_exact_AMBIGUOUS", "no_match"}
        # The test asserts the *method and ids are recorded*, not that a
        # match exists (per spec: "test asserts the matched polygon ids are
        # recorded, not that they exist").
        records = wp05_full._polygon_records(matched)
        assert isinstance(records, list)
        if name == "Vidigal":
            assert records == [{"objectid": 1, "cod_favela": 1, "nome": "Vidigal", "complexo": "Isolada"}]
        else:
            assert records == []  # not present in this tiny synthetic layer — recorded as such, not guessed
