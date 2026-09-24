"""MAREBOUND invariant for scripts/build_site_dashboard.py's compute_stats():
the near_boundary / edge-halo 15 m logic must measure distance to the site
DATA EXTENT (the bairro polygon) — never to the study-area outline (the
16-community union), even though the study area now governs which
cells/observers count elsewhere in the same stats dict.

Synthetic geometry, no real pipeline outputs needed: a data-extent square
much bigger than the union of two "communities" that sit well inside it, so
a point near the STUDY AREA's own edge but far from the DATA EXTENT edge
must NOT be flagged "near boundary". If the code accidentally swapped in
the study-area edge there, this exact point is close enough to trip it —
that is the "bug to avoid" MAREBOUND names.
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import build_site_dashboard as bsd  # noqa: E402

CRS = 31983


def _synthetic_d(observer_xy: tuple[float, float]) -> dict:
    # DATA EXTENT: a big square, (0,0)-(1000,1000).
    boundary = gpd.GeoDataFrame(geometry=[box(0, 0, 1000, 1000)], crs=CRS)

    # Two "communities" whose union sits 100 m inside the data extent on
    # every side — its own edge is nowhere near the true boundary.
    comm_a = box(100, 100, 500, 900)
    comm_b = box(500, 100, 900, 900)
    communities = gpd.GeoDataFrame(
        {"community": ["A", "B"]}, geometry=[comm_a, comm_b], crs=CRS,
    )
    study_area_geom = communities.geometry.union_all()
    excluded_communities = gpd.GeoDataFrame({"community": []}, geometry=[], crs=CRS)

    svf = gpd.GeoDataFrame(
        {"svf": [0.5], "street_id": [1], "distance_along": [0.0], "offset_distance": [0.0]},
        geometry=[Point(*observer_xy)], crs=CRS,
    )

    grid = pd.DataFrame({"center_x": [300.0, 700.0], "center_y": [500.0, 500.0]})
    from src.brisa_solar import mare_study_area as msa

    study_area_mask = msa.within_mask(grid["center_x"].to_numpy(), grid["center_y"].to_numpy(), study_area_geom)

    return dict(
        svf=svf, solar=None, seg=None, boundary=boundary, buildings=None,
        manifest={}, grid=grid, communities=communities,
        excluded_communities=excluded_communities,
        study_area_geom=study_area_geom, study_area_mask=study_area_mask,
    )


def test_edge_halo_uses_data_extent_not_study_area_edge():
    # 5 m inside the STUDY AREA's own edge (x=100), but 105 m inside the
    # true DATA EXTENT edge (x=0) — nowhere near the real 15 m boundary
    # buffer, so it must NOT be flagged as near_boundary.
    d = _synthetic_d((105.0, 500.0))
    stats = bsd.compute_stats(d)
    assert stats["edge_share"] == 0.0, (
        "an observer 105 m from the true site boundary but only 5 m from "
        "the study-area edge was flagged near_boundary — the 15 m buffer "
        "is measuring against the study area, not the data extent"
    )


def test_edge_halo_still_flags_true_data_extent_edge():
    # 5 m inside the DATA EXTENT's own edge (x=0) AND inside the study area
    # (community A is widened to touch x=0 for this one case) — this IS
    # within 15 m of the real boundary and must be flagged, proving the
    # test above isn't vacuously passing because nothing is ever flagged.
    d = _synthetic_d((5.0, 500.0))
    comm_a = box(0, 100, 500, 900)
    comm_b = box(500, 100, 900, 900)
    d["communities"] = gpd.GeoDataFrame({"community": ["A", "B"]}, geometry=[comm_a, comm_b], crs=CRS)
    d["study_area_geom"] = d["communities"].geometry.union_all()
    stats = bsd.compute_stats(d)
    assert stats["edge_share"] == 1.0


def test_non_mare_sites_unaffected_by_study_area_fields():
    """A site dict with study_area_geom=None (every non-Maré site) must
    take the pre-MAREBOUND code path exactly: n_grid_cells/area_km2/n_obs
    off the full boundary/grid/svf, no KeyError from the new fields."""
    boundary = gpd.GeoDataFrame(geometry=[box(0, 0, 1000, 1000)], crs=CRS)
    svf = gpd.GeoDataFrame(
        {"svf": [0.5, 0.6], "street_id": [1, 2], "distance_along": [0.0, 0.0], "offset_distance": [0.0, 0.0]},
        geometry=[Point(500, 500), Point(5, 500)], crs=CRS,
    )
    grid = pd.DataFrame({"center_x": [300.0], "center_y": [500.0]})
    d = dict(
        svf=svf, solar=None, seg=None, boundary=boundary, buildings=None,
        manifest={}, grid=grid, communities=None, excluded_communities=None,
        study_area_geom=None, study_area_mask=None,
    )
    stats = bsd.compute_stats(d)
    assert stats["n_grid_cells"] == 1
    assert stats["n_obs"] == 2
    assert stats["area_km2"] == pytest.approx(1.0)  # 1000x1000 m = 1 km²
    assert stats["edge_share"] == 0.5  # exactly one of the two points is within 15 m


# --- scripts/build_html_dashboard.py -----------------------------------------

import build_html_dashboard as bhd  # noqa: E402


def _synthetic_obs_seg():
    obs = gpd.GeoDataFrame(
        {
            "street_id": [1, 2], "distance_along": [0.0, 0.0],
            "svf": [0.5, 0.6], "solar_hours_annual": [5.0, 6.0],
        },
        geometry=[Point(500, 500), Point(500, 500)], crs=CRS,
    )
    seg = gpd.GeoDataFrame(
        {"length_m": [100.0], "svf_mean": [0.55]},
        geometry=[box(400, 490, 600, 510)], crs=CRS,
    )
    return obs, seg


def test_area_km2_override_used_when_given():
    obs, seg = _synthetic_obs_seg()
    boundary = gpd.GeoDataFrame(geometry=[box(0, 0, 1000, 1000)], crs=CRS)  # 1 km²
    stats = bhd.compute_site_stats("maré", obs, seg, boundary, area_km2_override=0.25)
    assert stats["area_km2"] == pytest.approx(0.25)
    assert stats["density_per_km2"] == pytest.approx(len(obs) / 0.25)


def test_area_km2_falls_back_to_boundary_without_override():
    obs, seg = _synthetic_obs_seg()
    boundary = gpd.GeoDataFrame(geometry=[box(0, 0, 1000, 1000)], crs=CRS)  # 1 km²
    stats = bhd.compute_site_stats("vidigal", obs, seg, boundary)
    assert stats["area_km2"] == pytest.approx(1.0)


def test_edge_share_in_compute_site_stats_uses_passed_in_boundary_only():
    # compute_site_stats has no notion of a study area at all — it is the
    # CALLER's job (build_site()/main()) to pass an already study-area-
    # filtered `obs`, while `boundary` stays the data extent. This test
    # locks that division of responsibility: edge_share must come out
    # identical whether or not an area_km2_override is passed, since the
    # override only touches area_km2/density, never the boundary used for
    # distance.
    obs, seg = _synthetic_obs_seg()
    boundary = gpd.GeoDataFrame(geometry=[box(0, 0, 1000, 1000)], crs=CRS)
    stats_a = bhd.compute_site_stats("maré", obs, seg, boundary, area_km2_override=0.25)
    stats_b = bhd.compute_site_stats("maré", obs, seg, boundary, area_km2_override=None)
    assert stats_a["edge_share"] == stats_b["edge_share"]


def test_write_communities_geojson_roundtrip(tmp_path):
    communities = gpd.GeoDataFrame(
        {"community": ["Alpha", "Beta"]},
        geometry=[box(0, 0, 10, 10), box(20, 20, 30, 30)], crs=CRS,
    )
    out = tmp_path / "communities.geojson"
    assert bhd.write_communities_geojson(communities, out) is True
    written = gpd.read_file(out)
    assert sorted(written["name"]) == ["Alpha", "Beta"]
    assert written.crs.to_epsg() == 4326


def test_write_communities_geojson_false_when_none():
    assert bhd.write_communities_geojson(None, Path("/tmp/should_not_be_written.geojson")) is False


def test_js_map_has_community_outline_layer_with_hover_tooltip():
    src = bhd.JS_MAP
    assert "communities.geojson" in src
    assert "bindTooltip" in src
    assert "properties.name" in src


def test_mare_site_meta_subtitle_has_no_stale_four_km2():
    # Pre-MAREBOUND subtitle said "~4 km²" (the whole bairro); the
    # analysed extent is now ~3.4 km² (the IPP Territórios Sociais
    # outline, promoted 2026-09-24 — bigger than the retired
    # union-of-communities definition's ~2 km², since the outline covers
    # ground between communities too). Checks the exact stale phrase, not
    # a bare "4 km" substring — the new correct value "~3.4 km²" itself
    # contains "4 km" as a substring.
    subtitle = bhd.SITE_META["maré"]["subtitle"]
    assert "~4 km" not in subtitle
    assert "15 of 16" in subtitle
