"""Tests for patch_selection.tiling module."""

import numpy as np
import pytest
import geopandas as gpd
from shapely.geometry import box

from src.patch_selection.tiling import (
    generate_tiles,
    classify_tiles,
    compute_buffered_extents,
    enrich_tiles,
    build_tile_grid,
)


class TestGenerateTiles:
    def test_tile_count_no_overlap(self, synthetic_boundary):
        """400m boundary / 100m tiles = 4×4 = 16 tiles."""
        tiles = generate_tiles(synthetic_boundary, tile_size=100.0)
        assert len(tiles) == 16

    def test_tile_count_with_overlap(self, synthetic_boundary):
        """With 50m overlap, step=50m → 8×8 = 64 tiles (approx)."""
        tiles = generate_tiles(synthetic_boundary, tile_size=100.0, overlap=50.0)
        # 400m / 50m step = 8 steps per axis
        assert len(tiles) == 64

    def test_crs_preserved(self, synthetic_boundary):
        tiles = generate_tiles(synthetic_boundary, tile_size=100.0)
        assert tiles.crs == synthetic_boundary.crs

    def test_tile_ids_unique(self, synthetic_boundary):
        tiles = generate_tiles(synthetic_boundary, tile_size=100.0)
        assert tiles["tile_id"].nunique() == len(tiles)

    def test_tile_geometry_is_square(self, synthetic_boundary):
        tiles = generate_tiles(synthetic_boundary, tile_size=100.0)
        for geom in tiles.geometry:
            minx, miny, maxx, maxy = geom.bounds
            assert abs((maxx - minx) - 100.0) < 0.01
            assert abs((maxy - miny) - 100.0) < 0.01

    def test_large_tile_single(self, synthetic_boundary):
        """Tile larger than boundary → 1 tile."""
        tiles = generate_tiles(synthetic_boundary, tile_size=500.0)
        assert len(tiles) == 1


class TestClassifyTiles:
    def test_interior_tiles_exist(self, synthetic_boundary):
        tiles = generate_tiles(synthetic_boundary, tile_size=100.0)
        classified = classify_tiles(tiles, synthetic_boundary)
        assert (classified["classification"] == "interior").sum() > 0

    def test_exterior_dropped(self):
        """Tile completely outside boundary is excluded."""
        boundary = gpd.GeoDataFrame(
            {"geometry": [box(0, 0, 100, 100)]}, crs="EPSG:31983"
        )
        tiles = gpd.GeoDataFrame(
            {"tile_id": ["outside"], "geometry": [box(500, 500, 600, 600)]},
            crs="EPSG:31983",
        )
        classified = classify_tiles(tiles, boundary)
        assert len(classified) == 0

    def test_full_overlap_is_interior(self, synthetic_boundary):
        """Tile fully inside boundary → interior."""
        tile_inside = gpd.GeoDataFrame(
            {"tile_id": ["inside"], "geometry": [box(679900, 7470900, 680000, 7471000)]},
            crs="EPSG:31983",
        )
        classified = classify_tiles(tile_inside, synthetic_boundary)
        assert len(classified) == 1
        assert classified.iloc[0]["classification"] == "interior"

    def test_partial_overlap_is_edge(self, synthetic_boundary):
        """Tile 25% inside boundary → edge."""
        # Overlaps the boundary corner by 50×50 out of 100×100
        tile_edge = gpd.GeoDataFrame(
            {"tile_id": ["edge"], "geometry": [box(680150, 7471150, 680250, 7471250)]},
            crs="EPSG:31983",
        )
        classified = classify_tiles(tile_edge, synthetic_boundary)
        assert len(classified) == 1
        assert classified.iloc[0]["classification"] == "edge"

    def test_overlap_frac_column(self, synthetic_boundary):
        tiles = generate_tiles(synthetic_boundary, tile_size=100.0)
        classified = classify_tiles(tiles, synthetic_boundary)
        assert "boundary_overlap_frac" in classified.columns
        assert classified["boundary_overlap_frac"].between(0, 1).all()


class TestBufferedExtents:
    def test_buffer_column_added(self, synthetic_boundary, synthetic_buildings):
        tiles = generate_tiles(synthetic_boundary, tile_size=100.0)
        tiles = classify_tiles(tiles, synthetic_boundary)
        tiles = compute_buffered_extents(tiles, synthetic_buildings)
        assert "geometry_buffered" in tiles.columns
        assert "buffer_distance_m" in tiles.columns

    def test_buffer_larger_than_tile(self, synthetic_boundary, synthetic_buildings):
        tiles = generate_tiles(synthetic_boundary, tile_size=100.0)
        tiles = classify_tiles(tiles, synthetic_boundary)
        tiles = compute_buffered_extents(tiles, synthetic_buildings)
        for _, row in tiles.iterrows():
            assert row.geometry_buffered.area > row.geometry.area

    def test_buffer_clamped(self, synthetic_boundary, synthetic_buildings):
        tiles = generate_tiles(synthetic_boundary, tile_size=100.0)
        tiles = classify_tiles(tiles, synthetic_boundary)
        tiles = compute_buffered_extents(tiles, synthetic_buildings, clamp=(50.0, 100.0))
        assert tiles["buffer_distance_m"].between(50.0, 100.0).all()


class TestEnrichTiles:
    def test_building_count(self, synthetic_boundary, synthetic_buildings, synthetic_dtm):
        tiles = generate_tiles(synthetic_boundary, tile_size=200.0)
        tiles = classify_tiles(tiles, synthetic_boundary)
        tiles = enrich_tiles(tiles, synthetic_buildings, synthetic_dtm)
        assert "n_buildings" in tiles.columns
        assert tiles["n_buildings"].sum() > 0

    def test_dtm_coverage(self, synthetic_boundary, synthetic_buildings, synthetic_dtm):
        tiles = generate_tiles(synthetic_boundary, tile_size=200.0)
        tiles = classify_tiles(tiles, synthetic_boundary)
        tiles = enrich_tiles(tiles, synthetic_buildings, synthetic_dtm)
        assert "has_dtm_coverage" in tiles.columns
        # Synthetic DTM is fully valid, so interior tiles should have coverage
        interior = tiles[tiles.get("classification", "") == "interior"]
        if len(interior) > 0:
            assert interior["has_dtm_coverage"].any()


class TestBuildTileGrid:
    def test_full_pipeline(self, synthetic_boundary, synthetic_buildings, synthetic_dtm):
        tiles = build_tile_grid(
            synthetic_boundary, synthetic_buildings, synthetic_dtm,
            tile_size=100.0,
        )
        expected_cols = {
            "tile_id", "geometry", "geometry_buffered", "classification",
            "boundary_overlap_frac", "n_buildings", "has_dtm_coverage",
            "buffer_distance_m",
        }
        assert expected_cols.issubset(set(tiles.columns))
        assert len(tiles) > 0
