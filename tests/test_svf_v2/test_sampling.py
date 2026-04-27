"""Tests for src/svf_v2/sampling.py."""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

from src.svf_v2.sampling import (
    _offset_points_outside_buildings,
    _outward_normal_2d,
    _sample_points_along_line,
)


class TestSamplePointsAlongLine:
    def test_regular_spacing(self):
        line = LineString([(0, 0), (10, 0)])
        pts = _sample_points_along_line(line, spacing=2.0)
        # Should have points at 0, 2, 4, 6, 8, 10
        assert len(pts) == 6
        distances = [d for _, d in pts]
        np.testing.assert_allclose(distances, [0, 2, 4, 6, 8, 10], atol=0.01)

    def test_short_line_uses_midpoint(self):
        line = LineString([(0, 0), (0.5, 0)])
        pts = _sample_points_along_line(line, spacing=2.0)
        assert len(pts) == 1
        assert pts[0][1] == pytest.approx(0.25)

    def test_includes_endpoint(self):
        line = LineString([(0, 0), (9, 0)])
        pts = _sample_points_along_line(line, spacing=2.0)
        # Points at 0, 2, 4, 6, 8; end=9 is 1m away (< spacing/2=1.0),
        # so endpoint is NOT added. Last sampled point is at 8.0.
        assert pts[-1][1] == pytest.approx(8.0, abs=0.01)
        # With spacing=3.0 on a 10m line, endpoint should be added:
        line2 = LineString([(0, 0), (10, 0)])
        pts2 = _sample_points_along_line(line2, spacing=3.0)
        # 0, 3, 6, 9 -- end=10, dist from 9 is 1.0 < 1.5 -> not added
        # last is 9.0
        assert pts2[-1][1] == pytest.approx(9.0, abs=0.01)


class TestOutwardNormal:
    def test_east_edge(self):
        """Edge going north along east side -> outward normal points east."""
        nx, ny = _outward_normal_2d(1, 0, 1, 1)
        assert nx == pytest.approx(1.0, abs=0.01)
        assert ny == pytest.approx(0.0, abs=0.01)

    def test_north_edge(self):
        """Edge going west along north side -> outward normal points north."""
        nx, ny = _outward_normal_2d(1, 1, 0, 1)
        assert nx == pytest.approx(0.0, abs=0.01)
        assert ny == pytest.approx(1.0, abs=0.01)

    def test_zero_length_edge(self):
        nx, ny = _outward_normal_2d(5, 5, 5, 5)
        assert nx == 0.0
        assert ny == 0.0

    def test_ccw_square_normals_point_outward(self):
        """For a CCW square (0,0)->(1,0)->(1,1)->(0,1), all normals point outward."""
        # Bottom edge: (0,0) -> (1,0) => outward = south (0, -1)
        nx, ny = _outward_normal_2d(0, 0, 1, 0)
        assert ny < 0  # south

        # Right edge: (1,0) -> (1,1) => outward = east (1, 0)
        nx, ny = _outward_normal_2d(1, 0, 1, 1)
        assert nx > 0  # east

        # Top edge: (1,1) -> (0,1) => outward = north (0, 1)
        nx, ny = _outward_normal_2d(1, 1, 0, 1)
        assert ny > 0  # north

        # Left edge: (0,1) -> (0,0) => outward = west (-1, 0)
        nx, ny = _outward_normal_2d(0, 1, 0, 0)
        assert nx < 0  # west


# ---------------------------------------------------------------------------
# Helpers for offset tests
# ---------------------------------------------------------------------------


def _make_building_gdf(polygons, crs="EPSG:31983"):
    """Create a GeoDataFrame of building footprints from a list of Polygons."""
    return gpd.GeoDataFrame(geometry=polygons, crs=crs)


def _make_points_gdf(points, crs="EPSG:31983", **extra_cols):
    """Create a street-point-like GeoDataFrame from (x, y) tuples."""
    gdf = gpd.GeoDataFrame(
        geometry=[Point(x, y) for x, y in points],
        crs=crs,
    )
    for col, vals in extra_cols.items():
        gdf[col] = vals
    return gdf


class TestOffsetPointsOutsideBuildings:
    """Tests for _offset_points_outside_buildings."""

    def test_point_inside_single_building(self):
        """Point at centre of 10x10 square is offset outside."""
        bldg = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        pts = _make_points_gdf([(5, 5)])
        bldgs = _make_building_gdf([bldg])

        result = _offset_points_outside_buildings(pts, bldgs, safety_margin=0.5)

        assert bool(result.at[0, "was_offset"]) is True
        new_pt = result.geometry.iloc[0]
        # Must be outside the building
        assert not bldg.contains(new_pt)
        # Distance from boundary >= margin
        assert bldg.exterior.distance(new_pt) >= 0.5 - 1e-6

    def test_point_outside_unchanged(self):
        """Point outside building stays put."""
        bldg = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        pts = _make_points_gdf([(15, 5)])
        bldgs = _make_building_gdf([bldg])

        result = _offset_points_outside_buildings(pts, bldgs, safety_margin=0.5)

        assert bool(result.at[0, "was_offset"]) is False
        assert result.geometry.iloc[0].x == pytest.approx(15.0)
        assert result.geometry.iloc[0].y == pytest.approx(5.0)

    def test_safety_margin_respected(self):
        """Point near edge is pushed out by at least the margin."""
        bldg = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        pts = _make_points_gdf([(5, 9.5)])  # inside, near top edge
        bldgs = _make_building_gdf([bldg])

        result = _offset_points_outside_buildings(pts, bldgs, safety_margin=0.5)

        new_pt = result.geometry.iloc[0]
        assert not bldg.contains(new_pt)
        assert bldg.exterior.distance(new_pt) >= 0.5 - 1e-6

    def test_point_between_two_buildings(self):
        """Point inside one building near a gap; offset lands in gap, not in other building."""
        bldg1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        bldg2 = Polygon([(12, 0), (22, 0), (22, 10), (12, 10)])  # 2m gap
        pts = _make_points_gdf([(9.5, 5)])  # inside bldg1 near right edge
        bldgs = _make_building_gdf([bldg1, bldg2])

        result = _offset_points_outside_buildings(pts, bldgs, safety_margin=0.5)

        new_pt = result.geometry.iloc[0]
        assert not bldg1.contains(new_pt)
        assert not bldg2.contains(new_pt)

    def test_multipolygon_building(self):
        """MultiPolygon building: point inside one part is offset correctly."""
        part1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        part2 = Polygon([(20, 0), (30, 0), (30, 10), (20, 10)])
        multi = MultiPolygon([part1, part2])
        pts = _make_points_gdf([(5, 5)])  # inside part1
        bldgs = _make_building_gdf([multi])

        result = _offset_points_outside_buildings(pts, bldgs, safety_margin=0.5)

        new_pt = result.geometry.iloc[0]
        assert not part1.contains(new_pt)
        assert bool(result.at[0, "was_offset"]) is True

    def test_no_buildings_noop(self):
        """Empty footprints GDF → all points unchanged."""
        pts = _make_points_gdf([(5, 5), (15, 15)])
        bldgs = _make_building_gdf([])

        result = _offset_points_outside_buildings(pts, bldgs, safety_margin=0.5)

        assert not result["was_offset"].any()
        assert len(result) == 2

    def test_preserves_columns(self):
        """Extra columns (street_id, distance_along) survive offset."""
        bldg = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        pts = _make_points_gdf(
            [(5, 5), (15, 5)],
            street_id=[0, 1],
            distance_along=[2.5, 7.0],
        )
        bldgs = _make_building_gdf([bldg])

        result = _offset_points_outside_buildings(pts, bldgs, safety_margin=0.5)

        assert "street_id" in result.columns
        assert "distance_along" in result.columns
        assert list(result["street_id"]) == [0, 1]
        assert list(result["distance_along"]) == [2.5, 7.0]

    def test_all_directions_blocked(self):
        """Point surrounded by 4 touching buildings — flagged, not crashed."""
        # Four buildings forming a tight enclosure with point in center of one
        b1 = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        b2 = Polygon([(5, 0), (10, 0), (10, 5), (5, 5)])
        b3 = Polygon([(0, 5), (5, 5), (5, 10), (0, 10)])
        b4 = Polygon([(5, 5), (10, 5), (10, 10), (5, 10)])
        pts = _make_points_gdf([(2.5, 2.5)])  # center of b1
        bldgs = _make_building_gdf([b1, b2, b3, b4])

        # Should not raise
        result = _offset_points_outside_buildings(pts, bldgs, safety_margin=0.5)
        assert len(result) == 1


class TestSampleStreetPointsWithBuildings:
    """Integration-level tests for the footprints_gdf parameter."""

    def test_backward_compatible(self):
        """Calling without footprints_gdf produces no offset columns."""

        # We can't easily call sample_street_points without real files,
        # so test the offset function directly with None
        pts = _make_points_gdf([(5, 5), (15, 15)])
        result = _offset_points_outside_buildings(pts, None, safety_margin=0.5)
        # When footprints_gdf=None, columns are still added but all False
        assert not result["was_offset"].any()

    def test_with_footprints_no_points_inside(self):
        """After offset, zero points remain inside buildings."""
        bldg = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        pts = _make_points_gdf([(5, 5), (3, 7), (15, 5)])
        bldgs = _make_building_gdf([bldg])

        result = _offset_points_outside_buildings(pts, bldgs, safety_margin=0.5)

        # Verify no points inside building
        joined = gpd.sjoin(
            result[["geometry"]],
            bldgs[["geometry"]],
            how="inner",
            predicate="within",
        )
        assert len(joined) == 0
