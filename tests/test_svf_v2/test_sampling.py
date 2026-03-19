"""Tests for src/svf_v2/sampling.py."""

import numpy as np
import pytest
from shapely.geometry import LineString

from src.svf_v2.sampling import _sample_points_along_line, _outward_normal_2d


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
